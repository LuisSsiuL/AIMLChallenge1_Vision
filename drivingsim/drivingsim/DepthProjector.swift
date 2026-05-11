//
//  DepthProjector.swift
//  drivingsim
//
//  Projects depth image rays into world space and updates an OccupancyGrid.
//  Uses pinhole camera model with ESP32-CAM OV2640 FOV (65° horizontal).
//
//  DepthAnythingV2 outputs DISPARITY normalized per-frame: large u8 = CLOSE.
//
//  Strategy (per-pixel dense ray cast + percentile bands):
//    - Sample every Nth column × Nth row across the obstacle band.
//    - For each pixel, compute its world ray direction (using x intrinsics).
//    - Use frame-wide percentile bands (P50 / P75 / P90) for tier classification.
//      Anything with disparity > P50 → some obstacle (range varies by tier).
//      <P50 OR below absolute floor → free.
//    - Optional bbox skip: pixels inside the box are ignored (used to mask
//      out a YOLO-tracked person so it doesn't get stamped as a wall).
//
//  Per-pixel dense sampling (not per-column max) ensures peripheral walls
//  in wide FOV get registered on the map even when something closer exists
//  ahead — without that, percentile thresholding made side walls "relatively
//  far" and dropped them as free space.
//

import Darwin
import Foundation
import simd

enum DepthProjector {

    // Camera intrinsics — DepthAnythingV2 input 518×518, FOV 65° horizontal.
    static let inputW:   Int   = 518
    static let inputH:   Int   = 518
    static let fovDeg:   Float = 65.0
    static let fx:       Float = (Float(inputW) / 2.0) / tan(fovDeg * Float.pi / 360.0)  // ≈ 406
    static let cx:       Float = Float(inputW) / 2.0
    static let cy:       Float = Float(inputH) / 2.0

    // Sampling stride for dense per-pixel ray cast.
    static let colStride: Int = 6
    static let rowStride: Int = 8

    // Vertical band: middle of image (skip ceiling + floor).
    static let rowTopFrac:    Float = 0.20
    static let rowBottomFrac: Float = 0.65

    // Range per tier.
    static let closeRangeM:    Float = 0.7
    static let midRangeM:      Float = 1.8
    static let distantRangeM:  Float = 3.5
    static let freeRangeM:     Float = 5.0

    // Absolute disparity floor — below this, always free regardless of percentile.
    // Prevents marking very-far walls as obstacles in featureless rooms.
    static let absoluteFreeFloorU8: UInt8 = 25

    // Spread guard: if frame is too uniform, treat all as free.
    static let minDepthSpreadU8: Int = 25

    /// Image-space bbox in normalized coords (cx, cy, w, h all in [0,1]).
    struct SkipBox {
        let cx: Float
        let cy: Float
        let w:  Float
        let h:  Float
    }

    /// Main entry. `skip` (optional) excludes pixels inside that bbox from
    /// being stamped as obstacles — used to mask a YOLO-tracked person.
    static func update(depthU8: [UInt8], w: Int, h: Int,
                       posePos: SIMD2<Float>, poseYaw: Float,
                       grid: inout OccupancyGrid,
                       skip: SkipBox? = nil) {
        guard depthU8.count == w * h, w > 0, h > 0 else { return }

        let stripTop = Int(Float(h) * rowTopFrac)
        let stripBot = Int(Float(h) * rowBottomFrac)
        guard stripBot > stripTop else { return }

        // Skip box bounds in pixel space (inclusive).
        let skipX0: Int; let skipX1: Int
        let skipY0: Int; let skipY1: Int
        if let s = skip {
            skipX0 = Int((s.cx - s.w * 0.5) * Float(w))
            skipX1 = Int((s.cx + s.w * 0.5) * Float(w))
            skipY0 = Int((s.cy - s.h * 0.5) * Float(h))
            skipY1 = Int((s.cy + s.h * 0.5) * Float(h))
        } else {
            skipX0 = -1; skipX1 = -1; skipY0 = -1; skipY1 = -1
        }
        func isSkipped(_ x: Int, _ y: Int) -> Bool {
            skip != nil && x >= skipX0 && x <= skipX1 && y >= skipY0 && y <= skipY1
        }

        // ── Histogram over the sampled band (skip-box excluded) ──────────
        var hist = [Int](repeating: 0, count: 256)
        var col = 0
        while col < w {
            defer { col += colStride }
            var row = stripTop
            while row < stripBot {
                defer { row += rowStride }
                if isSkipped(col, row) { continue }
                hist[Int(depthU8[row * w + col])] += 1
            }
        }
        let total = hist.reduce(0, +)
        guard total > 0 else { return }
        let p50 = percentile(hist: hist, total: total, frac: 0.50)
        let p75 = percentile(hist: hist, total: total, frac: 0.75)
        let p90 = percentile(hist: hist, total: total, frac: 0.90)
        let allFree = (Int(p90) - Int(p50)) < minDepthSpreadU8

        let startCell = OccupancyGrid.worldToCell(posePos)

        // ── Per-pixel ray cast ─────────────────────────────────────────────
        col = 0
        while col < w {
            defer { col += colStride }
            var row = stripTop
            while row < stripBot {
                defer { row += rowStride }
                if isSkipped(col, row) { continue }
                let v = depthU8[row * w + col]

                // Determine range tier.
                let rangeM: Float
                let isObstacle: Bool
                if allFree || v <= absoluteFreeFloorU8 {
                    rangeM = freeRangeM;    isObstacle = false
                } else if v >= p90 {
                    rangeM = closeRangeM;   isObstacle = true
                } else if v >= p75 {
                    rangeM = midRangeM;     isObstacle = true
                } else if v >= p50 {
                    rangeM = distantRangeM; isObstacle = true
                } else {
                    rangeM = freeRangeM;    isObstacle = false
                }

                // Horizontal world angle from column.
                let uNorm = Float(col) - cx
                let angleFromCenter = atan2(uNorm, fx)
                let worldAngle = poseYaw + angleFromCenter
                let rayDirX =  sin(worldAngle)
                let rayDirZ = -cos(worldAngle)
                let endPos  = posePos + SIMD2<Float>(rayDirX, rayDirZ) * rangeM
                let endCell = OccupancyGrid.worldToCell(endPos)

                if isObstacle {
                    grid.update(from: startCell, to: endCell)
                } else {
                    grid.updateFreeRay(from: startCell, to: endCell)
                }
            }
        }
    }

    /// Mean disparity over the central horizontal slice (used for stuck detection).
    /// Returns 0 if input invalid.
    static func centralDisparityMean(depthU8: [UInt8], w: Int, h: Int) -> Float {
        guard depthU8.count == w * h, w > 0, h > 0 else { return 0 }
        let xStart = w / 3
        let xEnd   = (2 * w) / 3
        let stripTop = Int(Float(h) * rowTopFrac)
        let stripBot = Int(Float(h) * rowBottomFrac)
        var sum: UInt64 = 0
        var n:   UInt64 = 0
        for y in stripTop..<stripBot {
            let base = y * w
            for x in xStart..<xEnd {
                sum &+= UInt64(depthU8[base + x])
                n   &+= 1
            }
        }
        return n > 0 ? Float(sum) / Float(n) : 0
    }

    private static func percentile(hist: [Int], total: Int, frac: Float) -> UInt8 {
        let target = Int(Float(total) * frac)
        var cum = 0
        for v in 0..<256 {
            cum += hist[v]
            if cum >= target { return UInt8(v) }
        }
        return 255
    }
}
