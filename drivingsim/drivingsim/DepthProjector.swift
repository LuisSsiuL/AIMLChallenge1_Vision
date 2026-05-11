//
//  DepthProjector.swift
//  drivingsim
//
//  Projects depth image rays into world space and updates an OccupancyGrid.
//  Uses pinhole camera model with ESP32-CAM OV2640 FOV (65° horizontal).
//
//  DepthAnythingV2 outputs RELATIVE depth normalized per-frame (0-255), so
//  absolute u8 thresholds don't translate to absolute distances.
//
//  Strategy: per-frame adaptive percentile thresholds. We compute the
//  distribution of depth values in the obstacle strip, then classify each
//  column's minimum depth against the frame's own statistics:
//    - bottom 15% of u8 values (closest in frame) → close obstacle at 0.7m
//    - 15-40%                                     → mid obstacle at 1.6m
//    - 40-70%                                     → distant obstacle at 3.5m
//    - top 30% (farthest)                         → free ray to 5m
//
//  Guard: if the frame is too uniform (no real depth variation), treat all
//  as free — prevents spurious obstacles from noise in featureless scenes.
//
//  Same ray-cast algorithm as Hector SLAM / GMapping, with relative-depth
//  adaptation in place of metric lidar returns.
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

    // Ray sampling stride. 4 → 130 rays per frame.
    static let colStride: Int = 4

    // Vertical strip — skip top 25% (ceiling/sky) and bottom 10% (floor near).
    static let rowTopFrac:    Float = 0.25
    static let rowBottomFrac: Float = 0.90

    // Range assigned to each obstacle tier.
    static let closeRangeM:    Float = 0.7
    static let midRangeM:      Float = 1.6
    static let distantRangeM:  Float = 3.5
    static let freeRangeM:     Float = 5.0

    // Degenerate-frame guard: if depth distribution is too flat, frame has no
    // useful obstacle info — treat everything as free instead of stamping noise.
    static let minDepthSpreadU8: Int = 30

    /// Main entry: update `grid` from current depth frame at given pose.
    static func update(depthU8: [UInt8], w: Int, h: Int,
                       posePos: SIMD2<Float>, poseYaw: Float,
                       grid: inout OccupancyGrid) {
        guard depthU8.count == w * h, w > 0, h > 0 else { return }

        let stripTop = Int(Float(h) * rowTopFrac)
        let stripBot = Int(Float(h) * rowBottomFrac)
        guard stripBot > stripTop else { return }

        // ── Pass 0: per-column min + frame histogram ───────────────────────
        // Single sweep across sampled columns: record per-column min depth
        // AND accumulate histogram for percentile computation.
        var colMins  = [UInt8](repeating: 255, count: w)
        var hist     = [Int](repeating: 0, count: 256)
        var col = 0
        while col < w {
            defer { col += colStride }
            var minDepth: UInt8 = 255
            for row in stripTop..<stripBot {
                let v = depthU8[row * w + col]
                if v < minDepth { minDepth = v }
                hist[Int(v)] += 1
            }
            colMins[col] = minDepth
        }

        // ── Compute frame percentiles from histogram ───────────────────────
        let total = hist.reduce(0, +)
        guard total > 0 else { return }
        let p15 = percentile(hist: hist, total: total, frac: 0.15)
        let p40 = percentile(hist: hist, total: total, frac: 0.40)
        let p70 = percentile(hist: hist, total: total, frac: 0.70)

        // Degenerate frame: not enough spread → no obstacles can be trusted.
        let allFree = (Int(p70) - Int(p15)) < minDepthSpreadU8

        let startCell = OccupancyGrid.worldToCell(posePos)

        // ── Per-column ray cast with adaptive classification ───────────────
        col = 0
        while col < w {
            defer { col += colStride }
            let minDepth = colMins[col]

            let rangeM: Float
            let isObstacle: Bool
            if allFree {
                rangeM = freeRangeM;   isObstacle = false
            } else if minDepth <= p15 {
                rangeM = closeRangeM;   isObstacle = true
            } else if minDepth <= p40 {
                rangeM = midRangeM;     isObstacle = true
            } else if minDepth <= p70 {
                rangeM = distantRangeM; isObstacle = true
            } else {
                rangeM = freeRangeM;    isObstacle = false
            }

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

    // MARK: - Helpers

    /// Returns the u8 value at the `frac` percentile of the histogram.
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
