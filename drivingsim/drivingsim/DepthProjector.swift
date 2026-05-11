//
//  DepthProjector.swift
//  drivingsim
//
//  Projects depth image rays into world space and updates an OccupancyGrid.
//  Uses pinhole camera model with ESP32-CAM OV2640 FOV (65° horizontal).
//
//  DepthAnythingV2 outputs DISPARITY normalized per-frame: large u8 = CLOSE,
//  small u8 = FAR. Per-frame normalization means absolute u8 thresholds are
//  meaningless — we use adaptive percentile thresholds over the obstacle
//  band of the image (excluding floor + ceiling).
//
//  Vertical band: skip top (ceiling/sky) and bottom (floor near robot).
//  Robot eye at 6cm above floor, camera looking forward → horizon is around
//  the vertical middle of the image. Floor pixels below horizon look CLOSE
//  but are not obstacles (traversable). Sample the obstacle-at-robot-height
//  band: rows ~25%-60% of image height.
//
//  Per-column statistic: MAX u8 in the band → largest disparity → closest
//  thing in that direction. Adaptive percentile classifies how close.
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

    // Vertical band: obstacles at approximate robot-eye height appear in the
    // middle horizontal slice. Skip top 25% (ceiling/distant background) and
    // bottom 40% (floor — looks close as disparity but is traversable).
    static let rowTopFrac:    Float = 0.25
    static let rowBottomFrac: Float = 0.60

    // Range assigned to each obstacle tier.
    static let closeRangeM:    Float = 0.7
    static let midRangeM:      Float = 1.6
    static let distantRangeM:  Float = 3.5
    static let freeRangeM:     Float = 5.0

    // Degenerate-frame guard: if disparity range too flat → treat as all free.
    static let minDepthSpreadU8: Int = 30

    /// Main entry: update `grid` from current depth frame at given pose.
    /// Input depthU8 is DISPARITY format (large = close, small = far).
    static func update(depthU8: [UInt8], w: Int, h: Int,
                       posePos: SIMD2<Float>, poseYaw: Float,
                       grid: inout OccupancyGrid) {
        guard depthU8.count == w * h, w > 0, h > 0 else { return }

        let stripTop = Int(Float(h) * rowTopFrac)
        let stripBot = Int(Float(h) * rowBottomFrac)
        guard stripBot > stripTop else { return }

        // ── Pass 0: per-column MAX (= closest disparity) + frame histogram ──
        var colMaxes = [UInt8](repeating: 0, count: w)
        var hist     = [Int](repeating: 0, count: 256)
        var col = 0
        while col < w {
            defer { col += colStride }
            var maxDisp: UInt8 = 0
            for row in stripTop..<stripBot {
                let v = depthU8[row * w + col]
                if v > maxDisp { maxDisp = v }
                hist[Int(v)] += 1
            }
            colMaxes[col] = maxDisp
        }

        // ── Frame percentiles: P30 / P60 / P85 over band ─────────────────
        // P85 = top 15% disparity (very close), P60 = next ~25% (mid), etc.
        let total = hist.reduce(0, +)
        guard total > 0 else { return }
        let p30 = percentile(hist: hist, total: total, frac: 0.30)
        let p60 = percentile(hist: hist, total: total, frac: 0.60)
        let p85 = percentile(hist: hist, total: total, frac: 0.85)

        let allFree = (Int(p85) - Int(p30)) < minDepthSpreadU8

        let startCell = OccupancyGrid.worldToCell(posePos)

        // ── Per-column ray cast with adaptive disparity classification ────
        col = 0
        while col < w {
            defer { col += colStride }
            let maxDisp = colMaxes[col]

            let rangeM: Float
            let isObstacle: Bool
            if allFree {
                rangeM = freeRangeM;    isObstacle = false
            } else if maxDisp >= p85 {
                rangeM = closeRangeM;   isObstacle = true
            } else if maxDisp >= p60 {
                rangeM = midRangeM;     isObstacle = true
            } else if maxDisp >= p30 {
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
