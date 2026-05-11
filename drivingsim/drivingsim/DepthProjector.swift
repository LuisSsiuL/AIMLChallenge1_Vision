//
//  DepthProjector.swift
//  drivingsim
//
//  Projects depth image rays into world space and updates an OccupancyGrid.
//  Uses pinhole camera model with ESP32-CAM OV2640 FOV (65° horizontal).
//
//  DepthAnythingV2 outputs RELATIVE depth normalized per-frame (0-255), so
//  u8 values don't translate directly to metres. Strategy here is to use u8
//  as a proximity signal:
//    - u8 below `closeThreshold` → close obstacle; cast short ray + mark hit
//    - u8 between mid and far → free space at medium range
//    - u8 near max → far/free, mark free out to maxRayRangeM
//
//  Same ray-cast algorithm used in Hector SLAM / GMapping.
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

    // Ray sampling stride. 4 → 130 rays per frame. Balance of coverage vs CPU.
    static let colStride: Int = 4

    // Vertical strip — top of image is far, bottom is floor.
    // Skip top 25% (often sky/ceiling) and bottom 10% (immediate floor).
    static let rowTopFrac:    Float = 0.25
    static let rowBottomFrac: Float = 0.90

    // Proximity → distance curve.
    // DepthAnythingV2 is per-frame normalized, so we calibrate by ASSUMPTION:
    //   u8 < 60     → very close obstacle, place hit at ~0.6m
    //   u8 60-130   → mid range obstacle, place hit at ~1.5m
    //   u8 130-200  → distant obstacle, place hit at ~3.5m
    //   u8 > 200    → free space, ray out to 5m (mark free along ray)
    static let closeMaxU8:  UInt8 = 60
    static let midMaxU8:    UInt8 = 130
    static let distantMaxU8: UInt8 = 200
    static let closeRangeM:    Float = 0.7
    static let midRangeM:      Float = 1.6
    static let distantRangeM:  Float = 3.5
    static let freeRangeM:     Float = 5.0

    /// Main entry: update `grid` from current depth frame at given pose.
    ///
    /// Two-pass strategy:
    ///   PASS A — ray cast per column for free-space sweep (visibility cone).
    ///   PASS B — dense obstacle stamping: every "obstacle-class" depth pixel in
    ///            the lower band gets its own world-cell marked occupied. This
    ///            makes depth-detected obstacles visible as dense clusters even
    ///            when their distance estimate is rough.
    static func update(depthU8: [UInt8], w: Int, h: Int,
                       posePos: SIMD2<Float>, poseYaw: Float,
                       grid: inout OccupancyGrid) {
        guard depthU8.count == w * h, w > 0, h > 0 else { return }

        let stripTop = Int(Float(h) * rowTopFrac)
        let stripBot = Int(Float(h) * rowBottomFrac)
        guard stripBot > stripTop else { return }

        let startCell = OccupancyGrid.worldToCell(posePos)

        // PASS A — per-column ray for free-space sweep
        var col = 0
        while col < w {
            defer { col += colStride }

            var minDepth: UInt8 = 255
            for row in stripTop..<stripBot {
                let v = depthU8[row * w + col]
                if v < minDepth { minDepth = v }
            }

            let rangeM: Float
            let isObstacle: Bool
            if minDepth <= closeMaxU8 {
                rangeM = closeRangeM;   isObstacle = true
            } else if minDepth <= midMaxU8 {
                rangeM = midRangeM;     isObstacle = true
            } else if minDepth <= distantMaxU8 {
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

        // PASS B — dense obstacle stamping: every close/mid-class pixel in lower
        // band gets projected to world XZ at its inferred range. Marks the cell
        // directly so obstacles thicken in the map even with noisy depth.
        let denseColStride = 8        // every 8th col
        let denseRowStride = 8        // every 8th row in obstacle band
        var dcol = 0
        while dcol < w {
            defer { dcol += denseColStride }
            var drow = stripTop
            while drow < stripBot {
                defer { drow += denseRowStride }
                let v = depthU8[drow * w + dcol]
                let rangeM: Float
                if v <= closeMaxU8        { rangeM = closeRangeM   }
                else if v <= midMaxU8     { rangeM = midRangeM     }
                else if v <= distantMaxU8 { rangeM = distantRangeM }
                else { continue }   // far/free pixel — skip in dense pass

                let uNorm = Float(dcol) - cx
                let angleFromCenter = atan2(uNorm, fx)
                let worldAngle = poseYaw + angleFromCenter
                let hitPos = posePos
                    + SIMD2<Float>(sin(worldAngle), -cos(worldAngle)) * rangeM
                let hitCell = OccupancyGrid.worldToCell(hitPos)
                // Stamp obstacle directly (no free-line traversal — already done in PASS A).
                grid.stampOccupied(cell: hitCell)
            }
        }
    }
}
