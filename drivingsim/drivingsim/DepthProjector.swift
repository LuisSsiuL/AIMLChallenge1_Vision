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
    static func update(depthU8: [UInt8], w: Int, h: Int,
                       posePos: SIMD2<Float>, poseYaw: Float,
                       grid: inout OccupancyGrid) {
        guard depthU8.count == w * h, w > 0, h > 0 else { return }

        let stripTop = Int(Float(h) * rowTopFrac)
        let stripBot = Int(Float(h) * rowBottomFrac)
        guard stripBot > stripTop else { return }

        let startCell = OccupancyGrid.worldToCell(posePos)

        var col = 0
        var hitCount = 0
        var freeCount = 0
        while col < w {
            defer { col += colStride }

            // Min depth in vertical strip — closest thing in that column direction.
            var minDepth: UInt8 = 255
            for row in stripTop..<stripBot {
                let v = depthU8[row * w + col]
                if v < minDepth { minDepth = v }
            }

            // Map u8 → distance + obstacle/free decision.
            let rangeM: Float
            let isObstacle: Bool
            if minDepth <= closeMaxU8 {
                rangeM = closeRangeM
                isObstacle = true
            } else if minDepth <= midMaxU8 {
                rangeM = midRangeM
                isObstacle = true
            } else if minDepth <= distantMaxU8 {
                rangeM = distantRangeM
                isObstacle = true
            } else {
                rangeM = freeRangeM
                isObstacle = false
            }

            // Camera-space horizontal angle.
            let uNorm = Float(col) - cx
            let angleFromCenter = atan2(uNorm, fx)
            let worldAngle = poseYaw + angleFromCenter

            let rayDirX =  sin(worldAngle)
            let rayDirZ = -cos(worldAngle)
            let endPos = posePos + SIMD2<Float>(rayDirX, rayDirZ) * rangeM
            let endCell = OccupancyGrid.worldToCell(endPos)

            if isObstacle {
                grid.update(from: startCell, to: endCell)
                hitCount += 1
            } else {
                grid.updateFreeRay(from: startCell, to: endCell)
                freeCount += 1
            }
        }
        _ = hitCount; _ = freeCount   // available for debug logging if needed
    }
}
