//
//  DepthProjector.swift
//  drivingsim
//
//  Projects depth image rays into world space and updates an OccupancyGrid.
//  Uses pinhole camera model with ESP32-CAM OV2640 FOV (65° horizontal).
//
//  Real-world ref: same ray-cast update used in Hector SLAM / GMapping for
//  lidar. Here depth U8 from DepthAnythingV2 replaces laser range measurements.
//
//  Depth U8 is RELATIVE (not metric): 255 = far, 0 = near. We scale using
//  `farRangeM` — the assumed real-world distance that depth=255 maps to.
//  Calibrate by knowing the spawn-to-north-wall distance (~18m) and observing
//  what depth value the wall registers at that distance.
//

import Foundation
import simd

enum DepthProjector {

    // Camera intrinsics — DepthAnythingV2 input 518×518, FOV 65° horizontal.
    static let inputW:   Int   = 518
    static let inputH:   Int   = 518
    static let fovDeg:   Float = 65.0
    static let fx:       Float = (Float(inputW) / 2.0) / tan(fovDeg * .pi / 360.0)  // ≈ 406
    static let cx:       Float = Float(inputW) / 2.0   // 259
    static let cy:       Float = Float(inputH) / 2.0   // 259

    // Depth scale: depth U8=255 → this many metres.
    // Room diagonal ≈ sqrt(30²+20²) ≈ 36m. Use 20m so obstacles at mid-range
    // map to reasonable grid cells. Tune if map looks stretched/compressed.
    static let farRangeM: Float = 20.0

    // Ray sampling stride (columns). 4 → 130 rays per frame, fast enough for 30Hz.
    static let colStride: Int = 4

    // Vertical strip for obstacle extraction.
    // Skip top 20% (ceiling/sky) and bottom 15% (floor).
    // Only ground-plane obstacles matter for 2D navigation.
    static let rowTopFrac:    Float = 0.20
    static let rowBottomFrac: Float = 0.85

    // Minimum depth score to register as an obstacle hit (not too far).
    // Obstacles below this depth (= very far) are ignored — beyond reliable range.
    static let minObstacleDepthU8: UInt8 = 5    // ignore depth=0..4 (no return)
    static let maxObstacleDepthU8: UInt8 = 230  // beyond 230 treat as free/far

    // Max ray range in metres — rays beyond this are marked free only.
    static let maxRayRangeM: Float = farRangeM * (Float(maxObstacleDepthU8) / 255.0)

    /// Main entry point: update `grid` using current depth frame and estimated pose.
    /// - Parameters:
    ///   - depthU8: flat array `[UInt8]` from `DepthDriver.depthBufferToU8`, size w×h.
    ///   - w, h: image dimensions (should be 518×518).
    ///   - posePos: estimated world (X, Z) from `PoseEstimator`.
    ///   - poseYaw: estimated yaw (radians) from `PoseEstimator`.
    ///   - grid: occupancy grid to mutate.
    static func update(depthU8: [UInt8], w: Int, h: Int,
                       posePos: SIMD2<Float>, poseYaw: Float,
                       grid: inout OccupancyGrid) {
        guard depthU8.count == w * h, w > 0, h > 0 else { return }

        let startCol = Int(Float(h) * rowTopFrac)
        let endCol   = Int(Float(h) * rowBottomFrac)
        guard endCol > startCol else { return }

        let startCell = OccupancyGrid.worldToCell(posePos)

        // Sample columns across the image.
        var col = 0
        while col < w {
            defer { col += colStride }

            // Find the minimum depth in the vertical strip for this column.
            // Min depth = closest obstacle = most relevant for navigation.
            var minDepth: UInt8 = 255
            for row in startCol..<endCol {
                let v = depthU8[row * w + col]
                if v < minDepth { minDepth = v }
            }

            // Convert depth U8 → metric distance.
            // U8=255 → far (farRangeM), U8=0 → 0m (too close / no return).
            let depthM = Float(minDepth) / 255.0 * farRangeM

            // Horizontal angle from camera centre → world ray direction.
            let uNorm = Float(col) - cx
            let angleFromCenter = atan2(uNorm, fx)   // radians
            let worldAngle = poseYaw + angleFromCenter

            let rayDirX =  sin(worldAngle)
            let rayDirZ = -cos(worldAngle)

            if minDepth < minObstacleDepthU8 || minDepth > maxObstacleDepthU8 {
                // Beyond reliable range — mark ray as free only.
                let farPos = posePos + SIMD2<Float>(rayDirX, rayDirZ) * maxRayRangeM
                let farCell = OccupancyGrid.worldToCell(farPos)
                grid.updateFreeRay(from: startCell, to: farCell)
            } else {
                // Obstacle hit within reliable range.
                let hitPos = posePos + SIMD2<Float>(rayDirX, rayDirZ) * depthM
                let hitCell = OccupancyGrid.worldToCell(hitPos)
                grid.update(from: startCell, to: hitCell)
            }
        }
    }
}
