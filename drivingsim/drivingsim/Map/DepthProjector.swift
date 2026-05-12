//
//  DepthProjector.swift
//  drivingsim
//
//  Projects a metric depth image into the world via pinhole intrinsics,
//  filters hits to (≤ maxRangeM) and reasonable height-band, and stamps
//  the OccupancyGrid: occupied at hit cells, free along the camera ray.
//
//  Designed for .mapMetric mode. Intrinsics derived from SimFPVRenderer's
//  75° HFOV. For real ESP32-CAM later: calibrate once, replace `hfovDeg`.
//

import Foundation
import simd

struct DepthProjector {
    // Camera intrinsics (square sensor). Must match SimFPVRenderer fieldOfView.
    var hfovDeg: Float = 75.0

    // Range gate — only project hits closer than this (meters).
    // Bumped from 5→20 to tolerate metric-depth hallucination on sim renders
    // (model trained on real photos; sim render scale ~3-4× off).
    var maxRangeM: Float = 20.0
    // Drop hits closer than this (model noise / sensor self-view).
    var minRangeM: Float = 0.25

    /// Multiplicative scale on raw model output. 1.0 = trust model. Use <1.0
    /// to correct over-estimation (sim render: try ~0.3). Calibrate once on
    /// real ESP32-CAM with a ruler.
    var metricScale: Float = 0.3

    // Camera mount height above ground (meters). Matches SimScene eyeHeight.
    var cameraHeightM: Float = 0.06

    // Height band relative to ground that counts as an obstacle.
    // Below floor or above ceiling = ignore.
    var minObstacleY: Float = 0.01   // floor clearance (m above ground)
    var maxObstacleY: Float = 0.40   // anything taller than this still counts

    // Subsample step in pixels. 4 → 1/16th of pixels processed.
    var pixelStep: Int = 4

    /// Project a metric depth frame and stamp the grid.
    /// `meters[v*w + u]` = depth in meters at pixel (u, v).
    /// `posePos` is car (X, Z) in world; `poseYaw` is heading (0 = +Z forward,
    /// matching SimScene convention — see fallbackToward in MapDriver).
    static func stamp(meters: [Float], width: Int, height: Int,
                      posePos: SIMD2<Float>, poseYaw: Float,
                      grid: inout OccupancyGrid,
                      config: DepthProjector = DepthProjector()) -> Int
    {
        guard width > 0, height > 0, meters.count == width * height else { return 0 }

        let cx = Float(width)  * 0.5
        let cy = Float(height) * 0.5
        let fx = cx / tanf(config.hfovDeg * .pi / 360.0)   // hfov/2 in radians
        let fy = fx   // assume square pixels — matches sim renderer

        // Camera origin in grid coords (used for ray-cast free-stamping).
        let camCell = OccupancyGrid.worldToCell(posePos)

        // Car-frame → world-frame rotation: same convention as fallbackToward
        // (yaw=0 → forward = -Z; +X = right). Inverted here we go camera → world.
        let cosY = cosf(poseYaw)
        let sinY = sinf(poseYaw)

        var hitCount = 0
        let step = max(1, config.pixelStep)
        for v in stride(from: 0, to: height, by: step) {
            for u in stride(from: 0, to: width, by: step) {
                let raw = meters[v * width + u]
                if !raw.isFinite { continue }
                let Z = raw * config.metricScale
                if Z < config.minRangeM || Z > config.maxRangeM { continue }

                // Pinhole back-project. Camera frame: +X right, +Y down, +Z forward.
                let Xc = (Float(u) - cx) * Z / fx
                let Yc = (Float(v) - cy) * Z / fy

                // Height band check (Yc grows downward in image; subtract from cam height).
                // World Y = cameraHeightM - Yc  (small Yc near horizon ≈ obstacle height).
                let worldY = config.cameraHeightM - Yc
                if worldY < config.minObstacleY || worldY > config.maxObstacleY {
                    continue
                }

                // Rotate (Xc, Zc) into world (X, Z) using car yaw.
                // SimScene convention: yaw=0 → forward = -Z. So world delta:
                //   dX =  cosY * Xc + sinY * Z
                //   dZ = -sinY * Xc * ??? – we need to match fallbackToward atan2(dx, -dz).
                // Effective rotation: forwardVec = (sin(yaw), -cos(yaw)) in (X, Z).
                let forwardX:  Float =  sinf(poseYaw)
                let forwardZ:  Float = -cosf(poseYaw)
                let rightX:    Float =  cosf(poseYaw)
                let rightZ:    Float =  sinf(poseYaw)
                _ = cosY; _ = sinY   // (kept for clarity; rotation computed via basis vectors)

                let worldX = posePos.x + rightX * Xc + forwardX * Z
                let worldZ = posePos.y + rightZ * Xc + forwardZ * Z

                let hitCell = OccupancyGrid.worldToCell(SIMD2<Float>(worldX, worldZ))
                if !grid.isValid(hitCell) { continue }

                // Free along the ray (camera → just before hit), occupied at hit.
                grid.rayCastFree(from: camCell, to: hitCell, freeDelta: 0.15)
                grid.bumpOccupied(at: hitCell, delta: 0.6)
                hitCount += 1
            }
        }
        return hitCount
    }
}
