//
//  DepthProjector.swift
//  drivingsim
//
//  Projects a metric depth image into the world via pinhole intrinsics,
//  filters hits to (≤ maxRangeM) and reasonable height-band, and stamps
//  the OccupancyGrid: occupied at hit cells, free along the camera ray.
//
//  Designed for .mapMetric mode. Intrinsics derived from SimFPVRenderer's
//  65° HFOV (AI-Thinker ESP32-CAM OV2640 stock lens). Calibrate per-unit if
//  swapping lens.
//

import Foundation
import simd

struct DepthProjector {
    // Camera intrinsics (square sensor). Must match SimFPVRenderer fieldOfView.
    var hfovDeg: Float = 65.0

    // Range gate — only project hits closer than this (meters).
    // Explore mode spec: 5m radius mapping window. Keep projection bounded to
    // match. (Previously 20 to absorb metric hallucination on sim renders;
    // metricScale below handles the over-estimation.)
    var maxRangeM: Float = 5.0
    // Drop hits closer than this (model noise / sensor self-view).
    var minRangeM: Float = 0.25

    /// Multiplicative scale on raw model output. 1.0 = trust model. Use <1.0
    /// to correct over-estimation (sim render: try ~0.3). Calibrate once on
    /// real ESP32-CAM with a ruler.
    var metricScale: Float = 0.3

    // Camera mount height above ground (meters). Matches SimScene eyeHeight.
    var cameraHeightM: Float = 0.06

    // Height band relative to ground that counts as an obstacle.
    // Below floor or above ceiling = ignore. Bumped from 0.01 → 0.05 to
    // suppress near-horizon floor pixels with worldY slightly above 0
    // (camera height = 6 cm, so floor pixels just below horizon would
    // otherwise pass the filter and stamp as low obstacles).
    var minObstacleY: Float = 0.05   // floor clearance (m above ground)
    var maxObstacleY: Float = 0.40   // anything taller than this still counts

    // Image-row cutoff: ignore pixels with v ≥ maxRowFrac * height. For a
    // horizontally-mounted camera near the floor, the bottom of the image is
    // guaranteed floor — the analytic height filter alone can't reject it
    // when the camera is barely above ground. 0.55 leaves a small margin
    // below the horizon (cy = height/2).
    var maxRowFrac: Float = 0.55

    // "Absolutely sure it's a wall" gate. Same-intensity (low-gradient) pixels
    // are skipped for occupancy bumps — only depth edges count. Sum of 4
    // neighbour |Δdepth| (model-meters, pre-metricScale) must exceed threshold.
    // Free-ray-cast still runs so corridors get cleared.
    var requireDepthEdge: Bool = true
    var minDepthGradientSum: Float = 0.20   // ≈ 0.05 m per neighbour

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
        let vCutoff = min(height, Int(Float(height) * config.maxRowFrac))
        for v in stride(from: 0, to: vCutoff, by: step) {
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

                // Free along the ray always (corridor evidence is cheap +
                // safe). Occupancy bump is gated on depth-edge strength so
                // uniform-intensity floor/wall noise doesn't false-stamp.
                grid.rayCastFree(from: camCell, to: hitCell, freeDelta: 0.20)

                var stampOccupied = true
                if config.requireDepthEdge {
                    let uL = max(0, u - step), uR = min(width - 1, u + step)
                    let vU = max(0, v - step), vD = min(height - 1, v + step)
                    let dL = meters[v * width + uL]
                    let dR = meters[v * width + uR]
                    let dU = meters[vU * width + u]
                    let dD = meters[vD * width + u]
                    let center = raw
                    let gradSum = abs(center - dL) + abs(center - dR)
                                + abs(center - dU) + abs(center - dD)
                    if !gradSum.isFinite || gradSum < config.minDepthGradientSum {
                        stampOccupied = false
                    }
                }
                if stampOccupied {
                    grid.bumpOccupied(at: hitCell, delta: 0.25)
                    hitCount += 1
                }
            }
        }
        return hitCount
    }
}
