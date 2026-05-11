//
//  WaypointFollower.swift
//  drivingsim
//
//  Heading-error controller: given next waypoint cell + current pose,
//  produce a DrivingCommand. Safety override from occupancy grid.
//
//  Real-world ref: simplified DWA (Dynamic Window Approach) local planner.
//

import Darwin
import Foundation
import simd

enum WaypointFollower {

    // Heading error threshold — within this, drive straight.
    static let alignThreshold:  Float = 0.25   // radians (~14°)
    // Waypoint reach radius — advance to next waypoint when within this.
    static let reachRadiusM:    Float = 0.3    // metres
    // Lookahead distance for obstacle safety check in grid cells.
    static let safetyLookahead: Int   = 3
    // Path cells to look ahead — pick a waypoint a few cells out for smoother heading.
    static let waypointLookahead: Int = 4

    /// Returns the DrivingCommand to follow the path given current pose.
    /// Mutates `path` to advance past reached waypoints.
    static func nextCommand(posePos: SIMD2<Float>,
                            poseYaw: Float,
                            path: inout [Cell],
                            grid: OccupancyGrid) -> DrivingCommand {

        // Advance past waypoints we've already reached.
        while !path.isEmpty {
            let wp = OccupancyGrid.cellToWorld(path[0])
            let dx = wp.x - posePos.x
            let dz = wp.y - posePos.y   // SIMD2.y = world Z
            let dist = (dx * dx + dz * dz).squareRoot()
            if dist < reachRadiusM {
                path.removeFirst()
            } else {
                break
            }
        }

        guard !path.isEmpty else { return .brake }

        // Look a few cells ahead from path[0] for smoother heading control.
        let targetIdx = min(waypointLookahead, path.count - 1)
        let targetCell = path[targetIdx]
        let targetWorld = OccupancyGrid.cellToWorld(targetCell)

        let dx = targetWorld.x - posePos.x
        let dz = targetWorld.y - posePos.y

        // atan2(x, -z) because yaw=0 faces -Z (north) and increases clockwise.
        let targetAngle = atan2(dx, -dz)
        var headingError = targetAngle - poseYaw

        // Normalize to [-π, π]
        while headingError >  Float.pi { headingError -= 2 * Float.pi }
        while headingError < -Float.pi { headingError += 2 * Float.pi }

        // Safety: check for occupied cells directly ahead.
        let frontCell = lookaheadCell(from: posePos, yaw: poseYaw, steps: safetyLookahead)
        if grid.state(frontCell) == .occupied {
            return .reverse
        }

        // Large heading error: turn in place.
        if headingError > alignThreshold  { return .forwardRight }
        if headingError < -alignThreshold { return .forwardLeft  }

        // Aligned: drive forward with gentle steering correction.
        if headingError > alignThreshold * 0.4  { return .forwardRight }
        if headingError < -alignThreshold * 0.4 { return .forwardLeft  }
        return .forward
    }

    // MARK: - Private

    private static func lookaheadCell(from pos: SIMD2<Float>, yaw: Float, steps: Int) -> Cell {
        let stepM = OccupancyGrid.resolution * Float(steps)
        let lookPos = pos + SIMD2<Float>(sin(yaw), -cos(yaw)) * stepM
        return OccupancyGrid.worldToCell(lookPos)
    }
}
