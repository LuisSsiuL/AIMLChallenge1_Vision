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
    // Safety scan: check cells 1..safetyLookaheadMax ahead — earlier brake.
    static let safetyLookaheadMin: Int = 1
    static let safetyLookaheadMax: Int = 6     // ~60cm — well before collision
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

        // Safety: scan a forward cone for occupied cells.
        // If anything in the cone is occupied → invalidate path, then choose
        // steer-around direction by checking left vs right shoulder clearance.
        let blockedAhead = scanForward(from: posePos, yaw: poseYaw, grid: grid)
        if blockedAhead {
            path.removeAll()    // force replan on next tick
            let leftClear  = isShoulderClear(from: posePos, yaw: poseYaw - .pi / 4, grid: grid)
            let rightClear = isShoulderClear(from: posePos, yaw: poseYaw + .pi / 4, grid: grid)
            if leftClear && !rightClear  { return .forwardLeft  }
            if rightClear && !leftClear  { return .forwardRight }
            if leftClear && rightClear   {
                // both clear — pick side that matches target direction
                return headingError < 0 ? .forwardLeft : .forwardRight
            }
            // both blocked — reverse with a turn to escape
            return .reverseLeft
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

    /// Scan cells 1..safetyLookaheadMax ahead. Return true if any is occupied.
    private static func scanForward(from pos: SIMD2<Float>, yaw: Float, grid: OccupancyGrid) -> Bool {
        for steps in safetyLookaheadMin...safetyLookaheadMax {
            let c = lookaheadCell(from: pos, yaw: yaw, steps: steps)
            if grid.state(c) == .occupied { return true }
        }
        return false
    }

    /// Check if a shoulder direction (e.g. ±45°) is reasonably clear for a 3-cell scan.
    private static func isShoulderClear(from pos: SIMD2<Float>, yaw: Float, grid: OccupancyGrid) -> Bool {
        for steps in 1...4 {
            let c = lookaheadCell(from: pos, yaw: yaw, steps: steps)
            if grid.state(c) == .occupied { return false }
        }
        return true
    }

    private static func lookaheadCell(from pos: SIMD2<Float>, yaw: Float, steps: Int) -> Cell {
        let stepM = OccupancyGrid.resolution * Float(steps)
        let lookPos = pos + SIMD2<Float>(sin(yaw), -cos(yaw)) * stepM
        return OccupancyGrid.worldToCell(lookPos)
    }
}
