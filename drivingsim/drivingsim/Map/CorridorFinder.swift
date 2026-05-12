//
//  CorridorFinder.swift
//  drivingsim
//
//  DFS-based deepest-corridor goal finder for .mapExplore.
//
//  Walks the occupancy grid depth-first from a starting cell, restricted to
//  cells with at least `minWidthCells` perpendicular clearance (no occupied
//  cells within a square radius). Returns the deepest reachable cell + a
//  reconstructed parent chain back to origin.
//
//  Used by MapDriver after each 360° scan to pick the next driving goal.
//

import Foundation

struct CorridorFinder {

    /// Minimum corridor width. 5 cells × 0.05m = 25cm. Relaxed from 7
    /// because stamped wall noise on sim renders breaks the 7-wide check.
    static let minWidthCells: Int = 5
    /// Half-radius of clearance check (2 → 5×5 square).
    static let clearanceRadius: Int = minWidthCells / 2
    /// Reject corridors shorter than this (in cells).
    static let minCorridorLength: Int = 6

    /// Returns the deepest reachable wide-free cell from `origin`, found via
    /// iterative DFS over 8-connected wide-free cells. Returns nil if no
    /// corridor of length ≥ minCorridorLength is reachable.
    ///
    /// "Deepest" = highest DFS visit-order index (path length from origin).
    /// Tie-broken by greater Euclidean distance from origin.
    @MainActor
    static func deepestCorridorEnd(grid: OccupancyGrid, origin: Cell) -> Cell? {
        let cols = OccupancyGrid.cols
        let rows = OccupancyGrid.rows
        let n = cols * rows
        @inline(__always) func key(_ c: Cell) -> Int { c.row * cols + c.col }

        // Clearance bitmap — true iff a square of radius clearanceRadius around
        // the cell contains no .occupied cells. Build on-demand (cached).
        var clearance = [Int8](repeating: -1, count: n)
        @inline(__always) func isWideFree(_ c: Cell) -> Bool {
            if !grid.isValid(c) { return false }
            let k = key(c)
            if clearance[k] >= 0 { return clearance[k] == 1 }
            let r = clearanceRadius
            for dr in -r...r {
                for dc in -r...r {
                    let nb = Cell(col: c.col + dc, row: c.row + dr)
                    if !grid.isValid(nb) { clearance[k] = 0; return false }
                    if grid.state(nb) == .occupied { clearance[k] = 0; return false }
                }
            }
            clearance[k] = 1
            return true
        }

        // Find a wide-free start. If origin isn't wide-free, BFS-expand outward
        // up to a small radius to find the nearest wide-free cell.
        let dfsOrigin: Cell = {
            if isWideFree(origin) { return origin }
            var queue: [Cell] = [origin]
            var seen: Set<Cell> = [origin]
            let limit = 64   // 64 cells radius ≈ 3.2m
            var hops = 0
            while !queue.isEmpty && hops < limit {
                let cur = queue.removeFirst()
                for (dc, dr) in [(-1,0),(1,0),(0,-1),(0,1)] {
                    let nb = Cell(col: cur.col + dc, row: cur.row + dr)
                    if seen.contains(nb) { continue }
                    seen.insert(nb)
                    if !grid.isValid(nb) { continue }
                    if grid.state(nb) == .occupied { continue }
                    if isWideFree(nb) { return nb }
                    queue.append(nb)
                }
                hops += 1
            }
            return origin
        }()

        // Iterative DFS. Track depth + parent chain for tie-breaking and
        // potential path reconstruction (currently we just return the leaf cell
        // and let A* reconstruct the actual driveable path).
        var visited = [Bool](repeating: false, count: n)
        var stack: [(cell: Cell, depth: Int)] = [(dfsOrigin, 0)]
        visited[key(dfsOrigin)] = true

        var bestCell:  Cell = dfsOrigin
        var bestDepth: Int  = 0
        var bestDist2: Int  = 0

        let dirs: [(Int, Int)] = [
            (-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)
        ]

        while let top = stack.popLast() {
            let dx = top.cell.col - dfsOrigin.col
            let dy = top.cell.row - dfsOrigin.row
            let d2 = dx * dx + dy * dy
            let isBetter = top.depth > bestDepth
                        || (top.depth == bestDepth && d2 > bestDist2)
            if isBetter {
                bestCell  = top.cell
                bestDepth = top.depth
                bestDist2 = d2
            }
            for (dc, dr) in dirs {
                let nb = Cell(col: top.cell.col + dc, row: top.cell.row + dr)
                if !grid.isValid(nb) { continue }
                let k = key(nb)
                if visited[k] { continue }
                if !isWideFree(nb) { continue }
                visited[k] = true
                stack.append((nb, top.depth + 1))
            }
        }

        if bestDepth < minCorridorLength { return nil }
        return bestCell
    }
}
