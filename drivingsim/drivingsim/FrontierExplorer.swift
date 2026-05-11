//
//  FrontierExplorer.swift
//  drivingsim
//
//  Frontier-based exploration: find boundary cells between free and unknown space,
//  cluster them, pick the best goal. Real-world ref: Yamauchi 1997
//  "A Frontier-Based Approach for Autonomous Exploration" — same algorithm
//  used in ROS explore_lite package.
//

import Foundation
import simd

struct FrontierCluster {
    let centroid: Cell
    let cells: [Cell]
    var size: Int { cells.count }
}

enum FrontierExplorer {

    // Minimum cluster size to be considered a valid exploration target.
    // Small isolated frontier cells are likely noise from depth map edges.
    static let minClusterSize: Int = 3

    // MARK: - Main API

    /// Find all frontier cells (free cells adjacent to unknown cells).
    static func findFrontiers(in grid: OccupancyGrid) -> [Cell] {
        var frontiers: [Cell] = []
        let cols = OccupancyGrid.cols
        let rows = OccupancyGrid.rows

        for row in 1..<(rows - 1) {
            for col in 1..<(cols - 1) {
                let cell = Cell(col: col, row: row)
                guard grid.state(cell) == .free else { continue }
                // Check 4-connected neighbors for unknown.
                let hasUnknownNeighbor =
                    grid.state(Cell(col: col,   row: row-1)) == .unknown ||
                    grid.state(Cell(col: col,   row: row+1)) == .unknown ||
                    grid.state(Cell(col: col-1, row: row))   == .unknown ||
                    grid.state(Cell(col: col+1, row: row))   == .unknown
                if hasUnknownNeighbor {
                    frontiers.append(cell)
                }
            }
        }
        return frontiers
    }

    /// Cluster frontier cells using flood-fill (8-connected).
    static func cluster(frontiers: [Cell]) -> [FrontierCluster] {
        guard !frontiers.isEmpty else { return [] }

        let frontierSet = Set(frontiers)
        var visited = Set<Cell>()
        var clusters: [FrontierCluster] = []

        for seed in frontiers {
            guard !visited.contains(seed) else { continue }

            // BFS flood-fill
            var clusterCells: [Cell] = []
            var queue = [seed]
            visited.insert(seed)

            while !queue.isEmpty {
                let current = queue.removeFirst()
                clusterCells.append(current)

                for (dr, dc) in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)] {
                    let nb = Cell(col: current.col + dc, row: current.row + dr)
                    if frontierSet.contains(nb) && !visited.contains(nb) {
                        visited.insert(nb)
                        queue.append(nb)
                    }
                }
            }

            guard clusterCells.count >= minClusterSize else { continue }

            // Centroid of cluster
            let avgCol = clusterCells.map(\.col).reduce(0, +) / clusterCells.count
            let avgRow = clusterCells.map(\.row).reduce(0, +) / clusterCells.count
            clusters.append(FrontierCluster(centroid: Cell(col: avgCol, row: avgRow),
                                            cells: clusterCells))
        }
        return clusters
    }

    /// Pick the best frontier goal. Strategy: nearest cluster centroid to `from`.
    /// Avoids the exit doorway area (south wall, high Z) when `avoidExit=true`.
    static func pickGoal(clusters: [FrontierCluster],
                         from: Cell,
                         avoidExitRow: Int? = nil) -> Cell? {
        guard !clusters.isEmpty else { return nil }

        return clusters.min { a, b in
            let da = distance(a.centroid, from)
            let db = distance(b.centroid, from)
            return da < db
        }?.centroid
    }

    // MARK: - Helpers

    static func distance(_ a: Cell, _ b: Cell) -> Float {
        let dc = Float(a.col - b.col), dr = Float(a.row - b.row)
        return (dc * dc + dr * dr).squareRoot()
    }
}
