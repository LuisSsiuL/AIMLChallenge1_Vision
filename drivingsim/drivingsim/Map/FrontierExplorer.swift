//
//  FrontierExplorer.swift
//  drivingsim
//
//  Yamauchi-style frontier detection on the occupancy grid.
//
//  Frontier cell = unknown cell with at least one free 4-neighbour. Adjacent
//  frontier cells are BFS-clustered. The MapDriver picks one cluster's
//  nearestFree cell as the next A* goal — repeatedly, until no frontiers
//  remain (room fully explored) or a target overrides.
//

import Foundation

struct FrontierExplorer {

    struct Cluster {
        let cells:       [Cell]
        let centroid:    Cell
        let nearestFree: Cell   // free cell adjacent to the cluster — A* goal
    }

    /// Scan the grid and return frontier clusters (BFS-connected).
    /// Drops clusters smaller than `minSize`; returns at most `maxClusters`,
    /// ordered by size descending.
    static func findClusters(grid: OccupancyGrid,
                             minSize: Int = 4,
                             maxClusters: Int = 12) -> [Cluster]
    {
        let cols = OccupancyGrid.cols
        let rows = OccupancyGrid.rows
        let n = cols * rows
        var visited = [Bool](repeating: false, count: n)

        @inline(__always) func key(_ c: Cell) -> Int { c.row * cols + c.col }

        @inline(__always) func isFrontier(_ c: Cell) -> Bool {
            if grid.state(c) != .unknown { return false }
            for (dc, dr) in [(-1,0),(1,0),(0,-1),(0,1)] {
                let n = Cell(col: c.col + dc, row: c.row + dr)
                if grid.isValid(n) && grid.state(n) == .free { return true }
            }
            return false
        }

        var clusters: [Cluster] = []
        for row in 0..<rows {
            for col in 0..<cols {
                let start = Cell(col: col, row: row)
                let k = key(start)
                if visited[k] { continue }
                if !isFrontier(start) { visited[k] = true; continue }

                // BFS over adjacent frontier cells.
                var cluster: [Cell] = []
                var queue: [Cell] = [start]
                visited[k] = true
                while let cur = queue.popLast() {
                    cluster.append(cur)
                    for (dc, dr) in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)] {
                        let nb = Cell(col: cur.col + dc, row: cur.row + dr)
                        if !grid.isValid(nb) { continue }
                        let nk = key(nb)
                        if visited[nk] { continue }
                        if isFrontier(nb) {
                            visited[nk] = true
                            queue.append(nb)
                        }
                    }
                }

                if cluster.count < minSize { continue }

                // Centroid (rounded).
                var sx = 0, sy = 0
                for c in cluster { sx += c.col; sy += c.row }
                let centroid = Cell(col: sx / cluster.count, row: sy / cluster.count)

                // Pick free neighbour nearest to centroid as A* goal.
                let goal = nearestFreeNeighbour(of: cluster, near: centroid, grid: grid)
                          ?? centroid
                clusters.append(Cluster(cells: cluster, centroid: centroid, nearestFree: goal))
            }
        }

        clusters.sort { $0.cells.count > $1.cells.count }
        if clusters.count > maxClusters { clusters = Array(clusters.prefix(maxClusters)) }
        return clusters
    }

    /// Free cell adjacent to any cell in `cluster`, closest to `near`.
    private static func nearestFreeNeighbour(of cluster: [Cell],
                                              near: Cell,
                                              grid: OccupancyGrid) -> Cell?
    {
        var best: Cell? = nil
        var bestD2 = Int.max
        for c in cluster {
            for (dc, dr) in [(-1,0),(1,0),(0,-1),(0,1)] {
                let nb = Cell(col: c.col + dc, row: c.row + dr)
                if !grid.isValid(nb) { continue }
                if grid.state(nb) != .free { continue }
                let dx = nb.col - near.col, dy = nb.row - near.row
                let d2 = dx * dx + dy * dy
                if d2 < bestD2 { bestD2 = d2; best = nb }
            }
        }
        return best
    }

    /// Pick the best cluster to drill into. Scores `size / sqrt(pathLen + 1)`
    /// — favours large reachable clusters; commitment ("DFS feel") is enforced
    /// by the caller, which keeps the chosen frontier until reached or blocked.
    /// Returns nil if no cluster is A*-reachable.
    static func pickBestCluster(_ clusters: [Cluster],
                                fromPose poseCell: Cell,
                                grid: OccupancyGrid) -> Cluster?
    {
        var bestScore: Float = -1
        var best: Cluster? = nil
        for c in clusters {
            let path = grid.aStar(from: poseCell, to: c.nearestFree, allowUnknown: true)
            if path.isEmpty { continue }
            let dist = Float(path.count)
            let score = Float(c.cells.count) / sqrtf(dist + 1.0)
            if score > bestScore { bestScore = score; best = c }
        }
        return best
    }

    /// BFS outward from `origin`, return first `.free` cell within `radius`.
    static func nearestFreeNear(_ origin: Cell, radius: Int, grid: OccupancyGrid) -> Cell? {
        if grid.isValid(origin) && grid.state(origin) == .free { return origin }
        var queue: [Cell] = [origin]
        var visited: Set<Cell> = [origin]
        let r2 = radius * radius
        while !queue.isEmpty {
            let cur = queue.removeFirst()
            for (dc, dr) in [(-1,0),(1,0),(0,-1),(0,1)] {
                let nb = Cell(col: cur.col + dc, row: cur.row + dr)
                if visited.contains(nb) { continue }
                visited.insert(nb)
                let dx = nb.col - origin.col, dy = nb.row - origin.row
                if dx * dx + dy * dy > r2 { continue }
                if !grid.isValid(nb) { continue }
                if grid.state(nb) == .free { return nb }
                if grid.state(nb) == .occupied { continue }
                queue.append(nb)
            }
        }
        return nil
    }
}
