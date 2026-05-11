//
//  OccupancyGrid.swift
//  drivingsim
//
//  2D log-odds occupancy grid + A* path planner.
//  Real-world ref: Elfes/Moravec 1985, Thrun "Probabilistic Robotics" Ch.9.
//  Same algorithm used in GMapping, Hector SLAM, ROS costmap_2d.
//

import CoreGraphics
import Foundation
import simd

// MARK: - Types

struct Cell: Hashable, Equatable {
    var col: Int   // X axis (east)
    var row: Int   // Z axis (south)
}

enum CellState: UInt8 {
    case unknown  = 0
    case free     = 1
    case occupied = 2
}

// MARK: - OccupancyGrid

struct OccupancyGrid {
    // World origin = grid corner [0,0] at world (-halfW, -halfD).
    // Room is 30×20m; car spawns at approx world (0, +9).
    static let resolution: Float = 0.1      // metres per cell
    static let cols:       Int   = 320      // 32m — slight margin past 30m room
    static let rows:       Int   = 220      // 22m — slight margin past 20m room
    static let originX:    Float = -16.0    // world X of cell (0,*)
    static let originZ:    Float = -11.0    // world Z of cell (*,0)

    // Log-odds update values. Tuned so a real obstacle (hit on most frames)
    // crosses the OCCUPIED threshold in ~1 frame, but a noise hit gets
    // cleared by 2 passing free rays. lOcc 0.7, lFree -0.4 — slight lFree
    // bias keeps maps clean while still letting genuine obstacles register
    // immediately.
    private static let lOcc:  Float =  0.7
    private static let lFree: Float = -0.4
    private static let lMin:  Float = -5.0
    private static let lMax:  Float =  5.0

    // Thresholds for state classification.
    private static let occupiedThresh: Float =  0.5
    private static let freeThresh:     Float = -0.5

    // Raw log-odds storage. Flat [row * cols + col].
    private(set) var logOdds: [Float]

    init() {
        logOdds = [Float](repeating: 0.0, count: OccupancyGrid.cols * OccupancyGrid.rows)
    }

    // MARK: - Coordinate helpers

    static func worldToCell(_ world: SIMD2<Float>) -> Cell {
        let col = Int((world.x - originX) / resolution)
        let row = Int((world.y - originZ) / resolution)
        return Cell(col: col, row: row)
    }

    static func cellToWorld(_ cell: Cell) -> SIMD2<Float> {
        let x = Float(cell.col) * resolution + originX + resolution * 0.5
        let z = Float(cell.row) * resolution + originZ + resolution * 0.5
        return SIMD2<Float>(x, z)
    }

    func isValid(_ cell: Cell) -> Bool {
        cell.col >= 0 && cell.col < OccupancyGrid.cols &&
        cell.row >= 0 && cell.row < OccupancyGrid.rows
    }

    private func idx(_ cell: Cell) -> Int { cell.row * OccupancyGrid.cols + cell.col }

    // MARK: - State read

    func state(_ cell: Cell) -> CellState {
        guard isValid(cell) else { return .occupied }
        let l = logOdds[idx(cell)]
        if l > OccupancyGrid.occupiedThresh { return .occupied }
        if l < OccupancyGrid.freeThresh     { return .free     }
        return .unknown
    }

    func logOddsAt(_ cell: Cell) -> Float {
        guard isValid(cell) else { return OccupancyGrid.lMax }
        return logOdds[idx(cell)]
    }

    // MARK: - Bayesian ray-cast update (Bresenham line)

    mutating func update(from start: Cell, to end: Cell) {
        // Mark all cells along the ray FREE, endpoint OCCUPIED.
        let cells = bresenham(start, end)
        for (i, cell) in cells.enumerated() {
            guard isValid(cell) else { continue }
            let ix = idx(cell)
            if i == cells.count - 1 {
                // Hit point: mark occupied.
                logOdds[ix] = min(OccupancyGrid.lMax, logOdds[ix] + OccupancyGrid.lOcc)
            } else {
                // Free space along ray.
                logOdds[ix] = max(OccupancyGrid.lMin, logOdds[ix] + OccupancyGrid.lFree)
            }
        }
    }

    /// Stamp a single cell as occupied. Used by dense obstacle pass.
    mutating func stampOccupied(cell: Cell) {
        guard isValid(cell) else { return }
        let i = idx(cell)
        logOdds[i] = min(OccupancyGrid.lMax, logOdds[i] + OccupancyGrid.lOcc)
    }

    /// Mark cells along ray as free only (no hit endpoint). Used when ray reaches max range.
    mutating func updateFreeRay(from start: Cell, to end: Cell) {
        for cell in bresenham(start, end) {
            guard isValid(cell) else { continue }
            let ix = idx(cell)
            logOdds[ix] = max(OccupancyGrid.lMin, logOdds[ix] + OccupancyGrid.lFree)
        }
    }

    // MARK: - Obstacle inflation (for path planning)

    /// Returns a copy with obstacle cells dilated by `cells` radius. Used so A* plans a
    /// path that keeps the robot body clear of walls. Radius ≈ carHalfX / resolution = 1-2 cells.
    func inflated(by radius: Int) -> OccupancyGrid {
        var copy = self
        for row in 0..<OccupancyGrid.rows {
            for col in 0..<OccupancyGrid.cols {
                let c = Cell(col: col, row: row)
                if state(c) == .occupied {
                    for dr in -radius...radius {
                        for dc in -radius...radius {
                            let n = Cell(col: col + dc, row: row + dr)
                            if isValid(n) {
                                let ix = n.row * OccupancyGrid.cols + n.col
                                copy.logOdds[ix] = max(copy.logOdds[ix], OccupancyGrid.occupiedThresh + 0.1)
                            }
                        }
                    }
                }
            }
        }
        return copy
    }

    // MARK: - A* path planner

    /// 8-connected A* on this grid. Returns list of cells from start (exclusive) to goal (inclusive).
    /// Unknown cells are treated as passable (allows exploration into unknown space).
    func aStar(from start: Cell, to goal: Cell) -> [Cell] {
        guard isValid(start), isValid(goal) else { return [] }
        if start == goal { return [] }

        struct Node: Comparable {
            let cell: Cell
            let f: Float
            static func < (a: Node, b: Node) -> Bool { a.f < b.f }
        }

        let n = OccupancyGrid.cols * OccupancyGrid.rows
        var gScore = [Float](repeating: .infinity, count: n)
        var cameFrom = [Int: Int]()
        var openSet = [Node]()

        func key(_ c: Cell) -> Int { c.row * OccupancyGrid.cols + c.col }
        func h(_ c: Cell) -> Float {
            let dc = Float(c.col - goal.col), dr = Float(c.row - goal.row)
            return (dc * dc + dr * dr).squareRoot()
        }

        let startKey = key(start)
        gScore[startKey] = 0
        openSet.append(Node(cell: start, f: h(start)))

        let dirs: [(Int, Int)] = [
            (-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)
        ]

        while !openSet.isEmpty {
            // Pop lowest f.
            let minIdx = openSet.indices.min(by: { openSet[$0].f < openSet[$1].f })!
            let current = openSet.remove(at: minIdx)

            if current.cell == goal {
                // Reconstruct path: walk cameFrom from goal back to (but not including) start.
                // Result: [step1, step2, ..., goal] in forward order.
                var path: [Cell] = []
                var k = key(goal)
                while let prev = cameFrom[k] {
                    path.append(Cell(col: k % OccupancyGrid.cols, row: k / OccupancyGrid.cols))
                    k = prev
                }
                return path.reversed()
            }

            for (dr, dc) in dirs {
                let nb = Cell(col: current.cell.col + dc, row: current.cell.row + dr)
                guard isValid(nb), state(nb) != .occupied else { continue }
                let moveCost: Float = (dr != 0 && dc != 0) ? 1.414 : 1.0
                let nbKey = key(nb)
                let tentG = gScore[key(current.cell)] + moveCost
                if tentG < gScore[nbKey] {
                    cameFrom[nbKey] = key(current.cell)
                    gScore[nbKey] = tentG
                    openSet.append(Node(cell: nb, f: tentG + h(nb)))
                }
            }
        }
        return []   // no path found
    }

    // MARK: - Visualization

    /// Render grid as RGBA CGImage. free=light, occupied=dark, unknown=mid-gray.
    func toCGImage(robotCell: Cell? = nil,
                   pathCells: [Cell] = [],
                   frontierCells: [Cell] = [],
                   personCells: [Cell] = []) -> CGImage? {
        let W = OccupancyGrid.cols
        let H = OccupancyGrid.rows
        var rgba = [UInt8](repeating: 0, count: W * H * 4)

        let frontierSet = Set(frontierCells)
        let pathSet     = Set(pathCells)
        let personSet   = Set(personCells)

        for row in 0..<H {
            for col in 0..<W {
                let cell = Cell(col: col, row: row)
                let i = (row * W + col) * 4
                if let rc = robotCell, rc == cell {
                    // Robot = red
                    rgba[i]=220; rgba[i+1]=30; rgba[i+2]=30; rgba[i+3]=255
                } else if personSet.contains(cell) {
                    // Person target = orange
                    rgba[i]=255; rgba[i+1]=160; rgba[i+2]=20; rgba[i+3]=255
                } else if pathSet.contains(cell) {
                    // Path = green
                    rgba[i]=50; rgba[i+1]=200; rgba[i+2]=50; rgba[i+3]=255
                } else if frontierSet.contains(cell) {
                    // Frontier = blue
                    rgba[i]=50; rgba[i+1]=100; rgba[i+2]=220; rgba[i+3]=255
                } else {
                    switch state(cell) {
                    case .free:     rgba[i]=230; rgba[i+1]=230; rgba[i+2]=230; rgba[i+3]=255
                    case .occupied: rgba[i]=30;  rgba[i+1]=30;  rgba[i+2]=30;  rgba[i+3]=255
                    case .unknown:  rgba[i]=128; rgba[i+1]=128; rgba[i+2]=128; rgba[i+3]=255
                    }
                }
            }
        }

        let cs = CGColorSpaceCreateDeviceRGB()
        guard let provider = CGDataProvider(data: Data(rgba) as CFData) else { return nil }
        return CGImage(width: W, height: H,
                       bitsPerComponent: 8, bitsPerPixel: 32,
                       bytesPerRow: W * 4, space: cs,
                       bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
                       provider: provider, decode: nil, shouldInterpolate: false,
                       intent: .defaultIntent)
    }

    /// Zoomed top-down render centered on `centerCell`. Crops to `halfWindowCells`
    /// radius and renders each grid cell as `pixelsPerCell` pixels.
    /// Default: 30-cell radius (3m × 0.1m/cell) = 6m×6m window, 4 px/cell = 240×240 image.
    func toZoomedCGImage(centerCell: Cell,
                         halfWindowCells: Int = 30,
                         pixelsPerCell: Int  = 4,
                         pathCells: [Cell] = [],
                         frontierCells: [Cell] = [],
                         personCells: [Cell]   = []) -> CGImage? {
        let cellSpan = halfWindowCells * 2 + 1     // window width in cells
        let pxSide   = cellSpan * pixelsPerCell    // image side in pixels
        var rgba = [UInt8](repeating: 0, count: pxSide * pxSide * 4)

        let frontierSet = Set(frontierCells)
        let pathSet     = Set(pathCells)
        let personSet   = Set(personCells)

        for dr in -halfWindowCells...halfWindowCells {
            for dc in -halfWindowCells...halfWindowCells {
                let gridCell = Cell(col: centerCell.col + dc,
                                    row: centerCell.row + dr)

                // Pick colour for this cell.
                let r: UInt8; let g: UInt8; let b: UInt8
                if dr == 0 && dc == 0 {
                    // Robot
                    r = 230; g = 30; b = 30
                } else if personSet.contains(gridCell) {
                    r = 255; g = 160; b = 20
                } else if pathSet.contains(gridCell) {
                    r = 50; g = 200; b = 50
                } else if frontierSet.contains(gridCell) {
                    r = 50; g = 100; b = 220
                } else if !isValid(gridCell) {
                    r = 90; g = 90; b = 90                  // out-of-grid: dark gray
                } else {
                    switch state(gridCell) {
                    case .free:     r = 240; g = 240; b = 240
                    case .occupied: r = 20;  g = 20;  b = 20
                    case .unknown:  r = 140; g = 140; b = 140
                    }
                }

                // Image coords: dc=-halfWin → leftmost block, dr=-halfWin → topmost.
                let imgCol = (dc + halfWindowCells) * pixelsPerCell
                let imgRow = (dr + halfWindowCells) * pixelsPerCell

                for py in 0..<pixelsPerCell {
                    for px in 0..<pixelsPerCell {
                        let i = ((imgRow + py) * pxSide + (imgCol + px)) * 4
                        rgba[i]   = r
                        rgba[i+1] = g
                        rgba[i+2] = b
                        rgba[i+3] = 255
                    }
                }
            }
        }

        // Robot heading indicator — draw a small line from centre indicating yaw.
        // (Caller can render this — we already mark robot cell.)

        let cs = CGColorSpaceCreateDeviceRGB()
        guard let provider = CGDataProvider(data: Data(rgba) as CFData) else { return nil }
        return CGImage(width: pxSide, height: pxSide,
                       bitsPerComponent: 8, bitsPerPixel: 32,
                       bytesPerRow: pxSide * 4, space: cs,
                       bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
                       provider: provider, decode: nil, shouldInterpolate: false,
                       intent: .defaultIntent)
    }

    // MARK: - Private

    private func bresenham(_ a: Cell, _ b: Cell) -> [Cell] {
        var cells: [Cell] = []
        var x0 = a.col, y0 = a.row
        let x1 = b.col, y1 = b.row
        let dx = abs(x1 - x0), sx = x0 < x1 ? 1 : -1
        let dy = -abs(y1 - y0), sy = y0 < y1 ? 1 : -1
        var err = dx + dy

        while true {
            cells.append(Cell(col: x0, row: y0))
            if x0 == x1 && y0 == y1 { break }
            let e2 = 2 * err
            if e2 >= dy { err += dy; x0 += sx }
            if e2 <= dx { err += dx; y0 += sy }
        }
        return cells
    }
}
