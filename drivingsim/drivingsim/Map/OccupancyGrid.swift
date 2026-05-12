//
//  OccupancyGrid.swift
//  drivingsim
//
//  2D grid of FREE / OCCUPIED / UNKNOWN cells with A* path planner.
//
//  Stripped down for trajectory-only mapping: the only cells that ever become
//  FREE are those the car has physically occupied (via markFreeDisc /
//  forceFreeDisc). No depth-based obstacle stamping anymore — obstacles are
//  implicit (= "everywhere we haven't been"). A* in homing uses allowUnknown:
//  false so paths stay inside the explored corridor.
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
    static let resolution: Float = 0.05     // metres per cell (5cm — matches BEV notebook)
    static let cols:       Int   = 640      // 32m — slight margin past 30m room
    static let rows:       Int   = 440      // 22m — slight margin past 20m room
    static let originX:    Float = -16.0    // world X of cell (0,*)
    static let originZ:    Float = -11.0    // world Z of cell (*,0)

    // Log-odds bounds. lOcc unused now (no obstacle stamping); kept for legend.
    private static let lMin:  Float = -5.0
    private static let lMax:  Float =  5.0

    // Thresholds for state classification.
    static let occupiedThresh: Float =  0.5
    static let freeThresh:     Float = -0.5

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

    /// Force-clear a disc around `cell` to FREE regardless of prior log-odds.
    mutating func forceFreeDisc(at cell: Cell, radius: Int) {
        for dr in -radius...radius {
            for dc in -radius...radius {
                if dr * dr + dc * dc > radius * radius { continue }
                let c = Cell(col: cell.col + dc, row: cell.row + dr)
                if !isValid(c) { continue }
                logOdds[idx(c)] = OccupancyGrid.freeThresh - 0.5
            }
        }
    }

    /// Stamp a disc of cells as FREE. Used per-frame to record the car's
    /// trajectory corridor (pose cell + car-body neighbours = known safe).
    /// Skips cells already firmly free to avoid pointless writes.
    mutating func markFreeDisc(at cell: Cell, radius: Int) {
        for dr in -radius...radius {
            for dc in -radius...radius {
                if dr * dr + dc * dc > radius * radius { continue }
                let c = Cell(col: cell.col + dc, row: cell.row + dr)
                if !isValid(c) { continue }
                let i = idx(c)
                if logOdds[i] < OccupancyGrid.freeThresh - 0.1 { continue }
                logOdds[i] = OccupancyGrid.freeThresh - 0.2
            }
        }
    }

    /// Increment log-odds toward OCCUPIED at a cell. Used by depth-projection
    /// to mark observed obstacle hits. Clamped at +lMax.
    mutating func bumpOccupied(at cell: Cell, delta: Float = 0.30) {
        guard isValid(cell) else { return }
        let i = idx(cell)
        let v = logOdds[i] + delta
        logOdds[i] = min(v, OccupancyGrid.lMax)
    }

    /// Decrement log-odds toward FREE along a ray. Used to clear cells the
    /// camera saw through (between sensor origin and hit point).
    mutating func bumpFree(at cell: Cell, delta: Float = 0.15) {
        guard isValid(cell) else { return }
        let i = idx(cell)
        let v = logOdds[i] - delta
        logOdds[i] = max(v, OccupancyGrid.lMin)
    }

    /// Bresenham ray-cast: bump every cell between `from` and `to` (exclusive
    /// of `to`) toward FREE. Used to record what the camera saw through.
    mutating func rayCastFree(from: Cell, to: Cell, freeDelta: Float = 0.15) {
        var x0 = from.col, y0 = from.row
        let x1 = to.col,   y1 = to.row
        let dx = abs(x1 - x0), sx = x0 < x1 ? 1 : -1
        let dy = -abs(y1 - y0), sy = y0 < y1 ? 1 : -1
        var err = dx + dy
        while true {
            if x0 == x1 && y0 == y1 { break }
            bumpFree(at: Cell(col: x0, row: y0), delta: freeDelta)
            let e2 = 2 * err
            if e2 >= dy { err += dy; x0 += sx }
            if e2 <= dx { err += dx; y0 += sy }
        }
    }

    /// Diagnostic: count cells by state.
    func stateCounts() -> (free: Int, occupied: Int, unknown: Int) {
        var f = 0, o = 0, u = 0
        for l in logOdds {
            if l > OccupancyGrid.occupiedThresh { o += 1 }
            else if l < OccupancyGrid.freeThresh { f += 1 }
            else { u += 1 }
        }
        return (f, o, u)
    }

    // MARK: - Obstacle inflation (for path planning)

    /// Returns a copy with obstacle cells dilated by `cells` radius. With
    /// trajectory-only mapping there are no obstacles to dilate, but this is
    /// kept for API symmetry and future expansion.
    func inflated(by radius: Int) -> OccupancyGrid {
        guard radius > 0 else { return self }
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

    /// 8-connected A* with binary-heap open set and octile heuristic.
    /// When `allowUnknown` is true, unknown cells are passable with a mild
    /// cost penalty (legacy frontier behaviour). When false (trajectory-corridor
    /// homing), unknown cells are blocked — only FREE cells are walkable.
    func aStar(from start: Cell, to goal: Cell, allowUnknown: Bool = true) -> [Cell] {
        guard isValid(start), isValid(goal) else { return [] }
        if start == goal { return [] }

        let cols = OccupancyGrid.cols
        let n    = cols * OccupancyGrid.rows

        let unknownPenalty: Float = 0.4
        let tieBreak: Float = 1.0 + 1.0 / 1000.0
        func h(_ c: Cell) -> Float {
            let dc = abs(c.col - goal.col)
            let dr = abs(c.row - goal.row)
            let mx = max(dc, dr), mn = min(dc, dr)
            return (Float(mx) + (sqrtf(2.0) - 1.0) * Float(mn)) * tieBreak
        }
        func key(_ c: Cell) -> Int { c.row * cols + c.col }

        var gScore   = [Float](repeating: .infinity, count: n)
        var cameFrom = [Int32](repeating: -1, count: n)
        var heap     = BinaryHeap()
        heap.reserveCapacity(256)

        let startKey = key(start)
        gScore[startKey] = 0
        heap.push(f: h(start), key: Int32(startKey))

        let dirs: [(Int, Int)] = [
            (-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)
        ]
        let goalKey = key(goal)

        while let top = heap.pop() {
            let curKey = Int(top.key)
            if top.f > gScore[curKey] + h(Cell(col: curKey % cols, row: curKey / cols)) * 0.001 + 1e-3 { continue }

            if curKey == goalKey {
                var path: [Cell] = []
                var k = goalKey
                while k != startKey, cameFrom[k] >= 0 {
                    path.append(Cell(col: k % cols, row: k / cols))
                    k = Int(cameFrom[k])
                }
                return path.reversed()
            }

            let cc = Cell(col: curKey % cols, row: curKey / cols)
            let curG = gScore[curKey]
            for (dr, dc) in dirs {
                let nb = Cell(col: cc.col + dc, row: cc.row + dr)
                guard isValid(nb) else { continue }
                let s = state(nb)
                if s == .occupied { continue }
                if s == .unknown && !allowUnknown { continue }
                let baseCost: Float = (dr != 0 && dc != 0) ? sqrtf(2.0) : 1.0
                let penalty: Float = (s == .unknown) ? unknownPenalty : 0
                let nbKey = key(nb)
                let tentG = curG + baseCost + penalty
                if tentG < gScore[nbKey] {
                    cameFrom[nbKey] = Int32(curKey)
                    gScore[nbKey]   = tentG
                    heap.push(f: tentG + h(nb), key: Int32(nbKey))
                }
            }
        }
        return []   // no path found
    }

    /// Lightweight binary min-heap used by A*.
    struct BinaryHeap {
        struct Entry { let f: Float; let key: Int32 }
        private var items: [Entry] = []
        mutating func reserveCapacity(_ n: Int) { items.reserveCapacity(n) }
        var isEmpty: Bool { items.isEmpty }
        mutating func push(f: Float, key: Int32) {
            items.append(Entry(f: f, key: key))
            siftUp(items.count - 1)
        }
        mutating func pop() -> Entry? {
            guard !items.isEmpty else { return nil }
            let top = items[0]
            let last = items.removeLast()
            if !items.isEmpty {
                items[0] = last
                siftDown(0)
            }
            return top
        }
        private mutating func siftUp(_ i: Int) {
            var i = i
            while i > 0 {
                let p = (i - 1) / 2
                if items[i].f < items[p].f { items.swapAt(i, p); i = p } else { break }
            }
        }
        private mutating func siftDown(_ i: Int) {
            var i = i
            let n = items.count
            while true {
                let l = 2 * i + 1, r = 2 * i + 2
                var best = i
                if l < n && items[l].f < items[best].f { best = l }
                if r < n && items[r].f < items[best].f { best = r }
                if best == i { break }
                items.swapAt(i, best); i = best
            }
        }
    }

    // MARK: - Visualization

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
                    rgba[i]=220; rgba[i+1]=30; rgba[i+2]=30; rgba[i+3]=255
                } else if personSet.contains(cell) {
                    rgba[i]=255; rgba[i+1]=160; rgba[i+2]=20; rgba[i+3]=255
                } else if pathSet.contains(cell) {
                    rgba[i]=50; rgba[i+1]=200; rgba[i+2]=50; rgba[i+3]=255
                } else if frontierSet.contains(cell) {
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

    /// Zoomed top-down render centered on `centerCell`.
    func toZoomedCGImage(centerCell: Cell,
                         halfWindowCells: Int = 30,
                         pixelsPerCell: Int  = 4,
                         pathCells: [Cell] = [],
                         frontierCells: [Cell] = [],
                         personCells: [Cell]   = []) -> CGImage? {
        let cellSpan = halfWindowCells * 2 + 1
        let pxSide   = cellSpan * pixelsPerCell
        var rgba = [UInt8](repeating: 0, count: pxSide * pxSide * 4)

        let frontierSet = Set(frontierCells)
        let pathSet     = Set(pathCells)
        let personSet   = Set(personCells)

        for dr in -halfWindowCells...halfWindowCells {
            for dc in -halfWindowCells...halfWindowCells {
                let gridCell = Cell(col: centerCell.col + dc,
                                    row: centerCell.row + dr)

                let r: UInt8; let g: UInt8; let b: UInt8
                if dr == 0 && dc == 0 {
                    r = 230; g = 30; b = 30
                } else if personSet.contains(gridCell) {
                    r = 255; g = 160; b = 20
                } else if pathSet.contains(gridCell) {
                    r = 50; g = 200; b = 50
                } else if frontierSet.contains(gridCell) {
                    r = 50; g = 100; b = 220
                } else if !isValid(gridCell) {
                    r = 90; g = 90; b = 90
                } else {
                    switch state(gridCell) {
                    case .free:     r = 240; g = 240; b = 240
                    case .occupied: r = 20;  g = 20;  b = 20
                    case .unknown:  r = 140; g = 140; b = 140
                    }
                }

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

        let cs = CGColorSpaceCreateDeviceRGB()
        guard let provider = CGDataProvider(data: Data(rgba) as CFData) else { return nil }
        return CGImage(width: pxSide, height: pxSide,
                       bitsPerComponent: 8, bitsPerPixel: 32,
                       bytesPerRow: pxSide * 4, space: cs,
                       bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
                       provider: provider, decode: nil, shouldInterpolate: false,
                       intent: .defaultIntent)
    }
}
