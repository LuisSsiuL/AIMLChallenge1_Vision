//
//  MapDriver.swift
//  drivingsim
//
//  Camera-only 2D SLAM navigation driver. Replaces DepthDriver's zone-based
//  decision tree with a persistent occupancy grid + frontier exploration + A*.
//
//  Pipeline (per frame):
//    CVPixelBuffer → DepthAnythingV2 → depth U8
//    → DepthProjector (ray cast) → OccupancyGrid update
//    → FrontierExplorer → A* path
//    → WaypointFollower → WASD keys
//
//  Pose source: PoseEstimator (dead reckoning from own WASD output).
//  No SimScene ground truth used — same interface as real RC car.
//
//  Public surface identical to DepthDriver so ContentView/SimScene can
//  treat both interchangeably.
//

@preconcurrency import CoreML
@preconcurrency import CoreVideo
import Combine
import CoreGraphics
import Foundation
import os
import SwiftUI

@MainActor
final class MapDriver: ObservableObject {
    // Same surface as DepthDriver — ContentView treats these identically.
    @Published private(set) var keys:    Set<UInt16>       = []
    @Published private(set) var command: DrivingCommand    = .brake
    @Published private(set) var depthImage:    CGImage?    // colourised depth (FPV preview)
    @Published private(set) var occupancyImage: CGImage?   // top-down map overlay

    var forward:  Bool { keys.contains(KeyboardMonitor.W) }
    var backward: Bool { keys.contains(KeyboardMonitor.S) }
    var left:     Bool { keys.contains(KeyboardMonitor.A) }
    var right:    Bool { keys.contains(KeyboardMonitor.D) }

    // Sub-components
    let poseEstimator = PoseEstimator()
    private var grid  = OccupancyGrid()
    private var currentPath: [Cell] = []
    private var currentGoal: Cell?  = nil
    private var frontierCells: [Cell] = []
    private var personCells:   [Cell] = []   // last known person positions on map

    // Replanning cadence — don't replan every frame (A* is O(N²) worst-case).
    private let replanIntervalFrames: Int = 6     // ~0.2s @ 30Hz inference
    private var framesSinceReplan:    Int = 0

    // Map visualization cadence.
    private let mapRenderInterval: Int = 5
    private var framesSinceMapRender: Int = 0

    // Inflation radius for A* (robot half-width / resolution = ~1 cell).
    private let inflationRadius: Int = 1

    // Frames between diagnostic prints.
    private let logEveryFrames: Int = 30
    private var framesSinceLog:  Int = 0

    // YOLO seek override — when set, A* targets this world position instead of frontier.
    private(set) var seekTarget: SIMD2<Float>? = nil

    // Latest YOLO detection bbox (image-space normalized) — masked from depth obstacles.
    private var seekBox: DepthProjector.SkipBox? = nil

    // Stuck detection: rolling history of central disparity mean.
    private var dispHistory: [Float] = []
    private let dispHistorySize = 12
    private let stuckDispStdMax: Float = 4.0     // very stable depth
    private let stuckDispMeanMin: Float = 100.0  // and something close ahead
    private var stuckReverseFrames: Int = 0
    private let stuckReverseDuration = 18

    // Model (shared inference with DepthDriver pattern)
    nonisolated(unsafe) private var model: MLModel?
    nonisolated(unsafe) private var modelInputName:  String = "image"
    nonisolated(unsafe) private var modelOutputName: String = "depth"
    nonisolated(unsafe) private var modelInputW: Int = 518
    nonisolated(unsafe) private var modelInputH: Int = 518

    private let visionQueue = DispatchQueue(label: "mapdriver.vision", qos: .userInitiated)
    nonisolated private let inFlight = OSAllocatedUnfairLock<Bool>(initialState: false)
    private var running = false

    // Stats
    nonisolated(unsafe) private var statFrames   = 0
    nonisolated(unsafe) private var statSumInfer = 0.0
    nonisolated(unsafe) private var statMaxInfer = 0.0
    nonisolated(unsafe) private var statWindowT  = CACurrentMediaTime()

    init() { loadModel() }

    // MARK: - Model loading (identical to DepthDriver)

    private func loadModel() {
        let cfg = MLModelConfiguration()
        cfg.computeUnits = .all
        guard let url = Bundle.main.url(forResource: "DepthAnythingV2SmallF16",
                                        withExtension: "mlmodelc")
                     ?? Bundle.main.url(forResource: "DepthAnythingV2SmallF16",
                                        withExtension: "mlpackage") else {
            print("[MapDriver] DepthAnythingV2SmallF16 not found in bundle")
            return
        }
        do {
            let m = try MLModel(contentsOf: url, configuration: cfg)
            self.model = m
            if let inDesc = m.modelDescription.inputDescriptionsByName.first {
                self.modelInputName = inDesc.key
                if let img = inDesc.value.imageConstraint {
                    self.modelInputW = img.pixelsWide
                    self.modelInputH = img.pixelsHigh
                }
            }
            if let outDesc = m.modelDescription.outputDescriptionsByName.first {
                self.modelOutputName = outDesc.key
            }
            print("[MapDriver] model loaded \(modelInputW)x\(modelInputH)")
        } catch {
            print("[MapDriver] model load failed: \(error)")
        }
    }

    var inputSize: (width: Int, height: Int) { (modelInputW, modelInputH) }

    // MARK: - Lifecycle

    func start() { running = true }

    func stop() {
        running = false
        keys = []
        command = .brake
        grid = OccupancyGrid()
        currentPath.removeAll()
        currentGoal = nil
        frontierCells.removeAll()
        personCells.removeAll()
        seekTarget = nil
        seekBox = nil
        dispHistory.removeAll()
        stuckReverseFrames = 0
        framesSinceReplan = 0
        inFlight.withLock { $0 = false }
        print("[MapDriver] stopped, map cleared")
    }

    // MARK: - Frame submission (drop-if-busy like DepthDriver)

    nonisolated func submit(_ pixelBuffer: CVPixelBuffer) {
        let busy = inFlight.withLock { state -> Bool in
            if state { return true }
            state = true; return false
        }
        if busy { return }
        visionQueue.async { [weak self] in self?.runInference(pixelBuffer) }
    }

    // MARK: - YOLO seek override

    /// Call from SimScene/ContentView when YOLO transitions to .seek.
    /// Pass the normalized detection (cx, cy) + estimated pose to compute world target.
    /// Called by ContentView each frame when YOLO is in SEEK state.
    /// Projects detection into world space, updates seekTarget + marks person on map.
    /// Also stores image-space bbox so DepthProjector can mask out the person.
    @MainActor
    func updateSeekTarget(detectionCx: Float, detectionCy: Float,
                          detectionW: Float, detectionH: Float,
                          posePos: SIMD2<Float>, poseYaw: Float) {
        let angleH = (detectionCx - 0.5) * DepthProjector.fovDeg * Float.pi / 180.0
        let worldAngle = poseYaw + angleH
        let estimatedDist = min(8.0, max(1.0, 0.55 / max(detectionH, 0.05)))
        let tx = posePos.x + sin(worldAngle) * estimatedDist
        let tz = posePos.y - cos(worldAngle) * estimatedDist
        let target = SIMD2<Float>(tx, tz)
        seekTarget = target

        // Image-space bbox — pad slightly to ensure person edges fully masked.
        let pad: Float = 1.25
        seekBox = DepthProjector.SkipBox(
            cx: detectionCx, cy: detectionCy,
            w:  min(1.0, detectionW * pad),
            h:  min(1.0, detectionH * pad)
        )

        let cell = OccupancyGrid.worldToCell(target)
        if personCells.last != cell {
            personCells.append(cell)
            if personCells.count > 5 { personCells.removeFirst() }
        }
    }

    @MainActor
    func clearSeekTarget() {
        seekTarget = nil
        seekBox = nil
        currentPath.removeAll()
    }

    // MARK: - Inference (off main actor)

    nonisolated private func runInference(_ pixelBuffer: CVPixelBuffer) {
        guard let model = self.model else {
            inFlight.withLock { $0 = false }
            return
        }

        let t0 = CACurrentMediaTime()

        let provider: MLFeatureProvider
        do {
            let dict: [String: MLFeatureValue] = [
                modelInputName: MLFeatureValue(pixelBuffer: pixelBuffer)
            ]
            provider = try MLDictionaryFeatureProvider(dictionary: dict)
        } catch {
            print("[MapDriver] feature build failed: \(error)")
            inFlight.withLock { $0 = false }
            return
        }

        let result: MLFeatureProvider
        do {
            result = try model.prediction(from: provider)
        } catch {
            print("[MapDriver] prediction failed: \(error)")
            inFlight.withLock { $0 = false }
            return
        }

        let inferMs = (CACurrentMediaTime() - t0) * 1000.0
        statFrames += 1; statSumInfer += inferMs
        if inferMs > statMaxInfer { statMaxInfer = inferMs }

        guard let outVal  = result.featureValue(for: modelOutputName),
              let outBuf  = outVal.imageBufferValue else {
            inFlight.withLock { $0 = false }
            return
        }

        // Reuse DepthDriver's static helpers (same module).
        // Raw output of DepthAnythingV2 is DISPARITY-format (large value =
        // CLOSE). Display uses raw so the colourised preview matches the
        // standard convention (yellow=close, blue=far). DepthProjector
        // handles the polarity flip internally.
        let (depthU8, w, h) = DepthDriver.depthBufferToU8(outBuf)
        let preview = DepthDriver.colorize(depthU8: depthU8, width: w, height: h)

        logStatsIfNeeded()

        Task { @MainActor [self] in
            self.processFrame(depthU8: depthU8, w: w, h: h, preview: preview)
            self.inFlight.withLock { $0 = false }
        }
    }

    // MARK: - Main processing (MainActor)

    @MainActor
    private func processFrame(depthU8: [UInt8], w: Int, h: Int, preview: CGImage?) {
        guard running, !depthU8.isEmpty else { return }

        let pose = (pos: poseEstimator.pos, yaw: poseEstimator.yaw)

        // 1. Ray-cast depth into occupancy grid. Mask out YOLO person bbox so
        //    the tracked target doesn't get stamped as a wall.
        DepthProjector.update(depthU8: depthU8, w: w, h: h,
                              posePos: pose.pos, poseYaw: pose.yaw,
                              grid: &grid, skip: seekBox)

        // 1b. Stuck detection — rolling central-disparity stability check.
        let centralMean = DepthProjector.centralDisparityMean(depthU8: depthU8, w: w, h: h)
        dispHistory.append(centralMean)
        if dispHistory.count > dispHistorySize { dispHistory.removeFirst() }

        // 2. Replan at interval.
        framesSinceReplan += 1
        if framesSinceReplan >= replanIntervalFrames {
            framesSinceReplan = 0
            replan(pose: pose)
        }

        // 3. Follow current path (or fallback exploration).
        let inflated = grid.inflated(by: inflationRadius)
        var cmd: DrivingCommand
        if currentPath.isEmpty {
            cmd = fallbackExplore(pose: pose, grid: inflated)
        } else {
            cmd = WaypointFollower.nextCommand(
                posePos: pose.pos,
                poseYaw: pose.yaw,
                path:    &currentPath,
                grid:    inflated
            )
        }

        // 3b. Stuck override — if commanding forward but depth profile
        //     stationary + close obstacle present, robot is wedged. Reverse
        //     for a fixed window, and DO NOT integrate forward into pose
        //     (otherwise odometry drifts through the wall).
        var poseShouldIntegrate = true
        let cmdIsForward = (cmd == .forward || cmd == .forwardLeft || cmd == .forwardRight)
        if stuckReverseFrames > 0 {
            stuckReverseFrames -= 1
            cmd = .reverseLeft   // back away with a turn so next forward differs
            poseShouldIntegrate = false
        } else if cmdIsForward, dispHistory.count == dispHistorySize {
            let mean = dispHistory.reduce(0, +) / Float(dispHistory.count)
            let variance = dispHistory.reduce(0) { $0 + ($1 - mean) * ($1 - mean) } / Float(dispHistory.count)
            let std = sqrtf(variance)
            if mean > stuckDispMeanMin && std < stuckDispStdMax {
                stuckReverseFrames = stuckReverseDuration
                dispHistory.removeAll()        // reset so detection re-arms after reverse
                cmd = .reverseLeft
                poseShouldIntegrate = false
                print(String(format: "[MapDriver] STUCK detected — dispMean=%.1f std=%.2f, reversing", mean, std))
            }
        }

        // 4. Publish keys + command.
        command = cmd
        keys    = keysFor(cmd)
        depthImage = preview

        // 5. Update pose estimator. Skip integration during stuck-reverse so
        //    estimated pose stops drifting forward through the wall.
        if poseShouldIntegrate {
            poseEstimator.tickFromCommand(keys: keys, dt: 1.0 / 30.0)
        } else {
            poseEstimator.tickFromCommand(keys: [], dt: 1.0 / 30.0)
        }

        // 6. Render zoomed map overlay (6m×6m window around robot, 4 px/cell).
        framesSinceMapRender += 1
        if framesSinceMapRender >= mapRenderInterval {
            framesSinceMapRender = 0
            let robotCell = OccupancyGrid.worldToCell(pose.pos)
            occupancyImage = grid.toZoomedCGImage(
                centerCell:      robotCell,
                halfWindowCells: 30,        // 6m × 6m view
                pixelsPerCell:   4,
                pathCells:       currentPath,
                frontierCells:   frontierCells,
                personCells:     personCells
            )
        }

        // 7. Periodic diagnostics.
        framesSinceLog += 1
        if framesSinceLog >= logEveryFrames {
            framesSinceLog = 0
            let robotCell = OccupancyGrid.worldToCell(pose.pos)
            print(String(format: "[MapDriver] pose=(%.2f,%.2f,y=%.2f) cell=(%d,%d) frontiers=%d path=%d cmd=%@",
                         pose.pos.x, pose.pos.y, pose.yaw,
                         robotCell.col, robotCell.row,
                         frontierCells.count, currentPath.count, cmd.rawValue))
        }
    }

    /// Fallback when A* gives no path: drive forward if direct lookahead is clear,
    /// otherwise turn in place. Ensures robot keeps mapping rather than freezing.
    @MainActor
    private func fallbackExplore(pose: (pos: SIMD2<Float>, yaw: Float),
                                 grid: OccupancyGrid) -> DrivingCommand {
        let stepM = OccupancyGrid.resolution * 4   // 0.4m lookahead
        let fwdPos = pose.pos + SIMD2<Float>(sin(pose.yaw), -cos(pose.yaw)) * stepM
        let fwdCell = OccupancyGrid.worldToCell(fwdPos)
        if grid.state(fwdCell) == .occupied { return .forwardRight }
        return .forward
    }

    // MARK: - Replanning

    @MainActor
    private func replan(pose: (pos: SIMD2<Float>, yaw: Float)) {
        let poseCell = OccupancyGrid.worldToCell(pose.pos)

        // Determine goal.
        let goalCell: Cell?
        if let seek = seekTarget {
            goalCell = OccupancyGrid.worldToCell(seek)
        } else {
            // Frontier exploration. Compute frontiers + clusters.
            let frontiers = FrontierExplorer.findFrontiers(in: grid)
            frontierCells = frontiers
            let clusters  = FrontierExplorer.cluster(frontiers: frontiers)
            goalCell = FrontierExplorer.pickGoal(clusters: clusters, from: poseCell)
        }

        guard let goal = goalCell else {
            // No goal — keep last path or empty (fallbackExplore will drive).
            currentGoal = nil
            return
        }

        // Always replan if goal changed; else keep existing path until depleted.
        let goalChanged = currentGoal != goal
        let pathDepleted = currentPath.count <= 1
        if !goalChanged && !pathDepleted { return }

        currentGoal = goal

        // Try A* on inflated grid first (safer path), then non-inflated as fallback.
        let inflated = grid.inflated(by: inflationRadius)
        currentPath  = inflated.aStar(from: poseCell, to: goal)
        if currentPath.isEmpty {
            currentPath = grid.aStar(from: poseCell, to: goal)
        }
    }

    // MARK: - Helpers

    private func keysFor(_ cmd: DrivingCommand) -> Set<UInt16> {
        switch cmd {
        case .forward:      return [KeyboardMonitor.W]
        case .forwardLeft:  return [KeyboardMonitor.W, KeyboardMonitor.A]
        case .forwardRight: return [KeyboardMonitor.W, KeyboardMonitor.D]
        case .reverse:      return [KeyboardMonitor.S]
        case .reverseLeft:  return [KeyboardMonitor.S, KeyboardMonitor.A]
        case .reverseRight: return [KeyboardMonitor.S, KeyboardMonitor.D]
        case .brake:        return []
        }
    }

    nonisolated private func logStatsIfNeeded() {
        guard statFrames >= 30 else { return }
        let elapsed = CACurrentMediaTime() - statWindowT
        let fps     = Double(statFrames) / max(elapsed, 1e-9)
        let avg     = statSumInfer / Double(max(1, statFrames))
        print(String(format: "[MapDriver] %.0f fps | infer avg %.1f ms  max %.1f ms",
                     fps, avg, statMaxInfer))
        statFrames = 0; statSumInfer = 0; statMaxInfer = 0
        statWindowT = CACurrentMediaTime()
    }
}
