//
//  MapDriver.swift
//  drivingsim
//
//  Trajectory-only mapper + A* return-home planner.
//
//  Roles by tour state:
//    .exploring          — passive. Records pose-disc cells as FREE. autoSeekPy drives.
//    .approachingTarget  — passive. Same recording. autoSeekPy seeks to person.
//    .homing             — active. A* on trajectory corridor → WaypointFollower → WASD.
//    .completed          — active. Brakes.
//
//  No depth-perception in this driver. Truth pose is fed in from SimScene.
//  Map = "cells the car has physically occupied" — guaranteed obstacle-free.
//

import Combine
import CoreGraphics
import Foundation
import simd
import SwiftUI

@MainActor
final class MapDriver: ObservableObject {
    // Public WASD surface — only non-empty when this driver is in control
    // (.homing / .completed). In exploring/approachingTarget the keys stay
    // empty so SimScene's OR-merge picks up autoSeekPy instead.
    @Published private(set) var keys:    Set<UInt16>    = []
    @Published private(set) var command: DrivingCommand = .brake
    @Published private(set) var occupancyImage: CGImage?    // map preview

    var forward:  Bool { keys.contains(KeyboardMonitor.W) }
    var backward: Bool { keys.contains(KeyboardMonitor.S) }
    var left:     Bool { keys.contains(KeyboardMonitor.A) }
    var right:    Bool { keys.contains(KeyboardMonitor.D) }

    // Sub-components
    let poseEstimator = PoseEstimator()
    private var grid  = OccupancyGrid()
    private var currentPath: [Cell] = []
    private var currentGoal: Cell?  = nil
    private var personCells:   [Cell] = []

    // Trajectory recording — disc radius around current pose every frame.
    // Car is 15×20cm, grid = 5cm/cell → radius 6 (30cm disc) = body + margin.
    private let trajectoryStampRadius: Int = 6

    // Replanning cadence in homing.
    private let replanIntervalFrames: Int = 6   // ~0.2s @ 30Hz tick
    private var framesSinceReplan:    Int = 0

    // Map render cadence.
    private let mapRenderInterval: Int = 5
    private var framesSinceMapRender: Int = 0

    // Frames between diagnostic prints.
    private let logEveryFrames: Int = 30
    private var framesSinceLog:  Int = 0

    // Truth pose feed from SimScene replaces dead reckoning.
    var useTruthPose: Bool = true

    // Tour state machine.
    // exploring / approachingTarget / homing / completed → used by .autoMap & .mapMetric.
    // scanning / driving → used by .mapExplore (discrete scan-then-drive).
    enum TourState { case exploring, approachingTarget, homing, completed,
                          scanning, driving, idle,
                          // Explore-auto sub-mode (.mapExplore + autoActive):
                          exploreSweepRotate, exploreSweepDwell,
                          explorePlanFrontier, exploreDrive, exploreDone }
    @Published private(set) var tourState: TourState = .exploring
    /// Inference gate — when true, ContentView's FPV pump should call
    /// `depth.submit(...)`. False during rotation + .driving / .homing to save
    /// CoreML compute. Only relevant in .mapExplore.
    @Published private(set) var wantsInference: Bool = false
    // mapExplore selector — set from ContentView on mode change so MapDriver
    // knows to enter sweepingInitial instead of exploring on start.
    var exploreMode: Bool = false
    // Explore-auto toggle. Manual sub-mode = false (default); Automatic = true.
    @Published private(set) var autoActive: Bool = false
    // 45° step sweep params for explore-auto.
    private let exploreSweepSteps:    Int   = 8
    private let exploreSweepStepRad:  Float = .pi / 4     // 45°
    private let exploreYawTolerance:  Float = 0.05
    private let exploreDwellTimeoutF: Int   = 300          // 5s @ 60Hz safety cap
    private var exploreSweepIndex:    Int   = 0
    private var exploreSweepStartYaw: Float = 0
    private var exploreSweepTargetYaw:Float = 0
    private var exploreSweepBaselineFrameID: UInt64 = 0
    private var exploreSweepDwellFrames: Int = 0
    private var exploreReplanFrames:  Int = 0
    private var startCell: Cell? = nil
    private(set) var markedTarget: Cell? = nil
    private let homeReachM:    Float = 0.5
    private let targetReachM:  Float = 0.6
    private let minExploreM:   Float = 2.0   // must roam before approaching
    private let minHomingFrames:   Int = 30
    private let minApproachFrames: Int = 30
    private var maxDistFromStart: Float = 0
    private var framesInHoming: Int = 0
    private var framesAtTarget: Int = 0

    // mapExplore: scan-then-drive cycle.
    // tick() runs @ 60Hz (called from SimScene.tick via updateTruthPose).
    // maxSteerRate = 1.8 rad/s → 18 frames @ 60Hz × 1.8 ≈ 0.54 rad ≈ 31°.
    private let sweepSteps:        Int = 12          // 360° / 30°
    private let sweepRotateFrames: Int = 18          // 30° per step
    private let sweepDwellFrames:  Int = 15          // 0.25s settle + inference wait
    private let sweepInferenceTimeoutFrames: Int = 180 // 3s max wait per step (forgiving)
    private var sweepStepIndex:    Int = 0
    private var sweepRotateRemaining: Int = 0
    private var sweepDwellRemaining:  Int = 0
    private var sweepWaitingInference: Bool = false
    private var sweepInferenceWaitFrames: Int = 0
    private var sweepBaselineFrameID: UInt64 = 0
    private let targetApproachAdjacentRadius: Int = 8   // cells (~40cm at 0.05m/cell)
    private let seekConfirmFrames: Int = 8
    private var seekConfirmCount: Int = 0
    private var corridorGoalCell: Cell? = nil

    // The map driver is registered as @StateObject in ContentView; `submit`
    // exists only so the FPV pump can still call into it without dispatching
    // separately. We don't actually run any inference here.
    nonisolated func submit(_ pixelBuffer: CVPixelBuffer) { /* no-op */ }
    var inputSize: (width: Int, height: Int) { (518, 518) }

    private var running = false

    // MARK: - Lifecycle

    func start() {
        running = true
        // Explore mode is now pure mapping — no autonomous sweep/drive. Sits in
        // .idle, publishes empty WASD, and only runs inference on `I`-key press.
        tourState = exploreMode ? .idle : .exploring
        sweepStepIndex = 0
        sweepRotateRemaining = 0
        sweepDwellRemaining = 0
        sweepWaitingInference = false
        sweepInferenceWaitFrames = 0
        sweepBaselineFrameID = 0
        corridorGoalCell = nil
        seekConfirmCount = 0
        wantsInference = false
    }

    func stop() {
        running = false
        keys = []
        command = .brake
        grid = OccupancyGrid()
        currentPath.removeAll()
        currentGoal = nil
        personCells.removeAll()
        framesSinceReplan = 0
        tourState = .exploring
        startCell = nil
        markedTarget = nil
        maxDistFromStart = 0
        framesInHoming = 0
        framesAtTarget = 0
        sweepStepIndex = 0
        sweepRotateRemaining = 0
        sweepDwellRemaining = 0
        sweepWaitingInference = false
        sweepInferenceWaitFrames = 0
        sweepBaselineFrameID = 0
        corridorGoalCell = nil
        seekConfirmCount = 0
        wantsInference = false
        autoActive = false
        exploreSweepIndex = 0
        exploreSweepDwellFrames = 0
        exploreReplanFrames = 0
        print("[MapDriver] stopped, map cleared")
    }

    // MARK: - External hooks (called from ContentView / SimScene)

    /// Truth pose from SimScene (sim only). Drives all map updates.
    @MainActor
    func updateTruthPose(_ pos: SIMD2<Float>, yaw: Float) {
        if useTruthPose {
            poseEstimator.reset(pos: pos, yaw: yaw)
        }
        tick(pose: (pos: pos, yaw: yaw))
    }

    /// Real-hardware entry: dead-reckon pose from commanded WASD, then tick.
    /// Used when SimScene isn't running (ESP32-CAM frame source). Caller must
    /// have already set `useTruthPose = false`.
    @MainActor
    func tickDeadReckoned(keys: Set<UInt16>, dt: Float) {
        poseEstimator.tickFromCommand(keys: keys, dt: dt)
        tick(pose: (pos: poseEstimator.pos, yaw: poseEstimator.yaw))
    }

    /// Called when YOLOPy detects the person. We use it only to mark the
    /// target cell so we know when the car has reached it (P2→P3 trigger).
    /// Target world position estimated from bbox center + height.
    /// One-shot inference request — Explore mode only. Sets `wantsInference`
    /// for one CoreML pass; cleared in `consumeMetricFrame` once the frame is
    /// ingested into the grid.
    @MainActor
    func requestInferenceOnce() {
        guard exploreMode else { return }
        wantsInference = true
    }

    /// Toggle automatic explore mode. Manual ↔ Automatic.
    /// Manual: tourState=.idle, no autonomous WASD. Press `I` for one-shot scan.
    /// Automatic: 30°×12 sweep → frontier A* → drive → re-sweep.
    @MainActor
    func toggleAutoExplore() {
        guard exploreMode else { return }
        if autoActive {
            autoActive = false
            tourState = .idle
            keys = []
            command = .brake
            currentPath.removeAll()
            corridorGoalCell = nil
            wantsInference = false
            print("[MapDriver] auto-explore OFF → manual")
        } else {
            autoActive = true
            exploreSweepIndex = 0
            exploreSweepStartYaw = poseEstimator.yaw
            exploreSweepTargetYaw = exploreSweepStartYaw
            exploreSweepBaselineFrameID = latestMetricFrameID
            exploreSweepDwellFrames = 0
            exploreReplanFrames = 0
            currentPath.removeAll()
            corridorGoalCell = nil
            tourState = .exploreSweepRotate
            print("[MapDriver] auto-explore ON → sweep (8 × 45°)")
        }
    }

    /// Wrap an angle into (-π, π].
    @inline(__always)
    private func wrapAngle(_ a: Float) -> Float {
        var x = a
        while x >  Float.pi { x -= 2 * .pi }
        while x < -Float.pi { x += 2 * .pi }
        return x
    }

    @MainActor
    func updateSeekTarget(detectionCx: Float, detectionCy: Float,
                          detectionW: Float, detectionH: Float,
                          posePos: SIMD2<Float>, poseYaw: Float) {
        // Explore mode = pure manual + mapping. No YOLO-driven autonomous
        // seek; ignore detections.
        if exploreMode { return }
        // Match the same camera FOV the FPV renderer uses (75°).
        let hfovDeg: Float = 75.0
        let angleH = (detectionCx - 0.5) * hfovDeg * Float.pi / 180.0
        let worldAngle = poseYaw + angleH
        let estimatedDist = min(8.0, max(1.0, 0.55 / max(detectionH, 0.05)))
        let tx = posePos.x + sin(worldAngle) * estimatedDist
        let tz = posePos.y - cos(worldAngle) * estimatedDist
        let target = SIMD2<Float>(tx, tz)

        let cell = OccupancyGrid.worldToCell(target)
        if personCells.last != cell {
            personCells.append(cell)
            if personCells.count > 5 { personCells.removeFirst() }
        }

        // mapExplore: just bookmark the target cell — the goal-picker in
        // scanning state will route to it once detection is confirmed. The
        // approachingTarget transition happens when the car reaches the
        // nearby free cell (handled in .driving completion).
        //
        // .autoMap / .mapMetric: legacy behaviour — flip straight into
        // .approachingTarget once roam-gate satisfied.
        let canBookmark = exploreMode
                          ? (tourState == .scanning || tourState == .driving)
                          : (tourState == .exploring && maxDistFromStart >= minExploreM)
        if markedTarget == nil && canBookmark {
            seekConfirmCount += 1
            if seekConfirmCount >= seekConfirmFrames {
                markedTarget = cell
                if !exploreMode {
                    tourState = .approachingTarget
                    framesAtTarget = 0
                    currentPath.removeAll()
                }
                print("[MapDriver] target sighted @ cell=(\(cell.col),\(cell.row)) — \(exploreMode ? "bookmarked (explore)" : "approaching")")
            }
        } else if markedTarget == nil {
            seekConfirmCount = 0
        }

        // Bbox-based near-target trigger. Counts toward P2→P3 even if the
        // truth-pose distance is noisy.
        if tourState == .approachingTarget && detectionH >= bboxNearH {
            framesAtTarget += 1
            if framesAtTarget >= minApproachFrames {
                tourState = .homing
                framesInHoming = 0
                currentPath.removeAll()
                print(String(format: "[MapDriver] bbox-near target (h=%.2f) — heading home", detectionH))
            }
        }
    }

    // Bbox-based "near target" threshold — bbox h ≥ this counts toward P2→P3.
    private let bboxNearH: Float = 0.55

    /// No-op kept for ContentView call site compatibility. With trajectory-
    /// only mapping there's no per-frame depth pass to mask.
    @MainActor
    func clearSeekBox() { /* no-op */ }

    /// Project a metric depth frame into the occupancy grid (.mapMetric mode).
    /// Only ≤5m hits stamped. Caller must pass meters straight from the metric
    /// CoreML model (NOT min-max normalized).
    @MainActor
    func ingestMetricDepth(meters: [Float], width: Int, height: Int,
                           posePos: SIMD2<Float>, poseYaw: Float)
    {
        let hits = DepthProjector.stamp(meters: meters, width: width, height: height,
                                        posePos: posePos, poseYaw: poseYaw,
                                        grid: &grid)
        metricHitsLastFrame = hits
    }
    private(set) var metricHitsLastFrame: Int = 0
    /// Most recent metric frame consumed. Used by .scanning to detect when
    /// the awaited CoreML inference has completed.
    private(set) var latestMetricFrameID: UInt64 = 0
    private var lastIngestedMetricFrameID: UInt64 {
        get { latestMetricFrameID } set { latestMetricFrameID = newValue }
    }
    func consumeMetricFrame(_ snap: DepthDriver.MetricFrame, posePos: SIMD2<Float>, poseYaw: Float) {
        if snap.frameID == lastIngestedMetricFrameID { return }
        lastIngestedMetricFrameID = snap.frameID
        ingestMetricDepth(meters: snap.meters, width: snap.width, height: snap.height,
                          posePos: snap.posePos, poseYaw: snap.poseYaw)
        // Explore manual: clear single-shot inference gate once stamped.
        // Auto: gate is managed by the sweep state machine, leave alone.
        if exploreMode && !autoActive { wantsInference = false }
    }

    /// Wipe target on mode change.
    @MainActor
    func clearSeekTarget() {
        markedTarget = nil
        currentPath.removeAll()
    }

    // MARK: - Per-frame tick (driven by truth pose feed)

    @MainActor
    private func tick(pose: (pos: SIMD2<Float>, yaw: Float)) {
        guard running else { return }

        // Capture start cell on first frame.
        if startCell == nil {
            let s = OccupancyGrid.worldToCell(pose.pos)
            startCell = s
            grid.forceFreeDisc(at: s, radius: 5)
            print("[MapDriver] start cell locked @ (\(s.col),\(s.row))  world=(\(pose.pos.x), \(pose.pos.y))")
        }

        // Track max distance from start (gate for approach transition).
        if let s = startCell {
            let sw = OccupancyGrid.cellToWorld(s)
            let dx = sw.x - pose.pos.x, dz = sw.y - pose.pos.y
            let dist = sqrtf(dx * dx + dz * dz)
            if dist > maxDistFromStart { maxDistFromStart = dist }

            // Homing → completed.
            if tourState == .homing {
                framesInHoming += 1
                if framesInHoming >= minHomingFrames && dist < homeReachM {
                    tourState = .completed
                    currentPath.removeAll()
                    print(String(format: "[MapDriver] arrived home — tour complete (%d frames homing, dist=%.2fm)", framesInHoming, dist))
                }
            }
        }

        // Approach → homing.
        if tourState == .approachingTarget, let tgt = markedTarget {
            let tw = OccupancyGrid.cellToWorld(tgt)
            let dx = tw.x - pose.pos.x, dz = tw.y - pose.pos.y
            let distToTgt = sqrtf(dx * dx + dz * dz)
            if distToTgt < targetReachM {
                framesAtTarget += 1
                if framesAtTarget >= minApproachFrames {
                    tourState = .homing
                    framesInHoming = 0
                    currentPath.removeAll()
                    print(String(format: "[MapDriver] reached target (dist=%.2fm) — heading home", distToTgt))
                }
            } else {
                framesAtTarget = 0
            }
        }

        let poseCell = OccupancyGrid.worldToCell(pose.pos)

        // Trajectory corridor — stamp wherever we drive (any non-terminal state).
        if tourState != .completed {
            grid.markFreeDisc(at: poseCell, radius: trajectoryStampRadius)
        }

        var cmd: DrivingCommand = .brake
        // Overlay: highlight current corridor goal so it's visible on the map.
        var frontierOverlay: [Cell] = corridorGoalCell.map { [$0] } ?? []

        switch tourState {
        case .scanning:
            // Discrete sweep: rotate → wait for inference → dwell → repeat.
            // wantsInference gates ContentView's depth.submit() call.
            if sweepRotateRemaining > 0 {
                cmd = .rotateLeft
                sweepRotateRemaining -= 1
                wantsInference = false
                if sweepRotateRemaining == 0 {
                    // Just finished rotating — kick off inference for this step.
                    sweepWaitingInference = true
                    sweepInferenceWaitFrames = 0
                    sweepBaselineFrameID = latestMetricFrameID
                }
            } else if sweepWaitingInference {
                cmd = .brake
                wantsInference = true
                sweepInferenceWaitFrames += 1
                let curID = latestMetricFrameID
                let timedOut = sweepInferenceWaitFrames >= sweepInferenceTimeoutFrames
                if curID > sweepBaselineFrameID || timedOut {
                    if timedOut && curID == sweepBaselineFrameID {
                        print("[MapDriver] step \(sweepStepIndex+1) inference TIMEOUT")
                    }
                    sweepWaitingInference = false
                    sweepDwellRemaining = sweepDwellFrames
                    sweepStepIndex += 1
                }
            } else if sweepDwellRemaining > 0 {
                cmd = .brake
                wantsInference = false
                sweepDwellRemaining -= 1
            } else if sweepStepIndex >= sweepSteps {
                // Sweep complete → pick goal → enter .driving.
                wantsInference = false
                if let goal = pickExploreGoal(from: poseCell) {
                    currentPath = grid.aStar(from: poseCell, to: goal, allowUnknown: true)
                    currentGoal = goal
                    corridorGoalCell = goal
                    if currentPath.isEmpty {
                        // Goal unreachable via A* — coast forward instead of
                        // immediately bailing home (gives a chance to discover
                        // new ground around the obstacle).
                        tourState = .driving
                        framesSinceReplan = 0
                        currentPath = []
                        corridorGoalCell = nil
                        print("[MapDriver] sweep done; goal A*-unreachable → forward coast")
                    } else {
                        tourState = .driving
                        framesSinceReplan = 0
                        print("[MapDriver] sweep done; goal @(\(goal.col),\(goal.row)), path=\(currentPath.count)")
                    }
                } else {
                    // No corridor found → drive forward to explore.
                    tourState = .driving
                    framesSinceReplan = 0
                    currentPath = []
                    corridorGoalCell = nil
                    print("[MapDriver] sweep done; no corridor → forward coast")
                }
            } else {
                // Start next rotation step.
                sweepRotateRemaining = sweepRotateFrames
                cmd = .rotateLeft
                wantsInference = false
            }

        case .driving:
            wantsInference = false
            let reachedGoal: Bool = {
                guard let g = corridorGoalCell else { return false }
                let dc = poseCell.col - g.col, dr = poseCell.row - g.row
                return (dc * dc + dr * dr) <= 4
            }()
            // Check for target adjacency (regardless of path state).
            if let tgt = markedTarget {
                let dc = poseCell.col - tgt.col, dr = poseCell.row - tgt.row
                if (dc * dc + dr * dr) <= targetApproachAdjacentRadius * targetApproachAdjacentRadius {
                    tourState = .approachingTarget
                    framesAtTarget = 0
                    currentPath.removeAll()
                    framesSinceReplan = 0
                    print("[MapDriver] target adjacent — entering approachingTarget")
                    break
                }
            }
            if reachedGoal {
                // Done with this corridor → re-scan from new vantage.
                tourState = .scanning
                sweepStepIndex = 0
                sweepRotateRemaining = sweepRotateFrames
                sweepDwellRemaining = 0
                sweepWaitingInference = false
                currentPath.removeAll()
                corridorGoalCell = nil
                print("[MapDriver] reached goal — re-scanning")
            } else if !currentPath.isEmpty {
                cmd = WaypointFollower.nextCommand(
                    posePos: pose.pos, poseYaw: pose.yaw,
                    path: &currentPath, grid: grid)
            } else {
                // No A* path. Coast forward — but stop if wall immediately ahead.
                let probeAhead = poseCellForward(from: poseCell, yaw: pose.yaw, distance: 10)
                if grid.isValid(probeAhead) && grid.state(probeAhead) == .occupied {
                    // Blocked → re-scan from here.
                    tourState = .scanning
                    sweepStepIndex = 0
                    sweepRotateRemaining = sweepRotateFrames
                    sweepDwellRemaining = 0
                    sweepWaitingInference = false
                    currentPath.removeAll()
                    corridorGoalCell = nil
                    print("[MapDriver] coast blocked → re-scanning")
                } else {
                    cmd = .forward
                }
            }

        case .approachingTarget:
            if let tgt = markedTarget {
                framesSinceReplan += 1
                let needReplan = currentPath.isEmpty || framesSinceReplan >= replanIntervalFrames
                if needReplan {
                    framesSinceReplan = 0
                    if let goal = FrontierExplorer.nearestFreeNear(tgt,
                                                                    radius: targetApproachAdjacentRadius,
                                                                    grid: grid) {
                        currentPath = grid.aStar(from: poseCell, to: goal, allowUnknown: true)
                        currentGoal = goal
                    } else {
                        currentPath.removeAll()
                    }
                }
                if !currentPath.isEmpty {
                    cmd = WaypointFollower.nextCommand(
                        posePos: pose.pos, poseYaw: pose.yaw,
                        path: &currentPath, grid: grid)
                }
            }

        case .homing:
            if let start = startCell {
                framesSinceReplan += 1
                if framesSinceReplan >= replanIntervalFrames || currentPath.isEmpty {
                    framesSinceReplan = 0
                    let path = grid.aStar(from: poseCell, to: start, allowUnknown: false)
                    currentPath = path
                    currentGoal = start
                }
                if !currentPath.isEmpty {
                    cmd = WaypointFollower.nextCommand(
                        posePos: pose.pos, poseYaw: pose.yaw,
                        path: &currentPath, grid: grid)
                } else {
                    cmd = fallbackToward(start: start, pose: pose)
                }
            }

        case .completed:
            cmd = .brake

        case .exploring:
            cmd = .brake   // autoSeekPy drives during exploring (.autoMap/.mapMetric)

        case .idle:
            cmd = .brake   // Explore manual — operator drives; MapDriver silent.

        case .exploreSweepRotate:
            // Rotate to next 45° target. While rotating, do NOT request
            // inference (CoreML idle until we settle on the next step).
            let target = exploreSweepStartYaw
                       - Float(exploreSweepIndex + 1) * exploreSweepStepRad
            exploreSweepTargetYaw = target
            let err = wrapAngle(target - pose.yaw)
            if abs(err) < exploreYawTolerance {
                // Settled at this step's yaw → arm inference and dwell.
                cmd = .brake
                exploreSweepBaselineFrameID = latestMetricFrameID
                exploreSweepDwellFrames = 0
                wantsInference = true
                tourState = .exploreSweepDwell
            } else {
                cmd = err < 0 ? .rotateLeft : .rotateRight
                wantsInference = false
            }

        case .exploreSweepDwell:
            // Hold position until one fresh CoreML frame is ingested into the
            // grid. Timeout exists only as a stall safety net.
            cmd = .brake
            wantsInference = true
            exploreSweepDwellFrames += 1
            let gotFrame = latestMetricFrameID > exploreSweepBaselineFrameID
            let timedOut = exploreSweepDwellFrames >= exploreDwellTimeoutF
            if gotFrame || timedOut {
                if timedOut && !gotFrame {
                    print("[MapDriver] sweep step \(exploreSweepIndex+1)/\(exploreSweepSteps) TIMEOUT")
                } else {
                    print("[MapDriver] sweep step \(exploreSweepIndex+1)/\(exploreSweepSteps) ingested")
                }
                wantsInference = false
                exploreSweepIndex += 1
                if exploreSweepIndex >= exploreSweepSteps {
                    tourState = .explorePlanFrontier
                } else {
                    tourState = .exploreSweepRotate
                }
            }

        case .explorePlanFrontier:
            // Brake while planning. BFS-flood from pose over free cells. Pick
            // the farthest free cell adjacent to an unknown cell as the goal.
            // BFS itself reconstructs the path — no A* fallback needed.
            cmd = .brake
            wantsInference = false
            if let pick = farthestReachableFrontierFree(from: poseCell), !pick.path.isEmpty {
                currentPath = pick.path
                currentGoal = pick.goal
                corridorGoalCell = pick.goal
                exploreReplanFrames = 0
                tourState = .exploreDrive
                print("[MapDriver] farthest frontier goal=(\(pick.goal.col),\(pick.goal.row)) bfsDist=\(pick.dist) pathLen=\(pick.path.count)")
            } else {
                print("[MapDriver] no reachable frontier — stopping (one-sweep mode)")
                autoActive = false
                tourState = .idle
                keys = []
                command = .brake
            }

        case .exploreDrive:
            // Drive the A* path. No CoreML inference during drive (saves
            // compute; map updated on next sweep).
            wantsInference = false
            let reached: Bool = {
                guard let g = corridorGoalCell else { return false }
                let dc = poseCell.col - g.col, dr = poseCell.row - g.row
                return (dc * dc + dr * dr) <= 4
            }()
            if reached {
                exploreSweepIndex = 0
                exploreSweepStartYaw = pose.yaw
                currentPath.removeAll()
                corridorGoalCell = nil
                tourState = .exploreSweepRotate
                cmd = .brake
                print("[MapDriver] arrived — re-sweeping")
            } else if !currentPath.isEmpty {
                exploreReplanFrames += 1
                if exploreReplanFrames >= replanIntervalFrames, let g = corridorGoalCell {
                    exploreReplanFrames = 0
                    let fresh = grid.aStar(from: poseCell, to: g, allowUnknown: true)
                    if !fresh.isEmpty { currentPath = fresh }
                }
                cmd = WaypointFollower.nextCommand(
                    posePos: pose.pos, poseYaw: pose.yaw,
                    path: &currentPath, grid: grid)
                if cmd == .brake {
                    // Path emptied or blocked — restart sweep cycle.
                    exploreSweepIndex = 0
                    exploreSweepStartYaw = pose.yaw
                    currentPath.removeAll()
                    corridorGoalCell = nil
                    tourState = .exploreSweepRotate
                    print("[MapDriver] drive halted — re-sweeping")
                }
            } else {
                // Unexpected empty path — re-sweep.
                exploreSweepIndex = 0
                exploreSweepStartYaw = pose.yaw
                corridorGoalCell = nil
                tourState = .exploreSweepRotate
                cmd = .brake
            }

        case .exploreDone:
            // Don't terminate — kick back into sweep so car never freezes.
            tourState = .exploreSweepRotate
            exploreSweepStartYaw = pose.yaw
            exploreSweepIndex = 0
            cmd = .rotateLeft
            wantsInference = true
        }

        // Publish WASD:
        //   .autoMap/.mapMetric — homing/completed
        //   .mapExplore auto    — sweep/drive/done (planning is brake-only)
        //   .mapExplore manual  — never (.idle)
        let mapDriverInControl =
               tourState == .homing
            || tourState == .completed
            || tourState == .exploreSweepRotate
            || tourState == .exploreSweepDwell
            || tourState == .explorePlanFrontier
            || tourState == .exploreDrive
            || tourState == .exploreDone
        if mapDriverInControl {
            command = cmd
            keys    = keysFor(cmd)
        } else {
            command = .brake
            keys    = []
        }

        // Render map every N frames.
        framesSinceMapRender += 1
        if framesSinceMapRender >= mapRenderInterval {
            framesSinceMapRender = 0
            let personOverlay = markedTarget.map { [$0] } ?? []
            occupancyImage = grid.toZoomedCGImage(
                centerCell:      poseCell,
                halfWindowCells: 100,                // 100 * 0.05m = 5m radius (10m × 10m window)
                pixelsPerCell:   2,                  // 201 * 2px ≈ 402px panel
                pathCells:       currentPath,
                frontierCells:   frontierOverlay,
                personCells:     personOverlay
            )
        }

        framesSinceLog += 1
        if framesSinceLog >= logEveryFrames {
            framesSinceLog = 0
            let counts = grid.stateCounts()
            let stateStr: String = {
                switch tourState {
                case .exploring:         return "exploring"
                case .approachingTarget: return "approachingTarget"
                case .homing:            return "homing"
                case .completed:         return "completed"
                case .scanning:          return "scanning(\(sweepStepIndex)/\(sweepSteps))"
                case .driving:           return "driving"
                case .idle:              return "idle(explore-manual)"
                case .exploreSweepRotate:return "exploreSweepRotate(\(exploreSweepIndex)/\(exploreSweepSteps))"
                case .exploreSweepDwell: return "exploreSweepDwell(\(exploreSweepIndex)/\(exploreSweepSteps))"
                case .explorePlanFrontier:return "explorePlanFrontier"
                case .exploreDrive:      return "exploreDrive"
                case .exploreDone:       return "exploreDone"
                }
            }()
            // Compact heartbeat — one line per 30 frames. Comment out for silence.
            print(String(format: "[MapDriver] %@ cmd=%@ free=%d unk=%d",
                         stateStr, cmd.rawValue, counts.free, counts.unknown))
        }
    }

    /// Heading-toward-start fallback if A* corridor is briefly disconnected.
    @MainActor
    private func fallbackToward(start: Cell, pose: (pos: SIMD2<Float>, yaw: Float)) -> DrivingCommand {
        let g = OccupancyGrid.cellToWorld(start)
        let dx = g.x - pose.pos.x
        let dz = g.y - pose.pos.y
        var dyaw = atan2(dx, -dz) - pose.yaw
        while dyaw >  Float.pi { dyaw -= 2 * Float.pi }
        while dyaw < -Float.pi { dyaw += 2 * Float.pi }
        let turnThresh: Float = 0.25
        if      dyaw >  turnThresh { return .forwardRight }
        else if dyaw < -turnThresh { return .forwardLeft  }
        else                       { return .forward      }
    }

    // MARK: - Helpers

    /// Single forward cell `distance` ahead of pose, used for coast-ahead
    /// blocked-check.
    @MainActor
    private func poseCellForward(from poseCell: Cell, yaw: Float, distance: Int) -> Cell {
        let fx = sin(yaw); let fz = -cos(yaw)
        let col = poseCell.col + Int((fx * Float(distance)).rounded())
        let row = poseCell.row + Int((fz * Float(distance)).rounded())
        return Cell(col: col, row: row)
    }

    /// BFS-flood from poseCell over `.free` cells. Return the farthest free
    /// cell that has at least one `.unknown` 4-neighbor (= reachable frontier
    /// boundary) AND the reconstructed BFS path to it. Bypasses A* entirely.
    @MainActor
    private func farthestReachableFrontierFree(from start: Cell)
        -> (goal: Cell, dist: Int, path: [Cell])?
    {
        let cols = OccupancyGrid.cols
        let rows = OccupancyGrid.rows
        let n = cols * rows
        var visited = [Bool](repeating: false, count: n)
        var parent  = [Int32](repeating: -1, count: n)
        let startKey = start.row * cols + start.col
        guard grid.isValid(start), grid.state(start) == .free else { return nil }
        visited[startKey] = true
        var queue: [Cell] = [start]
        var qHead = 0
        var bestKey: Int = -1
        var bestDist = 0
        var distMap = [Int32](repeating: 0, count: n)
        let nbrs8: [(Int,Int)] = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,1),(1,0)]
        let nbrs4: [(Int,Int)] = [(-1,0),(1,0),(0,-1),(0,1)]
        while qHead < queue.count {
            let cur = queue[qHead]; qHead += 1
            let curKey = cur.row * cols + cur.col
            let curDist = Int(distMap[curKey])
            // Check if frontier-adjacent.
            for (dc, dr) in nbrs4 {
                let nb = Cell(col: cur.col + dc, row: cur.row + dr)
                if !grid.isValid(nb) { continue }
                if grid.state(nb) == .unknown {
                    if curDist > bestDist { bestDist = curDist; bestKey = curKey }
                    break
                }
            }
            // Expand through free cells (8-conn for shorter paths).
            for (dc, dr) in nbrs8 {
                let nb = Cell(col: cur.col + dc, row: cur.row + dr)
                if !grid.isValid(nb) { continue }
                let k = nb.row * cols + nb.col
                if visited[k] { continue }
                if grid.state(nb) != .free { continue }
                visited[k] = true
                parent[k] = Int32(curKey)
                distMap[k] = Int32(curDist + 1)
                queue.append(nb)
            }
        }
        guard bestKey >= 0 else { return nil }
        // Reconstruct path from start to goal.
        var path: [Cell] = []
        var k = bestKey
        while k != startKey {
            path.append(Cell(col: k % cols, row: k / cols))
            let p = Int(parent[k])
            if p < 0 { break }
            k = p
        }
        path.reverse()
        let goal = Cell(col: bestKey % cols, row: bestKey / cols)
        return (goal, bestDist, path)
    }

    /// Goal picker after a 360° scan. Two modes:
    ///   • Target bookmarked (YOLO confirmed person) → return free cell next
    ///     to target, so A* drives us close.
    ///   • Otherwise → deepest reachable corridor end (DFS, 7-cell clearance).
    @MainActor
    private func pickExploreGoal(from poseCell: Cell) -> Cell? {
        if let tgt = markedTarget {
            if let g = FrontierExplorer.nearestFreeNear(tgt,
                                                        radius: targetApproachAdjacentRadius,
                                                        grid: grid) {
                return g
            }
        }
        return CorridorFinder.deepestCorridorEnd(grid: grid, origin: poseCell)
    }

    private func keysFor(_ cmd: DrivingCommand) -> Set<UInt16> {
        switch cmd {
        case .forward:      return [KeyboardMonitor.W]
        case .forwardLeft:  return [KeyboardMonitor.W, KeyboardMonitor.A]
        case .forwardRight: return [KeyboardMonitor.W, KeyboardMonitor.D]
        case .reverse:      return [KeyboardMonitor.S]
        case .reverseLeft:  return [KeyboardMonitor.S, KeyboardMonitor.A]
        case .reverseRight: return [KeyboardMonitor.S, KeyboardMonitor.D]
        case .rotateLeft:   return [KeyboardMonitor.A]
        case .rotateRight:  return [KeyboardMonitor.D]
        case .brake:        return []
        }
    }
}
