//
//  DepthDriver.swift
//  drivingsim
//
//  Native Swift port of live_depth_mlmodel_wasd 7.py.
//  v7: navigation gamma 0.6 applied to depth before zone scoring
//  (brightens close-range, makes low scores more discriminating).
//  Takes CVPixelBuffer frames (518×518 from SimFPVRenderer),
//  runs DepthAnythingV2 SmallF16 CoreML model, computes 7×7 zone grid
//  (FAR row top 40% + NEAR row bottom 60% of ROI, 7 columns each),
//  decides a WASD command, smooths via majority-vote buffer.
//
//  Same ObservableObject surface as HandJoystick.
//

@preconcurrency import CoreML
@preconcurrency import CoreVideo
import Combine
import Foundation
import os
import SwiftUI
import Vision

// ── Public types ──────────────────────────────────────────────────────────────

enum DrivingCommand: String {
    case forward
    case forwardLeft
    case forwardRight
    case reverse
    case reverseLeft
    case reverseRight
    case brake

    var label: String {
        switch self {
        case .forward:      return "^ FORWARD"
        case .forwardLeft:  return "<^ FWD+LEFT"
        case .forwardRight: return "^> FWD+RIGHT"
        case .reverse:      return "v REVERSE"
        case .reverseLeft:  return "<v REV+LEFT"
        case .reverseRight: return "v> REV+RIGHT"
        case .brake:        return "STOP"
        }
    }
}

enum ZoneState { case clear, uncertain, blocked }

struct ZoneScore: Identifiable {
    let id = UUID()
    let name: String
    let meanDepth: Float
    let obstacleScore: Float
    let state: ZoneState
}

// ── DepthDriver ───────────────────────────────────────────────────────────────

@MainActor
final class DepthDriver: ObservableObject {
    // Input gating surface — same as HandJoystick / KeyboardMonitor
    @Published private(set) var keys: Set<UInt16> = []
    @Published private(set) var command: DrivingCommand = .brake
    @Published private(set) var farZones:  [ZoneScore] = []   // F1–F7 (top 40% of ROI)
    @Published private(set) var nearZones: [ZoneScore] = []   // N1–N7 (bottom 60% of ROI)
    @Published private(set) var depthImage: CGImage?    // colourised depth for preview

    var forward:  Bool { keys.contains(KeyboardMonitor.W) }
    var backward: Bool { keys.contains(KeyboardMonitor.S) }
    var left:     Bool { keys.contains(KeyboardMonitor.A) }
    var right:    Bool { keys.contains(KeyboardMonitor.D) }

    // Tunables — live_depth_mlmodel_wasd 5.py.
    private let forwardThreshold:    Float = 120
    private let stuckThreshold:      Float = 60
    private let steerMinDiff:        Float = 15
    private let hysteresisBonus:     Float = 10
    private let navRowStart:         Float = 0.0   // no sky to skip indoors
    private let cmdSmoothFrames:     Int   = 3
    private let reverseEscapeFrames: Int   = 8
    // Navigation gamma (v7) — applied to depth values used for zone scoring.
    // < 1 brightens / compresses, making close-range obstacles more discriminating.
    private let navigationGamma: Float = 0.6
    // Close-range reverse safety (v6)
    private let reverseBlockedCenterThreshold:    Float = 70
    private let reverseBlockedMinNearZones:       Int   = 5
    private let reverseBlockedConfirmFrames:      Int   = 3
    private let reverseImmediateCenterThreshold: Float = 40
    // Exploration / inward-bias
    private let explorationEnabled:     Bool  = true
    // Exit avoidance (anti-open-space lure)
    private let exitAvoidanceEnabled:   Bool  = true
    private let exitHardBlockEnabled:   Bool  = true
    // Corridor mode
    private let corridorModeEnabled:    Bool  = true
    // FAR variance (indoorness signal)
    private let farVarianceEnabled:     Bool  = true
    private let farVarianceThreshold:   Float = 15.0
    private let farOpenMeanThreshold:   Float = 150.0
    // Doorway bias
    private let doorwayBiasEnabled:     Bool  = true
    private let doorwayMaxWidth:        Int   = 4
    private let doorwayBonus:           Float = 20
    // Loop breaker
    private let loopBreakerEnabled:     Bool  = true
    private let loopReverseWindow:      Int   = 60
    private let loopReverseThreshold:   Int   = 3
    private let loopBreakerFramesMax:   Int   = 20

    // Perception state.
    private var consecutiveReverse = 0
    private var escapePhase = 0
    private var smoother: [DrivingCommand] = []
    private var lastCommand: DrivingCommand = .brake
    private var zoneLogCount = 0
    private var exploration = ExplorationState()
    // Loop breaker state
    private var reverseHistory: [Bool] = []
    private var loopBreakerFrames: Int = 0
    private var loopBreakerDir: ExplorationState.Side = .none
    // Close-range reverse confirmation counter (v6)
    private var blockedNearFrames: Int = 0

    // Model
    nonisolated(unsafe) private var model: MLModel?
    nonisolated(unsafe) private var modelInputName: String = "image"
    nonisolated(unsafe) private var modelOutputName: String = "depth"
    nonisolated(unsafe) private var modelInputW: Int = 518
    nonisolated(unsafe) private var modelInputH: Int = 518

    private let visionQueue = DispatchQueue(label: "depthdriver.vision", qos: .userInitiated)
    nonisolated private let inFlight = OSAllocatedUnfairLock<Bool>(initialState: false)
    private var running = false

    // Stats
    nonisolated(unsafe) private var statFrames = 0
    nonisolated(unsafe) private var statSumInfer = 0.0
    nonisolated(unsafe) private var statMaxInfer = 0.0
    nonisolated(unsafe) private var statWindowT = CACurrentMediaTime()

    init() {
        loadModel()
    }

    private func loadModel() {
        let cfg = MLModelConfiguration()
        cfg.computeUnits = .all   // Neural Engine when available
        guard let url = Bundle.main.url(forResource: "DepthAnythingV2SmallF16",
                                        withExtension: "mlmodelc")
                     ?? Bundle.main.url(forResource: "DepthAnythingV2SmallF16",
                                        withExtension: "mlpackage") else {
            print("[DepthDriver] model not found in bundle")
            return
        }
        do {
            let m = try MLModel(contentsOf: url, configuration: cfg)
            self.model = m
            // Discover input/output names + shape from the model description
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
            print("[DepthDriver] model loaded — input \(modelInputName) \(modelInputW)x\(modelInputH), output \(modelOutputName)")
        } catch {
            print("[DepthDriver] failed to load model: \(error)")
        }
    }

    var inputSize: (width: Int, height: Int) { (modelInputW, modelInputH) }

    func start() { running = true }
    func stop() {
        running = false
        keys = []
        command = .brake
        farZones = []
        nearZones = []
        consecutiveReverse = 0
        escapePhase = 0
        smoother.removeAll()
        lastCommand = .brake
        zoneLogCount = 0
        exploration = ExplorationState()
        reverseHistory.removeAll()
        loopBreakerFrames = 0
        loopBreakerDir = .none
        blockedNearFrames = 0
        DepthDriver.diagLogged = false
        inFlight.withLock { $0 = false }
    }

    /// Submit a new frame for inference. Drops frame if previous is still running.
    nonisolated func submit(_ pixelBuffer: CVPixelBuffer) {
        let busy = inFlight.withLock { state -> Bool in
            if state { return true }
            state = true; return false
        }
        if busy { return }

        visionQueue.async { [weak self] in
            self?.runInference(pixelBuffer)
        }
    }

    nonisolated private func runInference(_ pixelBuffer: CVPixelBuffer) {
        guard let model = self.model else {
            inFlight.withLock { $0 = false }
            return
        }

        let t0 = CACurrentMediaTime()

        // Build input — model expects an image, name discovered at load time.
        let provider: MLFeatureProvider
        do {
            let dict: [String: MLFeatureValue] = [
                modelInputName: MLFeatureValue(pixelBuffer: pixelBuffer)
            ]
            provider = try MLDictionaryFeatureProvider(dictionary: dict)
        } catch {
            print("[DepthDriver] feature build failed: \(error)")
            inFlight.withLock { $0 = false }
            return
        }

        let result: MLFeatureProvider
        do {
            result = try model.prediction(from: provider)
        } catch {
            print("[DepthDriver] prediction failed: \(error)")
            inFlight.withLock { $0 = false }
            return
        }

        let inferMs = (CACurrentMediaTime() - t0) * 1000.0
        statFrames += 1
        statSumInfer += inferMs
        if inferMs > statMaxInfer { statMaxInfer = inferMs }

        // Output: grayscale image (CVPixelBuffer) — extract depth_u8 + colourised preview.
        guard let outVal = result.featureValue(for: modelOutputName),
              let outBuf = outVal.imageBufferValue else {
            print("[DepthDriver] output buffer missing for \(modelOutputName)")
            inFlight.withLock { $0 = false }
            return
        }

        let (depthU8, w, h) = Self.depthBufferToU8(outBuf)

        // Build colourised CGImage for preview from raw normalized depth.
        let preview = Self.colorize(depthU8: depthU8, width: w, height: h)

        // v7: apply navigation gamma (0.6) to depth values used for zone scoring.
        let depthU8Nav = Self.applyGamma(depthU8, gamma: navigationGamma)

        logStatsIfNeeded()

        Task { @MainActor [self] in
            self.processDepth(depthU8: depthU8Nav, w: w, h: h, preview: preview)
            self.inFlight.withLock { $0 = false }
        }
    }

    nonisolated static func applyGamma(_ a: [UInt8], gamma: Float) -> [UInt8] {
        guard gamma > 0 else { return a }
        // Build a 256-entry LUT once per call: out = (in/255)^gamma * 255
        var lut = [UInt8](repeating: 0, count: 256)
        for i in 0..<256 {
            let v = powf(Float(i) / 255.0, gamma) * 255.0
            lut[i] = UInt8(max(0, min(255, v.rounded())))
        }
        var out = [UInt8](repeating: 0, count: a.count)
        for i in 0..<a.count { out[i] = lut[Int(a[i])] }
        return out
    }

    nonisolated private func logStatsIfNeeded() {
        guard statFrames >= 30 else { return }
        let elapsed = CACurrentMediaTime() - statWindowT
        let fps = Double(statFrames) / max(elapsed, 1e-9)
        let avg = statSumInfer / Double(max(1, statFrames))
        print(String(format: "[DepthDriver] %.0f fps | infer avg %.1f ms  max %.1f ms",
                     fps, avg, statMaxInfer))
        statFrames = 0; statSumInfer = 0; statMaxInfer = 0
        statWindowT = CACurrentMediaTime()
    }

    // ── MainActor: ROI/zones/decision/smoothing/publish ──────────────────────

    @MainActor
    private func processDepth(depthU8: [UInt8], w: Int, h: Int, preview: CGImage?) {
        guard !depthU8.isEmpty, w > 0, h > 0 else { return }

        let stdGuess = Self.fastStd(depthU8)
        if stdGuess < 1e-3 {
            push(.brake)
            depthImage = preview
            return
        }

        let row0 = Int(Float(h) * navRowStart)
        let safeRow0 = min(max(0, row0), h - 1)

        let (fz, nz) = computeZoneScores(depthU8: depthU8, w: w, h: h, rowStart: safeRow0)
        farZones  = fz
        nearZones = nz

        // ── Pre-decide must-reverse gate (v6) ────────────────────────────
        // Operates on car-space (mirrored) NEAR scores to be consistent with decision tree.
        let nearScoresCar = nz.map(\.obstacleScore).reversed() as [Float]
        let nc3Min = min(nearScoresCar[2], min(nearScoresCar[3], nearScoresCar[4]))
        let nc3Avg = (nearScoresCar[2] + nearScoresCar[3] + nearScoresCar[4]) / 3
        let blockedCount = nearScoresCar.filter { $0 < reverseBlockedCenterThreshold }.count

        let blockedScene = (nc3Min < reverseBlockedCenterThreshold)
                           && (blockedCount >= reverseBlockedMinNearZones)
        if blockedScene { blockedNearFrames += 1 }
        else            { blockedNearFrames = 0 }

        let mustReverseNow = (nc3Avg < reverseImmediateCenterThreshold)
                             || (blockedNearFrames >= reverseBlockedConfirmFrames)

        let raw: DrivingCommand
        if mustReverseNow {
            consecutiveReverse += 1
            if consecutiveReverse > reverseEscapeFrames {
                let lAvg = (nearScoresCar[0] + nearScoresCar[1] + nearScoresCar[2]) / 3
                let rAvg = (nearScoresCar[4] + nearScoresCar[5] + nearScoresCar[6]) / 3
                raw = (lAvg >= rAvg) ? .reverseLeft : .reverseRight
            } else {
                raw = .reverse
            }
            escapePhase = 0
        } else {
            raw = decideCommand(farZones: fz, nearZones: nz)
        }
        var smoothed = push(raw)

        // Loop breaker: track reverses; force a sustained turn when looping.
        let isRev = (smoothed == .reverse || smoothed == .reverseLeft || smoothed == .reverseRight)
        reverseHistory.append(isRev)
        if reverseHistory.count > loopReverseWindow { reverseHistory.removeFirst() }

        // v6: must-reverse resets loop breaker so safety always wins.
        if mustReverseNow {
            loopBreakerFrames = 0
            reverseHistory.removeAll()
        }

        if loopBreakerEnabled && loopBreakerFrames == 0 {
            let revCount = reverseHistory.filter { $0 }.count
            if revCount >= loopReverseThreshold {
                let lefts  = exploration.turnHistory.filter { $0 == "left"  }.count
                let rights = exploration.turnHistory.filter { $0 == "right" }.count
                loopBreakerDir = (rights >= lefts) ? .left : .right
                loopBreakerFrames = loopBreakerFramesMax
                reverseHistory.removeAll()
            }
        }
        if loopBreakerEnabled && loopBreakerFrames > 0 && !mustReverseNow {
            loopBreakerFrames -= 1
            smoothed = (loopBreakerDir == .left) ? .forwardLeft : .forwardRight
        }

        lastCommand = smoothed
        exploration.recordCommand(smoothed)
        command = smoothed
        depthImage = preview

        zoneLogCount += 1
        if zoneLogCount >= 30 {
            zoneLogCount = 0
            let fStr = fz.map { String(format: "%3.0f", $0.obstacleScore) }.joined(separator: " ")
            let nStr = nz.map { String(format: "%3.0f", $0.obstacleScore) }.joined(separator: " ")
            print("[DepthDriver] FAR  [F1–F7] = \(fStr)  raw=\(raw.rawValue)  smooth=\(smoothed.rawValue)")
            print("[DepthDriver] NEAR [N1–N7] = \(nStr)")
        }

        switch smoothed {
        case .forward:      keys = [KeyboardMonitor.W]
        case .forwardLeft:  keys = [KeyboardMonitor.W, KeyboardMonitor.A]
        case .forwardRight: keys = [KeyboardMonitor.W, KeyboardMonitor.D]
        case .reverse:      keys = [KeyboardMonitor.S]
        case .reverseLeft:  keys = [KeyboardMonitor.S, KeyboardMonitor.A]
        case .reverseRight: keys = [KeyboardMonitor.S, KeyboardMonitor.D]
        case .brake:        keys = []
        }
    }

    // Returns (farZones[7], nearZones[7]).
    // FAR  = top 40% of ROI — path ahead.
    // NEAR = bottom 60% of ROI — immediate obstacles.
    @MainActor
    private func computeZoneScores(depthU8: [UInt8], w: Int, h: Int, rowStart: Int)
        -> ([ZoneScore], [ZoneScore])
    {
        let roiH = h - rowStart
        let farRowEnd = rowStart + Int(Float(roiH) * 0.4)
        let fz = computeRow(depthU8: depthU8, w: w, rowStart: rowStart, rowEnd: farRowEnd, prefix: "F")
        let nz = computeRow(depthU8: depthU8, w: w, rowStart: farRowEnd,  rowEnd: h,        prefix: "N")
        return (fz, nz)
    }

    @MainActor
    private func computeRow(depthU8: [UInt8], w: Int, rowStart: Int, rowEnd: Int, prefix: String) -> [ZoneScore] {
        let count = 7
        var out: [ZoneScore] = []
        let edges = (0...count).map { Int(Float(w) * Float($0) / Float(count)) }
        for i in 0..<count {
            let x0 = edges[i], x1 = edges[i + 1]
            var sum: UInt64 = 0
            var n:   UInt64 = 0
            for y in rowStart..<rowEnd {
                let row = y * w
                for x in x0..<x1 {
                    sum &+= UInt64(depthU8[row + x])
                    n   &+= 1
                }
            }
            let mean  = n > 0 ? Float(sum) / Float(n) : 0
            let score = max(0, min(255, mean))
            let state: ZoneState
            if      score > forwardThreshold { state = .clear }
            else if score < stuckThreshold   { state = .blocked }
            else                             { state = .uncertain }
            out.append(ZoneScore(name: "\(prefix)\(i + 1)", meanDepth: mean, obstacleScore: score, state: state))
        }
        return out
    }

    // Gap helper
    private struct Gap { let center: Float; let width: Int; let avg: Float }

    @MainActor
    private func findBestGap(_ scores: [Float]) -> Gap? {
        var gaps: [Gap] = []
        var gapStart: Int? = nil
        var gapSum: Float  = 0
        for (i, score) in scores.enumerated() {
            if score >= stuckThreshold {
                if gapStart == nil { gapStart = i; gapSum = score }
                else               { gapSum += score }
            } else if let start = gapStart {
                let width = i - start
                gaps.append(Gap(center: Float(start) + Float(width) / 2.0,
                                width:  width,
                                avg:    gapSum / Float(width)))
                gapStart = nil; gapSum = 0
            }
        }
        if let start = gapStart {
            let width = scores.count - start
            gaps.append(Gap(center: Float(start) + Float(width) / 2.0,
                            width:  width,
                            avg:    gapSum / Float(width)))
        }
        if gaps.isEmpty { return nil }

        // Doorway bias — narrow gaps (1..doorwayMaxWidth) get a bonus so an
        // indoor doorway competes with a wider exit-like opening.
        if doorwayBiasEnabled {
            return gaps.max { a, b in
                let aBonus: Float = (a.width >= 1 && a.width <= doorwayMaxWidth) ? doorwayBonus : 0
                let bBonus: Float = (b.width >= 1 && b.width <= doorwayMaxWidth) ? doorwayBonus : 0
                return (a.avg + aBonus) < (b.avg + bBonus)
            }
        }
        return gaps.max { a, b in a.width < b.width || (a.width == b.width && a.avg < b.avg) }
    }

    // Sample standard deviation
    private func stdev(_ a: [Float]) -> Float {
        guard !a.isEmpty else { return 0 }
        let mean = a.reduce(0, +) / Float(a.count)
        let v = a.reduce(0) { $0 + ($1 - mean) * ($1 - mean) } / Float(a.count)
        return v.squareRoot()
    }

    @MainActor
    private func decideCommand(farZones: [ZoneScore], nearZones: [ZoneScore]) -> DrivingCommand {
        // Mirror horizontally — depth image left/right is opposite to car steering left/right.
        let farScores  = farZones.map(\.obstacleScore).reversed() as [Float]
        let nearScores = nearZones.map(\.obstacleScore).reversed() as [Float]

        // Update outside-side memory before reading bonuses (mirror=false: scores already in car space).
        if explorationEnabled {
            exploration.observeScene(farScores: farScores, nearScores: nearScores,
                                     enabled: exitAvoidanceEnabled)
        }

        let expL = explorationEnabled ? exploration.leftBonus(exitAvoidEnabled: exitAvoidanceEnabled)  : 0 as Float
        let expR = explorationEnabled ? exploration.rightBonus(exitAvoidEnabled: exitAvoidanceEnabled) : 0 as Float

        // Center 3 zones: indices 2, 3, 4
        let nearCenter3 = [nearScores[2], nearScores[3], nearScores[4]]
        let farCenter3  = [farScores[2],  farScores[3],  farScores[4]]
        let nearCenterMin = nearCenter3.min() ?? 0
        let nearCenterAvg = nearCenter3.reduce(0, +) / 3.0
        let farCenterMin  = farCenter3.min()  ?? 0
        let bestNear      = nearScores.max()  ?? 0

        // 1. Escape manoeuvre
        if escapePhase != 0 {
            if escapePhase > 0 {
                escapePhase -= 1
                if escapePhase == 0 { escapePhase = -5 }
                return .reverseLeft
            } else {
                escapePhase += 1
                if escapePhase == 0 { return .forward }
                return .reverseRight
            }
        }

        // 2. Truly stuck → REVERSE (or escape)
        if bestNear < stuckThreshold {
            consecutiveReverse += 1
            if consecutiveReverse > reverseEscapeFrames {
                let lAvg = (nearScores[0] + nearScores[1] + nearScores[2]) / 3
                let rAvg = (nearScores[4] + nearScores[5] + nearScores[6]) / 3
                consecutiveReverse = 0
                if lAvg >= rAvg { escapePhase =  5; return .reverseLeft  }
                else            { escapePhase = -5; return .reverseRight }
            }
            return .reverse
        }
        consecutiveReverse = 0

        // 2.5. Hard exit redirect — turn away from confirmed outside side
        // when opposite side has any viable path.
        if exitAvoidanceEnabled && exitHardBlockEnabled && exploration.outsideTimer > 0 {
            let nL = (nearScores[0] + nearScores[1] + nearScores[2]) / 3
            let nR = (nearScores[4] + nearScores[5] + nearScores[6]) / 3
            switch exploration.outsideSide {
            case .left  where nR > forwardThreshold: return .forwardRight
            case .right where nL > forwardThreshold: return .forwardLeft
            default: break  // both sides blocked — fall through
            }
        }

        // 3. Center blocked — MUST steer, never go forward
        let centerIsBlocked = nearCenter3.contains { $0 < forwardThreshold }
        if centerIsBlocked {
            let lScores = [nearScores[0], nearScores[1], nearScores[2]]
            let rScores = [nearScores[4], nearScores[5], nearScores[6]]
            var lAvg = lScores.reduce(0, +) / 3.0
            var rAvg = rScores.reduce(0, +) / 3.0
            if lastCommand == .forwardLeft  { lAvg += hysteresisBonus }
            if lastCommand == .forwardRight { rAvg += hysteresisBonus }
            lAvg += expL; rAvg += expR
            let lMax = lScores.max() ?? 0
            let rMax = rScores.max() ?? 0
            if lAvg > rAvg + steerMinDiff { return .forwardLeft  }
            if rAvg > lAvg + steerMinDiff { return .forwardRight }
            return lMax >= rMax ? .forwardLeft : .forwardRight
        }

        // 4. Near center clear — check FAR
        let nearAllClear = nearCenterMin >= forwardThreshold
        let farAllClear  = farCenterMin  >= forwardThreshold
        if nearAllClear && farAllClear  {
            // FAR variance: uniformly open FAR row may indicate an exit/outdoor area.
            // Prefer a lateral turn to stay indoors.
            if farVarianceEnabled {
                let fStd = stdev(farScores)
                let fMean = farScores.reduce(0, +) / Float(farScores.count)
                if fStd < farVarianceThreshold && fMean > farOpenMeanThreshold {
                    let nL = (nearScores[0] + nearScores[1] + nearScores[2]) / 3 + expL
                    let nR = (nearScores[4] + nearScores[5] + nearScores[6]) / 3 + expR
                    if nL > nR + steerMinDiff { return .forwardLeft  }
                    if nR > nL + steerMinDiff { return .forwardRight }
                }
            }
            return .forward
        }

        // Near clear but far blocked — preemptive steer
        if nearAllClear && !farAllClear {
            // Corridor mode: if both near sides are walled, commit FORWARD rather
            // than steering away from a distant FAR obstacle (handle when it gets close).
            if corridorModeEnabled {
                let nL = (nearScores[0] + nearScores[1] + nearScores[2]) / 3
                let nR = (nearScores[4] + nearScores[5] + nearScores[6]) / 3
                if nL < forwardThreshold && nR < forwardThreshold { return .forward }
            }
            let flAvg = (farScores[0] + farScores[1] + farScores[2]) / 3 + expL
            let frAvg = (farScores[4] + farScores[5] + farScores[6]) / 3 + expR
            if flAvg > frAvg + steerMinDiff { return .forwardLeft  }
            if frAvg > flAvg + steerMinDiff { return .forwardRight }
            return .forward
        }

        // 5. Find best gap in near row and steer toward it
        guard let gap = findBestGap(nearScores) else { return .reverse }
        var gapCenter = gap.center
        if lastCommand == .forwardLeft  && gapCenter < 3.5 { gapCenter -= 0.3 }
        if lastCommand == .forwardRight && gapCenter > 3.5 { gapCenter += 0.3 }

        if gapCenter < 2.5 { return .forwardLeft  }
        if gapCenter > 4.5 { return .forwardRight }
        if gapCenter < 3.0 { return .forwardLeft  }
        if gapCenter > 4.0 { return .forwardRight }

        // Gap centered — check if clear enough to go straight
        if nearCenterAvg >= forwardThreshold * 0.8 { return .forward }
        let lAvg = (nearScores[0] + nearScores[1] + nearScores[2]) / 3 + expL
        let rAvg = (nearScores[4] + nearScores[5] + nearScores[6]) / 3 + expR
        if lAvg > rAvg + 10 { return .forwardLeft  }
        if rAvg > lAvg + 10 { return .forwardRight }
        return .forward
    }

    @MainActor
    @discardableResult
    private func push(_ cmd: DrivingCommand) -> DrivingCommand {
        // Reverse commands take effect immediately when boxed in (v6).
        if cmd == .reverse || cmd == .reverseLeft || cmd == .reverseRight {
            smoother.removeAll()
            smoother.append(cmd)
            return cmd
        }

        smoother.append(cmd)
        if smoother.count > cmdSmoothFrames { smoother.removeFirst() }
        if smoother.count < cmdSmoothFrames { return cmd }   // startup

        // BRAKE priority gate
        let brakeCount = smoother.filter { $0 == .brake }.count
        if brakeCount > smoother.count / 3 { return .brake }

        // Majority vote with rightmost-recent tie-break
        var counts: [DrivingCommand: Int] = [:]
        for c in smoother { counts[c, default: 0] += 1 }
        let maxCount = counts.values.max() ?? 0
        let tied = Set(counts.filter { $0.value == maxCount }.keys)
        for c in smoother.reversed() where tied.contains(c) { return c }
        return cmd
    }

    // ── Static helpers (depth post-processing) ───────────────────────────────

    /// Single-pass per-frame min/max normalization + invert.
    /// Matches live_depth_mlmodel_wasd.py:103-121 (no percentile, no EMA).
    nonisolated static func depthBufferToU8(_ buf: CVPixelBuffer) -> ([UInt8], Int, Int) {
        CVPixelBufferLockBaseAddress(buf, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buf, .readOnly) }
        let w = CVPixelBufferGetWidth(buf)
        let h = CVPixelBufferGetHeight(buf)
        let stride = CVPixelBufferGetBytesPerRow(buf)
        let format = CVPixelBufferGetPixelFormatType(buf)
        guard let base = CVPixelBufferGetBaseAddress(buf) else { return ([], 0, 0) }

        var floats = [Float](repeating: 0, count: w * h)
        switch format {
        case kCVPixelFormatType_OneComponent16Half:
            for y in 0..<h {
                let row = base.advanced(by: y * stride).assumingMemoryBound(to: UInt16.self)
                for x in 0..<w {
                    floats[y * w + x] = Float(Float16(bitPattern: row[x]))
                }
            }
        case kCVPixelFormatType_OneComponent32Float:
            for y in 0..<h {
                let row = base.advanced(by: y * stride).assumingMemoryBound(to: Float.self)
                for x in 0..<w {
                    floats[y * w + x] = row[x]
                }
            }
        case kCVPixelFormatType_OneComponent8:
            for y in 0..<h {
                let row = base.advanced(by: y * stride).assumingMemoryBound(to: UInt8.self)
                for x in 0..<w {
                    floats[y * w + x] = Float(row[x]) / 255.0
                }
            }
        default:
            for y in 0..<h {
                let row = base.advanced(by: y * stride).assumingMemoryBound(to: UInt16.self)
                for x in 0..<w {
                    floats[y * w + x] = Float(Float16(bitPattern: row[x]))
                }
            }
        }

        var dMin = Float.infinity
        var dMax = -Float.infinity
        for v in floats {
            if v < dMin { dMin = v }
            if v > dMax { dMax = v }
        }

        // One-time diagnostic: log raw values + format so we can confirm polarity.
        // top-mid pixel ≈ ceiling/wall (typically far), bottom-mid ≈ floor (close).
        // If model outputs DEPTH (large=far): top RAW < bottom RAW expected.
        // If model outputs DISPARITY (large=close): top RAW > bottom RAW.
        if !diagLogged {
            diagLogged = true
            let topMidRaw    = floats[5 * w + w / 2]
            let centerRaw    = floats[(h / 2) * w + w / 2]
            let bottomMidRaw = floats[(h - 5) * w + w / 2]
            let fmtName: String
            switch format {
            case kCVPixelFormatType_OneComponent16Half: fmtName = "F16"
            case kCVPixelFormatType_OneComponent32Float: fmtName = "F32"
            case kCVPixelFormatType_OneComponent8:       fmtName = "U8"
            default: fmtName = String(format: "0x%X", format)
            }
            print(String(format: "[DepthDriver] DIAG fmt=%@ %dx%d  raw min=%.4f max=%.4f  top=%.4f center=%.4f bottom=%.4f",
                         fmtName, w, h, dMin, dMax, topMidRaw, centerRaw, bottomMidRaw))
            print("[DepthDriver] DIAG  if bottom>top → DISPARITY (need invert);  if bottom<top → DEPTH (current code correct)")
        }

        let dRange = dMax - dMin
        var u8 = [UInt8](repeating: 0, count: w * h)
        if dRange > 1e-6 {
            for i in 0..<floats.count {
                var n = (floats[i] - dMin) / dRange
                if n < 0 { n = 0 } else if n > 1 { n = 1 }
                // No invert. Apple's "Estimated depth map" output: large value = far.
                // After normalization: high u8 = far (clear), low u8 = near (obstacle).
                // Decision tree thresholds operate on this directly.
                u8[i] = UInt8(n * 255.0)
            }
        }
        return (u8, w, h)
    }

    nonisolated(unsafe) private static var diagLogged = false

    nonisolated static func colorize(depthU8: [UInt8], width: Int, height: Int) -> CGImage? {
        guard !depthU8.isEmpty else { return nil }
        var rgba = [UInt8](repeating: 0, count: width * height * 4)
        for i in 0..<depthU8.count {
            let v = Float(depthU8[i]) / 255.0
            // Inferno-ish: dark→purple→red→orange→yellow
            let r = UInt8(min(255, max(0, v * 255.0 * 1.0)))
            let g = UInt8(min(255, max(0, (v - 0.3) * 255.0 * 1.4)))
            let b = UInt8(min(255, max(0, (0.4 - v) * 255.0 * 1.5)))
            rgba[i * 4 + 0] = r
            rgba[i * 4 + 1] = g
            rgba[i * 4 + 2] = b
            rgba[i * 4 + 3] = 255
        }
        let cs = CGColorSpaceCreateDeviceRGB()
        guard let provider = CGDataProvider(data: Data(rgba) as CFData) else { return nil }
        return CGImage(width: width, height: height,
                       bitsPerComponent: 8, bitsPerPixel: 32,
                       bytesPerRow: width * 4,
                       space: cs,
                       bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
                       provider: provider, decode: nil, shouldInterpolate: false,
                       intent: .defaultIntent)
    }

    nonisolated static func fastStd(_ a: [UInt8]) -> Float {
        guard !a.isEmpty else { return 0 }
        var sum: Double = 0
        var sumSq: Double = 0
        for v in a {
            let d = Double(v)
            sum += d
            sumSq += d * d
        }
        let n = Double(a.count)
        let mean = sum / n
        let variance = max(0, sumSq / n - mean * mean)
        return Float(variance.squareRoot())
    }

}

// ── ExplorationState ──────────────────────────────────────────────────────────
// Port of ExplorationState in live_depth_mlmodel_wasd 4.py.
// After a reverse the car biases away from where it came from (inward push).
// Counts left/right turns over a rolling window and balances them so the car
// explores both sides instead of wall-hugging.

struct ExplorationState {
    enum Side { case left, right, none }

    // Inward-bias tunables
    let inwardBias:           Float = 25
    let turnBalanceBonus:     Float = 12
    let turnBalanceThreshold: Int   = 6
    let reverseHeadingMemory: Int   = 30
    let windowSize:           Int   = 40
    // Exit-avoidance tunables
    let exitConfirmFrames:      Int   = 8
    let exitMemoryFrames:       Int   = 45
    let exitSideDiff:           Float = 22
    let exitFarOpenThreshold:   Float = 165
    let exitNearOpenThreshold:  Float = 135
    let exitAvoidancePenalty:   Float = 30

    var cameFromSide:  Side = .none
    var cameFromTimer: Int  = 0
    private(set) var turnHistory: [String] = []   // "left" | "right" | "fwd"
    // Exit-avoidance state
    var outsideSide:  Side = .none
    var outsideTimer: Int  = 0
    private var outsideConfirmLeft:  Int = 0
    private var outsideConfirmRight: Int = 0

    mutating func recordCommand(_ cmd: DrivingCommand) {
        switch cmd {
        case .reverseLeft:
            cameFromSide = .left;  cameFromTimer = reverseHeadingMemory
        case .reverseRight:
            cameFromSide = .right; cameFromTimer = reverseHeadingMemory
        case .reverse:
            cameFromTimer = reverseHeadingMemory
        case .forward, .forwardLeft, .forwardRight:
            if cameFromTimer > 0 { cameFromTimer -= 1 }
            turnHistory.append(cmd == .forwardLeft ? "left" : cmd == .forwardRight ? "right" : "fwd")
            if turnHistory.count > windowSize { turnHistory.removeFirst() }
        default:
            break
        }
    }

    /// Update outside-side memory from current FAR/NEAR side openness.
    /// Scores are expected in car-space orientation (already mirrored if needed).
    mutating func observeScene(farScores: [Float], nearScores: [Float], enabled: Bool) {
        guard enabled else { return }
        let fL = (farScores[0]  + farScores[1]  + farScores[2])  / 3
        let fR = (farScores[4]  + farScores[5]  + farScores[6])  / 3
        let nL = (nearScores[0] + nearScores[1] + nearScores[2]) / 3
        let nR = (nearScores[4] + nearScores[5] + nearScores[6]) / 3
        let lOpen = (fL + nL) / 2
        let rOpen = (fR + nR) / 2

        let leftOutside  = (lOpen > rOpen + exitSideDiff)
                            && fL >= exitFarOpenThreshold
                            && nL >= exitNearOpenThreshold
        let rightOutside = (rOpen > lOpen + exitSideDiff)
                            && fR >= exitFarOpenThreshold
                            && nR >= exitNearOpenThreshold

        if leftOutside {
            outsideConfirmLeft  = min(outsideConfirmLeft + 1, exitConfirmFrames)
            outsideConfirmRight = max(outsideConfirmRight - 1, 0)
        } else if rightOutside {
            outsideConfirmRight = min(outsideConfirmRight + 1, exitConfirmFrames)
            outsideConfirmLeft  = max(outsideConfirmLeft - 1, 0)
        } else {
            outsideConfirmLeft  = max(outsideConfirmLeft - 1, 0)
            outsideConfirmRight = max(outsideConfirmRight - 1, 0)
        }

        if outsideConfirmLeft >= exitConfirmFrames {
            outsideSide = .left;  outsideTimer = exitMemoryFrames; outsideConfirmLeft = 0
        } else if outsideConfirmRight >= exitConfirmFrames {
            outsideSide = .right; outsideTimer = exitMemoryFrames; outsideConfirmRight = 0
        } else if outsideTimer > 0 {
            outsideTimer -= 1
            if outsideTimer == 0 { outsideSide = .none }
        }
    }

    func leftBonus(exitAvoidEnabled: Bool = true) -> Float {
        var b: Float = 0
        if cameFromTimer > 0 && cameFromSide == .right { b += inwardBias }
        let l = turnHistory.filter { $0 == "left"  }.count
        let r = turnHistory.filter { $0 == "right" }.count
        if r - l >= turnBalanceThreshold { b += turnBalanceBonus }
        if exitAvoidEnabled && outsideTimer > 0 && outsideSide == .left { b -= exitAvoidancePenalty }
        return b
    }

    func rightBonus(exitAvoidEnabled: Bool = true) -> Float {
        var b: Float = 0
        if cameFromTimer > 0 && cameFromSide == .left { b += inwardBias }
        let l = turnHistory.filter { $0 == "left"  }.count
        let r = turnHistory.filter { $0 == "right" }.count
        if l - r >= turnBalanceThreshold { b += turnBalanceBonus }
        if exitAvoidEnabled && outsideTimer > 0 && outsideSide == .right { b -= exitAvoidancePenalty }
        return b
    }
}
