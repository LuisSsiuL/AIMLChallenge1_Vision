//
//  DepthDriver.swift
//  drivingsim
//
//  Native Swift port of live_depth_wasd.py:354-787.
//  Takes CVPixelBuffer frames (518×518 BGRA from SimFPVRenderer),
//  runs DepthAnythingV2 SmallF16 CoreML model, computes 5-zone obstacle
//  scores, decides a WASD command, smooths via majority-vote buffer.
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
    case forward, turnLeft, reverse, turnRight, brake

    var label: String {
        switch self {
        case .forward:   return "^ FORWARD"
        case .turnLeft:  return "< LEFT"
        case .reverse:   return "v REVERSE"
        case .turnRight: return "> RIGHT"
        case .brake:     return "STOP"
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
    @Published private(set) var zones: [ZoneScore] = []
    @Published private(set) var depthImage: CGImage?    // colourised depth for preview
    @Published private(set) var floorBaseline: Float = 20

    var forward:  Bool { keys.contains(KeyboardMonitor.W) }
    var backward: Bool { keys.contains(KeyboardMonitor.S) }
    var left:     Bool { keys.contains(KeyboardMonitor.A) }
    var right:    Bool { keys.contains(KeyboardMonitor.D) }

    // Tunables — same names/values as live_depth_wasd.py
    private let forwardThreshold:    Float = 150
    private let reverseThreshold:    Float = 80
    private let steerMinDiff:        Float = 10
    private let floorEmaAlpha:       Float = 0.15
    private let navRowStart:         Float = 0.30
    private let floorBandFraction:   Float = 0.10
    private let cmdSmoothFrames:     Int   = 5
    private let reverseEscapeFrames: Int   = 10
    private let depthEmaAlpha:       Float = 0.10
    private let depthPercentile:     Float = 2

    // Perception state
    nonisolated(unsafe) private var emaDmin: Float?
    nonisolated(unsafe) private var emaDmax: Float?
    private var floorBaselineState: Float?
    private var consecutiveReverse = 0
    private var escapePhase = 0
    private var smoother: [DrivingCommand] = []

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
        zones = []
        consecutiveReverse = 0
        escapePhase = 0
        smoother.removeAll()
        emaDmin = nil
        emaDmax = nil
        floorBaselineState = nil
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

        let (depthU8, w, h) = Self.depthBufferToU8(outBuf,
                                                    percentile: 2.0,
                                                    emaAlpha: 0.10,
                                                    emaDmin: &emaDmin,
                                                    emaDmax: &emaDmax)

        // Build colourised CGImage for preview (inferno-ish gradient).
        let preview = Self.colorize(depthU8: depthU8, width: w, height: h)

        logStatsIfNeeded()

        Task { @MainActor [self] in
            self.processDepth(depthU8: depthU8, w: w, h: h, preview: preview)
            self.inFlight.withLock { $0 = false }
        }
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

        // Degenerate guard
        let stdGuess = Self.fastStd(depthU8)
        if stdGuess < 1e-3 {
            push(.brake)
            depthImage = preview
            return
        }

        let row0 = Int(Float(h) * navRowStart)
        let safeRow0 = min(max(0, row0), h - 1)

        // Floor baseline EMA from bottom band
        let bandH = max(1, Int(Float(h - safeRow0) * floorBandFraction))
        let bandStart = h - bandH
        let bandMedian = Self.medianRows(depthU8, w: w, rowStart: bandStart, rowEnd: h)
        let newBaseline: Float
        if let cur = floorBaselineState {
            newBaseline = floorEmaAlpha * bandMedian + (1.0 - floorEmaAlpha) * cur
        } else {
            newBaseline = bandMedian
        }
        floorBaselineState = min(max(newBaseline, 20), 200)
        floorBaseline = floorBaselineState!

        // Zone scores
        let z = computeZoneScores(depthU8: depthU8, w: w, h: h, rowStart: safeRow0)
        zones = z

        // Decide command (decision tree with escape phase)
        let raw = decideCommand(zones: z)
        let smoothed = push(raw)
        command = smoothed
        depthImage = preview

        // Map command → keys
        switch smoothed {
        case .forward:   keys = [KeyboardMonitor.W]
        case .turnLeft:  keys = [KeyboardMonitor.A]
        case .reverse:   keys = [KeyboardMonitor.S]
        case .turnRight: keys = [KeyboardMonitor.D]
        case .brake:     keys = []
        }
    }

    @MainActor
    private func computeZoneScores(depthU8: [UInt8], w: Int, h: Int, rowStart: Int) -> [ZoneScore] {
        let names = ["FarLeft", "Left", "Center", "Right", "FarRight"]
        var out: [ZoneScore] = []
        let edges = (0...5).map { Int(Float(w) * Float($0) / 5.0) }
        for i in 0..<5 {
            let x0 = edges[i], x1 = edges[i + 1]
            var sum: UInt64 = 0
            var n: UInt64 = 0
            for y in rowStart..<h {
                let row = y * w
                for x in x0..<x1 {
                    sum &+= UInt64(depthU8[row + x])
                    n &+= 1
                }
            }
            let mean = n > 0 ? Float(sum) / Float(n) : 0
            let score = max(0, min(255, mean))
            let state: ZoneState
            if      score > forwardThreshold { state = .clear }
            else if score < reverseThreshold { state = .blocked }
            else                              { state = .uncertain }
            out.append(ZoneScore(name: names[i], meanDepth: mean, obstacleScore: score, state: state))
        }
        return out
    }

    @MainActor
    private func decideCommand(zones: [ZoneScore]) -> DrivingCommand {
        let fl = zones[0], l = zones[1], c = zones[2], r = zones[3], fr = zones[4]
        let leftScore  = (fl.obstacleScore + l.obstacleScore) / 2.0
        let rightScore = (fr.obstacleScore + r.obstacleScore) / 2.0
        let bestScore  = [fl, l, c, r, fr].map(\.obstacleScore).max() ?? 0

        // 1. Escape manoeuvre takes priority
        if escapePhase != 0 {
            if escapePhase > 0 {
                escapePhase -= 1
                if escapePhase == 0 { escapePhase = -5 }
                return .turnLeft
            } else {
                escapePhase += 1
                if escapePhase == 0 { return .forward }
                return .turnRight
            }
        }

        // 2. Everything too close → REVERSE, then escape
        if bestScore < reverseThreshold {
            consecutiveReverse += 1
            if consecutiveReverse > reverseEscapeFrames {
                consecutiveReverse = 0
                if leftScore >= rightScore {
                    escapePhase = 5
                    return .turnLeft
                } else {
                    escapePhase = -5
                    return .turnRight
                }
            }
            return .reverse
        }
        consecutiveReverse = 0

        // 3. Center open → FORWARD
        if c.obstacleScore >= forwardThreshold { return .forward }

        // 4. Steer toward higher-scoring side
        if leftScore  > rightScore + steerMinDiff { return .turnLeft }
        if rightScore > leftScore  + steerMinDiff { return .turnRight }

        // Tie-break with far zones
        if fl.obstacleScore > fr.obstacleScore { return .turnLeft }
        if fr.obstacleScore > fl.obstacleScore { return .turnRight }

        return .forward
    }

    @MainActor
    @discardableResult
    private func push(_ cmd: DrivingCommand) -> DrivingCommand {
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

    nonisolated static func depthBufferToU8(
        _ buf: CVPixelBuffer,
        percentile: Float,
        emaAlpha: Float,
        emaDmin: inout Float?,
        emaDmax: inout Float?
    ) -> ([UInt8], Int, Int) {
        CVPixelBufferLockBaseAddress(buf, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buf, .readOnly) }
        let w = CVPixelBufferGetWidth(buf)
        let h = CVPixelBufferGetHeight(buf)
        let stride = CVPixelBufferGetBytesPerRow(buf)
        let format = CVPixelBufferGetPixelFormatType(buf)
        guard let base = CVPixelBufferGetBaseAddress(buf) else { return ([], 0, 0) }

        // Read into [Float] then percentile-clip → EMA bounds → invert → u8
        var floats = [Float](repeating: 0, count: w * h)

        switch format {
        case kCVPixelFormatType_OneComponent16Half:
            // F16 grayscale — common for Apple's CoreML depth output
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
            // Fallback: treat as F16 anyway
            for y in 0..<h {
                let row = base.advanced(by: y * stride).assumingMemoryBound(to: UInt16.self)
                for x in 0..<w {
                    floats[y * w + x] = Float(Float16(bitPattern: row[x]))
                }
            }
        }

        // Percentile clip
        let sorted = floats.sorted()
        let lowIdx  = max(0, min(sorted.count - 1, Int(Float(sorted.count) * percentile / 100.0)))
        let highIdx = max(0, min(sorted.count - 1, Int(Float(sorted.count) * (100.0 - percentile) / 100.0)))
        let pLo = sorted[lowIdx]
        let pHi = sorted[highIdx]

        // EMA bounds
        if let dmin = emaDmin, let dmax = emaDmax {
            emaDmin = emaAlpha * pLo + (1 - emaAlpha) * dmin
            emaDmax = emaAlpha * pHi + (1 - emaAlpha) * dmax
        } else {
            emaDmin = pLo
            emaDmax = pHi
        }

        let dRange = (emaDmax ?? pHi) - (emaDmin ?? pLo)
        var u8 = [UInt8](repeating: 0, count: w * h)
        if dRange > 1e-6 {
            let dmin = emaDmin ?? pLo
            for i in 0..<floats.count {
                var n = (floats[i] - dmin) / dRange
                if n < 0 { n = 0 } else if n > 1 { n = 1 }
                u8[i] = 255 - UInt8(n * 255.0)   // invert: bright=near, dark=far
            }
        }
        return (u8, w, h)
    }

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

    nonisolated static func medianRows(_ a: [UInt8], w: Int, rowStart: Int, rowEnd: Int) -> Float {
        let count = max(0, (rowEnd - rowStart) * w)
        guard count > 0 else { return 0 }
        var slice = [UInt8](); slice.reserveCapacity(count)
        for y in rowStart..<rowEnd {
            let row = y * w
            for x in 0..<w { slice.append(a[row + x]) }
        }
        slice.sort()
        return Float(slice[slice.count / 2])
    }
}
