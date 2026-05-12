//
//  HandJoystick.swift
//  drivingsim
//
//  Native Swift port of controls/gesture_index_joystick.py with adaptive
//  smoothing (One Euro Filter, Casiez 2012) and velocity-based prediction
//  through brief Vision dropouts.
//
//  Tunables loaded from controls/joystick_config.json (bundled or repo path).
//

@preconcurrency import AVFoundation
import Combine
import CoreImage
import Foundation
import os
import SwiftUI
import Vision

// ── One Euro Filter ───────────────────────────────────────────────────────────
//
// Adaptive low-pass filter — cutoff rises with velocity so fast movement passes
// through with low lag while slow/stationary motion gets aggressively smoothed.
// Reference: https://gery.casiez.net/1euro/  (Casiez et al., SIGCHI 2012)
//
// Per-axis (scalar). For 21 landmarks × {x,y} we instantiate 42 of these.

private struct OneEuroFilterScalar {
    var minCutoff: Float = 1.0   // Hz — jitter rejection at rest
    var beta:      Float = 0.0   // velocity-cutoff coefficient
    var dCutoff:   Float = 1.0   // Hz — derivative low-pass cutoff

    private var initialized = false
    private var xPrev:  Float = 0       // last filtered value
    private var dxPrev: Float = 0       // last filtered derivative (px/s)

    private static func alpha(cutoff: Float, dt: Float) -> Float {
        // tau = 1/(2π·cutoff); α = 1/(1 + τ/dt)  — canonical Casiez derivation
        let tau = 1.0 / (2.0 * .pi * cutoff)
        return 1.0 / (1.0 + tau / dt)
    }

    mutating func filter(_ x: Float, dt: Float) -> Float {
        let dtSafe = max(dt, 1e-6)
        guard initialized else {
            initialized = true; xPrev = x; dxPrev = 0
            return x
        }
        let dx    = (x - xPrev) / dtSafe
        let aD    = Self.alpha(cutoff: dCutoff, dt: dtSafe)
        let dxHat = aD * dx + (1 - aD) * dxPrev
        let cutoff = minCutoff + beta * abs(dxHat)
        let aX    = Self.alpha(cutoff: cutoff, dt: dtSafe)
        let xHat  = aX * x + (1 - aX) * xPrev
        xPrev  = xHat
        dxPrev = dxHat
        return xHat
    }

    /// Smoothed velocity (units/s) — drives extrapolation during dropouts.
    var velocity: Float { dxPrev }
    /// Last filtered position — anchor for extrapolation.
    var lastValue: Float { xPrev }

    mutating func reset() { initialized = false; xPrev = 0; dxPrev = 0 }
}

// ── tunables ──────────────────────────────────────────────────────────────────

private struct JoystickConfig: Codable {
    var deadzone_len:    Float
    var deadzone_x:      Float
    var deadzone_y_neg:  Float
    var deadzone_y_pos:  Float
    var fist_dist:       Float
    var extension_ratio: Float
    var hold_frames:     Int
    var oef_min_cutoff:  Float
    var oef_beta:        Float
    var oef_dcutoff:     Float

    static let defaults = JoystickConfig(
        deadzone_len:    0.10,
        deadzone_x:      0.14,
        deadzone_y_neg:  0.006,
        deadzone_y_pos:  0.120,
        fist_dist:       0.65,
        extension_ratio: 1.3,
        hold_frames:     4,
        oef_min_cutoff:  1.0,
        oef_beta:        0.05,
        oef_dcutoff:     1.0
    )

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let d = JoystickConfig.defaults
        deadzone_len    = (try? c.decodeIfPresent(Float.self, forKey: .deadzone_len))    ?? d.deadzone_len
        deadzone_x      = (try? c.decodeIfPresent(Float.self, forKey: .deadzone_x))      ?? d.deadzone_x
        deadzone_y_neg  = (try? c.decodeIfPresent(Float.self, forKey: .deadzone_y_neg))  ?? d.deadzone_y_neg
        deadzone_y_pos  = (try? c.decodeIfPresent(Float.self, forKey: .deadzone_y_pos))  ?? d.deadzone_y_pos
        fist_dist       = (try? c.decodeIfPresent(Float.self, forKey: .fist_dist))       ?? d.fist_dist
        extension_ratio = (try? c.decodeIfPresent(Float.self, forKey: .extension_ratio)) ?? d.extension_ratio
        hold_frames     = (try? c.decodeIfPresent(Int.self,   forKey: .hold_frames))     ?? d.hold_frames
        oef_min_cutoff  = (try? c.decodeIfPresent(Float.self, forKey: .oef_min_cutoff))  ?? d.oef_min_cutoff
        oef_beta        = (try? c.decodeIfPresent(Float.self, forKey: .oef_beta))        ?? d.oef_beta
        oef_dcutoff     = (try? c.decodeIfPresent(Float.self, forKey: .oef_dcutoff))     ?? d.oef_dcutoff
    }

    init(deadzone_len: Float, deadzone_x: Float, deadzone_y_neg: Float, deadzone_y_pos: Float,
         fist_dist: Float, extension_ratio: Float, hold_frames: Int,
         oef_min_cutoff: Float, oef_beta: Float, oef_dcutoff: Float) {
        self.deadzone_len    = deadzone_len
        self.deadzone_x      = deadzone_x
        self.deadzone_y_neg  = deadzone_y_neg
        self.deadzone_y_pos  = deadzone_y_pos
        self.fist_dist       = fist_dist
        self.extension_ratio = extension_ratio
        self.hold_frames     = hold_frames
        self.oef_min_cutoff  = oef_min_cutoff
        self.oef_beta        = oef_beta
        self.oef_dcutoff     = oef_dcutoff
    }

    static func load() -> JoystickConfig {
        let candidates: [URL?] = [
            Bundle.main.url(forResource: "joystick_config", withExtension: "json"),
            URL(fileURLWithPath: #filePath)
                .deletingLastPathComponent()
                .deletingLastPathComponent()
                .deletingLastPathComponent()
                .appendingPathComponent("controls/joystick_config.json")
        ]
        for url in candidates.compactMap({ $0 }) {
            if let data = try? Data(contentsOf: url),
               let cfg  = try? JSONDecoder().decode(JoystickConfig.self, from: data) {
                print("[HandJoystick] config loaded: \(url.path)")
                return cfg
            }
        }
        print("[HandJoystick] using default config (no joystick_config.json found)")
        return .defaults
    }
}

// ── public API mirrors KeyboardMonitor ────────────────────────────────────────

@MainActor
final class HandJoystick: NSObject, ObservableObject {
    @Published private(set) var keys: Set<UInt16> = []
    @Published private(set) var lastVec: SIMD2<Float> = .zero
    @Published private(set) var lastActive: Bool = false
    @Published private(set) var landmarks: [SIMD2<Float>] = []
    @Published private(set) var frameSize: CGSize = .zero

    var forward:  Bool { keys.contains(KeyboardMonitor.W) }
    var backward: Bool { keys.contains(KeyboardMonitor.S) }
    var left:     Bool { keys.contains(KeyboardMonitor.A) }
    var right:    Bool { keys.contains(KeyboardMonitor.D) }

    var captureSession: AVCaptureSession { session }

    private let cfg = JoystickConfig.load()

    nonisolated(unsafe) private let session     = AVCaptureSession()
    nonisolated(unsafe) private let videoOutput = AVCaptureVideoDataOutput()
    private let videoQueue  = DispatchQueue(label: "handjoystick.video",  qos: .userInitiated)
    private let visionQueue = DispatchQueue(label: "handjoystick.vision", qos: .userInitiated)

    // Backlog coalesce — drop frame if MainActor still draining previous one.
    nonisolated private let inFlight = OSAllocatedUnfairLock<Bool>(initialState: false)

    // 1€ filters: per-landmark, per-axis. 21 × 2 = 42 scalars.
    private var oefX: [OneEuroFilterScalar] = Array(repeating: OneEuroFilterScalar(), count: 21)
    private var oefY: [OneEuroFilterScalar] = Array(repeating: OneEuroFilterScalar(), count: 21)
    private var lastFrameTime: CFTimeInterval = 0   // 0 = uninitialized

    // Temporal hold + prediction.
    private var noHandStreak = 0

    // ── MainActor pipeline stats ──
    private var pipeCount = 0
    private var pipeSumMs = 0.0
    private var pipeMaxMs = 0.0

    // ── Vision-queue timing stats ──
    nonisolated(unsafe) private var statFrames   = 0
    nonisolated(unsafe) private var statDetected = 0
    nonisolated(unsafe) private var statSumInfer = 0.0
    nonisolated(unsafe) private var statMaxInfer = 0.0
    nonisolated(unsafe) private var statWindowT  = CACurrentMediaTime()
    private let statInterval = 60

    override init() {
        super.init()
        applyFilterParams()
    }

    private func applyFilterParams() {
        for i in 0..<21 {
            oefX[i].minCutoff = cfg.oef_min_cutoff
            oefX[i].beta      = cfg.oef_beta
            oefX[i].dCutoff   = cfg.oef_dcutoff
            oefY[i].minCutoff = cfg.oef_min_cutoff
            oefY[i].beta      = cfg.oef_beta
            oefY[i].dCutoff   = cfg.oef_dcutoff
        }
    }

    private func resetFilters() {
        for i in 0..<21 { oefX[i].reset(); oefY[i].reset() }
        lastFrameTime = 0
    }

    func start() {
        AVCaptureDevice.requestAccess(for: .video) { [weak self] ok in
            guard ok else { print("[HandJoystick] camera access denied"); return }
            guard let self else { return }
            self.videoQueue.async { self.configureAndRun() }
        }
    }

    func stop() {
        let s = session
        videoQueue.async { if s.isRunning { s.stopRunning() } }
        keys = []
        lastActive = false
        noHandStreak = 0
        resetFilters()
        inFlight.withLock { $0 = false }
    }

    nonisolated private func configureAndRun() {
        session.beginConfiguration()
        session.sessionPreset = .vga640x480

        guard let device = AVCaptureDevice.default(for: .video),
              let input  = try? AVCaptureDeviceInput(device: device),
              session.canAddInput(input) else {
            print("[HandJoystick] no camera input available")
            session.commitConfiguration()
            return
        }
        session.addInput(input)

        videoOutput.alwaysDiscardsLateVideoFrames = true
        videoOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        videoOutput.setSampleBufferDelegate(self, queue: visionQueue)
        if session.canAddOutput(videoOutput) { session.addOutput(videoOutput) }

        session.commitConfiguration()
        session.startRunning()
        print("[HandJoystick] capture session running")
    }

    // ── joystick geometry ─────────────────────────────────────────────────────

    private func handSize(_ lms: [SIMD2<Float>]) -> Float {
        let xs = lms.map { $0.x }; let ys = lms.map { $0.y }
        return max((xs.max() ?? 0) - (xs.min() ?? 0),
                   (ys.max() ?? 0) - (ys.min() ?? 0)) + 1e-6
    }

    private func isNeutralPose(_ lms: [SIMD2<Float>]) -> Bool {
        let size = handSize(lms)
        return [8, 12, 16, 20].filter { simd_distance(lms[$0], lms[0]) / size < cfg.fist_dist }.count >= 4
    }

    private func computeKeys(_ lms: [SIMD2<Float>]) -> (Set<UInt16>, SIMD2<Float>, Bool) {
        if isNeutralPose(lms) { return ([], .zero, false) }
        let mcp = lms[5]; let tip = lms[8]; let wrist = lms[0]
        if simd_distance(tip, wrist) < simd_distance(mcp, wrist) * cfg.extension_ratio {
            return ([], .zero, false)
        }
        let vecN = (tip - mcp) / handSize(lms)
        if simd_length(vecN) < cfg.deadzone_len { return ([], vecN, false) }
        var out: Set<UInt16> = []
        if      vecN.y < -cfg.deadzone_y_neg { out.insert(KeyboardMonitor.W) }
        else if vecN.y >  cfg.deadzone_y_pos { out.insert(KeyboardMonitor.S) }
        if      vecN.x < -cfg.deadzone_x    { out.insert(KeyboardMonitor.A) }
        else if vecN.x >  cfg.deadzone_x    { out.insert(KeyboardMonitor.D) }
        return (out, vecN, true)
    }
}

// ── camera frame → landmarks → keys ───────────────────────────────────────────

extension HandJoystick: AVCaptureVideoDataOutputSampleBufferDelegate {
    nonisolated func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        let busy = inFlight.withLock { state -> Bool in
            if state { return true }
            state = true; return false
        }
        if busy { return }

        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            inFlight.withLock { $0 = false }
            return
        }
        let width  = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        let tFrame = CACurrentMediaTime()

        let request = VNDetectHumanHandPoseRequest()
        request.maximumHandCount = 1
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .up, options: [:])
        let tInfer = CACurrentMediaTime()
        do { try handler.perform([request]) } catch {
            inFlight.withLock { $0 = false }
            return
        }
        let inferMs = (CACurrentMediaTime() - tInfer) * 1000.0

        statFrames += 1

        guard let obs = request.results?.first as? VNHumanHandPoseObservation else {
            logStatsIfNeeded()
            Task { @MainActor in self.handleNoHand() }
            return
        }

        statDetected += 1
        statSumInfer += inferMs
        if inferMs > statMaxInfer { statMaxInfer = inferMs }

        let order: [VNHumanHandPoseObservation.JointName] = [
            .wrist,
            .thumbCMC, .thumbMP, .thumbIP, .thumbTip,
            .indexMCP, .indexPIP, .indexDIP, .indexTip,
            .middleMCP, .middlePIP, .middleDIP, .middleTip,
            .ringMCP, .ringPIP, .ringDIP, .ringTip,
            .littleMCP, .littlePIP, .littleDIP, .littleTip
        ]

        var lms: [SIMD2<Float>] = []
        lms.reserveCapacity(21)
        let allPoints: [VNHumanHandPoseObservation.JointName: VNRecognizedPoint]
        do { allPoints = try obs.recognizedPoints(.all) } catch {
            inFlight.withLock { $0 = false }
            return
        }

        for joint in order {
            guard let p = allPoints[joint], p.confidence > 0.3 else {
                logStatsIfNeeded()
                Task { @MainActor in self.handleNoHand() }
                return
            }
            // Vision: x∈[0,1] right, y∈[0,1] bottom-up → mirror x, flip y → image coords.
            lms.append(SIMD2<Float>((1.0 - Float(p.location.x)) * Float(width),
                                    (1.0 - Float(p.location.y)) * Float(height)))
        }

        let finalLms = lms
        let size     = CGSize(width: width, height: height)
        let tFrameC  = tFrame
        logStatsIfNeeded()
        Task { @MainActor in
            self.processLandmarks(finalLms, size: size,
                                  pipelineMs: (CACurrentMediaTime() - tFrameC) * 1000.0)
        }
    }

    nonisolated private func logStatsIfNeeded() {
        guard statFrames >= statInterval else { return }
        let elapsed  = CACurrentMediaTime() - statWindowT
        let fps      = Double(statFrames) / max(elapsed, 1e-9)
        let detRate  = Double(statDetected) / Double(max(1, statFrames)) * 100.0
        let avgInfer = statSumInfer / Double(max(1, statDetected))
        print(String(format: "[HandJoystick] %.0f fps | detection %.0f%% | Vision infer avg %.1f ms  max %.1f ms",
                     fps, detRate, avgInfer, statMaxInfer))
        statFrames = 0; statDetected = 0; statSumInfer = 0; statMaxInfer = 0
        statWindowT = CACurrentMediaTime()
    }

    @MainActor
    private func handleNoHand() {
        noHandStreak += 1

        // Past hold limit: give up, clear keys, reset filters so re-detection starts clean.
        if noHandStreak >= cfg.hold_frames {
            if !keys.isEmpty      { keys = [] }
            if !landmarks.isEmpty { landmarks = [] }
            lastActive = false
            resetFilters()
            inFlight.withLock { $0 = false }
            return
        }

        // Within hold window — extrapolate landmarks using filter-estimated velocity.
        // dt = elapsed since last real frame; lastFrameTime intentionally NOT updated, so
        // when a real frame returns, its dt covers the full gap (filter sees a single
        // larger jump rather than several extrapolated micro-steps).
        let now = CACurrentMediaTime()
        let dt: Float = lastFrameTime == 0 ? Float(1.0/30.0) : Float(now - lastFrameTime)

        let predicted: [SIMD2<Float>] = (0..<21).map { i in
            SIMD2(oefX[i].lastValue + oefX[i].velocity * dt,
                  oefY[i].lastValue + oefY[i].velocity * dt)
        }

        // Run joystick decision on predicted landmarks — keys stay live during dropout.
        let (newKeys, vec, active) = computeKeys(predicted)
        if newKeys != keys { keys = newKeys }
        lastVec    = vec
        lastActive = active
        landmarks  = predicted

        inFlight.withLock { $0 = false }
    }

    @MainActor
    private func processLandmarks(_ raw: [SIMD2<Float>], size: CGSize, pipelineMs: Double = 0) {
        noHandStreak = 0

        pipeCount += 1
        pipeSumMs += pipelineMs
        if pipelineMs > pipeMaxMs { pipeMaxMs = pipelineMs }
        if pipeCount >= statInterval {
            print(String(format: "[HandJoystick] pipeline (frame→publish) avg %.1f ms  max %.1f ms",
                         pipeSumMs / Double(pipeCount), pipeMaxMs))
            pipeCount = 0; pipeSumMs = 0; pipeMaxMs = 0
        }

        // dt for 1€ filter — real elapsed time since last real frame. First frame uses 1/30s.
        let now = CACurrentMediaTime()
        let dt: Float = lastFrameTime == 0 ? Float(1.0/30.0) : Float(now - lastFrameTime)
        lastFrameTime = now

        let smoothed: [SIMD2<Float>] = (0..<21).map { i in
            SIMD2(oefX[i].filter(raw[i].x, dt: dt),
                  oefY[i].filter(raw[i].y, dt: dt))
        }

        let (newKeys, vec, active) = computeKeys(smoothed)
        if newKeys != keys { keys = newKeys }
        lastVec    = vec
        lastActive = active
        landmarks  = raw            // overlay shows raw — no smoothing lag visible
        frameSize  = size

        inFlight.withLock { $0 = false }
    }
}
