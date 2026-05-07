//
//  HandJoystick.swift
//  drivingsim
//
//  Native Swift port of controls/gesture_index_joystick.py.
//  Uses AVFoundation for webcam + Vision (VNDetectHumanHandPoseRequest)
//  for 21 hand landmarks, then applies the same MCP→tip joystick geometry
//  to produce WASD intent. No Python, no synthetic keyboard.
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

// ── tunables ──────────────────────────────────────────────────────────────────

private struct JoystickConfig: Codable {
    var deadzone_len:    Float
    var deadzone_x:      Float
    var deadzone_y_neg:  Float
    var deadzone_y_pos:  Float
    var ema_alpha:       Float
    var fist_dist:       Float
    var extension_ratio: Float
    var hold_frames:     Int

    static let defaults = JoystickConfig(
        deadzone_len:    0.10,
        deadzone_x:      0.14,
        deadzone_y_neg:  0.006,
        deadzone_y_pos:  0.120,
        ema_alpha:       0.30,
        fist_dist:       0.65,
        extension_ratio: 1.3,
        hold_frames:     4
    )

    // Custom decoder: use decodeIfPresent per field so old JSON (missing hold_frames) still loads.
    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let d = JoystickConfig.defaults
        deadzone_len    = (try? c.decodeIfPresent(Float.self, forKey: .deadzone_len))    ?? d.deadzone_len
        deadzone_x      = (try? c.decodeIfPresent(Float.self, forKey: .deadzone_x))      ?? d.deadzone_x
        deadzone_y_neg  = (try? c.decodeIfPresent(Float.self, forKey: .deadzone_y_neg))  ?? d.deadzone_y_neg
        deadzone_y_pos  = (try? c.decodeIfPresent(Float.self, forKey: .deadzone_y_pos))  ?? d.deadzone_y_pos
        ema_alpha       = (try? c.decodeIfPresent(Float.self, forKey: .ema_alpha))       ?? d.ema_alpha
        fist_dist       = (try? c.decodeIfPresent(Float.self, forKey: .fist_dist))       ?? d.fist_dist
        extension_ratio = (try? c.decodeIfPresent(Float.self, forKey: .extension_ratio)) ?? d.extension_ratio
        hold_frames     = (try? c.decodeIfPresent(Int.self,   forKey: .hold_frames))     ?? d.hold_frames
    }

    // Memberwise init (used by defaults and tests)
    init(deadzone_len: Float, deadzone_x: Float, deadzone_y_neg: Float, deadzone_y_pos: Float,
         ema_alpha: Float, fist_dist: Float, extension_ratio: Float, hold_frames: Int) {
        self.deadzone_len    = deadzone_len
        self.deadzone_x      = deadzone_x
        self.deadzone_y_neg  = deadzone_y_neg
        self.deadzone_y_pos  = deadzone_y_pos
        self.ema_alpha       = ema_alpha
        self.fist_dist       = fist_dist
        self.extension_ratio = extension_ratio
        self.hold_frames     = hold_frames
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

    // Backlog coalesce: skip Vision + publish if previous MainActor update still pending.
    // OSAllocatedUnfairLock is safe to call from any thread/actor.
    nonisolated private let inFlight = OSAllocatedUnfairLock<Bool>(initialState: false)

    private var emaLandmarks: [SIMD2<Float>]?

    // Temporal hold: clear keys only after hold_frames consecutive no-hand frames.
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
        emaLandmarks = nil
        noHandStreak = 0
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
        // Drop frame if MainActor still processing previous one — prevents backlog drift.
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
            // Only log stats on no-hand path; don't accumulate infer sum (keeps avg honest).
            logStatsIfNeeded()
            Task { @MainActor in self.handleNoHand() }
            return
        }

        // Accumulate inference stats only on detected frames (prevents avg > max bug).
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
        if noHandStreak >= cfg.hold_frames {
            if !keys.isEmpty     { keys = [] }
            if !landmarks.isEmpty { landmarks = [] }
            lastActive   = false
            emaLandmarks = nil
        }
        inFlight.withLock { $0 = false }
    }

    @MainActor
    private func processLandmarks(_ raw: [SIMD2<Float>], size: CGSize, pipelineMs: Double = 0) {
        noHandStreak = 0   // hand detected — reset hold counter

        pipeCount += 1
        pipeSumMs += pipelineMs
        if pipelineMs > pipeMaxMs { pipeMaxMs = pipelineMs }
        if pipeCount >= statInterval {
            print(String(format: "[HandJoystick] pipeline (frame→publish) avg %.1f ms  max %.1f ms",
                         pipeSumMs / Double(pipeCount), pipeMaxMs))
            pipeCount = 0; pipeSumMs = 0; pipeMaxMs = 0
        }

        let smoothed: [SIMD2<Float>]
        if let prev = emaLandmarks, prev.count == raw.count {
            let a = cfg.ema_alpha
            smoothed = zip(prev, raw).map { p, r in a * p + (1 - a) * r }
        } else {
            smoothed = raw
        }
        emaLandmarks = smoothed

        let (newKeys, vec, active) = computeKeys(smoothed)
        if newKeys != keys { keys = newKeys }
        lastVec    = vec
        lastActive = active
        landmarks  = raw   // raw (unsmoothed) for overlay — no EMA lag
        frameSize  = size

        inFlight.withLock { $0 = false }
    }
}
