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

    static let defaults = JoystickConfig(
        deadzone_len:    0.10,
        deadzone_x:      0.14,
        deadzone_y_neg:  0.006,
        deadzone_y_pos:  0.120,
        ema_alpha:       0.50,
        fist_dist:       0.65,
        extension_ratio: 1.3
    )

    static func load() -> JoystickConfig {
        // Try bundle first, fall back to repo-relative path (works when running from Xcode build).
        let candidates: [URL?] = [
            Bundle.main.url(forResource: "joystick_config", withExtension: "json"),
            URL(fileURLWithPath: #filePath)
                .deletingLastPathComponent()                      // drivingsim/
                .deletingLastPathComponent()                      // drivingsim/drivingsim/ → drivingsim/
                .deletingLastPathComponent()                      // → repo root
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
    @Published private(set) var keys: Set<UInt16> = []   // reuses KeyboardMonitor codes
    @Published private(set) var lastVec: SIMD2<Float> = .zero      // for HUD
    @Published private(set) var lastActive: Bool = false           // for HUD
    @Published private(set) var landmarks: [SIMD2<Float>] = []     // 21 smoothed pixel coords
    @Published private(set) var frameSize: CGSize = .zero          // capture pixel dims

    var forward:  Bool { keys.contains(KeyboardMonitor.W) }
    var backward: Bool { keys.contains(KeyboardMonitor.S) }
    var left:     Bool { keys.contains(KeyboardMonitor.A) }
    var right:    Bool { keys.contains(KeyboardMonitor.D) }

    /// Live capture session — pass to AVCaptureVideoPreviewLayer for UI preview.
    var captureSession: AVCaptureSession { session }

    private let cfg = JoystickConfig.load()

    nonisolated(unsafe) private let session = AVCaptureSession()
    nonisolated(unsafe) private let videoOutput = AVCaptureVideoDataOutput()
    private let videoQueue = DispatchQueue(label: "handjoystick.video", qos: .userInitiated)
    private let visionQueue = DispatchQueue(label: "handjoystick.vision", qos: .userInitiated)

    private var emaLandmarks: [SIMD2<Float>]?      // 21 smoothed pixel landmarks

    // ── MainActor pipeline stats (stored here — extensions can't hold stored properties) ──
    private var pipeCount = 0
    private var pipeSumMs = 0.0
    private var pipeMaxMs = 0.0

    // ── timing stats (vision-queue writes, logged every statInterval frames) ──
    nonisolated(unsafe) private var statFrames     = 0
    nonisolated(unsafe) private var statDetected   = 0
    nonisolated(unsafe) private var statSumInfer   = 0.0   // ms
    nonisolated(unsafe) private var statMaxInfer   = 0.0   // ms
    nonisolated(unsafe) private var statWindowT    = CACurrentMediaTime()
    private let statInterval = 60

    func start() {
        AVCaptureDevice.requestAccess(for: .video) { [weak self] ok in
            guard ok else {
                print("[HandJoystick] camera access denied")
                return
            }
            guard let self else { return }
            self.videoQueue.async { self.configureAndRun() }
        }
    }

    func stop() {
        let s = session
        videoQueue.async {
            if s.isRunning { s.stopRunning() }
        }
        keys = []
        lastActive = false
        emaLandmarks = nil
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
        videoOutput.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        videoOutput.setSampleBufferDelegate(self, queue: visionQueue)
        if session.canAddOutput(videoOutput) { session.addOutput(videoOutput) }

        session.commitConfiguration()
        session.startRunning()
        print("[HandJoystick] capture session running")
    }

    // ── joystick geometry — mirrors compute_keys() in gesture_index_joystick.py

    private func handSize(_ lms: [SIMD2<Float>]) -> Float {
        let xs = lms.map { $0.x }
        let ys = lms.map { $0.y }
        let dx = (xs.max() ?? 0) - (xs.min() ?? 0)
        let dy = (ys.max() ?? 0) - (ys.min() ?? 0)
        return max(dx, dy) + 1e-6
    }

    private func isNeutralPose(_ lms: [SIMD2<Float>]) -> Bool {
        let tips = [8, 12, 16, 20]
        let size = handSize(lms)
        let tucked = tips.reduce(0) { acc, t in
            simd_distance(lms[t], lms[0]) / size < cfg.fist_dist ? acc + 1 : acc
        }
        return tucked >= 4
    }

    private func computeKeys(_ lms: [SIMD2<Float>]) -> (Set<UInt16>, SIMD2<Float>, Bool) {
        if isNeutralPose(lms) { return ([], .zero, false) }

        let mcp = lms[5]
        let tip = lms[8]
        let wrist = lms[0]

        let mcpWristDist = simd_distance(mcp, wrist)
        let tipWristDist = simd_distance(tip, wrist)
        if tipWristDist < mcpWristDist * cfg.extension_ratio {
            return ([], .zero, false)
        }

        let scale = handSize(lms)
        let vecN = (tip - mcp) / scale
        let mag = simd_length(vecN)
        if mag < cfg.deadzone_len {
            return ([], vecN, false)
        }

        var out: Set<UInt16> = []
        if vecN.y < -cfg.deadzone_y_neg { out.insert(KeyboardMonitor.W) }
        else if vecN.y > cfg.deadzone_y_pos { out.insert(KeyboardMonitor.S) }
        if vecN.x < -cfg.deadzone_x { out.insert(KeyboardMonitor.A) }
        else if vecN.x > cfg.deadzone_x { out.insert(KeyboardMonitor.D) }
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
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let width  = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)

        let tFrame = CACurrentMediaTime()

        let request = VNDetectHumanHandPoseRequest()
        request.maximumHandCount = 1
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .up, options: [:])
        let tInferStart = CACurrentMediaTime()
        do { try handler.perform([request]) } catch { return }
        let inferMs = (CACurrentMediaTime() - tInferStart) * 1000.0

        statFrames  += 1
        statSumInfer += inferMs
        if inferMs > statMaxInfer { statMaxInfer = inferMs }

        guard let obs = request.results?.first as? VNHumanHandPoseObservation else {
            logStatsIfNeeded()
            Task { @MainActor in self.handleNoHand() }
            return
        }
        statDetected += 1

        // Vision joint name → MediaPipe-style index 0..20
        let order: [VNHumanHandPoseObservation.JointName] = [
            .wrist,
            .thumbCMC, .thumbMP, .thumbIP, .thumbTip,                // 1..4
            .indexMCP, .indexPIP, .indexDIP, .indexTip,              // 5..8
            .middleMCP, .middlePIP, .middleDIP, .middleTip,          // 9..12
            .ringMCP, .ringPIP, .ringDIP, .ringTip,                  // 13..16
            .littleMCP, .littlePIP, .littleDIP, .littleTip           // 17..20
        ]

        var lms: [SIMD2<Float>] = []
        lms.reserveCapacity(21)
        let allPoints: [VNHumanHandPoseObservation.JointName: VNRecognizedPoint]
        do { allPoints = try obs.recognizedPoints(.all) } catch { return }

        for joint in order {
            guard let p = allPoints[joint], p.confidence > 0.3 else {
                Task { @MainActor in self.handleNoHand() }
                return
            }
            // Vision normalized: x∈[0,1] right, y∈[0,1] BOTTOM-up.
            // Python flips camera horizontally + uses image coords (y down).
            // Mirror x and flip y so the geometry signs match.
            let px = (1.0 - Float(p.location.x)) * Float(width)
            let py = (1.0 - Float(p.location.y)) * Float(height)
            lms.append(SIMD2<Float>(px, py))
        }

        let finalLms = lms
        let size = CGSize(width: width, height: height)
        let tFrameCopy = tFrame
        logStatsIfNeeded()
        Task { @MainActor in
            let pipeMs = (CACurrentMediaTime() - tFrameCopy) * 1000.0
            self.processLandmarks(finalLms, size: size, pipelineMs: pipeMs)
        }
    }

    nonisolated private func logStatsIfNeeded() {
        guard statFrames >= statInterval else { return }
        let elapsed = CACurrentMediaTime() - statWindowT
        let fps     = Double(statFrames) / elapsed
        let detRate = Double(statDetected) / Double(statFrames) * 100.0
        let avgInfer = statSumInfer / Double(max(1, statDetected > 0 ? statDetected : statFrames))
        print(String(format: "[HandJoystick] %.0f fps | detection %.0f%% | Vision infer avg %.1f ms  max %.1f ms",
                     fps, detRate, avgInfer, statMaxInfer))
        statFrames   = 0
        statDetected = 0
        statSumInfer = 0
        statMaxInfer = 0
        statWindowT  = CACurrentMediaTime()
    }

    @MainActor
    private func handleNoHand() {
        if !keys.isEmpty { keys = [] }
        lastActive = false
        emaLandmarks = nil
        if !landmarks.isEmpty { landmarks = [] }
    }

    @MainActor
    private func processLandmarks(_ raw: [SIMD2<Float>], size: CGSize, pipelineMs: Double = 0) {
        pipeCount  += 1
        pipeSumMs  += pipelineMs
        if pipelineMs > pipeMaxMs { pipeMaxMs = pipelineMs }
        if pipeCount >= 60 {
            print(String(format: "[HandJoystick] pipeline (frame→publish) avg %.1f ms  max %.1f ms",
                         pipeSumMs / Double(pipeCount), pipeMaxMs))
            pipeCount = 0; pipeSumMs = 0; pipeMaxMs = 0
        }
        // EMA on landmark positions for joystick decision (matches Python smoothing).
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
        lastVec = vec
        lastActive = active
        // Publish RAW landmarks for visual overlay — no EMA lag.
        landmarks = raw
        frameSize = size
    }
}
