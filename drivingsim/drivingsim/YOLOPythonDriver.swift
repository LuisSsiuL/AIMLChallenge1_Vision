//
//  YOLOPythonDriver.swift
//  drivingsim
//
//  Alternate YOLO source: spawns python_mps_detector.py from PyTorch-MPS as a
//  subprocess, ships JPEG frames over stdin, reads detections from stdout.
//  Same public surface as YOLODriver so SimScene/ContentView can swap with
//  minimal branching.
//

@preconcurrency import CoreVideo
import AppKit
import Combine
import CoreGraphics
import CoreImage
import Foundation
import SwiftUI
import os

@MainActor
final class YOLOPythonDriver: ObservableObject {
    @Published private(set) var detections: [PersonDetection] = []
    @Published private(set) var bestDetection: PersonDetection?
    @Published private(set) var hasTarget: Bool = false
    @Published private(set) var previewImage: CGImage?
    @Published private(set) var inputFrameSize: CGSize = CGSize(width: 518, height: 518)
    @Published private(set) var seekState: SeekState = .roam
    @Published private(set) var pythonReady: Bool = false

    // Config (matches python_mps_detector.py env contract)
    private let pythonPathCandidates: [String] = [
        ProcessInfo.processInfo.environment["YOLO_PYTHON"] ?? "",
        "/Users/christianluisefendy/.pyenv/versions/3.9.5/bin/python",
        "/Users/christianluisefendy/.pyenv/versions/3.13.3/bin/python",
        "/opt/homebrew/bin/python3",
        "/usr/bin/python3",
    ].filter { !$0.isEmpty }
    private let scriptPath = "/Users/christianluisefendy/Documents/PyTorch-MPS/WebCam/python_mps_detector.py"
    private let modelPath  = "/Users/christianluisefendy/Documents/PyTorch-MPS/WebCam/yolo26nbase.pt"

    private let confThreshold:   Float = 0.35
    private let personClassId:   Int   = 0
    private let seekEnterFrames: Int   = 3
    private let seekExitFrames:  Int   = 30
    private let seekEnterConf:   Float = 0.50
    private let jpegQuality:     CGFloat = 0.6
    private let jpegMaxSide:     CGFloat = 640

    private var seekDetectFrames = 0
    private var seekLostFrames   = 0

    // Subprocess state
    private var process: Process?
    private var stdinPipe:  Pipe?
    private var stdoutPipe: Pipe?
    private var stderrPipe: Pipe?
    private var stdoutBuffer = Data()
    private var running = false
    private var frameIdCounter: Int = 0
    private var pendingFrames: Int = 0
    private let maxPending: Int = 1   // drop-if-busy

    private let encodeQueue = DispatchQueue(label: "yolopy.encode", qos: .userInitiated)
    nonisolated(unsafe) private static let ciContext = CIContext(options: [.useSoftwareRenderer: false])

    // Stats
    private var statFrames = 0
    private var statDetsSum = 0
    private var statWindowT = CACurrentMediaTime()

    func start() {
        guard !running else { return }
        running = true
        seekState = .roam
        seekDetectFrames = 0
        seekLostFrames = 0
        pendingFrames = 0
        stdoutBuffer.removeAll()
        spawnIfNeeded()
    }

    func stop() {
        running = false
        terminateProcess()
        detections = []
        bestDetection = nil
        hasTarget = false
        previewImage = nil
        seekState = .roam
        seekDetectFrames = 0
        seekLostFrames = 0
        pendingFrames = 0
        pythonReady = false
        stdoutBuffer.removeAll()
    }

    private func spawnIfNeeded() {
        guard process == nil else { return }
        guard let py = pythonPathCandidates.first(where: { FileManager.default.isExecutableFile(atPath: $0) }) else {
            print("[YOLOPyDriver] no usable python interpreter found; tried \(pythonPathCandidates)")
            return
        }
        guard FileManager.default.fileExists(atPath: scriptPath) else {
            print("[YOLOPyDriver] script not found at \(scriptPath)")
            return
        }
        guard FileManager.default.fileExists(atPath: modelPath) else {
            print("[YOLOPyDriver] model not found at \(modelPath)")
            return
        }

        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: py)
        proc.arguments = [scriptPath]

        var env = ProcessInfo.processInfo.environment
        env["YOLO_MODEL_PATH"] = modelPath
        env["YOLO_DEVICE"] = "mps"
        env["YOLO_CONF_THRESHOLD"] = String(confThreshold)
        env["YOLO_TARGET_KEYWORDS"] = "person"
        env["YOLO_CONFIG_DIR"] = NSTemporaryDirectory() + "ultralytics"
        env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        proc.environment = env

        let sin = Pipe(); let sout = Pipe(); let serr = Pipe()
        proc.standardInput  = sin
        proc.standardOutput = sout
        proc.standardError  = serr
        self.stdinPipe = sin
        self.stdoutPipe = sout
        self.stderrPipe = serr

        sout.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            if data.isEmpty { return }
            Task { @MainActor [weak self] in self?.consumeStdout(data) }
        }
        serr.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            if data.isEmpty { return }
            if let s = String(data: data, encoding: .utf8) {
                for line in s.split(separator: "\n") {
                    print("[YOLOPy/stderr] \(line)")
                }
            }
        }
        proc.terminationHandler = { [weak self] p in
            Task { @MainActor [weak self] in
                self?.pythonReady = false
                print("[YOLOPyDriver] subprocess exited code=\(p.terminationStatus)")
            }
        }

        do {
            try proc.run()
            self.process = proc
            print("[YOLOPyDriver] spawned \(py) \(scriptPath)")
        } catch {
            print("[YOLOPyDriver] failed to launch python: \(error)")
            self.process = nil
        }
    }

    private func terminateProcess() {
        stdoutPipe?.fileHandleForReading.readabilityHandler = nil
        stderrPipe?.fileHandleForReading.readabilityHandler = nil
        process?.terminate()
        process = nil
        stdinPipe = nil
        stdoutPipe = nil
        stderrPipe = nil
    }

    // MARK: - Frame submission

    func submit(_ pixelBuffer: CVPixelBuffer) {
        guard running, pythonReady else { return }
        if pendingFrames >= maxPending { return }
        pendingFrames += 1
        frameIdCounter += 1
        let fid = frameIdCounter

        let w = CVPixelBufferGetWidth(pixelBuffer)
        let h = CVPixelBufferGetHeight(pixelBuffer)
        self.inputFrameSize = CGSize(width: w, height: h)

        // Encode off main actor (also build preview CGImage)
        let ci = CIImage(cvPixelBuffer: pixelBuffer)
        if let cg = Self.ciContext.createCGImage(ci, from: ci.extent) {
            self.previewImage = cg
        }
        encodeQueue.async { [weak self] in
            guard let self else { return }
            let scale = min(self.jpegMaxSide / CGFloat(w), self.jpegMaxSide / CGFloat(h), 1.0)
            let img = scale < 1.0
                ? ci.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
                : ci
            guard let jpeg = self.encodeJPEG(img) else {
                Task { @MainActor [weak self] in self?.pendingFrames = max(0, (self?.pendingFrames ?? 1) - 1) }
                return
            }
            let b64 = jpeg.base64EncodedString()
            let payload: [String: Any] = [
                "frame_id": fid,
                "conf": self.confThreshold,
                "jpeg_b64": b64,
            ]
            guard let data = try? JSONSerialization.data(withJSONObject: payload, options: []) else {
                Task { @MainActor [weak self] in self?.pendingFrames = max(0, (self?.pendingFrames ?? 1) - 1) }
                return
            }
            var line = data
            line.append(0x0A)   // newline delimiter
            Task { @MainActor [weak self] in
                self?.writeToStdin(line)
            }
        }
    }

    private func writeToStdin(_ data: Data) {
        guard let pipe = stdinPipe else {
            pendingFrames = max(0, pendingFrames - 1)
            return
        }
        do {
            try pipe.fileHandleForWriting.write(contentsOf: data)
        } catch {
            print("[YOLOPyDriver] stdin write failed: \(error)")
            pendingFrames = max(0, pendingFrames - 1)
        }
    }

    nonisolated private func encodeJPEG(_ ci: CIImage) -> Data? {
        let cs = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
        return Self.ciContext.jpegRepresentation(of: ci,
                                                 colorSpace: cs,
                                                 options: [kCGImageDestinationLossyCompressionQuality as CIImageRepresentationOption: jpegQuality])
    }

    // MARK: - Stdout parsing

    private func consumeStdout(_ chunk: Data) {
        stdoutBuffer.append(chunk)
        while let nlIdx = stdoutBuffer.firstIndex(of: 0x0A) {
            let lineData = stdoutBuffer.prefix(upTo: nlIdx)
            stdoutBuffer.removeSubrange(stdoutBuffer.startIndex...nlIdx)
            guard let obj = try? JSONSerialization.jsonObject(with: lineData) as? [String: Any] else {
                continue
            }
            handleMessage(obj)
        }
    }

    private func handleMessage(_ obj: [String: Any]) {
        if let ready = obj["ready"] as? Bool, ready {
            pythonReady = true
            print("[YOLOPyDriver] python ready device=\(obj["device"] ?? "?")")
            return
        }
        if let err = obj["error"] as? String {
            print("[YOLOPyDriver] python error: \(err)")
            return
        }
        // Detection message: { frame_id, detections: [{label, confidence, x, y, w, h}] }
        guard let detsArr = obj["detections"] as? [[String: Any]] else {
            // Some messages may omit; treat as zero detections
            publish(dets: [])
            pendingFrames = max(0, pendingFrames - 1)
            return
        }
        var dets: [PersonDetection] = []
        for d in detsArr {
            let label = (d["label"] as? String) ?? ""
            if label.lowercased() != "person" { continue }
            let conf = Float((d["confidence"] as? Double) ?? 0)
            let x = Float((d["x"] as? Double) ?? 0)
            let y = Float((d["y"] as? Double) ?? 0)
            let w = Float((d["w"] as? Double) ?? 0)
            let h = Float((d["h"] as? Double) ?? 0)
            // Python emits top-left x,y normalized; Swift uses center.
            dets.append(PersonDetection(cx: x + w/2, cy: y + h/2,
                                        w: w, h: h, confidence: conf))
        }
        publish(dets: dets)
        pendingFrames = max(0, pendingFrames - 1)

        statFrames += 1
        statDetsSum += dets.count
        if statFrames >= 30 {
            let elapsed = CACurrentMediaTime() - statWindowT
            let fps = Double(statFrames) / max(elapsed, 1e-9)
            print(String(format: "[YOLOPyDriver] %.0f fps | dets/frame=%.2f",
                         fps, Double(statDetsSum)/Double(statFrames)))
            statFrames = 0; statDetsSum = 0; statWindowT = CACurrentMediaTime()
        }
    }

    private func publish(dets: [PersonDetection]) {
        detections = dets
        let best = dets.max(by: { $0.confidence < $1.confidence })
        bestDetection = best
        hasTarget = (best != nil)

        if let b = best, b.confidence >= seekEnterConf {
            seekDetectFrames += 1
            seekLostFrames = 0
            if seekState == .roam && seekDetectFrames >= seekEnterFrames {
                seekState = .seek
                print("[YOLOPyDriver] ROAM → SEEK (conf=\(b.confidence))")
            }
        } else {
            seekLostFrames += 1
            seekDetectFrames = 0
            if seekState == .seek && seekLostFrames >= seekExitFrames {
                seekState = .roam
                print("[YOLOPyDriver] SEEK → ROAM (target lost)")
            }
        }
    }
}
