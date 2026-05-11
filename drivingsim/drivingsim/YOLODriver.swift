//
//  YOLODriver.swift
//  drivingsim
//
//  CoreML YOLO person detector. Peer to DepthDriver — consumes the same
//  CVPixelBuffer from SimFPVRenderer, publishes [PersonDetection] in
//  normalized [0,1] image-space. No steering logic — SimScene composes
//  detections with depth in its ROAM↔SEEK state machine.
//

@preconcurrency import CoreML
@preconcurrency import CoreVideo
import Combine
import CoreGraphics
import CoreImage
import Foundation
import os
import SwiftUI

enum SeekState { case roam, seek }

struct PersonDetection: Identifiable {
    let id = UUID()
    let cx: Float   // normalized [0,1] in model-input image space
    let cy: Float
    let w:  Float
    let h:  Float
    let confidence: Float
}

@MainActor
final class YOLODriver: ObservableObject {
    @Published private(set) var detections: [PersonDetection] = []
    @Published private(set) var bestDetection: PersonDetection?
    @Published private(set) var hasTarget: Bool = false
    @Published private(set) var previewImage: CGImage?
    @Published private(set) var inputFrameSize: CGSize = CGSize(width: 518, height: 518)
    @Published private(set) var seekState: SeekState = .roam

    // Tunables
    private let confThreshold:    Float = 0.35
    private let nmsIoUThreshold:  Float = 0.45
    private let personClassId:    Int   = 0
    private let seekEnterFrames:  Int   = 3
    private let seekExitFrames:   Int   = 30
    private let seekEnterConf:    Float = 0.50

    private var seekDetectFrames = 0
    private var seekLostFrames   = 0

    // Model
    nonisolated(unsafe) private var model: MLModel?
    nonisolated(unsafe) private var modelInputName: String  = "image"
    nonisolated(unsafe) private var modelInputW: Int = 640
    nonisolated(unsafe) private var modelInputH: Int = 640
    // Discovered output(s)
    nonisolated(unsafe) private var outputNames: [String] = []
    nonisolated(unsafe) private var outputShapes: [String: [Int]] = [:]
    // Mode A = single tensor [1, 84, 8400] (ultralytics raw)
    // Mode B = post-NMS coordinates + confidence (Vision-style)
    nonisolated(unsafe) private var decodeMode: DecodeMode = .unknown

    private enum DecodeMode { case unknown, rawYOLO, postNMS }

    private let visionQueue = DispatchQueue(label: "yolodriver.vision", qos: .userInitiated)
    nonisolated private let inFlight = OSAllocatedUnfairLock<Bool>(initialState: false)
    nonisolated(unsafe) private static let ciContext = CIContext(options: [.useSoftwareRenderer: false])
    nonisolated(unsafe) private var resizedPool: CVPixelBufferPool?
    private var running = false

    // Stats
    nonisolated(unsafe) private var statFrames = 0
    nonisolated(unsafe) private var statSumInfer = 0.0
    nonisolated(unsafe) private var statMaxInfer = 0.0
    nonisolated(unsafe) private var statDetsSum = 0
    nonisolated(unsafe) private var statWindowT = CACurrentMediaTime()

    init() { loadModel() }

    private func loadModel() {
        let cfg = MLModelConfiguration()
        cfg.computeUnits = .all
        guard let url = Bundle.main.url(forResource: "yolo26n", withExtension: "mlmodelc")
                     ?? Bundle.main.url(forResource: "yolo26n", withExtension: "mlpackage")
                     ?? Bundle.main.url(forResource: "yolo26nbase", withExtension: "mlmodelc")
                     ?? Bundle.main.url(forResource: "yolo26nbase", withExtension: "mlpackage") else {
            print("[YOLODriver] model not found in bundle (looked for yolo26n / yolo26nbase)")
            return
        }
        do {
            let m = try MLModel(contentsOf: url, configuration: cfg)
            self.model = m
            // Pick the IMAGE input (dict order is unstable; can't use .first).
            for (name, desc) in m.modelDescription.inputDescriptionsByName {
                if let img = desc.imageConstraint {
                    self.modelInputName = name
                    self.modelInputW = img.pixelsWide
                    self.modelInputH = img.pixelsHigh
                    break
                }
            }
            var names: [String] = []
            var shapes: [String: [Int]] = [:]
            for (name, desc) in m.modelDescription.outputDescriptionsByName {
                names.append(name)
                if let c = desc.multiArrayConstraint {
                    shapes[name] = c.shape.map { $0.intValue }
                }
            }
            self.outputNames = names.sorted()
            self.outputShapes = shapes

            // Heuristic: ultralytics CoreML export commonly emits a single
            // multiArray [1, 84, 8400] (raw). Post-NMS export emits separate
            // "coordinates"/"confidence" arrays.
            if shapes.count == 1, let only = shapes.values.first,
               only.count == 3, only[1] >= 5 && only[1] <= 256 && only[2] > 100 {
                self.decodeMode = .rawYOLO
            } else if shapes.keys.contains(where: { $0.lowercased().contains("coord") }) {
                self.decodeMode = .postNMS
            } else {
                self.decodeMode = .rawYOLO   // safest default
            }

            print("[YOLODriver] model loaded — input \(modelInputName) \(modelInputW)x\(modelInputH)")
            for n in outputNames {
                print("[YOLODriver]   output \(n) shape \(outputShapes[n] ?? [])")
            }
            print("[YOLODriver] decodeMode = \(decodeMode)")
        } catch {
            print("[YOLODriver] failed to load model: \(error)")
        }
    }

    var inputSize: (width: Int, height: Int) { (modelInputW, modelInputH) }

    func start() { running = true }
    func stop() {
        running = false
        detections = []
        bestDetection = nil
        hasTarget = false
        previewImage = nil
        seekState = .roam
        seekDetectFrames = 0
        seekLostFrames = 0
        inFlight.withLock { $0 = false }
    }

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

        // Ultralytics pipeline first stage is a strict 640×640 image — CoreML
        // does NOT auto-resize. Force-resize source CVPixelBuffer to model size.
        let srcW = CVPixelBufferGetWidth(pixelBuffer)
        let srcH = CVPixelBufferGetHeight(pixelBuffer)
        let inputBuf: CVPixelBuffer
        if srcW == modelInputW && srcH == modelInputH {
            inputBuf = pixelBuffer
        } else {
            guard let resized = resizeToBGRA(pixelBuffer, toW: modelInputW, toH: modelInputH) else {
                print("[YOLODriver] resize failed (src \(srcW)x\(srcH) → \(modelInputW)x\(modelInputH))")
                inFlight.withLock { $0 = false }
                return
            }
            inputBuf = resized
        }

        let provider: MLFeatureProvider
        do {
            var dict: [String: MLFeatureValue] = [
                modelInputName: MLFeatureValue(pixelBuffer: inputBuf)
            ]
            dict["iouThreshold"] = MLFeatureValue(double: Double(nmsIoUThreshold))
            dict["confidenceThreshold"] = MLFeatureValue(double: Double(confThreshold))
            provider = try MLDictionaryFeatureProvider(dictionary: dict)
        } catch {
            print("[YOLODriver] feature build failed: \(error)")
            inFlight.withLock { $0 = false }
            return
        }

        let result: MLFeatureProvider
        do {
            result = try model.prediction(from: provider)
        } catch {
            print("[YOLODriver] prediction failed: \(error)")
            inFlight.withLock { $0 = false }
            return
        }

        let inferMs = (CACurrentMediaTime() - t0) * 1000.0
        statFrames += 1
        statSumInfer += inferMs
        if inferMs > statMaxInfer { statMaxInfer = inferMs }

        // Decode by detected mode
        var dets: [PersonDetection] = []
        switch decodeMode {
        case .rawYOLO, .unknown:
            if let name = outputNames.first,
               let arr = result.featureValue(for: name)?.multiArrayValue {
                dets = Self.decodeRawYOLO(arr,
                                          inputW: modelInputW,
                                          inputH: modelInputH,
                                          conf: confThreshold,
                                          personClassId: personClassId)
                dets = Self.nms(dets, iou: nmsIoUThreshold)
            }
        case .postNMS:
            // Expect "coordinates" (xywh in normalized) + "confidence" (N×classes)
            if let coordVal = result.featureValue(for:
                    outputNames.first(where: { $0.lowercased().contains("coord") }) ?? "coordinates"),
               let confVal = result.featureValue(for:
                    outputNames.first(where: { $0.lowercased().contains("conf") }) ?? "confidence"),
               let coords = coordVal.multiArrayValue, let confs = confVal.multiArrayValue {
                dets = Self.decodePostNMS(coords: coords, confs: confs,
                                          conf: confThreshold,
                                          personClassId: personClassId)
            }
        }

        statDetsSum += dets.count

        let preview = Self.makePreviewCGImage(pixelBuffer)
        let frameW = CVPixelBufferGetWidth(pixelBuffer)
        let frameH = CVPixelBufferGetHeight(pixelBuffer)

        logStatsIfNeeded()

        Task { @MainActor [self] in
            self.publish(dets: dets, preview: preview, frameW: frameW, frameH: frameH)
            self.inFlight.withLock { $0 = false }
        }
    }

    @MainActor
    private func publish(dets: [PersonDetection], preview: CGImage?, frameW: Int, frameH: Int) {
        detections = dets
        let best = dets.max(by: { $0.confidence < $1.confidence })
        bestDetection = best
        hasTarget = (best != nil)
        previewImage = preview
        inputFrameSize = CGSize(width: frameW, height: frameH)

        // ── ROAM ↔ SEEK state machine with hysteresis ─────────────────
        if let b = best, b.confidence >= seekEnterConf {
            seekDetectFrames += 1
            seekLostFrames = 0
            if seekState == .roam && seekDetectFrames >= seekEnterFrames {
                seekState = .seek
                print("[YOLODriver] ROAM → SEEK (conf=\(b.confidence))")
            }
        } else {
            seekLostFrames += 1
            seekDetectFrames = 0
            if seekState == .seek && seekLostFrames >= seekExitFrames {
                seekState = .roam
                print("[YOLODriver] SEEK → ROAM (target lost)")
            }
        }
    }

    nonisolated private func logStatsIfNeeded() {
        guard statFrames >= 30 else { return }
        let elapsed = CACurrentMediaTime() - statWindowT
        let fps = Double(statFrames) / max(elapsed, 1e-9)
        let avg = statSumInfer / Double(max(1, statFrames))
        let dpf = Double(statDetsSum) / Double(max(1, statFrames))
        print(String(format: "[YOLODriver] %.0f fps | infer avg %.1f ms  max %.1f ms | dets/frame=%.2f",
                     fps, avg, statMaxInfer, dpf))
        statFrames = 0; statSumInfer = 0; statMaxInfer = 0; statDetsSum = 0
        statWindowT = CACurrentMediaTime()
    }

    // ── Resize helper ─────────────────────────────────────────────────────

    nonisolated private func resizeToBGRA(_ src: CVPixelBuffer, toW: Int, toH: Int) -> CVPixelBuffer? {
        if resizedPool == nil {
            let poolAttrs = [kCVPixelBufferPoolMinimumBufferCountKey: 3] as CFDictionary
            let pbAttrs: [CFString: Any] = [
                kCVPixelBufferPixelFormatTypeKey: kCVPixelFormatType_32BGRA,
                kCVPixelBufferWidthKey: toW,
                kCVPixelBufferHeightKey: toH,
                kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary
            ]
            var pool: CVPixelBufferPool?
            CVPixelBufferPoolCreate(nil, poolAttrs, pbAttrs as CFDictionary, &pool)
            resizedPool = pool
        }
        guard let pool = resizedPool else { return nil }
        var dst: CVPixelBuffer?
        CVPixelBufferPoolCreatePixelBuffer(nil, pool, &dst)
        guard let out = dst else { return nil }

        let srcW = CGFloat(CVPixelBufferGetWidth(src))
        let srcH = CGFloat(CVPixelBufferGetHeight(src))
        let sx = CGFloat(toW) / srcW
        let sy = CGFloat(toH) / srcH
        let ci = CIImage(cvPixelBuffer: src)
                    .transformed(by: CGAffineTransform(scaleX: sx, y: sy))
        Self.ciContext.render(ci, to: out)
        return out
    }

    // ── Decoders ──────────────────────────────────────────────────────────

    /// Decode ultralytics raw output [1, 4+C, A] (cx,cy,w,h in input pixels, then C class scores).
    nonisolated static func decodeRawYOLO(_ arr: MLMultiArray,
                                          inputW: Int, inputH: Int,
                                          conf: Float, personClassId: Int) -> [PersonDetection] {
        let shape = arr.shape.map { $0.intValue }
        // Expect [1, F, A] or [F, A].
        let F: Int
        let A: Int
        let batched: Bool
        if shape.count == 3 {
            F = shape[1]; A = shape[2]; batched = true
        } else if shape.count == 2 {
            F = shape[0]; A = shape[1]; batched = false
        } else {
            return []
        }
        guard F >= 5, A > 0 else { return [] }
        let numClasses = F - 4
        let personIdx = max(0, min(numClasses - 1, personClassId))
        let ptr = arr.dataPointer.assumingMemoryBound(to: Float32.self)
        let stridesArr = arr.strides.map { $0.intValue }
        // Compute base strides
        let sF: Int
        let sA: Int
        if batched {
            sF = stridesArr[1]; sA = stridesArr[2]
        } else {
            sF = stridesArr[0]; sA = stridesArr[1]
        }

        var dets: [PersonDetection] = []
        dets.reserveCapacity(64)
        let invW = 1.0 / Float(inputW)
        let invH = 1.0 / Float(inputH)

        for i in 0..<A {
            let pCls = ptr[(4 + personIdx) * sF + i * sA]
            let pScore = sigmoidIfNeeded(pCls)
            if pScore < conf { continue }
            // Quick max-class check — ensure person dominates (skip otherwise)
            var maxC = pScore
            for c in 0..<numClasses where c != personIdx {
                let v = sigmoidIfNeeded(ptr[(4 + c) * sF + i * sA])
                if v > maxC { maxC = v; break }
            }
            if maxC > pScore { continue }

            let cx = ptr[0 * sF + i * sA]
            let cy = ptr[1 * sF + i * sA]
            let bw = ptr[2 * sF + i * sA]
            let bh = ptr[3 * sF + i * sA]
            // Normalize to [0,1]
            let ncx = cx * invW
            let ncy = cy * invH
            let nw  = bw * invW
            let nh  = bh * invH
            if nw <= 0 || nh <= 0 { continue }
            if ncx < 0 || ncx > 1 || ncy < 0 || ncy > 1 { continue }
            dets.append(PersonDetection(cx: ncx, cy: ncy, w: nw, h: nh, confidence: pScore))
        }
        return dets
    }

    /// If raw scores look like logits (any |v|>1) apply sigmoid; otherwise pass through.
    /// Ultralytics CoreML exports usually already-sigmoid; we be defensive.
    nonisolated static func sigmoidIfNeeded(_ v: Float) -> Float {
        if v >= 0 && v <= 1 { return v }
        // sigmoid
        return 1.0 / (1.0 + expf(-v))
    }

    /// Decode post-NMS output (Vision-style: coordinates xywh normalized, confidence per-class).
    nonisolated static func decodePostNMS(coords: MLMultiArray, confs: MLMultiArray,
                                          conf: Float, personClassId: Int) -> [PersonDetection] {
        let cShape = coords.shape.map { $0.intValue }
        let kShape = confs.shape.map { $0.intValue }
        // coords: [N, 4]; confs: [N, C]
        let N = cShape.first ?? 0
        let C = kShape.last ?? 0
        guard N > 0, C > personClassId else { return [] }
        let cPtr = coords.dataPointer.assumingMemoryBound(to: Float32.self)
        let kPtr = confs.dataPointer.assumingMemoryBound(to: Float32.self)
        let cStride = coords.strides.map { $0.intValue }
        let kStride = confs.strides.map { $0.intValue }
        var dets: [PersonDetection] = []
        for i in 0..<N {
            let s = kPtr[i * kStride[0] + personClassId * (kStride.count > 1 ? kStride[1] : 1)]
            if s < conf { continue }
            let cx = cPtr[i * cStride[0] + 0]
            let cy = cPtr[i * cStride[0] + 1]
            let bw = cPtr[i * cStride[0] + 2]
            let bh = cPtr[i * cStride[0] + 3]
            dets.append(PersonDetection(cx: cx, cy: cy, w: bw, h: bh, confidence: s))
        }
        return dets
    }

    nonisolated static func nms(_ dets: [PersonDetection], iou: Float) -> [PersonDetection] {
        let sorted = dets.sorted { $0.confidence > $1.confidence }
        var keep: [PersonDetection] = []
        for d in sorted {
            var ok = true
            for k in keep {
                if iouOf(d, k) > iou { ok = false; break }
            }
            if ok { keep.append(d) }
            if keep.count >= 16 { break }
        }
        return keep
    }

    nonisolated static func iouOf(_ a: PersonDetection, _ b: PersonDetection) -> Float {
        let ax1 = a.cx - a.w/2, ax2 = a.cx + a.w/2
        let ay1 = a.cy - a.h/2, ay2 = a.cy + a.h/2
        let bx1 = b.cx - b.w/2, bx2 = b.cx + b.w/2
        let by1 = b.cy - b.h/2, by2 = b.cy + b.h/2
        let ix1 = max(ax1, bx1), iy1 = max(ay1, by1)
        let ix2 = min(ax2, bx2), iy2 = min(ay2, by2)
        let iw = max(0, ix2 - ix1), ih = max(0, iy2 - iy1)
        let inter = iw * ih
        let areaA = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        let areaB = max(0, bx2 - bx1) * max(0, by2 - by1)
        let uni = areaA + areaB - inter
        return uni > 1e-6 ? inter / uni : 0
    }

    // ── Preview ──────────────────────────────────────────────────────────

    nonisolated static func makePreviewCGImage(_ pb: CVPixelBuffer) -> CGImage? {
        let ci = CIImage(cvPixelBuffer: pb)
        let ctx = CIContext(options: [.useSoftwareRenderer: false])
        return ctx.createCGImage(ci, from: ci.extent)
    }
}
