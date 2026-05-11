//
//  LiveCameraSource.swift
//  drivingsim
//
//  AVCaptureSession wrapper that publishes the latest video frame as a
//  CVPixelBuffer at the inference resolution (518×518, BGRA, IOSurface-backed
//  for Metal compatibility). Designed as a drop-in replacement for
//  SimFPVRenderer when a real camera (built-in / external / Continuity)
//  feeds depth + YOLO instead of the sim render.
//

@preconcurrency import AVFoundation
@preconcurrency import CoreVideo
import Combine
import CoreImage
import Foundation
import os

@MainActor
final class LiveCameraSource: NSObject, ObservableObject {
    @Published private(set) var isRunning: Bool = false
    @Published private(set) var lastError: String?

    /// Latest captured frame, scaled + colour-converted to 518×518 BGRA.
    /// Read on MainActor; updated from capture callback.
    private(set) var latestPixelBuffer: CVPixelBuffer?

    /// Make the underlying AVCaptureSession available so SwiftUI can show
    /// a preview layer when wanted.
    nonisolated(unsafe) let session = AVCaptureSession()

    private let outputW: Int
    private let outputH: Int
    nonisolated(unsafe) private let videoOutput = AVCaptureVideoDataOutput()
    private let outputQueue = DispatchQueue(label: "livecam.output", qos: .userInitiated)
    private let ciContext = CIContext(options: [.useSoftwareRenderer: false])
    nonisolated(unsafe) private var pool: CVPixelBufferPool?

    nonisolated private let bufferStore = BufferStore()

    init(width: Int = 518, height: Int = 518) {
        self.outputW = width
        self.outputH = height
        super.init()
        configurePool()
    }

    /// Configure the session for a specific AVCaptureDevice (selected by
    /// uniqueID from CameraScanner). Idempotent — safe to call repeatedly.
    func configure(deviceUniqueID: String) {
        session.beginConfiguration()
        defer { session.commitConfiguration() }

        // Wipe any existing inputs/outputs.
        for input in session.inputs { session.removeInput(input) }
        for output in session.outputs { session.removeOutput(output) }

        guard let device = AVCaptureDevice(uniqueID: deviceUniqueID) else {
            lastError = "Device \(deviceUniqueID) not found"
            return
        }
        do {
            let input = try AVCaptureDeviceInput(device: device)
            if session.canAddInput(input) {
                session.addInput(input)
            } else {
                lastError = "Cannot add input for \(device.localizedName)"
                return
            }
        } catch {
            lastError = "AVCaptureDeviceInput failed: \(error)"
            return
        }

        videoOutput.setSampleBufferDelegate(self, queue: outputQueue)
        videoOutput.alwaysDiscardsLateVideoFrames = true
        videoOutput.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        if session.canAddOutput(videoOutput) {
            session.addOutput(videoOutput)
        } else {
            lastError = "Cannot add video output"
            return
        }

        if session.canSetSessionPreset(.high) {
            session.sessionPreset = .high
        }
        lastError = nil
    }

    func start() {
        guard !isRunning else { return }
        // AVCaptureSession.startRunning must run off main thread.
        outputQueue.async { [weak self] in
            guard let self else { return }
            if !self.session.isRunning { self.session.startRunning() }
            Task { @MainActor in self.isRunning = true }
        }
    }

    func stop() {
        guard isRunning else { return }
        outputQueue.async { [weak self] in
            guard let self else { return }
            if self.session.isRunning { self.session.stopRunning() }
            Task { @MainActor in self.isRunning = false }
        }
    }

    /// Returns the latest resized/converted frame, copying the pool buffer so
    /// the caller gets a stable reference. Returns nil if no frame yet.
    func pullLatest() -> CVPixelBuffer? {
        bufferStore.get()
    }

    // MARK: - Private

    private func configurePool() {
        let attrs: [CFString: Any] = [
            kCVPixelBufferPixelFormatTypeKey:    kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey:              outputW,
            kCVPixelBufferHeightKey:             outputH,
            kCVPixelBufferMetalCompatibilityKey: true,
            kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary,
        ]
        var p: CVPixelBufferPool?
        CVPixelBufferPoolCreate(nil, nil, attrs as CFDictionary, &p)
        self.pool = p
    }
}

extension LiveCameraSource: AVCaptureVideoDataOutputSampleBufferDelegate {
    nonisolated func captureOutput(_ output: AVCaptureOutput,
                                   didOutput sampleBuffer: CMSampleBuffer,
                                   from connection: AVCaptureConnection) {
        guard let raw = CMSampleBufferGetImageBuffer(sampleBuffer),
              let pool = self.pool else { return }

        let srcW = CGFloat(CVPixelBufferGetWidth(raw))
        let srcH = CGFloat(CVPixelBufferGetHeight(raw))
        let side = min(srcW, srcH)
        let xOff = (srcW - side) / 2
        let yOff = (srcH - side) / 2

        let ci = CIImage(cvPixelBuffer: raw)
            .cropped(to: CGRect(x: xOff, y: yOff, width: side, height: side))
            .transformed(by: CGAffineTransform(translationX: -xOff, y: -yOff))
            .transformed(by: CGAffineTransform(scaleX: CGFloat(outputW) / side,
                                                y: CGFloat(outputH) / side))

        var dst: CVPixelBuffer?
        CVPixelBufferPoolCreatePixelBuffer(nil, pool, &dst)
        guard let dst else { return }
        ciContext.render(ci, to: dst)

        bufferStore.set(dst)
    }
}

/// Reference-typed safe holder so CVPixelBuffer (not Sendable) can cross actors
/// through synchronised access without triggering strict-concurrency errors.
final class BufferStore: @unchecked Sendable {
    private let lock = NSLock()
    private var buf: CVPixelBuffer?
    func set(_ b: CVPixelBuffer) { lock.lock(); buf = b; lock.unlock() }
    func get() -> CVPixelBuffer? { lock.lock(); defer { lock.unlock() }; return buf }
}
