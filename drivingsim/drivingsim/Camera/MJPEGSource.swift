//
//  MJPEGSource.swift
//  drivingsim
//
//  Pulls http://<host>/stream from the ESP32-CAM (multipart MJPEG with
//  boundary=frame) and publishes the latest decoded frame as a 518×392
//  BGRA CVPixelBuffer — the same shape LiveCameraSource produces, so
//  ContentView's FPV timer + DepthDriver/YOLO submission paths are
//  source-agnostic.
//
//  ESP32 firmware (esp32/src/main.cpp:101 handleStream) emits:
//    --frame\r\nContent-Type: image/jpeg\r\nContent-Length: N\r\n\r\n
//    <N bytes JPEG>\r\n
//  …repeated. The parser scans for "--frame", reads the Content-Length
//  header, then slices N bytes of JPEG and decodes via ImageIO.
//

@preconcurrency import CoreVideo
import Combine
import CoreImage
import Foundation
import ImageIO

@MainActor
final class MJPEGSource: NSObject, ObservableObject {
    @Published private(set) var connected: Bool = false
    @Published private(set) var fps: Double = 0
    @Published private(set) var lastError: String?
    /// Latest decoded JPEG as a CGImage — for SwiftUI preview rendering.
    /// Inference pulls from `bufferStore` via `pullLatest()` instead.
    @Published private(set) var displayImage: CGImage?

    nonisolated(unsafe) private var outputW: Int = 518
    nonisolated(unsafe) private var outputH: Int = 392

    nonisolated(unsafe) private let bufferStore = BufferStore()
    nonisolated private let ciContext = CIContext(options: [.useSoftwareRenderer: false])
    nonisolated(unsafe) private var pool: CVPixelBufferPool?

    private var session: URLSession?
    private var task: URLSessionDataTask?
    private var running: Bool = false
    private var reconnecting: Bool = false   // re-entry guard
    private var reconnectAttempt: Int = 0
    private var stallWatchdog: Timer?
    private let stallTimeoutSec: TimeInterval = 5.0

    // Parser state (mutated only on the delegate queue, not Sendable-shared).
    nonisolated(unsafe) private var buffer = Data()
    nonisolated(unsafe) private var frameCount: Int = 0
    nonisolated(unsafe) private var lastFpsTick: Date = .init()
    nonisolated(unsafe) private var lastFrameTime: Date = .init()

    override init() {
        super.init()
        configurePool()
    }

    func setOutputSize(width: Int, height: Int) {
        guard width != outputW || height != outputH else { return }
        outputW = width
        outputH = height
        configurePool()
    }

    func start() {
        guard !running else { return }
        running = true
        connect()
    }

    func stop() {
        guard running else { return }
        running = false
        stopStallWatchdog()
        task?.cancel()
        task = nil
        session?.invalidateAndCancel()
        session = nil
        connected = false
    }

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
        pool = p
    }

    private func connect() {
        buffer.removeAll(keepingCapacity: true)
        let config = URLSessionConfiguration.ephemeral
        config.timeoutIntervalForRequest = 60
        config.timeoutIntervalForResource = .infinity
        let s = URLSession(configuration: config, delegate: self, delegateQueue: nil)
        session = s
        let req = URLRequest(url: ESP32Config.streamURL)
        print("[MJPEG] connecting → \(ESP32Config.streamURL)")
        let t = s.dataTask(with: req)
        task = t
        lastFrameTime = Date()
        startStallWatchdog()
        t.resume()
    }

    private func startStallWatchdog() {
        stallWatchdog?.invalidate()
        let timer = Timer(timeInterval: 1.0, repeats: true) { [weak self] _ in
            guard let self else { return }
            Task { @MainActor in
                let gap = Date().timeIntervalSince(self.lastFrameTime)
                if gap > self.stallTimeoutSec && self.running {
                    print(String(format: "[MJPEG] stall %.1fs — forcing reconnect", gap))
                    self.handleDisconnect(reason: "stall")
                }
            }
        }
        RunLoop.main.add(timer, forMode: .common)
        stallWatchdog = timer
    }

    private func stopStallWatchdog() {
        stallWatchdog?.invalidate()
        stallWatchdog = nil
    }

    private func handleDisconnect(reason: String) {
        // Re-entry guard — stall watchdog + URLSession completion both call
        // this. Only the first call actually schedules a reconnect.
        if reconnecting { return }
        reconnecting = true
        connected = false
        lastError = reason
        stopStallWatchdog()
        task?.cancel()
        task = nil
        session?.invalidateAndCancel()
        session = nil
        guard running else { reconnecting = false; return }
        reconnectAttempt = min(reconnectAttempt + 1, 4)
        let delay = min(0.5 * pow(2.0, Double(reconnectAttempt - 1)), 5.0)
        print(String(format: "[MJPEG] reconnect in %.1fs (attempt %d, reason=%@)", delay, reconnectAttempt, reason))
        Task { @MainActor [weak self] in
            try? await Task.sleep(nanoseconds: UInt64(delay * 1_000_000_000))
            guard let self, self.running else { return }
            self.reconnecting = false
            self.connect()
        }
    }

    // macOS URLSession natively parses multipart/x-mixed-replace: it strips
    // the --frame boundaries and fires `didReceive response` once per part.
    // Between two consecutive `didReceive response` events the body bytes
    // arriving via `didReceive data` are exactly one JPEG payload.
    //
    // So: accumulate bytes; on each new response (or stream end), the
    // accumulated buffer = one complete JPEG → decode + reset.
    fileprivate nonisolated func consumeBytes(_ data: Data) {
        buffer.append(data)
    }

    fileprivate nonisolated func flushFrame() {
        guard !buffer.isEmpty else { return }
        let jpeg = buffer
        buffer.removeAll(keepingCapacity: true)
        decodeAndStore(jpeg)
    }

    private nonisolated func decodeAndStore(_ jpeg: Data) {
        guard let pool = self.pool,
              let src = CGImageSourceCreateWithData(jpeg as CFData, nil),
              let cg  = CGImageSourceCreateImageAtIndex(src, 0, nil) else {
            print("[MJPEG] decode failed (\(jpeg.count)B)")
            return
        }
        // First-frame-only log; per-frame log is the fps line below.
        if frameCount == 0 { print("[MJPEG] decoded \(cg.width)x\(cg.height) (first frame this run)") }

        let srcW = CGFloat(cg.width)
        let srcH = CGFloat(cg.height)
        guard srcW > 0, srcH > 0 else { return }

        let dstAR = CGFloat(outputW) / CGFloat(outputH)
        let cropW: CGFloat
        let cropH: CGFloat
        if srcW / srcH > dstAR {
            cropH = srcH;     cropW = srcH * dstAR
        } else {
            cropW = srcW;     cropH = srcW / dstAR
        }
        let xOff = (srcW - cropW) / 2
        let yOff = (srcH - cropH) / 2

        var ci = CIImage(cgImage: cg)
            .cropped(to: CGRect(x: xOff, y: yOff, width: cropW, height: cropH))
            .transformed(by: CGAffineTransform(translationX: -xOff, y: -yOff))
            .transformed(by: CGAffineTransform(scaleX: CGFloat(outputW) / cropW,
                                                y: CGFloat(outputH) / cropH))

        var dst: CVPixelBuffer?
        CVPixelBufferPoolCreatePixelBuffer(nil, pool, &dst)
        guard let dst else { return }
        ciContext.render(ci, to: dst)
        bufferStore.set(dst)

        frameCount += 1
        let now = Date()
        lastFrameTime = now
        let dt = now.timeIntervalSince(lastFpsTick)
        let fpsUpdate: Double? = dt >= 1.0 ? (Double(frameCount) / dt) : nil
        if let f = fpsUpdate {
            print(String(format: "[MJPEG] %.1f fps  (%d frames in %.2fs)", f, frameCount, dt))
            frameCount = 0; lastFpsTick = now
        }

        // CVPixelBuffer is not Sendable — keep it in the lock-protected
        // bufferStore (pullLatest reads from there). Only ship the CGImage
        // (immutable, Sendable) and fps scalar across the actor boundary.
        Task { @MainActor [weak self, cg] in
            guard let self else { return }
            self.displayImage = cg
            if let f = fpsUpdate { self.fps = f }
        }
    }
}

extension MJPEGSource: URLSessionDataDelegate {
    nonisolated func urlSession(_ session: URLSession,
                                dataTask: URLSessionDataTask,
                                didReceive response: URLResponse,
                                completionHandler: @escaping (URLSession.ResponseDisposition) -> Void) {
        let status = (response as? HTTPURLResponse)?.statusCode ?? -1
        // Each new response = end of the previous multipart part. Decode
        // whatever bytes accumulated since the last response — that's one
        // complete JPEG frame.
        flushFrame()
        let mime = (response as? HTTPURLResponse)?.value(forHTTPHeaderField: "Content-Type") ?? ""
        // Log only the boundary header (first response), not every JPEG part.
        if mime.contains("multipart") { print("[MJPEG] response HTTP \(status) ct=\(mime)") }
        Task { @MainActor in
            self.connected = true
            self.lastError = nil
            self.reconnectAttempt = 0
        }
        completionHandler(.allow)
    }

    nonisolated func urlSession(_ session: URLSession,
                                dataTask: URLSessionDataTask,
                                didReceive data: Data) {
        consumeBytes(data)
    }

    nonisolated func urlSession(_ session: URLSession,
                                task: URLSessionTask,
                                didCompleteWithError error: Error?) {
        flushFrame()
        let reason = error?.localizedDescription ?? "stream closed"
        print("[MJPEG] complete: \(reason)")
        Task { @MainActor in
            self.handleDisconnect(reason: reason)
        }
    }
}
