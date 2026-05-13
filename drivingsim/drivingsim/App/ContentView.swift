//
//  ContentView.swift
//  drivingsim
//

import AVFoundation
import SwiftUI
import RealityKit

// MediaPipe-style hand skeleton edges (21 landmarks)
private let HAND_EDGES: [(Int, Int)] = [
    (0,1),(1,2),(2,3),(3,4),                 // thumb
    (0,5),(5,6),(6,7),(7,8),                 // index
    (0,9),(9,10),(10,11),(11,12),            // middle
    (0,13),(13,14),(14,15),(15,16),          // ring
    (0,17),(17,18),(18,19),(19,20),          // pinky
    (5,9),(9,13),(13,17)                     // palm
]

private struct LandmarkOverlay: View {
    let landmarks: [SIMD2<Float>]
    let frameSize: CGSize
    let active: Bool

    var body: some View {
        GeometryReader { geo in
            Canvas { ctx, _ in
                guard landmarks.count == 21,
                      frameSize.width > 0, frameSize.height > 0 else { return }
                let viewW  = geo.size.width
                let viewH  = geo.size.height
                let frameW = frameSize.width
                let frameH = frameSize.height
                let scale  = min(viewW / frameW, viewH / frameH)
                let drawnW = frameW * scale
                let drawnH = frameH * scale
                let offX   = (viewW - drawnW) / 2.0
                let offY   = (viewH - drawnH) / 2.0

                func pt(_ i: Int) -> CGPoint {
                    CGPoint(x: CGFloat(landmarks[i].x) * scale + offX,
                            y: CGFloat(landmarks[i].y) * scale + offY)
                }

                var path = Path()
                for (a, b) in HAND_EDGES {
                    path.move(to: pt(a))
                    path.addLine(to: pt(b))
                }
                ctx.stroke(path,
                           with: .color(active ? .green : .white.opacity(0.7)),
                           lineWidth: 2)

                for i in 0..<21 {
                    let p = pt(i)
                    let r: CGFloat = (i == 8 || i == 5) ? 4.5 : 3
                    let rect = CGRect(x: p.x - r, y: p.y - r, width: r*2, height: r*2)
                    ctx.fill(Path(ellipseIn: rect),
                             with: .color(i == 8 ? .yellow : .cyan))
                }
            }
        }
    }
}

private struct KeyCapView: View {
    let label: String
    let pressed: Bool

    var body: some View {
        Text(label)
            .font(.system(size: 14, weight: .bold, design: .monospaced))
            .foregroundColor(pressed ? .black : .white)
            .frame(width: 36, height: 36)
            .background(Color.white.opacity(pressed ? 0.9 : 0.15))
            .cornerRadius(6)
            .overlay(RoundedRectangle(cornerRadius: 6).stroke(Color.white.opacity(0.5), lineWidth: 1))
    }
}

struct ContentView: View {
    let source: CameraSource
    let esp32Profile: ESP32Profile

    @StateObject private var keyboard = KeyboardMonitor()
    @StateObject private var hand     = HandJoystick()
    // Metric model — used by .autoSeekPy + .autoMap
    @StateObject private var depth    = DepthDriver()
    // Relative model — retired (.mapDepth removed). Kept declared so any
    // residual references compile; stopped on every mode change.
    @StateObject private var depthRel = DepthDriver(modelName: "DepthAnythingV2SmallF16")
    @StateObject private var yolo     = YOLODriver()
    @StateObject private var yoloPy   = YOLOPythonDriver()
    @StateObject private var mapDrv   = MapDriver()
    @StateObject private var liveCam  = LiveCameraSource()
    @StateObject private var mjpeg    = MJPEGSource()
    @StateObject private var esp32WS  = ESP32WebSocketClient()
    @State private var mode: DrivingMode = .off
    @State private var scene = SimScene.make()
    @State private var fpv: SimFPVRenderer?
    @State private var fpvTimer: Timer?

    /// Equatable snapshot of the mode-gated WASD union — drives a single
    /// .onChange that forwards transitions to the ESP32 over WebSocket.
    private struct WASDState: Equatable { let f, b, l, r: Bool }
    private var wasd: WASDState {
        WASDState(f: anyForward, b: anyBackward, l: anyLeft, r: anyRight)
    }

    init(source: CameraSource, esp32Profile: ESP32Profile = .none) {
        self.source = source
        self.esp32Profile = esp32Profile
    }

    /// WASD union (same logic as `wasd` snapshot) packaged as the `Set<UInt16>`
    /// PoseEstimator expects for dead-reckoning on real hardware.
    private var activeKeySet: Set<UInt16> {
        var s: Set<UInt16> = []
        if anyForward  { s.insert(KeyboardMonitor.W) }
        if anyBackward { s.insert(KeyboardMonitor.S) }
        if anyLeft     { s.insert(KeyboardMonitor.A) }
        if anyRight    { s.insert(KeyboardMonitor.D) }
        return s
    }

    @ViewBuilder
    private var mainView: some View {
        if source.isLive {
            // Live camera FPV — no sim, no driving. Black background fills
            // letterbox when AVCaptureVideoPreviewLayer aspect-fits.
            ZStack {
                Color.black
                CameraPreview(session: liveCam.session,
                              mirror: false,
                              videoGravity: .resizeAspect)
            }
        } else if source.isESP32 {
            // ESP32-CAM MJPEG — decoded JPEG → CGImage rendered as Image.
            ZStack {
                Color.black
                if let cg = mjpeg.displayImage {
                    Image(decorative: cg, scale: 1.0, orientation: .up)
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                } else {
                    VStack(spacing: 6) {
                        ProgressView()
                        Text(mjpeg.connected ? "Waiting for first frame…"
                                              : "Connecting to \(ESP32Config.host)…")
                            .font(.caption)
                            .foregroundColor(.white.opacity(0.7))
                    }
                }
            }
        } else {
            RealityView { content in
                content.add(scene.root)
                scene.startUpdates(
                    keyboard: keyboard,
                    hand: hand,
                    depth: depth,
                    depthRel: depthRel,
                    yolo: yolo,
                    yoloPy: yoloPy,
                    map: mapDrv,
                    modeProvider: { mode }
                )
            } update: { _ in }
        }
    }

    var body: some View {
        mainView
        .frame(minWidth: 900, minHeight: 600)
        .onAppear {
            keyboard.start()
            // Always bring up the ESP32 WS bridge — KeyboardMonitor is live
            // even in .off mode, and the user expects W/A/S/D to drive the
            // real car regardless of which frame source is selected.
            esp32WS.start()
            if source.isLive, let uid = source.deviceUniqueID {
                // Match model's actual input size (518×392 for DepthAnythingV2),
                // not a hardcoded square. Apply ESP32 profile if user picked one.
                liveCam.setOutputSize(width:  mapDrv.inputSize.width,
                                       height: mapDrv.inputSize.height)
                liveCam.setESP32Profile(esp32Profile)
                liveCam.configure(deviceUniqueID: uid)
                liveCam.start()
            } else if source.isESP32 {
                // Same model-input size as liveCam; ESP32 stream is already
                // low-res so no further ESP32Profile downsample is applied.
                mjpeg.setOutputSize(width:  mapDrv.inputSize.width,
                                    height: mapDrv.inputSize.height)
                mjpeg.start()
                // SimScene won't run — pose must come from dead-reckoning
                // driven by commanded WASD inside the FPV pump.
                mapDrv.useTruthPose = false
            } else {
                // Sim source: build offscreen FPV renderer.
                fpv = SimFPVRenderer(obstacles: scene.obstacleSnapshot,
                                     personBillboards: scene.personBillboardSnapshot,
                                     eyeHeight: scene.eyeHeightPublic,
                                     width: depth.inputSize.width,
                                     height: depth.inputSize.height)
            }
        }
        .onChange(of: wasd) { _, w in
            esp32WS.publish(forward: w.f, backward: w.b, left: w.l, right: w.r)
        }
        .onChange(of: keyboard.inferenceCounter) { _, _ in
            // `I` key — single-shot depth inference, Explore mode only.
            if mode == .mapExplore { mapDrv.requestInferenceOnce() }
        }
        .onChange(of: keyboard.autoToggleCounter) { _, _ in
            // `O` key — flip Explore manual ↔ automatic.
            if mode == .mapExplore { mapDrv.toggleAutoExplore() }
        }
        .onDisappear {
            stopAll()
        }
        .onChange(of: mode) { _, newMode in updateMode(newMode) }
        .overlay {
            if mode.needsYOLO {
                BoundingBoxOverlay(driver: yolo, aspectFit: false)
            } else if mode.needsYOLOPy {
                BoundingBoxOverlay(driverPy: yoloPy, aspectFit: false)
            }
        }
        .overlay(alignment: .bottomLeading) {
            VStack(alignment: .leading, spacing: 8) {
                Text("W/S accel-brake   A/D steer")
                    .font(.caption)
                    .foregroundColor(.white.opacity(0.85))
                HStack(spacing: 6) {
                    Text("Mode")
                        .font(.caption)
                        .foregroundColor(.white.opacity(0.7))
                    Picker("", selection: $mode) {
                        ForEach(DrivingMode.allCases) { m in
                            Text(m.label).tag(m)
                        }
                    }
                    .pickerStyle(.menu)
                    .labelsHidden()
                    .tint(.white)
                    .frame(minWidth: 140)
                }
                HStack(spacing: 6) {
                    Circle()
                        .fill(esp32WS.connected ? Color.green : Color.red)
                        .frame(width: 8, height: 8)
                    Text("ESP32 \(esp32WS.connected ? "linked" : "offline") · \(ESP32Config.host)")
                        .font(.system(size: 11, design: .monospaced))
                        .foregroundColor(.white.opacity(0.9))
                    if source.isESP32 {
                        Text("· cam \(String(format: "%.0f", mjpeg.fps)) fps")
                            .font(.system(size: 11, design: .monospaced))
                            .foregroundColor(.white.opacity(0.7))
                    }
                }
            }
            .foregroundColor(.white)
            .padding(10)
            .background(.black.opacity(0.55))
            .cornerRadius(8)
            .padding(12)
        }
        .overlay(alignment: .topTrailing) {
            VStack(alignment: .trailing, spacing: 8) {
                if mode.needsHand {
                    VStack(spacing: 4) {
                        ZStack {
                            CameraPreview(session: hand.captureSession)
                            LandmarkOverlay(landmarks: hand.landmarks,
                                            frameSize: hand.frameSize,
                                            active: hand.lastActive)
                        }
                        .frame(width: 240, height: 180)
                        .cornerRadius(8)
                        .overlay(RoundedRectangle(cornerRadius: 8)
                            .stroke(hand.lastActive ? Color.green : Color.white.opacity(0.4),
                                    lineWidth: 2))
                        Text("Hand")
                            .font(.caption2)
                            .foregroundColor(.white.opacity(0.7))
                    }
                }
                if mode.needsDepth && mode != .autoSeekPy {
                    VStack(spacing: 4) {
                        DepthPreview(driver: depth)
                            .frame(width: 240, height: 240)
                            .cornerRadius(8)
                            .overlay(RoundedRectangle(cornerRadius: 8)
                                .stroke(Color.cyan.opacity(0.5), lineWidth: 2))
                        Text("Depth (\(mode.label))")
                            .font(.caption2)
                            .foregroundColor(.white.opacity(0.7))
                    }
                }
                if mode.needsYOLO {
                    VStack(spacing: 4) {
                        YOLOPreviewPanel(driver: yolo)
                            .frame(width: 240, height: 240)
                            .cornerRadius(8)
                            .overlay(RoundedRectangle(cornerRadius: 8)
                                .stroke(Color.orange.opacity(0.6), lineWidth: 2))
                        Text("YOLO/CoreML (\(yolo.seekState == .seek ? "SEEK" : "ROAM"))")
                            .font(.caption2)
                            .foregroundColor(.white.opacity(0.7))
                    }
                }
                if mode == .autoSeekPy {
                    // Relative DepthAnythingV2 preview + zone grid. YOLO disabled.
                    VStack(spacing: 4) {
                        DepthPreview(driver: depthRel)
                            .frame(width: 240, height: 240)
                            .cornerRadius(8)
                            .overlay(RoundedRectangle(cornerRadius: 8)
                                .stroke(Color.cyan.opacity(0.6), lineWidth: 2))
                        Text("Depth (SeekPy)")
                            .font(.caption2)
                            .foregroundColor(.white.opacity(0.7))
                    }
                }
                if mode.needsMap {
                    // Unified Map mode: YOLO cam (small) + occupancy map (large)
                    VStack(spacing: 6) {
                        // YOLO camera feed with bbox — small thumbnail
                        ZStack {
                            YOLOPyPreviewPanel(driver: yoloPy)
                            // ROAM/SEEK badge bottom-left of thumbnail
                            VStack {
                                Spacer()
                                HStack {
                                    HStack(spacing: 5) {
                                        Circle()
                                            .fill(yoloPy.seekState == .seek ? Color.red : Color.green)
                                            .frame(width: 7, height: 7)
                                        Text(yoloPy.seekState == .seek ? "SEEK" : "ROAM")
                                            .font(.system(size: 10, weight: .bold, design: .monospaced))
                                            .foregroundColor(.white)
                                        if !yoloPy.pythonReady {
                                            Text("(starting…)")
                                                .font(.system(size: 9))
                                                .foregroundColor(.white.opacity(0.6))
                                        }
                                    }
                                    .padding(.horizontal, 6).padding(.vertical, 3)
                                    .background(.black.opacity(0.6))
                                    .cornerRadius(5)
                                    .padding(5)
                                    Spacer()
                                }
                            }
                        }
                        .frame(width: 240, height: 135)
                        .cornerRadius(8)
                        .overlay(RoundedRectangle(cornerRadius: 8)
                            .stroke(yoloPy.seekState == .seek ? Color.red.opacity(0.8) : Color.purple.opacity(0.5),
                                    lineWidth: 2))

                        // DepthAnythingV2 output — metric (Map) or relative (MapDepth).
                        MapDepthPanel(driver: depth)
                            .frame(width: 240, height: 135)
                            .cornerRadius(8)
                            .overlay(RoundedRectangle(cornerRadius: 8)
                                .stroke(Color.cyan.opacity(0.6), lineWidth: 2))
                        Text("DepthAnythingV2 · metric")
                            .font(.caption2)
                            .foregroundColor(.white.opacity(0.6))

                        // Zoomed occupancy map — 6m × 6m window around robot
                        MapPreviewPanel(driver: mapDrv)
                            .frame(width: 320, height: 320)
                            .cornerRadius(8)
                            .overlay(RoundedRectangle(cornerRadius: 8)
                                .stroke(Color.green.opacity(0.6), lineWidth: 2))

                        Text("Map · 6m × 6m around robot")
                            .font(.caption2)
                            .foregroundColor(.white.opacity(0.7))
                            .frame(width: 320)

                        MapLegend()
                            .frame(width: 320)
                    }
                }
            }
            .padding(12)
        }
        .overlay(alignment: .bottomTrailing) {
            VStack(spacing: 4) {
                HStack {
                    KeyCapView(label: "W", pressed: anyForward)
                }
                HStack(spacing: 4) {
                    KeyCapView(label: "A", pressed: anyLeft)
                    KeyCapView(label: "S", pressed: anyBackward)
                    KeyCapView(label: "D", pressed: anyRight)
                }
                if mode == .mapExplore {
                    HStack(spacing: 4) {
                        KeyCapView(label: "I", pressed: keyboard.pressed.contains(KeyboardMonitor.I))
                        Text("scan")
                            .font(.system(size: 11, design: .monospaced))
                            .foregroundColor(.white.opacity(0.8))
                    }
                    HStack(spacing: 4) {
                        KeyCapView(label: "O", pressed: keyboard.pressed.contains(KeyboardMonitor.O))
                        Text(mapDrv.autoActive ? "auto · ON" : "auto · OFF")
                            .font(.system(size: 11, weight: .bold, design: .monospaced))
                            .foregroundColor(mapDrv.autoActive ? .green : .white.opacity(0.7))
                    }
                }
            }
            .padding(12)
        }
    }

    private var activeYoloSeek: SeekState {
        if mode == .autoSeek { return yolo.seekState }
        if mode == .autoSeekPy { return yoloPy.seekState }
        return .roam
    }
    private var activeYoloBest: PersonDetection? {
        if mode == .autoSeek { return yolo.bestDetection }
        if mode == .autoSeekPy { return yoloPy.bestDetection }
        return nil
    }
    private var seekActive: Bool { mode.anyYOLO && activeYoloSeek == .seek }
    private var roamActive: Bool { mode.anyYOLO && activeYoloSeek == .roam }

    /// Depth source for UI WASD reads: .autoSeekPy uses depthRel; others use depth.
    private var activeDepth: DepthDriver { mode == .autoSeekPy ? depthRel : depth }

    private var anyForward:  Bool {
        keyboard.forward
        || (mode.needsHand  && hand.forward)
        || (mode.needsDepth && activeDepth.forward)
        || (mode.needsMap   && mapDrv.forward)
        || (seekActive && (activeYoloBest?.h ?? 1) < 0.55)
    }
    private var anyBackward: Bool {
        keyboard.backward
        || (mode.needsHand  && hand.backward)
        || (mode.needsDepth && activeDepth.backward)
        || (mode.needsMap   && mapDrv.backward)
    }
    private var anyLeft:     Bool {
        keyboard.left
        || (mode.needsHand  && hand.left)
        || (mode == .automated && activeDepth.left)
        || (mode == .autoSeekPy && activeDepth.left)
        || (roamActive && activeDepth.left)
        || (mode.needsMap   && mapDrv.left)
        || (seekActive && (activeYoloBest?.cx ?? 0.5) < 0.45)
    }
    private var anyRight:    Bool {
        keyboard.right
        || (mode.needsHand  && hand.right)
        || (mode == .automated && activeDepth.right)
        || (mode == .autoSeekPy && activeDepth.right)
        || (roamActive && activeDepth.right)
        || (mode.needsMap   && mapDrv.right)
        || (seekActive && (activeYoloBest?.cx ?? 0.5) > 0.55)
    }

    private func updateMode(_ newMode: DrivingMode) {
        if newMode.needsHand {
            hand.start()
        } else {
            hand.stop()
        }
        // Two depth drivers:
        //   `depth`    — metric model, used by .autoMap/.mapMetric/.mapExplore + .assisted/.automated
        //   `depthRel` — relative DepthAnythingV2 SmallF16, used by .autoSeekPy
        let wantsRel    = (newMode == .autoSeekPy)
        let wantsMetric = (newMode.needsDepth && !wantsRel)
                           || newMode == .autoMap
                           || newMode == .mapMetric
                           || newMode == .mapExplore
        if wantsMetric { depth.start() }    else { depth.stop() }
        if wantsRel    { depthRel.start() } else { depthRel.stop() }
        // Publish raw float meters whenever DepthProjector consumes them.
        depth.setPublishMeters(newMode.usesMetricProjection)
        // Tell MapDriver which behaviour to enter on next start (sweep vs autoSeekPy passive).
        mapDrv.exploreMode = (newMode == .mapExplore)

        // Source frame size = active depth model's expected input.
        // autoSeekPy → depthRel (DepthAnythingV2SmallF16, typically 518×392).
        // others    → depth (metric, 518×518).
        let activeInputSize = (newMode == .autoSeekPy) ? depthRel.inputSize : depth.inputSize
        if !source.isExternalFrameSource {
            let (w, h) = activeInputSize
            let needRebuild: Bool = {
                guard let cur = fpv?.outputSize else { return true }
                return cur.width != w || cur.height != h
            }()
            if needRebuild {
                fpv = SimFPVRenderer(obstacles: scene.obstacleSnapshot,
                                     personBillboards: scene.personBillboardSnapshot,
                                     eyeHeight: scene.eyeHeightPublic,
                                     width: w, height: h)
            }
        } else if source.isESP32 {
            let (w, h) = activeInputSize
            mjpeg.setOutputSize(width: w, height: h)
        } else if source.isLive {
            let (w, h) = activeInputSize
            liveCam.setOutputSize(width: w, height: h)
        }
        if newMode.needsYOLO {
            yolo.start()
        } else {
            yolo.stop()
        }
        if newMode.needsYOLOPy {
            yoloPy.start()
        } else {
            yoloPy.stop()
        }
        if newMode.needsMap {
            mapDrv.start()
        } else {
            mapDrv.stop()
        }
        if newMode.needsDepth || newMode.anyYOLO || newMode.needsMap {
            startFPVTimer()
        } else {
            stopFPVTimer()
        }
    }

    private func startFPVTimer() {
        guard fpvTimer == nil else { return }
        if !source.isExternalFrameSource && fpv == nil { return }
        let t = Timer(timeInterval: 1.0 / 30.0, repeats: true) { _ in
            Task { @MainActor in
                let frameBuf: CVPixelBuffer? = {
                    if source.isLive {
                        return liveCam.pullLatest()
                    } else if source.isESP32 {
                        return mjpeg.pullLatest()
                    } else {
                        guard let renderer = fpv else { return nil }
                        return renderer.render(carPosition: scene.carWorldPosition,
                                                yaw: scene.carYaw,
                                                eyeHeight: scene.eyeHeightPublic)
                    }
                }()
                guard let buf = frameBuf else { return }
                // .mapExplore gates inference on the MapDriver's scan dwell window
                // so CoreML only runs during sweep stops (~12 inferences/cycle
                // instead of ~30/s). Other modes keep per-frame inference.
                let runMetric: Bool = {
                    if mode == .mapExplore { return mapDrv.wantsInference }
                    return (mode.needsDepth && mode != .autoSeekPy)
                        || mode == .autoMap || mode == .mapMetric
                }()
                if runMetric {
                    // Pose at submit time travels with the MetricFrame so
                    // consumeMetricFrame projects with the rendered yaw, not
                    // whatever yaw exists when inference completes.
                    let pSubmit = mapDrv.poseEstimator
                    depth.submit(buf, posePos: pSubmit.pos, poseYaw: pSubmit.yaw)
                }
                // autoSeekPy: relative DepthAnythingV2 (no metric projection).
                if mode == .autoSeekPy { depthRel.submit(buf) }
                if mode.needsYOLO    { yolo.submit(buf) }
                if mode.needsYOLOPy  { yoloPy.submit(buf) }
                // Sync YOLOPy seek state into MapDriver target tracker (map modes).
                if mode.needsMap {
                    // ESP32 path: drive pose dead-reckoning from commanded WASD
                    // so the occupancy grid + A* have a moving pose to project
                    // depth against. dt = 1/30 matches the timer interval.
                    if source.isESP32 {
                        mapDrv.tickDeadReckoned(keys: activeKeySet, dt: 1.0 / 30.0)
                    }
                    let pose = mapDrv.poseEstimator
                    if yoloPy.seekState == .seek, let det = yoloPy.bestDetection {
                        mapDrv.updateSeekTarget(
                            detectionCx: det.cx, detectionCy: det.cy,
                            detectionW:  det.w,  detectionH: det.h,
                            posePos: pose.pos, poseYaw: pose.yaw)
                    } else {
                        mapDrv.clearSeekBox()
                    }
                    // .mapMetric / .mapExplore: project ≤5m metric depth into occupancy grid.
                    if mode.usesMetricProjection, let snap = depth.latestMetric {
                        mapDrv.consumeMetricFrame(snap, posePos: pose.pos, poseYaw: pose.yaw)
                    }
                }
            }
        }
        RunLoop.main.add(t, forMode: .common)
        fpvTimer = t
    }

    private func stopFPVTimer() {
        fpvTimer?.invalidate()
        fpvTimer = nil
    }

    private func stopAll() {
        keyboard.stop()
        hand.stop()
        depth.stop()
        depthRel.stop()
        yolo.stop()
        yoloPy.stop()
        mapDrv.stop()
        liveCam.stop()
        mjpeg.stop()
        esp32WS.stop()
        scene.stopUpdates()
        stopFPVTimer()
    }
}

// Small wrapper view to show YOLO input frame + bbox overlay together.
private struct YOLOPreviewPanel: View {
    @ObservedObject var driver: YOLODriver
    var body: some View {
        ZStack {
            Color.black
            if let img = driver.previewImage {
                Image(decorative: img, scale: 1.0, orientation: .up)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
            }
            BoundingBoxOverlay(driver: driver, aspectFit: true)
        }
    }
}

private struct YOLOPyPreviewPanel: View {
    @ObservedObject var driver: YOLOPythonDriver
    var body: some View {
        ZStack {
            Color.black
            if let img = driver.previewImage {
                Image(decorative: img, scale: 1.0, orientation: .up)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
            }
            BoundingBoxOverlay(driverPy: driver, aspectFit: true)
        }
    }
}

private struct MapPreviewPanel: View {
    @ObservedObject var driver: MapDriver
    var body: some View {
        ZStack {
            Color(white: 0.5)   // unknown-gray background before first render
            if let img = driver.occupancyImage {
                Image(decorative: img, scale: 1.0, orientation: .up)
                    .interpolation(.none)   // keep pixel-art sharpness for grid cells
                    .resizable()
                    .aspectRatio(contentMode: .fit)
            } else {
                Text("Building map…")
                    .font(.caption2)
                    .foregroundColor(.white.opacity(0.8))
            }
        }
    }
}

/// Two-column legend below the map: colour swatch + label.
private struct MapLegend: View {
    private struct Item { let color: Color; let label: String }
    private let items: [Item] = [
        Item(color: Color(red: 0.90, green: 0.12, blue: 0.12), label: "Robot"),
        Item(color: Color(red: 1.00, green: 0.63, blue: 0.08), label: "Person"),
        Item(color: Color(red: 0.20, green: 0.78, blue: 0.20), label: "Planned path"),
        Item(color: Color(red: 0.08, green: 0.08, blue: 0.08), label: "Obstacle"),
        Item(color: Color(white: 0.94),                         label: "Free space"),
        Item(color: Color(white: 0.55),                         label: "Unknown"),
        Item(color: Color(white: 0.35),                         label: "Out of grid"),
    ]
    var body: some View {
        let columns = [GridItem(.flexible(), alignment: .leading),
                       GridItem(.flexible(), alignment: .leading)]
        LazyVGrid(columns: columns, alignment: .leading, spacing: 4) {
            ForEach(0..<items.count, id: \.self) { i in
                HStack(spacing: 6) {
                    RoundedRectangle(cornerRadius: 2)
                        .fill(items[i].color)
                        .frame(width: 12, height: 12)
                        .overlay(RoundedRectangle(cornerRadius: 2)
                            .stroke(Color.white.opacity(0.25), lineWidth: 0.5))
                    Text(items[i].label)
                        .font(.system(size: 10))
                        .foregroundColor(.white.opacity(0.85))
                }
            }
        }
        .padding(8)
        .background(.black.opacity(0.4))
        .cornerRadius(6)
    }
}

/// Colourised depth output from DepthAnythingV2 (Map mode's source for ray-casting).
private struct MapDepthPanel: View {
    @ObservedObject var driver: DepthDriver
    var body: some View {
        ZStack {
            Color.black
            if let img = driver.depthImage {
                Image(decorative: img, scale: 1.0, orientation: .up)
                    .resizable()
                    .interpolation(.low)
                    .aspectRatio(contentMode: .fit)
            } else {
                Text("Depth loading…")
                    .font(.caption2)
                    .foregroundColor(.white.opacity(0.6))
            }
        }
    }
}
