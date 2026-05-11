//
//  ContentView.swift
//  drivingsim
//

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
    @StateObject private var keyboard = KeyboardMonitor()
    @StateObject private var hand     = HandJoystick()
    @StateObject private var depth    = DepthDriver()
    @StateObject private var yolo     = YOLODriver()
    @StateObject private var yoloPy   = YOLOPythonDriver()
    @StateObject private var mapDrv   = MapDriver()
    @State private var mode: DrivingMode = .off
    @State private var scene = SimScene.make()
    @State private var fpv: SimFPVRenderer?
    @State private var fpvTimer: Timer?

    var body: some View {
        RealityView { content in
            content.add(scene.root)
            scene.startUpdates(
                keyboard: keyboard,
                hand: hand,
                depth: depth,
                yolo: yolo,
                yoloPy: yoloPy,
                map: mapDrv,
                modeProvider: { mode }
            )
        } update: { _ in }
        .frame(minWidth: 900, minHeight: 600)
        .onAppear {
            keyboard.start()
            // FPV renderer reads the obstacles laid down by SimScene.setup()
            fpv = SimFPVRenderer(obstacles: scene.obstacleSnapshot,
                                 personBillboards: scene.personBillboardSnapshot,
                                 eyeHeight: scene.eyeHeightPublic,
                                 width: depth.inputSize.width,
                                 height: depth.inputSize.height)
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
            VStack(alignment: .leading, spacing: 6) {
                Text("W/S accel-brake   A/D steer")
                    .font(.caption)
                Picker("Mode", selection: $mode) {
                    ForEach(DrivingMode.allCases) { m in
                        Text(m.label).tag(m)
                    }
                }
                .pickerStyle(.segmented)
                .frame(width: 280)
            }
            .foregroundColor(.white)
            .padding(8)
            .background(.black.opacity(0.45))
            .cornerRadius(6)
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
                if mode.needsDepth {
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
                if mode.needsYOLOPy {
                    VStack(spacing: 4) {
                        YOLOPyPreviewPanel(driver: yoloPy)
                            .frame(width: 240, height: 240)
                            .cornerRadius(8)
                            .overlay(RoundedRectangle(cornerRadius: 8)
                                .stroke(Color.purple.opacity(0.6), lineWidth: 2))
                        Text("YOLO/Python (\(yoloPy.seekState == .seek ? "SEEK" : "ROAM"))\(yoloPy.pythonReady ? "" : " starting…")")
                            .font(.caption2)
                            .foregroundColor(.white.opacity(0.7))
                    }
                }
                if mode.needsMap {
                    VStack(spacing: 4) {
                        MapPreviewPanel(driver: mapDrv)
                            .frame(width: 240, height: 160)
                            .cornerRadius(8)
                            .overlay(RoundedRectangle(cornerRadius: 8)
                                .stroke(Color.green.opacity(0.6), lineWidth: 2))
                        Text("Map (free=light · occ=dark · path=green · frontier=blue)")
                            .font(.caption2)
                            .foregroundColor(.white.opacity(0.7))
                            .multilineTextAlignment(.center)
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

    private var anyForward:  Bool {
        keyboard.forward
        || (mode.needsHand  && hand.forward)
        || (mode.needsDepth && depth.forward)
        || (seekActive && (activeYoloBest?.h ?? 1) < 0.55)
    }
    private var anyBackward: Bool {
        keyboard.backward
        || (mode.needsHand  && hand.backward)
        || (mode.needsDepth && depth.backward)
    }
    private var anyLeft:     Bool {
        keyboard.left
        || (mode.needsHand  && hand.left)
        || (mode == .automated && depth.left)
        || (roamActive && depth.left)
        || (seekActive && (activeYoloBest?.cx ?? 0.5) < 0.45)
    }
    private var anyRight:    Bool {
        keyboard.right
        || (mode.needsHand  && hand.right)
        || (mode == .automated && depth.right)
        || (roamActive && depth.right)
        || (seekActive && (activeYoloBest?.cx ?? 0.5) > 0.55)
    }

    private func updateMode(_ newMode: DrivingMode) {
        if newMode.needsHand {
            hand.start()
        } else {
            hand.stop()
        }
        if newMode.needsDepth {
            depth.start()
        } else {
            depth.stop()
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
        guard fpvTimer == nil, fpv != nil else { return }
        let t = Timer(timeInterval: 1.0 / 30.0, repeats: true) { _ in
            Task { @MainActor in
                guard let renderer = fpv else { return }
                let pos = scene.carWorldPosition
                let yaw = scene.carYaw
                if let buf = renderer.render(carPosition: pos, yaw: yaw,
                                              eyeHeight: scene.eyeHeightPublic) {
                    if mode.needsDepth   { depth.submit(buf) }
                    if mode.needsYOLO    { yolo.submit(buf) }
                    if mode.needsYOLOPy  { yoloPy.submit(buf) }
                    if mode.needsMap     { mapDrv.submit(buf) }
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
        yolo.stop()
        yoloPy.stop()
        mapDrv.stop()
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
