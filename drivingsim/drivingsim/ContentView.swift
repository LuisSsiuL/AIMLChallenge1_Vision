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
                // Replicate AVLayerVideoGravity.resizeAspect (letterbox) so this
                // overlay lives in the exact same rect as the displayed video.
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

                // edges
                var path = Path()
                for (a, b) in HAND_EDGES {
                    path.move(to: pt(a))
                    path.addLine(to: pt(b))
                }
                ctx.stroke(path,
                           with: .color(active ? .green : .white.opacity(0.7)),
                           lineWidth: 2)

                // joints
                for i in 0..<21 {
                    let p = pt(i)
                    let r: CGFloat = (i == 8 || i == 5) ? 4.5 : 3   // highlight index MCP/tip
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
    @State private var handEnabled    = false
    @State private var scene = SimScene.make()

    var body: some View {
        RealityView { content in
            content.add(scene.root)
            scene.startUpdates(keyboard: keyboard, hand: hand)
        } update: { _ in }
        .frame(minWidth: 900, minHeight: 600)
        .onAppear { keyboard.start() }
        .onDisappear {
            keyboard.stop()
            hand.stop()
            scene.stopUpdates()
        }
        .overlay(alignment: .bottomLeading) {
            VStack(alignment: .leading, spacing: 6) {
                Text("W/S accel-brake   A/D steer")
                    .font(.caption)
                Toggle("Hand Control", isOn: $handEnabled)
                    .toggleStyle(.switch)
                    .controlSize(.small)
                    .onChange(of: handEnabled) { _, on in
                        if on { hand.start() } else { hand.stop() }
                    }
            }
            .foregroundColor(.white)
            .padding(8)
            .background(.black.opacity(0.45))
            .cornerRadius(6)
            .padding(12)
        }
        .overlay(alignment: .topTrailing) {
            if handEnabled {
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
                    Text("Webcam")
                        .font(.caption2)
                        .foregroundColor(.white.opacity(0.7))
                }
                .padding(12)
            }
        }
        .overlay(alignment: .bottomTrailing) {
            VStack(spacing: 4) {
                HStack {
                    KeyCapView(label: "W", pressed: keyboard.forward  || hand.forward)
                }
                HStack(spacing: 4) {
                    KeyCapView(label: "A", pressed: keyboard.left     || hand.left)
                    KeyCapView(label: "S", pressed: keyboard.backward || hand.backward)
                    KeyCapView(label: "D", pressed: keyboard.right    || hand.right)
                }
                if handEnabled {
                    Text(hand.lastActive ? "active" : "deadzone")
                        .font(.caption2)
                        .foregroundColor(hand.lastActive ? .green : .gray)
                }
            }
            .padding(12)
        }
    }
}
