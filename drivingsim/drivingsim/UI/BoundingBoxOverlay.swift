//
//  BoundingBoxOverlay.swift
//  drivingsim
//
//  Draws YOLO detection boxes (normalized [0,1]) over a SwiftUI view, plus a
//  ROAM/SEEK state badge. Works with either YOLODriver (CoreML) or
//  YOLOPythonDriver via type-erased wrapper.
//

import SwiftUI

struct BoundingBoxOverlay: View {
    @ObservedObject private var coreml:  YOLODriver
    @ObservedObject private var python:  YOLOPythonDriver
    private let usePython: Bool
    var aspectFit: Bool = false

    // Shared dummy instances so SwiftUI's @ObservedObject always has a backing
    // object even when only one source is active.
    @MainActor private static let dummyCoreML  = YOLODriver()
    @MainActor private static let dummyPython  = YOLOPythonDriver()

    init(driver: YOLODriver, aspectFit: Bool = false) {
        self.coreml = driver
        self.python = Self.dummyPython
        self.usePython = false
        self.aspectFit = aspectFit
    }
    init(driverPy: YOLOPythonDriver, aspectFit: Bool = false) {
        self.coreml = Self.dummyCoreML
        self.python = driverPy
        self.usePython = true
        self.aspectFit = aspectFit
    }

    private var detections: [PersonDetection] {
        usePython ? python.detections : coreml.detections
    }
    private var inputFrameSize: CGSize {
        usePython ? python.inputFrameSize : coreml.inputFrameSize
    }
    private var seekState: SeekState {
        usePython ? python.seekState : coreml.seekState
    }
    private var backendLabel: String { usePython ? "PY" : "CM" }

    var body: some View {
        GeometryReader { geo in
            ZStack(alignment: .topLeading) {
                Canvas { ctx, _ in
                    let viewW = geo.size.width
                    let viewH = geo.size.height
                    let frameW = max(1, inputFrameSize.width)
                    let frameH = max(1, inputFrameSize.height)
                    let scale: CGFloat
                    let offX: CGFloat
                    let offY: CGFloat
                    if aspectFit {
                        let s = min(viewW / frameW, viewH / frameH)
                        scale = s
                        offX = (viewW - frameW * s) / 2
                        offY = (viewH - frameH * s) / 2
                    } else {
                        scale = 1; offX = 0; offY = 0
                    }

                    for d in detections {
                        let bw: CGFloat
                        let bh: CGFloat
                        let bx: CGFloat
                        let by: CGFloat
                        if aspectFit {
                            bw = CGFloat(d.w) * frameW * scale
                            bh = CGFloat(d.h) * frameH * scale
                            bx = CGFloat(d.cx - d.w/2) * frameW * scale + offX
                            by = CGFloat(d.cy - d.h/2) * frameH * scale + offY
                        } else {
                            bw = CGFloat(d.w) * viewW
                            bh = CGFloat(d.h) * viewH
                            bx = CGFloat(d.cx - d.w/2) * viewW
                            by = CGFloat(d.cy - d.h/2) * viewH
                        }
                        let rect = CGRect(x: bx, y: by, width: bw, height: bh)
                        var path = Path()
                        path.addRoundedRect(in: rect, cornerSize: CGSize(width: 4, height: 4))
                        ctx.stroke(path, with: .color(.red), lineWidth: 3)

                        let label = String(format: "person %.0f%%", d.confidence * 100)
                        let textRect = CGRect(x: bx, y: max(0, by - 16), width: 120, height: 14)
                        ctx.draw(Text(label)
                            .font(.system(size: 11, weight: .bold))
                            .foregroundColor(.white),
                            in: textRect)
                    }
                }

                HStack(spacing: 6) {
                    Circle()
                        .fill(seekState == .seek ? .red : .green)
                        .frame(width: 8, height: 8)
                    Text("\(seekState == .seek ? "SEEK" : "ROAM") · \(backendLabel)")
                        .font(.system(size: 11, weight: .bold, design: .monospaced))
                        .foregroundColor(.white)
                }
                .padding(.horizontal, 8)
                .padding(.vertical, 4)
                .background(.black.opacity(0.55))
                .cornerRadius(6)
                .padding(8)
            }
        }
        .allowsHitTesting(false)
    }
}
