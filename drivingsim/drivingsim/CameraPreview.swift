//
//  CameraPreview.swift
//  drivingsim
//
//  AppKit-backed AVCaptureVideoPreviewLayer wrapped for SwiftUI.
//  Mirrors horizontally to match the hand-joystick coordinate convention.
//

@preconcurrency import AVFoundation
import AppKit
import SwiftUI

struct CameraPreview: NSViewRepresentable {
    let session: AVCaptureSession

    final class PreviewView: NSView {
        let previewLayer = AVCaptureVideoPreviewLayer()

        init(session: AVCaptureSession) {
            super.init(frame: .zero)
            wantsLayer = true
            layer = CALayer()
            previewLayer.session = session
            previewLayer.videoGravity = .resizeAspect    // letterbox — predictable rect
            // Mirror horizontally via CALayer transform (anchor 0.5,0.5 → flips in place).
            // SwiftUI .scaleEffect does NOT reach AVCaptureVideoPreviewLayer (Metal-backed),
            // and connection.isVideoMirrored is unreliable at init time.
            previewLayer.transform = CATransform3DMakeScale(-1, 1, 1)
            layer?.addSublayer(previewLayer)
        }
        required init?(coder: NSCoder) { fatalError() }

        override func layout() {
            super.layout()
            previewLayer.frame = bounds
        }
    }

    func makeNSView(context: Context) -> PreviewView { PreviewView(session: session) }
    func updateNSView(_ nsView: PreviewView, context: Context) {
        // Re-attach in case session pointer changed
        if nsView.previewLayer.session !== session {
            nsView.previewLayer.session = session
        }
    }
}
