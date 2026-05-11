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
    var mirror: Bool = true
    var videoGravity: AVLayerVideoGravity = .resizeAspect

    final class PreviewView: NSView {
        let previewLayer = AVCaptureVideoPreviewLayer()

        init(session: AVCaptureSession, mirror: Bool, gravity: AVLayerVideoGravity) {
            super.init(frame: .zero)
            wantsLayer = true
            layer = CALayer()
            previewLayer.session = session
            previewLayer.videoGravity = gravity
            // Mirror horizontally via CALayer transform (anchor 0.5,0.5 → flips in place).
            // SwiftUI .scaleEffect does NOT reach AVCaptureVideoPreviewLayer (Metal-backed),
            // and connection.isVideoMirrored is unreliable at init time.
            previewLayer.transform = mirror ? CATransform3DMakeScale(-1, 1, 1) : CATransform3DIdentity
            layer?.addSublayer(previewLayer)
        }
        required init?(coder: NSCoder) { fatalError() }

        override func layout() {
            super.layout()
            previewLayer.frame = bounds
        }
    }

    func makeNSView(context: Context) -> PreviewView {
        PreviewView(session: session, mirror: mirror, gravity: videoGravity)
    }
    func updateNSView(_ nsView: PreviewView, context: Context) {
        if nsView.previewLayer.session !== session {
            nsView.previewLayer.session = session
        }
        nsView.previewLayer.videoGravity = videoGravity
        let want = mirror ? CATransform3DMakeScale(-1, 1, 1) : CATransform3DIdentity
        if !CATransform3DEqualToTransform(nsView.previewLayer.transform, want) {
            nsView.previewLayer.transform = want
        }
    }
}
