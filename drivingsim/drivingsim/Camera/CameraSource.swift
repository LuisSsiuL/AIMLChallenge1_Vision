//
//  CameraSource.swift
//  drivingsim
//
//  Frame-source selection — Sim Scene or a live AVCaptureDevice
//  (built-in camera, external UVC, Continuity Camera from iPhone, etc).
//
//  Continuity Camera devices surface as `.continuityCamera` device type on
//  macOS Ventura+. Built-in Mac cameras as `.builtInWideAngleCamera`.
//  External displays/cameras as `.external`.
//

@preconcurrency import AVFoundation
import Foundation

/// ESP32-CAM common resolution profile — applied as an intermediate
/// downsample step in LiveCameraSource to simulate the quality of an
/// ESP32-CAM stream (downsample → upsample to model input).
enum ESP32Profile: String, CaseIterable, Identifiable {
    case none      // no downsample — native cam quality
    case qqvga     // 160 × 120
    case qvga      // 320 × 240
    case hvga      // 480 × 320
    case vga       // 640 × 480
    case svga      // 800 × 600
    case xga       // 1024 × 768

    var id: String { rawValue }

    var label: String {
        switch self {
        case .none:  return "Native (no ESP32 sim)"
        case .qqvga: return "ESP32 QQVGA 160×120"
        case .qvga:  return "ESP32 QVGA 320×240"
        case .hvga:  return "ESP32 HVGA 480×320"
        case .vga:   return "ESP32 VGA 640×480"
        case .svga:  return "ESP32 SVGA 800×600"
        case .xga:   return "ESP32 XGA 1024×768"
        }
    }

    var size: CGSize? {
        switch self {
        case .none:  return nil
        case .qqvga: return CGSize(width: 160,  height: 120)
        case .qvga:  return CGSize(width: 320,  height: 240)
        case .hvga:  return CGSize(width: 480,  height: 320)
        case .vga:   return CGSize(width: 640,  height: 480)
        case .svga:  return CGSize(width: 800,  height: 600)
        case .xga:   return CGSize(width: 1024, height: 768)
        }
    }
}

enum CameraSource: Hashable, Identifiable {
    case sim
    case live(uniqueID: String, name: String)
    case esp32(host: String)

    var id: String {
        switch self {
        case .sim:                       return "sim"
        case .live(let uid, _):          return "live:" + uid
        case .esp32(let host):           return "esp32:" + host
        }
    }

    var label: String {
        switch self {
        case .sim:                       return "Sim Scene"
        case .live(_, let name):         return name
        case .esp32(let host):           return "ESP32-CAM (\(host))"
        }
    }

    /// True for any AVCapture-backed source (built-in, external, Continuity).
    /// Distinct from `.esp32` which streams MJPEG over HTTP.
    var isLive: Bool {
        if case .live = self { return true }
        return false
    }

    var isESP32: Bool {
        if case .esp32 = self { return true }
        return false
    }

    /// True for any non-sim source — i.e. anything that feeds frames from
    /// outside SimFPVRenderer (AVCapture OR ESP32 MJPEG).
    var isExternalFrameSource: Bool { isLive || isESP32 }

    var deviceUniqueID: String? {
        if case .live(let uid, _) = self { return uid }
        return nil
    }
}

enum CameraScanner {
    /// Enumerates Sim + every AVCaptureDevice that can produce video frames.
    static func discover() -> [CameraSource] {
        var sources: [CameraSource] = [.sim]

        var types: [AVCaptureDevice.DeviceType] = [.builtInWideAngleCamera, .external]
        #if os(macOS)
        if #available(macOS 14.0, *) {
            types.append(.continuityCamera)
        }
        #endif

        let session = AVCaptureDevice.DiscoverySession(
            deviceTypes: types,
            mediaType: .video,
            position: .unspecified
        )
        for d in session.devices {
            sources.append(.live(uniqueID: d.uniqueID, name: d.localizedName))
        }
        return sources
    }
}
