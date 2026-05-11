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

enum CameraSource: Hashable, Identifiable {
    case sim
    case live(uniqueID: String, name: String)

    var id: String {
        switch self {
        case .sim:                       return "sim"
        case .live(let uid, _):          return "live:" + uid
        }
    }

    var label: String {
        switch self {
        case .sim:                       return "Sim Scene"
        case .live(_, let name):         return name
        }
    }

    var isLive: Bool {
        if case .live = self { return true }
        return false
    }

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
