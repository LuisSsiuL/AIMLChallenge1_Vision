//
//  DrivingMode.swift
//  drivingsim
//
//  Mutually-exclusive control modes. Picks which input sources contribute
//  which axes (W/S vs A/D) in SimScene.tick.
//

import Foundation

enum DrivingMode: String, CaseIterable, Identifiable {
    case off         // keyboard only
    case hand        // keyboard + hand
    case assisted    // depth → W/S; (kb || hand) → A/D
    case automated   // depth → all four; kb/hand ignored
    case autoSeek    // depth-sweep (ROAM) ↔ YOLO-drive (SEEK), CoreML backend
    case autoSeekPy  // same state machine, Python ultralytics subprocess backend

    var id: String { rawValue }

    var label: String {
        switch self {
        case .off:        return "Off"
        case .hand:       return "Hand"
        case .assisted:   return "Assisted"
        case .automated:  return "Auto"
        case .autoSeek:   return "Seek"
        case .autoSeekPy: return "SeekPy"
        }
    }

    var needsHand:    Bool { self == .hand || self == .assisted }
    var needsDepth:   Bool { self == .assisted || self == .automated || self == .autoSeek || self == .autoSeekPy }
    var needsYOLO:    Bool { self == .autoSeek }
    var needsYOLOPy:  Bool { self == .autoSeekPy }
    var anyYOLO:      Bool { needsYOLO || needsYOLOPy }
}
