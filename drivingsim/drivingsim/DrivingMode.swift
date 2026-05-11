//
//  DrivingMode.swift
//  drivingsim
//
//  Mutually-exclusive control modes. Picks which input sources contribute
//  which axes (W/S vs A/D) in SimScene.tick.
//

import Foundation

enum DrivingMode: String, Identifiable {
    case off         // keyboard only
    case hand        // keyboard + hand
    case assisted    // depth → W/S; (kb || hand) → A/D  [hidden from picker]
    case automated   // depth → all four; kb/hand ignored [hidden from picker]
    case autoSeek    // CoreML YOLO ROAM↔SEEK             [hidden from picker]
    case autoSeekPy  // Python YOLO ROAM↔SEEK
    case autoMap     // camera-only SLAM: occupancy grid + frontier exploration + A*

    var id: String { rawValue }

    // Picker shows only these three — autoMap is the unified final mode.
    static let allCases: [DrivingMode] = [.off, .hand, .autoMap]

    var label: String {
        switch self {
        case .off:        return "Off"
        case .hand:       return "Hand"
        case .assisted:   return "Assisted"
        case .automated:  return "Auto"
        case .autoSeek:   return "Seek"
        case .autoSeekPy: return "SeekPy"
        case .autoMap:    return "Map"
        }
    }

    var needsHand:    Bool { self == .hand || self == .assisted }
    var needsDepth:   Bool { self == .assisted || self == .automated || self == .autoSeek || self == .autoSeekPy }
    var needsYOLO:    Bool { self == .autoSeek }
    var needsYOLOPy:  Bool { self == .autoSeekPy || self == .autoMap }
    var anyYOLO:      Bool { needsYOLO || needsYOLOPy }
    var needsMap:     Bool { self == .autoMap }
}
