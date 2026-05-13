//
//  KeyboardMonitor.swift
//  drivingsim
//
//  Tracks WASD pressed state via NSEvent local monitor.
//

import AppKit
import Combine

@MainActor
final class KeyboardMonitor: ObservableObject {
    @Published private(set) var pressed: Set<UInt16> = []
    // Edge-triggered counter for `I` key. SwiftUI observers .onChange this to
    // detect single-shot inference requests in Explore mode.
    @Published private(set) var inferenceCounter: UInt64 = 0
    // Edge-triggered counter for `O` key. Explore mode toggles auto-explore.
    @Published private(set) var autoToggleCounter: UInt64 = 0

    // US-layout key codes
    static let W: UInt16 = 13
    static let A: UInt16 = 0
    static let S: UInt16 = 1
    static let D: UInt16 = 2
    static let I: UInt16 = 34
    static let O: UInt16 = 31

    private var monitor: Any?

    private static let consumed: Set<UInt16> = [W, A, S, D, I, O]

    func start() {
        monitor = NSEvent.addLocalMonitorForEvents(matching: [.keyDown, .keyUp]) { [weak self] event in
            guard let self else { return event }
            Task { @MainActor in
                if event.type == .keyDown && !event.isARepeat {
                    self.pressed.insert(event.keyCode)
                    if event.keyCode == KeyboardMonitor.I {
                        self.inferenceCounter &+= 1
                    }
                    if event.keyCode == KeyboardMonitor.O {
                        self.autoToggleCounter &+= 1
                    }
                } else if event.type == .keyUp {
                    self.pressed.remove(event.keyCode)
                }
            }
            // Consume WASD + I events so AppKit doesn't beep on unhandled key-repeats
            return KeyboardMonitor.consumed.contains(event.keyCode) ? nil : event
        }
    }

    func stop() {
        if let m = monitor { NSEvent.removeMonitor(m) }
        monitor = nil
        pressed = []
    }

    var forward:  Bool { pressed.contains(KeyboardMonitor.W) }
    var backward: Bool { pressed.contains(KeyboardMonitor.S) }
    var left:     Bool { pressed.contains(KeyboardMonitor.A) }
    var right:    Bool { pressed.contains(KeyboardMonitor.D) }
}
