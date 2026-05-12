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

    // US-layout key codes
    static let W: UInt16 = 13
    static let A: UInt16 = 0
    static let S: UInt16 = 1
    static let D: UInt16 = 2

    private var monitor: Any?

    private static let consumed: Set<UInt16> = [W, A, S, D]

    func start() {
        monitor = NSEvent.addLocalMonitorForEvents(matching: [.keyDown, .keyUp]) { [weak self] event in
            guard let self else { return event }
            Task { @MainActor in
                if event.type == .keyDown && !event.isARepeat {
                    self.pressed.insert(event.keyCode)
                } else if event.type == .keyUp {
                    self.pressed.remove(event.keyCode)
                }
            }
            // Consume WASD events so AppKit doesn't beep on unhandled key-repeats
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
