//
//  ESP32WebSocketClient.swift
//  drivingsim
//
//  Pushes WASD axis state to the ESP32-CAM over ws://<host>:81/.
//  Wire format matches esp32/src/main.cpp `applyKeyMsg`: plain text
//  "<key><state>" per axis transition (e.g. "w1", "a0", "s1", "d0").
//
//  ContentView calls `publish(forward:backward:left:right:)` whenever the
//  WASD union changes. The client diffs against the last sent state and
//  emits one WS message per axis that flipped. ESP32 clears all keys on
//  WS disconnect (firmware deadman); we mirror that by resetting `lastSent`
//  so the next connect re-emits whichever axes are still held.
//

import Combine
import Foundation

@MainActor
final class ESP32WebSocketClient: NSObject, ObservableObject {
    @Published private(set) var connected: Bool = false
    @Published private(set) var lastError: String?

    /// Latest WASD state requested by the app. The client re-asserts this
    /// after reconnects so a key held during an outage stays pressed
    /// (ESP32 itself drops everything on disconnect — we restore it).
    private var desired: (w: Bool, a: Bool, s: Bool, d: Bool) = (false, false, false, false)

    /// Last state actually transmitted; nil entries mean "never sent since
    /// last (re)connect" so the next publish forces an emit.
    private var lastSent: (w: Bool?, a: Bool?, s: Bool?, d: Bool?) = (nil, nil, nil, nil)

    private var task: URLSessionWebSocketTask?
    private var session: URLSession?
    private var running: Bool = false
    private var reconnectAttempt: Int = 0

    func start() {
        guard !running else { return }
        running = true
        connect()
    }

    func stop() {
        guard running else { return }
        running = false
        // Best-effort all-keys-up before tearing down. ESP32 clears on
        // disconnect anyway, but explicit zeros help if the firmware
        // changes the deadman behaviour later.
        sendAxis("w", false)
        sendAxis("a", false)
        sendAxis("s", false)
        sendAxis("d", false)
        task?.cancel(with: .goingAway, reason: nil)
        task = nil
        connected = false
    }

    /// Called by ContentView whenever the mode-gated WASD union changes.
    /// Emits at most one WS message per axis that flipped since the last
    /// publish; idempotent if nothing changed.
    func publish(forward: Bool, backward: Bool, left: Bool, right: Bool) {
        desired = (forward, left, backward, right)   // (w, a, s, d)
        flush()
    }

    // MARK: - Internal

    private func connect() {
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 10
        let s = URLSession(configuration: config, delegate: self, delegateQueue: nil)
        session = s
        let t = s.webSocketTask(with: ESP32Config.wsURL)
        task = t
        t.resume()
        listen()
    }

    private func listen() {
        task?.receive { [weak self] result in
            Task { @MainActor in
                guard let self else { return }
                switch result {
                case .failure(let err):
                    self.handleDisconnect(reason: "receive: \(err.localizedDescription)")
                case .success:
                    // Server is one-way; just drain and re-arm.
                    if self.running { self.listen() }
                }
            }
        }
    }

    private func flush() {
        if !connected { return }
        if lastSent.w != desired.w { sendAxis("w", desired.w); lastSent.w = desired.w }
        if lastSent.a != desired.a { sendAxis("a", desired.a); lastSent.a = desired.a }
        if lastSent.s != desired.s { sendAxis("s", desired.s); lastSent.s = desired.s }
        if lastSent.d != desired.d { sendAxis("d", desired.d); lastSent.d = desired.d }
    }

    private func sendAxis(_ key: Character, _ down: Bool) {
        guard let t = task else { return }
        let msg = "\(key)\(down ? 1 : 0)"
        t.send(.string(msg)) { [weak self] err in
            if let err {
                Task { @MainActor in
                    self?.handleDisconnect(reason: "send: \(err.localizedDescription)")
                }
            }
        }
    }

    private func handleDisconnect(reason: String) {
        guard connected || task != nil else { return }
        connected = false
        lastError = reason
        task?.cancel()
        task = nil
        // ESP32 firmware clears all keys on disconnect. Forget what we
        // "sent" so the next connect re-asserts whatever is still held.
        lastSent = (nil, nil, nil, nil)
        guard running else { return }
        // Backoff: 0.5s, 1s, 2s, capped at 5s.
        reconnectAttempt = min(reconnectAttempt + 1, 4)
        let delay = min(0.5 * pow(2.0, Double(reconnectAttempt - 1)), 5.0)
        Task { @MainActor [weak self] in
            try? await Task.sleep(nanoseconds: UInt64(delay * 1_000_000_000))
            guard let self, self.running else { return }
            self.connect()
        }
    }
}

extension ESP32WebSocketClient: URLSessionWebSocketDelegate {
    nonisolated func urlSession(_ session: URLSession,
                                webSocketTask: URLSessionWebSocketTask,
                                didOpenWithProtocol protocol: String?) {
        Task { @MainActor in
            self.connected = true
            self.lastError = nil
            self.reconnectAttempt = 0
            // Re-assert current desired state on (re)connect.
            self.flush()
        }
    }

    nonisolated func urlSession(_ session: URLSession,
                                webSocketTask: URLSessionWebSocketTask,
                                didCloseWith closeCode: URLSessionWebSocketTask.CloseCode,
                                reason: Data?) {
        Task { @MainActor in
            self.handleDisconnect(reason: "close \(closeCode.rawValue)")
        }
    }
}
