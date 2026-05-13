//
//  Config.swift
//  drivingsim
//
//  Static configuration for the ESP32-CAM RC car bridge.
//
//  Three ways to address the ESP, in priority order:
//    1. manualHost (non-nil)  → use that literal IP/host
//    2. useAPMode (true)      → use apHost (ESP runs its own WiFi)
//    3. otherwise             → mdnsHost via Bonjour ("espcar.local")
//
//  Sandbox is disabled (drivingsim.entitlements) so Bonjour .local
//  resolution needs no extra entitlement.
//

import Foundation

enum ESP32Config {
    /// Non-nil → use this address verbatim and skip mDNS + AP.
    /// Handy before the mDNS firmware is flashed, or when .local is flaky.
    /// Set to `nil` to fall through to AP / mDNS rules.
    static let manualHost: String? = nil

    /// Mirror to `WIFI_AP_MODE` in `esp32/include/config.h`.
    /// true  → ESP creates its own WiFi (ESPCar). Mac must join that SSID.
    /// false → ESP joins router (default; uses mDNS host below).
    static let useAPMode: Bool = true

    /// Bonjour hostname advertised by the firmware mDNS responder.
    /// Matches `MDNS_HOSTNAME` in esp32/src/main.cpp.
    static let mdnsHost = "espcar.local"

    /// Fixed AP-mode IP from `WiFi.softAPIP()` default. Literal IP avoids
    /// flaky mDNS resolution over the ESP's own AP network.
    static let apHost   = "192.168.4.1"

    static var host: String {
        if let m = manualHost { return m }
        return useAPMode ? apHost : mdnsHost
    }

    static let wsPort: Int   = 81
    static let httpPort: Int = 80

    static var streamURL: URL { URL(string: "http://\(host):\(httpPort)/stream")! }
    static var wsURL: URL     { URL(string: "ws://\(host):\(wsPort)/")! }
}
