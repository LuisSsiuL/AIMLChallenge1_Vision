# ESP32-CAM Session Handoff

Last updated: 2026-05-12

## TL;DR — current working state

- AI-Thinker ESP32-CAM on Bluino Motor Shield v3.1 → MJPEG stream + WebSocket WASD + OTA + runtime resolution switching + mDNS (`espcar.local`).
- W/A/S/D all working. S needed PWM duty 255 to break reverse static friction; `MOTOR_SPEED=255` globally so all keys equal.
- WiFi STA mode (joins `AP RITZ ROYAL`). AP mode toggle in `config.h` flip `WIFI_AP_MODE 0→1` when you want zero-hop low-latency Mac↔ESP.
- **drivingsim Swift app is now the front-facing controller.** `.esp32` source in SourcePickerView → MJPEG preview + WASD over WS. Keyboard/Hand/Depth drivers feed the real car the same way they fed SimScene.
- Build verified — both `pio run` (ESP32) and `xcodebuild` (drivingsim) SUCCESS.

## Files touched this session

| File | What changed |
|---|---|
| `include/config.h` | `WIFI_AP_MODE` toggle + AP creds; `MOTOR_SPEED=255` (was 200) |
| `platformio.ini` | Added `links2004/WebSockets`, `ayushsharma82/ElegantOTA`; build flag `-DELEGANTOTA_USE_ASYNC_WEBSERVER=0`; pinned `upload_port=/dev/cu.usbserial-110` |
| `src/main.cpp` | WebSocket server on :81, ElegantOTA `/update`, `/res` runtime resolution endpoint, sensor tuning, fb_count=2, AP/STA branch, disconnect-clears-keys safety |
| `src/dashboard.cpp` | WS client + auto-reconnect + HTTP fallback, resolution+quality selectors, OTA link, fixed `?t=` query string bug |
| `NOTES.md` | Updated S=255 root cause, Enhancements section, resolution trade-off table |

## Endpoints

| URL | Purpose |
|---|---|
| `http://<ip>/` | Dashboard UI |
| `http://<ip>/stream` | MJPEG video |
| `http://<ip>/cmd?k=w&s=1` | HTTP key cmd (legacy fallback) |
| `http://<ip>/res?size=vga&q=12` | Runtime resolution + quality |
| `http://<ip>/update` | ElegantOTA firmware upload |
| `ws://<ip>:81/` | WebSocket cmd (primary, msg fmt `"w1"`/`"w0"`) |

## Flashing

```bash
# USB-TTL adapter on /dev/cu.usbserial-110, IO0→GND, press reset, release IO0
cd /Users/christianluisefendy/Documents/AIML_Challenge1_Vision/esp32
~/.platformio/penv/bin/pio run -t upload
~/.platformio/penv/bin/pio device monitor
```

After first flash, future updates via OTA:
1. `pio run` → produces `.pio/build/esp32cam/firmware.bin`
2. Browser → `http://<esp-ip>/update`
3. Drag-drop bin → auto-reboot

## Switching WiFi mode

`include/config.h`:
```c
#define WIFI_AP_MODE 0   // STA — joins router. Default. Keeps internet.
#define WIFI_AP_MODE 1   // AP  — ESP creates ESPCar/esprc1234. Mac joins ESP directly.
```

AP mode → browse `http://192.168.4.1/`.

## Camera resolution

- Default: VGA 640×480, jpeg_quality=12.
- Switch at runtime via dashboard dropdown or `GET /res?size=svga&q=15`.
- Sizes: `qqvga, qvga, cif, vga, svga, xga, hd, sxga, uxga`.
- Rule of thumb: bigger size → bump quality number (looser compression) to avoid crashes/corruption. UXGA q≥20.

## Motor pin map (locked in, do not change)

| Function | GPIO | LEDC ch |
|---|---|---|
| LEFT_DRV | 12 | 0 |
| LEFT_DIR | 14 | — |
| RIGHT_DRV | 13 | 4 |
| RIGHT_DIR | 15 | — |
| BUZZER | 2 | held LOW |

Camera reserves LEDC ch 2/3 + timer 1 → motors stay on ch 0/4.

WASD logic:
| Key | L_DIR | R_DIR | PWM |
|---|---|---|---|
| W | HIGH | HIGH | 255 |
| S | LOW  | LOW  | 255 |
| A | HIGH | LOW  | 255 |
| D | LOW  | HIGH | 255 |

`cameraInit()` MUST run before `motorsInit()` or SCCB fails `0x20002`.

## Latency tuning (added at end of session)

Cmd lag was severe at -85 dBm RSSI. Root cause: `handleStream` blocks Arduino loop on core 1 sending each frame (~300ms at VGA on weak WiFi); `server.handleClient()` between frames meant cmd path waited a full frame.

**Fixes applied:**
- `ws.loop()` moved to its own FreeRTOS task pinned to **core 0** (`wsTask`). Frame sending on core 1 no longer starves WebSocket cmd.
- **XCLK 20MHz → 16MHz** — proven to reduce camera↔WiFi interference on classic ESP32.
- Default res reset to **QVGA q=15** (smallest viable; runtime upgradable via dashboard).
- Added `/stats` endpoint (JSON: rssi, fps, kbps, heap, psram).
- Dashboard shows live stats bar (1Hz poll) + 4 presets + auto-tune button.

**Presets:**

| Name | Size | Quality | Use case |
|---|---|---|---|
| ultra-low-lat | QQVGA 160×120 | 20 | Weakest WiFi, max FPS, control-priority |
| low-lat | QVGA 320×240 | 15 | Default, balanced for ≤ -75 dBm |
| balanced | CIF 400×296 | 14 | Decent WiFi |
| quality | VGA 640×480 | 12 | Strong WiFi (≥ -65 dBm) |

**Auto-tune:** cycles all 4 presets, samples FPS for 3s each, picks winner.

**If still laggy:**
1. Watch live stats. RSSI < -80 → switch to AP mode (config.h flip), or move closer.
2. kbps near WiFi ceiling (~2-3 Mbps at weak signal) → drop to ultra-low-lat preset.
3. fps below 5 → camera is bottleneck, not network — try XCLK 8MHz in `cameraInit`.

## Known issues / caveats

- **No deadman safety net.** If WiFi drops a keyup packet motors run forever. Disconnect of WS client force-clears all keys (added this session). HTTP fallback has no such guarantee.
- **RSSI on `AP RITZ ROYAL` measured -85 dBm at last test.** Bad signal → stream stutters, cmd latency spikes. Move closer or flip to AP mode.
- **Bluino shield reverse needs full duty.** Static friction + gear backlash + cogging asymmetry. `MOTOR_SPEED=255` works; lower values may stall S only.
- **Clangd shows fake errors** (`'Arduino.h' file not found`). PlatformIO include paths not loaded into clangd index. Ignore — actual PIO build succeeds.

## Restore points (git)

- `d7c1ffc` — original edge-triggered WASD, W/A/D working, S broken
- `c3922fe` — pulse/deadman experiment (reverted, was laggy)
- Current HEAD: WebSocket + OTA + resolution switching + S working at MOTOR_SPEED=255

Quick rollback to base state:
```bash
git checkout d7c1ffc -- esp32/
```

## Recent fixes worth remembering

- **Parser error `arg missing value: 0`** → caused by `'/stream?'+Date.now()` (no `=`). Fixed to `?t=<n>` in `dashboard.cpp`.
- **C++ const internal linkage** in `dashboard.cpp` — `extern const char DASHBOARD_HTML[]` required on definition.
- **handleStream blocked /cmd** → added `server.handleClient()` + `ws.loop()` inside stream loop.
- **camera init 0x20002** → cameraInit before motorsInit.
- **MJPEGSource parser never found `--frame`** → macOS `URLSession` natively parses `multipart/x-mixed-replace`: strips boundaries, fires `didReceive response` once per part. Body bytes between two response events = one full JPEG. Fix in `drivingsim/Camera/MJPEGSource.swift`: drop the boundary scanner, treat each new response as "flush previous buffer = decode one JPEG".

## drivingsim ↔ ESP32 bridge (new this session)

Swift macOS app drives the real car. Three pieces:

| File | Role |
|---|---|
| `drivingsim/Shared/Config.swift` | Host resolution: `manualHost` → AP `192.168.4.1` → mDNS `espcar.local`. Priority top-down. |
| `drivingsim/Drivers/ESP32WebSocketClient.swift` | `URLSessionWebSocketTask` to `ws://host:81/`. Diffs WASD axes; emits `"w1"`/`"w0"` per transition. Auto-reconnect w/ exp backoff. Re-asserts held keys on (re)connect since firmware deadman clears keys on WS drop. |
| `drivingsim/Camera/MJPEGSource.swift` | Pulls `http://host/stream`; CGImage for SwiftUI preview + `CVPixelBufferPool` (518×392 BGRA) for DepthDriver/YOLO. Mirrors `LiveCameraSource.pullLatest()` API so the FPV timer is source-agnostic. |
| `drivingsim/Camera/CameraSource.swift` | Added `.esp32(host:)` case + `isExternalFrameSource` helper. |
| `drivingsim/App/SourcePickerView.swift` | Appends `.esp32(host: ESP32Config.host)` to discovery list with antenna icon. |
| `drivingsim/App/ContentView.swift` | `@StateObject mjpeg, esp32WS`; mode-gated WASD union (`anyForward/anyBackward/anyLeft/anyRight`) wrapped in `WASDState: Equatable`, pushed via `.onChange` → `esp32WS.publish(...)`. HUD pill shows `ESP32 linked · host · cam X fps`. |

**Wire protocol (locked):** plain text `"<key><01>"` per axis transition. NOT JSON. Defined in `esp32/src/main.cpp:66 applyKeyMsg`.

**mDNS:** firmware calls `MDNS.begin("espcar")` + advertises `_http._tcp:80` and `_ws._tcp:81`. Works in both STA and AP modes. Mac sandbox is disabled (`drivingsim.entitlements`) so `.local` resolution needs no extra entitlement.

**Mode coverage on the real car:** identical to sim. `Off` = keyboard only. `Hand` = keyboard + gestures. `Assisted` = depth handles W/S (using ESP32-CAM frames as input!), kb/hand handles A/D. `Auto` = depth drives all four. `Map*` modes work but ESP32 stream is QVGA → metric depth accuracy degrades vs sim FPV.

**Known gotcha:** holding the ESP32 dashboard in a browser tab blocks the Swift app from getting `/stream` because ESP32 WebServer serves MJPEG in a blocking `while (client.connected())` loop. Close the browser tab before launching drivingsim.

## Suggested next steps (ordered)

1. **Per-motor speed split** — `ledcWrite(0,left); ledcWrite(4,right);` + slider for smooth curve steering instead of pivot-only.
2. **Open palm = emergency stop.** All 5 finger tips extended → clear all keys in `HandJoystick.swift`. Deferred when wiring the bridge — pick up now.
3. **Mac-side WS heartbeat + firmware key-expiry deadman.** ESP32 currently clears keys only on TCP disconnect. If Mac app crashes mid-press while TCP holds, car keeps running. Send ping every 200ms; firmware expires keys after 500ms silence.
4. **Onboard person detection** — Edge Impulse FOMO, 96×96 grayscale, fits ESP32 classic. ER survivor MVP. If accuracy poor → RPi/Jetson with YOLOv8-nano (CLAUDE.md original plan).
5. **Soft-start motor kick** — pulse 255 for 100ms then drop to 200. Better battery life + less noise. Currently 255 always.
6. **Battery runtime measurement** — characterize 18650 endurance under continuous streaming + occasional drive.
7. **DMA-streamed FPV** ([jeanlemotan/esp32-cam-fpv](https://github.com/jeanlemotan/esp32-cam-fpv)) only if MJPEG latency still hurts. Big rewrite.

## Reference repos consulted this session

- [jeanlemotan/esp32-cam-fpv](https://github.com/jeanlemotan/esp32-cam-fpv) — DMA low-latency FPV
- [RomanLut/hx-esp32-cam-fpv](https://github.com/RomanLut/hx-esp32-cam-fpv) — DMA fork
- [mattsroufe/esp32_rc_cars](https://github.com/mattsroufe/esp32_rc_cars) — WS+motor reference
- [vitorccs/esp32cam-rc-car](https://github.com/vitorccs/esp32cam-rc-car) — virtual joystick web UI
- [rzeldent/esp32cam-rtsp](https://github.com/rzeldent/esp32cam-rtsp) — RTSP alternative
- [VinhCao09/Robot_Car_using_ESP32_Cam](https://github.com/VinhCao09/Robot_Car_using_ESP32_Cam) — OTA-enabled robot car
- [sonysunny-com/esp32_tinyML](https://github.com/sonysunny-com/esp32_tinyML) — onboard person detection
- [Edge Impulse FOMO tutorial](https://www.makerguides.com/train-an-object-detection-model-with-edge-impulse-for-esp32-cam/)
- [espressif/esp32-camera sensor.h](https://github.com/espressif/esp32-camera/blob/master/driver/include/sensor.h) — runtime sensor settings reference
