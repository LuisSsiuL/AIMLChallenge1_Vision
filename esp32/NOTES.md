# ESP32-CAM Robot Car — Build Notes

## Hardware

- **Board:** Bluino ESP32-CAM Motor Shield v3.1 (AI-Thinker ESP32-CAM module on top)
- **Chip:** Classic ESP32 (NOT ESP32-S3). PSRAM detected at boot.
- **Camera:** OV2640 @ AI-Thinker pinout
- **Programming:** External USB-TTL adapter (CH340), `/dev/cu.usbserial-110` @ 460800 baud upload
- **WiFi:** ESP32 = 2.4GHz 802.11n only. Real throughput ~10–15 Mbps best case.
- **Power:** 18650 battery in shield holder + ON/OFF slide switch on shield (must be ON to drive motors)
- **Shield URL:** bluino.com / Play Store app `kbduino` ships custom firmware

## PlatformIO Setup

```ini
[env:esp32cam]
platform = espressif32@6.7.0
board = esp32cam
framework = arduino
monitor_speed = 115200
upload_speed = 460800
upload_port  = /dev/cu.usbserial-110
monitor_port = /dev/cu.usbserial-110
build_flags =
    -DBOARD_HAS_PSRAM
    -DCORE_DEBUG_LEVEL=3
    -DELEGANTOTA_USE_ASYNC_WEBSERVER=0
lib_deps =
    esp32-camera
    links2004/WebSockets@^2.4.1
    ayushsharma82/ElegantOTA@^3.1.5
```

## Camera Config

- Resolution: `FRAMESIZE_VGA` (640×480) default. Runtime switch via `/res?size=qvga|cif|vga|svga|xga|hd|sxga|uxga` + dashboard dropdown.
- `jpeg_quality = 12`, `fb_count = 2` (double-buffer for smoother MJPEG)
- Camera uses `LEDC_CHANNEL_2` + `LEDC_TIMER_1` internally
- **`cameraInit()` MUST run BEFORE `motorsInit()`** (otherwise SCCB write fails `0x20002`)
- Sensor tuning post-init: brightness/contrast/saturation +1, AWB on, AEC2 on, lens correction on.

### Resolution / FPS / bandwidth trade-off (OV2640 @ 20MHz XCLK, PSRAM)

| Size  | Pixels      | Approx FPS  | Notes                                              |
|-------|-------------|-------------|----------------------------------------------------|
| QVGA  | 320×240     | 20–30       | Lowest latency, decent on weak WiFi                |
| CIF   | 400×296     | ~20         |                                                    |
| VGA   | 640×480     | 15–20       | **Default — sweet spot**                           |
| SVGA  | 800×600     | 10–15       | Bandwidth tight on hotspot                         |
| XGA   | 1024×768    | 7–10        |                                                    |
| HD    | 1280×720    | 5–8         | OV2640 max useful for streaming                    |
| UXGA  | 1600×1200   | 1–3         | Crashes at high quality; usable for snapshots only |

Bumping resolution → push `jpeg_quality` to 15–20 to avoid corruption (esp32-camera issue #252).

### Runtime sensor tuning (post-init)

```cpp
sensor_t *s = esp_camera_sensor_get();
s->set_brightness(s, 1);     // -2..2
s->set_contrast(s, 1);
s->set_saturation(s, 1);
s->set_whitebal(s, 1);       // AWB on
s->set_awb_gain(s, 1);
s->set_wb_mode(s, 0);        // 0=auto 1=sunny 2=cloudy 3=office 4=home
s->set_exposure_ctrl(s, 1);  // AEC on
s->set_aec2(s, 1);           // AEC DSP
s->set_gain_ctrl(s, 1);      // AGC on
s->set_lenc(s, 1);           // lens correction
```

Caveat from sensor.h: OV2640's SDE register zeros sibling effect bits on each set — call all three (brightness/contrast/saturation) so the last call leaves them all enabled.

### XCLK notes
- Default 20 MHz works. 24 MHz claims +FPS but increases WiFi interference risk on classic ESP32.
- 8 MHz fallback if streaming corrupts or WiFi flakes.

## WiFi / HTTP

- Phone hotspot subnet: `172.16.16.x` or `172.20.10.x` (iPhone)
- Router gave `192.168.1.51` but had **client isolation** → Mac couldn't reach ESP
- iPhone hotspot: must enable **Maximize Compatibility** → 2.4GHz mode. Keep hotspot screen open while ESP boots.

### Critical fix — single-threaded server
`handleStream` had infinite `while(client.connected)` loop → blocked `server.handleClient()` for `/cmd` requests → POSTs got "network connection lost" while stream open.

**Fix:** call `server.handleClient()` inside the stream loop between frames.

### C++ linkage gotcha — dashboard.cpp
`const char DASHBOARD_HTML[]` at namespace scope has **internal linkage by default in C++**, so main.cpp's `extern` declaration couldn't find it. Fix:
```cpp
extern const char DASHBOARD_HTML[] PROGMEM = R"HTML(...)HTML";
```

## Motor Pin Map (CONFIRMED working W/A/S/D)

| Function       | GPIO | LEDC ch | Notes                                |
|----------------|------|---------|--------------------------------------|
| LEFT_DRV_PIN   | 12   | 0       | PWM speed (LEDC)                     |
| LEFT_DIR_PIN   | 14   | —       | digitalWrite direction               |
| RIGHT_DRV_PIN  | 13   | 4       | PWM speed (LEDC)                     |
| RIGHT_DIR_PIN  | 15   | —       | digitalWrite direction               |
| BUZZER_PIN     | 2    | —       | held LOW to silence                  |

**Avoid LEDC channels 2/3** — share a timer with camera XCLK (LEDC_TIMER_1) and will break PWM.

## Control Logic (Bluino tutorial pattern)

| Key | LEFT_DIR (pin 14) | RIGHT_DIR (pin 15) | DRV PWM | Result               |
|-----|-------------------|--------------------|---------|----------------------|
| W   | HIGH              | HIGH               | 200     | forward ✅           |
| A   | HIGH              | LOW                | 200     | turn left ✅         |
| D   | LOW               | HIGH               | 200     | turn right ✅        |
| S   | LOW               | LOW                | 200     | reverse ✅           |

### S Reverse — fix history
Initially S didn't move at `MOTOR_SPEED=200` while W/A/D did → looked like a shield-protocol issue, but root cause was static friction in reverse direction. Raising S duty to 255 worked. Later unified all four keys to `MOTOR_SPEED` per user preference — if S stalls again, bump `MOTOR_SPEED` itself rather than carving out a special case.

```cpp
int speed = active ? MOTOR_SPEED : 0;
ledcWrite(0, speed);
ledcWrite(4, speed);
```

## Enhancements (current build)

### 1. WiFi mode toggle — STA / AP
`config.h`:
```c
#define WIFI_AP_MODE 0   // 0 = STA (join network), 1 = AP (ESP creates network)
```
- **STA (default):** joins router/hotspot. Keeps internet for Mac → ChatGPT/Claude/etc while driving.
- **AP:** ESP becomes its own WiFi (`ESPCar` / `esprc1234`). Mac joins ESP directly → lowest latency, no router hop, no isolation issues. No internet though.

Switch by flipping `WIFI_AP_MODE` and re-flashing (or use OTA).

### 2. WebSocket command channel — `ws://<ip>:81/`
HTTP `/cmd` had a TCP handshake per key event (20-80ms each). Replaced with a persistent WebSocket. Message format: `"<key><state>"` e.g. `"w1"` for W-down, `"s0"` for S-up.

- Library: `links2004/WebSockets@^2.4.1`
- Port 81 (HTTP stays on 80)
- Dashboard auto-connects on load, auto-reconnects on disconnect
- Falls back to HTTP `/cmd` if WS unavailable (legacy preserved)
- On client disconnect, all 4 keys force-cleared (safety: prevent runaway if browser tab closes mid-keypress)

Expected: noticeably lower input latency vs edge-triggered HTTP, especially on weak RSSI.

### 3. ElegantOTA — wireless firmware updates
No more USB-TTL + `IO0→GND` ritual.

- Library: `ayushsharma82/ElegantOTA@^3.1.5`
- Build flag: `-DELEGANTOTA_USE_ASYNC_WEBSERVER=0` (we use sync `WebServer`, not Async)
- Endpoint: `http://<ip>/update` — drag-and-drop `.bin` upload UI
- Firmware bin location after `pio run`: `.pio/build/esp32cam/firmware.bin`

Flow: `pio run` → open `/update` → upload bin → ESP reboots into new firmware.

### 4. Camera double-buffer
`fb_count = 2` (was 1). Camera captures next frame in PSRAM while current frame ships out. Smoother MJPEG, higher achievable FPS at QVGA. PSRAM has plenty of headroom.

### Endpoints summary

| URL                     | Purpose                              |
|-------------------------|--------------------------------------|
| `http://<ip>/`          | Dashboard UI                         |
| `http://<ip>/stream`    | MJPEG video                          |
| `http://<ip>/cmd?k=&s=` | HTTP key cmd (legacy fallback)       |
| `http://<ip>/res?size=&q=` | Runtime resolution + jpeg quality |
| `http://<ip>/update`    | ElegantOTA firmware upload           |
| `ws://<ip>:81/`         | WebSocket cmd (primary)              |

### Enhancements still TODO

- **ESP-NOW cmd path** — peer-to-peer sub-2ms; needs second ESP on Mac side. Skip for now.
- **DMA-streamed FPV** (jeanlemotan/RomanLut style) — sub-100ms glass-to-glass, but full rewrite of camera/network layer. Defer unless MJPEG latency still hurts after WS.
- **RTSP server** (rzeldent/esp32cam-rtsp) — better OpenCV/VLC integration vs MJPEG multipart parse.
- **Onboard person detection** — Edge Impulse FOMO 96×96 grayscale. ER survivor detect MVP. Try before committing to RPi/Jetson.

## Q&A Notes

### Does WiFi speed affect stream + control stability?
**Yes, heavily.**
- **Stream:** MJPEG = ~5–15 KB per frame at QVGA. Needs ~1–2 Mbps stable for ~10 FPS. Weak signal → stutters, freezes, drops.
- **Commands:** tiny payload but latency sensitive. Slow WiFi → noticeable lag between keypress and wheel response.
- **Bottlenecks worst → best:** phone hotspot (cell uplink) < shared WiFi (isolation, congestion) < direct router 2.4 GHz < **ESP in AP mode** (Mac joins ESP's own WiFi, no router, zero hop)
- Best stability path → switch ESP to **WiFi.mode(WIFI_AP)** with `WiFi.softAP("ESPCar","password")`. Mac connects directly. No internet, minimum latency.

### Can we control voltage/speed per motor?
**Yes. PWM duty controls effective voltage.**
- Effective voltage = battery × (duty / 255). 7.4V battery × duty 200 → motors see ~5.8V avg.
- Currently both motors share `MOTOR_SPEED`. Easy to split:
  ```cpp
  ledcWrite(0, leftSpeed);   // 0–255
  ledcWrite(4, rightSpeed);
  ```
- Use cases: smooth curve steering (not pivot), motor mismatch trim, slow-precision mode.
- API options:
  - Query param: `/cmd?k=w&s=1&speed=150`
  - Separate endpoint: `/drive?left=200&right=150`
  - Dashboard slider for global speed

## Timeline of What We Did

1. **Initial config** — assumed Freenove ESP32-S3-CAM (board comment in platformio.ini). Build linked but upload failed: `chip is ESP32 not ESP32-S3`.
2. **Switched to AI-Thinker ESP32-CAM** — rewrote platformio.ini (`board = esp32cam`) and config.h camera pins (32/0/26/27/35-21 etc.).
3. **Fixed C++ linkage bug** in `dashboard.cpp` (added `extern` to const).
4. **WiFi connect attempts**: router blocked us (client isolation), tried 5GHz mismatch, eventually used iPhone hotspot `Luis` / `Satusampe10`.
5. **Camera streamed** at QVGA after dropping from VGA. Direct IP `http://172.16.16.122/` in browser.
6. **Fixed stream/cmd race** — added `server.handleClient()` inside stream loop.
7. **Motor pin discovery** — long iteration through L298N, L9110, phase/enable, dual-PWM modes. Final working pin map confirmed by behavior: pin 12 + pin 14 = LEFT pair, pin 13 + pin 15 = RIGHT pair. Camera channel conflict (ch2/3) ruled out via switching to ch0/4.
8. **W/A/D work via Bluino tutorial logic.** S documented as hardware-shielded.

## Stack / Files

- `esp32/platformio.ini` — board config
- `esp32/src/main.cpp` — camera + HTTP + motors
- `esp32/src/dashboard.cpp` — embedded HTML (`DASHBOARD_HTML`)
- `esp32/include/config.h` — WiFi creds + pin map + `MOTORS_ENABLED`
- `esp32/NOTES.md` — this file

## Next Steps

1. Per-motor speed: split `ledcWrite` calls + slider/param for smooth curve steering vs pivot.
2. Wire `drivingsim/` Swift `HandJoystick` keys → WebSocket client to the ESP. Same WASD surface, just point at `ws://<esp-ip>:81/`.
3. Onboard human detection — try Edge Impulse FOMO first (fits ESP32 classic); RPi/Jetson if accuracy too low.
4. Analog stop gesture (open palm → all keys clear).
5. Battery runtime measurement.
