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
build_flags =
    -DBOARD_HAS_PSRAM
    -DCORE_DEBUG_LEVEL=3
lib_deps =
    esp32-camera
```

## Camera Config

- Resolution: `FRAMESIZE_QVGA` (320×240). VGA stuttered over hotspot.
- `jpeg_quality = 15`, `fb_count = 1`
- Camera uses `LEDC_CHANNEL_2` + `LEDC_TIMER_1` internally
- **`cameraInit()` MUST run BEFORE `motorsInit()`** (otherwise SCCB write fails `0x20002`)

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

## Motor Pin Map (CONFIRMED working for W/A/D, S unsolvable)

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
| S   | LOW               | LOW                | 200     | **NOT WORKING** ❌    |

Serial print confirms correct pin state on S — wheels just don't move. Driver IC ignores/brakes that combo.

## S Reverse — Software Attempts That FAILED

| Attempt | Result |
|---------|--------|
| Both DIR LOW + both PWM (tutorial logic) | no motion |
| Dual-PWM mode, pin pairs `{12,14}` and `{13,15}` | **all keys** dead |
| L298N pairing (AlexeyMal), pin pairs `{14,15}` and `{13,12}` | **all keys** dead — shield NOT wired like a standard L298N car |
| Inverted PWM duty (duty 55 with DIR HIGH) | no motion |
| Hybrid: digitalWrite HIGH on dir pin + PWM=0 | buzz only |
| LEDC on all 4 pins, swap PWM↔dir roles | no reverse |

### Likely conclusion
Bluino ESP32-CAM Motor Shield v3.1 appears designed so reverse direction needs a special control signal from their `kbduino` app firmware (possibly I2S/UART command to an on-board MP3 chip that gates the H-bridge). Reverse not driveable from raw ESP32 GPIO pin toggling.

### Things still untried
- Lower `MOTOR_SPEED` to 130 to rule out brown-out
- Read H-bridge IC label on shield underside (L9110S / TB6612 / DRV8833 / etc.)
- Decompile Bluino's `kbduino` Android app to find their cmd protocol
- Open Bluino's "Esp32 Camera WiFi Robot Car" Play Store app, use upload-sketch feature — likely ships a working firmware we can extract pin protocol from

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

1. (Optional) Try AP mode for stable streaming with low latency.
2. (Optional) Per-motor speed: split `ledcWrite` calls + add slider/param.
3. Wire dashboard JS POST to hand-gesture driver from `drivingsim/` Swift.
4. Decide on onboard human detection (YOLOv8-nano / MobileNet-SSD); host on Mac or RPi/Jetson.
5. Battery runtime measurement.
6. Solve S reverse via Bluino's official app firmware extraction.
