# ESP32-S3-CAM RC Control Node

Live MJPEG dashboard + WASD remote control for Freenove ESP32-S3-WROOM CAM.

## What gets built
- HTTP server on the ESP32 serving:
  - `/`         → control dashboard (single HTML page)
  - `/stream`   → MJPEG camera stream
  - `/cmd?k=w&s=1` → key down/up events (W/A/S/D)
- 4-motor differential drive (left pair + right pair via two H-bridge channels)

## 1. Install toolchain

1. VSCode → install **PlatformIO IDE** extension.
2. Reopen VSCode after install. PlatformIO icon appears in sidebar.
3. Open this `esp32/` folder as a PlatformIO project (PlatformIO → Open Project).

First build downloads the Espressif32 platform (~5 min, one-time).

## 2. Edit config

Open `include/config.h`:

```c
#define WIFI_SSID     "your-wifi"
#define WIFI_PASSWORD "your-password"
```

If your motor driver pins differ from the defaults (L298N-style), also edit `LEFT_*` and `RIGHT_*` pin defines. Until you wire motors, leave `MOTORS_ENABLED 0` so the ESP only streams + logs keys.

## 3. Wire it up

**Camera:** already wired on the Freenove S3-CAM board. Nothing to do.

**Motors (4× DC, no steering — tank drive):**
Wire front-left + rear-left motors to one H-bridge channel ("LEFT"), front-right + rear-right to the other ("RIGHT").

Default pin map (edit `config.h` if different):

| Side  | IN1 | IN2 | PWM (ENA/ENB) |
|-------|-----|-----|---------------|
| LEFT  | 41  | 42  | 1             |
| RIGHT | 45  | 46  | 2             |

Power motors from a **separate battery** (not the ESP 5V rail) — DC motors brown out the regulator. Common ground between motor driver and ESP.

## 4. Flash

Plug ESP32-S3 into Mac via USB-C. PlatformIO bottom bar:

- **→ (Upload)** — builds + flashes
- **🔌 (Serial Monitor)** — view logs (115200 baud)

On first connect macOS may ask permission for the USB serial device.

Expected serial output:
```
[boot] esp32-cam control node
[wifi] connecting to your-wifi....
[wifi] ip = http://192.168.1.42/
[http] server up
```

## 5. Use the dashboard

Open `http://<that-ip>/` in any browser on the same WiFi. Click the page to give it keyboard focus, then hold W / A / S / D.

- W = both motors forward
- S = both motors reverse
- A = pivot left (left rev, right fwd)
- D = pivot right (left fwd, right rev)
- Release all keys → stop

Visual key indicators light up green; serial monitor prints every key event.

## 6. Tuning

| Knob | Where | Effect |
|------|-------|--------|
| `MOTOR_SPEED` | `config.h` | 0–255 PWM duty |
| `FRAMESIZE_VGA` | `main.cpp` (`cameraInit`) | drop to QVGA for higher FPS |
| `jpeg_quality` | `main.cpp` | 0–63, lower = sharper, larger |

## Troubleshooting

- **camera init failed 0x105** — PSRAM not detected. Check `board_build.flash_size` and `BOARD_HAS_PSRAM` flag.
- **WiFi stuck connecting** — wrong SSID/password, or 5GHz-only network (ESP32 is 2.4GHz only).
- **Upload fails** — hold BOOT button while pressing RESET, then release RESET, then release BOOT. Retry upload.
- **Motors spin wrong direction** — swap IN1/IN2 wires on that side, or swap them in `config.h`.
- **Stream stutters** — phone/laptop on weak WiFi; or lower `frame_size` to `FRAMESIZE_QVGA`.

## Next steps

Once the dashboard drives the car manually, replace the JS key handler with output from your hand-gesture or depth-driver controllers — they already produce W/A/S/D, just POST to `/cmd` instead of injecting OS key events.
