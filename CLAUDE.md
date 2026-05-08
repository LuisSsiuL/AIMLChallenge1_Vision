# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Hand-gesture-controlled RC car for ER disaster response. Operator controls car via hand pose; onboard camera detects survivors in earthquake rubble. Connects over WiFi (ESP32/Raspberry Pi bridge). Python ML stack on macOS (Apple Silicon).

**Three subsystems:**
1. **Hand pose tracking** (host machine) — MediaPipe detects hand, MobRecon reconstructs 3D mesh → gesture classifier
2. **WiFi command bridge** — translates gestures to RC car motor commands sent over UDP/WebSocket
3. **Onboard human detection** — lightweight model (YOLOv8-nano or similar) running on car's compute unit

## Environment Setup

```bash
# Python 3.8.10 (pinned — matches shebang in mobrecon_live.py)
pyenv install 3.8.10
pyenv local 3.8.10

# HandMesh / MobRecon deps
conda create -n handmesh python=3.9   # HandMesh uses 3.9
conda activate handmesh
pip install torch==1.11.0 torchvision==0.12.0  # cuda11.3 or cpu
pip install -r attempt1/HandMesh/requirements.txt
pip install mediapipe fvcore

# simpleHand deps (separate env or same if compatible)
pip install -r attempt1/simpleHand/requirements.txt
```

## Running Hand Tracking

```bash
# Live webcam demo (MobRecon + MediaPipe)
python attempt1/mobrecon_live.py

# Static image mode
python attempt1/mobrecon_live.py --image picture.avif --out-prefix out/annotated

# Jupyter notebook (exploratory)
jupyter notebook attempt1/mobrecon_demo.ipynb
```

Keys in live demo: `M` cycle display modes (Silhouette → Landmarks → Mesh → All), `Q` quit.

## Architecture: attempt1/

```
attempt1/
├── mobrecon_live.py      # main entry — camera loop, MediaPipe → MobRecon → overlay
├── cell5_demo_fixed.py   # earlier notebook-extracted version (simpler alignment)
├── HandMesh/             # MobRecon model code (mobrecon/models/mobrecon_ds.py)
│   ├── mobrecon/         # model, config, tools
│   └── template/         # right_faces.npy, j_reg.npy (MANO topology)
└── simpleHand/           # alternate hand net (Transformer-based, GPU-heavy)
```

**Inference pipeline in `mobrecon_live.py`:**
- MediaPipe detects hand → square bbox
- Crop resized to 128×128, normalised (mean=0.5, std=0.5) → `MobRecon_DS`
- Model outputs `verts` (778,3) + `joint_img` (21,2) in MANO camera space
- `verts_to_px`: Umeyama 2D similarity aligns model joints → MediaPipe pixel landmarks
- Left hand: crop is horizontally flipped before model, pixel coords unflipped after
- EMA smoothing (`α=0.6`) on vertex pixel positions across frames

**Device priority:** MPS → CUDA → CPU. Checkpoint loaded with `strict=False`; key rename `backbone.conv.` → `backbone.reduce.` handles older checkpoint format.

**openmesh / psbody shims:** `mobrecon_live.py` injects pure-Python stubs for these C++ libs so MobRecon imports succeed without compiling them.

## Key Design Decisions

- MobRecon is right-hand-only — left hand input is mirror-flipped to right before inference
- `J_REG @ verts` (778→21) converts mesh vertices to MANO joints for alignment; these are then converted from MANO order to MPII order via `mano_to_mpii()`
- Do NOT flip the camera frame — model expects natural (non-mirrored) video; only the crop is flipped for left hands

## drivingsim/ — macOS Swift driving simulator

RealityKit FPV sim used to develop and test gesture control before deploying to real RC car.

```
drivingsim/drivingsim/
├── SimScene.swift          # FPV scene: procedural office room (6m×4m, walls/desks/chairs/cabinets/clutter, ~24 obstacles) at RC-car scale (15×20cm car, 6cm eye-height, 1.5m/s max), AABB collision; tick() gates inputs by mode
├── ContentView.swift       # SwiftUI root: mode picker (Off/Hand/Assisted/Auto), hand + depth previews, key-cap HUD
├── DrivingMode.swift       # enum Off/Hand/Assisted/Auto + needsHand/needsDepth helpers
├── HandJoystick.swift      # Native Vision hand gesture driver (see below)
├── DepthDriver.swift       # CoreML depth → 5-zone scoring → WASD (port of live_depth_wasd.py)
├── SimFPVRenderer.swift    # RealityRenderer offscreen FPV camera → CVPixelBuffer (parallel scene)
├── DepthPreview.swift      # SwiftUI depth+zone overlay (top-right when Assisted/Auto)
├── CameraPreview.swift     # NSViewRepresentable wrapping AVCaptureVideoPreviewLayer
├── KeyboardMonitor.swift   # NSEvent WASD monitor; returns nil to consume (no key-repeat beep)
├── DepthAnythingV2SmallF16.mlpackage  # Apple CoreML depth model (518×518, F16, ~47MB)
└── drivingsim.entitlements # sandbox + camera permission
```

### Driving modes

- **Off** — keyboard only (regression baseline)
- **Hand** — keyboard + hand both contribute all four axes
- **Assisted** — depth → W/S only; (kb || hand) → A/D. Operator steers, depth handles collision avoidance
- **Auto** — depth → all four; kb/hand ignored. Self-driving mode

`SimScene.tick(kb:hand:depth:mode:dt:)` ORs the active sources per mode. `ContentView` starts/stops `HandJoystick`, `SimFPVRenderer`, and `DepthDriver` on mode change.

### DepthDriver.swift — CoreML perception

**Pipeline:** `CVPixelBuffer (518×518)` → `MLModel.prediction` → grayscale depth → percentile-clip + EMA bounds → invert (close=high, far=low) → 5-zone obstacle scores (FarLeft/Left/Center/Right/FarRight) → decision tree (escape > reverse > forward > steer-best) → 5-frame majority-vote smoother → WASD `Set<UInt16>`.

Direct CoreML (not VNCoreMLRequest) — depth is dense pixel regression, Vision is for classification/detection.

**Frame source:** `SimFPVRenderer` renders the sim from the car's eye position to a CVPixelBuffer each tick using `RealityRenderer` with a parallel scene (mirrors obstacles + sync car position). Same architecture works unchanged when ESP32-CAM streams from the real RC car later — just swap the frame source.

**Timing logging** every 30 frames:
- `[DepthDriver] XX fps | infer avg X.X ms  max X.X ms`

### HandJoystick.swift — native Vision gesture driver

Ports `controls/gesture_index_joystick.py` to Swift with no Python dependency.

**Pipeline:** `AVCaptureSession (640×480 BGRA)` → `VNDetectHumanHandPoseRequest` → 21 landmarks → One Euro Filter smooth → MCP→tip vector → WASD `Set<UInt16>`

**Key design:**
- Vision joint order mapped to MediaPipe indices (wrist=0, indexMCP=5, indexTip=8, etc.)
- Vision coords: x∈[0,1] right, y∈[0,1] bottom-up → flip both to match Python image convention
- One Euro Filter (Casiez 2012) per axis, per landmark — 21×2 = 42 scalar filters. Adaptive cutoff `cutoff = min_cutoff + beta·|velocity|`; jitter-free at rest, low lag (~5–8ms) on fast motion. Replaced fixed-α EMA. Used internally for joystick decisions; raw landmarks published for overlay (no lag on visual).
- Velocity-based prediction during dropouts: filter's smoothed velocity (`dxPrev`) extrapolates landmarks (`predicted = lastValue + velocity·dt`) when Vision misses a hand. `computeKeys` runs on predicted positions so WASD stays live up to `hold_frames`. Past cap → clear keys + `resetFilters()` for clean re-detection. dt accumulates across missed frames so the next real frame closes the gap in one filter step (no drift artifacts).
- Config: `oef_min_cutoff=1.0, beta=0.05, dcutoff=1.0` in `controls/joystick_config.json` (Casiez defaults). Python side (`gesture_index_joystick.py`) still uses EMA via `ema_alpha`; Swift JSONDecoder ignores it.
- `nonisolated(unsafe)` on `session`/`videoOutput`; `@preconcurrency import AVFoundation` for Swift 6
- Same surface as `KeyboardMonitor`: `.forward/.backward/.left/.right` bools, OR'd in `SimScene.tick`

**Shared config:** `controls/joystick_config.json` — read by both Python and Swift at startup

**Timing logging** (printed to console every 60 frames):
- `[HandJoystick] XX fps | detection XX% | Vision infer avg X.X ms  max X.X ms`
- `[HandJoystick] pipeline (frame→publish) avg X.X ms  max X.X ms`
- Vision inference on Apple Silicon M-series: typically 8–20 ms per frame
- Total pipeline (frame receipt → MainActor key publish): typically 15–40 ms

### Coordinate conventions
- Camera frame NOT flipped — only landmark x is mirrored (`1 - p.location.x`) to match Python's `cv2.flip`
- Left hand: Vision detects chirality natively; no explicit flip needed (unlike MobRecon which is right-hand-only)

## Planned Components (not yet implemented)

- **Open palm stop gesture** — all 5 tips extended → override keys to empty (emergency stop)
- **Analog output** — expose `lastVec` magnitude to SimScene for proportional speed/steer
- **WiFi TX** — UDP socket sending command packets to ESP32 on car
- **Onboard human detector** — YOLOv8-nano or MobileNet-SSD running on Raspberry Pi / Jetson Nano
- **Car firmware** — ESP32 motor PWM control receiving WiFi commands
