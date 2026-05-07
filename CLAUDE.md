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

## Planned Components (not yet implemented)

- **Gesture classifier** — map 3D joint angles / finger curl to discrete RC commands (forward, back, turn L/R, stop)
- **WiFi TX** — UDP socket sending command packets to ESP32 on car
- **Onboard human detector** — YOLOv8-nano or MobileNet-SSD running on Raspberry Pi / Jetson Nano
- **Car firmware** — ESP32 motor PWM control receiving WiFi commands
