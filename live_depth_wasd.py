import sys
import os
import time
import threading
from collections import deque
from enum import Enum
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# ESP32-CAM + Depth Anything V2 — WASD Navigation
#
# Extends the live depth pipeline with zone-based driving commands:
#   - Splits the depth map ROI into Left / Center / Right zones
#   - Maps obstacle positions to WASD key presses
#   - Simulates keypresses via pynput (held key, fires on-change only)
#   - Overlays zone highlights + active command on the display window
#
# depth_u8 semantics (AFTER inversion on line 231): 
# 0 = farthest (clear/dark), 255 = closest (obstacle/bright)
#
# Requires:  pip install pynput
# macOS note: grant Accessibility permission to your Terminal app in
#             System Preferences > Privacy & Security > Accessibility.
# ---------------------------------------------------------------------------

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Depth-Anything-V2"))
from depth_anything_v2.dpt import DepthAnythingV2

import numpy as np
import cv2
import torch
import torch.nn.functional as F
import urllib.request
from pynput.keyboard import Controller

# ===========================================================================
# CONFIGURATION
# ===========================================================================

# --- Source ---
# Camera is selected interactively at startup via select_camera().
# Set CAMERA_SCAN_LIMIT to control how many webcam indices are probed.
CAMERA_SCAN_LIMIT = 6   # probe indices 0..N-1 for local/USB/Continuity cameras

# --- Model ---
MODEL_ENCODER = "vitl"
ENCODER_CONFIGS = {
    "vits": {"features": 64,  "out_channels": [48,  96,  192,  384]},
    "vitb": {"features": 128, "out_channels": [96,  192, 384,  768]},
    "vitl": {"features": 256, "out_channels": [256, 512, 1024, 1024]},
}
CHECKPOINT = os.path.join(
    os.path.dirname(__file__), "checkpoints",
    f"depth_anything_v2_{MODEL_ENCODER}.pth",
)

MODEL_INPUT_SIZE   = 252
USE_HALF_PRECISION = True

# --- Capture ---
CAPTURE_WIDTH  = 320
CAPTURE_HEIGHT = 240
FRAME_SKIP     = 1
COLORMAP       = cv2.COLORMAP_INFERNO

# --- Flip ---
FLIP_HORIZONTAL = False  # True = mirror left-right
FLIP_VERTICAL   = False  # True = flip upside-down

# --- Navigation ---
# ObstacleScore displayed on screen: LOW = close/obstacle, HIGH = far/open space
# Always steer toward the zone with the HIGHEST score (most open space).
FORWARD_THRESHOLD     = 150    # Center score above this → safe to go FORWARD
REVERSE_THRESHOLD     = 80     # All zone scores below this → too close, REVERSE
STEER_MIN_DIFF        = 10     # Min score difference between sides to prefer one over the other
FLOOR_EMA_ALPHA       = 0.15   # EMA weight for FloorBaseline update (0 = no adapt, 1 = instant)
NAV_ROW_START         = 0.30   # top of ROI as fraction of frame height (ignore sky/background)
FLOOR_BAND_FRACTION   = 0.10   # FloorBand height as fraction of ROI height (near-floor calibration strip)
CMD_SMOOTH_FRAMES     = 5      # majority-vote window size to stabilise noisy commands
REVERSE_ESCAPE_FRAMES = 10     # consecutive frames all-close before triggering escape manoeuvre
REL_MARGIN            = 15.0   # min ObstacleScore difference to prefer one side over the other
DEBUG_NAVIGATION      = False  # Set to True to print zone scores and decisions

# --- Depth normalization stability ---
# EMA smoothing for depth scale — prevents colormap flickering across frames.
# Lower alpha = more stable but slower to adapt (0.05–0.15 recommended).
DEPTH_EMA_ALPHA    = 0.1   # EMA weight for updating running depth scale
DEPTH_PERCENTILE   = 2     # clip bottom/top N% of depth values before scaling

# ===========================================================================
# Startup validation — clamp navigation constants to safe ranges
# ===========================================================================

# --- 7.3  Validate NAV_ROW_START ---
if not (0.0 <= NAV_ROW_START <= 0.9):
    _clamped_nav = max(0.0, min(0.9, NAV_ROW_START))
    print(
        f"[NAV][WARN] NAV_ROW_START={NAV_ROW_START} is outside [0.0, 0.9]; "
        f"clamping to {_clamped_nav}"
    )
    NAV_ROW_START = _clamped_nav

# --- 2.5  Validate FLOOR_BAND_FRACTION ---
if not (0.05 <= FLOOR_BAND_FRACTION <= 0.5):
    _clamped = max(0.05, min(0.5, FLOOR_BAND_FRACTION))
    print(
        f"[NAV][WARN] FLOOR_BAND_FRACTION={FLOOR_BAND_FRACTION} is outside [0.05, 0.5]; "
        f"clamping to {_clamped}"
    )
    FLOOR_BAND_FRACTION = _clamped

# ===========================================================================
# Device & model setup
# ===========================================================================
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

use_fp16 = USE_HALF_PRECISION and device.type in ("cuda", "mps")
print(f"[INFO] device={device}  fp16={use_fp16}  encoder={MODEL_ENCODER}  input_size={MODEL_INPUT_SIZE}")

cfg   = ENCODER_CONFIGS[MODEL_ENCODER]
model = DepthAnythingV2(encoder=MODEL_ENCODER, **cfg)
model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu"))
model.to(device).eval()
if use_fp16:
    model.half()

# Warm-up pass — avoids slow first-frame penalty on MPS/CUDA
with torch.no_grad():
    _dummy = torch.zeros(
        1, 3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE,
        dtype=torch.float16 if use_fp16 else torch.float32,
    ).to(device)
    model(_dummy)
print("[INFO] Warm-up done.")

# ===========================================================================
# MJPEG stream reader (background thread for ESP32-CAM)
# ===========================================================================
class MJPEGCapture:
    """Non-blocking MJPEG-over-HTTP reader (ESP32-CAM default stream)."""

    def __init__(self, url: str):
        self.url    = url
        self._frame = None
        self._lock  = threading.Lock()
        self._stop  = threading.Event()
        t = threading.Thread(target=self._reader, daemon=True)
        t.start()

    def _reader(self):
        stream = urllib.request.urlopen(self.url, timeout=10)
        buf = b""
        while not self._stop.is_set():
            buf += stream.read(4096)
            a  = buf.find(b"\xff\xd8")   # JPEG SOI
            b_ = buf.find(b"\xff\xd9")   # JPEG EOI
            if a != -1 and b_ != -1 and b_ > a:
                jpg = buf[a : b_ + 2]
                buf = buf[b_ + 2:]
                img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    with self._lock:
                        self._frame = img

    def read(self):
        with self._lock:
            if self._frame is None:
                return False, None
            return True, self._frame.copy()

    def release(self):
        self._stop.set()


# ===========================================================================
# Inference — returns (colorized BGR, depth_u8 grayscale)
# ===========================================================================

# EMA state for stable depth normalization across frames.
# Tracks smoothed percentile bounds so the colormap doesn't jump when
# scene content changes (e.g. empty room vs. person walking in).
_ema_d_min: float = None
_ema_d_max: float = None


def run_inference(frame_bgr: np.ndarray):
    """
    Run Depth Anything V2 on a BGR frame.
    Returns (colorized_bgr, depth_u8) both resized to original frame dims.

    Normalization uses an EMA of per-frame percentile bounds so the colormap
    stays consistent across scene changes instead of re-stretching every frame.
    """
    global _ema_d_min, _ema_d_max

    h, w      = frame_bgr.shape[:2]
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    with torch.no_grad():
        if use_fp16:
            image, (oh, ow) = model.image2tensor(frame_rgb, MODEL_INPUT_SIZE)
            image = image.half()
            depth = model(image)
            depth = F.interpolate(
                depth[:, None], (oh, ow), mode="bilinear", align_corners=True
            )[0, 0].float().cpu().numpy()
        else:
            depth = model.infer_image(frame_rgb, input_size=MODEL_INPUT_SIZE)

    # Percentile clip removes extreme outlier pixels before computing scale.
    # This prevents a single bright/dark pixel from collapsing the whole range.
    p_lo = float(np.percentile(depth, DEPTH_PERCENTILE))
    p_hi = float(np.percentile(depth, 100 - DEPTH_PERCENTILE))

    # Bootstrap EMA on first frame, then blend toward current frame's range.
    if _ema_d_min is None:
        _ema_d_min = p_lo
        _ema_d_max = p_hi
    else:
        _ema_d_min = DEPTH_EMA_ALPHA * p_lo + (1.0 - DEPTH_EMA_ALPHA) * _ema_d_min
        _ema_d_max = DEPTH_EMA_ALPHA * p_hi + (1.0 - DEPTH_EMA_ALPHA) * _ema_d_max

    d_range = _ema_d_max - _ema_d_min
    if d_range > 1e-6:
        depth_norm = np.clip((depth - _ema_d_min) / d_range, 0.0, 1.0)
        depth_u8   = (depth_norm * 255).astype(np.uint8)
    else:
        depth_u8 = np.zeros_like(depth, dtype=np.uint8)

    depth_u8  = 255 - depth_u8   # invert: bright = near/obstacle, dark = far/clear
    # NOTE: After inversion, runtime shows LOW values for close objects, HIGH for far.
    # Navigation logic uses this directly: high score = open space, low score = obstacle.

    # Gamma < 1 darkens the overall image without clipping highlights.
    # 0.6 gives a noticeably dimmer map; raise toward 1.0 to brighten back.
    DISPLAY_GAMMA = 0.6
    depth_display = (np.power(depth_u8 / 255.0, DISPLAY_GAMMA) * 255).astype(np.uint8)

    colorized = cv2.applyColorMap(depth_display, COLORMAP)
    if colorized.shape[:2] != (h, w):
        colorized = cv2.resize(colorized, (w, h), interpolation=cv2.INTER_LINEAR)
        depth_u8  = cv2.resize(depth_u8,  (w, h), interpolation=cv2.INTER_LINEAR)

    return colorized, depth_u8


# ===========================================================================
# Navigation data types
# ===========================================================================

class ZoneState(Enum):
    """Classification of a NavigationZone based on its ObstacleScore."""
    CLEAR     = "clear"      # ObstacleScore < CLEAR_THRESHOLD — safe to navigate toward
    UNCERTAIN = "uncertain"  # CLEAR_THRESHOLD ≤ ObstacleScore ≤ OBSTACLE_THRESHOLD — brake
    BLOCKED   = "blocked"    # ObstacleScore > OBSTACLE_THRESHOLD — obstacle present


class DrivingCommand(Enum):
    """Driving commands mapped to their pynput key characters.
    BRAKE maps to an empty string (no key held)."""
    FORWARD    = "w"   # drive forward
    TURN_LEFT  = "a"   # turn left
    REVERSE    = "s"   # reverse
    TURN_RIGHT = "d"   # turn right
    BRAKE      = ""    # release all keys / stop


@dataclass
class ZoneInfo:
    """Per-zone assessment produced by compute_zone_scores each frame."""
    name:           str        # "FarLeft" | "Left" | "Center" | "Right" | "FarRight"
    mean_depth:     float      # raw mean depth_u8 value in this zone
    obstacle_score: float      # floor-corrected score clamped to [0, 255]
    state:          ZoneState  # CLEAR / UNCERTAIN / BLOCKED classification


# ===========================================================================
# Navigation helpers — floor baseline calibration
# ===========================================================================

def update_floor_baseline(
    depth_u8: np.ndarray,
    roi_row0: int,
    floor_band_fraction: float,
    current_baseline: float | None,
    alpha: float = FLOOR_EMA_ALPHA,
) -> float:
    """
    Compute an EMA-smoothed FloorBaseline from the bottom strip of the ROI.

    The FloorBand is the bottom ``floor_band_fraction`` of the ROI — the strip
    of pixels expected to contain only near-floor tiles.  Its median depth is
    blended into the running baseline via an exponential moving average so the
    system adapts to changing floor textures without reacting to single-frame
    spikes.

    Args:
        depth_u8:           Full depth frame (H×W uint8).
        roi_row0:           First row of the ROI (pixels above this are ignored).
        floor_band_fraction: Height of the FloorBand as a fraction of ROI height.
                            Caller is responsible for passing a validated value.
        current_baseline:   Previous baseline value, or ``None`` on the first call
                            (bootstraps directly from the median).
        alpha:              EMA smoothing weight (0 = no adapt, 1 = instant).

    Returns:
        Updated FloorBaseline clamped to [20, 200].
    """
    # --- 2.1  Extract the FloorBand slice ---
    roi = depth_u8[roi_row0:, :]
    roi_h = roi.shape[0]
    band_h = max(1, int(roi_h * floor_band_fraction))
    floor_band = roi[-band_h:, :]          # bottom band_h rows of the ROI

    # --- 2.2  Compute the median of the FloorBand ---
    median_val = float(np.median(floor_band))

    # --- 2.3  EMA blend; bootstrap on first call ---
    if current_baseline is None:
        # First frame: initialise directly from the observed median
        new_baseline = median_val
    else:
        new_baseline = alpha * median_val + (1.0 - alpha) * current_baseline

    # --- 2.4  Clamp to [20, 200] and log when clamping fires ---
    if new_baseline < 20.0:
        print(
            f"[NAV][DIAG] FloorBaseline clamped from {new_baseline:.2f} → 20 "
            f"(median={median_val:.1f})"
        )
        new_baseline = 20.0
    elif new_baseline > 200.0:
        print(
            f"[NAV][DIAG] FloorBaseline clamped from {new_baseline:.2f} → 200 "
            f"(median={median_val:.1f})"
        )
        new_baseline = 200.0

    return new_baseline


# ===========================================================================
# Navigation logic — zone scoring
# ===========================================================================

ZONE_NAMES = ("FarLeft", "Left", "Center", "Right", "FarRight")


def compute_zone_scores(roi: np.ndarray, floor_baseline: float) -> list:
    """
    Divide the ROI into five equal-width columns and compute obstacle scores.

    Pure function — no side effects, no global state.

    Args:
        roi:            2-D uint8 NumPy array (H×W) representing the navigation
                        Region of Interest (already cropped to the ROI rows).
        floor_baseline: Current FloorBaseline scalar (clamped to [20, 200]).

    Returns:
        List of five ZoneInfo objects in left-to-right order:
        [FarLeft, Left, Center, Right, FarRight].
    """
    h, w = roi.shape

    # --- 3.1  Divide ROI into five equal-width columns ---
    # Use np.array_split so the last zone absorbs any remainder pixels,
    # keeping all five zones non-empty even for narrow frames.
    col_edges = [int(w * i / 5) for i in range(6)]  # 6 edges → 5 intervals

    zones: list = []
    for i, name in enumerate(ZONE_NAMES):
        x0 = col_edges[i]
        x1 = col_edges[i + 1]
        zone_slice = roi[:, x0:x1]

        # --- 3.2  Compute mean depth_u8 for this zone ---
        mean_depth = float(zone_slice.mean())

        # --- 3.3  ObstacleScore = mean depth directly ---
        # LOW score = close/obstacle, HIGH score = far/open space
        obstacle_score = float(np.clip(mean_depth, 0.0, 255.0))

        # --- 3.4  Classify zone: high score = CLEAR (open), low score = BLOCKED (obstacle) ---
        if obstacle_score > FORWARD_THRESHOLD:
            state = ZoneState.CLEAR       # high = far = safe
        elif obstacle_score < REVERSE_THRESHOLD:
            state = ZoneState.BLOCKED     # low = close = obstacle
        else:
            state = ZoneState.UNCERTAIN   # in between

        zones.append(ZoneInfo(
            name=name,
            mean_depth=mean_depth,
            obstacle_score=obstacle_score,
            state=state,
        ))

    # --- 3.5  Return list of five ZoneInfo objects in left-to-right order ---
    return zones


# ===========================================================================
# Navigation logic — five-zone decision tree
# ===========================================================================

def decide_command(
    zones: list,
    consecutive_reverse: int,
    escape_phase: int,
) -> tuple:
    """
    Navigation rule: always steer toward the zone with the HIGHEST score.
    HIGH score = far away = open space. LOW score = close = obstacle.

    Priority order:
      1. Escape manoeuvre (if currently escaping)
      2. Center is open enough → FORWARD
      3. Pick the side with the highest score → TURN_LEFT or TURN_RIGHT
      4. Everything too close → REVERSE, then escape turn

    WASD:  W=forward  A=left  S=reverse  D=right

    Returns: (raw_command, new_consecutive_reverse, new_escape_phase)
    """
    far_left, left, center, right, far_right = zones

    # Aggregate scores per side (average of the two side zones)
    left_score  = (far_left.obstacle_score + left.obstacle_score)  / 2.0
    right_score = (far_right.obstacle_score + right.obstacle_score) / 2.0
    best_score  = max(far_left.obstacle_score, left.obstacle_score,
                      center.obstacle_score,
                      right.obstacle_score, far_right.obstacle_score)

    if DEBUG_NAVIGATION:
        print(
            f"[NAV] FL={far_left.obstacle_score:.0f} L={left.obstacle_score:.0f} "
            f"C={center.obstacle_score:.0f} R={right.obstacle_score:.0f} "
            f"FR={far_right.obstacle_score:.0f}  "
            f"left_avg={left_score:.0f} right_avg={right_score:.0f}  "
            f"[high=open, low=close]"
        )

    # ------------------------------------------------------------------
    # 1. Escape manoeuvre — takes priority
    # ------------------------------------------------------------------
    if escape_phase != 0:
        if escape_phase > 0:
            new_escape_phase = escape_phase - 1
            if new_escape_phase == 0:
                new_escape_phase = -5   # switch to right phase
            if DEBUG_NAVIGATION:
                print(f"[NAV] → TURN_LEFT (escape {escape_phase})")
            return (DrivingCommand.TURN_LEFT, 0, new_escape_phase)
        else:
            new_escape_phase = escape_phase + 1
            if new_escape_phase == 0:
                if DEBUG_NAVIGATION:
                    print(f"[NAV] → FORWARD (escape done)")
                return (DrivingCommand.FORWARD, 0, 0)
            if DEBUG_NAVIGATION:
                print(f"[NAV] → TURN_RIGHT (escape {escape_phase})")
            return (DrivingCommand.TURN_RIGHT, 0, new_escape_phase)

    # ------------------------------------------------------------------
    # 2. Everything is too close → REVERSE
    # ------------------------------------------------------------------
    if best_score < REVERSE_THRESHOLD:
        new_consecutive = consecutive_reverse + 1
        if new_consecutive > REVERSE_ESCAPE_FRAMES:
            # Stuck too long — start escape turn toward the less-blocked side
            if left_score >= right_score:
                if DEBUG_NAVIGATION:
                    print(f"[NAV] → TURN_LEFT (escape start, left_score={left_score:.0f})")
                return (DrivingCommand.TURN_LEFT, 0, 5)
            else:
                if DEBUG_NAVIGATION:
                    print(f"[NAV] → TURN_RIGHT (escape start, right_score={right_score:.0f})")
                return (DrivingCommand.TURN_RIGHT, 0, -5)
        if DEBUG_NAVIGATION:
            print(f"[NAV] → REVERSE (best_score={best_score:.0f} < {REVERSE_THRESHOLD}, consecutive={new_consecutive})")
        return (DrivingCommand.REVERSE, new_consecutive, 0)

    # Reset reverse counter when not reversing
    consecutive_reverse = 0

    # ------------------------------------------------------------------
    # 3. Center is open → FORWARD
    # ------------------------------------------------------------------
    if center.obstacle_score >= FORWARD_THRESHOLD:
        if DEBUG_NAVIGATION:
            print(f"[NAV] → FORWARD (center={center.obstacle_score:.0f})")
        return (DrivingCommand.FORWARD, 0, 0)

    # ------------------------------------------------------------------
    # 4. Center is not open enough — steer toward the highest-scoring side
    # ------------------------------------------------------------------
    # If one side is clearly better, steer that way
    if left_score > right_score + STEER_MIN_DIFF:
        if DEBUG_NAVIGATION:
            print(f"[NAV] → TURN_LEFT (left={left_score:.0f} > right={right_score:.0f})")
        return (DrivingCommand.TURN_LEFT, 0, 0)

    if right_score > left_score + STEER_MIN_DIFF:
        if DEBUG_NAVIGATION:
            print(f"[NAV] → TURN_RIGHT (right={right_score:.0f} > left={left_score:.0f})")
        return (DrivingCommand.TURN_RIGHT, 0, 0)

    # Scores are close — check individual far zones to break the tie
    if far_left.obstacle_score > far_right.obstacle_score:
        if DEBUG_NAVIGATION:
            print(f"[NAV] → TURN_LEFT (tie-break: far_left={far_left.obstacle_score:.0f} > far_right={far_right.obstacle_score:.0f})")
        return (DrivingCommand.TURN_LEFT, 0, 0)

    if far_right.obstacle_score > far_left.obstacle_score:
        if DEBUG_NAVIGATION:
            print(f"[NAV] → TURN_RIGHT (tie-break: far_right={far_right.obstacle_score:.0f} > far_left={far_left.obstacle_score:.0f})")
        return (DrivingCommand.TURN_RIGHT, 0, 0)

    # Truly symmetric — keep going forward
    if DEBUG_NAVIGATION:
        print(f"[NAV] → FORWARD (symmetric, center={center.obstacle_score:.0f})")
    return (DrivingCommand.FORWARD, 0, 0)


# ===========================================================================
# Command smoothing — majority-vote buffer with BRAKE priority
# ===========================================================================

class Smoother:
    """
    Sliding-window majority-vote smoother for DrivingCommands.

    Maintains a deque of the last ``window`` raw commands and returns a
    smoothed command on each ``push`` call according to the following rules
    (in priority order):

    1. **Startup** — while the buffer has fewer than ``window`` entries,
       return the most recent raw command directly (no waiting).
    2. **BRAKE priority** — if BRAKE appears in more than one-third of the
       window entries, return BRAKE regardless of the majority vote.
    3. **Majority vote** — return the command that appears most frequently;
       ties are broken by returning the most recent command among the tied
       commands (i.e. the rightmost occurrence in the deque).
    """

    def __init__(self, window: int):
        # 5.1  Maintain a deque of max length CMD_SMOOTH_FRAMES
        self._buf: deque = deque(maxlen=window)

    def push(self, cmd: DrivingCommand) -> DrivingCommand:
        """Add raw command to the window and return the smoothed command."""
        self._buf.append(cmd)

        # 5.4  Startup: fewer than window entries → return most recent raw command
        if len(self._buf) < self._buf.maxlen:
            return cmd

        # 5.2  BRAKE priority: BRAKE in more than one-third of window → return BRAKE
        brake_count = sum(1 for c in self._buf if c == DrivingCommand.BRAKE)
        if brake_count > len(self._buf) / 3:
            return DrivingCommand.BRAKE

        # 5.3  Majority vote with tie-breaking by most recent command
        # Count occurrences of each command
        counts: dict = {}
        for c in self._buf:
            counts[c] = counts.get(c, 0) + 1

        max_count = max(counts.values())

        # Collect all commands tied at the maximum count
        tied = {c for c, n in counts.items() if n == max_count}

        # Among tied commands, return the one that appears most recently
        # (rightmost in the deque = last element that is in the tied set)
        for c in reversed(self._buf):
            if c in tied:
                return c

        # Fallback (should never be reached)
        return cmd


class KeyController:
    """
    Translates a DrivingCommand into held WASD keypresses via pynput.

    Design rules:
    - ``_last`` is initialised to BRAKE so the car starts stopped (6.4).
    - ``send()`` is a no-op when the command has not changed (6.2).
    - On a command change, the previously held key is released and the new
      key is pressed within the same call — no gap between release and press (6.2).
    - BRAKE maps to no key; sending BRAKE releases any held key (6.1).
    - ``release_all()`` releases any currently held key and resets state (6.3).
    """

    # 6.1  Map DrivingCommand → pynput key character (BRAKE → None = no key)
    _CMD_KEY: dict = {
        DrivingCommand.FORWARD:    DrivingCommand.FORWARD.value,    # "w"
        DrivingCommand.TURN_LEFT:  DrivingCommand.TURN_LEFT.value,  # "a"
        DrivingCommand.REVERSE:    DrivingCommand.REVERSE.value,    # "s"
        DrivingCommand.TURN_RIGHT: DrivingCommand.TURN_RIGHT.value, # "d"
        DrivingCommand.BRAKE:      None,                            # no key
    }

    def __init__(self):
        # 6.4  Start in BRAKE state so the car is stopped on initialisation
        self._last: DrivingCommand = DrivingCommand.BRAKE
        self._kb: Controller = Controller()

    def send(self, cmd: DrivingCommand) -> None:
        """
        Release the previously held key and press the new key in one call.
        No-op when the command is unchanged (6.2).
        """
        # 6.2  No-op if command has not changed
        if cmd == self._last:
            return

        prev_key = self._CMD_KEY[self._last]
        new_key  = self._CMD_KEY[cmd]

        # Release the previously held key (if any)
        if prev_key is not None:
            self._kb.release(prev_key)

        # Press the new key (if any — BRAKE means no key held)
        if new_key is not None:
            self._kb.press(new_key)

        self._last = cmd

    def release_all(self) -> None:
        """
        Release any currently held key and reset state to BRAKE (6.3).
        Safe to call multiple times or when no key is held.
        """
        current_key = self._CMD_KEY[self._last]
        if current_key is not None:
            self._kb.release(current_key)
        self._last = DrivingCommand.BRAKE



# ===========================================================================
# Navigator — stateful wrapper that drives the full per-frame pipeline
# ===========================================================================

class Navigator:
    """
    Stateful wrapper around the navigation pipeline.

    Owns all mutable state:
      - EMA FloorBaseline accumulator
      - Smoother instance
      - Consecutive-reverse counter and escape-phase counter
      - Consecutive inference-failure counter and error-banner flag

    The ``process_frame`` method is the single entry point called once per
    frame.  It orchestrates the pure helper functions and returns everything
    the overlay renderer needs.

    Note: ``run_inference`` is called *outside* this class (in ``main``).
    ``process_frame`` accepts the already-computed ``depth_u8`` array.
    The ``try/except`` around ``run_inference`` is handled by the caller
    (``main``), which passes the result here or calls
    ``Navigator.handle_inference_error()`` on failure.
    """

    def __init__(self):
        # 7.1  EMA state, smoother, reverse counter, escape phase
        self._floor_baseline: float | None = None
        self._smoother: Smoother = Smoother(CMD_SMOOTH_FRAMES)
        self._consecutive_reverse: int = 0
        self._escape_phase: int = 0

        # 7.5 / 7.6  Inference-failure tracking
        self._consecutive_failures: int = 0
        self.error_banner: bool = False   # True when >= 3 consecutive failures

    # ------------------------------------------------------------------
    # 7.2  Main per-frame entry point
    # ------------------------------------------------------------------
    def process_frame(
        self, depth_u8: np.ndarray
    ) -> tuple:
        """
        Run the full navigation pipeline on a single depth frame.

        Steps (in order):
          1. Degenerate-frame guard (7.4)
          2. update_floor_baseline
          3. compute_zone_scores
          4. decide_command
          5. Smoother.push

        Args:
            depth_u8: Full-frame uint8 depth map (H×W), 0=far/clear, 255=near/obstacle.

        Returns:
            (smoothed_command, zone_infos, floor_baseline)
              smoothed_command : DrivingCommand after majority-vote smoothing
              zone_infos       : list[ZoneInfo] — five zones, left-to-right
              floor_baseline   : float — current EMA FloorBaseline value
        """
        # 7.4  Degenerate depth map guard — all values identical (std ≈ 0)
        if float(depth_u8.std()) < 1e-3:
            print(
                f"[NAV][WARN] Degenerate depth map detected "
                f"(std={depth_u8.std():.6f} < 1e-3); emitting BRAKE"
            )
            smoothed = self._smoother.push(DrivingCommand.BRAKE)
            # Return a flat set of UNCERTAIN zones as a safe placeholder
            h, w = depth_u8.shape
            roi_row0 = int(h * NAV_ROW_START)
            roi = depth_u8[roi_row0:, :]
            placeholder_zones = [
                ZoneInfo(name=n, mean_depth=0.0, obstacle_score=0.0, state=ZoneState.UNCERTAIN)
                for n in ("FarLeft", "Left", "Center", "Right", "FarRight")
            ]
            baseline = self._floor_baseline if self._floor_baseline is not None else 20.0
            return (smoothed, placeholder_zones, baseline)

        # Step 1 — update FloorBaseline via EMA
        h, w = depth_u8.shape
        roi_row0 = int(h * NAV_ROW_START)
        self._floor_baseline = update_floor_baseline(
            depth_u8=depth_u8,
            roi_row0=roi_row0,
            floor_band_fraction=FLOOR_BAND_FRACTION,
            current_baseline=self._floor_baseline,
            alpha=FLOOR_EMA_ALPHA,
        )

        # Step 2 — compute per-zone obstacle scores
        roi = depth_u8[roi_row0:, :]
        zone_infos = compute_zone_scores(roi, self._floor_baseline)

        # Step 3 — decide raw command (pure decision tree)
        raw_cmd, self._consecutive_reverse, self._escape_phase = decide_command(
            zones=zone_infos,
            consecutive_reverse=self._consecutive_reverse,
            escape_phase=self._escape_phase,
        )

        # Step 4 — smooth via majority-vote buffer
        smoothed_cmd = self._smoother.push(raw_cmd)

        # 7.7  Return (smoothed_command, zone_infos, floor_baseline)
        return (smoothed_cmd, zone_infos, self._floor_baseline)

    # ------------------------------------------------------------------
    # 7.5 / 7.6  Inference-failure handling (called by main on exception)
    # ------------------------------------------------------------------
    def handle_inference_error(self, exc: Exception) -> DrivingCommand:
        """
        Called by the main loop when ``run_inference`` raises an exception.

        Emits BRAKE, logs the error with a timestamp, increments the failure
        counter, and sets the error-banner flag when failures reach 3.

        Returns:
            DrivingCommand.BRAKE (always)
        """
        self._consecutive_failures += 1
        ts = time.strftime("%Y-%m-%dT%H:%M:%S")
        print(
            f"[NAV][ERROR] {ts} run_inference failed "
            f"(consecutive={self._consecutive_failures}): {exc}"
        )

        # 7.6  Set error banner when >= 3 consecutive failures
        if self._consecutive_failures >= 3:
            if not self.error_banner:
                print(
                    f"[NAV][ERROR] {self._consecutive_failures} consecutive inference "
                    "failures — activating error banner"
                )
            self.error_banner = True

        return self._smoother.push(DrivingCommand.BRAKE)

    def handle_inference_success(self) -> None:
        """
        Called by the main loop after a successful ``run_inference`` call.
        Resets the failure counter and clears the error banner.
        """
        if self._consecutive_failures > 0:
            print(
                f"[NAV][INFO] Inference recovered after "
                f"{self._consecutive_failures} consecutive failure(s); "
                "clearing error banner"
            )
        self._consecutive_failures = 0
        self.error_banner = False


# ===========================================================================
# Visual overlay — zone rectangles + active command text
# ===========================================================================

# Human-readable labels for each DrivingCommand used in the overlay.
_CMD_LABELS: dict[DrivingCommand, str] = {
    DrivingCommand.FORWARD:    "^ FORWARD",
    DrivingCommand.TURN_LEFT:  "< LEFT",
    DrivingCommand.REVERSE:    "v REVERSE",
    DrivingCommand.TURN_RIGHT: "> RIGHT",
    DrivingCommand.BRAKE:      "STOP",
}

# BGR fill colours for each ZoneState (used for semi-transparent zone fills).
_ZONE_COLORS: dict[ZoneState, tuple[int, int, int]] = {
    ZoneState.CLEAR:     (0, 200, 0),    # green
    ZoneState.UNCERTAIN: (0, 200, 220),  # yellow (BGR)
    ZoneState.BLOCKED:   (0, 0, 220),    # red
}


def draw_nav_overlay(
    display: np.ndarray,
    depth_u8: np.ndarray,
    zones: list,
    raw_cmd: DrivingCommand,
    smoothed_cmd: DrivingCommand,
    floor_baseline: float,
    error_banner: bool = False,
) -> None:
    """
    Draw the five-zone navigation overlay on *display* in-place.

    8.1  Accepts zones, raw_cmd, smoothed_cmd, and floor_baseline.
    8.2  Draws five zone rectangles with semi-transparent fills:
           green = CLEAR, yellow = UNCERTAIN, red = BLOCKED.
    8.3  Displays the ObstacleScore as a centred text label in each zone.
    8.4  Displays the current FloorBaseline value in the HUD area.
    8.5  Displays the smoothed command as a large centred label above the ROI.
    8.6  Displays the raw (pre-smooth) command as a smaller secondary label.
    8.7  Displays a visible red error banner when error_banner is True.

    Args:
        display:       BGR image to annotate (modified in-place).
        depth_u8:      Uint8 depth map — used only for frame dimensions.
        zones:         list[ZoneInfo] — five zones in left-to-right order.
        raw_cmd:       DrivingCommand before smoothing.
        smoothed_cmd:  DrivingCommand after majority-vote smoothing.
        floor_baseline: Current EMA FloorBaseline value.
        error_banner:  When True, draw a red error banner across the top.
    """
    h, w = depth_u8.shape
    row0 = int(h * NAV_ROW_START)

    # ------------------------------------------------------------------
    # 8.2  Semi-transparent zone fills
    # ------------------------------------------------------------------
    overlay = display.copy()
    zone_width = w // 5
    for i, zone in enumerate(zones):
        x1 = i * zone_width
        x2 = (i + 1) * zone_width if i < 4 else w   # last zone absorbs rounding remainder
        color = _ZONE_COLORS.get(zone.state, (128, 128, 128))
        cv2.rectangle(overlay, (x1, row0), (x2 - 1, h - 1), color, -1)
    cv2.addWeighted(overlay, 0.28, display, 0.72, 0, display)

    # ------------------------------------------------------------------
    # Zone border lines + 8.3  ObstacleScore centred label
    # ------------------------------------------------------------------
    for i, zone in enumerate(zones):
        x1 = i * zone_width
        x2 = (i + 1) * zone_width if i < 4 else w
        cx = (x1 + x2) // 2

        # Zone border
        cv2.rectangle(display, (x1, row0), (x2 - 1, h - 1),
                      (255, 255, 255), 1, cv2.LINE_AA)

        # Zone name (small, near top of zone)
        cv2.putText(
            display, zone.name,
            (x1 + 3, row0 + 14),
            cv2.FONT_HERSHEY_SIMPLEX, 0.32, (220, 220, 220), 1, cv2.LINE_AA,
        )

        # ObstacleScore — centred vertically in the zone, large enough to read
        score_label = f"{zone.obstacle_score:.0f}"
        (sw, sh), _ = cv2.getTextSize(score_label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        sx = cx - sw // 2
        sy = (row0 + h) // 2 + sh // 2
        # Dark outline for contrast against any fill colour
        cv2.putText(display, score_label, (sx, sy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(display, score_label, (sx, sy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # 8.4  FloorBaseline HUD label (bottom-left corner)
    # ------------------------------------------------------------------
    hud_text = f"Floor: {floor_baseline:.1f}"
    cv2.putText(display, hud_text, (4, h - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(display, hud_text, (4, h - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 230, 255), 1, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # 8.5  Smoothed command — large centred label above the ROI
    # ------------------------------------------------------------------
    smooth_label = _CMD_LABELS.get(smoothed_cmd, smoothed_cmd.name)
    (tw, th), _ = cv2.getTextSize(smooth_label, cv2.FONT_HERSHEY_DUPLEX, 1.1, 2)
    tx = (w - tw) // 2
    ty = max(th + 4, row0 - 8)
    cv2.putText(display, smooth_label, (tx, ty),
                cv2.FONT_HERSHEY_DUPLEX, 1.1, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(display, smooth_label, (tx, ty),
                cv2.FONT_HERSHEY_DUPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # 8.6  Raw (pre-smooth) command — smaller secondary label, top-right
    # ------------------------------------------------------------------
    raw_label = "raw: " + _CMD_LABELS.get(raw_cmd, raw_cmd.name)
    (rw, rh), _ = cv2.getTextSize(raw_label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    rx = w - rw - 4
    ry = rh + 4
    cv2.putText(display, raw_label, (rx, ry),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(display, raw_label, (rx, ry),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # 8.7  Red error banner when consecutive inference failures >= 3
    # ------------------------------------------------------------------
    if error_banner:
        banner_h = 22
        banner_overlay = display.copy()
        cv2.rectangle(banner_overlay, (0, 0), (w - 1, banner_h), (0, 0, 200), -1)
        cv2.addWeighted(banner_overlay, 0.75, display, 0.25, 0, display)
        banner_text = "! DEPTH MODEL ERROR — BRAKING !"
        (bw, bh), _ = cv2.getTextSize(banner_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        bx = (w - bw) // 2
        by = (banner_h + bh) // 2
        cv2.putText(display, banner_text, (bx, by),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


# ===========================================================================
# Camera selector — terminal prompt at startup
# ===========================================================================

def _probe_local_cameras() -> list[tuple[int, str]]:
    """
    Probe cv2.VideoCapture indices 0..CAMERA_SCAN_LIMIT-1.
    Returns a list of (index, label) for every index that opens successfully.
    On macOS, Continuity Camera (iPhone) typically appears at index 1 or 2.
    """
    found = []
    for idx in range(CAMERA_SCAN_LIMIT):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            # Try to grab a frame to confirm it's a real source
            ok, _ = cap.read()
            if ok:
                # cv2 doesn't expose a friendly name on all platforms,
                # so we label by index and note the likely Continuity slot.
                label = f"Camera {idx}"
                if idx == 0:
                    label += " (built-in / default)"
                found.append((idx, label))
        cap.release()
    return found


def select_camera():
    """
    Interactive terminal camera picker.

    Probes local webcam indices, then offers URL-based options for:
      - ESP32-CAM  (MJPEG over HTTP)
      - RTSP stream (IP cameras, DroidCam, etc.)
      - Custom HTTP MJPEG URL

    Returns a ready-to-use capture object — either a cv2.VideoCapture
    or an MJPEGCapture instance — both expose .read() and .release().
    """
    print("\n" + "=" * 56)
    print("  Camera Selector")
    print("=" * 56)

    # --- Enumerate local cameras ---
    print("Scanning for local / USB / Continuity cameras…")
    local_cams = _probe_local_cameras()

    options: list[tuple[str, str]] = []   # (display label, source key)

    for idx, label in local_cams:
        options.append((label, f"local:{idx}"))

    # --- Fixed URL-based options ---
    options.append(("ESP32-CAM  (enter IP)", "esp32"))
    options.append(("RTSP stream (enter URL)", "rtsp"))
    options.append(("HTTP MJPEG  (enter URL)", "mjpeg"))

    if not local_cams:
        print("  (no local cameras found)\n")
    else:
        print()

    for i, (label, _) in enumerate(options):
        print(f"  [{i}] {label}")

    print()

    # --- Get user choice ---
    while True:
        raw = input(f"Select source [0–{len(options) - 1}]: ").strip()
        if raw.isdigit() and 0 <= int(raw) < len(options):
            choice_idx = int(raw)
            break
        print(f"  Please enter a number between 0 and {len(options) - 1}.")

    _, source_key = options[choice_idx]

    # --- Build capture object ---
    if source_key.startswith("local:"):
        cam_idx = int(source_key.split(":")[1])
        print(f"\n[INFO] Opening camera index {cam_idx}…")
        cap = cv2.VideoCapture(cam_idx)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAPTURE_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            print(f"[ERROR] Could not open camera {cam_idx}. Exiting.")
            sys.exit(1)
        return cap

    elif source_key == "esp32":
        ip = input("  ESP32-CAM IP (e.g. 192.168.1.100): ").strip()
        url = f"http://{ip}:81/stream"
        print(f"\n[INFO] Connecting to ESP32-CAM: {url}")
        return MJPEGCapture(url)

    elif source_key == "rtsp":
        url = input("  RTSP URL (e.g. rtsp://192.168.1.x:554/stream): ").strip()
        print(f"\n[INFO] Opening RTSP stream: {url}")
        cap = cv2.VideoCapture(url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            print(f"[ERROR] Could not open RTSP stream. Check the URL and try again.")
            sys.exit(1)
        return cap

    elif source_key == "mjpeg":
        url = input("  HTTP MJPEG URL (e.g. http://192.168.1.x:8080/video): ").strip()
        print(f"\n[INFO] Connecting to MJPEG stream: {url}")
        return MJPEGCapture(url)


# ===========================================================================
# Main loop
# ===========================================================================
def main():
    cap = select_camera()

    # 9.1  Instantiate Navigator and KeyController before the capture loop
    navigator  = Navigator()
    controller = KeyController()

    prev_time     = time.perf_counter()
    frame_count   = 0
    last_depth    = None
    last_depth_u8 = None

    # Initialise to safe defaults so draw_nav_overlay always has valid data
    last_zone_infos   = [
        ZoneInfo(name=n, mean_depth=0.0, obstacle_score=0.0, state=ZoneState.UNCERTAIN)
        for n in ("FarLeft", "Left", "Center", "Right", "FarRight")
    ]
    last_raw_cmd      = DrivingCommand.BRAKE
    last_smoothed_cmd = DrivingCommand.BRAKE
    last_floor_base   = 20.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.005)
                continue

            frame_count += 1
            if FLIP_HORIZONTAL and FLIP_VERTICAL:
                frame = cv2.flip(frame, -1)
            elif FLIP_HORIZONTAL:
                frame = cv2.flip(frame, 1)
            elif FLIP_VERTICAL:
                frame = cv2.flip(frame, 0)

            if frame_count % FRAME_SKIP == 0:
                # 7.5  Wrap run_inference in try/except; delegate error handling to Navigator
                try:
                    last_depth, last_depth_u8 = run_inference(frame)
                    navigator.handle_inference_success()
                except Exception as exc:
                    smoothed_err = navigator.handle_inference_error(exc)
                    controller.send(smoothed_err)
                    last_smoothed_cmd = smoothed_err
                    last_raw_cmd      = DrivingCommand.BRAKE

            if last_depth is None:
                time.sleep(0.005)
                continue

            display = last_depth.copy()

            # 9.2  Replace compute_wasd / send_wasd with Navigator.process_frame + KeyController.send
            smoothed_cmd, zone_infos, floor_baseline = navigator.process_frame(last_depth_u8)

            # Recover the raw command from the Navigator's last decide_command result.
            # Navigator exposes the raw command indirectly via the smoother; we reconstruct
            # it by peeking at the smoother buffer's most-recent entry.
            raw_cmd = navigator._smoother._buf[-1] if navigator._smoother._buf else smoothed_cmd

            controller.send(smoothed_cmd)

            # 9.3  Cache results for overlay (keeps overlay valid even on skipped frames)
            last_zone_infos   = zone_infos
            last_raw_cmd      = raw_cmd
            last_smoothed_cmd = smoothed_cmd
            last_floor_base   = floor_baseline

            # 9.3  Pass zone_infos, raw_cmd, smoothed_cmd, and floor_baseline to draw_nav_overlay
            draw_nav_overlay(
                display,
                last_depth_u8,
                last_zone_infos,
                last_raw_cmd,
                last_smoothed_cmd,
                last_floor_base,
                error_banner=navigator.error_banner,
            )

            # HUD
            now = time.perf_counter()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            cv2.putText(display, f"FPS: {fps:.1f}", (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(display, f"{MODEL_ENCODER.upper()} | {MODEL_INPUT_SIZE}px",
                        (8, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

            side = np.hstack([frame, display])
            cv2.imshow("Depth WASD Navigation", side)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        # 9.4  Call KeyController.release_all() in the finally block
        controller.release_all()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
