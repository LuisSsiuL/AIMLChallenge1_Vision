import os
import time
import threading
from collections import deque
from enum import Enum
from dataclasses import dataclass

import cv2
import numpy as np
import urllib.request
from pynput.keyboard import Controller

from depth_anything_v2_coreml import DepthAnythingV2CoreML

# ---------------------------------------------------------------------------
# ESP32-CAM + Depth Anything V2 CoreML — WASD Navigation
#
# Uses CoreML .mlpackage for faster inference on macOS (Apple Silicon / Intel).
# Same navigation logic as live_depth_wasd.py but with CoreML backend.
#
# Requires:  pip install coremltools pynput
# macOS note: grant Accessibility permission to your Terminal app in
#             System Preferences > Privacy & Security > Accessibility.
# ---------------------------------------------------------------------------

# ===========================================================================
# CONFIGURATION
# ===========================================================================

# --- Source ---
CAMERA_SCAN_LIMIT = 6   # probe indices 0..N-1 for local/USB/Continuity cameras

# --- CoreML Model ---
MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "DepthAnythingV2SmallF16.mlpackage",
)

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
FORWARD_THRESHOLD     = 120    # Center score above this → safe to go straight
STUCK_THRESHOLD       = 60     # All zones below this → truly stuck, REVERSE
STEER_MIN_DIFF        = 15     # Min score difference between sides to prefer one over the other
HYSTERESIS_BONUS      = 10     # Score bonus for continuing current direction
NAV_ROW_START         = 0.30   # top of ROI as fraction of frame height (ignore sky/background)
CMD_SMOOTH_FRAMES     = 3      # majority-vote window size (lower = faster reaction)
REVERSE_ESCAPE_FRAMES = 8      # consecutive stuck frames before triggering escape manoeuvre
DEBUG_NAVIGATION      = False  # Set to True to print zone scores and decisions

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

def run_inference(frame_bgr: np.ndarray, model: DepthAnythingV2CoreML):
    """
    Run CoreML depth inference on a BGR frame.
    Returns (colorized_bgr, depth_u8) both resized to original frame dims.
    """
    h, w = frame_bgr.shape[:2]

    # Get raw depth from CoreML model
    depth = model.predict_depth(frame_bgr)  # float32 HxW

    # Normalize to 0-255
    d_min, d_max = float(depth.min()), float(depth.max())
    if d_max > d_min:
        depth_u8 = ((depth - d_min) / (d_max - d_min) * 255.0).astype(np.uint8)
    else:
        depth_u8 = np.zeros_like(depth, dtype=np.uint8)

    # Invert: low values = close/obstacle, high values = far/open
    depth_u8 = 255 - depth_u8

    # Apply gamma for display
    DISPLAY_GAMMA = 0.6
    depth_display = (np.power(depth_u8 / 255.0, DISPLAY_GAMMA) * 255).astype(np.uint8)

    # Colorize
    colorized = cv2.applyColorMap(depth_display, COLORMAP)

    # Resize to original frame size
    if colorized.shape[:2] != (h, w):
        colorized = cv2.resize(colorized, (w, h), interpolation=cv2.INTER_LINEAR)
        depth_u8  = cv2.resize(depth_u8,  (w, h), interpolation=cv2.INTER_LINEAR)

    return colorized, depth_u8


# ===========================================================================
# Navigation data types
# ===========================================================================

class ZoneState(Enum):
    """Classification of a NavigationZone based on its ObstacleScore."""
    CLEAR     = "clear"
    UNCERTAIN = "uncertain"
    BLOCKED   = "blocked"


class DrivingCommand(Enum):
    """Driving commands mapped to their pynput key characters."""
    FORWARD         = "w"      # forward only
    FORWARD_LEFT    = "wa"     # forward + steer left
    FORWARD_RIGHT   = "wd"     # forward + steer right
    REVERSE         = "s"      # reverse
    REVERSE_LEFT    = "sa"     # reverse + steer left
    REVERSE_RIGHT   = "sd"     # reverse + steer right
    BRAKE           = ""       # stop


@dataclass
class ZoneInfo:
    """Per-zone assessment produced by compute_zone_scores each frame."""
    name:           str
    mean_depth:     float
    obstacle_score: float
    state:          ZoneState


# ===========================================================================
# Navigation logic — zone scoring
# ===========================================================================

ZONE_NAMES = ("FarLeft", "Left", "Center", "Right", "FarRight")


def compute_zone_scores(roi: np.ndarray) -> list:
    """
    Divide the ROI into five equal-width columns and compute obstacle scores.
    
    Returns:
        List of five ZoneInfo objects in left-to-right order.
    """
    h, w = roi.shape
    col_edges = [int(w * i / 5) for i in range(6)]

    zones: list = []
    for i, name in enumerate(ZONE_NAMES):
        x0 = col_edges[i]
        x1 = col_edges[i + 1]
        zone_slice = roi[:, x0:x1]

        mean_depth = float(zone_slice.mean())
        obstacle_score = float(np.clip(mean_depth, 0.0, 255.0))

        # Classify: high score = CLEAR (open), low score = BLOCKED (obstacle)
        if obstacle_score > FORWARD_THRESHOLD:
            state = ZoneState.CLEAR
        elif obstacle_score < STUCK_THRESHOLD:
            state = ZoneState.BLOCKED
        else:
            state = ZoneState.UNCERTAIN

        zones.append(ZoneInfo(
            name=name,
            mean_depth=mean_depth,
            obstacle_score=obstacle_score,
            state=state,
        ))

    return zones


# ===========================================================================
# Navigation logic — decision tree
# ===========================================================================

def decide_command(
    zones: list,
    consecutive_reverse: int,
    escape_phase: int,
    last_command: DrivingCommand = None,
) -> tuple:
    """
    Navigation with forward bias: always moving, steer left/right while going forward.
    
    Philosophy:
    - Default = FORWARD (W)
    - Obstacle on left → FORWARD_RIGHT (W+D)
    - Obstacle on right → FORWARD_LEFT (W+A)
    - Truly stuck (all zones < STUCK_THRESHOLD) → REVERSE
    
    Hysteresis: prefer continuing current direction to avoid oscillation.

    Returns: (raw_command, new_consecutive_reverse, new_escape_phase)
    """
    far_left, left, center, right, far_right = zones

    # Weighted side scores: far zones count more (they show the path ahead)
    left_score  = (far_left.obstacle_score * 0.6 + left.obstacle_score * 0.4)
    right_score = (far_right.obstacle_score * 0.6 + right.obstacle_score * 0.4)
    
    # Apply hysteresis bonus to current direction
    if last_command == DrivingCommand.FORWARD_LEFT:
        left_score += HYSTERESIS_BONUS
    elif last_command == DrivingCommand.FORWARD_RIGHT:
        right_score += HYSTERESIS_BONUS
    
    best_score = max(far_left.obstacle_score, left.obstacle_score,
                     center.obstacle_score,
                     right.obstacle_score, far_right.obstacle_score)

    if DEBUG_NAVIGATION:
        print(
            f"[NAV] FL={far_left.obstacle_score:.0f} L={left.obstacle_score:.0f} "
            f"C={center.obstacle_score:.0f} R={right.obstacle_score:.0f} "
            f"FR={far_right.obstacle_score:.0f}  "
            f"left_w={left_score:.0f} right_w={right_score:.0f} best={best_score:.0f}  "
            f"[high=open, low=close]"
        )

    # ------------------------------------------------------------------
    # 1. Escape manoeuvre (if active)
    # ------------------------------------------------------------------
    if escape_phase != 0:
        if escape_phase > 0:
            new_escape_phase = escape_phase - 1
            if new_escape_phase == 0:
                new_escape_phase = -5
            if DEBUG_NAVIGATION:
                print(f"[NAV] → REVERSE_LEFT (escape {escape_phase})")
            return (DrivingCommand.REVERSE_LEFT, 0, new_escape_phase)
        else:
            new_escape_phase = escape_phase + 1
            if new_escape_phase == 0:
                if DEBUG_NAVIGATION:
                    print(f"[NAV] → FORWARD (escape done)")
                return (DrivingCommand.FORWARD, 0, 0)
            if DEBUG_NAVIGATION:
                print(f"[NAV] → REVERSE_RIGHT (escape {escape_phase})")
            return (DrivingCommand.REVERSE_RIGHT, 0, new_escape_phase)

    # ------------------------------------------------------------------
    # 2. Truly stuck (all zones very low) → REVERSE
    # ------------------------------------------------------------------
    if best_score < STUCK_THRESHOLD:
        new_consecutive = consecutive_reverse + 1
        if new_consecutive > REVERSE_ESCAPE_FRAMES:
            # Start escape: reverse+turn toward the less-blocked side
            if left_score >= right_score:
                if DEBUG_NAVIGATION:
                    print(f"[NAV] → REVERSE_LEFT (escape start, left_w={left_score:.0f})")
                return (DrivingCommand.REVERSE_LEFT, 0, 5)
            else:
                if DEBUG_NAVIGATION:
                    print(f"[NAV] → REVERSE_RIGHT (escape start, right_w={right_score:.0f})")
                return (DrivingCommand.REVERSE_RIGHT, 0, -5)
        
        # Just reverse straight back
        if DEBUG_NAVIGATION:
            print(f"[NAV] → REVERSE (stuck: best={best_score:.0f} < {STUCK_THRESHOLD}, consecutive={new_consecutive})")
        return (DrivingCommand.REVERSE, new_consecutive, 0)

    # Reset reverse counter when not stuck
    consecutive_reverse = 0

    # ------------------------------------------------------------------
    # 3. Center is very open → FORWARD straight
    # ------------------------------------------------------------------
    if center.obstacle_score >= FORWARD_THRESHOLD:
        if DEBUG_NAVIGATION:
            print(f"[NAV] → FORWARD (center={center.obstacle_score:.0f} >= {FORWARD_THRESHOLD})")
        return (DrivingCommand.FORWARD, 0, 0)

    # ------------------------------------------------------------------
    # 4. Center not ideal — steer while moving forward
    # ------------------------------------------------------------------
    score_diff = abs(left_score - right_score)
    
    if left_score > right_score + STEER_MIN_DIFF:
        if DEBUG_NAVIGATION:
            print(f"[NAV] → FORWARD_LEFT (left_w={left_score:.0f} > right_w={right_score:.0f}, diff={score_diff:.0f})")
        return (DrivingCommand.FORWARD_LEFT, 0, 0)

    if right_score > left_score + STEER_MIN_DIFF:
        if DEBUG_NAVIGATION:
            print(f"[NAV] → FORWARD_RIGHT (right_w={right_score:.0f} > left_w={left_score:.0f}, diff={score_diff:.0f})")
        return (DrivingCommand.FORWARD_RIGHT, 0, 0)

    # Scores are close — use far zones as tiebreaker
    if far_left.obstacle_score > far_right.obstacle_score + 5:
        if DEBUG_NAVIGATION:
            print(f"[NAV] → FORWARD_LEFT (tie-break: far_left={far_left.obstacle_score:.0f})")
        return (DrivingCommand.FORWARD_LEFT, 0, 0)

    if far_right.obstacle_score > far_left.obstacle_score + 5:
        if DEBUG_NAVIGATION:
            print(f"[NAV] → FORWARD_RIGHT (tie-break: far_right={far_right.obstacle_score:.0f})")
        return (DrivingCommand.FORWARD_RIGHT, 0, 0)

    # Truly symmetric — go straight
    if DEBUG_NAVIGATION:
        print(f"[NAV] → FORWARD (symmetric, center={center.obstacle_score:.0f})")
    return (DrivingCommand.FORWARD, 0, 0)


# ===========================================================================
# Command smoothing
# ===========================================================================

class Smoother:
    """Sliding-window majority-vote smoother for DrivingCommands."""

    def __init__(self, window: int):
        self._buf: deque = deque(maxlen=window)

    def push(self, cmd: DrivingCommand) -> DrivingCommand:
        """Add raw command to the window and return the smoothed command."""
        self._buf.append(cmd)

        if len(self._buf) < self._buf.maxlen:
            return cmd

        brake_count = sum(1 for c in self._buf if c == DrivingCommand.BRAKE)
        if brake_count > len(self._buf) / 3:
            return DrivingCommand.BRAKE

        counts: dict = {}
        for c in self._buf:
            counts[c] = counts.get(c, 0) + 1

        max_count = max(counts.values())
        tied = {c for c, n in counts.items() if n == max_count}

        for c in reversed(self._buf):
            if c in tied:
                return c

        return cmd


class KeyController:
    """Translates a DrivingCommand into held WASD keypresses via pynput."""

    def __init__(self):
        self._last: DrivingCommand = DrivingCommand.BRAKE
        self._kb: Controller = Controller()
        self._pressed_keys: set = set()  # Track currently pressed keys

    def send(self, cmd: DrivingCommand) -> None:
        """Release old keys and press new keys for the command."""
        if cmd == self._last:
            return

        # Release all currently pressed keys
        for key in self._pressed_keys:
            self._kb.release(key)
        self._pressed_keys.clear()

        # Press new keys based on command
        keys_to_press = []
        if cmd == DrivingCommand.FORWARD:
            keys_to_press = ['w']
        elif cmd == DrivingCommand.FORWARD_LEFT:
            keys_to_press = ['w', 'a']
        elif cmd == DrivingCommand.FORWARD_RIGHT:
            keys_to_press = ['w', 'd']
        elif cmd == DrivingCommand.REVERSE:
            keys_to_press = ['s']
        elif cmd == DrivingCommand.REVERSE_LEFT:
            keys_to_press = ['s', 'a']
        elif cmd == DrivingCommand.REVERSE_RIGHT:
            keys_to_press = ['s', 'd']
        # BRAKE = no keys

        for key in keys_to_press:
            self._kb.press(key)
            self._pressed_keys.add(key)

        self._last = cmd

    def release_all(self) -> None:
        """Release any currently held keys and reset state to BRAKE."""
        for key in self._pressed_keys:
            self._kb.release(key)
        self._pressed_keys.clear()
        self._last = DrivingCommand.BRAKE


# ===========================================================================
# Navigator
# ===========================================================================

class Navigator:
    """Stateful wrapper around the navigation pipeline."""

    def __init__(self):
        self._smoother: Smoother = Smoother(CMD_SMOOTH_FRAMES)
        self._consecutive_reverse: int = 0
        self._escape_phase: int = 0
        self._last_command: DrivingCommand = DrivingCommand.BRAKE

    def process_frame(self, depth_u8: np.ndarray) -> tuple:
        """
        Run the full navigation pipeline on a single depth frame.
        
        Returns:
            (smoothed_command, zone_infos)
        """
        # Degenerate depth map guard
        if float(depth_u8.std()) < 1e-3:
            print(f"[NAV][WARN] Degenerate depth map detected; emitting BRAKE")
            smoothed = self._smoother.push(DrivingCommand.BRAKE)
            placeholder_zones = [
                ZoneInfo(name=n, mean_depth=0.0, obstacle_score=0.0, state=ZoneState.UNCERTAIN)
                for n in ZONE_NAMES
            ]
            return (smoothed, placeholder_zones)

        # Extract ROI
        h, w = depth_u8.shape
        roi_row0 = int(h * NAV_ROW_START)
        roi = depth_u8[roi_row0:, :]

        # Compute zone scores
        zone_infos = compute_zone_scores(roi)

        # Decide raw command (with hysteresis based on last command)
        raw_cmd, self._consecutive_reverse, self._escape_phase = decide_command(
            zones=zone_infos,
            consecutive_reverse=self._consecutive_reverse,
            escape_phase=self._escape_phase,
            last_command=self._last_command,
        )

        # Smooth
        smoothed_cmd = self._smoother.push(raw_cmd)
        self._last_command = smoothed_cmd  # Track for next frame

        return (smoothed_cmd, zone_infos)


# ===========================================================================
# Visual overlay
# ===========================================================================

_CMD_LABELS: dict[DrivingCommand, str] = {
    DrivingCommand.FORWARD:       "↑ FORWARD",
    DrivingCommand.FORWARD_LEFT:  "↖ FWD+LEFT",
    DrivingCommand.FORWARD_RIGHT: "↗ FWD+RIGHT",
    DrivingCommand.REVERSE:       "↓ REVERSE",
    DrivingCommand.REVERSE_LEFT:  "↙ REV+LEFT",
    DrivingCommand.REVERSE_RIGHT: "↘ REV+RIGHT",
    DrivingCommand.BRAKE:         "■ STOP",
}

_ZONE_COLORS: dict[ZoneState, tuple[int, int, int]] = {
    ZoneState.CLEAR:     (0, 200, 0),
    ZoneState.UNCERTAIN: (0, 200, 220),
    ZoneState.BLOCKED:   (0, 0, 220),
}


def draw_nav_overlay(
    display: np.ndarray,
    depth_u8: np.ndarray,
    zones: list,
    raw_cmd: DrivingCommand,
    smoothed_cmd: DrivingCommand,
) -> None:
    """Draw the five-zone navigation overlay on display in-place."""
    h, w = depth_u8.shape
    row0 = int(h * NAV_ROW_START)

    # Semi-transparent zone fills
    overlay = display.copy()
    zone_width = w // 5
    for i, zone in enumerate(zones):
        x1 = i * zone_width
        x2 = (i + 1) * zone_width if i < 4 else w
        color = _ZONE_COLORS.get(zone.state, (128, 128, 128))
        cv2.rectangle(overlay, (x1, row0), (x2 - 1, h - 1), color, -1)
    cv2.addWeighted(overlay, 0.28, display, 0.72, 0, display)

    # Zone borders + scores
    for i, zone in enumerate(zones):
        x1 = i * zone_width
        x2 = (i + 1) * zone_width if i < 4 else w
        cx = (x1 + x2) // 2

        cv2.rectangle(display, (x1, row0), (x2 - 1, h - 1),
                      (255, 255, 255), 1, cv2.LINE_AA)

        cv2.putText(display, zone.name, (x1 + 3, row0 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (220, 220, 220), 1, cv2.LINE_AA)

        score_label = f"{zone.obstacle_score:.0f}"
        (sw, sh), _ = cv2.getTextSize(score_label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        sx = cx - sw // 2
        sy = (row0 + h) // 2 + sh // 2
        cv2.putText(display, score_label, (sx, sy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(display, score_label, (sx, sy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    # Smoothed command
    smooth_label = _CMD_LABELS.get(smoothed_cmd, smoothed_cmd.name)
    (tw, th), _ = cv2.getTextSize(smooth_label, cv2.FONT_HERSHEY_DUPLEX, 1.1, 2)
    tx = (w - tw) // 2
    ty = max(th + 4, row0 - 8)
    cv2.putText(display, smooth_label, (tx, ty),
                cv2.FONT_HERSHEY_DUPLEX, 1.1, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(display, smooth_label, (tx, ty),
                cv2.FONT_HERSHEY_DUPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)

    # Raw command
    raw_label = "raw: " + _CMD_LABELS.get(raw_cmd, raw_cmd.name)
    (rw, rh), _ = cv2.getTextSize(raw_label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    rx = w - rw - 4
    ry = rh + 4
    cv2.putText(display, raw_label, (rx, ry),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(display, raw_label, (rx, ry),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)


# ===========================================================================
# Camera selector
# ===========================================================================

def _probe_local_cameras() -> list[tuple[int, str]]:
    """Probe cv2.VideoCapture indices 0..CAMERA_SCAN_LIMIT-1."""
    found = []
    for idx in range(CAMERA_SCAN_LIMIT):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                label = f"Camera {idx}"
                if idx == 0:
                    label += " (built-in / default)"
                found.append((idx, label))
        cap.release()
    return found


def select_camera():
    """Interactive terminal camera picker."""
    print("\n" + "=" * 56)
    print("  Camera Selector")
    print("=" * 56)

    print("Scanning for local / USB / Continuity cameras…")
    local_cams = _probe_local_cameras()

    options: list[tuple[str, str]] = []

    for idx, label in local_cams:
        options.append((label, f"local:{idx}"))

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

    while True:
        raw = input(f"Select source [0–{len(options) - 1}]: ").strip()
        if raw.isdigit() and 0 <= int(raw) < len(options):
            choice_idx = int(raw)
            break
        print(f"  Please enter a number between 0 and {len(options) - 1}.")

    _, source_key = options[choice_idx]

    if source_key.startswith("local:"):
        cam_idx = int(source_key.split(":")[1])
        print(f"\n[INFO] Opening camera index {cam_idx}…")
        cap = cv2.VideoCapture(cam_idx)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAPTURE_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            print(f"[ERROR] Could not open camera {cam_idx}. Exiting.")
            exit(1)
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
            exit(1)
        return cap

    elif source_key == "mjpeg":
        url = input("  HTTP MJPEG URL (e.g. http://192.168.1.x:8080/video): ").strip()
        print(f"\n[INFO] Connecting to MJPEG stream: {url}")
        return MJPEGCapture(url)


# ===========================================================================
# Main loop
# ===========================================================================
def main():
    print("[INFO] Loading CoreML depth model...")
    depth_model = DepthAnythingV2CoreML(MODEL_PATH, colormap=COLORMAP)
    print(f"[INFO] Model ready: {MODEL_PATH}")
    print(f"       input={depth_model.input_width}x{depth_model.input_height}")

    cap = select_camera()

    navigator  = Navigator()
    controller = KeyController()

    prev_time     = time.perf_counter()
    frame_count   = 0
    last_depth    = None
    last_depth_u8 = None

    last_zone_infos   = [
        ZoneInfo(name=n, mean_depth=0.0, obstacle_score=0.0, state=ZoneState.UNCERTAIN)
        for n in ZONE_NAMES
    ]
    last_raw_cmd      = DrivingCommand.BRAKE
    last_smoothed_cmd = DrivingCommand.BRAKE

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
                try:
                    last_depth, last_depth_u8 = run_inference(frame, depth_model)
                except Exception as exc:
                    print(f"[ERROR] Inference failed: {exc}")
                    controller.send(DrivingCommand.BRAKE)
                    continue

            if last_depth is None:
                time.sleep(0.005)
                continue

            display = last_depth.copy()

            smoothed_cmd, zone_infos = navigator.process_frame(last_depth_u8)
            raw_cmd = navigator._smoother._buf[-1] if navigator._smoother._buf else smoothed_cmd

            controller.send(smoothed_cmd)

            last_zone_infos   = zone_infos
            last_raw_cmd      = raw_cmd
            last_smoothed_cmd = smoothed_cmd

            draw_nav_overlay(display, last_depth_u8, last_zone_infos, last_raw_cmd, last_smoothed_cmd)

            # HUD
            now = time.perf_counter()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            cv2.putText(display, f"FPS: {fps:.1f}", (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(display, "CoreML (DepthAnythingV2SmallF16)",
                        (8, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

            side = np.hstack([frame, display])
            cv2.imshow("Depth WASD Navigation (CoreML)", side)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        controller.release_all()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
