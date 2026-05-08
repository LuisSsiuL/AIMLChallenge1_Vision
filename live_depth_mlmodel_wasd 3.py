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
# Navigation logic — zone scoring (2-row 7x7 grid)
# ===========================================================================

def compute_zone_scores(roi: np.ndarray) -> tuple:
    """
    Divide the ROI into 2 horizontal rows, each with 7 columns:
    - FAR ROW (top 40%): 7 zones — shows path ahead
    - NEAR ROW (bottom 60%): 7 zones — shows immediate obstacles
    
    Returns:
        (far_zones, near_zones) — two lists of 7 ZoneInfo objects each
    """
    h, w = roi.shape
    
    # Split into far (top 40%) and near (bottom 60%) rows
    far_row_end = int(h * 0.4)
    far_roi = roi[:far_row_end, :]
    near_roi = roi[far_row_end:, :]
    
    # ------------------------------------------------------------------
    # FAR ROW: 7 zones
    # ------------------------------------------------------------------
    far_zones = []
    far_col_edges = [int(w * i / 7) for i in range(8)]
    
    for i in range(7):
        x0 = far_col_edges[i]
        x1 = far_col_edges[i + 1]
        zone_slice = far_roi[:, x0:x1]
        
        mean_depth = float(zone_slice.mean())
        obstacle_score = float(np.clip(mean_depth, 0.0, 255.0))
        
        if obstacle_score > FORWARD_THRESHOLD:
            state = ZoneState.CLEAR
        elif obstacle_score < STUCK_THRESHOLD:
            state = ZoneState.BLOCKED
        else:
            state = ZoneState.UNCERTAIN
        
        far_zones.append(ZoneInfo(
            name=f"F{i+1}",
            mean_depth=mean_depth,
            obstacle_score=obstacle_score,
            state=state,
        ))
    
    # ------------------------------------------------------------------
    # NEAR ROW: 7 zones
    # ------------------------------------------------------------------
    near_zones = []
    near_col_edges = [int(w * i / 7) for i in range(8)]
    
    for i in range(7):
        x0 = near_col_edges[i]
        x1 = near_col_edges[i + 1]
        zone_slice = near_roi[:, x0:x1]
        
        mean_depth = float(zone_slice.mean())
        obstacle_score = float(np.clip(mean_depth, 0.0, 255.0))
        
        if obstacle_score > FORWARD_THRESHOLD:
            state = ZoneState.CLEAR
        elif obstacle_score < STUCK_THRESHOLD:
            state = ZoneState.BLOCKED
        else:
            state = ZoneState.UNCERTAIN
        
        near_zones.append(ZoneInfo(
            name=f"N{i+1}",
            mean_depth=mean_depth,
            obstacle_score=obstacle_score,
            state=state,
        ))
    
    return (far_zones, near_zones)


# ===========================================================================
# Navigation logic — decision tree
# ===========================================================================

def decide_command(
    far_zones: list,
    near_zones: list,
    consecutive_reverse: int,
    escape_phase: int,
    last_command: DrivingCommand = None,
) -> tuple:
    """
    Navigation with 7x7 grid and STRICT forward requirements.
    
    FORWARD only if:
    - Center 3 zones in NEAR are ALL clear (N3, N4, N5)
    - Center 3 zones in FAR are ALL clear (F3, F4, F5)
    - Otherwise, steer toward the best gap
    
    Returns: (raw_command, new_consecutive_reverse, new_escape_phase)
    """
    far_scores = [z.obstacle_score for z in far_zones]
    near_scores = [z.obstacle_score for z in near_zones]
    
    # Center 3 zones (indices 2, 3, 4)
    near_center_3 = [near_scores[2], near_scores[3], near_scores[4]]
    far_center_3 = [far_scores[2], far_scores[3], far_scores[4]]
    
    near_center_min = min(near_center_3)
    near_center_avg = sum(near_center_3) / 3.0
    far_center_min = min(far_center_3)
    far_center_avg = sum(far_center_3) / 3.0
    
    best_near = max(near_scores)
    best_far = max(far_scores)

    if DEBUG_NAVIGATION:
        far_str = " ".join(f"{s:.0f}" for s in far_scores)
        near_str = " ".join(f"{s:.0f}" for s in near_scores)
        print(f"[NAV] FAR:  {far_str}")
        print(f"[NAV] NEAR: {near_str}")
        print(f"[NAV] near_center_3: min={near_center_min:.0f} avg={near_center_avg:.0f}")
        print(f"[NAV] far_center_3: min={far_center_min:.0f} avg={far_center_avg:.0f}")

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
    # 2. Truly stuck (all near zones very low) → REVERSE
    # ------------------------------------------------------------------
    if best_near < STUCK_THRESHOLD:
        new_consecutive = consecutive_reverse + 1
        if new_consecutive > REVERSE_ESCAPE_FRAMES:
            left_avg = sum(near_scores[:3]) / 3
            right_avg = sum(near_scores[4:]) / 3
            
            if left_avg >= right_avg:
                if DEBUG_NAVIGATION:
                    print(f"[NAV] → REVERSE_LEFT (escape start)")
                return (DrivingCommand.REVERSE_LEFT, 0, 5)
            else:
                if DEBUG_NAVIGATION:
                    print(f"[NAV] → REVERSE_RIGHT (escape start)")
                return (DrivingCommand.REVERSE_RIGHT, 0, -5)
        
        if DEBUG_NAVIGATION:
            print(f"[NAV] → REVERSE (stuck: best_near={best_near:.0f}, consecutive={new_consecutive})")
        return (DrivingCommand.REVERSE, new_consecutive, 0)

    consecutive_reverse = 0

    # ------------------------------------------------------------------
    # 3. Check if center is blocked — if so, MUST steer, never go forward
    # ------------------------------------------------------------------
    # Center zones: N3, N4, N5 (indices 2, 3, 4)
    center_is_blocked = any(score < FORWARD_THRESHOLD for score in near_center_3)
    
    if center_is_blocked:
        # Center has obstacle — MUST steer left or right, never forward
        if DEBUG_NAVIGATION:
            print(f"[NAV] Center blocked (scores: {near_center_3}), must steer")
        
        # Find which side is more open
        left_scores = near_scores[:3]   # N1, N2, N3
        right_scores = near_scores[4:]  # N5, N6, N7
        
        left_max = max(left_scores)
        right_max = max(right_scores)
        left_avg = sum(left_scores) / len(left_scores)
        right_avg = sum(right_scores) / len(right_scores)
        
        # Apply hysteresis
        if last_command == DrivingCommand.FORWARD_LEFT:
            left_avg += HYSTERESIS_BONUS
        elif last_command == DrivingCommand.FORWARD_RIGHT:
            right_avg += HYSTERESIS_BONUS
        
        if DEBUG_NAVIGATION:
            print(f"[NAV] Left: max={left_max:.0f} avg={left_avg:.0f}  Right: max={right_max:.0f} avg={right_avg:.0f}")
        
        # Steer toward the more open side
        if left_avg > right_avg + STEER_MIN_DIFF:
            if DEBUG_NAVIGATION:
                print(f"[NAV] → FORWARD_LEFT (center blocked, left more open)")
            return (DrivingCommand.FORWARD_LEFT, 0, 0)
        elif right_avg > left_avg + STEER_MIN_DIFF:
            if DEBUG_NAVIGATION:
                print(f"[NAV] → FORWARD_RIGHT (center blocked, right more open)")
            return (DrivingCommand.FORWARD_RIGHT, 0, 0)
        else:
            # Both sides similar — pick the one with higher max score
            if left_max >= right_max:
                if DEBUG_NAVIGATION:
                    print(f"[NAV] → FORWARD_LEFT (center blocked, left max higher)")
                return (DrivingCommand.FORWARD_LEFT, 0, 0)
            else:
                if DEBUG_NAVIGATION:
                    print(f"[NAV] → FORWARD_RIGHT (center blocked, right max higher)")
                return (DrivingCommand.FORWARD_RIGHT, 0, 0)
    
    # ------------------------------------------------------------------
    # 4. Center is clear — check if FAR is also clear
    # ------------------------------------------------------------------
    near_center_all_clear = near_center_min >= FORWARD_THRESHOLD
    far_center_all_clear = far_center_min >= FORWARD_THRESHOLD
    
    if near_center_all_clear and far_center_all_clear:
        if DEBUG_NAVIGATION:
            print(f"[NAV] → FORWARD (near+far center all clear)")
        return (DrivingCommand.FORWARD, 0, 0)
    
    # Near is clear but far has obstacle — steer preemptively
    if near_center_all_clear and not far_center_all_clear:
        # Check which side of FAR is more open
        far_left_avg = sum(far_scores[:3]) / 3
        far_right_avg = sum(far_scores[4:]) / 3
        
        if far_left_avg > far_right_avg + STEER_MIN_DIFF:
            if DEBUG_NAVIGATION:
                print(f"[NAV] → FORWARD_LEFT (near clear, far obstacle ahead, left better)")
            return (DrivingCommand.FORWARD_LEFT, 0, 0)
        elif far_right_avg > far_left_avg + STEER_MIN_DIFF:
            if DEBUG_NAVIGATION:
                print(f"[NAV] → FORWARD_RIGHT (near clear, far obstacle ahead, right better)")
            return (DrivingCommand.FORWARD_RIGHT, 0, 0)
        else:
            # Far is unclear but near is clear — go forward cautiously
            if DEBUG_NAVIGATION:
                print(f"[NAV] → FORWARD (near clear, far unclear)")
            return (DrivingCommand.FORWARD, 0, 0)

    # ------------------------------------------------------------------
    # 4. Center not fully clear — find best gap and steer
    # ------------------------------------------------------------------
    def find_best_gap(scores):
        """Find widest gap in a score array."""
        gaps = []
        gap_start = None
        gap_sum = 0
        
        for i, score in enumerate(scores):
            if score >= STUCK_THRESHOLD:
                if gap_start is None:
                    gap_start = i
                    gap_sum = score
                else:
                    gap_sum += score
            else:
                if gap_start is not None:
                    gap_width = i - gap_start
                    gap_center = gap_start + gap_width / 2.0
                    gap_avg = gap_sum / gap_width
                    gaps.append((gap_center, gap_width, gap_avg))
                    gap_start = None
                    gap_sum = 0
        
        # Close last gap
        if gap_start is not None:
            gap_width = len(scores) - gap_start
            gap_center = gap_start + gap_width / 2.0
            gap_avg = gap_sum / gap_width
            gaps.append((gap_center, gap_width, gap_avg))
        
        if not gaps:
            return None
        
        # Pick widest gap, then highest average
        return max(gaps, key=lambda g: (g[1], g[2]))
    
    near_gap = find_best_gap(near_scores)
    
    if near_gap is None:
        if DEBUG_NAVIGATION:
            print(f"[NAV] → REVERSE (no near gaps)")
        return (DrivingCommand.REVERSE, 1, 0)
    
    gap_center, gap_width, gap_avg = near_gap
    
    # Apply hysteresis
    if last_command == DrivingCommand.FORWARD_LEFT and gap_center < 3.5:
        gap_center -= 0.3
    elif last_command == DrivingCommand.FORWARD_RIGHT and gap_center > 3.5:
        gap_center += 0.3
    
    if DEBUG_NAVIGATION:
        print(f"[NAV] Near gap: center={gap_center:.1f} width={gap_width} avg={gap_avg:.0f}")

    # ------------------------------------------------------------------
    # 5. Steer toward gap (7 zones: center is 3.0)
    # ------------------------------------------------------------------
    # Zone indices: 0 1 2 3 4 5 6
    # Center is zone 3 (index 3)
    
    # Gap is significantly left (< 2.5)
    if gap_center < 2.5:
        if DEBUG_NAVIGATION:
            print(f"[NAV] → FORWARD_LEFT (gap at {gap_center:.1f})")
        return (DrivingCommand.FORWARD_LEFT, 0, 0)
    
    # Gap is significantly right (> 4.5)
    if gap_center > 4.5:
        if DEBUG_NAVIGATION:
            print(f"[NAV] → FORWARD_RIGHT (gap at {gap_center:.1f})")
        return (DrivingCommand.FORWARD_RIGHT, 0, 0)
    
    # Gap is slightly left (2.5 - 3.0)
    if gap_center < 3.0:
        if DEBUG_NAVIGATION:
            print(f"[NAV] → FORWARD_LEFT (gap slightly left at {gap_center:.1f})")
        return (DrivingCommand.FORWARD_LEFT, 0, 0)
    
    # Gap is slightly right (4.0 - 4.5)
    if gap_center > 4.0:
        if DEBUG_NAVIGATION:
            print(f"[NAV] → FORWARD_RIGHT (gap slightly right at {gap_center:.1f})")
        return (DrivingCommand.FORWARD_RIGHT, 0, 0)
    
    # Gap is centered (3.0 - 4.0) but center zones not all clear
    # This means there's a narrow path — be cautious
    if near_center_avg >= FORWARD_THRESHOLD * 0.8:
        # Center is mostly clear, go forward
        if DEBUG_NAVIGATION:
            print(f"[NAV] → FORWARD (gap centered, mostly clear: avg={near_center_avg:.0f})")
        return (DrivingCommand.FORWARD, 0, 0)
    else:
        # Center has some obstacles, steer toward the better side
        left_avg = sum(near_scores[:3]) / 3
        right_avg = sum(near_scores[4:]) / 3
        
        if left_avg > right_avg + 10:
            if DEBUG_NAVIGATION:
                print(f"[NAV] → FORWARD_LEFT (center unclear, left better: {left_avg:.0f} vs {right_avg:.0f})")
            return (DrivingCommand.FORWARD_LEFT, 0, 0)
        elif right_avg > left_avg + 10:
            if DEBUG_NAVIGATION:
                print(f"[NAV] → FORWARD_RIGHT (center unclear, right better: {right_avg:.0f} vs {left_avg:.0f})")
            return (DrivingCommand.FORWARD_RIGHT, 0, 0)
        else:
            if DEBUG_NAVIGATION:
                print(f"[NAV] → FORWARD (gap centered at {gap_center:.1f}, sides similar)")
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
            (smoothed_command, far_zones, near_zones)
        """
        # Degenerate depth map guard
        if float(depth_u8.std()) < 1e-3:
            print(f"[NAV][WARN] Degenerate depth map detected; emitting BRAKE")
            smoothed = self._smoother.push(DrivingCommand.BRAKE)
            placeholder_far = [
                ZoneInfo(name=f"F{i+1}", mean_depth=0.0, obstacle_score=0.0, state=ZoneState.UNCERTAIN)
                for i in range(7)
            ]
            placeholder_near = [
                ZoneInfo(name=f"N{i+1}", mean_depth=0.0, obstacle_score=0.0, state=ZoneState.UNCERTAIN)
                for i in range(7)
            ]
            return (smoothed, placeholder_far, placeholder_near)

        # Extract ROI
        h, w = depth_u8.shape
        roi_row0 = int(h * NAV_ROW_START)
        roi = depth_u8[roi_row0:, :]

        # Compute zone scores (2 rows)
        far_zones, near_zones = compute_zone_scores(roi)

        # Decide raw command
        raw_cmd, self._consecutive_reverse, self._escape_phase = decide_command(
            far_zones=far_zones,
            near_zones=near_zones,
            consecutive_reverse=self._consecutive_reverse,
            escape_phase=self._escape_phase,
            last_command=self._last_command,
        )

        # Smooth
        smoothed_cmd = self._smoother.push(raw_cmd)
        self._last_command = smoothed_cmd

        return (smoothed_cmd, far_zones, near_zones)


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
    far_zones: list,
    near_zones: list,
    raw_cmd: DrivingCommand,
    smoothed_cmd: DrivingCommand,
) -> None:
    """Draw the 2-row perspective-aware navigation overlay."""
    h, w = depth_u8.shape
    row0 = int(h * NAV_ROW_START)
    roi_h = h - row0
    
    # Split ROI into far (top 40%) and near (bottom 60%)
    far_row_end = row0 + int(roi_h * 0.4)
    
    overlay = display.copy()
    
    # ------------------------------------------------------------------
    # FAR ROW: 7 zones
    # ------------------------------------------------------------------
    far_zone_width = w // 7
    for i, zone in enumerate(far_zones):
        x1 = i * far_zone_width
        x2 = (i + 1) * far_zone_width if i < 6 else w
        color = _ZONE_COLORS.get(zone.state, (128, 128, 128))
        cv2.rectangle(overlay, (x1, row0), (x2 - 1, far_row_end - 1), color, -1)
    
    # ------------------------------------------------------------------
    # NEAR ROW: 7 zones
    # ------------------------------------------------------------------
    near_zone_width = w // 7
    for i, zone in enumerate(near_zones):
        x1 = i * near_zone_width
        x2 = (i + 1) * near_zone_width if i < 6 else w
        color = _ZONE_COLORS.get(zone.state, (128, 128, 128))
        cv2.rectangle(overlay, (x1, far_row_end), (x2 - 1, h - 1), color, -1)
    
    cv2.addWeighted(overlay, 0.28, display, 0.72, 0, display)
    
    # ------------------------------------------------------------------
    # FAR ROW borders + scores
    # ------------------------------------------------------------------
    for i, zone in enumerate(far_zones):
        x1 = i * far_zone_width
        x2 = (i + 1) * far_zone_width if i < 6 else w
        cx = (x1 + x2) // 2
        
        cv2.rectangle(display, (x1, row0), (x2 - 1, far_row_end - 1),
                      (255, 255, 255), 1, cv2.LINE_AA)
        
        # Zone label
        cv2.putText(display, zone.name, (x1 + 2, row0 + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (220, 220, 220), 1, cv2.LINE_AA)
        
        # Score
        score_label = f"{zone.obstacle_score:.0f}"
        (sw, sh), _ = cv2.getTextSize(score_label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        sx = cx - sw // 2
        sy = (row0 + far_row_end) // 2 + sh // 2
        cv2.putText(display, score_label, (sx, sy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(display, score_label, (sx, sy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    
    # ------------------------------------------------------------------
    # NEAR ROW borders + scores
    # ------------------------------------------------------------------
    for i, zone in enumerate(near_zones):
        x1 = i * near_zone_width
        x2 = (i + 1) * near_zone_width if i < 6 else w
        cx = (x1 + x2) // 2
        
        cv2.rectangle(display, (x1, far_row_end), (x2 - 1, h - 1),
                      (255, 255, 255), 1, cv2.LINE_AA)
        
        # Zone label
        cv2.putText(display, zone.name, (x1 + 2, far_row_end + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (220, 220, 220), 1, cv2.LINE_AA)
        
        # Score
        score_label = f"{zone.obstacle_score:.0f}"
        (sw, sh), _ = cv2.getTextSize(score_label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        sx = cx - sw // 2
        sy = (far_row_end + h) // 2 + sh // 2
        cv2.putText(display, score_label, (sx, sy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(display, score_label, (sx, sy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    
    # ------------------------------------------------------------------
    # Command labels
    # ------------------------------------------------------------------
    smooth_label = _CMD_LABELS.get(smoothed_cmd, smoothed_cmd.name)
    (tw, th), _ = cv2.getTextSize(smooth_label, cv2.FONT_HERSHEY_DUPLEX, 1.0, 2)
    tx = (w - tw) // 2
    ty = max(th + 4, row0 - 8)
    cv2.putText(display, smooth_label, (tx, ty),
                cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(display, smooth_label, (tx, ty),
                cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    
    raw_label = "raw: " + _CMD_LABELS.get(raw_cmd, raw_cmd.name)
    (rw, rh), _ = cv2.getTextSize(raw_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    rx = w - rw - 4
    ry = rh + 4
    cv2.putText(display, raw_label, (rx, ry),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(display, raw_label, (rx, ry),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA)


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

    last_far_zones = [
        ZoneInfo(name=f"F{i+1}", mean_depth=0.0, obstacle_score=0.0, state=ZoneState.UNCERTAIN)
        for i in range(7)
    ]
    last_near_zones = [
        ZoneInfo(name=f"N{i+1}", mean_depth=0.0, obstacle_score=0.0, state=ZoneState.UNCERTAIN)
        for i in range(7)
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

            smoothed_cmd, far_zones, near_zones = navigator.process_frame(last_depth_u8)
            raw_cmd = navigator._smoother._buf[-1] if navigator._smoother._buf else smoothed_cmd

            controller.send(smoothed_cmd)

            last_far_zones = far_zones
            last_near_zones = near_zones
            last_raw_cmd = raw_cmd
            last_smoothed_cmd = smoothed_cmd

            draw_nav_overlay(display, last_depth_u8, last_far_zones, last_near_zones, last_raw_cmd, last_smoothed_cmd)

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
