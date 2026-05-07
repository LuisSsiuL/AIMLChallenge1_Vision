#!/Users/christianluisefendy/.pyenv/versions/3.8.10/bin/python
# macOS: re-exec with known-good Python + fork-safety fix before any C++ libs load
import os, sys
_PY = "/Users/christianluisefendy/.pyenv/versions/3.8.10/bin/python"
if sys.executable != _PY or os.environ.get("OBJC_DISABLE_INITIALIZE_FORK_SAFETY") != "YES":
    os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
    os.execv(_PY, [_PY] + sys.argv)

"""Index-finger joystick WASD keymapper using MediaPipe.

Gesture scheme:
  Direction vector from index MCP (landmark 5) → index tip (landmark 8)
  maps to W/A/S/D like a joystick.  Only the index finger is used —
  other fingers can be in any pose.

  Dead zone: finger must be extended past a minimum length to register.

Controls:
  Q    quit
  M    toggle mirror mode
  SPACE pause / resume key output
"""

import json
import os
import sys
import time

import cv2
import mediapipe as mp
import numpy as np

try:
    from pynput.keyboard import Controller as KbController
    _kb = KbController()
    _PYNPUT_OK = True
except Exception as e:
    print(f"[WARN] pynput unavailable ({e}). Key output disabled.")
    _kb = None
    _PYNPUT_OK = False

# ── constants (loaded from joystick_config.json with fallback defaults) ───────
_DEFAULTS = {
    "deadzone_len":    0.10,
    "deadzone_x":      0.14,
    "deadzone_y_neg":  0.006,
    "deadzone_y_pos":  0.120,
    "ema_alpha":       0.50,
    "fist_dist":       0.65,
    "extension_ratio": 1.3,
}

def _load_config():
    cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "joystick_config.json")
    cfg = dict(_DEFAULTS)
    try:
        with open(cfg_path, "r") as f:
            cfg.update({k: float(v) for k, v in json.load(f).items() if k in _DEFAULTS})
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
        print(f"[INFO] joystick_config.json not loaded ({e}); using defaults.")
    return cfg

_CFG = _load_config()
DEADZONE_LEN    = _CFG["deadzone_len"]
DEADZONE_X      = _CFG["deadzone_x"]
DEADZONE_Y_NEG  = _CFG["deadzone_y_neg"]
DEADZONE_Y_POS  = _CFG["deadzone_y_pos"]
EMA_ALPHA       = _CFG["ema_alpha"]
FIST_DIST       = _CFG["fist_dist"]
EXTENSION_RATIO = _CFG["extension_ratio"]

# ── MediaPipe setup ───────────────────────────────────────────────────────────
mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles  = mp.solutions.drawing_styles


def get_mp_landmarks(mp_res, H, W):
    if not mp_res.multi_hand_landmarks:
        return None
    lms = mp_res.multi_hand_landmarks[0].landmark
    return np.array([[lm.x * W, lm.y * H] for lm in lms], dtype=np.float32)


def hand_size(lms):
    xs, ys = lms[:, 0], lms[:, 1]
    return max(xs.max() - xs.min(), ys.max() - ys.min()) + 1e-6


def is_neutral_pose(lms):
    """Return True if hand is a closed fist → deadzone.

    Knuckle check removed: pointing index down is geometrically identical to
    a knuckle pose (MCP above PIP, tip below PIP), causing false positives.
    Wrist-distance fist check is direction-agnostic: a pointed finger in any
    direction keeps the index tip far from the wrist, so it never fires.
    """
    fingers = [(5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20)]
    size = hand_size(lms)
    tucked = sum(np.linalg.norm(lms[tip] - lms[0]) / size < FIST_DIST
                 for _, _, tip in fingers)
    return tucked >= 4


def compute_keys(lms):
    """
    Index MCP (5) → index tip (8) = joystick vector.
    Knuckle / fist pose → deadzone regardless of vector.
    Normalise by hand_size. Outside deadzone: dominant+secondary axis → WASD.
    Returns (keys: set, vec_norm: ndarray, active: bool).
    """
    if is_neutral_pose(lms):
        return set(), np.zeros(2), False

    mcp = lms[5]
    tip = lms[8]

    # Index must be clearly extended: tip must be farther from wrist than MCP.
    # Catches relaxed/splayed hand where finger isn't truly pointing.
    mcp_wrist_dist = np.linalg.norm(mcp - lms[0])
    tip_wrist_dist = np.linalg.norm(tip - lms[0])
    if tip_wrist_dist < mcp_wrist_dist * EXTENSION_RATIO:
        return set(), np.zeros(2), False

    vec = tip - mcp                             # raw pixel vector
    scale = hand_size(lms)
    vec_n = vec / scale                         # normalised, ~[-1, 1] each axis

    mag = np.linalg.norm(vec_n)
    if mag < DEADZONE_LEN:
        return set(), vec_n, False

    keys = set()
    if vec_n[1] < -DEADZONE_Y_NEG:
        keys.add('w')
    elif vec_n[1] > DEADZONE_Y_POS:
        keys.add('s')
    if vec_n[0] < -DEADZONE_X:
        keys.add('a')
    elif vec_n[0] > DEADZONE_X:
        keys.add('d')

    return keys, vec_n, True


# ── key press/release ─────────────────────────────────────────────────────────
_pressed: set = set()

def apply_keys(desired: set):
    if not _PYNPUT_OK or _kb is None:
        return
    for k in _pressed - desired:
        try:
            _kb.release(k)
        except Exception:
            pass
    for k in desired - _pressed:
        try:
            _kb.press(k)
        except Exception:
            pass
    _pressed.clear()
    _pressed.update(desired)


def release_all():
    if _PYNPUT_OK and _kb:
        for k in list(_pressed):
            try:
                _kb.release(k)
            except Exception:
                pass
    _pressed.clear()


# ── HUD ───────────────────────────────────────────────────────────────────────
KEY_LABEL = {
    frozenset():          "DEADZONE",
    frozenset({'w'}):     "W ↑",
    frozenset({'s'}):     "S ↓",
    frozenset({'a'}):     "A ←",
    frozenset({'d'}):     "D →",
    frozenset({'w','a'}): "W+A ↖",
    frozenset({'w','d'}): "W+D ↗",
    frozenset({'s','a'}): "S+A ↙",
    frozenset({'s','d'}): "S+D ↘",
}

def draw_hud(frame, lms, keys, vec_n, active, fps, paused, W, H):
    # draw joystick vector from index MCP
    if lms is not None:
        mcp = lms[5].astype(int)
        tip = lms[8].astype(int)

        # guide circle centred on MCP
        radius = int(hand_size(lms) * DEADZONE_LEN)
        cv2.circle(frame, tuple(mcp), max(radius, 8), (80, 80, 80), 1)

        # vector arrow MCP → tip
        color = (0, 255, 0) if active else (100, 100, 100)
        cv2.arrowedLine(frame, tuple(mcp), tuple(tip), color, 3, tipLength=0.3)

        # highlight index tip
        cv2.circle(frame, tuple(tip), 10, (0, 255, 255), -1)

    label = KEY_LABEL.get(frozenset(keys), str(keys))
    color = (0, 255, 0) if active else (180, 180, 180)
    cv2.putText(frame, label, (16, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
    cv2.putText(frame, f"FPS {fps:.0f}", (16, H - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    if paused:
        cv2.putText(frame, "PAUSED (SPACE to resume)", (16, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
    if not _PYNPUT_OK:
        cv2.putText(frame, "pynput missing – keys not sent", (16, H - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 100, 255), 1)


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    tracker = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.exit("Could not open camera.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("="*55)
    print("  Index-finger joystick WASD keymapper")
    print("  Point UP=W  DOWN=S  LEFT=A  RIGHT=D")
    print("  Neutral (finger relaxed/vertical) = no key")
    print("  Q=quit  M=mirror  SPACE=pause")
    print("="*55)
    if not _PYNPUT_OK:
        print("[WARN] Key output disabled. Install pynput + grant Accessibility.")
    print()

    mirror   = True
    paused   = False
    prev_lms = None
    last_t   = time.time()
    fps      = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if mirror:
                frame = cv2.flip(frame, 1)

            H, W = frame.shape[:2]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_res    = tracker.process(frame_rgb)
            lms       = get_mp_landmarks(mp_res, H, W)

            # agile EMA filter
            if lms is None:
                prev_lms = None
            else:
                if prev_lms is not None and prev_lms.shape == lms.shape:
                    lms = EMA_ALPHA * prev_lms + (1 - EMA_ALPHA) * lms
                prev_lms = lms

            keys, vec_n, active = set(), np.zeros(2), False
            if lms is not None and not paused:
                keys, vec_n, active = compute_keys(lms)
                apply_keys(keys)
            else:
                release_all()

            if mp_res.multi_hand_landmarks:
                for hl in mp_res.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hl, mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style(),
                    )

            now    = time.time()
            fps    = 0.9 * fps + 0.1 * (1.0 / max(now - last_t, 1e-6))
            last_t = now

            draw_hud(frame, lms, keys, vec_n, active, fps, paused, W, H)
            cv2.imshow("Index Joystick WASD", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('m'):
                mirror = not mirror
            elif key == ord(' '):
                paused = not paused
                if paused:
                    release_all()

    finally:
        release_all()
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
