#!/Users/christianluisefendy/.pyenv/versions/3.8.10/bin/python
# macOS: re-exec with known-good Python + fork-safety fix before any C++ libs load
import os, sys
_PY = "/Users/christianluisefendy/.pyenv/versions/3.8.10/bin/python"
if sys.executable != _PY or os.environ.get("OBJC_DISABLE_INITIALIZE_FORK_SAFETY") != "YES":
    os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
    os.execv(_PY, [_PY] + sys.argv)

"""Hand-gesture WASD keymapper using MediaPipe.

Gesture scheme (position-based):
  Hand centroid offset from frame center → W/A/S/D (held while active)
  Pinch (thumb tip ↔ index tip < 5 % of hand size) → emergency stop (all keys released)

Controls:
  Q    quit
  M    toggle mirror mode
  SPACE pause / resume key output
"""

import sys
import time

import cv2
import mediapipe as mp
import numpy as np

try:
    from pynput.keyboard import Controller as KbController, Key
    _kb = KbController()
    _PYNPUT_OK = True
except Exception as e:
    print(f"[WARN] pynput unavailable ({e}). Key output disabled.")
    _kb = None
    _PYNPUT_OK = False

# ── constants ─────────────────────────────────────────────────────────────────
DEADZONE      = 0.06    # normalised offset below which no key fires
KNUCKLE_COUNT = 3       # min fingers that must show knuckle pose to trigger stop
EMA_ALPHA     = 0.50    # agile filter weight (prev frame); higher = smoother
PALM_ANCHORS  = [0, 5, 9, 13, 17]   # landmark indices used for centroid

# ── helpers ───────────────────────────────────────────────────────────────────
mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles  = mp.solutions.drawing_styles


def get_mp_landmarks(mp_res, H, W):
    if not mp_res.multi_hand_landmarks:
        return None
    lms = mp_res.multi_hand_landmarks[0].landmark
    return np.array([[lm.x * W, lm.y * H] for lm in lms], dtype=np.float32)


def is_knuckle(lms):
    """Knuckle gesture: MCP raised above PIP, tip curled below PIP (y down = larger).
    Detects fingers bent ~90° at MCP — knuckles forward, tips tucked."""
    # per-finger: (mcp, pip, tip)
    fingers = [(5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20)]
    knuckled = sum(
        lms[mcp][1] < lms[pip][1] and lms[tip][1] > lms[pip][1]
        for mcp, pip, tip in fingers
    )
    return knuckled >= KNUCKLE_COUNT


def compute_keys(lms, W, H):
    """Return set of WASD chars to press, or empty set for stop/neutral."""
    if is_knuckle(lms):
        return set(), True   # knuckle → emergency stop

    centroid = lms[PALM_ANCHORS].mean(axis=0)   # (cx, cy) in pixels
    dx = (centroid[0] - W / 2) / W              # normalised, range ~[-0.5, 0.5]
    dy = (centroid[1] - H / 2) / H

    keys = set()
    if abs(dy) > DEADZONE:
        keys.add('w' if dy < 0 else 's')   # hand up = W (forward), down = S
    if abs(dx) > DEADZONE:
        keys.add('d' if dx > 0 else 'a')   # right = D, left = A
    return keys, False


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


# ── HUD helpers ───────────────────────────────────────────────────────────────
KEY_LABEL = {frozenset(): "NEUTRAL", frozenset({'w'}): "W ↑",
             frozenset({'s'}): "S ↓", frozenset({'a'}): "A ←",
             frozenset({'d'}): "D →", frozenset({'w','a'}): "W+A ↖",
             frozenset({'w','d'}): "W+D ↗", frozenset({'s','a'}): "S+A ↙",
             frozenset({'s','d'}): "S+D ↘"}

def draw_hud(frame, lms, keys, pinching, fps, paused, W, H):
    cx, cy = int(W / 2), int(H / 2)
    dz_px = int(DEADZONE * W)
    cv2.circle(frame, (cx, cy), dz_px, (80, 80, 80), 1)   # deadzone ring

    if lms is not None:
        # centroid
        ctr = lms[PALM_ANCHORS].mean(axis=0).astype(int)
        cv2.circle(frame, tuple(ctr), 8, (0, 255, 255), -1)
        # line from frame center to centroid
        cv2.line(frame, (cx, cy), tuple(ctr), (0, 255, 255), 1)

    label = "KNUCKLE STOP" if pinching else KEY_LABEL.get(frozenset(keys), str(keys))
    color = (0, 0, 255) if pinching else (0, 255, 0) if keys else (180, 180, 180)
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
    print("  Gesture WASD keymapper — MediaPipe")
    print("  Hand UP=W  DOWN=S  LEFT=A  RIGHT=D")
    print("  Pinch = stop all keys")
    print("  Q=quit  M=mirror  SPACE=pause")
    print("="*55)
    if not _PYNPUT_OK:
        print("[WARN] Key output disabled. Install pynput + grant Accessibility.")
    print()

    mirror     = True    # mirror frame for natural feel
    paused     = False
    prev_lms   = None    # EMA state
    last_t     = time.time()
    fps        = 0.0

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

            # ── agile filter (EMA) ────────────────────────────────────────────
            if lms is None:
                prev_lms = None
            else:
                if prev_lms is not None and prev_lms.shape == lms.shape:
                    lms = EMA_ALPHA * prev_lms + (1 - EMA_ALPHA) * lms
                prev_lms = lms

            # ── gesture → keys ────────────────────────────────────────────────
            keys, pinching = set(), False
            if lms is not None and not paused:
                keys, pinching = compute_keys(lms, W, H)
                apply_keys(keys)
            else:
                release_all()

            # ── draw landmarks ────────────────────────────────────────────────
            if mp_res.multi_hand_landmarks:
                for hl in mp_res.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hl, mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style(),
                    )

            # ── FPS ───────────────────────────────────────────────────────────
            now   = time.time()
            fps   = 0.9 * fps + 0.1 * (1.0 / max(now - last_t, 1e-6))
            last_t = now

            draw_hud(frame, lms, keys, pinching, fps, paused, W, H)
            cv2.imshow("Gesture WASD", frame)

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
