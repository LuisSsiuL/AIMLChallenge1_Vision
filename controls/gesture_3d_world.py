#!/usr/bin/env python3
"""
gesture_3d_world.py — Option D: 3D world-landmark palm-normal steering.

No recording needed. Hold palm facing camera = neutral.
  Pitch hand forward (knuckles toward screen) → W
  Pitch hand back   (palm faces ceiling)      → S
  Roll hand right   (thumb side up)           → D
  Roll hand left    (pinky side up)            → A

Controls: Q quit | M mirror | SPACE pause
"""

import os
import sys

_PY38 = os.path.expanduser("~/.pyenv/versions/3.8.10/bin/python")
if sys.version_info >= (3, 9) and os.path.exists(_PY38):
    os.execv(_PY38, [_PY38] + sys.argv)

import argparse
import math
import time

import cv2
import mediapipe as mp
import numpy as np

try:
    from Quartz import (CGEventCreateKeyboardEvent, CGEventPost,
                        kCGHIDEventTap, CGEventSetFlags)
    HAS_QUARTZ = True
except ImportError:
    HAS_QUARTZ = False
    print("WARNING: Quartz not available — keyboard events disabled (dry-run).")

# ── tunables ──────────────────────────────────────────────────────────────────

PITCH_DEAD_DEG  = 15.0   # ±° from neutral before W/S fires
ROLL_DEAD_DEG   = 15.0   # ±° from neutral before A/D fires
EMA_ALPHA       = 0.40   # smoothing (higher = more lag, less noise)
CAMERA          = 0

# ── key codes (US layout) ─────────────────────────────────────────────────────

KEY_CODES = {"W": 13, "A": 0, "S": 1, "D": 2}

def send_key(label, down):
    code = KEY_CODES.get(label)
    if code is None:
        return
    if HAS_QUARTZ:
        ev = CGEventCreateKeyboardEvent(None, code, down)
        CGEventSetFlags(ev, 0)
        CGEventPost(kCGHIDEventTap, ev)
    else:
        print(f"  [dry-run] {'DOWN' if down else 'UP  '} {label}")

# ── palm normal from 3D world landmarks ───────────────────────────────────────
#
# MediaPipe world_landmarks: metric-scale, origin at hand geometric centre.
# x = right, y = up, z = toward camera.
# Palm normal: cross( wrist→index_MCP, wrist→pinky_MCP ).
# For a right hand facing camera the normal points toward +z (camera).
#
# Pitch = rotation about X axis  → normal.y component
# Roll  = rotation about Y axis  → normal.x component
#
# Sign conventions chosen so:
#   pitch > 0  →  knuckles tip toward screen  →  W
#   pitch < 0  →  palm tips toward ceiling    →  S
#   roll  > 0  →  thumb side rises            →  D
#   roll  < 0  →  pinky side rises            →  A

def _wl_to_np(world_lms):
    return np.array([[lm.x, lm.y, lm.z] for lm in world_lms], dtype=np.float64)

def palm_normal(wl_np):
    wrist  = wl_np[0]
    idx    = wl_np[5]   # index MCP
    pinky  = wl_np[17]  # pinky MCP
    v1 = idx   - wrist
    v2 = pinky - wrist
    n  = np.cross(v1, v2)
    nrm = np.linalg.norm(n)
    return n / nrm if nrm > 1e-6 else np.array([0.0, 0.0, 1.0])

def normal_to_angles(n):
    """Return (pitch_deg, roll_deg) from palm normal vector."""
    pitch = math.degrees(math.asin(float(np.clip(-n[2], -1, 1))))
    roll  = math.degrees(math.asin(float(np.clip( n[0], -1, 1))))
    return pitch, roll

def angles_to_keys(pitch, roll):
    keys = set()
    if pitch >  PITCH_DEAD_DEG:  keys.add("S")
    if pitch < -PITCH_DEAD_DEG:  keys.add("W")
    if roll  >  ROLL_DEAD_DEG:   keys.add("D")
    if roll  < -ROLL_DEAD_DEG:   keys.add("A")
    return keys

# ── HUD ───────────────────────────────────────────────────────────────────────

MP_EDGES = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

def draw_skeleton(img, lms_2d, color=(0, 255, 120)):
    for a, b in MP_EDGES:
        if a < len(lms_2d) and b < len(lms_2d):
            cv2.line(img,
                     (int(lms_2d[a][0]), int(lms_2d[a][1])),
                     (int(lms_2d[b][0]), int(lms_2d[b][1])),
                     color, 2)
    for pt in lms_2d:
        cv2.circle(img, (int(pt[0]), int(pt[1])), 4, (0, 220, 255), -1)

def draw_normal_arrow(img, lms_2d, n):
    """Project normal onto 2D and draw from wrist."""
    wrist = lms_2d[0].astype(int)
    tip   = (wrist[0] + int(n[0] * 80),
             wrist[1] - int(n[1] * 80))   # y flipped for image coords
    cv2.arrowedLine(img, tuple(wrist), tip, (255, 80, 0), 3, tipLength=0.25)

def draw_gauge(img, value, dead, label, x, y, w=180, h=18):
    """Horizontal gauge centred at x showing value in [-90,90] with deadzone band."""
    half = w // 2
    cx   = x + half
    # background
    cv2.rectangle(img, (x, y), (x + w, y + h), (40, 40, 40), -1)
    # deadzone band
    dz_px = int(dead / 90.0 * half)
    cv2.rectangle(img, (cx - dz_px, y + 2), (cx + dz_px, y + h - 2), (80, 80, 30), -1)
    # value bar
    fill = int(np.clip(value / 90.0, -1, 1) * half)
    col  = (0, 200, 80) if abs(value) > dead else (80, 80, 80)
    if fill > 0:
        cv2.rectangle(img, (cx, y + 2), (cx + fill, y + h - 2), col, -1)
    elif fill < 0:
        cv2.rectangle(img, (cx + fill, y + 2), (cx, y + h - 2), col, -1)
    # centre line
    cv2.line(img, (cx, y), (cx, y + h), (150, 150, 150), 1)
    cv2.putText(img, f"{label} {value:+.1f}°", (x, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

def draw_hud(img, pitch, roll, keys, normal, lms_2d, fps, paused):
    H, W = img.shape[:2]

    # skeleton + normal arrow
    if lms_2d is not None:
        draw_skeleton(img, lms_2d)
        if normal is not None:
            draw_normal_arrow(img, lms_2d, normal)

    # top bar
    overlay = img[:52, :].copy()
    cv2.rectangle(img, (0, 0), (W, 52), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.25, img[:52, :], 0.75, 0, img[:52, :])

    key_str  = " + ".join(sorted(keys)) if keys else "—"
    mode_tag = "LIVE" if HAS_QUARTZ else "DRY-RUN"
    cv2.putText(img, f"Keys: {key_str}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(img, mode_tag, (W - 130, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (0, 200, 0) if HAS_QUARTZ else (0, 140, 255), 2)
    cv2.putText(img, f"FPS {fps:.0f}", (10, 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 140), 1)

    if paused:
        cv2.putText(img, "PAUSED (SPACE)", (W // 2 - 80, H // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)

    # gauges bottom-left
    draw_gauge(img, pitch, PITCH_DEAD_DEG, "Pitch W/S", 10, H - 90)
    draw_gauge(img, roll,  ROLL_DEAD_DEG,  "Roll  A/D", 10, H - 50)

    # WASD key grid bottom-right
    cell_w = 52
    ph     = 52
    y0     = H - ph
    keys_order = ["W", "A", "S", "D"]
    for i, key in enumerate(keys_order):
        active = key in keys
        x      = W - (len(keys_order) - i) * (cell_w + 4) - 8
        bg     = (0, 80, 0) if active else (30, 30, 30)
        cv2.rectangle(img, (x, y0 + 4), (x + cell_w - 4, y0 + ph - 4), bg, -1)
        cv2.putText(img, key, (x + 14, y0 + 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                    (0, 255, 0) if active else (120, 120, 120), 2)

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera",         type=int,   default=CAMERA)
    parser.add_argument("--pitch-dead",     type=float, default=PITCH_DEAD_DEG)
    parser.add_argument("--roll-dead",      type=float, default=ROLL_DEAD_DEG)
    parser.add_argument("--ema",            type=float, default=EMA_ALPHA)
    args = parser.parse_args()

    pitch_dead = args.pitch_dead
    roll_dead  = args.roll_dead
    ema        = args.ema

    tracker = mp.solutions.hands.Hands(
        static_image_mode=False, max_num_hands=1,
        min_detection_confidence=0.7, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(args.camera, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit("Could not open camera.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("3D world-landmark gesture driver")
    print(f"  Pitch dead ±{pitch_dead}°   Roll dead ±{roll_dead}°")
    print("  Hold palm facing camera = neutral")
    print("  Tip knuckles toward screen = W  |  Palm ceiling = S")
    print("  Thumb side up = D               |  Pinky side up = A")
    print("  Q quit  M mirror  SPACE pause\n")

    mirror       = True
    paused       = False
    active_keys  : set = set()
    ema_normal   = None
    last_t       = time.time()
    fps          = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if mirror:
                frame = cv2.flip(frame, 1)

            H, W      = frame.shape[:2]
            rgb       = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result    = tracker.process(rgb)

            pitch, roll = 0.0, 0.0
            normal      = None
            lms_2d      = None
            keys        = set()

            if (result.multi_hand_world_landmarks and result.multi_hand_landmarks):
                wl  = _wl_to_np(result.multi_hand_world_landmarks[0].landmark)
                n   = palm_normal(wl)

                # EMA on normal vector
                if ema_normal is None:
                    ema_normal = n
                else:
                    ema_normal = ema * ema_normal + (1 - ema) * n
                    nrm = np.linalg.norm(ema_normal)
                    if nrm > 1e-6:
                        ema_normal = ema_normal / nrm

                normal = ema_normal
                pitch, roll = normal_to_angles(normal)

                lms_raw = result.multi_hand_landmarks[0].landmark
                lms_2d  = np.array([[lm.x * W, lm.y * H] for lm in lms_raw],
                                   dtype=np.float32)

                if not paused:
                    keys = angles_to_keys(pitch, roll)
            else:
                ema_normal = None

            # fire key events on state change
            if not paused:
                for key in KEY_CODES:
                    was = key in active_keys
                    now_active = key in keys
                    if now_active and not was:
                        send_key(key, True)
                    elif not now_active and was:
                        send_key(key, False)
                active_keys = keys
            else:
                for key in active_keys:
                    send_key(key, False)
                active_keys = set()

            now    = time.time()
            fps    = 0.9 * fps + 0.1 / max(now - last_t, 1e-6)
            last_t = now

            draw_hud(frame, pitch, roll, keys, normal, lms_2d, fps, paused)
            cv2.imshow("3D World Gesture Driver", frame)

            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            elif k == ord("m"):
                mirror = not mirror
            elif k == ord(" "):
                paused = not paused

    except KeyboardInterrupt:
        pass
    finally:
        for key in active_keys:
            send_key(key, False)
        cap.release()
        cv2.destroyAllWindows()
        tracker.close()
        print("Closed.")

if __name__ == "__main__":
    main()
