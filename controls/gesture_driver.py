#!/usr/bin/env python3
"""
gesture_driver.py — reads gestures.json, detects hand poses live,
fires real macOS keyboard events (W/A/S/D) to the focused app.

Requirements:
  - gestures.json recorded via landmarks_live.py
  - macOS Accessibility permission for the terminal running this script
    (System Settings → Privacy & Security → Accessibility → toggle on Terminal/iTerm)

Usage:
  python gesture_driver.py
  python gesture_driver.py --camera 1 --threshold 0.30
"""

import os
import sys

_PY38 = os.path.expanduser("~/.pyenv/versions/3.8.10/bin/python")
if sys.version_info >= (3, 9) and os.path.exists(_PY38):
    os.execv(_PY38, [_PY38] + sys.argv)

import argparse  # noqa: E402
import json      # noqa: E402
import signal    # noqa: E402

import cv2
import mediapipe as mp
import numpy as np

try:
    from Quartz import (CGEventCreateKeyboardEvent, CGEventPost,
                        kCGHIDEventTap, CGEventSetFlags)
    HAS_QUARTZ = True
except ImportError:
    HAS_QUARTZ = False
    print("WARNING: Quartz not available — keyboard events disabled (dry-run mode).")

# ─── Paths / config ───────────────────────────────────────────────────────────

SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
GESTURES_FILE = os.path.join(SCRIPT_DIR, "gestures.json")
CONFIG_FILE   = os.path.join(SCRIPT_DIR, "config.json")

KEY_CODES = {"W": 13, "A": 0, "S": 1, "D": 2}

MP_EDGES = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

COL_RIGHT_SKEL = (30,  100, 255)
COL_RIGHT_KPT  = (0,   255, 255)
COL_LEFT_SKEL  = (255, 100,  30)
COL_LEFT_KPT   = (0,   200,   0)
COL_ACTIVE     = (0,   255,   0)
COL_INACTIVE   = (60,   60,  60)

# ─── macOS key event ──────────────────────────────────────────────────────────

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

# ─── Feature / matching (same logic as landmarks_live.py) ────────────────────

def hand_features(pts):
    delta = pts[9] - pts[0]
    pitch = float(np.clip(-delta[1] / 150.0, -1, 1))
    roll  = float(np.clip( delta[0] / 150.0, -1, 1))
    return pitch, roll

def get_hands(mp_res, H, W):
    if not mp_res.multi_hand_landmarks:
        return []
    out = []
    for i, hand_lms in enumerate(mp_res.multi_hand_landmarks):
        label = "Right"
        if mp_res.multi_handedness and i < len(mp_res.multi_handedness):
            label = mp_res.multi_handedness[i].classification[0].label
        label = "Left" if label == "Right" else "Right"
        pts = np.array([[lm.x * W, lm.y * H] for lm in hand_lms.landmark],
                       dtype=np.float32)
        out.append((label, pts))
    return out

def match_gestures(hands, db, threshold):
    matched = set()
    for label, pts in hands:
        pitch, roll = hand_features(pts)
        best_dist, best_key = float("inf"), None
        for key, rec in db.items():
            if rec["hand"] != label:
                continue
            dist = ((pitch - rec["pitch"])**2 + (roll - rec["roll"])**2) ** 0.5
            if dist < best_dist:
                best_dist, best_key = dist, key
        if best_key is not None and best_dist < threshold:
            matched.add(best_key)
    return matched

# ─── Drawing ─────────────────────────────────────────────────────────────────

def draw_skeleton(img, lms, color, thickness=2):
    for a, b in MP_EDGES:
        if a < len(lms) and b < len(lms):
            cv2.line(img, (int(lms[a][0]), int(lms[a][1])),
                          (int(lms[b][0]), int(lms[b][1])), color, thickness)

def draw_keypoints(img, lms, color, radius=4):
    for pt in lms:
        cv2.circle(img, (int(pt[0]), int(pt[1])), radius, color, -1)

def draw_hands(img, hands, matched, db):
    for lbl, pts in hands:
        active = any(db.get(m, {}).get("hand") == lbl for m in matched)
        sk = COL_ACTIVE if active else (COL_LEFT_SKEL  if lbl == "Left" else COL_RIGHT_SKEL)
        kp = COL_ACTIVE if active else (COL_LEFT_KPT   if lbl == "Left" else COL_RIGHT_KPT)
        draw_skeleton(img, pts, sk)
        draw_keypoints(img, pts, kp)

def draw_hud(img, matched, db):
    H, W = img.shape[:2]

    # Top bar — active gestures
    bar = img[:44, :].copy()
    cv2.rectangle(img, (0, 0), (W, 44), (0, 0, 0), -1)
    cv2.addWeighted(bar, 0.25, img[:44, :], 0.75, 0, img[:44, :])
    active_str = " + ".join(sorted(matched)) if matched else "—"
    cv2.putText(img, f"Active: {active_str}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.80, (255, 255, 255), 2)
    mode_tag = "LIVE" if HAS_QUARTZ else "DRY-RUN"
    cv2.putText(img, mode_tag, (W - 120, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (0, 200, 0) if HAS_QUARTZ else (0, 140, 255), 2)

    # Bottom key grid
    ph = 52
    y0 = H - ph
    ov = img[y0:, :].copy()
    cv2.rectangle(img, (0, y0), (W, H), (15, 15, 15), -1)
    cv2.addWeighted(ov, 0.2, img[y0:, :], 0.8, 0, img[y0:, :])

    cv2.putText(img, "Q: quit  |  focus driving sim window then use hands",
                (10, y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (150,150,150), 1)

    keys   = list(KEY_CODES.keys())
    cell_w = W // len(keys)
    for i, key in enumerate(keys):
        rec    = db.get(key)
        active = key in matched
        x      = i * cell_w
        bg     = (0, 80, 0) if active else (30, 30, 30)
        cv2.rectangle(img, (x + 4, y0 + 24), (x + cell_w - 4, y0 + ph - 4), bg, -1)
        label_color = (0, 255, 0) if active else ((180, 180, 180) if rec else (60, 60, 60))
        cv2.putText(img, key, (x + cell_w // 2 - 10, y0 + 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, label_color, 2)

# ─── Signal ───────────────────────────────────────────────────────────────────

running = True

def _stop(*_):
    global running
    running = False

signal.signal(signal.SIGINT, _stop)

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Gesture → keyboard driver")
    parser.add_argument("--camera",    type=int,   default=None)
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()

    # Load config
    cfg = {"camera": 0, "threshold": 0.35}
    if os.path.isfile(CONFIG_FILE):
        with open(CONFIG_FILE) as f:
            cfg.update(json.load(f))
    if args.camera    is not None: cfg["camera"]    = args.camera
    if args.threshold is not None: cfg["threshold"] = args.threshold

    # Load gestures
    if not os.path.isfile(GESTURES_FILE):
        sys.exit(f"No gestures.json found. Run landmarks_live.py first to record gestures.")
    with open(GESTURES_FILE) as f:
        db = json.load(f)
    if not db:
        sys.exit("gestures.json is empty. Record gestures in landmarks_live.py first.")

    recorded = sorted(db.keys())
    missing  = [k for k in KEY_CODES if k not in db]
    print(f"Loaded gestures: {recorded}")
    if missing:
        print(f"WARNING: not recorded yet: {missing}")
    print("Accessibility permission required for key injection.")
    print("Focus the driving sim window, then use your hands.\n")

    tracker = mp.solutions.hands.Hands(
        static_image_mode=False, max_num_hands=2,
        min_detection_confidence=0.7, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(cfg["camera"], cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        cap = cv2.VideoCapture(cfg["camera"])
    if not cap.isOpened():
        sys.exit("Could not open camera.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    active_keys: set = set()

    try:
        while running:
            ok, frame = cap.read()
            if not ok:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            H, W = frame.shape[:2]
            mp_res  = tracker.process(frame_rgb)
            hands   = get_hands(mp_res, H, W)
            matched = match_gestures(hands, db, cfg["threshold"])

            # Fire key events only on state changes
            for key in KEY_CODES:
                was_active = key in active_keys
                is_active  = key in matched
                if is_active and not was_active:
                    send_key(key, True)
                elif not is_active and was_active:
                    send_key(key, False)
            active_keys = matched

            draw_hands(frame, hands, matched, db)
            draw_hud(frame, matched, db)
            cv2.imshow("Gesture Driver", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        # Release all held keys on exit
        for key in active_keys:
            send_key(key, False)
        cap.release()
        cv2.destroyAllWindows()
        tracker.close()
        print("Gesture driver closed.")

if __name__ == "__main__":
    main()
