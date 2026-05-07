#!/usr/bin/env python3
"""
landmarks_live.py — MediaPipe two-hand gesture recorder + live demo.

In the window:
  W / A / S / D  — start recording that gesture (3s ready → 5s capture)
  C              — clear ALL saved gestures
  Q              — quit

Left hand: cyan/green   Right hand: orange/yellow   Matched: bright green
"""

import os
import sys

_PY38 = os.path.expanduser("~/.pyenv/versions/3.8.10/bin/python")
if sys.version_info >= (3, 9) and os.path.exists(_PY38):
    os.execv(_PY38, [_PY38] + sys.argv)

import argparse   # noqa: E402
import json       # noqa: E402
import signal     # noqa: E402
import time       # noqa: E402
from collections import defaultdict  # noqa: E402

import cv2
import mediapipe as mp
import numpy as np

# ─── Paths ────────────────────────────────────────────────────────────────────

SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
GESTURES_FILE = os.path.join(SCRIPT_DIR, "gestures.json")
CONFIG_FILE   = os.path.join(SCRIPT_DIR, "config.json")

# ─── Config ───────────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "camera":       0,
    "ready_secs":   3.0,
    "record_secs":  5.0,
    "threshold":    0.35,
}

def load_config():
    if os.path.isfile(CONFIG_FILE):
        with open(CONFIG_FILE) as f:
            data = json.load(f)
        cfg = {**DEFAULT_CONFIG, **data}
    else:
        cfg = dict(DEFAULT_CONFIG)
    save_config(cfg)   # write defaults for any missing keys
    return cfg

def save_config(cfg):
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)

# ─── Gesture DB ───────────────────────────────────────────────────────────────

def load_gestures():
    if not os.path.isfile(GESTURES_FILE):
        return {}
    with open(GESTURES_FILE) as f:
        return json.load(f)

def save_gestures(db):
    with open(GESTURES_FILE, "w") as f:
        json.dump(db, f, indent=2)

# ─── Colours (BGR) ────────────────────────────────────────────────────────────

COL_RIGHT_SKEL = (30,  100, 255)
COL_RIGHT_KPT  = (0,   255, 255)
COL_LEFT_SKEL  = (255, 100,  30)
COL_LEFT_KPT   = (0,   200,   0)
COL_MATCH      = (0,   255,   0)
COL_REC_READY  = (0,   200, 255)
COL_REC_GO     = (0,    50, 255)
COL_PANEL_BG   = (20,   20,  20)

MP_EDGES = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

RECORDABLE = ["W", "A", "S", "D"]

# Fixed hand assignment — enforced during recording AND matching
HAND_FOR_KEY = {"W": "Left", "S": "Left", "A": "Right", "D": "Right"}

# ─── Feature extraction ───────────────────────────────────────────────────────

def hand_features(pts):
    """(pitch, roll) in [-1,1]. Position-invariant — wrist→middle-MCP direction."""
    delta = pts[9] - pts[0]
    pitch = float(np.clip(-delta[1] / 150.0, -1, 1))
    roll  = float(np.clip( delta[0] / 150.0, -1, 1))
    return pitch, roll

# ─── Hand extraction ──────────────────────────────────────────────────────────

def get_hands(mp_res, H, W):
    if not mp_res.multi_hand_landmarks:
        return []
    out = []
    for i, hand_lms in enumerate(mp_res.multi_hand_landmarks):
        label = "Right"
        if mp_res.multi_handedness and i < len(mp_res.multi_handedness):
            label = mp_res.multi_handedness[i].classification[0].label
        label = "Left" if label == "Right" else "Right"   # mirror flip
        pts = np.array([[lm.x * W, lm.y * H] for lm in hand_lms.landmark],
                       dtype=np.float32)
        out.append((label, pts))
    return out

# ─── Gesture matching ─────────────────────────────────────────────────────────

def match_gestures(hands, db, threshold):
    matched = []
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
            matched.append(best_key)
    return matched

# ─── Drawing ──────────────────────────────────────────────────────────────────

def draw_skeleton(img, lms, color, thickness=2):
    for a, b in MP_EDGES:
        if a < len(lms) and b < len(lms):
            cv2.line(img, (int(lms[a][0]), int(lms[a][1])),
                          (int(lms[b][0]), int(lms[b][1])), color, thickness)

def draw_keypoints(img, lms, color, radius=4):
    for pt in lms:
        cv2.circle(img, (int(pt[0]), int(pt[1])), radius, color, -1)

def draw_hands(img, hands, matched_labels, db):
    for lbl, pts in hands:
        matched_here = any(db.get(m, {}).get("hand") == lbl for m in matched_labels)
        if matched_here:
            draw_skeleton(img, pts, COL_MATCH)
            draw_keypoints(img, pts, COL_MATCH)
        elif lbl == "Left":
            draw_skeleton(img, pts, COL_LEFT_SKEL)
            draw_keypoints(img, pts, COL_LEFT_KPT)
        else:
            draw_skeleton(img, pts, COL_RIGHT_SKEL)
            draw_keypoints(img, pts, COL_RIGHT_KPT)

def draw_top_hud(img, line1, line2=""):
    W = img.shape[1]
    h = 52 if line2 else 36
    overlay = img[:h, :].copy()
    cv2.rectangle(img, (0, 0), (W, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.3, img[:h, :], 0.7, 0, img[:h, :])
    cv2.putText(img, line1, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255,255,255), 2)
    if line2:
        cv2.putText(img, line2, (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180,180,180), 1)

def draw_bottom_panel(img, db):
    """Status bar: shows each recordable key and whether it's saved."""
    H, W = img.shape[:2]
    ph = 56
    y0 = H - ph
    overlay = img[y0:, :].copy()
    cv2.rectangle(img, (0, y0), (W, H), COL_PANEL_BG, -1)
    cv2.addWeighted(overlay, 0.2, img[y0:, :], 0.8, 0, img[y0:, :])

    # Key hint row
    hints = "W/S = Left hand   A/D = Right hand   |   C = clear all   |   Q = quit"
    cv2.putText(img, hints, (10, y0 + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 180, 180), 1)

    # Gesture status row
    cell_w = W // len(RECORDABLE)
    for i, key in enumerate(RECORDABLE):
        rec      = db.get(key)
        x        = i * cell_w + 8
        req_hand = HAND_FOR_KEY[key][0]   # "L" or "R"
        if rec:
            tag   = f"[{key}:{req_hand}]  p={rec['pitch']:+.2f} r={rec['roll']:+.2f}"
            color = (0, 220, 0)
        else:
            tag   = f"[{key}:{req_hand}]  not recorded"
            color = (80, 80, 80)
        cv2.putText(img, tag, (x, y0 + 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)

def draw_record_overlay(img, label, phase, remaining):
    H, W = img.shape[:2]
    cy = H // 2
    # Dark band
    cv2.rectangle(img, (0, cy - 50), (W, cy + 50), (0, 0, 0), -1)
    required = HAND_FOR_KEY.get(label, "?")
    if phase == "ready":
        secs = int(remaining) + 1
        msg  = f"  GET READY [{label}]  Use {required} hand  —  {secs}s"
        color = COL_REC_READY
    else:
        bar_w = int((1 - remaining / 5.0) * (W - 40))
        cv2.rectangle(img, (20, cy + 28), (20 + bar_w, cy + 44), COL_REC_GO, -1)
        cv2.rectangle(img, (20, cy + 28), (W - 20,     cy + 44), (100,100,100), 1)
        msg  = f"  RECORDING [{label}]  {required} hand  —  {remaining:.1f}s left"
        color = COL_REC_GO
    cv2.putText(img, msg, (10, cy + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.80, color, 2)

# ─── Signal handler ───────────────────────────────────────────────────────────

running = True

def _stop(*_):
    global running
    running = False

signal.signal(signal.SIGINT, _stop)

# ─── Camera helper ────────────────────────────────────────────────────────────

def _open_camera(index):
    cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        sys.exit("Could not open camera.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return cap

# ─── Image loader ─────────────────────────────────────────────────────────────

def _load_image_any(path):
    img = cv2.imread(path)
    if img is not None:
        return img
    try:
        from PIL import Image
        try:
            import pillow_avif  # type: ignore  # noqa: F401
        except ImportError:
            pass
        return cv2.cvtColor(np.array(Image.open(path).convert("RGB")),
                            cv2.COLOR_RGB2BGR)
    except Exception as e:
        sys.exit(f"ERROR: could not load image {path}: {e}")

# ─── Live + record mode (single unified window) ───────────────────────────────

def run_live(camera_index, cfg):
    db = load_gestures()

    tracker = mp.solutions.hands.Hands(
        static_image_mode=False, max_num_hands=2,
        min_detection_confidence=0.7, min_tracking_confidence=0.5)
    cap = _open_camera(camera_index)

    # Recording state machine
    rec_label   = None    # currently recording label, or None
    rec_phase   = None    # "ready" | "record"
    rec_samples = []
    rec_t_start = 0.0

    print("Window open. Press W/A/S/D to record, C to clear, Q to quit.")

    try:
        while running:
            ok, frame = cap.read()
            if not ok:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            H, W = frame.shape[:2]
            mp_res = tracker.process(frame_rgb)
            hands  = get_hands(mp_res, H, W)

            # ── Recording state machine ──
            if rec_label:
                elapsed   = time.time() - rec_t_start
                remaining = (cfg["ready_secs"] if rec_phase == "ready"
                             else cfg["record_secs"]) - elapsed

                if rec_phase == "ready" and remaining <= 0:
                    rec_phase   = "record"
                    rec_samples = []
                    rec_t_start = time.time()
                    remaining   = cfg["record_secs"]

                if rec_phase == "record":
                    required_hand = HAND_FOR_KEY[rec_label]
                    for lbl, pts in hands:
                        if lbl == required_hand:          # only accept correct hand
                            p, r = hand_features(pts)
                            rec_samples.append((p, r))
                    if remaining <= 0:
                        if rec_samples:
                            db[rec_label] = {
                                "hand":    HAND_FOR_KEY[rec_label],
                                "pitch":   float(np.mean([v[0] for v in rec_samples])),
                                "roll":    float(np.mean([v[1] for v in rec_samples])),
                                "samples": len(rec_samples),
                            }
                            save_gestures(db)
                            print(f"Saved [{rec_label}]: {db[rec_label]}")
                        else:
                            print(f"No {HAND_FOR_KEY[rec_label]} hand detected — [{rec_label}] not saved.")
                        rec_label = rec_phase = None

            # ── Draw ──
            matched = match_gestures(hands, db, cfg["threshold"]) if db else []
            draw_hands(frame, hands, matched, db)

            if rec_label:
                draw_record_overlay(frame, rec_label, rec_phase, remaining)
                hand_str = ", ".join(h[0] for h in hands) if hands else "none"
                draw_top_hud(frame, f"Hands: {hand_str}", "")
            else:
                hand_str  = ", ".join(h[0] for h in hands) if hands else "none"
                match_str = ("  →  " + " + ".join(matched)) if matched else ""
                draw_top_hud(frame,
                             f"Hands ({len(hands)}): {hand_str}{match_str}",
                             f"threshold={cfg['threshold']:.2f}")

            draw_bottom_panel(frame, db)
            cv2.imshow("Landmark Gesture Demo", frame)

            # ── Key handling ──
            key = cv2.waitKey(1) & 0xFF
            ch  = chr(key).upper() if key < 128 else ""

            if ch == "Q":
                break
            elif ch == "C" and not rec_label:
                db = {}
                save_gestures(db)
                print("Cleared all gestures.")
            elif ch in RECORDABLE and not rec_label:
                rec_label   = ch
                rec_phase   = "ready"
                rec_samples = []
                rec_t_start = time.time()
                print(f"Starting record for [{ch}] …")

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        tracker.close()
        print("Demo closed.")

# ─── Static image mode ────────────────────────────────────────────────────────

def run_image(path, out_prefix):
    if not os.path.isabs(path):
        path = os.path.join(SCRIPT_DIR, path)
    if not os.path.isfile(path):
        sys.exit(f"ERROR: image not found: {path}")

    frame     = _load_image_any(path)
    H, W      = frame.shape[:2]
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    tracker = mp.solutions.hands.Hands(
        static_image_mode=True, max_num_hands=2,
        min_detection_confidence=0.5)
    mp_res = tracker.process(frame_rgb)
    tracker.close()

    hands = get_hands(mp_res, H, W)
    db    = load_gestures()
    cfg   = load_config()
    matched = match_gestures(hands, db, cfg["threshold"]) if db else []
    draw_hands(frame, hands, matched, db)

    label_str = ", ".join(h[0] for h in hands) if hands else "none"
    draw_top_hud(frame, f"Hands ({len(hands)}): {label_str}")
    draw_bottom_panel(frame, db)

    if not os.path.isabs(out_prefix):
        out_prefix = os.path.join(SCRIPT_DIR, out_prefix)
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
    out_path = out_prefix + ".png"
    cv2.imwrite(out_path, frame)
    print(f"Wrote: {out_path}")

# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MediaPipe gesture demo")
    parser.add_argument("--image",      default="",
                        help="Static image path (skips live mode)")
    parser.add_argument("--out-prefix", default="out/landmarks")
    parser.add_argument("--camera",     type=int, default=None,
                        help="Camera index (overrides config.json)")
    parser.add_argument("--threshold",  type=float, default=None,
                        help="Match threshold (overrides config.json)")
    args = parser.parse_args()

    cfg = load_config()
    if args.camera    is not None: cfg["camera"]    = args.camera
    if args.threshold is not None: cfg["threshold"] = args.threshold

    if args.image:
        run_image(args.image, args.out_prefix)
    else:
        run_live(cfg["camera"], cfg)
