# ── Cell 5: Live Camera Demo — Ctrl+C or Q to quit ───────────────────────────
import cv2, signal, numpy as np, torch
import mediapipe as mp

INPUT_SIZE = 128
IMG_MEAN   = 0.5
IMG_STD    = 0.5
BBOX_PAD   = 0.35
MESH_COLOR = (0, 220, 50)
SKEL_COLOR = (30, 100, 255)
KPT_COLOR  = (0, 255, 255)

mp_hands     = mp.solutions.hands
hand_tracker = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)

# ── Helper functions ──────────────────────────────────────────────────────────

def get_bbox(mp_res, H, W, pad=BBOX_PAD):
    """Return (x1,y1,x2,y2) SQUARE padded hand bbox, or None."""
    if not mp_res.multi_hand_landmarks:
        return None
    lms = mp_res.multi_hand_landmarks[0].landmark
    xs  = [lm.x * W for lm in lms]
    ys  = [lm.y * H for lm in lms]
    cx, cy = (max(xs)+min(xs))/2, (max(ys)+min(ys))/2
    side = max(max(xs)-min(xs), max(ys)-min(ys))
    half = side * (1 + pad) / 2
    x1 = max(0, int(cx - half))
    y1 = max(0, int(cy - half))
    x2 = min(W, int(cx + half))
    y2 = min(H, int(cy + half))
    return (x1, y1, x2, y2)


def get_mp_landmarks(mp_res, H, W):
    """Get MediaPipe 21 landmarks as (21, 2) pixel coords [x, y]."""
    if not mp_res.multi_hand_landmarks:
        return None
    lms = mp_res.multi_hand_landmarks[0].landmark
    return np.array([[lm.x * W, lm.y * H] for lm in lms], dtype=np.float32)


def preprocess(frame_rgb, bbox):
    """Crop hand, resize to 128x128, normalise -> (1,3,128,128) tensor."""
    x1, y1, x2, y2 = bbox
    crop = frame_rgb[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    crop = cv2.resize(crop, (INPUT_SIZE, INPUT_SIZE))
    crop = (crop.astype(np.float32) / 255.0 - IMG_MEAN) / IMG_STD
    return torch.from_numpy(crop).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)


def verts_to_px(verts3d, mp_lms):
    """
    Align 3D mesh vertex XY to MediaPipe 2D landmarks via scale + translate.
    verts3d: (778, 3) model output in MANO space.
    mp_lms:  (21, 2) MediaPipe landmarks in pixel coords.
    """
    v = verts3d[:, :2].copy().astype(np.float32)
    # Flip Y: MANO Y points up, screen Y points down
    v[:, 1] = -v[:, 1]

    # Uniform scale + translate to match MediaPipe landmark extent
    v_min, v_max = v.min(0), v.max(0)
    v_center = (v_min + v_max) / 2
    v_span   = max((v_max - v_min).max(), 1e-6)

    mp_min, mp_max = mp_lms.min(0), mp_lms.max(0)
    mp_center = (mp_min + mp_max) / 2
    mp_span   = max((mp_max - mp_min).max(), 1e-6)

    scale = mp_span / v_span * 0.95
    out = (v - v_center) * scale + mp_center
    return out.astype(np.int32)


def draw_skeleton(img, pts, edges, color, thickness=2):
    for a, b in edges:
        if a < len(pts) and b < len(pts):
            cv2.line(img, tuple(pts[a].astype(int)), tuple(pts[b].astype(int)), color, thickness)


def draw_wireframe(img, verts_px, faces, color, thickness=1):
    H, W = img.shape[:2]
    for f in faces:
        p = verts_px[f]
        if (p < 0).any() or (p[:, 0] >= W).any() or (p[:, 1] >= H).any():
            continue
        cv2.line(img, tuple(p[0]), tuple(p[1]), color, thickness)
        cv2.line(img, tuple(p[1]), tuple(p[2]), color, thickness)
        cv2.line(img, tuple(p[2]), tuple(p[0]), color, thickness)


# MediaPipe skeleton edges (21 landmarks)
MP_SKELETON = [
    (0,1),(1,2),(2,3),(3,4),          # thumb
    (0,5),(5,6),(6,7),(7,8),          # index
    (0,9),(9,10),(10,11),(11,12),     # middle
    (0,13),(13,14),(14,15),(15,16),   # ring
    (0,17),(17,18),(18,19),(19,20),   # pinky
    (5,9),(9,13),(13,17),             # palm
]

# ── Signal handling ───────────────────────────────────────────────────────────
running = True
def _stop(sig, frame_):
    global running
    running = False
signal.signal(signal.SIGINT, _stop)

# ── Open camera ──────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
assert cap.isOpened(), 'Could not open camera.'

cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)

print('Demo running — hold a hand in front of the camera.')
print('Press Q in the window or Ctrl+C here to quit.\n')

try:
    while running:
        ok, frame = cap.read()
        if not ok:
            break

        frame     = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        H, W      = frame.shape[:2]

        mp_res  = hand_tracker.process(frame_rgb)
        bbox    = get_bbox(mp_res, H, W)
        mp_lms  = get_mp_landmarks(mp_res, H, W)

        if bbox is not None and mp_lms is not None:
            tensor = preprocess(frame_rgb, bbox)
            if tensor is not None:
                with torch.no_grad():
                    out = model(tensor)

                verts_3d = out['verts'][0].cpu().numpy()  # (778, 3)

                # Align mesh to MediaPipe landmarks
                vpts = verts_to_px(verts_3d, mp_lms)

                # Draw mesh wireframe
                draw_wireframe(frame, vpts, FACES, MESH_COLOR)

                # Draw skeleton using MediaPipe landmarks (accurate)
                draw_skeleton(frame, mp_lms, MP_SKELETON, SKEL_COLOR)
                for pt in mp_lms:
                    cv2.circle(frame, (int(pt[0]), int(pt[1])), 4, KPT_COLOR, -1)

            # Bounding box
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (180, 180, 180), 1)

        cv2.putText(frame, 'MobRecon  |  Q to quit',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow('MobRecon Demo', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    pass

finally:
    cap.release()
    cv2.destroyAllWindows()
    hand_tracker.close()
    print('Demo closed.')
