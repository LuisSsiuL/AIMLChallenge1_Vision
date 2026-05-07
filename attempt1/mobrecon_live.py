#!/Users/christianluisefendy/.pyenv/versions/3.8.10/bin/python
"""
MobRecon — Live Hand Mesh Demo
===============================
Real-time 3D hand mesh reconstruction using MobRecon + MediaPipe.

Controls:
  M  — cycle mode: Silhouette → Landmarks → Mesh → All
  Q  — quit

Usage:
  python mobrecon_live.py
"""

import sys, os, types, glob, signal, argparse

ap = argparse.ArgumentParser()
ap.add_argument("--image", default=None, help="Run inference on a single image instead of camera")
ap.add_argument("--out-prefix", default="picture_annotated", help="Output filename prefix for static image mode")
args = ap.parse_args()

# ─── Resolve paths ────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR   = os.path.join(SCRIPT_DIR, "HandMesh")

if not os.path.isdir(REPO_DIR):
    sys.exit(f"ERROR: HandMesh repo not found at {REPO_DIR}")

if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

# ─── Openmesh / psbody shims ─────────────────────────────────────────────────
import numpy as np

class _VertexHandle:
    __slots__ = ("_idx",)
    def __init__(self, idx):
        self._idx = idx._idx if isinstance(idx, _VertexHandle) else int(idx)
    def idx(self):   return self._idx
    def __int__(self):   return self._idx
    def __index__(self): return self._idx

class _TriMesh:
    def __init__(self, vertices, faces):
        self._points = np.asarray(vertices, dtype=np.float64).copy()
        self._faces  = np.asarray(faces, dtype=np.int64).copy()
        self._build_topology()
    def _build_topology(self):
        n_v = len(self._points)
        v_face_pos = [[] for _ in range(n_v)]
        for fi, face in enumerate(self._faces):
            for i in range(3):
                v_face_pos[int(face[i])].append((fi, i))
        edge_to_fp = {}
        for fi, face in enumerate(self._faces):
            for i in range(3):
                edge_to_fp[(int(face[i]), int(face[(i+1)%3]))] = (fi, i)
        next_fp, prev_fp = {}, {}
        for fi, face in enumerate(self._faces):
            for i in range(3):
                v_, u_ = int(face[i]), int(face[(i+2)%3])
                key = (v_, u_)
                if key in edge_to_fp and edge_to_fp[key] != (fi, i):
                    next_fp[(fi, i)] = edge_to_fp[key]
                    prev_fp[edge_to_fp[key]] = (fi, i)
        self._neighbors = [[] for _ in range(n_v)]
        for v_ in range(n_v):
            fps = v_face_pos[v_]
            if not fps: continue
            start = next((fp for fp in fps if fp not in prev_fp), fps[0])
            walk, cur, seen = [], start, set()
            while cur is not None and cur not in seen:
                seen.add(cur); walk.append(cur); cur = next_fp.get(cur)
            res = [int(self._faces[fi][(i+1)%3]) for (fi, i) in walk]
            if cur is None and walk:
                last_fi, last_i = walk[-1]
                a_ = int(self._faces[last_fi][(last_i+2)%3])
                if a_ not in res: res.append(a_)
            self._neighbors[v_] = res
    def vertices(self):
        for i in range(len(self._points)): yield _VertexHandle(i)
    def vv(self, vh):
        v = vh.idx() if hasattr(vh, "idx") else int(vh)
        for n in self._neighbors[v]: yield _VertexHandle(n)
    def points(self): return self._points

_shim = types.ModuleType("openmesh")
_shim.VertexHandle = _VertexHandle
_shim.TriMesh = _TriMesh
_shim.read_trimesh = lambda path: None
_shim.write_mesh = lambda *a, **k: None
sys.modules["openmesh"] = _shim

if "psbody" not in sys.modules:
    try: import psbody.mesh
    except ImportError:
        _ps = types.ModuleType("psbody")
        _ps_msh = types.ModuleType("psbody.mesh")
        class _StubMesh:
            def __init__(self, *a, **k): raise RuntimeError("psbody stub")
        _ps_msh.Mesh = _StubMesh
        _ps.mesh = _ps_msh
        sys.modules["psbody"] = _ps
        sys.modules["psbody.mesh"] = _ps_msh

# ─── Imports ──────────────────────────────────────────────────────────────────
import torch
import cv2
import mediapipe as mp
import fvcore  # noqa
from mobrecon.configs.config import get_cfg
from mobrecon.models.mobrecon_ds import MobRecon_DS
from mobrecon.tools.kinematics import mano_to_mpii

# ─── Build model ──────────────────────────────────────────────────────────────
CKPT_DIR = os.path.join(REPO_DIR, "mobrecon", "checkpoints")
candidates = (glob.glob(os.path.join(CKPT_DIR, "**", "*.pt"), recursive=True) +
              glob.glob(os.path.join(CKPT_DIR, "**", "*.pth"), recursive=True))
mobrecon_ckpts = [f for f in candidates if "mobrecon" in os.path.basename(f).lower()]
if not mobrecon_ckpts and not candidates:
    sys.exit(f"ERROR: No checkpoint in {CKPT_DIR}")
CKPT_PATH = (mobrecon_ckpts or candidates)[0]

print("Building MobRecon_DS...")
cfg = get_cfg()
model = MobRecon_DS(cfg)

print(f"Loading weights from:\n  {CKPT_PATH}")
raw = torch.load(CKPT_PATH, map_location="cpu")
state = raw.get("model_state_dict") or raw.get("state_dict") or raw.get("model") or raw if isinstance(raw, dict) else raw
# Older checkpoints name the kpts-reducing conv "backbone.conv" while the current code
# names it "backbone.reduce". Rename so the kpts head loads its trained weights.
state = {k.replace("backbone.conv.", "backbone.reduce."): v for k, v in state.items()}
missing, unexpected = model.load_state_dict(state, strict=False)
if missing:    print(f"  Missing keys  : {len(missing)}  ({missing[:3]}{'...' if len(missing)>3 else ''})")
if unexpected: print(f"  Unexpected keys: {len(unexpected)}  ({unexpected[:3]}{'...' if len(unexpected)>3 else ''})")

DEVICE = ("mps" if torch.backends.mps.is_available() else
          "cuda" if torch.cuda.is_available() else "cpu")
model.to(DEVICE).eval()

FACES = np.load(os.path.join(REPO_DIR, "template", "right_faces.npy")).astype(np.int32)
J_REG = np.load(os.path.join(REPO_DIR, "template", "j_reg.npy"))  # (21, 778)

print(f"\nModel ready  |  device: {DEVICE}  |  faces: {len(FACES)}")

# ═══════════════════════════════════════════════════════════════════════════════
#  DEMO CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

INPUT_SIZE = 128
IMG_MEAN, IMG_STD = 0.5, 0.5
BBOX_PAD = 0.10
STD_SCALE = 0.20  # official vertex scale factor

# Colours (BGR)
COL_MESH = (0, 220, 50)
COL_SKEL = (30, 100, 255)
COL_KPT  = (0, 255, 255)
COL_SILH = (200, 120, 50)

MODES    = ["Silhouette", "Landmarks", "Mesh", "All"]
mode_idx = 3

MP_EDGES = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]


# ─── Projection helpers ──────────────────────────────────────────────────────

def get_bbox(mp_res, H, W, pad=BBOX_PAD):
    """Square bbox of side `s*(1+pad)`, centered on hand. Pinned to image so the crop stays square."""
    if not mp_res.multi_hand_landmarks: return None
    lms = mp_res.multi_hand_landmarks[0].landmark
    xs, ys = [lm.x*W for lm in lms], [lm.y*H for lm in lms]
    cx, cy = (max(xs)+min(xs))/2, (max(ys)+min(ys))/2
    side = max(max(xs)-min(xs), max(ys)-min(ys)) * (1 + pad)
    side = min(side, float(min(W, H)))            # cannot exceed image
    half = side / 2
    # Shift center to keep full square inside image
    cx = min(max(cx, half), W - half)
    cy = min(max(cy, half), H - half)
    x1, y1 = int(round(cx - half)), int(round(cy - half))
    x2, y2 = x1 + int(round(side)), y1 + int(round(side))
    return (x1, y1, x2, y2)

def get_mp_landmarks(mp_res, H, W):
    if not mp_res.multi_hand_landmarks: return None
    lms = mp_res.multi_hand_landmarks[0].landmark
    return np.array([[lm.x*W, lm.y*H] for lm in lms], dtype=np.float32)

def preprocess(frame_rgb, bbox, flip_x=False):
    x1, y1, x2, y2 = bbox
    crop = frame_rgb[y1:y2, x1:x2]
    if crop.size == 0: return None
    crop = cv2.resize(crop, (INPUT_SIZE, INPUT_SIZE))
    if flip_x:
        crop = cv2.flip(crop, 1)            # left hand → mirror to right hand for model
    crop = (crop.astype(np.float32)/255.0 - IMG_MEAN) / IMG_STD
    return torch.from_numpy(crop).permute(2,0,1).unsqueeze(0).float().to(DEVICE)

def get_handedness(mp_res):
    """Return 'Right' / 'Left' / None for the first detected hand (camera view)."""
    if not mp_res.multi_handedness:
        return None
    return mp_res.multi_handedness[0].classification[0].label

def umeyama_2d(src, dst):
    """Closed-form 2D similarity (uniform scale s, rotation R, translation t) so that s*R@src + t ≈ dst."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s_c, d_c = src - mu_s, dst - mu_d
    var_s = (s_c ** 2).sum() / len(src)
    cov = (d_c.T @ s_c) / len(src)             # 2x2
    U, S, Vt = np.linalg.svd(cov)
    D = np.eye(2)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[1, 1] = -1
    R = U @ D @ Vt
    s = (S * np.diag(D)).sum() / max(var_s, 1e-12)
    t = mu_d - s * (R @ mu_s)
    return float(s), R.astype(np.float32), t.astype(np.float32)

def verts_to_px(verts3d, joint_img_norm, bbox, flip_x=False):
    """
    Rigid 2D similarity alignment: Procrustes-fit (J_REG @ verts) XY → model joint_img mapped to frame.
    Returns (verts_px, rmse_px) where rmse_px is reprojection error of the 21 joints.
    """
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1

    j2d = joint_img_norm.astype(np.float32).copy()    # (21, 2) in [0,1] of crop
    if flip_x:
        j2d[:, 0] = 1.0 - j2d[:, 0]                   # undo crop horizontal flip
    j2d_px = np.stack([j2d[:, 0] * bw + x1, j2d[:, 1] * bh + y1], axis=1)

    # MPII-ordered joint XY (J_REG yields MANO order; convert to MPII to match joint_img output)
    src_j = mano_to_mpii(J_REG @ verts3d)[:, :2].astype(np.float32)
    # NOTE: do NOT negate Y — MANO finger axis (-Y) already aligns with image (top of image = small pixel Y)
    if flip_x:
        src_j[:, 0] = -src_j[:, 0]

    s, R, t = umeyama_2d(src_j, j2d_px)

    v = verts3d[:, :2].astype(np.float32).copy()
    if flip_x:
        v[:, 0] = -v[:, 0]
    v_px = (s * (v @ R.T) + t)

    j_pred = s * (src_j @ R.T) + t
    rmse = float(np.sqrt(((j_pred - j2d_px) ** 2).sum(axis=1).mean()))
    return v_px.astype(np.int32), rmse


# ─── Drawing functions ────────────────────────────────────────────────────────

def draw_silhouette(img, verts_px, faces, color=COL_SILH):
    overlay = img.copy()
    H, W = img.shape[:2]
    for f in faces:
        p = verts_px[f]
        if (p < -50).any() or (p[:,0] > W+50).any() or (p[:,1] > H+50).any():
            continue
        cv2.fillConvexPoly(overlay, p.reshape(-1,1,2), color)
    cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)

def draw_wireframe(img, verts_px, faces, color=COL_MESH, thickness=1):
    H, W = img.shape[:2]
    for f in faces:
        p = verts_px[f]
        if (p < 0).any() or (p[:,0] >= W).any() or (p[:,1] >= H).any():
            continue
        cv2.line(img, tuple(p[0]), tuple(p[1]), color, thickness)
        cv2.line(img, tuple(p[1]), tuple(p[2]), color, thickness)
        cv2.line(img, tuple(p[2]), tuple(p[0]), color, thickness)

def draw_skeleton(img, lms, edges=MP_EDGES, color=COL_SKEL, thickness=2):
    for a, b in edges:
        if a < len(lms) and b < len(lms):
            cv2.line(img, (int(lms[a][0]),int(lms[a][1])),
                          (int(lms[b][0]),int(lms[b][1])), color, thickness)

def draw_keypoints(img, lms, color=COL_KPT, radius=4):
    for pt in lms:
        cv2.circle(img, (int(pt[0]),int(pt[1])), radius, color, -1)

def draw_hud(img, mode_name):
    H, W = img.shape[:2]
    bar = img[:40,:].copy()
    cv2.rectangle(img, (0,0), (W,40), (0,0,0), -1)
    cv2.addWeighted(bar, 0.3, img[:40,:], 0.7, 0, img[:40,:])
    cv2.putText(img, f"MobRecon  |  Mode: {mode_name}  |  M: cycle  Q: quit",
                (10,28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)


# ─── Signal / Camera ─────────────────────────────────────────────────────────
running = True
def _stop(sig, frame_):
    global running; running = False
signal.signal(signal.SIGINT, _stop)

mp_hands = mp.solutions.hands


def _load_image_any(path):
    """Load BGR uint8 numpy array. Tries cv2.imread, then PIL (handles AVIF if pillow_avif installed)."""
    img = cv2.imread(path)
    if img is not None:
        return img
    try:
        from PIL import Image
        try:
            import pillow_avif  # noqa: F401  registers AVIF decoder
        except ImportError:
            pass
        pil = Image.open(path).convert("RGB")
        rgb = np.array(pil)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception as e:
        sys.exit(f"ERROR: could not load image {path}: {e}")


def _apply_overlays(frame, vpts, mp_lms, mode):
    if mode in ("Silhouette", "All") and vpts is not None:
        draw_silhouette(frame, vpts, FACES)
    if mode in ("Landmarks", "All") and mp_lms is not None:
        draw_skeleton(frame, mp_lms)
        draw_keypoints(frame, mp_lms)
    if mode in ("Mesh", "All") and vpts is not None:
        draw_wireframe(frame, vpts, FACES)


# ═══════════════════════════════════════════════════════════════════════════════
#  STATIC IMAGE MODE
# ═══════════════════════════════════════════════════════════════════════════════
if args.image:
    img_path = args.image
    if not os.path.isabs(img_path):
        img_path = os.path.join(SCRIPT_DIR, img_path)
    if not os.path.isfile(img_path):
        sys.exit(f"ERROR: image not found: {img_path}")

    frame = _load_image_any(img_path)
    H, W = frame.shape[:2]
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    static_tracker = mp_hands.Hands(
        static_image_mode=True, max_num_hands=1,
        min_detection_confidence=0.5)
    mp_res = static_tracker.process(frame_rgb)
    static_tracker.close()

    bbox = get_bbox(mp_res, H, W)
    if bbox is None:
        sys.exit("ERROR: MediaPipe found no hand in image.")
    mp_lms = get_mp_landmarks(mp_res, H, W)
    handed = get_handedness(mp_res)
    flip_for_model = (handed == "Left")

    tensor = preprocess(frame_rgb, bbox, flip_x=flip_for_model)
    if tensor is None:
        sys.exit("ERROR: preprocessing failed (empty crop).")
    with torch.no_grad():
        out = model(tensor)

    verts_3d = out["verts"][0].cpu().numpy()
    joint_img = out["joint_img"][0].cpu().numpy()
    vpts, rmse = verts_to_px(verts_3d, joint_img, bbox, flip_x=flip_for_model)

    print("\n--- Static inference diagnostics ---")
    print(f"  image       : {img_path}  ({W}x{H})")
    print(f"  bbox        : {bbox}")
    print(f"  handedness  : {handed}  (flip_for_model={flip_for_model})")
    print(f"  verts XY    : x[{verts_3d[:,0].min():+.3f}, {verts_3d[:,0].max():+.3f}]"
          f"  y[{verts_3d[:,1].min():+.3f}, {verts_3d[:,1].max():+.3f}]")
    bbox_diag = float(np.hypot(bbox[2] - bbox[0], bbox[3] - bbox[1]))
    rmse_pct = 100.0 * rmse / max(bbox_diag, 1.0)
    print(f"  reproj RMSE : {rmse:.2f} px  ({rmse_pct:.1f}% of bbox diag — {'OK' if rmse_pct < 6 else 'HIGH'})")

    out_paths = []
    for mode in MODES:
        canvas = frame.copy()
        _apply_overlays(canvas, vpts, mp_lms, mode)
        x1, y1, x2, y2 = bbox
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (100, 100, 100), 1)
        draw_hud(canvas, mode)
        out_path = os.path.join(SCRIPT_DIR, f"{args.out_prefix}_{mode}.png")
        cv2.imwrite(out_path, canvas)
        out_paths.append(out_path)
        print(f"  wrote       : {out_path}")
    print("------------------------------------\n")
    sys.exit(0)


# ═══════════════════════════════════════════════════════════════════════════════
#  LIVE CAMERA MODE
# ═══════════════════════════════════════════════════════════════════════════════
hand_tracker = mp_hands.Hands(
    static_image_mode=False, max_num_hands=1,
    min_detection_confidence=0.7, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
if not cap.isOpened(): cap = cv2.VideoCapture(0)
if not cap.isOpened(): sys.exit("Could not open camera.")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("\n" + "="*60)
print("  Demo running — hold a hand in front of the camera.")
print("  Press M to cycle modes, Q to quit.")
print("="*60 + "\n")

EMA_ALPHA = 0.6   # higher = more smoothing
prev_vpts = None

# ─── Main loop ────────────────────────────────────────────────────────────────
try:
    while running:
        ok, frame = cap.read()
        if not ok: break
        # frame = cv2.flip(frame, 1)  # DO NOT FLIP! Model expects a Right hand. Mirroring turns it into a Left hand.
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        H, W = frame.shape[:2]

        mp_res = hand_tracker.process(frame_rgb)
        bbox   = get_bbox(mp_res, H, W)
        mp_lms = get_mp_landmarks(mp_res, H, W)
        handed = get_handedness(mp_res)            # "Right"/"Left"/None
        flip_for_model = (handed == "Left")        # MobRecon = right-hand-only
        cur_mode = MODES[mode_idx]

        if bbox is None:
            prev_vpts = None

        if bbox is not None and mp_lms is not None:
            tensor = preprocess(frame_rgb, bbox, flip_x=flip_for_model)
            if tensor is not None:
                with torch.no_grad():
                    out = model(tensor)

                verts_3d  = out["verts"][0].cpu().numpy()
                joint_img = out["joint_img"][0].cpu().numpy()
                vpts, _   = verts_to_px(verts_3d, joint_img, bbox, flip_x=flip_for_model)

                if prev_vpts is not None and prev_vpts.shape == vpts.shape:
                    vpts = (EMA_ALPHA * prev_vpts + (1 - EMA_ALPHA) * vpts).astype(np.int32)
                prev_vpts = vpts

                _apply_overlays(frame, vpts, mp_lms, cur_mode)

            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1,y1), (x2,y2), (100,100,100), 1)

        draw_hud(frame, cur_mode)
        cv2.imshow("MobRecon Demo", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"): break
        elif key == ord("m"):
            mode_idx = (mode_idx + 1) % len(MODES)
            print(f"  → Mode: {MODES[mode_idx]}")

except KeyboardInterrupt:
    pass
finally:
    cap.release()
    cv2.destroyAllWindows()
    hand_tracker.close()
    print("Demo closed.")
