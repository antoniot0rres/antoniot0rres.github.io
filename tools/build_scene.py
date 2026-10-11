"""Build the layered scene used by index.html from a single photo.

    python3 tools/build_scene.py my-photo.jpg            # writes scene/ next to index.html

What it produces (scene/):
  image.jpg            the photo, resized to 2400 px wide
  labels.png           R = planar region id per pixel, G/B = 1st/2nd surface hidden behind it
  labels2.png          R = 3rd hidden surface
  fill1..3.jpg         colour of those hidden surfaces (mirror-filled from the region's own texture)
  layers.json          per region: plane (depth = a*u + b*v + c) and bounding box, in uv units

Method:
  1. Relative depth with Depth Anything (ViT-B, ONNX, CPU), downloaded on first run (~390 MB).
  2. Depth edges snapped to the photo's edges with an image-guided weighted median.
  3. Superpixels on colour + depth, each fitted with a plane; neighbours are merged greedily while
     the merged plane still fits (boundary disagreement and RMS thresholds).
  4. Region borders re-decided per pixel with a guided filter, so they follow real edges.
  5. Surfaces hidden behind nearer regions are filled up to the largest shift the effect can make.

Why planes: when the viewpoint moves, a plane moves by an affine map, and affine maps keep straight
lines straight. Every region is a plane, so nothing in the picture bends.

Requirements: pip install numpy opencv-contrib-python-headless scikit-image scipy onnxruntime
"""
import argparse
import heapq
import json
import os
import sys
import urllib.request

import cv2
import numpy as np
import onnxruntime as ort
from scipy.ndimage import distance_transform_edt
from skimage.segmentation import slic

MODEL_URL = ("https://github.com/fabio-sim/Depth-Anything-ONNX/releases/download/"
             "v1.0.0/depth_anything_vitb14.onnx")

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("image", help="source photo (any size; landscape works best)")
ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scene"),
                help="output folder (default: scene/ next to index.html)")
ap.add_argument("--width", type=int, default=2400, help="output width in px (default 2400)")
ap.add_argument("--model", default=os.path.expanduser("~/.cache/depth-anything/depth_anything_vitb14.onnx"))
ap.add_argument("--debug", action="store_true", help="also save intermediate .npy arrays")
args = ap.parse_args()
out_dir = os.path.abspath(args.out)
os.makedirs(out_dir, exist_ok=True)


def log(msg):
    print(msg, flush=True)


# ---------------- parameters ----------------
N_SEG = 7000          # initial superpixels
DEPTH_W = 4.0         # weight of depth vs colour when forming superpixels
TAU_BOUNDARY = 0.035  # max plane disagreement along a shared boundary to merge (depth units 0..1)
TAU_RMS = 0.014       # max RMS residual of the merged plane
MIN_AREA = 900        # regions smaller than this (px) are absorbed by their best neighbour
MAX_REGIONS = 250     # region ids must fit in one byte of labels.png
FILL_BAND = 0.05      # upper bound (fraction of width) for extending hidden surfaces
FILL_EPS = 0.03       # a neighbour must be this much nearer to count as an occluder
STRENGTH_MAX = 0.045  # largest effect strength the fill supports (page default is 0.034)

# ---------------- 0. photo and depth ----------------
src = cv2.imread(args.image, cv2.IMREAD_COLOR)
if src is None:
    sys.exit(f"Could not read {args.image}")
h0, w0 = src.shape[:2]
W = args.width
H = int(round(W * h0 / w0))
bgr = cv2.resize(src, (W, H), interpolation=cv2.INTER_AREA if w0 > W else cv2.INTER_CUBIC)

if not os.path.exists(args.model):
    os.makedirs(os.path.dirname(args.model), exist_ok=True)
    log(f"downloading depth model to {args.model} ...")
    urllib.request.urlretrieve(MODEL_URL, args.model + ".part")
    os.replace(args.model + ".part", args.model)

log("estimating depth ...")
tw = 1204                                  # inference width (multiple of 14); detail vs speed
th = int(round(tw * H / W / 14)) * 14
sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
x = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (tw, th), interpolation=cv2.INTER_AREA)
x = ((x.astype(np.float32) / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225])
x = x.transpose(2, 0, 1)[None].astype(np.float32)
raw = sess.run(None, {sess.get_inputs()[0].name: x})[0].squeeze().astype(np.float32)
lo, hi = np.percentile(raw, [0.5, 99.8])
dep = np.clip((cv2.resize(raw, (W, H), interpolation=cv2.INTER_CUBIC) - lo) / (hi - lo), 0, 1)
dep = dep.astype(np.float32)
# The network blurs depth across object borders (ramps of several px). A median weighted by colour
# similarity picks one side or the other instead of averaging, so every depth step lands exactly on
# the photo's edge and no "in-between" slivers survive to become floating fragments.
dep = cv2.ximgproc.weightedMedianFilter(bgr, dep, 13, 25.5, cv2.ximgproc.WMF_JAC)
dep = cv2.ximgproc.weightedMedianFilter(bgr, dep, 9, 25.5, cv2.ximgproc.WMF_JAC)

log("segmenting into planes ...")
ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
U = (xs + 0.5) / W
V = (ys + 0.5) / H
Xn = (xs + 0.5) / W           # plane coordinates: both axes in units of image width
Yn = (ys + 0.5) / W

# ---------------- 1. superpixels on colour + depth ----------------
lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
feat = np.dstack([lab[..., 0] / 255.0, (lab[..., 1] - 128) / 255.0, (lab[..., 2] - 128) / 255.0,
                  dep * DEPTH_W / 3.0])
sp = slic(feat, n_segments=N_SEG, compactness=0.06, channel_axis=-1, start_label=0,
          enforce_connectivity=True, convert2lab=False, max_num_iter=10)
n = sp.max() + 1
log(f"superpixels: {n}")

# normal-equation sums per region: S = sum [x y 1]^T [x y 1], t = sum [x y 1]^T d, q = sum d^2
lf = sp.ravel()
X1 = np.stack([Xn.ravel(), Yn.ravel(), np.ones(H * W, np.float32)], 1).astype(np.float64)
D = dep.ravel().astype(np.float64)
S = np.zeros((n, 3, 3))
for i in range(3):
    for j in range(3):
        S[:, i, j] = np.bincount(lf, X1[:, i] * X1[:, j], n)
T = np.stack([np.bincount(lf, X1[:, i] * D, n) for i in range(3)], 1)
Q = np.bincount(lf, D * D, n)
A = np.bincount(lf, minlength=n).astype(np.float64)


SLOPE_RIDGE = 4e-4    # slope prior: a region needs ~2% of the width of extent along an axis
                      # before a tilt along that axis is trusted (thin strips stay fronto-parallel)


def fit(Si, ti, qi, ai):
    reg = np.diag([SLOPE_RIDGE, SLOPE_RIDGE, 0.0]) * ai
    beta = np.linalg.solve(Si + reg + np.eye(3) * 1e-12, ti)
    rss = qi - 2 * beta @ ti + beta @ Si @ beta
    return beta, np.sqrt(max(rss, 0) / max(ai, 1))


# ---------------- 2. adjacency with boundary pixel pairs ----------------
pairs = {}
for dy, dx in ((0, 1), (1, 0)):
    a = sp[: H - dy, : W - dx]
    b = sp[dy:, dx:]
    m = a != b
    pa, pb = a[m], b[m]
    py = (ys[: H - dy, : W - dx][m] + dy * 0.5) / W + 0.5 / W
    px = (xs[: H - dy, : W - dx][m] + dx * 0.5) / W + 0.5 / W
    for k1, k2, X, Y in zip(np.minimum(pa, pb), np.maximum(pa, pb), px, py):
        pairs.setdefault((int(k1), int(k2)), []).append((X, Y))
edges = {k: np.array(v, np.float64) for k, v in pairs.items()}
nbrs = [set() for _ in range(n)]
for (p, q) in edges:
    nbrs[p].add(q)
    nbrs[q].add(p)

parent = list(range(n))
beta = [None] * n
rms = np.zeros(n)
for i in range(n):
    beta[i], rms[i] = fit(S[i], T[i], Q[i], A[i])


def disagreement(p, q, pts):
    P = np.c_[pts, np.ones(len(pts))]
    return float(np.median(np.abs(P @ beta[p] - P @ beta[q])))


def key(p, q):
    return (p, q) if p < q else (q, p)


heap = []
for (p, q), pts in edges.items():
    heapq.heappush(heap, (disagreement(p, q, pts), p, q))

alive = np.ones(n, bool)
while heap:
    w, p, q = heapq.heappop(heap)
    if not (alive[p] and alive[q]) or key(p, q) not in edges:
        continue
    pts = edges[key(p, q)]
    w_now = disagreement(p, q, pts)
    if w_now > w + 1e-9:                 # stale entry: re-queue with current cost
        heapq.heappush(heap, (w_now, p, q))
        continue
    if w_now > TAU_BOUNDARY:
        break
    Sm, Tm, Qm, Am = S[p] + S[q], T[p] + T[q], Q[p] + Q[q], A[p] + A[q]
    bm, rm = fit(Sm, Tm, Qm, Am)
    if rm > TAU_RMS:
        continue
    # merge q into p
    S[p], T[p], Q[p], A[p], beta[p], rms[p] = Sm, Tm, Qm, Am, bm, rm
    alive[q] = False
    parent[q] = p
    del edges[key(p, q)]
    nbrs[p].discard(q)
    for r in list(nbrs[q]):
        if r == p:
            continue
        e = edges.pop(key(q, r))
        k = key(p, r)
        edges[k] = np.vstack([edges[k], e]) if k in edges else e
        nbrs[r].discard(q)
        nbrs[r].add(p)
        nbrs[p].add(r)
    nbrs[q] = set()
    for r in nbrs[p]:
        heapq.heappush(heap, (disagreement(p, r, edges[key(p, r)]), p, r))


def find(i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


# ---------------- 3. absorb small regions into the most compatible neighbour ----------------
def absorb_small(limit_area, max_regions):
    changed = True
    while changed:
        changed = False
        live = [i for i in range(n) if alive[i]]
        live.sort(key=lambda i: A[i])
        too_many = len(live) > max_regions
        for i in live:
            if not alive[i] or (A[i] >= limit_area and not too_many):
                continue
            if not nbrs[i]:
                continue
            best = min(nbrs[i], key=lambda r: disagreement(i, r, edges[key(i, r)]))
            Sm, Tm, Qm, Am = S[best] + S[i], T[best] + T[i], Q[best] + Q[i], A[best] + A[i]
            S[best], T[best], Q[best], A[best] = Sm, Tm, Qm, Am
            beta[best], rms[best] = fit(Sm, Tm, Qm, Am)
            alive[i] = False
            parent[i] = best
            del edges[key(i, best)]
            nbrs[best].discard(i)
            for r in list(nbrs[i]):
                if r == best:
                    continue
                e = edges.pop(key(i, r))
                k = key(best, r)
                edges[k] = np.vstack([edges[k], e]) if k in edges else e
                nbrs[r].discard(i)
                nbrs[r].add(best)
                nbrs[best].add(r)
            nbrs[i] = set()
            changed = True
            if too_many and sum(alive) <= max_regions:
                too_many = False
        if sum(alive) > max_regions:
            limit_area *= 1.5
            changed = True


absorb_small(MIN_AREA, MAX_REGIONS)

roots = np.array([find(i) for i in range(n)])
ids = {r: k for k, r in enumerate(sorted(set(roots.tolist())))}
lab_map = np.vectorize(ids.get)(roots)[sp].astype(np.int32)
R = len(ids)
planes = np.zeros((R, 3))
for r, k in ids.items():
    planes[k] = beta[r]
log(f"planar regions: {R}")

# ---------------- 3b. snap region borders to the photo's own edges ----------------
# Superpixel borders are only approximately on image edges. A soft mask per region, smoothed by a
# guided filter that uses the photo as guide, re-decides every border pixel by colour evidence.
GF_R, GF_EPS = 5, 2e-4
guide = bgr.astype(np.float32) / 255.0
best_v = np.full((H, W), -1.0, np.float32)
best_l = lab_map.copy()
pad = 3 * GF_R
for k in range(R):
    m = lab_map == k
    yy, xx = np.where(m)
    y0, y1 = max(0, yy.min() - pad), min(H, yy.max() + pad + 1)
    x0, x1 = max(0, xx.min() - pad), min(W, xx.max() + pad + 1)
    soft = cv2.ximgproc.guidedFilter(guide[y0:y1, x0:x1], m[y0:y1, x0:x1].astype(np.float32), GF_R, GF_EPS)
    bv = best_v[y0:y1, x0:x1]
    upd = soft > bv
    bv[upd] = soft[upd]
    best_l[y0:y1, x0:x1][upd] = k
# keep each region connected to its own body: drop stray islands created by the refinement
refined = best_l
for k in range(R):
    m = (refined == k).astype(np.uint8)
    if not m.any():
        continue
    nlab, cc, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=4)
    if nlab <= 2:
        continue
    keep = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    stray = (cc > 0) & (cc != keep)
    refined[stray] = lab_map[stray]
lab_map = refined

# ---------------- 3c. silhouettes belong to the nearer object ----------------
# Along a strong depth step, the few pixels right at the border are almost always part of the near
# object (a frame's rubber gasket, a dark outline, a contact shadow). Give them to the near region,
# otherwise they stay glued to the far background and visibly detach when the view moves.
SIL_R, SIL_GAP = 3, 0.10
P0 = np.stack([Xn, Yn, np.ones_like(Xn)], -1).astype(np.float32)
pd = np.einsum("hwk,hwk->hw", P0, planes[lab_map].astype(np.float32))
best_d, best_lab = pd.copy(), lab_map.copy()
for dy in range(-SIL_R, SIL_R + 1):
    for dx in range(-SIL_R, SIL_R + 1):
        if dy * dy + dx * dx > SIL_R * SIL_R or (dy == 0 and dx == 0):
            continue
        nl = np.roll(np.roll(lab_map, dy, 0), dx, 1)
        # depth of the neighbour's plane evaluated here (the near object's plane extended 1-3 px)
        nd = np.einsum("hwk,hwk->hw", P0, planes[nl].astype(np.float32))
        take = (nd > pd + SIL_GAP) & (nd > best_d)
        best_d[take], best_lab[take] = nd[take], nl[take]
# np.roll wraps around; keep the outer frame untouched
best_lab[:SIL_R], best_lab[-SIL_R:] = lab_map[:SIL_R], lab_map[-SIL_R:]
best_lab[:, :SIL_R], best_lab[:, -SIL_R:] = lab_map[:, :SIL_R], lab_map[:, -SIL_R:]
lab_map = best_lab
del P0, pd, best_d, best_lab

# plane depth of the owning region at every pixel
P = np.stack([Xn, Yn, np.ones_like(Xn)], -1).astype(np.float64)
own = np.einsum("hwk,hwk->hw", P, planes[lab_map]).astype(np.float32)
own = np.clip(own, 0, 1)

# ---------------- 4. hidden-surface fill (up to K surfaces per pixel) ----------------
# A pixel hidden behind a near object can be revealed as more than one surface, depending on where
# the viewer moves (e.g. behind a window mullion: another piece of frame, then the landscape).
# Each pixel keeps the K nearest hidden surfaces, ordered nearest first.
K = 3
band = max(3, int(round(FILL_BAND * W)))
f_lab = np.full((K, H, W), -1, np.int32)
f_dep = np.full((K, H, W), -1.0, np.float32)
f_col = np.zeros((K, H, W, 3), np.uint8)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band + 1, 2 * band + 1))
for k in range(R):
    m = (lab_map == k).astype(np.uint8)
    yy, xx = np.where(m)
    y0, y1 = max(0, yy.min() - band), min(H, yy.max() + band + 1)
    x0, x1 = max(0, xx.min() - band), min(W, xx.max() + band + 1)
    mc = m[y0:y1, x0:x1]
    grow = cv2.dilate(mc, kernel) > 0
    plane_here = (P[y0:y1, x0:x1] @ planes[k]).astype(np.float32)
    # extend only underneath regions that are clearly nearer than this plane, and only as far as
    # the largest relative shift between this plane and that occluder can ever reveal
    gap = own[y0:y1, x0:x1] - plane_here
    dist = cv2.distanceTransform((mc == 0).astype(np.uint8), cv2.DIST_L2, 5)
    reach = gap * STRENGTH_MAX * W * 1.15 + 3
    cand = grow & (mc == 0) & (gap > FILL_EPS) & (dist <= reach)
    if not cand.any():
        continue
    crop = bgr[y0:y1, x0:x1]
    # Mirror fill: copy the region's own pixel reflected across the nearest border point. Texture
    # (leaves, tiles, grain) is preserved and the seam is continuous, unlike diffusion inpainting.
    # Mirror around a border pulled 2 px inside the region: the outermost ring of a region often
    # carries the neighbour's edge (a dark gasket, an anti-aliased halo) and would be copied as a stripe.
    core = cv2.erode(mc, np.ones((5, 5), np.uint8))
    src_m = core if core.sum() >= 0.25 * mc.sum() else mc
    iy, ix = distance_transform_edt(src_m == 0, return_distances=False, return_indices=True)
    py, px = np.mgrid[0:y1 - y0, 0:x1 - x0]
    qy = np.clip(2 * iy - py, 0, y1 - y0 - 1)
    qx = np.clip(2 * ix - px, 0, x1 - x0 - 1)
    ok = src_m[qy, qx] > 0
    # A region only extends behind an occluder as far as its own mirrored texture reaches: a thin
    # strip is not stretched into a wide smear; farther surfaces fill the rest.
    cand &= ok
    if not cand.any():
        continue
    c_col = crop[qy, qx]
    c_lab = np.where(cand, k, -1).astype(np.int32)
    c_dep = np.where(cand, plane_here, -1.0).astype(np.float32)
    # insertion into the per-pixel list, nearest hidden surface first
    for j in range(K):
        sl, sd, sc = f_lab[j, y0:y1, x0:x1], f_dep[j, y0:y1, x0:x1], f_col[j, y0:y1, x0:x1]
        swap = (c_lab >= 0) & (c_dep > sd)
        if not swap.any():
            continue
        ol, od, oc = sl[swap].copy(), sd[swap].copy(), sc[swap].copy()
        sl[swap], sd[swap], sc[swap] = c_lab[swap], c_dep[swap], c_col[swap]
        c_lab[swap], c_dep[swap], c_col[swap] = ol, od, oc

# ---------------- 5. export ----------------
cv2.imwrite(os.path.join(out_dir, "image.jpg"), bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])
# labels.png: R = region id, G = hidden surface 1, B = hidden surface 2 (0 = none, ids from 1)
# labels2.png: R = hidden surface 3
lab_png = np.zeros((H, W, 3), np.uint8)
lab_png[..., 2] = lab_map + 1
lab_png[..., 1] = f_lab[0] + 1
lab_png[..., 0] = f_lab[1] + 1
cv2.imwrite(os.path.join(out_dir, "labels.png"), lab_png)
lab2 = np.zeros((H, W, 3), np.uint8)
lab2[..., 2] = f_lab[2] + 1
cv2.imwrite(os.path.join(out_dir, "labels2.png"), lab2)
# Fill colours. Where a slot is empty, a very smooth copy of the photo keeps JPEG ringing away
# from the real fill pixels and keeps the files small.
smooth = cv2.resize(cv2.resize(bgr, (W // 32, H // 32), interpolation=cv2.INTER_AREA), (W, H),
                    interpolation=cv2.INTER_CUBIC)
for j in range(K):
    img_j = np.where((f_lab[j] >= 0)[..., None], f_col[j], smooth)
    cv2.imwrite(os.path.join(out_dir, f"fill{j + 1}.jpg"), img_j, [cv2.IMWRITE_JPEG_QUALITY, 88])

regions = []
for k in range(R):
    yy, xx = np.where((lab_map == k) | (f_lab == k).any(0))
    a, b, c = planes[k]
    regions.append([
        # plane in uv (u = x/W, v = y/H, y down): depth = a*u + b*v + c
        round(float(a), 6), round(float(b) * H / W, 6), round(float(c), 6),
        # bbox in uv
        round(xx.min() / W, 5), round(yy.min() / H, 5), round((xx.max() + 1) / W, 5), round((yy.max() + 1) / H, 5),
    ])
json.dump({"version": 1, "width": W, "height": H, "slots": K, "regions": regions},
          open(os.path.join(out_dir, "layers.json"), "w"), separators=(",", ":"))
if args.debug:
    np.save(os.path.join(out_dir, "labels.npy"), lab_map)
    np.save(os.path.join(out_dir, "fill_lab.npy"), f_lab)
    np.save(os.path.join(out_dir, "planes.npy"), planes)
log(f"done: {R} regions, files written to {out_dir}")
