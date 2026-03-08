"""
make_visual_annotations_debug.py

Improved .mat parsing + debug logging when parsing fails.

Saves:
 - annotated images -> <split>/annotations_visual/
 - debug info for failed .mat files -> mat_debug_samples.txt

Edit BASE_DIR at top if needed.
"""
import os
import re
import cv2
import numpy as np
import scipy.io as sio
from tqdm import tqdm
from collections.abc import Iterable

# ===== CONFIG =====
BASE_DIR = r"C:\Users\vicky\Desktop\CVVV\DroneCrowd"   # <-- change if needed
SPLITS = ["train_data", "val_data", "test_data"]
IMG_SUBFOLDER = "images"
GT_SUBFOLDER = "ground_truth"
OUT_SUBFOLDER = "annotations_visual"
DOT_COLOR = (0, 0, 255)  # BGR
DOT_REL = 0.01
THICKNESS = -1
MAT_DEBUG_OUT = os.path.join(BASE_DIR, "mat_debug_samples.txt")
MAX_DEBUG_ENTRIES = 20  # how many failing mats to log in detail
# ====================

POSSIBLE_KEYS = [
    "annPoints", "points", "pts", "gt", "locations", "loc", "image_info",
    "annotation", "people", "HeadLocations", "head", "GT"
]

def list_mat_variables(matpath):
    try:
        d = sio.whosmat(matpath)
        # sio.whosmat returns list of (name, shape, dtype)
        return d
    except Exception as e:
        return [("ERROR", str(e), None)]

def load_mat(matpath):
    try:
        return sio.loadmat(matpath, struct_as_record=False, squeeze_me=True)
    except Exception as e:
        # try with Matlab v7.3? loadmat can't load HDF5 .mat v7.3, would need h5py.
        # We'll just return None here.
        return None

def flatten_array_like(obj):
    """Try to convert variant array-like to Nx2 numpy array if possible."""
    if obj is None:
        return None
    arr = None
    if isinstance(obj, np.ndarray):
        if obj.ndim == 2 and obj.shape[1] in (2,3,4):
            arr = obj[:, :2].astype(float)
            return arr
        # if shape (2,N) -> transpose
        if obj.ndim == 2 and obj.shape[0] == 2:
            return obj.T.astype(float)
        # other shapes: try flatten
        if obj.size >= 2:
            try:
                flat = obj.reshape(-1)
            except:
                flat = None
            if flat is not None and flat.size % 2 == 0:
                try:
                    arr = flat.reshape(-1, 2).astype(float)
                    return arr
                except:
                    pass
    # lists or tuples of coordinate pairs
    if isinstance(obj, (list, tuple)):
        try:
            arr = np.array(obj, dtype=float)
            if arr.ndim == 2 and arr.shape[1] >= 2:
                return arr[:, :2]
            if arr.ndim == 1 and arr.size % 2 == 0:
                return arr.reshape(-1, 2)
        except:
            pass
    return None

def extract_points_from_loaded_mat(matdict):
    # 1) try known keys:
    for key in POSSIBLE_KEYS:
        if key in matdict:
            pts = flatten_array_like(matdict[key])
            if pts is not None and pts.shape[0] > 0:
                return pts
            # if it's a nested struct, try deeper
            val = matdict[key]
            # If struct-like, convert to dict
            if hasattr(val, "__dict__"):
                d2 = val.__dict__
                for k2, v2 in d2.items():
                    pts = flatten_array_like(v2)
                    if pts is not None and pts.shape[0] > 0:
                        return pts
    # 2) scan all variables in matdict for anything array-like Nx2
    for k, v in matdict.items():
        if k.startswith("__"):
            continue
        pts = flatten_array_like(v)
        if pts is not None and pts.shape[0] > 0 and pts.shape[1] == 2:
            return pts
        # if it's a struct (numpy object) with attributes
        if hasattr(v, "__dict__"):
            d2 = v.__dict__
            for k2, v2 in d2.items():
                pts = flatten_array_like(v2)
                if pts is not None and pts.shape[0] > 0:
                    return pts
    # not found
    return None

def mat_debug_summary(matpath):
    lines = []
    lines.append("=== DEBUG " + matpath + " ===")
    varlist = list_mat_variables(matpath)
    lines.append("Variables (name, shape, dtype):")
    for name, shape, dtype in varlist:
        lines.append(f"  - {name} | shape={shape} | dtype={dtype}")
    # try loading first-level keys
    loaded = load_mat(matpath)
    if loaded is None:
        lines.append("LOAD FAILED (maybe v7.3 MAT/HDF5).")
        # If v7.3 we can attempt h5py but skip here; user can upload sample if needed.
        return "\n".join(lines)
    lines.append("Top-level keys and types:")
    for k, v in loaded.items():
        if k.startswith("__"):
            continue
        t = type(v).__name__
        try:
            shape = getattr(v, "shape", None)
        except:
            shape = None
        lines.append(f"  - {k}: type={t}, shape={shape}")
        # If small, print preview
        if hasattr(v, "__array__") and np.array(v).size <= 40:
            try:
                lines.append("    preview: " + str(np.array(v).tolist()))
            except:
                pass
    # also attempt to find any Nx2 arrays and show small sample
    found = extract_points_from_loaded_mat(loaded)
    if found is not None:
        lines.append("INFERRED Nx2 points sample (first 10):")
        for row in found[:10]:
            lines.append(f"  {row[0]:.2f}, {row[1]:.2f}")
    else:
        lines.append("NO Nx2 array inferred by heuristics.")
    return "\n".join(lines)

def annotate_image_with_points(img_path, pts, out_path):
    img = cv2.imread(img_path)
    if img is None:
        return False
    h, w = img.shape[:2]
    r = max(1, int(min(w, h) * DOT_REL))
    for p in pts:
        x, y = float(p[0]), float(p[1])
        xi = int(round(np.clip(x, 0, w - 1)))
        yi = int(round(np.clip(y, 0, h - 1)))
        cv2.circle(img, (xi, yi), r, DOT_COLOR, THICKNESS)
    cv2.imwrite(out_path, img)
    return True

def main():
    total_images = 0
    annotated = 0
    missing_or_bad = 0
    debug_entries = 0
    if os.path.exists(MAT_DEBUG_OUT):
        os.remove(MAT_DEBUG_OUT)

    for split in SPLITS:
        img_dir = os.path.join(BASE_DIR, split, IMG_SUBFOLDER)
        gt_dir_candidate = os.path.join(BASE_DIR, split, GT_SUBFOLDER)
        if not os.path.isdir(gt_dir_candidate):
            gt_dir_candidate = os.path.join(BASE_DIR, GT_SUBFOLDER)
        out_dir = os.path.join(BASE_DIR, split, OUT_SUBFOLDER)
        os.makedirs(out_dir, exist_ok=True)

        if not os.path.isdir(img_dir):
            print(f"[WARN] missing image dir for split {split}: {img_dir}")
            continue

        images = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        print(f"Processing {split}: {len(images)} images (GT folder: {gt_dir_candidate})")
        for img_name in tqdm(images, desc=split):
            total_images += 1
            base = os.path.splitext(img_name)[0]
            img_path = os.path.join(img_dir, img_name)

            # try direct GT match patterns
            candidates = [
                os.path.join(gt_dir_candidate, f"GT_{base}.mat"),
                os.path.join(gt_dir_candidate, f"{base}.mat"),
                os.path.join(gt_dir_candidate, f"{base}.npy"),
                os.path.join(gt_dir_candidate, f"{base}.txt"),
                os.path.join(gt_dir_candidate, f"{base}.csv"),
                os.path.join(gt_dir_candidate, f"GT_{base}.txt"),
            ]
            gt_path = None
            for c in candidates:
                if os.path.exists(c):
                    gt_path = c
                    break

            # fallback: try numeric mapping (img011001 -> 011001 or 11001 or 11001 padded)
            if gt_path is None:
                m = re.search(r'(\d{1,7})', base)
                if m:
                    num = int(m.group(1))
                    offsets = [0, -11000, 11000, -10000, 10000]
                    pads = [5,6]
                    for off in offsets:
                        candnum = num + off
                        if candnum <= 0: continue
                        for pad in pads:
                            candname = str(candnum).zfill(pad)
                            for ext in [".mat", ".npy", ".txt", ".csv"]:
                                candpath = os.path.join(gt_dir_candidate, candname + ext)
                                if os.path.exists(candpath):
                                    gt_path = candpath
                                    break
                            if gt_path: break
                        if gt_path: break

            if gt_path is None:
                missing_or_bad += 1
                if debug_entries < MAX_DEBUG_ENTRIES:
                    with open(MAT_DEBUG_OUT, "a", encoding="utf-8") as fh:
                        fh.write(f"NO_GT_FOUND for image: {img_name}\n")
                    debug_entries += 1
                continue

            # try to load and extract points
            pts = None
            if gt_path.lower().endswith(".mat"):
                loaded = load_mat(gt_path)
                if loaded is None:
                    # write debug
                    if debug_entries < MAX_DEBUG_ENTRIES:
                        with open(MAT_DEBUG_OUT, "a", encoding="utf-8") as fh:
                            fh.write(mat_debug_summary(gt_path) + "\n\n")
                        debug_entries += 1
                    missing_or_bad += 1
                    continue
                pts = extract_points_from_loaded_mat(loaded)

            elif gt_path.lower().endswith(".npy"):
                try:
                    arr = np.load(gt_path, allow_pickle=True)
                    pts = flatten_array_like(arr)
                except:
                    pts = None
            elif gt_path.lower().endswith((".txt", ".csv")):
                try:
                    with open(gt_path, "r", encoding="utf-8", errors="ignore") as fh:
                        lines = [l.strip() for l in fh if l.strip()]
                    pts_list = []
                    for ln in lines:
                        parts = re.split(r'[,\s]+', ln)
                        nums = []
                        for t in parts:
                            try:
                                nums.append(float(t))
                            except:
                                pass
                            if len(nums) >= 2:
                                break
                        if len(nums) >= 2:
                            pts_list.append([nums[0], nums[1]])
                    if len(pts_list) > 0:
                        pts = np.array(pts_list)
                except:
                    pts = None

            if pts is None or pts.shape[0] == 0:
                # write debug for the mat to understand structure
                if debug_entries < MAX_DEBUG_ENTRIES:
                    with open(MAT_DEBUG_OUT, "a", encoding="utf-8") as fh:
                        fh.write(mat_debug_summary(gt_path) + "\n\n")
                    debug_entries += 1
                missing_or_bad += 1
                continue

            # annotate and save
            out_path = os.path.join(out_dir, base + ".png")
            ok = annotate_image_with_points(img_path, pts, out_path)
            if ok:
                annotated += 1
            else:
                missing_or_bad += 1

    # summary
    print("\n=== SUMMARY ===")
    print("Total images scanned:   ", total_images)
    print("Annotated images:       ", annotated)
    print("Missing / malformed GT: ", missing_or_bad)
    print(f"Debug samples (up to {MAX_DEBUG_ENTRIES}) written to: {MAT_DEBUG_OUT}")
    print("If many are missing, open mat_debug_samples.txt and paste first ~100 lines here so I can inspect.")
    print("Done.")

if __name__ == "__main__":
    main()
