# generate_density_h5_gpu.py
"""
Generate .h5 density maps from red-dot annotation visuals using GPU Gaussian smoothing.

Requirements:
  - Python3
  - torch (with CUDA for GPU smoothing) or CPU-only torch (will run conv on CPU)
  - h5py
  - opencv-python
  - numpy
  - matplotlib (optional, for visualization)
"""

import os
import cv2
import h5py
import math
import torch
import numpy as np
from scipy.ndimage import gaussian_filter  # only used for fallback if torch conv unavailable
import matplotlib.pyplot as plt
from tqdm import tqdm

# ---------------- CONFIG ----------------
ANNOT_DIR = r"C:\Users\vicky\Desktop\CVVV\DroneCrowd\train_data\annotations_visual"  # red-dot images
OUT_H5_DIR = r"C:\Users\vicky\Desktop\CVVV\DroneCrowd\train_data\density_maps"      # output .h5
OUTPUT_SIZE = (64, 64)   # (H_out, W_out) — set to your network output resolution (e.g., 64)
GAUSSIAN_SIGMA = 2.0     # sigma in output-pixels (float)
VISUALIZE_N = 4          # number of examples to show at end (0 to skip)
THRESH_RED = 120         # absolute threshold for R channel
THRESH_RG_DIFF = 40      # R - G difference threshold
THRESH_RB_DIFF = 40      # R - B difference threshold
WRITE_EMPTY = True       # if no points found, write empty density map (True) or skip file (False)
USE_GPU = True           # use GPU if available

# ---------------- end CONFIG ----------------

os.makedirs(OUT_H5_DIR, exist_ok=True)

device = torch.device("cuda" if (torch.cuda.is_available() and USE_GPU) else "cpu")
print("Device for smoothing:", device)

def find_points_from_annotation_image(img):
    """
    Detect dot centers in annotation image.
    Input: img = cv2.imread(...) (BGR) or grayscale
    Return: list of (x, y) coordinates in image pixel coords (x=col, y=row)
    """
    if img is None:
        return []

    if img.ndim == 3 and img.shape[2] == 3:
        b, g, r = cv2.split(img)
        mask = (r.astype(np.int32) > THRESH_RED) & ((r.astype(np.int32) - g.astype(np.int32)) > THRESH_RG_DIFF) & ((r.astype(np.int32) - b.astype(np.int32)) > THRESH_RB_DIFF)
        mask = mask.astype(np.uint8) * 255
    else:
        gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

    # small morphological cleanup
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)

    # connected components to get centroids
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    points = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= 1:
            cx, cy = centroids[i]
            points.append((float(cx), float(cy)))
    return points

def make_gaussian_kernel(sigma, kernel_size=None, device='cpu', dtype=torch.float32):
    """
    Create a 2D Gaussian kernel (kernel_size odd). sigma in pixels.
    If kernel_size is None, choose size = ceil(6*sigma) | ensure odd and >=3.
    Returns: kernel tensor shape (1,1,kH,kW) on requested device.
    """
    if kernel_size is None:
        k = max(3, int(math.ceil(6 * sigma)))
        if k % 2 == 0:
            k += 1
        kernel_size = k
    k = kernel_size
    ax = np.linspace(-(k-1)/2., (k-1)/2., k)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    kernel = kernel / np.sum(kernel)
    kern_t = torch.from_numpy(kernel.astype(np.float32)).unsqueeze(0).unsqueeze(0)  # 1x1xkHxkW
    return kern_t.to(device=device, dtype=dtype)

def generate_density_map_from_points(points, in_size, out_size, sigma, device):
    """
    Create sparse map (out_size) from points in in_size coords, then smooth with Gaussian on device.
    Returns a numpy array (H_out, W_out) dtype float32 with sum == number of points (within float rounding).
    """
    H_in, W_in = in_size
    H_out, W_out = out_size
    sparse = np.zeros((H_out, W_out), dtype=np.float32)
    if len(points) == 0:
        return sparse

    sx = W_out / float(W_in)
    sy = H_out / float(H_in)
    for (x, y) in points:
        ox = int(round(x * sx))
        oy = int(round(y * sy))
        if 0 <= ox < W_out and 0 <= oy < H_out:
            sparse[oy, ox] += 1.0

    # convert to torch tensor and convolve with gaussian kernel on device
    sparse_t = torch.from_numpy(sparse).unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.float32)  # 1x1xH_outxW_out

    # choose kernel size proportional to sigma
    ksize = max(3, int(math.ceil(6 * sigma)))
    if ksize % 2 == 0:
        ksize += 1
    kernel = make_gaussian_kernel(sigma=sigma, kernel_size=ksize, device=device)
    # padding to preserve size
    pad = ksize // 2
    try:
        smoothed = torch.nn.functional.conv2d(sparse_t, kernel, padding=pad)
        smoothed = smoothed.squeeze().cpu().numpy()
    except Exception as e:
        # fallback to scipy if conv2d fails for any reason
        print("Warning: torch conv2d failed, using scipy gaussian_filter fallback:", e)
        smoothed = gaussian_filter(sparse, sigma=sigma, mode='constant')

    # renormalize so sum equals number of points
    s = smoothed.sum()
    if s > 0:
        smoothed = smoothed * (len(points) / s)
    return smoothed.astype(np.float32)

def save_h5_density(path, density_map):
    with h5py.File(path, 'w') as f:
        f.create_dataset('density', data=density_map.astype(np.float32), compression='gzip')

# ---------------- main loop ----------------
files = sorted([f for f in os.listdir(ANNOT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
print(f"Found {len(files)} annotation images in {ANNOT_DIR}")

examples = []
created = 0
skipped = 0
failed = 0

for fname in tqdm(files, desc="Processing annotations"):
    base = os.path.splitext(fname)[0]
    in_path = os.path.join(ANNOT_DIR, fname)
    out_path = os.path.join(OUT_H5_DIR, base + ".h5")
    try:
        img = cv2.imread(in_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            skipped += 1
            continue
        points = find_points_from_annotation_image(img)
        H_in, W_in = img.shape[0], img.shape[1]
        if len(points) == 0 and not WRITE_EMPTY:
            skipped += 1
            continue
        density = generate_density_map_from_points(points, (H_in, W_in), OUTPUT_SIZE, GAUSSIAN_SIGMA, device)
        save_h5_density(out_path, density)
        created += 1
        if len(examples) < VISUALIZE_N:
            examples.append((img, density, base, points))
    except Exception as e:
        print(f"FAILED {fname}: {e}")
        failed += 1
        continue

print(f"\nDone. Created: {created}, Skipped: {skipped}, Failed: {failed}")
print(f"Saved .h5 to: {OUT_H5_DIR}")

# ---------------- optional visualization ----------------
if VISUALIZE_N > 0 and len(examples) > 0:
    for (img, density, base, pts) in examples:
        fig, axs = plt.subplots(1,3, figsize=(12,4))
        if img.ndim == 3 and img.shape[2] == 3:
            ann_vis = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            ann_vis = img
        axs[0].imshow(ann_vis); axs[0].set_title(f"Annotation visual: {base}"); axs[0].axis('off')
        axs[1].imshow(density, cmap='jet'); axs[1].set_title('Density (out res)'); axs[1].axis('off')
        # overlay detected points
        blank = np.ones((img.shape[0], img.shape[1], 3), dtype=np.uint8) * 255
        for (x,y) in pts:
            cv2.circle(blank, (int(round(x)), int(round(y))), 4, (255,0,0), -1)
        axs[2].imshow(cv2.cvtColor(blank, cv2.COLOR_BGR2RGB)); axs[2].set_title('Detected points'); axs[2].axis('off')
    plt.tight_layout(); plt.show()

print("All finished.")
