#!/usr/bin/env python3
"""
Dual-Head ResNet-FPN CANNet (density + annotation heatmap)
- Optimized skip-frames (one-of-5) rotation with progress monitoring
- Fast detection metrics with limits and progress feedback
- 60 epochs with robust validation loop
"""

import os
import time
import random
import csv
import shutil
import math
from datetime import datetime, timedelta

import numpy as np
import cv2
import h5py
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import models
from sklearn.model_selection import train_test_split

# Optional: imageio for GIF generation
try:
    import imageio
    IMAGEIO_AVAILABLE = True
except Exception:
    IMAGEIO_AVAILABLE = False

# -------------------------
# DEVICE & CONFIGURATION
# -------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)
if DEVICE.type == "cuda":
    try:
        print("GPU:", torch.cuda.get_device_name(0))
    except Exception:
        pass

# ---- Edit these paths ----
IMAGES_PATH = r"C:\Users\vicky\Desktop\CVVV\DroneCrowd\train_data\images"
ANNOTATIONS_PATH = r"C:\Users\vicky\Desktop\CVVV\DroneCrowd\train_data\annotations_visual"
DENSITY_MAPS_PATH = r"C:\Users\vicky\Desktop\CVVV\DroneCrowd\train_data\density_maps"
OUTPUT_DIR = r"C:\Users\vicky\Desktop\CVVV\DroneCrowd\output_dual_head"
# --------------------------

IMG_HEIGHT, IMG_WIDTH = 512, 512
OUT_HEIGHT, OUT_WIDTH = 64, 64

BATCH_SIZE = 4
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-4

EPOCHS = 60  # 60 epochs
NUM_WORKERS = 0
VIS_SAMPLES = 5
GIF_SAMPLES = 3

PERIODIC_SAVE_EVERY = 10
GRAD_CLIP = 2.0
USE_AMP = True


SHIFT_ENABLED = True
SHIFT_GROUP = 3
SHUFFLE_WITHIN_BUCKET = True
APPLY_SHIFT_TO_VAL = True

# Detection metrics limits
MAX_PREDICTIONS = 2000
MAX_GT_POINTS = 2000

MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
EVAL_DIR = os.path.join(OUTPUT_DIR, "epoch_evaluations")
HISTORY_DIR = os.path.join(OUTPUT_DIR, "sample_history")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "final_results")
CSV_LOG_PATH = os.path.join(OUTPUT_DIR, "training_log.csv")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)
os.makedirs(HISTORY_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if DEVICE.type == 'cuda':
    torch.cuda.manual_seed_all(SEED)

# -------------------------
# UTIL
# -------------------------
def format_seconds(s):
    try:
        return str(timedelta(seconds=int(round(s))))
    except Exception:
        return "N/A"

def safe_mkdir(p):
    os.makedirs(p, exist_ok=True)

# -------------------------
# MODEL: ResNet-FPN + ContextualModule + backend + heads
# -------------------------
class ContextualModule(nn.Module):
    def __init__(self, features, out_features=256, sizes=(1,2,3,6)):
        super().__init__()
        self.scales = nn.ModuleList([self._make_scale(features, s) for s in sizes])
        self.weight_net = nn.Conv2d(features, features, kernel_size=1)
        self.bottleneck = nn.Conv2d(features * 2, out_features, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_weight(self, feature, scale_feature):
        return torch.sigmoid(self.weight_net(feature - scale_feature))

    def _make_scale(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.shape[2], feats.shape[3]
        multi_scales = [F.interpolate(scale(feats), size=(h, w), mode='bilinear', align_corners=True) for scale in self.scales]
        weights = [self._make_weight(feats, scale) for scale in multi_scales]
        overall_feat = sum(s * w for s, w in zip(multi_scales, weights))
        weights_sum = sum(weights) + 1e-8
        overall_feat = overall_feat / weights_sum
        concat = torch.cat([overall_feat, feats], dim=1)
        return self.relu(self.bottleneck(concat))

def make_layers(cfg, in_channels=256, batch_norm=False, dilation=False):
    d_rate = 2 if dilation else 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            conv = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
            if batch_norm:
                layers += [conv, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

class ResNetFPNBackbone(nn.Module):
    def __init__(self, pretrained=True, fpn_out_channels=256):
        super().__init__()
        res = models.resnet34(pretrained=pretrained)
        self.conv1 = res.conv1
        self.bn1 = res.bn1
        self.relu = res.relu
        self.maxpool = res.maxpool
        self.layer1 = res.layer1
        self.layer2 = res.layer2
        self.layer3 = res.layer3
        self.layer4 = res.layer4

        self.lateral4 = nn.Conv2d(512, fpn_out_channels, kernel_size=1)
        self.lateral3 = nn.Conv2d(256, fpn_out_channels, kernel_size=1)
        self.lateral2 = nn.Conv2d(128, fpn_out_channels, kernel_size=1)
        self.smooth3 = nn.Conv2d(fpn_out_channels, fpn_out_channels, kernel_size=3, padding=1)
        self.smooth2 = nn.Conv2d(fpn_out_channels, fpn_out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        c1 = self.conv1(x); c1 = self.bn1(c1); c1 = self.relu(c1)
        p = self.maxpool(c1)
        c2 = self.layer1(p)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        p5 = self.lateral4(c5)
        p4 = self.lateral3(c4) + F.interpolate(p5, size=c4.shape[2:], mode='bilinear', align_corners=True)
        p4 = self.smooth3(p4)
        p3 = self.lateral2(c3) + F.interpolate(p4, size=c3.shape[2:], mode='bilinear', align_corners=True)
        p3 = self.smooth2(p3)
        return p3

class DualHeadCANNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = ResNetFPNBackbone(pretrained=pretrained, fpn_out_channels=256)
        self.context = ContextualModule(256, 256, sizes=(1,2,3,6))
        self.backend_cfg = [256, 256, 256, 128, 64]
        self.backend = make_layers(self.backend_cfg, in_channels=256, batch_norm=True, dilation=True)
        self.density_head = nn.Conv2d(64, 1, kernel_size=1)
        self.heatmap_head = nn.Conv2d(64, 1, kernel_size=1)
        self._initialize_weights()

    def forward(self, x):
        feats = self.backbone(x)
        feats = self.context(feats)
        x = self.backend(feats)
        density_out = self.density_head(x)
        heatmap_logits = self.heatmap_head(x)
        return density_out, heatmap_logits

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

# -------------------------
# DATASET (CoordConv channels added)
# -------------------------
class DualCrowdDataset(Dataset):
    def __init__(self, file_list, images_dir, annotations_dir, density_dir,
                 img_size=(IMG_HEIGHT, IMG_WIDTH), out_size=(OUT_HEIGHT, OUT_WIDTH)):
        self.files = file_list
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.density_dir = density_dir
        self.img_h, self.img_w = img_size
        self.out_h, self.out_w = out_size

    def __len__(self):
        return len(self.files)

    def load_image(self, path):
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_w, self.img_h), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        return img

    def add_coord_channels(self, img):
        H, W = img.shape[0], img.shape[1]
        xs = np.linspace(-1, 1, W, dtype=np.float32)
        ys = np.linspace(-1, 1, H, dtype=np.float32)
        xv = np.tile(xs[np.newaxis, :], (H, 1))
        yv = np.tile(ys[:, np.newaxis], (1, W))
        coord = np.stack([xv, yv], axis=2)
        img5 = np.concatenate([img, coord], axis=2)
        return img5

    def load_density_h5(self, path):
        try:
            with h5py.File(path, 'r') as f:
                if 'density' in f:
                    density = f['density'][:]
                else:
                    keys = list(f.keys())
                    if len(keys) == 0:
                        raise RuntimeError("Empty h5 file")
                    density = f[keys[0]][:]
            density = np.asarray(density, dtype=np.float32)
            density_full = cv2.resize(density, (self.img_w, self.img_h), interpolation=cv2.INTER_AREA)
            density_down = cv2.resize(density, (self.out_w, self.out_h), interpolation=cv2.INTER_AREA)
            sum_full = density_full.sum()
            sum_down = density_down.sum()
            if sum_down > 0 and sum_full > 0:
                density_down = density_down * (sum_full / (sum_down + 1e-9))
            return density_full, density_down
        except Exception as e:
            print(f"Warning: Failed to load {path}: {e}")
            return None, None

    def load_annotation(self, path):
        """
        Load annotation heatmap - expects RGB image where red channel marks people locations.
        Red dots (high red channel value) indicate person locations.
        """
        ann = cv2.imread(path)  # Load as BGR
        if ann is None:
            raise FileNotFoundError(f"Annotation not found: {path}")
        
        # Extract RED channel (index 2 in BGR format)
        # The red dots in annotations_visual are in the red channel
        ann_red = ann[:, :, 2].astype(np.float32)
        
        # Normalize to 0-1 range and create binary mask for points
        # High red values (>200) indicate person locations
        ann_binary = (ann_red > 200).astype(np.float32)
        
        # If no points detected, try different threshold or use full red channel
        if ann_binary.sum() == 0:
            # Fallback: normalize red channel directly
            ann_binary = ann_red / 255.0
        
        # Resize to full resolution
        ann_full = cv2.resize(ann_binary, (self.img_w, self.img_h), interpolation=cv2.INTER_AREA)
        
        # For heatmap training, we want values between 0 and 1
        # High values where people are located
        ann_full = np.clip(ann_full, 0.0, 1.0)
        
        # Downsample for training
        ann_down = cv2.resize(ann_full, (self.out_w, self.out_h), interpolation=cv2.INTER_AREA)
        ann_down = np.clip(ann_down, 0.0, 1.0)
        
        return ann_full, ann_down

    def __getitem__(self, idx):
        fname = self.files[idx]
        img_path = os.path.join(self.images_dir, f"{fname}.jpg")
        img = self.load_image(img_path)
        img5 = self.add_coord_channels(img)

        density_path = os.path.join(self.density_dir, f"{fname}.h5")
        density_full, density_down = self.load_density_h5(density_path)

        ann_path = os.path.join(self.annotations_dir, f"{fname}.png")
        ann_full, ann_down = self.load_annotation(ann_path)

        if density_full is None:
            density_full = cv2.GaussianBlur(ann_full, (0, 0), sigmaX=4, sigmaY=4)
            density_down = cv2.resize(density_full, (self.out_w, self.out_h), interpolation=cv2.INTER_AREA)
            if density_down.sum() > 0 and density_full.sum() > 0:
                density_down = density_down * (density_full.sum() / (density_down.sum() + 1e-9))

        img_t = torch.from_numpy(img5).permute(2, 0, 1).float()
        density_down_t = torch.from_numpy(density_down).unsqueeze(0).float()
        ann_down_t = torch.from_numpy(ann_down).unsqueeze(0).float()
        density_full_t = torch.from_numpy(density_full).unsqueeze(0).float()
        ann_full_t = torch.from_numpy(ann_full).unsqueeze(0).float()

        return img_t, density_down_t, ann_down_t, density_full_t, ann_full_t, fname

# -------------------------
# LOSSES
# -------------------------
def gaussian_focal_loss(pred_logits, gt_heatmap, alpha=2.0, beta=4.0, eps=1e-6):
    pred = torch.sigmoid(pred_logits)
    pos_mask = (gt_heatmap > 0.5).float()
    neg_mask = (1 - pos_mask)
    pos_loss = - ((1 - pred) ** alpha) * torch.log(pred + eps) * pos_mask
    neg_weight = (1 - gt_heatmap) ** beta
    neg_loss = - (pred ** alpha) * torch.log(1 - pred + eps) * neg_weight * neg_mask
    num_pos = pos_mask.sum()
    pos_loss_sum = pos_loss.sum()
    neg_loss_sum = neg_loss.sum()
    if num_pos == 0:
        loss = neg_loss_sum
    else:
        loss = (pos_loss_sum + neg_loss_sum) / num_pos
    return loss.mean()

class DensityLoss(nn.Module):
    def __init__(self, alpha=0.5, count_weight=0.1):
        super().__init__()
        self.alpha = alpha
        self.count_weight = count_weight
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()
    def forward(self, pred, target):
        mse_loss = self.mse(pred, target)
        mae_loss = self.mae(pred, target)
        pred_count = pred.sum(dim=(1,2,3))
        target_count = target.sum(dim=(1,2,3))
        count_loss = F.l1_loss(pred_count, target_count)
        return mse_loss + self.alpha * mae_loss + self.count_weight * count_loss

class AnnotationLoss(nn.Module):
    def __init__(self, alpha=2.0, beta=4.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
    def forward(self, logits, target):
        return gaussian_focal_loss(logits, target, alpha=self.alpha, beta=self.beta)

# -------------------------
# OPTIMIZED METRICS with limits and progress
# -------------------------
def nms_peaks(heatmap, thresh=0.2, kernel=3):
    maxm = maximum_filter(heatmap, size=kernel)
    peaks = (heatmap == maxm) & (heatmap >= thresh)
    ys, xs = np.where(peaks)
    scores = heatmap[ys, xs]
    pts = [(int(x), int(y), float(s)) for y, x, s in zip(ys, xs, scores)]
    pts = sorted(pts, key=lambda x: -x[2])
    return pts

def greedy_match(pred_pts, gt_pts, radius):
    """Optimized matching with squared distance"""
    if len(pred_pts) == 0:
        return 0, 0, len(gt_pts)
    if len(gt_pts) == 0:
        return 0, len(pred_pts), 0
    
    # Limit for performance
    if len(pred_pts) > MAX_PREDICTIONS:
        pred_pts = pred_pts[:MAX_PREDICTIONS]
    if len(gt_pts) > MAX_GT_POINTS:
        gt_pts = gt_pts[:MAX_GT_POINTS]
    
    pred_coords = [(float(x), float(y)) for x, y, *_ in pred_pts]
    gt_coords = [(float(x), float(y)) for x, y in gt_pts]
    
    matched_gt = set()
    tp = 0
    radius_sq = radius * radius  # Use squared distance
    
    for px, py in pred_coords:
        best_d_sq = float('inf')
        best_j = -1
        for j, (gx, gy) in enumerate(gt_coords):
            if j in matched_gt:
                continue
            d_sq = (px - gx) ** 2 + (py - gy) ** 2
            if d_sq < best_d_sq:
                best_d_sq = d_sq
                best_j = j
        
        if best_j >= 0 and best_d_sq <= radius_sq:
            tp += 1
            matched_gt.add(best_j)
    
    fp = len(pred_coords) - tp
    fn = len(gt_coords) - tp
    return tp, fp, fn

def compute_detection_metrics_all(pred_heat, gt_heat, radii=(4,8), thresh=0.2):
    """
    Optimized with limits and error handling.
    GT heatmap: binary mask where 1.0 = person location, 0.0 = background
    """
    if pred_heat.size == 0 or gt_heat.size == 0:
        results = {}
        for r in radii:
            results[r] = {'tp': 0, 'fp': 0, 'fn': 0, 'precision': 0.0, 
                         'recall': 0.0, 'f1': 0.0, 'pred_count': 0, 'gt_count': 0}
        return results
    
    # Get ground truth points - look for high values in the heatmap
    # Use higher threshold to find actual marked points (not just slight activations)
    gt_mask = (gt_heat > 0.5).astype(np.uint8)
    gt_ys, gt_xs = np.where(gt_mask > 0)
    gt_pts = [(int(x), int(y)) for x, y in zip(gt_xs, gt_ys)]
    
    # If no GT points found with 0.5 threshold, try lower threshold
    if len(gt_pts) == 0:
        gt_mask = (gt_heat > 0.1).astype(np.uint8)
        gt_ys, gt_xs = np.where(gt_mask > 0)
        gt_pts = [(int(x), int(y)) for x, y in zip(gt_xs, gt_ys)]
    
    pred = np.clip(pred_heat, 0.0, 1.0)
    pred_pts = nms_peaks(pred, thresh=thresh, kernel=3)
    
    # Limit predictions
    if len(pred_pts) > MAX_PREDICTIONS:
        pred_pts = pred_pts[:MAX_PREDICTIONS]
    if len(gt_pts) > MAX_GT_POINTS:
        gt_pts = gt_pts[:MAX_GT_POINTS]
    
    results = {}
    for r in radii:
        try:
            tp, fp, fn = greedy_match(pred_pts, gt_pts, radius=r)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            results[r] = {'tp': tp, 'fp': fp, 'fn': fn, 'precision': prec, 
                         'recall': rec, 'f1': f1, 'pred_count': len(pred_pts), 
                         'gt_count': len(gt_pts)}
        except Exception as e:
            results[r] = {'tp': 0, 'fp': 0, 'fn': 0, 'precision': 0.0, 
                         'recall': 0.0, 'f1': 0.0, 'pred_count': 0, 'gt_count': 0}
    
    return results

# -------------------------
# VISUALIZATION
# -------------------------
def save_dual_comparison(img, gt_density, pred_density, gt_annot, pred_annot, 
                        save_path, counts=None):
    """
    Visualize results with proper annotation handling.
    gt_annot and pred_annot are binary-like heatmaps where high values = person locations
    """
    vmax = max(float(gt_density.max()), float(pred_density.max()), 1e-6)
    gt_density_vis = plt.cm.jet(np.clip(gt_density / vmax, 0, 1))[:, :, :3]
    pred_density_vis = plt.cm.jet(np.clip(pred_density / vmax, 0, 1))[:, :, :3]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes[0, 0].imshow(img); axes[0,0].set_title('Input Image'); axes[0,0].axis('off')
    axes[0, 1].imshow(gt_density_vis); axes[0,1].set_title(f'GT Density\nCount: {gt_density.sum():.0f}'); axes[0,1].axis('off')
    axes[0, 2].imshow(pred_density_vis); axes[0,2].set_title(f'Pred Density\nCount: {pred_density.sum():.0f}'); axes[0,2].axis('off')

    img_gt = (img * 255).astype(np.uint8).copy() if img.max() <= 1.0 else img.astype(np.uint8).copy()
    img_pred = img_gt.copy()

    # Extract GT points - use higher threshold to get actual marked points
    gt_points = (gt_annot > 0.5).astype(np.uint8)
    ys, xs = np.where(gt_points > 0)
    
    # If no points with 0.5 threshold, try 0.1
    if len(ys) == 0:
        gt_points = (gt_annot > 0.1).astype(np.uint8)
        ys, xs = np.where(gt_points > 0)
    
    # Draw GT points in red (limit to reasonable number for visualization)
    MAX_VIS_POINTS = 5000
    if len(ys) > MAX_VIS_POINTS:
        # Sample random subset for visualization
        indices = np.random.choice(len(ys), MAX_VIS_POINTS, replace=False)
        ys = ys[indices]
        xs = xs[indices]
    
    for y, x in zip(ys, xs):
        cv2.circle(img_gt, (x, y), 2, (255, 0, 0), -1)
    
    gt_total_points = int((gt_annot > 0.5).sum())
    if gt_total_points == 0:
        gt_total_points = int((gt_annot > 0.1).sum())
    
    axes[1, 0].imshow(cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title(f'GT Points\nTotal: {gt_total_points}')
    axes[1, 0].axis('off')

    # Extract predicted points using NMS for better point detection
    # Instead of simple thresholding, use peak detection
    pred_sigmoid = 1.0 / (1.0 + np.exp(-pred_annot))  # Already sigmoid in main code, but ensure it here
    
    # Use NMS to find actual peaks (person locations)
    pred_peaks = nms_peaks(pred_sigmoid, thresh=0.3, kernel=5)
    
    # Draw predicted points in green
    drawn_points = 0
    for x, y, score in pred_peaks[:MAX_VIS_POINTS]:  # Limit visualization
        cv2.circle(img_pred, (x, y), 3, (0, 255, 0), -1)
        drawn_points += 1
    
    pred_total_points = len(pred_peaks)
    
    axes[1, 1].imshow(cv2.cvtColor(img_pred, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title(f'Pred Points (NMS)\nTotal: {pred_total_points}')
    axes[1, 1].axis('off')

    error = np.abs(gt_density - pred_density)
    axes[1, 2].imshow(error, cmap='hot')
    axes[1, 2].set_title(f'Absolute Error\nMAE: {error.mean():.4f}')
    axes[1, 2].axis('off')

    if counts:
        gt_count, pred_count, gt_pts, pred_pts = counts
        fig.suptitle(f'GT Count: {int(gt_count)} | Pred Count: {int(pred_count)} | GT Points: {gt_total_points} | Pred Points: {pred_total_points}', 
                    fontsize=16, y=0.98)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

# -------------------------
# CHECKPOINT LOADER
# -------------------------
def load_checkpoint_maybe(path, model, optimizer=None, map_location=None):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=map_location)
    start_epoch = 1
    best_val = None
    if isinstance(ckpt, dict):
        if 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
            if optimizer is not None and 'optimizer_state_dict' in ckpt:
                try:
                    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                except Exception as e:
                    print("Warning: failed to load optimizer state:", e)
            if 'epoch' in ckpt:
                start_epoch = int(ckpt['epoch']) + 1
            if 'best_val_loss' in ckpt:
                best_val = float(ckpt['best_val_loss'])
            if 'val_loss' in ckpt and best_val is None:
                best_val = float(ckpt.get('val_loss', best_val))
        else:
            try:
                model.load_state_dict(ckpt)
                start_epoch = 1
            except Exception as e:
                raise RuntimeError("Checkpoint dict not understood: " + str(e))
    else:
        raise RuntimeError("Unsupported checkpoint format")
    return {'loaded': True, 'start_epoch': start_epoch, 'best_val_loss': best_val}

# -------------------------
# BUCKET INDICES (one-of-5) - robust
# -------------------------
def build_bucket_only_indices(n_items, epoch_index, group=5, shuffle_bucket=True, seed_base=SEED):
    """
    One-of-N rotation: take every Nth sample starting from offset
    offset = (epoch-1) % group
    """
    if n_items <= 0:
        return []
    
    if group <= 1:
        indices = list(range(n_items))
    elif group >= n_items:
        offset = (epoch_index - 1) % group
        indices = [i for i in range(offset, n_items, group)]
        if len(indices) == 0:
            indices = list(range(n_items))
    else:
        offset = (epoch_index - 1) % group
        indices = [i for i in range(offset, n_items, group)]
    
    if len(indices) == 0:
        indices = list(range(n_items))
    
    if shuffle_bucket and len(indices) > 1:
        rnd = random.Random(seed_base + epoch_index * 131 + ((epoch_index - 1) % group) * 7)
        rnd.shuffle(indices)
    
    return indices

# -------------------------
# TRAIN LOOP with progress monitoring
# -------------------------
def train():
    print("\n" + "="*80)
    print("DUAL-HEAD CANNET TRAINING - 60 EPOCHS WITH SKIP-FRAMES (1-of-5)")
    print("="*80)
    print("\nResume options: S - Start NEW, R - Resume")
    choice = input("Enter S or R (default S): ").strip().upper() or "S"
    resume_ckpt = None

    # Build file lists
    img_files = set([f.replace('.jpg','') for f in os.listdir(IMAGES_PATH) if f.endswith('.jpg')])
    ann_files = set([f.replace('.png','') for f in os.listdir(ANNOTATIONS_PATH) if f.endswith('.png')])
    density_files = set([f.replace('.h5','') for f in os.listdir(DENSITY_MAPS_PATH) if f.endswith('.h5')])
    matched = sorted(list(img_files.intersection(ann_files).intersection(density_files)))
    
    if len(matched) == 0:
        print("ERROR: No matched image-annotation-density triplets found! Exiting.")
        return

    if choice == "R":
        resume_ckpt = input("Enter path to checkpoint (.pth) to resume from: ").strip()
        if not resume_ckpt or not os.path.exists(resume_ckpt):
            print("No valid checkpoint provided — starting new training.")
            choice = "S"

    train_files, val_files = train_test_split(matched, test_size=0.2, random_state=SEED)
    print(f"\nDataset: {len(matched)} total samples")
    print(f"Train: {len(train_files)} samples | Val: {len(val_files)} samples")
    print(f"Skip-frames enabled: Using 1-of-{SHIFT_GROUP} samples per epoch")
    print(f"Expected samples per epoch: Train ~{len(train_files) // SHIFT_GROUP} | Val ~{len(val_files) // SHIFT_GROUP}")
    
    train_ds = DualCrowdDataset(train_files, IMAGES_PATH, ANNOTATIONS_PATH, DENSITY_MAPS_PATH)
    val_ds = DualCrowdDataset(val_files, IMAGES_PATH, ANNOTATIONS_PATH, DENSITY_MAPS_PATH)
    
    # Note: val_loader will be recreated each epoch with skip-frames if enabled
    # For now, create a full val_loader for initial setup
    val_loader_full = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=False)

    # Build model
    print("\nBuilding model...")
    model = DualHeadCANNet(pretrained=True).to(DEVICE)

    # Adapt conv1 to 5 channels (RGB + xy coords)
    backbone_conv1 = model.backbone.conv1
    if getattr(backbone_conv1, 'in_channels', 3) != 5:
        try:
            w = backbone_conv1.weight.data
            new_conv = nn.Conv2d(5, backbone_conv1.out_channels, kernel_size=backbone_conv1.kernel_size,
                                 stride=backbone_conv1.stride, padding=backbone_conv1.padding, bias=False)
            with torch.no_grad():
                new_conv.weight[:, :3, :, :].copy_(w)
                new_conv.weight[:, 3:, :, :].zero_()
            model.backbone.conv1 = new_conv.to(DEVICE)
            print("✓ Adapted backbone.conv1 to 5 channels (RGB + xy)")
        except Exception as e:
            print(f"Warning: couldn't adapt conv1: {e}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    density_criterion = DensityLoss(alpha=0.5, count_weight=0.1).to(DEVICE)
    annot_criterion = AnnotationLoss(alpha=2.0, beta=4.0).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=6, verbose=True)
    scaler = torch.amp.GradScaler('cuda') if (USE_AMP and DEVICE.type == 'cuda') else None

    # CSV init/resume
    header = ['epoch','train_loss','val_loss','mae_count','rmse_count',
              'P@4','R@4','F1@4','P@8','R@8','F1@8','lr','epoch_time_s','timestamp']
    
    if choice == "S":
        with open(CSV_LOG_PATH, 'w', newline='') as csvf:
            writer = csv.writer(csvf)
            writer.writerow(header)
        start_epoch = 1
        best_f1_8 = -1.0
        best_mae = float('inf')
        best_path = None
        epoch_times = []
        print("\n✓ Starting NEW training run")
    else:
        info = load_checkpoint_maybe(resume_ckpt, model, optimizer, map_location=DEVICE)
        start_epoch = info['start_epoch']
        best_f1_8 = -1.0
        best_mae = float('inf')
        best_path = None
        if os.path.exists(CSV_LOG_PATH):
            try:
                with open(CSV_LOG_PATH, 'r') as csvf:
                    rows = list(csv.reader(csvf))
                    for row in rows[1:]:
                        try:
                            f1_8 = float(row[10])
                            mae = float(row[3])
                            if f1_8 > best_f1_8 or (abs(f1_8 - best_f1_8) < 1e-6 and mae < best_mae):
                                best_f1_8 = f1_8
                                best_mae = mae
                        except Exception:
                            pass
            except Exception:
                pass
        epoch_times = []
        print(f"\n✓ Resuming from epoch {start_epoch}")

    total_epochs = EPOCHS
    start_total = time.time()
    gif_candidates = val_files[:GIF_SAMPLES]
    n_train = len(train_files)
    n_val = len(val_files)

    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80 + "\n")

    # MAIN TRAINING LOOP
    for epoch in range(start_epoch, total_epochs + 1):
        print(f"\n{'='*80}")
        print(f"EPOCH {epoch}/{total_epochs}")
        print(f"{'='*80}")
        
        epoch_start = time.time()
        model.train()
        train_loss = 0.0
        train_batches = 0

        # Build dataloader with skip-frames
        try:
            if SHIFT_ENABLED:
                group = max(1, min(SHIFT_GROUP, max(1, n_train)))
                bucket_indices = build_bucket_only_indices(n_train, epoch_index=epoch, group=group,
                                                           shuffle_bucket=SHUFFLE_WITHIN_BUCKET, seed_base=SEED)
                
                if len(bucket_indices) == 0:
                    print(f"[WARNING] Empty bucket for epoch {epoch} -> using full dataset")
                    bucket_indices = list(range(n_train))
                    rnd = random.Random(SEED + epoch)
                    rnd.shuffle(bucket_indices)
                
                print(f"[SHIFT] Using {len(bucket_indices)}/{n_train} samples (offset={(epoch-1)%group})")
                
                sampler = SubsetRandomSampler(bucket_indices)
                train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                                          shuffle=False, num_workers=0, pin_memory=False)
            else:
                train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, 
                                         num_workers=0, pin_memory=False)
                print(f"[NO SHIFT] Using full dataset: {n_train} samples")
            
            n_batches = len(train_loader)
            print(f"Total batches: {n_batches}")
            
        except Exception as e:
            print(f"[ERROR] Failed to create DataLoader: {e}")
            import traceback
            traceback.print_exc()
            break

        # TRAINING BATCHES
        for batch_idx, (imgs, d_down, a_down, d_full, a_full, _) in enumerate(train_loader, 1):
            imgs = imgs.to(DEVICE)
            d_down = d_down.to(DEVICE)
            a_down = a_down.to(DEVICE)

            optimizer.zero_grad()

            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    pred_density, pred_annot = model(imgs)
                    if pred_density.shape[2:] != d_down.shape[2:]:
                        pred_density = F.interpolate(pred_density, size=d_down.shape[2:], mode='bilinear', align_corners=True)
                    if pred_annot.shape[2:] != a_down.shape[2:]:
                        pred_annot = F.interpolate(pred_annot, size=a_down.shape[2:], mode='bilinear', align_corners=True)

                    loss_d = density_criterion(pred_density, d_down)
                    loss_a = annot_criterion(pred_annot, a_down)
                    loss = loss_d + loss_a

                scaler.scale(loss).backward()
                if GRAD_CLIP and GRAD_CLIP > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
            else:
                pred_density, pred_annot = model(imgs)
                if pred_density.shape[2:] != d_down.shape[2:]:
                    pred_density = F.interpolate(pred_density, size=d_down.shape[2:], mode='bilinear', align_corners=True)
                if pred_annot.shape[2:] != a_down.shape[2:]:
                    pred_annot = F.interpolate(pred_annot, size=a_down.shape[2:], mode='bilinear', align_corners=True)

                loss_d = density_criterion(pred_density, d_down)
                loss_a = annot_criterion(pred_annot, a_down)
                loss = loss_d + loss_a

                loss.backward()
                if GRAD_CLIP and GRAD_CLIP > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()

            train_loss += float(loss.item())
            train_batches += 1

            # Progress display
            elapsed = time.time() - epoch_start
            progress = batch_idx / float(n_batches) if n_batches > 0 else 1.0
            epoch_eta = (elapsed / progress - elapsed) if progress > 0 else None
            
            if (batch_idx % 20 == 0) or (batch_idx == n_batches):
                epoch_eta_str = format_seconds(epoch_eta) if epoch_eta else "N/A"
                print(f"\r[TRAIN] Batch {batch_idx}/{n_batches} | Loss: {loss.item():.4f} | ETA: {epoch_eta_str}    ", 
                      end='', flush=True)

        avg_train_loss = train_loss / max(train_batches, 1)
        print(f"\n[TRAIN] Completed in {format_seconds(time.time() - epoch_start)}")

        # VALIDATION
        print(f"\n{'='*80}")
        print(f"VALIDATION")
        print(f"{'='*80}")
        
        # Create validation loader with skip-frames if enabled
        if SHIFT_ENABLED and APPLY_SHIFT_TO_VAL:
            group_val = max(1, min(SHIFT_GROUP, max(1, n_val)))
            val_bucket_indices = build_bucket_only_indices(n_val, epoch_index=epoch, group=group_val,
                                                          shuffle_bucket=False, seed_base=SEED)
            
            if len(val_bucket_indices) == 0:
                val_bucket_indices = list(range(n_val))
            
            print(f"[SHIFT VAL] Using {len(val_bucket_indices)}/{n_val} samples (offset={(epoch-1)%group_val})")
            
            val_sampler = SubsetRandomSampler(val_bucket_indices)
            val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, sampler=val_sampler,
                                   shuffle=False, num_workers=0, pin_memory=False)
        else:
            val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, 
                                   num_workers=0, pin_memory=False)
            print(f"[NO SHIFT VAL] Using full validation set: {n_val} samples")
        
        model.eval()
        val_loss = 0.0
        val_batches = 0
        mae_sum = 0.0
        rmse_sum = 0.0
        n_val_samples = 0
        tp4 = fp4 = fn4 = 0
        tp8 = fp8 = fn8 = 0

        val_start_time = time.time()
        total_val_batches = len(val_loader)

        with torch.no_grad():
            for batch_idx, (imgs, d_down, a_down, d_full, a_full, fnames) in enumerate(val_loader, 1):
                imgs = imgs.to(DEVICE)
                d_down = d_down.to(DEVICE)
                a_down = a_down.to(DEVICE)

                pred_density, pred_annot = model(imgs)

                if pred_density.shape[2:] != d_down.shape[2:]:
                    pred_density = F.interpolate(pred_density, size=d_down.shape[2:], mode='bilinear', align_corners=True)
                if pred_annot.shape[2:] != a_down.shape[2:]:
                    pred_annot = F.interpolate(pred_annot, size=a_down.shape[2:], mode='bilinear', align_corners=True)

                loss_d = density_criterion(pred_density, d_down)
                loss_a = annot_criterion(pred_annot, a_down)
                loss = loss_d + loss_a
                val_loss += float(loss.item())
                val_batches += 1

                pred_d_full = F.interpolate(pred_density, size=(IMG_HEIGHT, IMG_WIDTH), mode='bilinear', align_corners=True).cpu().numpy()
                pred_a_full = F.interpolate(pred_annot, size=(IMG_HEIGHT, IMG_WIDTH), mode='bilinear', align_corners=True).cpu().numpy()
                
                B = imgs.size(0)
                for i in range(B):
                    gt_d = d_full[i].cpu().squeeze().numpy()
                    pr_d = pred_d_full[i].squeeze()
                    gt_a = a_full[i].cpu().squeeze().numpy()
                    pr_a = 1.0 / (1.0 + np.exp(-pred_a_full[i].squeeze()))
                    
                    gt_count = float(gt_d.sum())
                    pr_count = float(pr_d.sum())
                    mae_sum += abs(pr_count - gt_count)
                    rmse_sum += (pr_count - gt_count) ** 2
                    n_val_samples += 1
                    
                    # Detection metrics with error handling
                    try:
                        det = compute_detection_metrics_all(pr_a, gt_a, radii=(4,8), thresh=0.3)
                        tp4 += det[4]['tp']; fp4 += det[4]['fp']; fn4 += det[4]['fn']
                        tp8 += det[8]['tp']; fp8 += det[8]['fp']; fn8 += det[8]['fn']
                    except Exception as e:
                        print(f"\n[WARNING] Metrics failed for {fnames[i]}: {e}")
                
                # Progress update
                if (batch_idx % 10 == 0) or (batch_idx == total_val_batches):
                    elapsed_val = time.time() - val_start_time
                    progress = batch_idx / total_val_batches
                    eta_val = (elapsed_val / progress - elapsed_val) if progress > 0 else 0
                    print(f"\r[VAL] Batch {batch_idx}/{total_val_batches} | Loss: {loss.item():.4f} | ETA: {format_seconds(eta_val)}    ", 
                          end='', flush=True)

        print(f"\n[VAL] Completed in {format_seconds(time.time() - val_start_time)}")

        avg_val_loss = val_loss / max(val_batches, 1)
        scheduler.step(avg_val_loss)

        # Compute metrics
        mae = mae_sum / max(n_val_samples, 1)
        rmse = math.sqrt(rmse_sum / max(n_val_samples, 1))
        prec4 = tp4 / (tp4 + fp4) if (tp4 + fp4) > 0 else 0.0
        rec4 = tp4 / (tp4 + fn4) if (tp4 + fn4) > 0 else 0.0
        f14 = 2 * prec4 * rec4 / (prec4 + rec4) if (prec4 + rec4) > 0 else 0.0
        prec8 = tp8 / (tp8 + fp8) if (tp8 + fp8) > 0 else 0.0
        rec8 = tp8 / (tp8 + fn8) if (tp8 + fn8) > 0 else 0.0
        f18 = 2 * prec8 * rec8 / (prec8 + rec8) if (prec8 + rec8) > 0 else 0.0

        # Save best model (by F1@8, tie-break by MAE)
        saved_now = False
        saved_paths = []
        
        if (f18 > best_f1_8) or (abs(f18 - best_f1_8) < 1e-6 and mae < best_mae):
            best_f1_8 = f18
            best_mae = mae
            best_name = f'best_F1{f18:.4f}_MAE{mae:.2f}_ep{epoch:03d}.pth'
            best_path = os.path.join(MODELS_DIR, best_name)
            torch.save(model.state_dict(), best_path)
            saved_now = True
            saved_paths.append(best_path)
            print(f"\n✓ NEW BEST MODEL saved: {best_name}")

            # Save visualizations
            eval_epoch_dir = os.path.join(EVAL_DIR, f'epoch_{epoch:03d}_best')
            safe_mkdir(eval_epoch_dir)
            saved_vis = 0
            
            # Use full validation set for visualizations (not skip-frame subset)
            vis_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, 
                                   num_workers=0, pin_memory=False)
            
            with torch.no_grad():
                for imgs, d_down, a_down, d_full, a_full, fnames in vis_loader:
                    imgs = imgs.to(DEVICE)
                    pred_d, pred_a = model(imgs)
                    pred_d_full = F.interpolate(pred_d, size=(IMG_HEIGHT, IMG_WIDTH), mode='bilinear', align_corners=True).cpu().numpy()
                    pred_a_full = F.interpolate(pred_a, size=(IMG_HEIGHT, IMG_WIDTH), mode='bilinear', align_corners=True).cpu().numpy()
                    B = imgs.size(0)
                    for i in range(B):
                        if saved_vis >= VIS_SAMPLES:
                            break
                        img_np = imgs[i].cpu().permute(1,2,0).numpy()[:, :, :3]
                        if img_np.max() > 1.0:
                            img_np = img_np / 255.0
                        gt_d = d_full[i].cpu().squeeze().numpy()
                        pr_d = pred_d_full[i].squeeze()
                        gt_a = a_full[i].cpu().squeeze().numpy()
                        pr_a = 1.0 / (1.0 + np.exp(-pred_a_full[i].squeeze()))
                        
                        gt_count = float(gt_d.sum())
                        pr_count = float(pr_d.sum())
                        
                        # Count actual points using proper thresholding and NMS
                        gt_pts_mask = (gt_a > 0.5).sum()
                        if gt_pts_mask == 0:
                            gt_pts_mask = (gt_a > 0.1).sum()
                        gt_pts = int(gt_pts_mask)
                        
                        # Use NMS to count predicted points
                        pr_peaks = nms_peaks(pr_a, thresh=0.3, kernel=5)
                        pr_pts = len(pr_peaks)
                        
                        vis_name = f'ep{epoch:03d}_{fnames[i]}.png'
                        vis_path = os.path.join(eval_epoch_dir, vis_name)
                        save_dual_comparison(img_np, gt_d, pr_d, gt_a, pr_a, vis_path, 
                                           counts=(gt_count, pr_count, gt_pts, pr_pts))
                        saved_vis += 1
                        
                        # Save to history
                        sample_hist_dir = os.path.join(HISTORY_DIR, fnames[i])
                        safe_mkdir(sample_hist_dir)
                        hist_path = os.path.join(sample_hist_dir, f'epoch{epoch:03d}.png')
                        shutil.copy(vis_path, hist_path)
                    if saved_vis >= VIS_SAMPLES:
                        break

        # Periodic save
        if PERIODIC_SAVE_EVERY > 0 and (epoch % PERIODIC_SAVE_EVERY == 0):
            periodic_name = f"checkpoint_ep{epoch:03d}.pth"
            periodic_path = os.path.join(MODELS_DIR, periodic_name)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'best_f1_8': best_f1_8,
                'best_mae': best_mae
            }, periodic_path)
            saved_now = True
            saved_paths.append(periodic_path)
            print(f"✓ PERIODIC checkpoint saved: {periodic_name}")

        # CSV logging
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        remaining_epochs = total_epochs - epoch
        est_remaining = avg_epoch_time * remaining_epochs
        elapsed_total = time.time() - start_total
        est_total = elapsed_total + est_remaining
        current_lr = optimizer.param_groups[0]['lr']

        with open(CSV_LOG_PATH, 'a', newline='') as csvf:
            writer = csv.writer(csvf)
            writer.writerow([epoch, round(avg_train_loss,6), round(avg_val_loss,6),
                             round(mae,6), round(rmse,6),
                             round(prec4,6), round(rec4,6), round(f14,6),
                             round(prec8,6), round(rec8,6), round(f18,6),
                             current_lr, round(epoch_time,2), datetime.now().isoformat()])
            csvf.flush()

        # Summary
        print(f"\n{'='*80}")
        print(f"EPOCH {epoch}/{total_epochs} SUMMARY")
        print(f"{'='*80}")
        print(f"Train Loss:     {avg_train_loss:.6f}")
        print(f"Val Loss:       {avg_val_loss:.6f}")
        print(f"MAE:            {mae:.4f}")
        print(f"RMSE:           {rmse:.4f}")
        print(f"P@4/R@4/F1:     {prec4:.4f} / {rec4:.4f} / {f14:.4f}")
        print(f"P@8/R@8/F1:     {prec8:.4f} / {rec8:.4f} / {f18:.4f}")
        print(f"Best F1@8:      {best_f1_8:.4f} (MAE: {best_mae:.4f})")
        print(f"LR:             {current_lr:.2e}")
        print(f"-" * 80)
        print(f"Epoch time:     {format_seconds(epoch_time)}")
        print(f"Avg/epoch:      {format_seconds(avg_epoch_time)}")
        print(f"Elapsed total:  {format_seconds(elapsed_total)}")
        print(f"Est. remaining: {format_seconds(est_remaining)}")
        print(f"Est. total:     {format_seconds(est_total)}")
        if saved_now:
            print(f"Saved:          {', '.join([os.path.basename(p) for p in saved_paths])}")
        print(f"{'='*80}\n")

    # TRAINING COMPLETED
    total_time = time.time() - start_total
    final_model_path = os.path.join(MODELS_DIR, f'final_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pth')
    torch.save(model.state_dict(), final_model_path)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETED!")
    print("="*80)
    print(f"Total time:     {format_seconds(total_time)}")
    print(f"Best F1@8:      {best_f1_8:.4f}")
    print(f"Best MAE:       {best_mae:.4f}")
    print(f"Final model:    {final_model_path}")
    print(f"Best model:     {best_path}")
    print(f"Training log:   {CSV_LOG_PATH}")
    print(f"Results dir:    {OUTPUT_DIR}")
    print("="*80)

    # TorchScript export
    try:
        model.eval()
        dummy = torch.randn(1, 5, IMG_HEIGHT, IMG_WIDTH).to(DEVICE)
        traced = torch.jit.trace(model, dummy, strict=False)
        ts_path = os.path.join(MODELS_DIR, f'traced_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pt')
        traced.save(ts_path)
        print(f"✓ TorchScript model: {ts_path}")
    except Exception as e:
        print(f"TorchScript export failed: {e}")

    # GIF creation
    if IMAGEIO_AVAILABLE:
        print("\nCreating progress GIFs...")
        created = 0
        for fname in gif_candidates:
            sample_dir = os.path.join(HISTORY_DIR, fname)
            if not os.path.exists(sample_dir):
                continue
            frames = sorted([os.path.join(sample_dir, f) for f in os.listdir(sample_dir) if f.endswith('.png')])
            if len(frames) < 2:
                continue
            imgs = [imageio.imread(f) for f in frames]
            gif_path = os.path.join(RESULTS_DIR, f'{fname}_progress.gif')
            imageio.mimsave(gif_path, imgs, duration=0.8)
            print(f"✓ GIF: {gif_path}")
            created += 1
            if created >= GIF_SAMPLES:
                break
    else:
        print("\nSkipping GIF creation (imageio not installed)")

    print(f"\nAll results saved to: {OUTPUT_DIR}")
    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    train()