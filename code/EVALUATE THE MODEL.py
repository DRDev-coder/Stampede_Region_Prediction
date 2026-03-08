#!/usr/bin/env python3
"""
Dual-Head CANNet Video Stampede Risk Analysis
- Upload video file
- Extract frames and analyze density
- Track density flow in 8x8 grid cells
- Identify 2 most stampede-prone regions based on:
  * Positive density change (crowd accumulation)
  * Inward flow > outward flow
  * Rate of density increase
- SAVES RESULTS FOR EVERY 10TH PROCESSED FRAME
"""

import os
import tkinter as tk
from tkinter import filedialog
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# -------------------------
# CONFIGURATION
# -------------------------
MODEL_PATH = r"C:\Users\vicky\Desktop\CVVV\code\TRAINEDMODEL.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_HEIGHT, IMG_WIDTH = 512, 512
GRID_SIZE = 8  # 8x8 grid

# Video processing settings
FRAME_SKIP = 2  # Process every Nth frame (for speed)
TEMPORAL_WINDOW = 5  # Number of frames to analyze for flow
DANGER_THRESHOLD = 0.1  # Threshold for density increase to flag danger
SAVE_EVERY_N_FRAMES = 10  # Save visualization every N processed frames

print("="*80)
print("DUAL-HEAD CANNET - VIDEO STAMPEDE RISK ANALYZER")
print("="*80)
print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    try:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    except:
        pass
print("="*80 + "\n")

# -------------------------
# MODEL ARCHITECTURE (same as training)
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
    def __init__(self, pretrained=False, fpn_out_channels=256):
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
    def __init__(self, pretrained=False):
        super().__init__()
        self.backbone = ResNetFPNBackbone(pretrained=pretrained, fpn_out_channels=256)
        self.context = ContextualModule(256, 256, sizes=(1,2,3,6))
        self.backend_cfg = [256, 256, 256, 128, 64]
        self.backend = make_layers(self.backend_cfg, in_channels=256, batch_norm=True, dilation=True)
        self.density_head = nn.Conv2d(64, 1, kernel_size=1)
        self.heatmap_head = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        feats = self.backbone(x)
        feats = self.context(feats)
        x = self.backend(feats)
        density_out = self.density_head(x)
        heatmap_logits = self.heatmap_head(x)
        return density_out, heatmap_logits

# -------------------------
# IMAGE PREPROCESSING
# -------------------------
def add_coord_channels(img):
    """Add coordinate channels (x, y) to image"""
    H, W = img.shape[0], img.shape[1]
    xs = np.linspace(-1, 1, W, dtype=np.float32)
    ys = np.linspace(-1, 1, H, dtype=np.float32)
    xv = np.tile(xs[np.newaxis, :], (H, 1))
    yv = np.tile(ys[:, np.newaxis], (1, W))
    coord = np.stack([xv, yv], axis=2)
    img5 = np.concatenate([img, coord], axis=2)
    return img5

def preprocess_frame(frame):
    """Preprocess video frame for model input"""
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Resize to model input size
    img_resized = cv2.resize(img_rgb, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)
    
    # Normalize to [0, 1]
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    # Add coordinate channels
    img_with_coords = add_coord_channels(img_normalized)
    
    # Convert to tensor (C, H, W)
    img_tensor = torch.from_numpy(img_with_coords).permute(2, 0, 1).float()
    
    # Add batch dimension (1, C, H, W)
    img_tensor = img_tensor.unsqueeze(0)
    
    return img_tensor, img_resized

# -------------------------
# GRID ANALYSIS
# -------------------------
def compute_grid_densities(density_map, grid_size=8):
    """Compute density for each grid cell"""
    H, W = density_map.shape
    cell_h = H // grid_size
    cell_w = W // grid_size
    
    grid_densities = np.zeros((grid_size, grid_size))
    
    for i in range(grid_size):
        for j in range(grid_size):
            y_start = i * cell_h
            y_end = (i + 1) * cell_h if i < grid_size - 1 else H
            x_start = j * cell_w
            x_end = (j + 1) * cell_w if j < grid_size - 1 else W
            
            cell_density = density_map[y_start:y_end, x_start:x_end].sum()
            grid_densities[i, j] = cell_density
    
    return grid_densities

def compute_flow_metrics(grid_history):
    """
    Analyze density flow over time to detect stampede risk
    
    Risk indicators:
    1. Positive density change (accumulation)
    2. High rate of increase
    3. Sustained increase over multiple frames
    
    Returns: risk_scores (grid_size x grid_size), flow_direction
    """
    if len(grid_history) < 2:
        return np.zeros((GRID_SIZE, GRID_SIZE)), np.zeros((GRID_SIZE, GRID_SIZE))
    
    grid_size = grid_history[0].shape[0]
    
    # Compute temporal derivatives (density change)
    density_changes = []
    for i in range(1, len(grid_history)):
        change = grid_history[i] - grid_history[i-1]
        density_changes.append(change)
    
    density_changes = np.array(density_changes)
    
    # Risk scoring:
    # 1. Average positive change (accumulation)
    avg_change = np.mean(density_changes, axis=0)
    positive_change = np.maximum(avg_change, 0)
    
    # 2. Rate of change (acceleration)
    if len(density_changes) >= 2:
        acceleration = np.mean(density_changes[1:] - density_changes[:-1], axis=0)
        positive_acceleration = np.maximum(acceleration, 0)
    else:
        positive_acceleration = np.zeros_like(positive_change)
    
    # 3. Consistency of increase (how many frames show increase)
    increase_count = np.sum(density_changes > 0, axis=0)
    consistency_score = increase_count / len(density_changes)
    
    # 4. Current density (absolute risk)
    current_density = grid_history[-1]
    normalized_density = current_density / (np.max(current_density) + 1e-6)
    
    # Combined risk score
    risk_score = (
        0.35 * positive_change / (np.max(positive_change) + 1e-6) +
        0.25 * positive_acceleration / (np.max(np.abs(positive_acceleration)) + 1e-6) +
        0.20 * consistency_score +
        0.20 * normalized_density
    )
    
    # Flow direction: positive = inward flow, negative = outward flow
    flow_direction = avg_change
    
    return risk_score, flow_direction

def analyze_stampede_risk(grid_history):
    """
    Identify top 2 most stampede-prone regions
    
    Returns: top_risk_cells, risk_scores, risk_info
    """
    risk_scores, flow_direction = compute_flow_metrics(grid_history)
    
    # Find top 2 highest risk cells
    flat_indices = np.argsort(risk_scores.ravel())[::-1][:2]
    top_cells = [(idx // GRID_SIZE, idx % GRID_SIZE) for idx in flat_indices]
    
    # Analyze each top cell
    risk_info = []
    for row, col in top_cells:
        risk_level = risk_scores[row, col]
        flow = flow_direction[row, col]
        
        # Classify risk
        if flow > DANGER_THRESHOLD:
            status = "HIGH RISK - CROWD ACCUMULATION"
            color = "red"
        elif flow > 0:
            status = "MODERATE RISK - Slight Increase"
            color = "orange"
        elif flow < -DANGER_THRESHOLD:
            status = "SAFE - Crowd Dispersing"
            color = "green"
        else:
            status = "STABLE - No Significant Change"
            color = "yellow"
        
        # Get density trend
        if len(grid_history) >= 2:
            history_list = list(grid_history)
            recent = history_list[-5:]  # last up to 5 entries
            recent_densities = [g[row, col] for g in recent]
            if len(recent_densities) >= 2 and recent_densities[-1] > recent_densities[0]:
                trend = "INCREASING"
            elif len(recent_densities) >= 2 and recent_densities[-1] < recent_densities[0]:
                trend = "DECREASING"
            else:
                trend = "STABLE"
        else:
            trend = "N/A"
        
        risk_info.append({
            'cell': (row, col),
            'risk_score': risk_level,
            'flow': flow,
            'status': status,
            'color': color,
            'trend': trend,
            'current_density': grid_history[-1][row, col]
        })
    
    return top_cells, risk_scores, risk_info

def get_cell_bounds(row, col, grid_size, img_h, img_w):
    """Get pixel boundaries for a grid cell"""
    cell_h = img_h // grid_size
    cell_w = img_w // grid_size
    
    y_start = row * cell_h
    y_end = (row + 1) * cell_h if row < grid_size - 1 else img_h
    x_start = col * cell_w
    x_end = (col + 1) * cell_w if col < grid_size - 1 else img_w
    
    return y_start, y_end, x_start, x_end

# -------------------------
# VISUALIZATION
# -------------------------
def create_frame_visualization(frame_resized, density_map, grid_densities, 
                               risk_info, frame_num, total_frames):
    """Create visualization for a single frame"""
    
    fig = plt.figure(figsize=(20, 10))
    
    H, W = frame_resized.shape[:2]
    cell_h = H // GRID_SIZE
    cell_w = W // GRID_SIZE
    
    # 1. Input frame with grid
    ax1 = plt.subplot(2, 3, 1)
    ax1.imshow(frame_resized)
    ax1.set_title(f'Frame {frame_num}/{total_frames}\nInput Video Frame', 
                  fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Draw grid
    for i in range(GRID_SIZE + 1):
        ax1.axhline(y=i * cell_h, color='white', linewidth=0.5, alpha=0.5)
        ax1.axvline(x=i * cell_w, color='white', linewidth=0.5, alpha=0.5)
    
    # 2. Density map
    ax2 = plt.subplot(2, 3, 2)
    density_vis = ax2.imshow(density_map, cmap='jet', interpolation='bilinear')
    ax2.set_title(f'Density Map\nTotal Count: {density_map.sum():.0f}', 
                  fontsize=14, fontweight='bold')
    ax2.axis('off')
    plt.colorbar(density_vis, ax=ax2, fraction=0.046, pad=0.04)
    
    # 3. Grid density heatmap
    ax3 = plt.subplot(2, 3, 3)
    grid_vis = ax3.imshow(grid_densities, cmap='YlOrRd', interpolation='nearest')
    ax3.set_title('Grid Cell Densities', fontsize=14, fontweight='bold')
    ax3.set_xticks(range(GRID_SIZE))
    ax3.set_yticks(range(GRID_SIZE))
    
    # Annotate with density values
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            ax3.text(j, i, f'{grid_densities[i, j]:.0f}',
                    ha="center", va="center", color="black", fontsize=7)
    
    plt.colorbar(grid_vis, ax=ax3, fraction=0.046, pad=0.04)
    
    # Highlight risk cells
    if risk_info:
        for info in risk_info:
            row, col = info['cell']
            rect = patches.Rectangle((col - 0.5, row - 0.5), 1, 1,
                                    linewidth=3, edgecolor=info['color'],
                                    facecolor='none', linestyle='--')
            ax3.add_patch(rect)
    
    # 4. Risk zones on frame
    ax4 = plt.subplot(2, 3, 4)
    ax4.imshow(frame_resized)
    ax4.set_title('STAMPEDE RISK ZONES', fontsize=14, fontweight='bold', color='red')
    ax4.axis('off')
    
    # Draw grid
    for i in range(GRID_SIZE + 1):
        ax4.axhline(y=i * cell_h, color='white', linewidth=0.5, alpha=0.3)
        ax4.axvline(x=i * cell_w, color='white', linewidth=0.5, alpha=0.3)
    
    # Highlight risk zones
    if risk_info:
        for idx, info in enumerate(risk_info, 1):
            row, col = info['cell']
            y_start, y_end, x_start, x_end = get_cell_bounds(row, col, GRID_SIZE, H, W)
            
            rect = patches.Rectangle((x_start, y_start), x_end - x_start, y_end - y_start,
                                    linewidth=5, edgecolor=info['color'],
                                    facecolor=info['color'], alpha=0.3)
            ax4.add_patch(rect)
            
            # Add warning label
            label_x = x_start + (x_end - x_start) // 2
            label_y = y_start + 30
            ax4.text(label_x, label_y, f'RISK #{idx}\n({row},{col})',
                    ha='center', va='top', color='white',
                    fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor=info['color'], alpha=0.8))
    
    # 5. Risk analysis table
    ax5 = plt.subplot(2, 3, 5)
    ax5.axis('off')
    ax5.set_title('RISK ANALYSIS', fontsize=14, fontweight='bold', color='red')
    
    if risk_info:
        table_data = []
        for idx, info in enumerate(risk_info, 1):
            row, col = info['cell']
            table_data.append([
                f"Zone #{idx}",
                f"({row},{col})",
                f"{info['current_density']:.0f}",
                f"{info['flow']:+.2f}",
                info['trend'],
                info['status']
            ])
        
        table = ax5.table(cellText=table_data,
                         colLabels=['Zone', 'Position', 'Density', 'Flow', 'Trend', 'Status'],
                         cellLoc='left',
                         loc='center',
                         colWidths=[0.12, 0.12, 0.12, 0.12, 0.15, 0.37])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 3)
        
        # Color code rows
        for i, info in enumerate(risk_info, 1):
            for j in range(6):
                cell = table[(i, j)]
                cell.set_facecolor(info['color'])
                cell.set_alpha(0.3)
    else:
        ax5.text(0.5, 0.5, 'Analyzing...', ha='center', va='center', fontsize=16)
    
    # 6. Legend and info
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    ax6.set_title('RISK INDICATORS', fontsize=14, fontweight='bold')
    
    legend_text = """
    RISK ASSESSMENT:
    
    🔴 HIGH RISK - Rapid crowd accumulation
       • Positive density flow (inward movement)
       • Sustained increase over time
       • Potential bottleneck or congestion
    
    🟠 MODERATE RISK - Slight density increase
       • Minor accumulation detected
       • Monitor for escalation
    
    🟡 STABLE - No significant change
       • Density relatively constant
    
    🟢 SAFE - Crowd dispersing
       • Negative flow (outward movement)
       • Decreasing density
    
    FLOW METRICS:
    • Positive flow = Inward/Accumulation
    • Negative flow = Outward/Dispersal
    • Zero flow = Stable
    """
    
    ax6.text(0.1, 0.9, legend_text, fontsize=10, verticalalignment='top',
            family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    return fig

# -------------------------
# MODEL LOADING
# -------------------------
def load_model(model_path, device):
    """Load trained model"""
    print(f"Loading model from: {model_path}")
    
    model = DualHeadCANNet(pretrained=False).to(device)
    
    # Adapt conv1 to 5 channels
    backbone_conv1 = model.backbone.conv1
    if getattr(backbone_conv1, 'in_channels', 3) != 5:
        w = backbone_conv1.weight.data
        new_conv = nn.Conv2d(5, backbone_conv1.out_channels, 
                            kernel_size=backbone_conv1.kernel_size,
                            stride=backbone_conv1.stride, 
                            padding=backbone_conv1.padding, bias=False)
        with torch.no_grad():
            new_conv.weight[:, :3, :, :].copy_(w)
            new_conv.weight[:, 3:, :, :].zero_()
        model.backbone.conv1 = new_conv.to(device)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        model.load_state_dict(checkpoint)
        print("✓ Loaded model weights")
    
    model.eval()
    print(f"✓ Model ready on {device}\n")
    return model

# -------------------------
# VIDEO PROCESSING
# -------------------------
def process_video(model, video_path, device, output_dir):
    """Process video and analyze stampede risk - SAVES EVERY 10TH PROCESSED FRAME"""
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\n{'='*80}")
    print(f"VIDEO ANALYSIS")
    print(f"{'='*80}")
    print(f"Video: {os.path.basename(video_path)}")
    print(f"FPS: {fps}")
    print(f"Total frames: {total_frames}")
    print(f"Processing every {FRAME_SKIP} frames")
    print(f"Saving visualization every {SAVE_EVERY_N_FRAMES} processed frames")
    print(f"Temporal window: {TEMPORAL_WINDOW} frames")
    print(f"{'='*80}\n")
    
    # Storage for temporal analysis
    grid_history = deque(maxlen=TEMPORAL_WINDOW)
    frame_results = []
    
    frame_count = 0
    processed_count = 0
    saved_count = 0
    
    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Skip frames for efficiency
            if frame_count % FRAME_SKIP != 0:
                continue
            
            processed_count += 1
            
            # Preprocess frame
            img_tensor, img_resized = preprocess_frame(frame)
            img_tensor = img_tensor.to(device)
            
            # Predict density
            density_pred, _ = model(img_tensor)
            density_full = F.interpolate(density_pred, size=(IMG_HEIGHT, IMG_WIDTH), 
                                        mode='bilinear', align_corners=True)
            density_map = density_full[0, 0].cpu().numpy()
            
            # Compute grid densities
            grid_densities = compute_grid_densities(density_map, GRID_SIZE)
            grid_history.append(grid_densities)
            
            # Analyze stampede risk (need at least 2 frames)
            if len(grid_history) >= 2:
                top_cells, risk_scores, risk_info = analyze_stampede_risk(grid_history)
            else:
                top_cells = []
                risk_scores = np.zeros((GRID_SIZE, GRID_SIZE))
                risk_info = []
            
            # Store results
            frame_results.append({
                'frame_num': frame_count,
                'processed_num': processed_count,
                'frame': img_resized,
                'density_map': density_map,
                'grid_densities': grid_densities,
                'risk_info': risk_info,
                'total_density': density_map.sum()
            })
            
            # Save visualization every N processed frames
            if processed_count % SAVE_EVERY_N_FRAMES == 0:
                result = frame_results[-1]
                fig = create_frame_visualization(
                    result['frame'],
                    result['density_map'],
                    result['grid_densities'],
                    result['risk_info'],
                    result['frame_num'],
                    total_frames
                )
                
                output_path = os.path.join(output_dir, f"analysis_frame_{result['frame_num']:06d}_proc_{processed_count:04d}.png")
                fig.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                saved_count += 1
                print(f"✓ Saved frame {result['frame_num']} (processed #{processed_count}) - Total saved: {saved_count}")
            
            # Progress update
            if processed_count % 10 == 0:
                print(f"Processed {processed_count} frames (original frame {frame_count}/{total_frames})...", end='\r')
    
    cap.release()
    print(f"\n\n✓ Processed {processed_count} frames from {total_frames} total frames")
    print(f"✓ Saved {saved_count} visualization images\n")
    
    # Generate summary report
    if len(frame_results) > 0:
        generate_summary_report(frame_results, video_path, output_dir)
    
    return frame_results

def generate_summary_report(frame_results, video_path, output_dir):
    """Generate text summary of stampede risk analysis"""
    
    report_path = os.path.join(output_dir, "stampede_risk_report.txt")
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("STAMPEDE RISK ANALYSIS REPORT\n")
        f.write("="*80 + "\n\n")
        f.write(f"Video: {os.path.basename(video_path)}\n")
        f.write(f"Frames Analyzed: {len(frame_results)}\n")
        f.write(f"Analysis Date: {cv2.__version__}\n\n")
        
        f.write("-"*80 + "\n")
        f.write("HIGH RISK INCIDENTS DETECTED\n")
        f.write("-"*80 + "\n\n")
        
        high_risk_count = 0
        for result in frame_results:
            if result['risk_info']:
                for info in result['risk_info']:
                    if 'HIGH RISK' in info['status']:
                        high_risk_count += 1
                        f.write(f"Frame {result['frame_num']:04d}:\n")
                        f.write(f"  Location: Row {info['cell'][0]}, Col {info['cell'][1]}\n")
                        f.write(f"  Status: {info['status']}\n")
                        f.write(f"  Density: {info['current_density']:.1f}\n")
                        f.write(f"  Flow: {info['flow']:+.2f}\n")
                        f.write(f"  Trend: {info['trend']}\n\n")
        
        if high_risk_count == 0:
            f.write("No high-risk incidents detected.\n\n")
        else:
            f.write(f"\nTotal high-risk incidents: {high_risk_count}\n\n")
        
        f.write("-"*80 + "\n")
        f.write("OVERALL STATISTICS\n")
        f.write("-"*80 + "\n\n")
        
        # Compute statistics
        avg_density = np.mean([r['total_density'] for r in frame_results])
        max_density = np.max([r['total_density'] for r in frame_results])
        
        f.write(f"Average Total Density: {avg_density:.1f}\n")
        f.write(f"Maximum Total Density: {max_density:.1f}\n")
        f.write(f"High Risk Frames: {high_risk_count}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")
    
    print(f"✓ Summary report saved: {report_path}")

# -------------------------
# FILE DIALOG
# -------------------------
def select_file():
    """Open file dialog to select image or video"""
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    
    file_path = filedialog.askopenfilename(
        title="Select Image or Video for Analysis",
        filetypes=[
            ("Video files", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv"),
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
            ("All files", "*.*")
        ]
    )
    
    root.destroy()
    return file_path

# -------------------------
# SINGLE IMAGE PROCESSING
# -------------------------
def process_single_image(model, image_path, device, output_dir):
    """Process single image (original functionality)"""
    from scipy.ndimage import maximum_filter
    
    print(f"\nProcessing image: {os.path.basename(image_path)}")
    
    # Read and preprocess
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_with_coords = add_coord_channels(img_normalized)
    img_tensor = torch.from_numpy(img_with_coords).permute(2, 0, 1).float().unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        density_pred, _ = model(img_tensor)
        density_full = F.interpolate(density_pred, size=(IMG_HEIGHT, IMG_WIDTH), 
                                    mode='bilinear', align_corners=True)
        density_map = density_full[0, 0].cpu().numpy()
    
    # Compute grid
    grid_densities = compute_grid_densities(density_map, GRID_SIZE)
    
    # Find top 2 densest cells (static analysis)
    flat_indices = np.argsort(grid_densities.ravel())[::-1][:2]
    top_cells = [(idx // GRID_SIZE, idx % GRID_SIZE) for idx in flat_indices]
    
    # Create fake risk info for visualization
    risk_info = []
    for idx, (row, col) in enumerate(top_cells):
        risk_info.append({
            'cell': (row, col),
            'risk_score': grid_densities[row, col],
            'flow': 0.0,
            'status': f'Density Rank #{idx+1}',
            'color': 'red' if idx == 0 else 'orange',
            'trend': 'STATIC',
            'current_density': grid_densities[row, col]
        })
    
    # Visualize
    fig = create_frame_visualization(img_resized, density_map, grid_densities, 
                                    risk_info, 1, 1)
    
    output_path = os.path.join(output_dir, f"analysis_{os.path.basename(image_path)}")
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"✓ Results saved: {output_path}")
    print(f"\nTop 2 Densest Cells:")
    for idx, (row, col) in enumerate(top_cells, 1):
        print(f"  #{idx}: Row {row}, Col {col} - Density: {grid_densities[row, col]:.1f}")

# -------------------------
# MAIN
# -------------------------
def main():
    # Check model
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at {MODEL_PATH}")
        return
    
    # Load model
    model = load_model(MODEL_PATH, DEVICE)
    
    # Select file
    print("Opening file dialog...")
    file_path = select_file()
    
    if not file_path:
        print("No file selected. Exiting.")
        return
    
    print(f"\nSelected: {file_path}")
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(MODEL_PATH), "stampede_analysis")
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine if video or image
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    file_ext = os.path.splitext(file_path)[1].lower()
    
    try:
        if file_ext in video_extensions:
            print("\n🎬 VIDEO MODE: Analyzing temporal flow for stampede risk")
            print(f"📊 Saving visualization every {SAVE_EVERY_N_FRAMES} processed frames")
            frame_results = process_video(model, file_path, DEVICE, output_dir)
            print(f"\n✓ Analysis complete! Results saved to: {output_dir}")
        else:
            print("\n📷 IMAGE MODE: Static density analysis")
            process_single_image(model, file_path, DEVICE, output_dir)
            print(f"\n✓ Analysis complete! Results saved to: {output_dir}")
            
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()