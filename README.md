# stampede-region-prediction: Crowd Counting + Stampede Risk Analysis

**Authors**: Darshan R & Vignesh Balamurugan M.B 

## Overview
This repository contains the source code for a deep learning pipeline designed to estimate crowd density and highlight potential stampede risk zones in both static images and video streams. 

By leveraging a **Dual-Head CANNet-style architecture** with a **ResNet-34 FPN backbone**, this project aims to provide real-time, actionable insights for crowd management and public safety.

---

## ❓ Why This Project?
In densely populated events or crowded public spaces, the sudden accumulation of people and rapid inward flows can quickly escalate into hazardous situations, including stampedes. Traditional surveillance relies heavily on human monitoring, which is often slow and prone to error in high-density scenarios. 

This project automates the monitoring process by:
- Accurately estimating the number of individuals in a scene.
- Pinpointing exact person locations.
- Analyzing temporal crowd dynamics to **alert authorities of imminent stampede risks** before they happen.

## 🧠 What It Does
The core of this system is a **Dual-Head Architecture**:
1. **Density Estimation Head**: Predicts a continuous density map to accurately count the total number of people in the frame.
2. **Localization Head**: Predicts a discrete annotation heatmap for exact person localization.

**Key Features:**
- **ResNet-34 FPN Backbone**: Adapted for aerial surveillance constraints, featuring a Multi-scale Context Module to capture global scene semantics.
- **Temporal Risk Assessment**: Analyzes consecutive video frames using a grid-based approach to detect rapid accumulation and inward movement (the two primary indicators of a stampede).
- **Video & Image Support**: Fully functional on single images or continuous video streams.

## ⚙️ How It Works (Pipeline)

1. **Data Preprocessing**: Converts raw ground-truth coordinates (`.mat` files) into continuous density maps using Gaussian kernel smoothing, and generates visual annotation overlays.
2. **Model Training**: Trains the dual-head model using a combination of Density Loss and Localization Loss.
3. **Inference & Risk Analysis**: Processes input videos frame-by-frame. The output highlights high-risk zones and generates a comprehensive `stampede_risk_report.txt`.

---

## 📂 Repository Layout

```text
stampede-region-prediction/
│
├── code/
│   ├── RAW IMAGE TO ANNOTATED.py         # Step 1: Visualizes GT as red-dot annotations
│   ├── ANNOTATED IMAGE TO DENSITY MAP.py # Step 2: Generates density maps (.h5) from GT
│   ├── MODEL TRAINING CODE.py            # Step 3: Trains the Dual-Head CANNet
│   ├── EVALUATE THE MODEL.py             # Step 4: Runs inference & temporal risk analysis
│   ├── CONTRIBUTIONS.txt                 # Author credits & module details
│   └── TRAINEDMODEL.pth                  # Pre-trained model weights (if applicable)
│
├── RESULTS/                              # Output directory for evaluations
│   ├── INPUT VIDEO.mp4                   # Sample input video
│   ├── analysis_frame_*.png              # Output frames with risk heatmaps
│   └── stampede_risk_report.txt          # Generated risk logs
│
└── README.md                             # Project documentation
```

---

## 📊 Dataset & Repository Policy
- **No Private Data**: We do not upload any dataset files to this repository to comply with data distribution guidelines.
- **Supported Format**: The pipeline was developed and validated using the public **DroneCrowd** dataset structure.
- **Reproducibility**: To reproduce our results, point the scripts to your local dataset copy (or your own DroneCrowd-like data). Paths are configured at the top of each Python script; nothing in the code assumes private data is bundled here.

---

## 🛠️ Requirements & Installation

A GPU is strongly recommended for training and fast video inference. If using CPU-only, consider reducing the input size or frame skip to keep inference responsive.

1. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   
   # Windows
   .venv\Scripts\activate
   # Linux/Mac
   source .venv/bin/activate
   ```

2. **Install core packages:**
   ```bash
   pip install --upgrade pip
   
   # Install PyTorch (Update the index-url depending on your CUDA version)
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   
   # Install other dependencies
   pip install numpy scipy scikit-learn opencv-python h5py matplotlib tqdm imageio
   ```

---

## 🚀 Quick Start Guide

### Step 1: Prepare Annotations
Edit `BASE_DIR` in `code/RAW IMAGE TO ANNOTATED.py` to point at your dataset directory, then run:
```bash
python "code/RAW IMAGE TO ANNOTATED.py"
```

### Step 2: Generate Density Maps (.h5)
Edit `ANNOT_DIR`, `OUT_H5_DIR`, `OUTPUT_SIZE`, and `GAUSSIAN_SIGMA` in `code/ANNOTATED IMAGE TO DENSITY MAP.py`, then run:
```bash
python "code/ANNOTATED IMAGE TO DENSITY MAP.py"
```
*This produces `.h5` files normalized so the sum approximates the number of people.*

### Step 3: Train the Model
Edit the configuration paths (`IMAGES_PATH`, `ANNOTATIONS_PATH`, `DENSITY_MAPS_PATH`, `OUTPUT_DIR`) at the top of `code/MODEL TRAINING CODE.py`, then run:
```bash
python "code/MODEL TRAINING CODE.py"
```
*Optional knobs: batch size, epochs, AMP (`USE_AMP`), gradient clipping, etc.*

### Step 4: Evaluate and Analyze Video/Images
Set `MODEL_PATH` in `code/EVALUATE THE MODEL.py` to the trained weights (e.g., `code/TRAINEDMODEL.pth`).
```bash
python "code/EVALUATE THE MODEL.py"
```
*For videos, it processes every $N$th frame and saves a visualization every `SAVE_EVERY_N_FRAMES`. Results and stampede risk warnings are outputted to the `RESULTS/` directory.*

---

## 🛑 Troubleshooting

- **.mat Parsing Fails**: If `.mat` annotation parsing fails during Step 1, inspect the generated `mat_debug_samples.txt` and adjust the loading heuristics in the script.
- **Path Errors**: Ensure all dataset paths in the configuration sections of the scripts exist and point to the correct locations.
