# SolarScan – Solar Panel Defect Detection System
## Complete Project Report

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Problem Statement](#2-problem-statement)
3. [Dataset](#3-dataset)
4. [System Architecture](#4-system-architecture)
5. [Model 1 – ResNet-50 CNN Classifier](#5-model-1--resnet-50-cnn-classifier)
6. [GradCAM Label Generation](#6-gradcam-label-generation)
7. [Model 2 – YOLOv8 Object Detector](#7-model-2--yolov8-object-detector)
8. [Damage Percentage System](#8-damage-percentage-system)
9. [Backend API](#9-backend-api)
10. [Frontend Dashboard](#10-frontend-dashboard)
11. [Authentication System](#11-authentication-system)
12. [Results & Metrics](#12-results--metrics)
13. [Technology Stack](#13-technology-stack)
14. [Project Structure](#14-project-structure)
15. [How to Run](#15-how-to-run)
16. [Challenges & Solutions](#16-challenges--solutions)
17. [Summary of Work Done](#17-summary-of-work-done)

---

## 1. Project Overview

**SolarScan** is an end-to-end AI-powered solar panel defect detection and analysis system. It combines a **CNN classifier (ResNet-50)** and a **YOLOv8 object detector** to:

- Detect defects on solar panels from photographs
- Draw **precise bounding boxes** around the damaged region (not the whole image)
- Calculate a **damage percentage** (0–100%) per detection
- Provide **actionable repair recommendations** for each defect type
- Present results through a **modern web dashboard** with charts, history, and diagnosis reports

**GitHub Repository:** https://github.com/Shubhammadiwalar/repoproject1

---

## 2. Problem Statement

Solar panels degrade over time due to various environmental and physical factors. Manual inspection is:
- Time-consuming and expensive
- Inconsistent across inspectors
- Difficult to scale across large solar farms

**Goal:** Build an automated AI system that can analyse a photograph of a solar panel and:
1. Classify the type of defect present
2. Localise the exact damaged region with a bounding box
3. Quantify the severity as a damage percentage
4. Recommend specific repair actions

---

## 3. Dataset

### Source
Custom solar panel defect image dataset with 6 classes.

### Split Summary

| Split | Unique Images | Purpose |
|-------|--------------|---------|
| Train | **925** | Model training |
| Val   | **549** | Hyperparameter tuning & early stopping |
| Test  | **95**  | Final evaluation |
| **Total** | **1,569** | |

### Class Distribution

| Class | Train | Val | Test | Total |
|-------|-------|-----|------|-------|
| Bird-drop | 177 | 104 | 17 | **298** |
| Clean | 169 | 102 | 18 | **289** |
| Dusty | 162 | 97 | 16 | **275** |
| Electrical-damage | 135 | 77 | 13 | **225** |
| Physical-Damage | 132 | 78 | 15 | **225** |
| Snow-Covered | 154 | 92 | 16 | **262** |

### Image Characteristics
- Formats: JPG, JPEG, PNG
- Sizes: Varied (158×318 to 3000×2250 pixels)
- Sources: Real-world solar panel photographs
- Folder structure: `split/ClassName/image.jpg`

---

## 4. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        SolarScan Pipeline                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Input Image                                                      │
│      │                                                            │
│      ▼                                                            │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │  STAGE 1: ResNet-50 CNN (Classification + GradCAM)      │     │
│  │  • Classifies defect type (6 classes)                   │     │
│  │  • GradCAM generates activation heatmap                 │     │
│  │  • Heatmap → tight bounding box labels                  │     │
│  │  • Labels used to train YOLO                            │     │
│  └─────────────────────────────────────────────────────────┘     │
│      │                                                            │
│      ▼ (YOLO label files)                                         │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │  STAGE 2: YOLOv8m Detection Model                       │     │
│  │  • CSPDarknet CNN backbone (pretrained COCO)            │     │
│  │  • PANet feature pyramid neck                           │     │
│  │  • Decoupled detection head                             │     │
│  │  • Output: bbox + class + confidence                    │     │
│  └─────────────────────────────────────────────────────────┘     │
│      │                                                            │
│      ▼                                                            │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │  STAGE 3: Damage Analysis                               │     │
│  │  • damage% = class_weight × confidence × 100           │     │
│  │  • Severity classification (6 levels)                   │     │
│  │  • Diagnosis + repair suggestions                       │     │
│  └─────────────────────────────────────────────────────────┘     │
│      │                                                            │
│      ▼                                                            │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │  STAGE 4: Web Dashboard                                 │     │
│  │  • Flask REST API backend                               │     │
│  │  • Purple-themed dashboard frontend                     │     │
│  │  • Login / Signup authentication                        │     │
│  └─────────────────────────────────────────────────────────┘     │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. Model 1 – ResNet-50 CNN Classifier

### Architecture
- **Base model**: ResNet-50 (pretrained on ImageNet IMAGENET1K_V2)
- **Fine-tuned layers**: layer3, layer4, custom FC head
- **Frozen layers**: layer1, layer2 (feature extraction)
- **Custom head**:
  ```
  Dropout(0.4) → Linear(2048→512) → ReLU → Dropout(0.3) → Linear(512→6)
  ```

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Input size | 224 × 224 |
| Batch size | 32 |
| Epochs | 50 (early stopping, patience=10) |
| Optimizer | AdamW (lr=1e-4, weight_decay=1e-4) |
| Scheduler | CosineAnnealingLR (T_max=50, eta_min=1e-6) |
| Loss | CrossEntropyLoss (label_smoothing=0.1) |
| AMP | Enabled (FP16 on RTX 3050) |

### Data Augmentation (Training)
- Random crop (256→224)
- Random horizontal/vertical flip
- ColorJitter (brightness, contrast, saturation, hue)
- Random rotation (±15°)
- Normalisation (ImageNet mean/std)

### CNN Test Results (95 test images)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Bird-drop | 1.00 | 1.00 | 1.00 | 17 |
| Clean | 0.94 | 0.94 | 0.94 | 18 |
| Dusty | 0.94 | 1.00 | 0.97 | 16 |
| Electrical-damage | 1.00 | 0.92 | 0.96 | 13 |
| Physical-Damage | 1.00 | 1.00 | 1.00 | 15 |
| Snow-Covered | 1.00 | 1.00 | 1.00 | 16 |
| **Overall Accuracy** | | | **0.98** | **95** |
| **Macro Avg** | **0.98** | **0.98** | **0.98** | 95 |

**Saved weights:** `cnn_best.pth` (98 MB)

---

## 6. GradCAM Label Generation

### Problem with Whole-Image Labels
Initially, all YOLO labels were whole-image bounding boxes (`0.5 0.5 1.0 1.0`), meaning YOLO learned to draw a box around the entire panel instead of the actual damaged region.

### Solution: GradCAM-Based Precise Labels

**GradCAM (Gradient-weighted Class Activation Mapping)** uses the trained CNN to find exactly where the damage is:

```
1. Forward pass image through ResNet-50
2. Compute gradients of predicted class score
   w.r.t. last convolutional layer (layer4)
3. Global average pool gradients → weights
4. Weighted sum of feature maps → activation heatmap
5. Apply ReLU → normalise to [0,1]
6. Threshold heatmap (0.40) → binary mask
7. Find contours → fit tight bounding box
8. Add 6% padding → write YOLO label
```

### Results: Box Size Comparison

| Class | Old (whole-image) | New (GradCAM) | Reduction |
|-------|-------------------|---------------|-----------|
| Bird-drop | 100% | **27%** avg | -73% |
| Clean | 100% | **64%** (centred) | -36% |
| Dusty | 100% | **31%** avg | -69% |
| Electrical-damage | 100% | **37%** avg | -63% |
| Physical-Damage | 100% | **37%** avg | -63% |
| Snow-Covered | 100% | **49%** avg | -51% |

**Labels generated:** 1,574 total (925 train + 549 val + 95 test)

---

## 7. Model 2 – YOLOv8 Object Detector

### Architecture
- **Model**: YOLOv8m (medium variant)
- **Parameters**: 25,859,794
- **GFLOPs**: 79.1
- **Backbone**: CSPDarknet (CNN, pretrained on COCO)
- **Neck**: PANet (Path Aggregation Network) feature pyramid
- **Head**: Decoupled detection head (bbox regression + classification)
- **Input**: 640 × 640

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 120 |
| Batch size | 16 |
| Image size | 640 × 640 |
| Optimizer | AdamW (auto-tuned, lr=0.001) |
| AMP | Enabled (FP16) |
| Early stopping | Patience = 25 |
| Cache | RAM caching |
| GPU | RTX 3050 6GB |

### Augmentation Strategy
Reduced from standard to preserve tight GradCAM boxes:

| Augmentation | Value |
|-------------|-------|
| HSV hue shift | 0.015 |
| HSV saturation | 0.7 |
| HSV brightness | 0.4 |
| Rotation | ±10° |
| Translation | 0.05 (reduced) |
| Scale | 0.4 (reduced) |
| Mosaic | 0.8 |
| MixUp | 0.1 |
| Flip LR | 0.5 |

### YOLOv8 Training Results (solar_precise3 run)

| Metric | Best Epoch (119/120) | Final Epoch (120/120) |
|--------|---------------------|----------------------|
| **mAP50** | **0.9001** | 0.8955 |
| **mAP50-95** | **0.6190** | 0.6226 |
| **Precision** | **0.8561** | 0.8702 |
| **Recall** | **0.8534** | 0.8366 |
| Train box loss | — | 0.9442 |
| Train cls loss | — | 0.7349 |
| Val box loss | — | 1.1009 |
| Val cls loss | — | 0.8152 |

**Total training time:** ~5.5 hours on RTX 3050 6GB  
**Best weights:** `runs/detect/runs/solar_precise3/weights/best.pt` (52 MB)

### Training Runs History

| Run Name | Model | Labels | Epochs | Notes |
|----------|-------|--------|--------|-------|
| solar_panel_yolo | YOLOv8m | Whole-image | 100 | Initial run |
| solar_panel_yolo2–7 | YOLOv8m | Whole-image | 100 | Config fixes |
| solar_precise | YOLOv8m | GradCAM | 120 | First precise run |
| solar_precise2 | YOLOv8m | GradCAM | 120 | Tuned augmentation |
| **solar_precise3** | **YOLOv8m** | **GradCAM** | **120** | **Best model** |

---

## 8. Damage Percentage System

### Formula
```
damage% = class_weight × yolo_confidence × 100
```

### Class Weights

| Class | Weight | Max Damage | Rationale |
|-------|--------|------------|-----------|
| Clean | 0.00 | 0% | No damage |
| Dusty | 0.35 | 35% | Light, reversible |
| Bird-drop | 0.60 | 60% | Moderate, corrosive |
| Snow-Covered | 0.50 | 50% | Temporary blockage |
| Physical-Damage | 0.90 | 90% | Structural damage |
| Electrical-damage | 0.95 | 95% | Critical, fire risk |

### Severity Scale

| Range | Level | Colour |
|-------|-------|--------|
| 0% | No Damage | Green |
| 1–19% | Minimal | Light green |
| 20–39% | Low | Yellow |
| 40–59% | Moderate | Orange |
| 60–79% | High | Red |
| ≥80% | **Critical** | Dark red |

### Diagnosis System
Each class has a structured diagnosis with:
- **What happened**: Plain-language description of the defect
- **Impact on panel**: Performance effects (3–4 bullet points)
- **How to improve**: Step-by-step repair recommendations (5–6 steps)

---

## 9. Backend API

### Framework
Flask 3.1.3 with flask-cors and flask-login

### Endpoints

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| GET | `/` | — | Serves login or dashboard |
| GET | `/login.html` | — | Login page |
| GET | `/signup.html` | — | Signup page |
| POST | `/api/auth/register` | — | Create account |
| POST | `/api/auth/login` | — | Login, set session |
| POST | `/api/auth/logout` | ✓ | Clear session |
| GET | `/api/auth/me` | ✓ | Current user info |
| GET | `/api/health` | ✓ | Model status + GPU info |
| GET | `/api/classes` | ✓ | 6 defect classes + metadata |
| POST | `/api/predict` | ✓ | Upload image → full analysis |

### `/api/predict` Response Structure
```json
{
  "success": true,
  "annotated_image": "<base64 JPEG>",
  "primary_class": "Electrical-damage",
  "primary_color": "#DC2626",
  "avg_damage": 91.3,
  "severity": "Critical",
  "detection_count": 1,
  "detections": [{
    "class": "Electrical-damage",
    "confidence": 96.1,
    "damage_pct": 91.3,
    "severity": "Critical",
    "color": "#DC2626",
    "bbox": [45, 32, 280, 198]
  }],
  "diagnosis": {
    "what_happened": "Electrical damage detected (burn marks / arc damage).",
    "impact": ["Panel may produce zero output.", "Risk of fire hazard.", ...],
    "suggestions": ["IMMEDIATELY disconnect panel.", "Call a technician.", ...]
  }
}
```

---

## 10. Frontend Dashboard

### Design
Inspired by modern fintech dashboards (FinSet-style):
- **Colour scheme**: Purple/violet (#7C3AED) accent on white cards
- **Background**: Soft lavender (#E8E4F8)
- **Layout**: Fixed left sidebar + scrollable main content
- **Typography**: Inter font

### Pages

| Page | Description |
|------|-------------|
| **Dashboard** | 4 stat cards + bar chart + donut chart + recent scans table |
| **Analyse Panel** | Drag-drop upload + confidence slider + mini stat cards |
| **Results** | Summary strip + damage meter + annotated image + diagnosis + detection table |
| **History** | Full scan history table for the session |
| **Defect Classes** | 6 class cards with colour coding and damage ranges |
| **Settings** | Model info + detection configuration |

### Charts (Canvas API — no external library)
- **Bar chart**: Detection count + average damage % per class
- **Donut chart**: Severity distribution across all scans

### Key UI Components
- Animated damage severity meter (green → red gradient)
- Colour-coded severity badges
- Confidence bar charts in detection table
- Real-time GPU status indicator in sidebar
- Session scan counter

---

## 11. Authentication System

### Implementation
Flask session-based authentication using `werkzeug.security` for password hashing.

### Flow
```
User visits / → Not logged in → Redirect to /login.html
                                        ↓
                              Enter email + password
                                        ↓
                              POST /api/auth/login
                                        ↓
                              Session cookie set
                                        ↓
                              Redirect to dashboard
```

### Security Features
- Passwords hashed with `werkzeug.security.generate_password_hash` (PBKDF2-SHA256)
- Session-based auth (server-side, not JWT)
- All API endpoints protected with `@login_required` decorator
- 401 response with redirect hint for unauthenticated requests

### Login Page Design (Sleeknote-inspired)
- **Left panel** (white): Logo, "Welcome Back" heading, Google OAuth button (UI), email/password fields, "Keep me logged in" checkbox, "Forgot password?" link, purple submit button
- **Right panel** (light blue): Brand tagline, SVG illustration, feature pills
- Demo credentials hint box

### Signup Page
- Same split layout
- Additional fields: Full name, confirm password
- Live password strength meter (5 levels: Very Weak → Very Strong)
- Terms of service checkbox

### Demo Credentials
```
Email:    admin@solarscan.com
Password: admin123
```

---

## 12. Results & Metrics

### CNN Classifier (ResNet-50)

| Metric | Value |
|--------|-------|
| Test Accuracy | **98.0%** |
| Macro Precision | **0.98** |
| Macro Recall | **0.98** |
| Macro F1-Score | **0.98** |
| Perfect classes (F1=1.0) | Bird-drop, Physical-Damage, Snow-Covered |

### YOLOv8 Detector (solar_precise3)

| Metric | Value |
|--------|-------|
| **mAP50** | **0.9001** |
| **mAP50-95** | **0.6190** |
| **Precision** | **0.8561** |
| **Recall** | **0.8534** |
| Training epochs | 120 |
| Training time | ~5.5 hours |
| GPU | RTX 3050 6GB |

### Bounding Box Precision
- Old (whole-image): 100% of image area
- New (GradCAM): **27–49%** of image area (damage classes)
- Improvement: **51–73% reduction** in box area → boxes now tightly surround actual damage

### Inference Speed
- GPU (RTX 3050): ~15–25ms per image
- CPU fallback: ~200–400ms per image

---

## 13. Technology Stack

### Hardware
| Component | Specification |
|-----------|--------------|
| GPU | NVIDIA GeForce RTX 3050 6GB Laptop GPU |
| CUDA | 12.8 |
| OS | Windows 11 |

### Software

| Library | Version | Purpose |
|---------|---------|---------|
| Python | 3.14.2 | Runtime |
| PyTorch | 2.11.0+cu128 | Deep learning framework |
| Ultralytics | 8.4.34 | YOLOv8 implementation |
| torchvision | 0.26.0 | ResNet-50, transforms |
| OpenCV | 4.13.0 | Image processing, drawing |
| NumPy | 2.4.2 | Array operations |
| Pillow | 12.1.1 | Image loading |
| scikit-learn | 1.8.0 | Classification report, confusion matrix |
| Matplotlib | 3.10.8 | Training curves, plots |
| Seaborn | — | Confusion matrix heatmap |
| Flask | 3.1.3 | Web API server |
| flask-cors | — | Cross-origin requests |
| Werkzeug | 3.1.6 | Password hashing |

---

## 14. Project Structure

```
project/
│
├── backend/
│   └── app.py                  # Flask REST API + Auth (280 lines)
│
├── frontend/
│   ├── index.html              # Dashboard (600+ lines)
│   ├── login.html              # Login page (250 lines)
│   ├── signup.html             # Signup page (280 lines)
│   ├── css/
│   │   ├── style.css           # Dashboard styles (580 lines)
│   │   └── auth.css            # Auth page styles (350 lines)
│   └── js/
│       ├── config.js           # API base URL config
│       └── app.js              # Dashboard logic + charts (400 lines)
│
├── predict.py                  # Core detection + damage % + suggestions (450 lines)
├── train_yolo.py               # YOLOv8 training script (120 lines)
├── train_cnn.py                # ResNet-50 training script (200 lines)
├── generate_labels.py          # GradCAM label generator (200 lines)
├── visualise_labels.py         # Label preview tool (70 lines)
├── dataset.yaml                # YOLO dataset config
│
├── cnn_best.pth                # CNN weights (98 MB)
├── cnn_classification_report.txt
├── cnn_confusion_matrix.png
├── cnn_training_curves.png
│
├── .gitignore
└── README.md

Code Statistics:
  Backend Python:   8 files,  2,125 lines
  Frontend HTML:    4 files,  1,334 lines
  Frontend CSS:     3 files,  1,228 lines
  Frontend JS:      3 files,    718 lines
  Total:           ~5,400 lines of code
```

---

## 15. How to Run

### Prerequisites
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install ultralytics opencv-python pillow numpy matplotlib seaborn scikit-learn flask flask-cors werkzeug
```

### Step 1 – Train CNN (if not already done)
```bash
python train_cnn.py
# Output: cnn_best.pth
```

### Step 2 – Generate GradCAM Labels
```bash
python generate_labels.py
# Output: .txt label files alongside each image
```

### Step 3 – Train YOLOv8
```bash
python train_yolo.py
# Output: runs/detect/runs/solar_precise/weights/best.pt
```

### Step 4 – Run the Dashboard
```bash
python backend/app.py
# Open: http://127.0.0.1:5000
```

### Step 5 – Run CLI Inference (optional)
```bash
python predict.py --source test --output predictions --conf 0.25
```

---

## 16. Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| **Whole-image bounding boxes** — YOLO drew boxes around entire panel, not damage | Implemented GradCAM on ResNet-50 to generate tight, damage-localised labels |
| **CPU-only PyTorch** — `torch 2.11.0+cpu` installed, no GPU | Reinstalled with `--index-url https://download.pytorch.org/whl/cu128` |
| **YOLO label path mismatch** — Ultralytics looked for labels next to images, not in `labels/` folder | Regenerated labels alongside images (same directory, `.txt` extension) |
| **Doubled project path** — `runs/detect/runs/detect/...` | Fixed `PROJECT = "runs"` (Ultralytics appends `/detect/` automatically) |
| **Flask catch-all route blocking API** — `/<path:filename>` intercepted POST to `/api/auth/login` | Replaced catch-all with explicit `/css/`, `/js/`, `/assets/` routes |
| **Stale YOLO cache** — Old `.cache` files caused wrong label counts | Deleted all `.cache` files before each retrain |
| **PowerShell stderr** — Git informational output treated as errors | Confirmed success by checking `origin/main` tracking, not exit code |
| **Port conflict** — Old server still running on 5000 | Killed all Python processes with `Get-Process python | Stop-Process` |

---

## 17. Summary of Work Done

### Phase 1 – Data & Environment Setup
- ✅ Analysed dataset structure (6 classes, train/val/test splits)
- ✅ Installed PyTorch with CUDA 12.8 support for RTX 3050
- ✅ Verified GPU detection and tensor operations
- ✅ Generated initial whole-image YOLO labels

### Phase 2 – CNN Training
- ✅ Built ResNet-50 fine-tuning pipeline with AMP
- ✅ Implemented data augmentation (crop, flip, colour jitter, rotation)
- ✅ Trained for 50 epochs with early stopping
- ✅ Achieved **98% test accuracy** across 6 classes
- ✅ Generated confusion matrix, training curves, classification report

### Phase 3 – GradCAM Label Generation
- ✅ Implemented GradCAM on ResNet-50's layer4
- ✅ Heatmap → binary mask → contour → tight bounding box
- ✅ Progressive threshold fallback (0.40 → 0.30 → 0.20 → peak-based)
- ✅ Generated 1,574 precise labels (27–49% box area vs 100% before)
- ✅ Built `visualise_labels.py` to preview GradCAM boxes

### Phase 4 – YOLOv8 Training
- ✅ Configured YOLOv8m with GradCAM labels
- ✅ Tuned augmentation to preserve tight boxes (reduced translate/scale)
- ✅ Trained 120 epochs on GPU with AMP
- ✅ Achieved **mAP50 = 0.9001**, Precision = 0.856, Recall = 0.853
- ✅ Multiple training runs with progressive improvements

### Phase 5 – Inference & Damage Analysis
- ✅ Built `predict.py` with full detection pipeline
- ✅ Implemented damage percentage formula (class_weight × confidence)
- ✅ Built rich bounding box drawing (class + confidence + damage% + severity bar)
- ✅ Added summary panel at bottom of each image
- ✅ Built suggestion sidebar with what/impact/fix per class
- ✅ Tested on all 95 test images

### Phase 6 – Backend API
- ✅ Flask REST API with 7 endpoints
- ✅ Session-based authentication (register/login/logout)
- ✅ `@login_required` decorator protecting all ML endpoints
- ✅ Base64 image encoding for API response
- ✅ Serves frontend from same server

### Phase 7 – Frontend Dashboard
- ✅ Purple/violet themed dashboard (FinSet-inspired design)
- ✅ 6 pages: Dashboard, Analyse, Results, History, Classes, Settings
- ✅ Bar chart + donut chart (Canvas API, no external library)
- ✅ Animated damage severity meter
- ✅ Session scan history with stats
- ✅ Responsive layout with sidebar navigation

### Phase 8 – Authentication UI
- ✅ Login page (Sleeknote-inspired split layout)
- ✅ Signup page with password strength meter
- ✅ SVG illustrations on right panel
- ✅ Google OAuth button (UI placeholder)
- ✅ Auth flow: unauthenticated → login → dashboard → logout

### Phase 9 – GitHub
- ✅ Initialised git repository
- ✅ Created `.gitignore` (excludes weights, datasets, runs)
- ✅ Wrote comprehensive `README.md`
- ✅ 2 commits pushed to https://github.com/Shubhammadiwalar/repoproject1

---

## Final Metrics Summary

| Model | Metric | Value |
|-------|--------|-------|
| ResNet-50 CNN | Test Accuracy | **98.0%** |
| ResNet-50 CNN | Macro F1 | **0.98** |
| YOLOv8m | mAP50 | **0.9001** |
| YOLOv8m | mAP50-95 | **0.6190** |
| YOLOv8m | Precision | **0.8561** |
| YOLOv8m | Recall | **0.8534** |
| GradCAM | Box area reduction | **51–73%** |
| System | Total images | **1,569** |
| System | Total code lines | **~5,400** |

---

*Report generated for SolarScan project — Solar Panel Defect Detection System*
*GitHub: https://github.com/Shubhammadiwalar/repoproject1*
