# SolarScan – Solar Panel Defect Detection System
## Complete Project Report (Final Version)

**GitHub:** https://github.com/Shubhammadiwalar/repoproject1
**Date:** May 2026
**Developer:** Shubham Madiwalar

---

## Table of Contents
1. Project Overview
2. Problem Statement
3. System Architecture
4. Dataset
5. Model 1 – ResNet-50 CNN Classifier
6. GradCAM++ Label Generation
7. Model 2 – YOLOv8 Object Detector
8. Training History
9. Damage Percentage System
10. Backend API
11. MongoDB Database
12. Frontend Dashboard
13. Authentication System
14. Forgot Password Flow
15. PDF Report Generation
16. Farm / Aerial Image Detection
17. Results and Metrics
18. Technology Stack
19. Project Structure
20. How to Run
21. Challenges and Solutions
22. Summary of All Work Done

---

## 1. Project Overview

**SolarScan** is a full-stack AI-powered solar panel defect detection and analysis system. It combines a CNN classifier (ResNet-50) and a YOLOv8 object detector to:

- Detect defects on solar panels from photographs with precise bounding boxes
- Calculate a damage percentage (0-100%) per detection
- Provide actionable repair recommendations for each defect type
- Present results through a modern purple-themed web dashboard
- Store all data persistently in MongoDB Atlas
- Send password reset emails via Gmail SMTP
- Generate downloadable PDF reports with full diagnosis

---

## 2. Problem Statement

Solar panels degrade over time due to environmental and physical factors. Manual inspection is time-consuming, expensive, and inconsistent. This system automates defect detection using AI to:

1. Classify the type of defect (6 classes)
2. Localise the exact damaged region with a tight bounding box
3. Quantify severity as a damage percentage
4. Recommend specific repair actions
5. Track scan history per user in a database

---

## 3. System Architecture

```
Input Image
    |
    v
[ResNet-50 CNN + GradCAM++]
    |-- Classifies defect type
    |-- Generates tight bounding box labels
    v
[YOLOv8m Detection Model]
    |-- CSPDarknet backbone (pretrained COCO)
    |-- PANet neck
    |-- Decoupled detection head
    |-- Output: bbox + class + confidence
    v
[Damage Analysis Engine]
    |-- damage% = class_weight x confidence x 100
    |-- Severity classification (6 levels)
    |-- Diagnosis + repair suggestions
    v
[Flask REST API + MongoDB]
    |-- Auth: email/password + Google OAuth
    |-- Forgot password via Gmail SMTP
    |-- Scan history persisted in MongoDB
    v
[Purple Dashboard Frontend]
    |-- 6 pages: Dashboard, Analyse, Results, History, Classes, Settings
    |-- Canvas charts (bar + donut)
    |-- PDF report download
    |-- Farm/aerial grid detection
```

---

## 4. Dataset

### Split Summary

| Split | Unique Images |
|-------|--------------|
| Train | 925 |
| Val   | 549 |
| Test  | 95  |
| **Total** | **1,569** |

### Class Distribution

| Class | Train | Val | Test | Total |
|-------|-------|-----|------|-------|
| Bird-drop | 177 | 104 | 17 | 298 |
| Clean | 169 | 102 | 18 | 289 |
| Dusty | 162 | 97 | 16 | 275 |
| Electrical-damage | 135 | 77 | 13 | 225 |
| Physical-Damage | 132 | 78 | 15 | 225 |
| Snow-Covered | 154 | 92 | 16 | 262 |

---

## 5. Model 1 – ResNet-50 CNN Classifier

### Architecture
- Base: ResNet-50 (pretrained ImageNet IMAGENET1K_V2)
- Fine-tuned: layer3, layer4, custom FC head
- Head: Dropout(0.4) -> Linear(2048->512) -> ReLU -> Dropout(0.3) -> Linear(512->6)

### Training Config

| Parameter | Value |
|-----------|-------|
| Input size | 224 x 224 |
| Batch size | 32 |
| Epochs | 50 (early stopping, patience=10) |
| Optimizer | AdamW (lr=1e-4, weight_decay=1e-4) |
| Scheduler | CosineAnnealingLR |
| Loss | CrossEntropyLoss (label_smoothing=0.1) |
| AMP | Enabled (FP16) |

### CNN Test Results (95 images)

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Bird-drop | 1.00 | 1.00 | 1.00 | 17 |
| Clean | 0.94 | 0.94 | 0.94 | 18 |
| Dusty | 0.94 | 1.00 | 0.97 | 16 |
| Electrical-damage | 1.00 | 0.92 | 0.96 | 13 |
| Physical-Damage | 1.00 | 1.00 | 1.00 | 15 |
| Snow-Covered | 1.00 | 1.00 | 1.00 | 16 |
| **Overall Accuracy** | | | **0.98** | 95 |

---

## 6. GradCAM++ Label Generation

### Problem
Initial labels were whole-image boxes (0.5 0.5 1.0 1.0) — YOLO drew boxes around the entire panel instead of the damage.

### Solution: GradCAM++ (Gradient-weighted Class Activation Mapping++)

```
1. Forward pass image through ResNet-50
2. Compute pixel-wise gradient weights (GradCAM++ formula)
3. Weight feature maps -> activation heatmap
4. Apply Gaussian blur to reduce noise
5. Threshold heatmap (per-class thresholds)
6. Fit tight bounding box around high-activation region
7. Add 4% padding -> write YOLO label
```

### Per-Class Thresholds and Max Area

| Class | Threshold | Max Box Area |
|-------|-----------|-------------|
| Bird-drop | 0.55 | 30% |
| Clean | N/A | 75% (centred) |
| Dusty | 0.45 | 55% |
| Electrical-damage | 0.50 | 40% |
| Physical-Damage | 0.50 | 45% |
| Snow-Covered | 0.40 | 65% |

### Box Size Improvement

| Class | Old (whole-image) | New (GradCAM++) | Reduction |
|-------|-------------------|-----------------|-----------|
| Bird-drop | 100% | 17.6% avg | -82% |
| Electrical-damage | 100% | 21.3% avg | -79% |
| Physical-Damage | 100% | 30.2% avg | -70% |
| Dusty | 100% | 37.8% avg | -62% |
| Snow-Covered | 100% | 32.2% avg | -68% |

---

## 7. Model 2 – YOLOv8 Object Detector

### Architecture
- Model: YOLOv8m (medium, 25.8M parameters, 79.1 GFLOPs)
- Backbone: CSPDarknet CNN (pretrained COCO)
- Neck: PANet feature pyramid
- Head: Decoupled detection (bbox regression + classification)
- Input: 640 x 640

### Training Config (solar_v4 – final model)

| Parameter | Value |
|-----------|-------|
| Epochs | 150 |
| Batch | 16 |
| Image size | 640 x 640 |
| Optimizer | AdamW (auto-tuned, lr=0.001) |
| AMP | Enabled (FP16) |
| Early stopping | Patience=25 |
| GPU | RTX 3050 6GB |
| Labels | GradCAM++ tight boxes |

### Augmentation (reduced to preserve tight boxes)

| Augmentation | Value |
|-------------|-------|
| HSV hue | 0.015 |
| HSV saturation | 0.7 |
| Rotation | +-10 deg |
| Translation | 0.05 (reduced) |
| Scale | 0.4 (reduced) |
| Mosaic | 0.8 |
| MixUp | 0.1 |

---

## 8. Training History

| Run Name | Labels | Epochs | Best mAP50 | Precision | Recall | Time |
|----------|--------|--------|-----------|-----------|--------|------|
| solar_panel_yolo | Whole-image | 28 | 0.957 | — | — | — |
| solar_panel_yolo2 | Whole-image | 100 | **0.989** | 0.964 | 0.976 | — |
| solar_precise | GradCAM v1 | 34 | 0.564 | — | — | — |
| solar_precise3 | GradCAM v1 | 120 | 0.900 | 0.856 | 0.853 | 5.5h |
| **solar_v4** | **GradCAM++** | **150** | **0.913** | **0.919** | **0.881** | **~7h** |

**Active model:** `solar_v4` (GradCAM++ tight boxes, mAP50=0.913)

Note: solar_panel_yolo2 had higher mAP50 (0.989) but used whole-image labels — boxes covered 99%+ of image. solar_v4 uses tight labels with boxes covering 17-38% of image, giving precise damage localisation.

---

## 9. Damage Percentage System

### Formula
```
damage% = class_weight x yolo_confidence x 100
```

### Class Weights

| Class | Weight | Max Damage |
|-------|--------|------------|
| Clean | 0.00 | 0% |
| Dusty | 0.35 | 35% |
| Bird-drop | 0.60 | 60% |
| Snow-Covered | 0.50 | 50% |
| Physical-Damage | 0.90 | 90% |
| Electrical-damage | 0.95 | 95% |

### Severity Scale

| Range | Level | Colour |
|-------|-------|--------|
| 0% | No Damage | Green |
| 1-19% | Minimal | Light green |
| 20-39% | Low | Yellow |
| 40-59% | Moderate | Orange |
| 60-79% | High | Red |
| >=80% | Critical | Dark red |

### Diagnosis Database
Each class has structured diagnosis with:
- What happened (plain-language description)
- Impact on panel (3-4 bullet points)
- How to improve (5-6 numbered repair steps)

---

## 10. Backend API

### Framework
Flask 3.1.3 + flask-cors + flask-mail + Authlib + PyMongo + ReportLab

### All Endpoints

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| GET | / | — | Login or dashboard |
| GET | /login.html | — | Login page |
| GET | /signup.html | — | Signup page |
| GET | /forgot.html | — | Forgot password page |
| GET | /reset-password.html | — | Reset password page |
| POST | /api/auth/register | — | Create account |
| POST | /api/auth/login | — | Login, set session |
| POST | /api/auth/logout | — | Clear session |
| GET | /api/auth/me | — | Current user info |
| GET | /api/auth/google | — | Start Google OAuth |
| GET | /api/auth/google/callback | — | Google OAuth callback |
| POST | /api/auth/forgot-password | — | Send reset email |
| POST | /api/auth/reset-password | — | Update password |
| GET | /api/auth/validate-token | — | Check reset token |
| GET | /api/health | Protected | Model + GPU status |
| GET | /api/classes | Protected | 6 defect classes |
| POST | /api/predict | Protected | Full detection analysis |
| GET | /api/history | Protected | User scan history |
| POST | /api/history | Protected | Save scan to DB |
| POST | /api/report | Protected | Generate PDF report |

---

## 11. MongoDB Database

### Connection
- Provider: MongoDB Atlas (Free M0 tier)
- Cluster: intershippro1.nkhshzd.mongodb.net
- Database: solarscan

### Collections

| Collection | Purpose | Indexes |
|------------|---------|---------|
| users | User accounts (email, name, hashed password, avatar, provider) | email (unique) |
| reset_tokens | Password reset tokens with 30-min TTL auto-expiry | token (unique), expires_at (TTL) |
| scan_history | All scan results per user | user_email + time |

### DB Helper Functions
- db_get_user(email) — fetch user
- db_create_user(...) — register new user
- db_update_user(email, updates) — update fields
- db_save_token(token, email, expires_at) — store reset token
- db_get_token(token) — validate and fetch token
- db_delete_token(token) — one-time use deletion
- db_save_scan(user_email, scan) — persist scan result
- db_get_history(user_email, limit) — fetch scan history

All functions have in-memory fallback when MongoDB is not configured.

---

## 12. Frontend Dashboard

### Design
Purple/violet (#7C3AED) accent on white cards, lavender outer background, Inter font.
Inspired by modern fintech dashboard (FinSet-style).

### Pages

| Page | Description |
|------|-------------|
| Dashboard | 4 stat cards + bar chart + donut chart + recent scans table |
| Analyse Panel | Drag-drop upload + confidence slider + mini stat cards |
| Results | Summary strip + damage meter + annotated image + diagnosis + detection table + farm grid |
| History | Full scan history, clickable rows open detail modal |
| Defect Classes | 6 class cards with colour coding |
| Settings | Model info + detection config |

### Key Features
- Clickable stat cards with arrow navigation
- History row click -> full diagnosis modal with zoomed image
- Canvas bar chart (detections by class) + donut chart (severity split)
- Animated damage severity meter (green to red gradient)
- Farm/aerial grid map showing panel matrix positions
- PDF report download button

---

## 13. Authentication System

### Methods
1. Email/password (bcrypt hashed via Werkzeug)
2. Google OAuth 2.0 (Authlib + Google Cloud Console)

### Security
- Passwords hashed with PBKDF2-SHA256
- Server-side session cookies
- login_required decorator on all ML endpoints
- Email enumeration protection on forgot-password

### Demo Credentials
- Email: admin@solarscan.com
- Password: admin123

---

## 14. Forgot Password Flow

### Flow
1. User clicks "Forgot your password?" on login page
2. Enters email on /forgot.html
3. Server generates 32-byte cryptographic token (30-min TTL)
4. Token saved to MongoDB reset_tokens collection
5. Gmail SMTP sends HTML email with reset button
6. Email link uses network IP (works from phone on same WiFi)
7. /reset-password.html validates token from MongoDB
8. User enters new password -> bcrypt hash saved to MongoDB
9. Token deleted (one-time use)

### Email Config
- SMTP: smtp.gmail.com:587 (STARTTLS)
- From: shubhammadiwalar717@gmail.com
- App Password: configured in .env

---

## 15. PDF Report Generation

### Library: ReportLab 4.4.10

### PDF Contents
1. Header: SolarScan logo + report title
2. User info: name, email, scan time, filename
3. Summary cards: class, damage%, severity, detection count
4. Damage severity meter (colour gradient bar)
5. Annotated image (full size with bounding boxes)
6. Diagnosis: What Happened / Impact / How to Improve
7. Detection details table (grid position, class, confidence, damage%, severity)
8. Footer: timestamp + model info

### Download
- Button: "Download Report" on Results page
- Format: PDF
- Filename: SolarScan_ClassName_XX.Xpct.pdf
- Generated server-side via /api/report POST endpoint

---

## 16. Farm / Aerial Image Detection

### Feature
For drone/aerial images showing multiple panels in a grid:

1. Detect individual panel regions using HSV colour segmentation + contour analysis
2. Cluster panels into rows and columns
3. Assign matrix coordinates (A1, B2, C3...)
4. Match YOLO detections to grid positions
5. Crop zoomed view of each defective panel
6. Generate visual grid map (green=OK, coloured=defect)

### Grid Map
- Rows labelled A, B, C...
- Columns labelled 1, 2, 3...
- Each cell shows grid label + damage%
- Defective cells highlighted in class colour
- Shown in Results page below detection table

---

## 17. Results and Metrics

### CNN Classifier (ResNet-50)

| Metric | Value |
|--------|-------|
| Test Accuracy | 98.0% |
| Macro Precision | 0.98 |
| Macro Recall | 0.98 |
| Macro F1 | 0.98 |
| Perfect F1=1.0 classes | Bird-drop, Physical-Damage, Snow-Covered |

### YOLOv8m – solar_v4 (Active Model)

| Metric | Value |
|--------|-------|
| mAP50 | 0.9126 |
| mAP50-95 | 0.7330 |
| Precision | 0.9194 |
| Recall | 0.8811 |
| Best epoch | 148/150 |
| Training time | ~7 hours |
| GPU | RTX 3050 6GB |

### Bounding Box Precision

| Class | Old box area | New box area | Improvement |
|-------|-------------|-------------|-------------|
| Bird-drop | 99.6% | 17.6% | -82% |
| Electrical-damage | 99.9% | 21.3% | -79% |
| Physical-Damage | 99.6% | 30.2% | -70% |
| Dusty | 99.8% | 37.8% | -62% |
| Snow-Covered | 99.7% | 32.2% | -68% |

---

## 18. Technology Stack

### Hardware
| Component | Spec |
|-----------|------|
| GPU | NVIDIA GeForce RTX 3050 6GB Laptop GPU |
| CUDA | 12.8 |
| OS | Windows 11 |
| Python | 3.14.2 |

### Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| PyTorch | 2.11.0+cu128 | Deep learning |
| Ultralytics | 8.4.34 | YOLOv8 |
| torchvision | 0.26.0 | ResNet-50 |
| OpenCV | 4.13.0 | Image processing |
| NumPy | 2.4.2 | Arrays |
| Pillow | 12.1.1 | Image loading |
| scikit-learn | 1.8.0 | Metrics |
| Matplotlib | 3.10.8 | Plots |
| Flask | 3.1.3 | Web API |
| flask-cors | — | CORS |
| flask-mail | 0.10.0 | Email |
| Authlib | 1.7.2 | Google OAuth |
| Werkzeug | 3.1.6 | Password hashing |
| PyMongo | 4.17.0 | MongoDB |
| ReportLab | 4.4.10 | PDF generation |

---

## 19. Project Structure

```
project/
|
|-- backend/
|   +-- app.py                  Flask API + Auth + MongoDB + PDF (760 lines)
|
|-- frontend/
|   |-- index.html              Dashboard (600+ lines)
|   |-- login.html              Login page (Sleeknote design)
|   |-- signup.html             Signup with password strength meter
|   |-- forgot.html             Forgot password page
|   |-- forgot-password.html    (legacy, redirects)
|   |-- reset-password.html     Reset password with token validation
|   |-- css/
|   |   |-- style.css           Dashboard styles (580+ lines)
|   |   +-- auth.css            Auth page styles (350+ lines)
|   +-- js/
|       |-- config.js           API base URL
|       +-- app.js              Dashboard logic + charts (400+ lines)
|
|-- predict.py                  Detection + damage% + drawing (450 lines)
|-- train_yolo.py               YOLOv8 training (120 lines)
|-- train_cnn.py                ResNet-50 training (200 lines)
|-- generate_labels.py          Original GradCAM labels
|-- improve_labels.py           GradCAM++ improved labels
|-- farm_detect.py              Farm/aerial grid detection (350 lines)
|-- visualise_labels.py         Label preview tool
|-- dataset.yaml                YOLO dataset config
|-- .env                        Credentials (not committed)
|-- .env.example                Template for credentials
|-- README.md                   Setup documentation
|-- PROJECT_REPORT.md           This report
|
Code Statistics:
  Backend Python:  12 files, 3,601 lines
  Frontend HTML:    7 files, 2,219 lines
  Frontend CSS:     3 files, 1,402 lines
  Frontend JS:      3 files, 1,002 lines
  Total:          ~8,224 lines of code
  Git commits:    17
```

---

## 20. How to Run

### Prerequisites
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install ultralytics opencv-python pillow numpy matplotlib seaborn scikit-learn
pip install flask flask-cors flask-mail authlib pymongo[srv] reportlab werkzeug
```

### Step 1 – Train CNN
```bash
python train_cnn.py
# Output: cnn_best.pth
```

### Step 2 – Generate GradCAM++ Labels
```bash
python improve_labels.py
# Output: .txt label files alongside each image
```

### Step 3 – Train YOLOv8
```bash
python train_yolo.py
# Output: runs/detect/runs/solar_v4/weights/best.pt
```

### Step 4 – Configure .env
```
GOOGLE_CLIENT_ID=...
GOOGLE_CLIENT_SECRET=...
SECRET_KEY=...
MAIL_USERNAME=your@gmail.com
MAIL_PASSWORD=your-app-password
MONGO_URI=mongodb+srv://user:pass@cluster.mongodb.net/solarscan
```

### Step 5 – Run Dashboard
```bash
python backend/app.py
# Open: http://127.0.0.1:5000
```

---

## 21. Challenges and Solutions

| Challenge | Solution |
|-----------|----------|
| Whole-image bounding boxes | GradCAM++ with per-class thresholds and max area caps |
| CPU-only PyTorch | Reinstalled with --index-url https://download.pytorch.org/whl/cu128 |
| YOLO label path mismatch | Labels placed alongside images (same directory) |
| Doubled project path in runs/ | Fixed PROJECT = "runs" (Ultralytics appends /detect/) |
| Flask catch-all blocking API POST | Replaced catch-all with explicit /css/, /js/ routes |
| .env created as directory on Windows | Used Out-File -Encoding utf8 instead of heredoc |
| UTF-8 BOM in .env breaking parsing | Changed encoding to utf-8-sig |
| Google OAuth 400 Bad Request | Hardcoded redirect_uri in authorize_redirect() |
| Google OAuth duplicate redirect_uri | Removed redirect_uri from authorize_access_token() |
| Google OAuth deleted_client | Created new OAuth client, updated .env |
| Browser caching old forgot-password JS | Renamed to /forgot.html (new URL, no cache) |
| Reset link 127.0.0.1 not reachable on phone | Used request.host to generate network-IP link |
| MongoDB .env not loading | Fixed utf-8-sig encoding + is_file() check |
| USERS dict not replaced in forgot_password | Replaced all remaining USERS references with db_get_user() |
| Stale server on port 5000 | Kill all Python processes before restart |

---

## 22. Summary of All Work Done

### Phase 1 – Environment & Data
- Analysed dataset (6 classes, 1,569 images, train/val/test splits)
- Installed PyTorch CUDA 12.8 for RTX 3050
- Generated initial whole-image YOLO labels

### Phase 2 – CNN Training
- Built ResNet-50 fine-tuning pipeline with AMP
- Achieved 98% test accuracy across 6 classes
- Generated confusion matrix, training curves, classification report

### Phase 3 – GradCAM Label Generation (v1)
- Implemented standard GradCAM on ResNet-50 layer4
- Generated 1,574 labels with 27-49% box area

### Phase 4 – GradCAM++ Improved Labels
- Upgraded to GradCAM++ (sharper, more localised)
- Added Gaussian blur smoothing
- Per-class thresholds and max area caps
- Reduced box area to 17-38% (82% improvement for Bird-drop)

### Phase 5 – YOLOv8 Training (Multiple Runs)
- 8 training runs total
- Final model solar_v4: mAP50=0.913, Precision=0.919, Recall=0.881
- 150 epochs, ~7 hours on RTX 3050

### Phase 6 – Inference & Damage Analysis
- Built predict.py with full detection pipeline
- Damage% formula with class weights
- Rich bounding box drawing (thick corners, label pill, severity dot)
- Slim summary bar at bottom of image

### Phase 7 – Backend API
- Flask REST API with 20 endpoints
- Session-based authentication
- Google OAuth 2.0 (Authlib)
- login_required decorator

### Phase 8 – MongoDB Integration
- MongoDB Atlas free tier
- 3 collections: users, reset_tokens, scan_history
- TTL index on reset_tokens (auto-expire in 30 min)
- In-memory fallback when MongoDB not configured

### Phase 9 – Forgot Password
- Secure token generation (secrets.token_urlsafe(32))
- Gmail SMTP email sending (flask-mail)
- Network IP in reset link (works from phone)
- Token validation + one-time use deletion

### Phase 10 – Frontend Dashboard
- Purple/violet themed dashboard (FinSet-inspired)
- 6 pages with sidebar navigation
- Canvas bar chart + donut chart
- Animated damage severity meter
- Clickable stat card arrows

### Phase 11 – History & Modals
- Clickable history rows open full diagnosis modal
- Modal shows: zoomed image, damage meter, what/impact/fix
- "View in Results Page" button in modal

### Phase 12 – Farm/Aerial Detection
- HSV colour segmentation to detect individual panels
- Grid assignment (A1, B2, C3...)
- Panel crop zoom for each defective panel
- Visual grid map overlay

### Phase 13 – PDF Report
- ReportLab PDF generation
- Contains: user info, annotated image, damage meter, diagnosis, detection table
- Download button on Results page
- Filename: SolarScan_ClassName_XX.Xpct.pdf

### Phase 14 – Auth UI
- Login page (Sleeknote split-layout design)
- Signup page with password strength meter
- Forgot/reset password pages
- Google OAuth button

### Phase 15 – GitHub
- 17 commits pushed to https://github.com/Shubhammadiwalar/repoproject1
- .gitignore excludes weights, datasets, .env, runs

---

## Final Metrics Summary

| Model | Metric | Value |
|-------|--------|-------|
| ResNet-50 CNN | Test Accuracy | 98.0% |
| ResNet-50 CNN | Macro F1 | 0.98 |
| YOLOv8m (solar_v4) | mAP50 | 0.9126 |
| YOLOv8m (solar_v4) | mAP50-95 | 0.7330 |
| YOLOv8m (solar_v4) | Precision | 0.9194 |
| YOLOv8m (solar_v4) | Recall | 0.8811 |
| GradCAM++ | Box area reduction | 62-82% |
| System | Total images | 1,569 |
| System | Total code lines | ~8,224 |
| System | Git commits | 17 |
| System | API endpoints | 20 |
| System | Frontend pages | 6 |

---

*SolarScan – Solar Panel Defect Detection System*
*GitHub: https://github.com/Shubhammadiwalar/repoproject1*
*Developer: Shubham Madiwalar | May 2026*