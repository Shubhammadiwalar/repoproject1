"""
backend/api.py  –  SolarScan REST API
======================================
Pure JSON API. No HTML serving. Frontend is completely separate.

Endpoints:
  GET  /api/health          → model status
  POST /api/predict         → run detection on uploaded image
  GET  /api/classes         → list of defect classes + metadata

Run:
    cd backend
    python api.py
    → http://localhost:5000
"""

import base64
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

# ── resolve project root so we can import predict.py ──────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from predict import (
    CLASSES,
    DIAGNOSIS,
    PALETTE,
    build_suggestion_panel,
    compute_damage_pct,
    draw_box,
    find_best_weights,
    severity_color,
    severity_label,
    _draw_summary_panel,
)
from ultralytics import YOLO

# ── App ────────────────────────────────────────────────────────────────────
FRONTEND_DIR = ROOT / "frontend"

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# ── Load model once ────────────────────────────────────────────────────────
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
YOLO_WEIGHTS = find_best_weights(str(ROOT / "runs/detect/solar_panel_yolo/weights/best.pt"))
print(f"[API] Loading model : {YOLO_WEIGHTS}")
print(f"[API] Device        : {DEVICE}")
yolo_model   = YOLO(YOLO_WEIGHTS)
print("[API] Model ready. Listening on http://localhost:5000")

# ── Class metadata (sent to frontend) ─────────────────────────────────────
CLASS_META = {
    "Bird-drop":         {"color": "#F97316", "max_damage": 60,  "icon": "🐦"},
    "Clean":             {"color": "#22C55E", "max_damage": 0,   "icon": "✅"},
    "Dusty":             {"color": "#EAB308", "max_damage": 35,  "icon": "🌫️"},
    "Electrical-damage": {"color": "#DC2626", "max_damage": 95,  "icon": "⚡"},
    "Physical-Damage":   {"color": "#3B82F6", "max_damage": 90,  "icon": "💥"},
    "Snow-Covered":      {"color": "#06B6D4", "max_damage": 50,  "icon": "❄️"},
}


# ── Routes ─────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    """Serve the frontend index.html."""
    return send_from_directory(str(FRONTEND_DIR), "index.html")


@app.route("/<path:filename>")
def frontend_files(filename):
    """Serve frontend static files (css/, js/, assets/)."""
    return send_from_directory(str(FRONTEND_DIR), filename)


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status":   "ok",
        "device":   DEVICE,
        "model":    str(YOLO_WEIGHTS),
        "classes":  len(CLASSES),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    })


@app.route("/api/classes", methods=["GET"])
def get_classes():
    result = []
    for cls in CLASSES:
        meta = CLASS_META.get(cls, {})
        diag = DIAGNOSIS.get(cls, {})
        result.append({
            "name":          cls,
            "color":         meta.get("color", "#94A3B8"),
            "max_damage":    meta.get("max_damage", 50),
            "icon":          meta.get("icon", "🔍"),
            "what_happened": diag.get("what_happened", ""),
            "impact":        diag.get("impact", []),
            "suggestions":   diag.get("suggestions", []),
        })
    return jsonify({"classes": result})


@app.route("/api/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file in request. Use field name 'image'."}), 400

    file     = request.files["image"]
    conf_thr = float(request.form.get("conf", 0.25))
    conf_thr = max(0.01, min(0.95, conf_thr))

    # Decode image
    img_bytes = file.read()
    np_arr    = np.frombuffer(img_bytes, np.uint8)
    img_bgr   = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return jsonify({"error": "Cannot decode image. Ensure it is a valid JPG/PNG."}), 400

    h, w = img_bgr.shape[:2]
    vis  = img_bgr.copy()

    # Write temp file for YOLO
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name
        cv2.imwrite(tmp_path, img_bgr)

    try:
        detections_list = []
        damage_pcts     = []
        primary_class   = None

        # First pass at requested confidence
        results    = yolo_model.predict(source=tmp_path, conf=conf_thr,
                                        imgsz=640, device=DEVICE, verbose=False)
        detections = results[0].boxes

        if detections is not None and len(detections):
            for box in detections:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls_id     = int(box.cls[0].item())
                conf       = float(box.conf[0].item())
                class_name = CLASSES[cls_id] if cls_id < len(CLASSES) else f"cls{cls_id}"
                color      = PALETTE[cls_id % len(PALETTE)]
                dpct       = draw_box(vis, x1, y1, x2, y2, class_name, conf, color)
                damage_pcts.append(dpct)
                primary_class = class_name
                detections_list.append({
                    "class":      class_name,
                    "confidence": round(conf * 100, 1),
                    "damage_pct": dpct,
                    "severity":   severity_label(dpct),
                    "color":      CLASS_META.get(class_name, {}).get("color", "#94A3B8"),
                    "bbox":       [x1, y1, x2, y2],
                })
        else:
            # Fallback: lowest-conf prediction
            res2  = yolo_model.predict(source=tmp_path, conf=0.01,
                                       imgsz=640, device=DEVICE, verbose=False)
            boxes = res2[0].boxes
            if boxes is not None and len(boxes):
                best       = max(boxes, key=lambda b: float(b.conf[0]))
                cls_id     = int(best.cls[0].item())
                conf       = float(best.conf[0].item())
                class_name = CLASSES[cls_id] if cls_id < len(CLASSES) else f"cls{cls_id}"
                color      = PALETTE[cls_id % len(PALETTE)]
                dpct       = draw_box(vis, 0, 0, w - 1, h - 1, class_name, conf, color)
                damage_pcts.append(dpct)
                primary_class = class_name
                detections_list.append({
                    "class":      class_name,
                    "confidence": round(conf * 100, 1),
                    "damage_pct": dpct,
                    "severity":   severity_label(dpct),
                    "color":      CLASS_META.get(class_name, {}).get("color", "#94A3B8"),
                    "bbox":       [0, 0, w - 1, h - 1],
                })
    finally:
        os.unlink(tmp_path)

    avg_damage = round(sum(damage_pcts) / len(damage_pcts), 1) if damage_pcts else 0.0
    primary    = primary_class or "Clean"

    # Draw summary bar on annotated image
    if damage_pcts:
        _draw_summary_panel(vis, avg_damage, primary)

    # Encode annotated image → base64
    _, buf   = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 92])
    img_b64  = base64.b64encode(buf).decode("utf-8")

    # Diagnosis
    diag = DIAGNOSIS.get(primary, DIAGNOSIS["Clean"])

    return jsonify({
        "success":         True,
        "annotated_image": img_b64,
        "image_size":      {"width": w, "height": h},
        "primary_class":   primary,
        "primary_color":   CLASS_META.get(primary, {}).get("color", "#94A3B8"),
        "avg_damage":      avg_damage,
        "severity":        severity_label(avg_damage),
        "detection_count": len(detections_list),
        "detections":      detections_list,
        "diagnosis": {
            "what_happened": diag["what_happened"],
            "impact":        diag["impact"],
            "suggestions":   diag["suggestions"],
        },
    })


# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000, threaded=True)
