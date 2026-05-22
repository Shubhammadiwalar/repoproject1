"""
backend/app.py  –  SolarScan API + Auth (Email + Google OAuth)
===============================================================
Endpoints:
  POST /api/auth/register        → create account with email
  POST /api/auth/login           → login with email/password
  POST /api/auth/logout          → clear session
  GET  /api/auth/me              → current user info
  GET  /api/auth/google          → start Google OAuth flow
  GET  /api/auth/google/callback → Google OAuth callback
  GET  /api/health               → model status (protected)
  POST /api/predict              → run detection (protected)
  GET  /api/classes              → defect classes (protected)

Run:
    python backend/app.py
    → http://localhost:5000
"""

import base64
import os
import sys
import tempfile
import secrets
import time
from pathlib import Path
from functools import wraps

import cv2
import numpy as np
import torch
import requests as http_requests
from flask import Flask, jsonify, request, send_from_directory, session, redirect, url_for
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
from authlib.integrations.flask_client import OAuth

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
from farm_detect import analyse_farm_image
from ultralytics import YOLO

# ── Load .env if present ───────────────────────────────────────────────────
env_path = ROOT / ".env"
try:
    if env_path.is_file() and env_path.stat().st_size > 0:
        # utf-8-sig strips BOM if present (Windows PowerShell adds it)
        for line in env_path.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
        print(f"[API] Loaded .env from {env_path}")
except Exception as e:
    print(f"[API] Warning: could not read .env — {e}")

GOOGLE_CLIENT_ID     = os.environ.get("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "")
SECRET_KEY           = os.environ.get("SECRET_KEY", "solarscan-secret-key-change-in-production")
GOOGLE_CONFIGURED    = bool(GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET)

# SMTP config (optional — for real email sending)
MAIL_SERVER   = os.environ.get("MAIL_SERVER", "smtp.gmail.com")
MAIL_PORT     = int(os.environ.get("MAIL_PORT", "587"))
MAIL_USERNAME = os.environ.get("MAIL_USERNAME", "")
MAIL_PASSWORD = os.environ.get("MAIL_PASSWORD", "")
MAIL_FROM     = os.environ.get("MAIL_FROM", MAIL_USERNAME)
MAIL_ENABLED  = bool(MAIL_USERNAME and MAIL_PASSWORD)

# ── MongoDB ────────────────────────────────────────────────────────────────
from pymongo import MongoClient, ASCENDING
from pymongo.errors import DuplicateKeyError

MONGO_URI = os.environ.get("MONGO_URI", "")
if MONGO_URI:
    try:
        _mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        _mongo_client.admin.command("ping")   # test connection
        _db            = _mongo_client.get_default_database()
        users_col      = _db["users"]
        tokens_col     = _db["reset_tokens"]
        history_col    = _db["scan_history"]

        # Indexes
        users_col.create_index("email", unique=True)
        tokens_col.create_index("token", unique=True)
        tokens_col.create_index("expires_at", expireAfterSeconds=0)  # TTL auto-delete
        history_col.create_index([("user_email", ASCENDING), ("time", ASCENDING)])

        MONGO_ENABLED = True
        print(f"[API] MongoDB      : CONNECTED ({_db.name})")

        # Seed default admin if not present
        if not users_col.find_one({"email": "admin@solarscan.com"}):
            users_col.insert_one({
                "email":         "admin@solarscan.com",
                "name":          "Admin User",
                "password_hash": generate_password_hash("admin123"),
                "avatar":        None,
                "provider":      "email",
            })
            print("[API] MongoDB      : Seeded admin@solarscan.com")

    except Exception as e:
        print(f"[API] MongoDB      : FAILED — {e}")
        MONGO_ENABLED = False
        users_col = tokens_col = history_col = None
else:
    MONGO_ENABLED = False
    users_col = tokens_col = history_col = None
    print("[API] MongoDB      : DISABLED (set MONGO_URI in .env)")

# ── DB helper functions (work with Mongo or in-memory fallback) ────────────
# In-memory fallback (used when MongoDB is not configured)
_MEM_USERS: dict = {
    "admin@solarscan.com": {
        "email": "admin@solarscan.com", "name": "Admin User",
        "password_hash": generate_password_hash("admin123"),
        "avatar": None, "provider": "email",
    }
}
_MEM_TOKENS:  dict = {}
_MEM_HISTORY: list = []


def db_get_user(email: str):
    if MONGO_ENABLED:
        return users_col.find_one({"email": email}, {"_id": 0})
    return _MEM_USERS.get(email)


def db_create_user(email: str, name: str, password_hash: str,
                   avatar=None, provider="email"):
    doc = {"email": email, "name": name, "password_hash": password_hash,
           "avatar": avatar, "provider": provider}
    if MONGO_ENABLED:
        try:
            users_col.insert_one(doc)
            return True
        except DuplicateKeyError:
            return False
    if email in _MEM_USERS:
        return False
    _MEM_USERS[email] = doc
    return True


def db_update_user(email: str, updates: dict):
    if MONGO_ENABLED:
        users_col.update_one({"email": email}, {"$set": updates})
    elif email in _MEM_USERS:
        _MEM_USERS[email].update(updates)


def db_save_token(token: str, email: str, expires_at: float):
    import datetime
    doc = {"token": token, "email": email,
           "expires_at": datetime.datetime.utcfromtimestamp(expires_at)}
    if MONGO_ENABLED:
        tokens_col.insert_one(doc)
    else:
        _MEM_TOKENS[token] = {"email": email, "expires_at": expires_at}


def db_get_token(token: str):
    if MONGO_ENABLED:
        import datetime
        doc = tokens_col.find_one({"token": token}, {"_id": 0})
        if not doc:
            return None
        # Check expiry
        exp = doc["expires_at"]
        if isinstance(exp, datetime.datetime):
            if exp < datetime.datetime.utcnow():
                tokens_col.delete_one({"token": token})
                return None
        return {"email": doc["email"], "expires_at": exp.timestamp() if hasattr(exp,"timestamp") else exp}
    rec = _MEM_TOKENS.get(token)
    if rec and time.time() > rec["expires_at"]:
        del _MEM_TOKENS[token]
        return None
    return rec


def db_delete_token(token: str):
    if MONGO_ENABLED:
        tokens_col.delete_one({"token": token})
    else:
        _MEM_TOKENS.pop(token, None)


def db_save_scan(user_email: str, scan: dict):
    import datetime
    doc = {"user_email": user_email, "time": datetime.datetime.utcnow(), **scan}
    if MONGO_ENABLED:
        history_col.insert_one(doc)
    else:
        _MEM_HISTORY.insert(0, doc)


def db_get_history(user_email: str, limit: int = 50):
    if MONGO_ENABLED:
        docs = list(history_col.find(
            {"user_email": user_email},
            {"_id": 0, "user_email": 0}
        ).sort("time", -1).limit(limit))
        # Convert datetime to string
        for d in docs:
            if hasattr(d.get("time"), "strftime"):
                d["time"] = d["time"].strftime("%H:%M:%S")
        return docs
    return [d for d in _MEM_HISTORY if d.get("user_email") == user_email][:limit]

# ── App ────────────────────────────────────────────────────────────────────
FRONTEND_DIR = ROOT / "frontend"

app = Flask(__name__)
app.secret_key = SECRET_KEY
CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=True)

# ── Google OAuth via Authlib ───────────────────────────────────────────────
oauth = OAuth(app)

if GOOGLE_CONFIGURED:
    google = oauth.register(
        name="google",
        client_id=GOOGLE_CLIENT_ID,
        client_secret=GOOGLE_CLIENT_SECRET,
        server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
        client_kwargs={"scope": "openid email profile"},
    )
    print(f"[API] Google OAuth : ENABLED (client_id={GOOGLE_CLIENT_ID[:20]}…)")
else:
    google = None
    print("[API] Google OAuth : DISABLED (set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET in .env)")

# ── Auth helpers ───────────────────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_email" not in session:
            return jsonify({"error": "Unauthorized", "redirect": "/login.html"}), 401
        return f(*args, **kwargs)
    return decorated

# ── Load model once ────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[API] Device: {DEVICE}")

def _load_yolo(path_hint):
    """Load YOLO model — downloads weights if not found locally."""
    try:
        w = find_best_weights(path_hint)
        print(f"[API] Loading model: {w}")
        return YOLO(w), w
    except FileNotFoundError:
        # On Railway/cloud: download pretrained YOLOv8m as fallback
        print("[API] Local weights not found — using pretrained yolov8m.pt")
        return YOLO("yolov8m.pt"), "yolov8m.pt"

# Primary: high-accuracy classifier
try:
    _cls_path = str(ROOT / "runs/detect/runs/solar_panel_yolo2/weights/best.pt")
    yolo_model, YOLO_WEIGHTS = _load_yolo(_cls_path)
    print("[API] Classification model: solar_panel_yolo2 (mAP50=0.989)")
except Exception as e:
    print(f"[API] Classifier load error: {e}")
    yolo_model   = YOLO("yolov8m.pt")
    YOLO_WEIGHTS = "yolov8m.pt"

# Secondary: tight-box model
yolo_box_model = None
try:
    _box_path = str(ROOT / "runs/detect/runs/solar_v4/weights/best.pt")
    if Path(_box_path).exists():
        yolo_box_model = YOLO(_box_path)
        print("[API] Box model: solar_v4 (GradCAM++ tight boxes)")
except Exception as e:
    print(f"[API] Box model not loaded: {e}")

print("[API] Models ready.")

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
    if "user_email" not in session:
        return send_from_directory(str(FRONTEND_DIR), "login.html")
    resp = send_from_directory(str(FRONTEND_DIR), "index.html")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
    resp.headers["Pragma"]        = "no-cache"
    return resp


@app.route("/login.html")
def login_page():
    return send_from_directory(str(FRONTEND_DIR), "login.html")


@app.route("/signup.html")
def signup_page():
    return send_from_directory(str(FRONTEND_DIR), "signup.html")


@app.route("/forgot.html")
def forgot_page_new():
    resp = send_from_directory(str(FRONTEND_DIR), "forgot.html")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
    resp.headers["Pragma"]        = "no-cache"
    resp.headers["Expires"]       = "0"
    return resp


@app.route("/forgot-password.html")
def forgot_password_page():
    resp = send_from_directory(str(FRONTEND_DIR), "forgot-password.html")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
    resp.headers["Pragma"]        = "no-cache"
    resp.headers["Expires"]       = "0"
    return resp


@app.route("/reset-password.html")
def reset_password_page():
    resp = send_from_directory(str(FRONTEND_DIR), "reset-password.html")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
    resp.headers["Pragma"]        = "no-cache"
    resp.headers["Expires"]       = "0"
    return resp


@app.route("/dashboard")
def dashboard():
    if "user_email" not in session:
        return send_from_directory(str(FRONTEND_DIR), "login.html")
    return send_from_directory(str(FRONTEND_DIR), "index.html")


# Static file routes — explicit paths only, never catch API routes
@app.route("/css/<path:filename>")
def serve_css(filename):
    return send_from_directory(str(FRONTEND_DIR / "css"), filename)

@app.route("/js/<path:filename>")
def serve_js(filename):
    resp = send_from_directory(str(FRONTEND_DIR / "js"), filename)
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
    resp.headers["Pragma"]        = "no-cache"
    return resp

@app.route("/assets/<path:filename>")
def serve_assets(filename):
    return send_from_directory(str(FRONTEND_DIR / "assets"), filename)


# ── Auth endpoints ──────────────────────────────────────────────────────────
@app.route("/api/auth/register", methods=["POST"])
def register():
    data     = request.get_json()
    name     = (data.get("name") or "").strip()
    email    = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    if not name or not email or not password:
        return jsonify({"error": "Name, email and password are required."}), 400
    if "@" not in email:
        return jsonify({"error": "Invalid email address."}), 400
    if len(password) < 6:
        return jsonify({"error": "Password must be at least 6 characters."}), 400
    if db_get_user(email):
        return jsonify({"error": "An account with this email already exists."}), 409

    ok = db_create_user(email, name, generate_password_hash(password))
    if not ok:
        return jsonify({"error": "An account with this email already exists."}), 409

    session["user_email"]  = email
    session["user_name"]   = name
    session["user_avatar"] = None
    return jsonify({"success": True, "name": name, "email": email}), 201


@app.route("/api/auth/login", methods=["POST"])
def login():
    data     = request.get_json()
    email    = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    user = db_get_user(email)
    if not user or not check_password_hash(user.get("password_hash", ""), password):
        return jsonify({"error": "Invalid email or password."}), 401

    session["user_email"]  = email
    session["user_name"]   = user["name"]
    session["user_avatar"] = user.get("avatar")
    return jsonify({"success": True, "name": user["name"], "email": email})


@app.route("/api/auth/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"success": True})


@app.route("/api/auth/me", methods=["GET"])
def me():
    if "user_email" not in session:
        return jsonify({"authenticated": False}), 401
    return jsonify({
        "authenticated": True,
        "email":  session["user_email"],
        "name":   session["user_name"],
        "avatar": session.get("user_avatar"),
    })


# ── Forgot / Reset password ─────────────────────────────────────────────────
@app.route("/api/auth/forgot-password", methods=["POST"])
def forgot_password():
    data  = request.get_json()
    email = (data.get("email") or "").strip().lower()

    if not email or "@" not in email:
        return jsonify({"error": "Please enter a valid email address."}), 400

    # Always return success to prevent email enumeration
    if not db_get_user(email):
        return jsonify({
            "success":    True,
            "email_sent": False,
            "reset_url":  None,
            "token":      None,
            "message":    "If that email exists, a reset link has been sent.",
        })

    # Generate secure token (expires in 30 minutes)
    token      = secrets.token_urlsafe(32)
    expires_at = time.time() + 1800   # 30 min
    db_save_token(token, email, expires_at)

    # Use network IP so link works from any device on the same WiFi
    host      = request.host
    reset_url = f"http://{host}/reset-password.html?token={token}"

    # Try to send real email if SMTP is configured
    email_sent = False
    if MAIL_ENABLED:
        try:
            import smtplib
            from email.mime.multipart import MIMEMultipart
            from email.mime.text import MIMEText

            msg            = MIMEMultipart("alternative")
            msg["Subject"] = "SolarScan – Reset Your Password"
            msg["From"]    = MAIL_FROM
            msg["To"]      = email

            html = f"""
            <div style="font-family:Inter,sans-serif;max-width:480px;margin:0 auto;padding:32px">
              <div style="text-align:center;margin-bottom:24px">
                <span style="font-size:1.4rem;font-weight:800;color:#1E1B4B">Solar<span style="color:#7C3AED">Scan</span></span>
              </div>
              <h2 style="color:#1E1B4B;margin-bottom:8px">Reset Your Password</h2>
              <p style="color:#6B7280;margin-bottom:24px">
                Click the button below to reset your password. This link expires in <strong>30 minutes</strong>.
              </p>
              <a href="{reset_url}"
                 style="display:inline-block;background:#7C3AED;color:#fff;padding:12px 28px;
                        border-radius:10px;text-decoration:none;font-weight:700;font-size:.95rem">
                Reset Password
              </a>
              <p style="color:#9CA3AF;font-size:.8rem;margin-top:24px">
                If you didn't request this, ignore this email. Your password won't change.
              </p>
            </div>
            """
            msg.attach(MIMEText(html, "html"))

            with smtplib.SMTP(MAIL_SERVER, MAIL_PORT) as smtp:
                smtp.starttls()
                smtp.login(MAIL_USERNAME, MAIL_PASSWORD)
                smtp.sendmail(MAIL_FROM, email, msg.as_string())
            email_sent = True
        except Exception as e:
            print(f"[Mail Error] {e}")

    return jsonify({
        "success":    True,
        "email_sent": email_sent,
        "message":    "Reset link sent to your email." if email_sent
                      else "Reset link generated (email not configured).",
        # In dev mode without email, return the link directly
        "reset_url":  reset_url if not email_sent else None,
        "token":      token     if not email_sent else None,
    })


@app.route("/api/auth/reset-password", methods=["POST"])
def reset_password():
    data     = request.get_json()
    token    = (data.get("token") or "").strip()
    password = data.get("password") or ""

    if not token or not password:
        return jsonify({"error": "Token and new password are required."}), 400
    if len(password) < 6:
        return jsonify({"error": "Password must be at least 6 characters."}), 400

    record = db_get_token(token)
    if not record:
        return jsonify({"error": "Invalid or expired reset link. Please request a new one."}), 400

    email = record["email"]
    if not db_get_user(email):
        return jsonify({"error": "Account not found."}), 404

    # Update password and delete token (one-time use)
    db_update_user(email, {"password_hash": generate_password_hash(password)})
    db_delete_token(token)

    return jsonify({"success": True, "message": "Password updated successfully. You can now log in."})


@app.route("/api/auth/validate-token", methods=["GET"])
def validate_token():
    """Check if a reset token is valid (used by reset page on load)."""
    token  = request.args.get("token", "")
    record = db_get_token(token)
    if not record:
        return jsonify({"valid": False, "error": "Invalid or expired reset link."})
    return jsonify({"valid": True, "email": record["email"]})


# ── Google OAuth routes ─────────────────────────────────────────────────────
@app.route("/api/auth/google")
def google_login():
    if not GOOGLE_CONFIGURED:
        return redirect("/login.html?error=google_not_configured")
    # Hardcode the callback URL to avoid Flask generating wrong host/port
    callback_url = "http://127.0.0.1:5000/api/auth/google/callback"
    return google.authorize_redirect(callback_url)


@app.route("/api/auth/google/callback")
def google_callback():
    if not GOOGLE_CONFIGURED:
        return redirect("/login.html?error=google_not_configured")
    try:
        # Authlib stores redirect_uri in session from authorize_redirect()
        # Do NOT pass it again here — it causes "multiple values" error
        token     = google.authorize_access_token()
        user_info = token.get("userinfo")
        if not user_info:
            # Fallback: fetch from userinfo endpoint
            resp      = google.get("https://openidconnect.googleapis.com/v1/userinfo")
            user_info = resp.json()

        email  = user_info.get("email", "").lower()
        name   = user_info.get("name") or user_info.get("given_name") or email.split("@")[0]
        avatar = user_info.get("picture")

        if not email:
            return redirect("/login.html?error=no_email")

        # Create account if first time, otherwise update avatar
        existing = db_get_user(email)
        if not existing:
            db_create_user(email, name, "", avatar=avatar, provider="google")
        else:
            db_update_user(email, {"avatar": avatar, "provider": "google"})

        session["user_email"]  = email
        session["user_name"]   = name
        session["user_avatar"] = avatar

        return redirect("/")

    except Exception as e:
        print(f"[Google OAuth Error] {e}")
        return redirect(f"/login.html?error=oauth_failed")


@app.route("/api/health", methods=["GET"])
@login_required
def health():
    return jsonify({
        "status":   "ok",
        "device":   DEVICE,
        "model":    str(YOLO_WEIGHTS),
        "classes":  len(CLASSES),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    })


@app.route("/api/classes", methods=["GET"])
@login_required
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


# ── Solar Panel Validator ───────────────────────────────────────────────────
def is_solar_panel(img_bgr: np.ndarray) -> tuple[bool, str]:
    """
    Determines if an image contains a solar panel using:
    1. Trained MobileNetV3 binary classifier (primary — most accurate)
    2. Colour/edge heuristics as fallback

    Returns (is_panel: bool, reason: str)
    """
    import torch.nn.functional as F_nn
    from torchvision import transforms
    from PIL import Image as PILImage

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # ── Stage 1: MobileNetV3 binary classifier (trained on solar vs non-solar) ──
    if panel_validator_model is not None:
        try:
            rgb    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            tensor = tf(PILImage.fromarray(rgb)).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                logits = panel_validator_model(tensor)
                probs  = F_nn.softmax(logits, dim=1)[0]
                not_panel_prob = float(probs[0])
                is_panel_prob  = float(probs[1])

            # Only accept if model is very confident it IS a panel
            if is_panel_prob < 0.80:
                return False, (
                    f"AI classifier is {not_panel_prob:.0%} confident this is NOT a solar panel "
                    f"(solar panel score: {is_panel_prob:.0%}). "
                    f"Please upload a solar panel photograph."
                )
            # High confidence acceptance — still run heuristics below
        except Exception as e:
            print(f"[Validator] MobileNetV3 error: {e}")

    # ── Stage 2: Heuristic fallback ──────────────────────────────────────
    h, w   = img_bgr.shape[:2]
    hsv    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    gray   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Colour ratios
    mask_blue  = cv2.inRange(hsv, (90, 20, 10),  (140, 255, 180))
    mask_grey  = cv2.inRange(hsv, (0,  0,  10),  (180, 55,  145))
    panel_ratio = float(cv2.bitwise_or(mask_blue, mask_grey).sum()) / (h * w * 255)

    mask_white  = cv2.inRange(hsv, (0, 0, 195), (180, 35, 255))
    white_ratio = float(mask_white.sum()) / (h * w * 255)

    mask_green  = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
    green_ratio = float(mask_green.sum()) / (h * w * 255)

    mask_skin   = cv2.inRange(hsv, (5, 30, 80), (25, 180, 255))
    skin_ratio  = float(mask_skin.sum()) / (h * w * 255)

    # Vivid/saturated colours (movie posters, game art, cartoons)
    # Solar panels are mostly desaturated (low saturation)
    avg_sat = float(hsv[:,:,1].mean())

    # Red/orange dominant (fire, robots, movie posters)
    mask_red1 = cv2.inRange(hsv, (0,  60, 60), (15,  255, 255))
    mask_red2 = cv2.inRange(hsv, (165,60, 60), (180, 255, 255))
    mask_orange = cv2.inRange(hsv, (10, 80, 80), (25, 255, 255))
    vivid_red_ratio = float(cv2.bitwise_or(cv2.bitwise_or(mask_red1,mask_red2),mask_orange).sum()) / (h*w*255)

    edges        = cv2.Canny(gray, 30, 100)
    edge_density = float(edges.sum()) / (h * w * 255)
    variance     = float(gray.var())

    # Face detection (only reject if skin also present)
    try:
        fc = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = fc.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=6, minSize=(80, 80))
        if len(faces) > 0 and skin_ratio > 0.06:
            return False, (
                f"A human face was detected. "
                f"Please upload a solar panel photograph, not a selfie."
            )
    except Exception:
        pass

    # Document / paper
    if white_ratio > 0.35 and panel_ratio < 0.15:
        return False, f"Image appears to be a document (white: {white_ratio:.0%})."

    # Plants / nature
    if green_ratio > 0.25 and panel_ratio < 0.20:
        return False, f"Image appears to contain vegetation (green: {green_ratio:.0%})."

    # Movie poster / game art / illustration — vivid saturated colours
    # Solar panels have avg saturation < 80; posters/art > 100
    if avg_sat > 95 and panel_ratio < 0.35:
        return False, (
            f"Image appears to be a movie poster, artwork, or illustration "
            f"(avg saturation: {avg_sat:.0f}, panel area: {panel_ratio:.0%}). "
            f"Please upload a real solar panel photograph."
        )

    # Vivid red/orange dominant (fire, robots, movie scenes)
    if vivid_red_ratio > 0.15 and panel_ratio < 0.30:
        return False, (
            f"Image contains vivid red/orange colours not typical of solar panels "
            f"(red/orange: {vivid_red_ratio:.0%}). "
            f"Please upload a solar panel photograph."
        )

    # No grid texture (laptop, wall, floor)
    if edge_density < 0.08:
        return False, (
            f"Image lacks solar panel grid texture (edge density: {edge_density:.3f}). "
            f"Solar panels score 0.10+."
        )

    return True, f"Heuristic pass (panel={panel_ratio:.0%} edge={edge_density:.3f} sat={avg_sat:.0f})"


# Load CNN for validation (same weights as classifier)
try:
    from torchvision import models as _tv_models
    import torch.nn as _nn

    # ── Primary: trained binary panel validator (MobileNetV3) ──────────
    _val_path = ROOT / "panel_validator.pth"
    if _val_path.exists():
        _mv3 = _tv_models.mobilenet_v3_small(weights=None)
        _mv3.classifier[-1] = _nn.Linear(_mv3.classifier[-1].in_features, 2)
        _mv3.load_state_dict(torch.load(str(_val_path), map_location=DEVICE))
        _mv3.to(DEVICE).eval()
        panel_validator_model = _mv3
        print(f"[API] Panel validator : LOADED (MobileNetV3, {_val_path.stat().st_size/1e6:.1f}MB)")
    else:
        panel_validator_model = None
        print("[API] Panel validator : NOT FOUND (panel_validator.pth missing)")

    # ── Fallback: ResNet-50 CNN classifier ──────────────────────────────
    _cnn_path = ROOT / "cnn_best.pth"
    if _cnn_path.exists():
        _rn50 = _tv_models.resnet50(weights=None)
        _rn50.fc = _nn.Sequential(
            _nn.Dropout(0.4), _nn.Linear(2048, 512),
            _nn.ReLU(), _nn.Dropout(0.3), _nn.Linear(512, len(CLASSES)),
        )
        _rn50.load_state_dict(torch.load(str(_cnn_path), map_location=DEVICE))
        _rn50.to(DEVICE).eval()
        cnn_validator = _rn50
        print("[API] CNN validator   : LOADED (ResNet-50 fallback)")
    else:
        cnn_validator = None
        print("[API] CNN validator   : NOT FOUND")

except Exception as e:
    panel_validator_model = None
    cnn_validator         = None
    print(f"[API] Validators      : FAILED — {e}")


def _nms_boxes(boxes, iou_thresh=0.4):
    """Simple NMS for (x1,y1,x2,y2,cls_id,conf) tuples."""
    if not boxes:
        return boxes
    boxes = sorted(boxes, key=lambda b: b[5], reverse=True)
    keep = []
    for b in boxes:
        bx1,by1,bx2,by2 = b[0],b[1],b[2],b[3]
        overlap = False
        for k in keep:
            kx1,ky1,kx2,ky2 = k[0],k[1],k[2],k[3]
            ix1 = max(bx1,kx1); iy1 = max(by1,ky1)
            ix2 = min(bx2,kx2); iy2 = min(by2,ky2)
            inter = max(0,ix2-ix1)*max(0,iy2-iy1)
            if inter == 0: continue
            area_b = (bx2-bx1)*(by2-by1)
            area_k = (kx2-kx1)*(ky2-ky1)
            iou = inter/(area_b+area_k-inter+1e-6)
            if iou > iou_thresh:
                overlap = True; break
        if not overlap:
            keep.append(b)
    return keep


@app.route("/api/predict", methods=["POST"])
@login_required
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

    # ── Solar panel validation ──────────────────────────────────────────
    is_panel, reason = is_solar_panel(img_bgr)
    if not is_panel:
        return jsonify({
            "error":        "not_solar_panel",
            "message":      "This does not appear to be a solar panel image.",
            "reason":       reason,
            "suggestion":   "Please upload a photograph of a solar panel. "
                            "Supported: single panel close-ups, aerial farm images, "
                            "or drone shots of solar installations.",
        }), 422

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

        # ── Dual-model detection with tiled inference ────────────────────
        # For images with multiple panels, tile the image and detect on each tile
        # This gives precise per-panel bounding boxes

        results    = yolo_model.predict(source=tmp_path, conf=conf_thr,
                                        imgsz=640, device=DEVICE, verbose=False)
        detections = results[0].boxes

        # Get ALL tight boxes from box model using tiled inference
        box_detections_all = []  # list of (x1,y1,x2,y2,cls_id,conf)
        if yolo_box_model is not None:
            try:
                # Run on full image first
                box_res = yolo_box_model.predict(
                    source=tmp_path, conf=max(0.08, conf_thr*0.4),
                    imgsz=640, device=DEVICE, verbose=False, iou=0.25, max_det=30
                )
                if box_res[0].boxes is not None:
                    for b in box_res[0].boxes:
                        box_detections_all.append((
                            *map(int, b.xyxy[0].tolist()),
                            int(b.cls[0].item()), float(b.conf[0].item())
                        ))

                # Tile inference: divide image into 2x2 grid with overlap
                tile_size = max(320, min(h, w) // 2)
                stride    = int(tile_size * 0.7)
                for ty in range(0, h - tile_size//4, stride):
                    for tx in range(0, w - tile_size//4, stride):
                        tx2 = min(tx + tile_size, w)
                        ty2 = min(ty + tile_size, h)
                        tile = img_bgr[ty:ty2, tx:tx2]
                        if tile.shape[0] < 64 or tile.shape[1] < 64:
                            continue
                        import tempfile as _tf
                        with _tf.NamedTemporaryFile(suffix=".jpg", delete=False) as t:
                            tile_path = t.name
                            cv2.imwrite(tile_path, tile)
                        try:
                            tr = yolo_box_model.predict(
                                source=tile_path, conf=max(0.10, conf_thr*0.5),
                                imgsz=640, device=DEVICE, verbose=False, iou=0.25, max_det=10
                            )
                            if tr[0].boxes is not None:
                                for b in tr[0].boxes:
                                    bx1,by1,bx2,by2 = map(int, b.xyxy[0].tolist())
                                    # Map back to original image coords
                                    ox1 = tx + bx1; oy1 = ty + by1
                                    ox2 = tx + bx2; oy2 = ty + by2
                                    box_detections_all.append((
                                        ox1, oy1, ox2, oy2,
                                        int(b.cls[0].item()), float(b.conf[0].item())
                                    ))
                        finally:
                            import os as _os
                            _os.unlink(tile_path)

                # NMS across all tile detections
                if box_detections_all:
                    box_detections_all = _nms_boxes(box_detections_all, iou_thresh=0.4)

            except Exception as e:
                print(f"[Box model error] {e}")
                box_detections_all = []

        if detections is not None and len(detections):
            # Get primary class from high-accuracy classifier
            primary_cls_id   = int(detections[0].cls[0].item())
            primary_cls_name = CLASSES[primary_cls_id] if primary_cls_id < len(CLASSES) else f"cls{primary_cls_id}"
            primary_conf     = float(detections[0].conf[0].item())

            if box_detections_all:
                # Draw all tight boxes with classifier's class label
                for (bx1, by1, bx2, by2, box_cls_id, box_conf) in box_detections_all:
                    if primary_conf > 0.5:
                        class_name = primary_cls_name
                        conf       = primary_conf
                        cls_id     = primary_cls_id
                    else:
                        class_name = CLASSES[box_cls_id] if box_cls_id < len(CLASSES) else f"cls{box_cls_id}"
                        conf       = box_conf
                        cls_id     = box_cls_id
                    color = PALETTE[cls_id % len(PALETTE)]
                    dpct  = draw_box(vis, bx1, by1, bx2, by2, class_name, conf, color)
                    damage_pcts.append(dpct)
                    primary_class = class_name
                    detections_list.append({
                        "class":      class_name,
                        "confidence": round(conf * 100, 1),
                        "damage_pct": dpct,
                        "severity":   severity_label(dpct),
                        "color":      CLASS_META.get(class_name, {}).get("color", "#94A3B8"),
                        "bbox":       [bx1, by1, bx2, by2],
                    })
            else:
                # Fallback: use classifier boxes
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

    # ── Farm / multi-panel analysis ─────────────────────────────────────
    farm_data = analyse_farm_image(img_bgr, detections_list)

    # Use farm-annotated image (has grid overlay) if panels were found
    if farm_data["total_panels"] > 1:
        farm_vis = farm_data["annotated_farm"].copy()
        # Also draw YOLO boxes on top of grid overlay
        for det in detections_list:
            x1, y1, x2, y2 = det["bbox"]
            color = PALETTE[CLASSES.index(det["class"]) % len(PALETTE)] if det["class"] in CLASSES else (128,128,128)
            draw_box(farm_vis, x1, y1, x2, y2, det["class"], det["confidence"]/100, color)
        if damage_pcts:
            _draw_summary_panel(farm_vis, avg_damage, primary)
        _, buf = cv2.imencode(".jpg", farm_vis, [cv2.IMWRITE_JPEG_QUALITY, 92])
    else:
        _, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 92])

    img_b64 = base64.b64encode(buf).decode("utf-8")

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
        "detections":      farm_data["detections"],   # enriched with grid_label
        "diagnosis": {
            "what_happened": diag["what_happened"],
            "impact":        diag["impact"],
            "suggestions":   diag["suggestions"],
        },
        # Farm-specific fields
        "farm_mode":       farm_data["farm_mode"],
        "total_panels":    farm_data["total_panels"],
        "affected_panels": farm_data["affected_panels"],
        "grid_map_b64":    farm_data["grid_map_b64"],
        "panel_crops":     farm_data["panel_crops"],
        "grid_panels":     farm_data["grid_panels"],
    })

# ── Scan history endpoints ──────────────────────────────────────────────────
@app.route("/api/history", methods=["GET"])
@login_required
def get_history():
    email   = session["user_email"]
    history = db_get_history(email, limit=100)
    return jsonify({"history": history})


@app.route("/api/history", methods=["POST"])
@login_required
def save_scan():
    data  = request.get_json()
    email = session["user_email"]
    scan  = {
        "file":     data.get("file", ""),
        "cls":      data.get("cls", ""),
        "conf":     data.get("conf", 0),
        "damage":   data.get("damage", 0),
        "severity": data.get("severity", ""),
        "color":    data.get("color", "#7C3AED"),
    }
    db_save_scan(email, scan)
    return jsonify({"success": True})


# ── PDF Report endpoint ────────────────────────────────────────────────────
@app.route("/api/report", methods=["POST"])
@login_required
def generate_report():
    """
    Generate a full PDF report containing:
    - User info
    - Annotated image
    - Damage summary & severity meter
    - What happened / Impact / Suggestions
    - Detection details table
    """
    import io, base64, datetime
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                    Table, TableStyle, Image as RLImage,
                                    HRFlowable, KeepTogether)
    from reportlab.graphics.shapes import Drawing, Rect, String
    from reportlab.graphics import renderPDF

    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    user_email    = session.get("user_email", "Unknown")
    user_name     = session.get("user_name",  "Unknown")
    primary_class = data.get("primary_class", "Unknown")
    primary_color = data.get("primary_color", "#7C3AED")
    avg_damage    = data.get("avg_damage", 0)
    severity      = data.get("severity", "Unknown")
    detections    = data.get("detections", [])
    diagnosis     = data.get("diagnosis", {})
    img_b64       = data.get("annotated_image", "")
    filename      = data.get("filename", "panel.jpg")
    scan_time     = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Severity colour
    sev_colors = {
        "No Damage": colors.HexColor("#16A34A"),
        "Minimal":   colors.HexColor("#16A34A"),
        "Low":       colors.HexColor("#CA8A04"),
        "Moderate":  colors.HexColor("#EA580C"),
        "High":      colors.HexColor("#DC2626"),
        "Critical":  colors.HexColor("#991B1B"),
    }
    sev_col = sev_colors.get(severity, colors.HexColor("#7C3AED"))

    # ── Build PDF in memory ──────────────────────────────────────────────
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm,
    )
    W, H = A4
    content_w = W - 4*cm

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle("title",
        fontSize=22, fontName="Helvetica-Bold",
        textColor=colors.HexColor("#1E1B4B"),
        spaceAfter=4, leading=26)
    sub_style = ParagraphStyle("sub",
        fontSize=10, fontName="Helvetica",
        textColor=colors.HexColor("#6B7280"),
        spaceAfter=2)
    h2_style = ParagraphStyle("h2",
        fontSize=13, fontName="Helvetica-Bold",
        textColor=colors.HexColor("#1E1B4B"),
        spaceBefore=14, spaceAfter=6)
    h3_style = ParagraphStyle("h3",
        fontSize=10, fontName="Helvetica-Bold",
        textColor=colors.HexColor("#374151"),
        spaceBefore=8, spaceAfter=4)
    body_style = ParagraphStyle("body",
        fontSize=9.5, fontName="Helvetica",
        textColor=colors.HexColor("#374151"),
        leading=14, spaceAfter=3)
    bullet_style = ParagraphStyle("bullet",
        fontSize=9, fontName="Helvetica",
        textColor=colors.HexColor("#374151"),
        leading=13, leftIndent=12, spaceAfter=2,
        bulletIndent=0)

    story = []

    # ── Header ───────────────────────────────────────────────────────────
    # Logo row
    logo_table = Table([[
        Paragraph("<b><font color='#7C3AED'>Solar</font><font color='#1E1B4B'>Scan</font></b>",
                  ParagraphStyle("logo", fontSize=18, fontName="Helvetica-Bold")),
        Paragraph(f"<font color='#9CA3AF'>Solar Panel Defect Analysis Report</font>",
                  ParagraphStyle("logor", fontSize=9, fontName="Helvetica",
                                 alignment=2))
    ]], colWidths=[content_w*0.5, content_w*0.5])
    logo_table.setStyle(TableStyle([
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("BOTTOMPADDING", (0,0), (-1,-1), 8),
    ]))
    story.append(logo_table)
    story.append(HRFlowable(width=content_w, thickness=2,
                            color=colors.HexColor("#7C3AED"), spaceAfter=12))

    # ── User & scan info ─────────────────────────────────────────────────
    info_data = [
        ["Generated By", user_name, "Email", user_email],
        ["Scan Time",    scan_time,  "File",  filename],
    ]
    info_table = Table(info_data, colWidths=[3*cm, content_w*0.35, 2.5*cm, content_w*0.35])
    info_table.setStyle(TableStyle([
        ("FONTNAME",  (0,0), (0,-1), "Helvetica-Bold"),
        ("FONTNAME",  (2,0), (2,-1), "Helvetica-Bold"),
        ("FONTSIZE",  (0,0), (-1,-1), 8.5),
        ("TEXTCOLOR", (0,0), (0,-1), colors.HexColor("#6B7280")),
        ("TEXTCOLOR", (2,0), (2,-1), colors.HexColor("#6B7280")),
        ("TEXTCOLOR", (1,0), (1,-1), colors.HexColor("#1E1B4B")),
        ("TEXTCOLOR", (3,0), (3,-1), colors.HexColor("#1E1B4B")),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ("TOPPADDING",    (0,0), (-1,-1), 4),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 10))

    # ── Summary cards ────────────────────────────────────────────────────
    card_data = [[
        Paragraph(f"<b><font color='#6B7280' size='8'>DETECTED CLASS</font></b><br/>"
                  f"<font color='{primary_color}' size='14'><b>{primary_class}</b></font>",
                  body_style),
        Paragraph(f"<b><font color='#6B7280' size='8'>DAMAGE LEVEL</font></b><br/>"
                  f"<font color='#DC2626' size='14'><b>{avg_damage}%</b></font>",
                  body_style),
        Paragraph(f"<b><font color='#6B7280' size='8'>SEVERITY</font></b><br/>"
                  f"<font size='14'><b>{severity}</b></font>",
                  ParagraphStyle("sev", fontSize=9, fontName="Helvetica",
                                 textColor=sev_col)),
        Paragraph(f"<b><font color='#6B7280' size='8'>DETECTIONS</font></b><br/>"
                  f"<font color='#1E1B4B' size='14'><b>{len(detections)}</b></font>",
                  body_style),
    ]]
    card_table = Table(card_data, colWidths=[content_w/4]*4)
    card_table.setStyle(TableStyle([
        ("BOX",         (0,0), (-1,-1), 0.5, colors.HexColor("#EDE9FE")),
        ("INNERGRID",   (0,0), (-1,-1), 0.5, colors.HexColor("#EDE9FE")),
        ("BACKGROUND",  (0,0), (-1,-1), colors.HexColor("#F8F7FF")),
        ("TOPPADDING",  (0,0), (-1,-1), 10),
        ("BOTTOMPADDING",(0,0),(-1,-1), 10),
        ("LEFTPADDING", (0,0), (-1,-1), 10),
        ("VALIGN",      (0,0), (-1,-1), "MIDDLE"),
        ("ROUNDEDCORNERS", (0,0), (-1,-1), [6,6,6,6]),
    ]))
    story.append(card_table)
    story.append(Spacer(1, 8))

    # ── Damage severity bar ───────────────────────────────────────────────
    story.append(Paragraph("Damage Severity Meter", h3_style))
    bar_w = float(content_w)
    bar_h = 14
    pct   = min(max(avg_damage, 0), 100) / 100

    d = Drawing(bar_w, bar_h + 20)
    # Background gradient track
    segments = [
        (0.00, 0.40, "#22C55E"),
        (0.40, 0.65, "#EAB308"),
        (0.65, 0.85, "#F97316"),
        (0.85, 1.00, "#DC2626"),
    ]
    for s, e, c in segments:
        d.add(Rect(bar_w*s, 6, bar_w*(e-s), bar_h,
                   fillColor=colors.HexColor(c), strokeColor=None))
    # White overlay for unfilled portion
    d.add(Rect(bar_w*pct, 6, bar_w*(1-pct), bar_h,
               fillColor=colors.white, strokeColor=None, fillOpacity=0.6))
    # Thumb
    tx = bar_w * pct
    d.add(Rect(max(0, tx-6), 2, 12, bar_h+8,
               fillColor=colors.white,
               strokeColor=colors.HexColor("#1E1B4B"), strokeWidth=2))
    # Labels
    for lbl, pos in [("No Damage",0), ("Low",0.33), ("Moderate",0.55), ("Critical",0.85)]:
        d.add(String(bar_w*pos, 0, lbl,
                     fontSize=6, fontName="Helvetica",
                     fillColor=colors.HexColor("#6B7280")))
    story.append(d)
    story.append(Spacer(1, 8))

    # ── Annotated image ───────────────────────────────────────────────────
    if img_b64:
        try:
            img_bytes = base64.b64decode(img_b64)
            img_buf   = io.BytesIO(img_bytes)
            rl_img    = RLImage(img_buf, width=content_w, height=content_w*0.6)
            story.append(Paragraph("Annotated Image", h2_style))
            story.append(rl_img)
            story.append(Spacer(1, 10))
        except Exception:
            pass

    # ── Diagnosis ─────────────────────────────────────────────────────────
    story.append(HRFlowable(width=content_w, thickness=1,
                            color=colors.HexColor("#EDE9FE"), spaceAfter=8))
    story.append(Paragraph("Diagnosis Report", h2_style))

    # What happened
    what = diagnosis.get("what_happened", "")
    if what:
        story.append(Paragraph("🔶 What Happened", h3_style))
        story.append(Paragraph(what, body_style))
        story.append(Spacer(1, 6))

    # Impact
    impact = diagnosis.get("impact", [])
    if impact:
        story.append(Paragraph("🔴 Impact on Panel", h3_style))
        for item in impact:
            story.append(Paragraph(f"• {item}", bullet_style))
        story.append(Spacer(1, 6))

    # Suggestions
    suggestions = diagnosis.get("suggestions", [])
    if suggestions:
        story.append(Paragraph("🟢 How to Improve", h3_style))
        for i, step in enumerate(suggestions, 1):
            clean = step.lstrip("0123456789. ")
            story.append(Paragraph(f"{i}. {clean}", bullet_style))
        story.append(Spacer(1, 8))

    # ── Detection details table ───────────────────────────────────────────
    if detections:
        story.append(HRFlowable(width=content_w, thickness=1,
                                color=colors.HexColor("#EDE9FE"), spaceAfter=8))
        story.append(Paragraph("Detection Details", h2_style))

        tbl_data = [["#", "Grid", "Class", "Confidence", "Damage %", "Severity"]]
        for i, det in enumerate(detections, 1):
            tbl_data.append([
                str(i),
                det.get("grid_label", "—"),
                det.get("class", ""),
                f"{det.get('confidence', 0):.1f}%",
                f"{det.get('damage_pct', 0):.1f}%",
                det.get("severity", ""),
            ])

        det_table = Table(tbl_data,
                          colWidths=[1*cm, 1.5*cm, 4*cm, 3*cm, 3*cm, 3*cm])
        det_table.setStyle(TableStyle([
            ("BACKGROUND",   (0,0), (-1,0), colors.HexColor("#EDE9FE")),
            ("TEXTCOLOR",    (0,0), (-1,0), colors.HexColor("#7C3AED")),
            ("FONTNAME",     (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE",     (0,0), (-1,-1), 8.5),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),
             [colors.white, colors.HexColor("#F8F7FF")]),
            ("GRID",         (0,0), (-1,-1), 0.4, colors.HexColor("#EDE9FE")),
            ("TOPPADDING",   (0,0), (-1,-1), 5),
            ("BOTTOMPADDING",(0,0), (-1,-1), 5),
            ("LEFTPADDING",  (0,0), (-1,-1), 6),
            ("VALIGN",       (0,0), (-1,-1), "MIDDLE"),
        ]))
        story.append(det_table)

    # ── Footer ────────────────────────────────────────────────────────────
    story.append(Spacer(1, 16))
    story.append(HRFlowable(width=content_w, thickness=1,
                            color=colors.HexColor("#EDE9FE"), spaceAfter=6))
    story.append(Paragraph(
        f"<font color='#9CA3AF' size='8'>Generated by SolarScan AI · {scan_time} · "
        f"YOLOv8m + ResNet-50 · GradCAM Precision Detection</font>",
        ParagraphStyle("footer", fontSize=8, fontName="Helvetica",
                       textColor=colors.HexColor("#9CA3AF"), alignment=1)))

    # ── Build ─────────────────────────────────────────────────────────────
    doc.build(story)
    buf.seek(0)

    safe_name = f"SolarScan_{primary_class.replace(' ','-')}_{avg_damage}pct.pdf"
    from flask import send_file
    return send_file(
        buf,
        mimetype="application/pdf",
        as_attachment=True,
        download_name=safe_name,
    )


# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port, threaded=True)
