"""
farm_detect.py
--------------
Solar Farm Grid Detection & Panel Matrix Mapping

Labels panels as:  Row 1 Col 1,  Row 1 Col 2,  Row 2 Col 1 …
Short form:        R1C1,  R1C2,  R2C1 …
"""

import cv2
import numpy as np
import base64
from typing import List, Dict, Tuple


# ── Panel detection ────────────────────────────────────────────────────────
def detect_panel_regions(img_bgr: np.ndarray) -> List[Dict]:
    h, w = img_bgr.shape[:2]
    hsv  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Solar panels: dark blue / dark grey
    mask1 = cv2.inRange(hsv, (90,  20,  20), (140, 255, 180))   # dark blue
    mask2 = cv2.inRange(hsv, (0,   0,   10), (180,  60, 120))   # dark grey
    mask3 = cv2.inRange(hsv, (100, 30,  30), (130, 200, 160))   # navy blue
    mask  = cv2.bitwise_or(mask1, cv2.bitwise_or(mask2, mask3))

    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    k_open  = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k_open)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = (w * h) * 0.003
    max_area = (w * h) * 0.25

    panels = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (min_area <= area <= max_area):
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        aspect = bw / max(bh, 1)
        if not (0.3 <= aspect <= 5.0):
            continue
        hull_area = cv2.contourArea(cv2.convexHull(cnt))
        if area / max(hull_area, 1) < 0.5:
            continue
        panels.append({"x1": x, "y1": y, "x2": x+bw, "y2": y+bh,
                        "cx": x+bw//2, "cy": y+bh//2,
                        "area": area, "w": bw, "h": bh})

    return _nms_panels(panels, 0.4)


def _nms_panels(panels, iou_thresh):
    panels = sorted(panels, key=lambda p: p["area"], reverse=True)
    keep = []
    for p in panels:
        if not any(_iou(p, k) > iou_thresh for k in keep):
            keep.append(p)
    return keep


def _iou(a, b):
    ix1, iy1 = max(a["x1"], b["x1"]), max(a["y1"], b["y1"])
    ix2, iy2 = min(a["x2"], b["x2"]), min(a["y2"], b["y2"])
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    if inter == 0:
        return 0.0
    return inter / ((a["x2"]-a["x1"])*(a["y2"]-a["y1"]) +
                    (b["x2"]-b["x1"])*(b["y2"]-b["y1"]) - inter)


# ── Grid assignment — pure numeric Row/Col ─────────────────────────────────
def assign_grid_positions(panels: List[Dict], row_gap_ratio: float = 0.6) -> List[Dict]:
    """
    Cluster panels into rows and columns.
    Labels:  grid_row (0-based int), grid_col (0-based int)
             grid_label = "R{row+1}C{col+1}"   e.g. R1C1, R2C3
    """
    if not panels:
        return panels

    panels = sorted(panels, key=lambda p: p["cy"])
    med_h  = float(np.median([p["h"] for p in panels]))

    rows: List[List[Dict]] = []
    cur = [panels[0]]
    for p in panels[1:]:
        if abs(p["cy"] - cur[-1]["cy"]) < med_h * row_gap_ratio:
            cur.append(p)
        else:
            rows.append(cur)
            cur = [p]
    rows.append(cur)

    for row in rows:
        row.sort(key=lambda p: p["cx"])

    for r_idx, row in enumerate(rows):
        for c_idx, panel in enumerate(row):
            panel["grid_row"]   = r_idx
            panel["grid_col"]   = c_idx
            panel["grid_label"] = f"R{r_idx+1}C{c_idx+1}"

    return panels


# ── Match YOLO detections → grid ───────────────────────────────────────────
def match_detections_to_grid(detections, grid_panels, img_h, img_w):
    for det in detections:
        dx1, dy1, dx2, dy2 = det["bbox"]
        det_cx = (dx1 + dx2) / 2
        det_cy = (dy1 + dy2) / 2

        if grid_panels:
            best, best_score = None, -1
            for gp in grid_panels:
                if gp["x1"] <= det_cx <= gp["x2"] and gp["y1"] <= det_cy <= gp["y2"]:
                    score = 2.0
                else:
                    score = _iou({"x1":dx1,"y1":dy1,"x2":dx2,"y2":dy2}, gp)
                if score > best_score:
                    best_score, best = score, gp

            if best and best_score > 0:
                det["grid_label"] = best["grid_label"]
                det["grid_row"]   = best["grid_row"]
                det["grid_col"]   = best["grid_col"]
                det["panel_bbox"] = [best["x1"], best["y1"], best["x2"], best["y2"]]
            else:
                det.update(_fallback_pos(det_cx, det_cy, img_w, img_h))
        else:
            det.update(_fallback_pos(det_cx, det_cy, img_w, img_h))

    return detections


def _fallback_pos(cx, cy, w, h):
    row = 1 if cy < h / 2 else 2
    col = 1 if cx < w / 2 else 2
    return {"grid_label": f"R{row}C{col}", "grid_row": row-1,
            "grid_col": col-1, "panel_bbox": None}


# ── Crop zoomed panel ──────────────────────────────────────────────────────
def crop_panel_zoom(img_bgr, bbox, pad_ratio=0.15):
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = bbox
    px = int((x2-x1) * pad_ratio)
    py = int((y2-y1) * pad_ratio)
    cx1, cy1 = max(0, x1-px), max(0, y1-py)
    cx2, cy2 = min(w, x2+px), min(h, y2+py)
    crop = img_bgr[cy1:cy2, cx1:cx2]
    tw   = 400
    th   = int(crop.shape[0] * tw / max(crop.shape[1], 1))
    if tw > 0 and th > 0:
        crop = cv2.resize(crop, (tw, th), interpolation=cv2.INTER_LANCZOS4)
    return crop


# ── Grid map — numeric Row/Col labels ─────────────────────────────────────
def draw_grid_map(grid_panels, detections, img_h, img_w):
    if not grid_panels:
        return _no_grid_map(detections)

    n_rows = max(p["grid_row"] for p in grid_panels) + 1
    n_cols = max(p["grid_col"] for p in grid_panels) + 1

    # Cell dimensions
    CELL_W, CELL_H = 80, 56
    PAD            = 8
    HDR_W          = 52   # left header width (Row labels)
    HDR_H          = 28   # top header height (Col labels)
    MARGIN         = 12

    map_w = MARGIN + HDR_W + n_cols * (CELL_W + PAD) + MARGIN
    map_h = MARGIN + HDR_H + n_rows * (CELL_H + PAD) + MARGIN + 20  # +20 legend

    canvas = np.full((map_h, map_w, 3), 245, dtype=np.uint8)

    # Affected lookup
    affected = {d.get("grid_label"): d for d in detections if d.get("grid_label")}

    # ── Column headers ────────────────────────────────────────────────────
    for c in range(n_cols):
        cx = MARGIN + HDR_W + c * (CELL_W + PAD) + CELL_W // 2
        label = f"Col {c+1}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        cv2.putText(canvas, label, (cx - tw//2, MARGIN + HDR_H - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 80, 80), 1, cv2.LINE_AA)

    # ── Row headers + cells ───────────────────────────────────────────────
    for r in range(n_rows):
        ry = MARGIN + HDR_H + r * (CELL_H + PAD)

        # Row label on left
        row_lbl = f"Row {r+1}"
        (tw, th), _ = cv2.getTextSize(row_lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        cv2.putText(canvas, row_lbl,
                    (MARGIN, ry + CELL_H//2 + th//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 80, 80), 1, cv2.LINE_AA)

        for c in range(n_cols):
            lbl = f"R{r+1}C{c+1}"
            cx1 = MARGIN + HDR_W + c * (CELL_W + PAD)
            cy1 = ry
            cx2 = cx1 + CELL_W
            cy2 = cy1 + CELL_H

            exists = any(p["grid_row"] == r and p["grid_col"] == c for p in grid_panels)
            if not exists:
                # Empty slot — draw faint placeholder
                cv2.rectangle(canvas, (cx1, cy1), (cx2, cy2), (220, 220, 220), 1)
                continue

            if lbl in affected:
                det   = affected[lbl]
                color = _hex_bgr(det.get("color", "#EF4444"))
                # Tinted background
                bg = tuple(int(v * 0.12 + 245 * 0.88) for v in color)
                cv2.rectangle(canvas, (cx1, cy1), (cx2, cy2), bg, -1)
                cv2.rectangle(canvas, (cx1, cy1), (cx2, cy2), color, 2)

                # Panel ID  e.g. "R1C2"
                cv2.putText(canvas, lbl,
                            (cx1+5, cy1+16),
                            cv2.FONT_HERSHEY_DUPLEX, 0.38, color, 1, cv2.LINE_AA)
                # Damage %
                dmg = f"{det.get('damage_pct',0):.0f}% dmg"
                cv2.putText(canvas, dmg,
                            (cx1+5, cy1+30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1, cv2.LINE_AA)
                # Class (short)
                cls_short = det.get("class","?")[:8]
                cv2.putText(canvas, cls_short,
                            (cx1+5, cy1+44),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.28, color, 1, cv2.LINE_AA)
                # Warning triangle
                pts = np.array([[cx2-8, cy1+4], [cx2-16, cy1+18], [cx2-1, cy1+18]], np.int32)
                cv2.fillPoly(canvas, [pts], color)
                cv2.putText(canvas, "!", (cx2-12, cy1+17),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.28, (255,255,255), 1, cv2.LINE_AA)
            else:
                # Clean panel — green
                cv2.rectangle(canvas, (cx1, cy1), (cx2, cy2), (225, 245, 225), -1)
                cv2.rectangle(canvas, (cx1, cy1), (cx2, cy2), (140, 195, 140), 1)
                cv2.putText(canvas, lbl,
                            (cx1+5, cy1+16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (60, 140, 60), 1, cv2.LINE_AA)
                cv2.putText(canvas, "OK",
                            (cx1+5, cy1+32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (60, 140, 60), 1, cv2.LINE_AA)

    # ── Legend ────────────────────────────────────────────────────────────
    ly = map_h - 8
    cv2.rectangle(canvas, (MARGIN, ly-10), (MARGIN+12, ly), (140,195,140), -1)
    cv2.putText(canvas, "= OK", (MARGIN+16, ly-1),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100,100,100), 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (MARGIN+70, ly-10), (MARGIN+82, ly), (60,60,220), -1)
    cv2.putText(canvas, "= Defect", (MARGIN+86, ly-1),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100,100,100), 1, cv2.LINE_AA)

    return canvas


def _no_grid_map(detections):
    canvas = np.full((140, 340, 3), 245, dtype=np.uint8)
    cv2.putText(canvas, "Grid map unavailable", (12, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160,160,160), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{len(detections)} defect(s) detected", (12, 72),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,100,200), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Upload an aerial/farm image for grid view", (12, 104),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180,180,180), 1, cv2.LINE_AA)
    return canvas


def _hex_bgr(hex_color: str) -> Tuple[int, int, int]:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return (b, g, r)


# ── Full farm analysis pipeline ────────────────────────────────────────────
def analyse_farm_image(img_bgr: np.ndarray, detections: List[Dict]) -> Dict:
    h, w = img_bgr.shape[:2]

    # 1. Detect panels
    grid_panels = detect_panel_regions(img_bgr)
    if grid_panels:
        grid_panels = assign_grid_positions(grid_panels)

    # 2. Match detections → grid
    detections = match_detections_to_grid(detections, grid_panels, h, w)

    # 3. Grid map image
    gm_img = draw_grid_map(grid_panels, detections, h, w)
    _, gm_buf = cv2.imencode(".png", gm_img)
    grid_map_b64 = base64.b64encode(gm_buf).decode("utf-8")

    # 4. Zoomed crops of defective panels
    panel_crops, seen = [], set()
    for det in detections:
        lbl = det.get("grid_label", "?")
        if lbl in seen:
            continue
        seen.add(lbl)
        bbox = det.get("panel_bbox") or det["bbox"]
        crop = crop_panel_zoom(img_bgr, bbox, pad_ratio=0.12)
        _, c_buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
        panel_crops.append({
            "grid_label":  lbl,
            "grid_row":    det.get("grid_row", 0) + 1,   # 1-based for display
            "grid_col":    det.get("grid_col", 0) + 1,
            "crop_b64":    base64.b64encode(c_buf).decode("utf-8"),
            "class":       det.get("class", "Unknown"),
            "confidence":  det.get("confidence", 0),
            "damage_pct":  det.get("damage_pct", 0),
            "severity":    det.get("severity", "Unknown"),
            "color":       det.get("color", "#7C3AED"),
            "bbox":        det["bbox"],
            "panel_bbox":  bbox,
        })

    # 5. Draw grid overlay on original image
    annotated = img_bgr.copy()
    affected_labels = {d.get("grid_label") for d in detections}
    for gp in grid_panels:
        lbl       = gp.get("grid_label", "")
        is_bad    = lbl in affected_labels
        box_color = (0, 60, 220) if is_bad else (0, 180, 60)
        thickness = 2 if is_bad else 1
        cv2.rectangle(annotated, (gp["x1"], gp["y1"]), (gp["x2"], gp["y2"]),
                      box_color, thickness)
        # Label background pill
        (tw, th), bl = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        lx, ly = gp["x1"] + 3, gp["y1"] + th + 4
        cv2.rectangle(annotated, (gp["x1"]+1, gp["y1"]+1),
                      (gp["x1"]+tw+6, gp["y1"]+th+6),
                      box_color, -1)
        cv2.putText(annotated, lbl, (lx, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255,255,255), 1, cv2.LINE_AA)

    return {
        "grid_panels":    [{"grid_label": p.get("grid_label",""),
                            "grid_row": p.get("grid_row",0)+1,
                            "grid_col": p.get("grid_col",0)+1,
                            "x1":p["x1"],"y1":p["y1"],"x2":p["x2"],"y2":p["y2"]}
                           for p in grid_panels],
        "detections":     detections,
        "grid_map_b64":   grid_map_b64,
        "panel_crops":    panel_crops,
        "total_panels":   len(grid_panels),
        "affected_panels":len(panel_crops),
        "farm_mode":      True,
        "annotated_farm": annotated,
    }
