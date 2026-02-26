import io
import os
import zipfile
from datetime import datetime

import cv2
import numpy as np
from flask import Flask, render_template, request, send_file, jsonify

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MASK_48 = os.path.join(BASE_DIR, "48.png")
MASK_96 = os.path.join(BASE_DIR, "96.png")

app = Flask(__name__)


def load_alpha(mask_path: str) -> np.ndarray:
    bg = cv2.imread(mask_path)
    if bg is None:
        raise FileNotFoundError(f"Missing mask: {mask_path}")
    return np.max(bg, axis=2) / 255.0


ALPHA_48 = load_alpha(MASK_48)
ALPHA_96 = load_alpha(MASK_96)


def _fixed_anchor(img: np.ndarray, sz: int, pad: int):
    """固定使用右下角区域（Gemini 水印规则位置）。"""
    h, w = img.shape[:2]
    y0 = h - pad - sz
    x0 = w - pad - sz
    if y0 < 0 or x0 < 0:
        return None
    return (y0, x0)


def _find_best_anchor(img: np.ndarray, sz: int, pad: int, a_map: np.ndarray, search: int = 20):
    """右下角附近搜索最可能的水印位置（亮度+alpha权重）。"""
    h, w = img.shape[:2]
    base = _fixed_anchor(img, sz, pad)
    if base is None:
        return None
    y0, x0 = base

    y_min = max(0, y0 - search)
    y_max = min(h - sz, y0 + search)
    x_min = max(0, x0 - search)
    x_max = min(w - sz, x0 + search)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    wm = (a_map > 0.08).astype(np.float32)
    if wm.sum() <= 1:
        return (y0, x0)

    best = (y0, x0)
    best_score = -1e18
    for yy in range(y_min, y_max + 1):
        for xx in range(x_min, x_max + 1):
            patch = gray[yy:yy + sz, xx:xx + sz]
            if patch.shape[0] != sz or patch.shape[1] != sz:
                continue
            score = float((patch * wm).sum() - 0.35 * patch.mean() * wm.sum())
            if score > best_score:
                best_score = score
                best = (yy, xx)
    return best


def _cleanup_corner_residual(out: np.ndarray) -> np.ndarray:
    """兜底：清理右下角残余小水印（如小星标/边缘残影）。"""
    h, w = out.shape[:2]
    box = 120
    y0, x0 = max(0, h - box), max(0, w - box)
    roi = out[y0:h, x0:w].copy()

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(th, connectivity=8)
    mask = np.zeros_like(th)
    for i in range(1, num_labels):
        x, y, ww, hh, area = stats[i]
        if 8 <= area <= 1200 and ww <= 80 and hh <= 80:
            mask[labels == i] = 255

    # 强制兜底：右下角固定小框（Gemini角标常驻区域）
    fy2, fx2 = max(0, roi.shape[0] - 16), max(0, roi.shape[1] - 16)
    fy1, fx1 = max(0, fy2 - 56), max(0, fx2 - 56)
    corner = gray[fy1:fy2, fx1:fx2]
    if corner.size:
        hard = (corner > 80).astype(np.uint8) * 255
        mask[fy1:fy2, fx1:fx2] = cv2.max(mask[fy1:fy2, fx1:fx2], hard)

    if mask.any():
        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=2)
        repaired = cv2.inpaint(roi, mask, 4, cv2.INPAINT_TELEA)
        out[y0:h, x0:w] = repaired
    return out


def clean_watermark(img: np.ndarray, mode: str = "fixed") -> np.ndarray:
    """
    基于 Reverse Alpha Blending 的无损去水印。
    默认固定右下角；对 1024x572 横版使用实测坐标。
    """
    h, w = img.shape[:2]

    # 用户锁定：1024x572 横版固定清理框 [x:959.15~1005.08, y:509.06~551.51]
    if abs(w - 1024) <= 2 and abs(h - 572) <= 2 and mode != "search":
        x_start, y_start = 959, 509
        x_end, y_end = 1006, 552
        x_end = min(x_end, w)
        y_end = min(y_end, h)
        if x_start >= x_end or y_start >= y_end:
            return img

        out = img.copy()
        crop = out[y_start:y_end, x_start:x_end].astype(np.float32)

        alpha = cv2.resize(ALPHA_48.astype(np.float32), (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_LINEAR)
        alpha = np.clip(alpha, 0.0, 0.995)
        alpha3 = alpha[..., None]
        denom = np.maximum(1.0 - alpha3, 1e-6)

        recovered = (crop - alpha3 * 255.0) / denom
        recovered = np.where(alpha3 > 1e-4, recovered, crop)
        rec_u8 = np.clip(recovered, 0, 255).astype(np.uint8)

        edge_mask = ((alpha > 0.003) & (alpha < 0.55)).astype(np.uint8)
        edge_mask = cv2.dilate(edge_mask, np.ones((5, 5), np.uint8), iterations=2)
        med = cv2.medianBlur(rec_u8, 3)
        smooth = cv2.GaussianBlur(med, (7, 7), 0)
        em = edge_mask.astype(bool)
        rec_u8[em] = smooth[em]

        out[y_start:y_end, x_start:x_end] = rec_u8

        # 精确兜底：只按水印形状掩膜修复，避免整块区域被抹糊
        precise_mask = (alpha > 0.06).astype(np.uint8) * 255
        precise_mask = cv2.dilate(precise_mask, np.ones((3, 3), np.uint8), iterations=1)
        out[y_start:y_end, x_start:x_end] = cv2.inpaint(
            out[y_start:y_end, x_start:x_end], precise_mask, 2, cv2.INPAINT_TELEA
        )

        return out

    if w > 1024 and h > 1024:
        sz, pad, a_map = 96, 64, ALPHA_96
    else:
        sz, pad, a_map = 48, 32, ALPHA_48

    if mode == "search":
        anchor = _find_best_anchor(img, sz, pad, a_map, search=20)
    else:
        anchor = _fixed_anchor(img, sz, pad)
    if anchor is None:
        return img

    y_start, x_start = anchor
    out = img.copy()
    crop = out[y_start:y_start + sz, x_start:x_start + sz].astype(np.float32)

    alpha = np.clip(a_map.astype(np.float32), 0.0, 0.995)
    alpha3 = alpha[..., None]
    denom = np.maximum(1.0 - alpha3, 1e-6)

    recovered = (crop - alpha3 * 255.0) / denom
    recovered = np.where(alpha3 > 1e-4, recovered, crop)
    rec_u8 = np.clip(recovered, 0, 255).astype(np.uint8)

    # 对水印边缘残留做轻量滤波（仅作用于 alpha 边缘带）
    edge_mask = ((alpha > 0.003) & (alpha < 0.55)).astype(np.uint8)
    edge_mask = cv2.dilate(edge_mask, np.ones((5, 5), np.uint8), iterations=2)
    # 适中强度：轻中值 + 轻高斯
    med = cv2.medianBlur(rec_u8, 3)
    smooth = cv2.GaussianBlur(med, (7, 7), 0)
    em = edge_mask.astype(bool)
    rec_u8[em] = smooth[em]

    out[y_start:y_start + sz, x_start:x_start + sz] = rec_u8
    return _cleanup_corner_residual(out)


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/batch-clean")
def batch_clean():
    files = request.files.getlist("images")
    mode = (request.form.get("mode") or "fixed").strip().lower()
    if mode not in ("fixed", "search"):
        mode = "fixed"
    if not files:
        return jsonify({"ok": False, "message": "请先上传图片"}), 400

    zip_buffer = io.BytesIO()
    processed = []

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            data = np.frombuffer(f.read(), np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is None:
                continue

            cleaned = clean_watermark(img, mode=mode)
            name, ext = os.path.splitext(f.filename or "image.jpg")
            ext = ext.lower() if ext.lower() in [".jpg", ".jpeg", ".png", ".webp"] else ".jpg"
            ok, encoded = cv2.imencode(ext if ext != ".jpeg" else ".jpg", cleaned)
            if not ok:
                continue

            out_name = f"{name}_clean{ext}"
            zf.writestr(out_name, encoded.tobytes())
            processed.append(out_name)

    if not processed:
        return jsonify({"ok": False, "message": "没有可处理的有效图片"}), 400

    zip_buffer.seek(0)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return send_file(
        zip_buffer,
        mimetype="application/zip",
        as_attachment=True,
        download_name=f"nano-banana-clean-{ts}.zip",
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050, debug=True)
