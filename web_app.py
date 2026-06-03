import io
import os
import zipfile
from datetime import datetime

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request, send_file


def imread_unicode(path: str, flags=cv2.IMREAD_COLOR):
    """兼容 Windows 中文路径读取。"""
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, flags)
    except Exception:
        return None


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MASK_48 = os.path.join(BASE_DIR, "48.png")
MASK_96 = os.path.join(BASE_DIR, "96.png")

app = Flask(__name__)


def load_alpha(mask_path: str) -> np.ndarray:
    bg = imread_unicode(mask_path)
    if bg is None:
        raise FileNotFoundError(f"缺少遮罩文件: {mask_path}")
    return np.max(bg, axis=2) / 255.0


ALPHA_48 = load_alpha(MASK_48)
ALPHA_96 = load_alpha(MASK_96)


def _fixed_anchor(img: np.ndarray, sz: int, pad: int):
    """固定使用右下角区域。"""
    h, w = img.shape[:2]
    y0 = h - pad - sz
    x0 = w - pad - sz
    if y0 < 0 or x0 < 0:
        return None
    return y0, x0


def _find_best_anchor(img: np.ndarray, sz: int, pad: int, alpha_map: np.ndarray, search: int = 20):
    """在右下角附近搜索最可能的水印位置。"""
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
    watermark_weight = (alpha_map > 0.08).astype(np.float32)
    if watermark_weight.sum() <= 1:
        return y0, x0

    best = (y0, x0)
    best_score = -1e18
    for yy in range(y_min, y_max + 1):
        for xx in range(x_min, x_max + 1):
            patch = gray[yy:yy + sz, xx:xx + sz]
            if patch.shape[0] != sz or patch.shape[1] != sz:
                continue
            score = float((patch * watermark_weight).sum() - 0.35 * patch.mean() * watermark_weight.sum())
            if score > best_score:
                best_score = score
                best = (yy, xx)
    return best


def _refine_edge_band(recovered_patch: np.ndarray, alpha: np.ndarray, strength: str = "normal") -> np.ndarray:
    """在反解结果上进一步修复偏黑的边缘残影。"""
    is_strong = strength == "strong"
    core = (alpha > 0.07).astype(np.uint8) * 255
    if not core.any():
        return recovered_patch

    outer = cv2.dilate(core, np.ones((3, 3), np.uint8), iterations=3 if is_strong else 2)
    inner = cv2.erode(core, np.ones((3, 3), np.uint8), iterations=1)
    ring = cv2.subtract(outer, inner)
    ring = cv2.bitwise_and(ring, ((alpha > 0.01).astype(np.uint8) * 255))
    if not ring.any():
        return recovered_patch

    gray = cv2.cvtColor(recovered_patch, cv2.COLOR_BGR2GRAY)
    dark_thr = 74 if is_strong else 64
    dark_mask = cv2.bitwise_and(((gray < dark_thr).astype(np.uint8) * 255), ring)
    if not dark_mask.any():
        return recovered_patch

    dark_mask = cv2.dilate(dark_mask, np.ones((3, 3), np.uint8), iterations=2 if is_strong else 1)
    repaired = cv2.inpaint(recovered_patch, dark_mask, 3 if is_strong else 2, cv2.INPAINT_TELEA)

    out = recovered_patch.copy()
    mask = dark_mask.astype(bool)
    out[mask] = repaired[mask]
    return out


def _cleanup_corner_residual(out: np.ndarray) -> np.ndarray:
    """兜底清理右下角残留的小块内容。"""
    h, w = out.shape[:2]
    box = 120
    y0, x0 = max(0, h - box), max(0, w - box)
    roi = out[y0:h, x0:w].copy()

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)
    threshold = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(threshold, connectivity=8)
    mask = np.zeros_like(threshold)
    for i in range(1, num_labels):
        x, y, ww, hh, area = stats[i]
        if 8 <= area <= 1200 and ww <= 80 and hh <= 80:
            mask[labels == i] = 255

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


def _process_rect(
    img: np.ndarray,
    x_start: int,
    y_start: int,
    x_end: int,
    y_end: int,
    alpha_src: np.ndarray,
    strength: str = "normal",
) -> np.ndarray:
    if x_start >= x_end or y_start >= y_end:
        return img

    out = img.copy()
    crop = out[y_start:y_end, x_start:x_end].astype(np.float32)
    if crop.size == 0:
        return img

    alpha = cv2.resize(alpha_src.astype(np.float32), (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_LINEAR)
    alpha = np.clip(alpha, 0.0, 0.995)
    alpha3 = alpha[..., None]
    denom = np.maximum(1.0 - alpha3, 1e-6)

    recovered = (crop - alpha3 * 255.0) / denom
    recovered = np.where(alpha3 > 1e-4, recovered, crop)
    recovered_u8 = np.clip(recovered, 0, 255).astype(np.uint8)
    fixed_u8 = _refine_edge_band(recovered_u8, alpha, strength=strength)
    out[y_start:y_end, x_start:x_end] = fixed_u8
    return out


def clean_watermark(img: np.ndarray, mode: str = "fixed", edge_strength: str = "normal") -> np.ndarray:
    """基于 Reverse Alpha Blending 清理右下角水印。"""
    h, w = img.shape[:2]

    if abs(w - 1024) <= 2 and abs(h - 572) <= 2 and mode != "search":
        x_start, y_start = 959, 509
        x_end, y_end = min(1006, w), min(552, h)
        processed = _process_rect(img, x_start, y_start, x_end, y_end, ALPHA_48, strength=edge_strength)
        if processed is not img:
            return processed

    if w > 1024 and h > 1024:
        sz, pad, alpha_map = 96, 64, ALPHA_96
    else:
        sz, pad, alpha_map = 48, 32, ALPHA_48

    anchor = _find_best_anchor(img, sz, pad, alpha_map, search=20) if mode == "search" else _fixed_anchor(img, sz, pad)
    if anchor is None:
        return img

    y_start, x_start = anchor
    out = img.copy()
    crop = out[y_start:y_start + sz, x_start:x_start + sz].astype(np.float32)

    alpha = np.clip(alpha_map.astype(np.float32), 0.0, 0.995)
    alpha3 = alpha[..., None]
    denom = np.maximum(1.0 - alpha3, 1e-6)

    recovered = (crop - alpha3 * 255.0) / denom
    recovered = np.where(alpha3 > 1e-4, recovered, crop)
    recovered_u8 = np.clip(recovered, 0, 255).astype(np.uint8)
    fixed_u8 = _refine_edge_band(recovered_u8, alpha, strength=edge_strength)

    out[y_start:y_start + sz, x_start:x_start + sz] = fixed_u8
    return _cleanup_corner_residual(out)


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/batch-clean")
def batch_clean():
    files = request.files.getlist("images")
    mode = (request.form.get("mode") or "fixed").strip().lower()
    if mode not in ("fixed", "search", "2752"):
        mode = "fixed"

    edge_strength = (request.form.get("edge_strength") or "normal").strip().lower()
    if edge_strength not in ("normal", "strong"):
        edge_strength = "normal"

    if not files:
        return jsonify({"ok": False, "message": "请先上传图片。"}), 400

    zip_buffer = io.BytesIO()
    processed = []

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for file in files:
            data = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is None:
                continue

            cleaned = clean_watermark(img, mode=mode, edge_strength=edge_strength)
            name, ext = os.path.splitext(file.filename or "image.jpg")
            ext = ext.lower() if ext.lower() in [".jpg", ".jpeg", ".png", ".webp"] else ".jpg"
            ok, encoded = cv2.imencode(ext if ext != ".jpeg" else ".jpg", cleaned)
            if not ok:
                continue

            out_name = f"{name}_clean{ext}"
            archive.writestr(out_name, encoded.tobytes())
            processed.append(out_name)

    if not processed:
        return jsonify({"ok": False, "message": "没有可处理的有效图片。"}), 400

    zip_buffer.seek(0)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return send_file(
        zip_buffer,
        mimetype="application/zip",
        as_attachment=True,
        download_name=f"nano-banana-clean-{ts}.zip",
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050, debug=False)
