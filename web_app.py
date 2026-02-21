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


def clean_watermark(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    if w > 1024 and h > 1024:
        sz, pad, a_map = 96, 64, ALPHA_96
    else:
        sz, pad, a_map = 48, 32, ALPHA_48

    y, x = h - pad, w - pad
    y_start, x_start = y - sz, x - sz
    if y_start < 0 or x_start < 0:
        return img

    crop = img[y_start:y, x_start:x].astype(float)
    a_map = np.clip(a_map, 0, 0.999)
    norm = 1.0 - a_map
    res = np.zeros_like(crop)

    for i in range(3):
        res[:, :, i] = (crop[:, :, i] - (a_map * 255.0)) / norm

    out = img.copy()
    out[y_start:y, x_start:x] = np.clip(res, 0, 255)
    return out


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/batch-clean")
def batch_clean():
    files = request.files.getlist("images")
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

            cleaned = clean_watermark(img)
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
