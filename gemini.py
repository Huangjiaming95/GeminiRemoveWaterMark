import sys
import os
import cv2
import numpy as np


def refine_edge_band(patch: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    # 以 alpha 形状构造“核心+边缘带”，只修边不糊主体
    core = (alpha > 0.07).astype(np.uint8) * 255
    if not core.any():
        return patch

    outer = cv2.dilate(core, np.ones((3, 3), np.uint8), iterations=2)
    inner = cv2.erode(core, np.ones((3, 3), np.uint8), iterations=1)
    ring = cv2.subtract(outer, inner)
    ring = cv2.bitwise_and(ring, ((alpha > 0.01).astype(np.uint8) * 255))
    if not ring.any():
        return patch

    # 先用核心mask估计“周围背景颜色”，再只在ring上回填
    bg_est = cv2.inpaint(patch, core, 3, cv2.INPAINT_TELEA)
    out = patch.copy()

    m = ring.astype(bool)
    if m.any():
        p = out[m].astype(np.float32)
        b = bg_est[m].astype(np.float32)

        # 对黑边（明显偏暗）做更强替换；其余做轻量混合
        gp = p.mean(axis=1)
        gb = b.mean(axis=1)
        dark_idx = gp < (gb - 8)

        p2 = p.copy()
        if dark_idx.any():
            p2[dark_idx] = 0.72 * b[dark_idx] + 0.28 * p[dark_idx]
        if (~dark_idx).any():
            p2[~dark_idx] = 0.45 * b[~dark_idx] + 0.55 * p[~dark_idx]

        out[m] = np.clip(p2, 0, 255).astype(np.uint8)

    return out


def cleanup_corner_residual(out: np.ndarray) -> np.ndarray:
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


if len(sys.argv) < 2:
    print("Drag and drop a file onto the script.")
    input()
    sys.exit()

f_path = sys.argv[1]
print(f"Fixing: {f_path}")

img = cv2.imread(f_path)
if img is None:
    sys.exit("Bad file")

h, w = img.shape[:2]
base_dir = os.path.dirname(os.path.abspath(__file__))

mask_48 = os.path.join(base_dir, "48.png")
mask_96 = os.path.join(base_dir, "96.png")
if not os.path.exists(mask_48) or not os.path.exists(mask_96):
    sys.exit("Missing 48.png or 96.png")

alpha_48 = np.max(cv2.imread(mask_48), axis=2).astype(np.float32) / 255.0
alpha_96 = np.max(cv2.imread(mask_96), axis=2).astype(np.float32) / 255.0

# 1024x572 横版固定坐标特判
if abs(w - 1024) <= 2 and abs(h - 572) <= 2:
    x_start, y_start = 959, 509
    x_end, y_end = min(1006, w), min(552, h)
    if x_start >= x_end or y_start >= y_end:
        sys.exit("Invalid custom watermark box")

    crop = img[y_start:y_end, x_start:x_end].astype(np.float32)
    alpha = cv2.resize(alpha_48, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_LINEAR)
    alpha = np.clip(alpha, 0.0, 0.995)
    alpha3 = alpha[..., None]
    denom = np.maximum(1.0 - alpha3, 1e-6)

    res = (crop - alpha3 * 255.0) / denom
    res = np.where(alpha3 > 1e-4, res, crop)
    res_u8 = np.clip(res, 0, 255).astype(np.uint8)
    res_u8 = refine_edge_band(res_u8, alpha)

    img[y_start:y_end, x_start:x_end] = res_u8
else:
    # 固定右下角区域：横版竖版一致
    if w > 1024 and h > 1024:
        sz, pad, a_map = 96, 64, alpha_96
    else:
        sz, pad, a_map = 48, 32, alpha_48

    y_start = h - pad - sz
    x_start = w - pad - sz
    if y_start < 0 or x_start < 0:
        sys.exit("Image too small for watermark region")

    crop = img[y_start:y_start + sz, x_start:x_start + sz].astype(np.float32)
    alpha = np.clip(a_map, 0.0, 0.995)
    alpha3 = alpha[..., None]
    denom = np.maximum(1.0 - alpha3, 1e-6)

    res = (crop - alpha3 * 255.0) / denom
    res = np.where(alpha3 > 1e-4, res, crop)
    res_u8 = np.clip(res, 0, 255).astype(np.uint8)
    res_u8 = refine_edge_band(res_u8, alpha)

    img[y_start:y_start + sz, x_start:x_start + sz] = res_u8

img = cleanup_corner_residual(img)

name, ext = os.path.splitext(f_path)
out = f"{name}_clean{ext or '.png'}"
cv2.imwrite(out, img)

print("Done.")
