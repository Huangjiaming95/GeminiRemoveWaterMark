import os
import sys

import cv2
import numpy as np


def imread_unicode(path: str, flags=cv2.IMREAD_COLOR):
    """兼容 Windows 中文路径读取。"""
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, flags)
    except Exception:
        return None


def imwrite_unicode(path: str, img: np.ndarray) -> bool:
    """兼容 Windows 中文路径写入。"""
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return False
    try:
        buf.tofile(path)
        return True
    except Exception:
        return False


def refine_edge_band(recovered_patch: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """先反解，再只修复水印边缘残留的黑边。"""
    core = (alpha > 0.07).astype(np.uint8) * 255
    if not core.any():
        return recovered_patch

    outer = cv2.dilate(core, np.ones((3, 3), np.uint8), iterations=2)
    inner = cv2.erode(core, np.ones((3, 3), np.uint8), iterations=1)
    ring = cv2.subtract(outer, inner)
    ring = cv2.bitwise_and(ring, ((alpha > 0.01).astype(np.uint8) * 255))
    if not ring.any():
        return recovered_patch

    gray = cv2.cvtColor(recovered_patch, cv2.COLOR_BGR2GRAY)
    dark_mask = cv2.bitwise_and(((gray < 64).astype(np.uint8) * 255), ring)
    if not dark_mask.any():
        return recovered_patch

    dark_mask = cv2.dilate(dark_mask, np.ones((3, 3), np.uint8), iterations=1)
    repaired = cv2.inpaint(recovered_patch, dark_mask, 2, cv2.INPAINT_TELEA)

    out = recovered_patch.copy()
    mask = dark_mask.astype(bool)
    out[mask] = repaired[mask]
    return out


def cleanup_corner_residual(out: np.ndarray) -> np.ndarray:
    """兜底清理右下角残留的小角标或边缘残影。"""
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


def process_rect(
    img: np.ndarray,
    x_start: int,
    y_start: int,
    x_end: int,
    y_end: int,
    alpha_src: np.ndarray,
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
    fixed_u8 = refine_edge_band(recovered_u8, alpha)
    out[y_start:y_end, x_start:x_end] = fixed_u8
    return out


def main():
    if len(sys.argv) < 2:
        print("请把图片拖到这个脚本上，或在命令行后面传入图片路径。")
        input()
        return 1

    file_path = sys.argv[1]
    print(f"正在处理: {file_path}")

    img = imread_unicode(file_path)
    if img is None:
        print("图片读取失败，请检查文件是否存在或格式是否正确。")
        return 1

    h, w = img.shape[:2]
    base_dir = os.path.dirname(os.path.abspath(__file__))

    mask_48 = os.path.join(base_dir, "48.png")
    mask_96 = os.path.join(base_dir, "96.png")
    if not os.path.exists(mask_48) or not os.path.exists(mask_96):
        print("缺少 48.png 或 96.png。")
        return 1

    m48 = imread_unicode(mask_48)
    m96 = imread_unicode(mask_96)
    if m48 is None or m96 is None:
        print("48.png 或 96.png 无法读取，请检查文件是否损坏。")
        return 1

    alpha_48 = np.max(m48, axis=2).astype(np.float32) / 255.0
    alpha_96 = np.max(m96, axis=2).astype(np.float32) / 255.0

    if abs(w - 1024) <= 2 and abs(h - 572) <= 2:
        x_start, y_start = 959, 509
        x_end, y_end = min(1006, w), min(552, h)
        img = process_rect(img, x_start, y_start, x_end, y_end, alpha_48)
    elif abs(w - 2752) <= 2 and abs(h - 1536) <= 2:
        x_start, y_start = 2577, 1368
        x_end, y_end = min(2704, w), min(1482, h)
        img = process_rect(img, x_start, y_start, x_end, y_end, alpha_96)
    else:
        if w > 1024 and h > 1024:
            sz, pad, alpha_map = 96, 64, alpha_96
        else:
            sz, pad, alpha_map = 48, 32, alpha_48

        y_start = h - pad - sz
        x_start = w - pad - sz
        if y_start < 0 or x_start < 0:
            print("图片太小，无法定位右下角水印区域。")
            return 1

        crop = img[y_start:y_start + sz, x_start:x_start + sz].astype(np.float32)
        alpha = np.clip(alpha_map, 0.0, 0.995)
        alpha3 = alpha[..., None]
        denom = np.maximum(1.0 - alpha3, 1e-6)

        recovered = (crop - alpha3 * 255.0) / denom
        recovered = np.where(alpha3 > 1e-4, recovered, crop)
        recovered_u8 = np.clip(recovered, 0, 255).astype(np.uint8)
        fixed_u8 = refine_edge_band(recovered_u8, alpha)
        img[y_start:y_start + sz, x_start:x_start + sz] = fixed_u8

    img = cleanup_corner_residual(img)

    name, ext = os.path.splitext(file_path)
    output_path = f"{name}_clean{ext or '.png'}"
    if not imwrite_unicode(output_path, img):
        print("保存处理结果失败。")
        return 1

    print(f"处理完成: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
