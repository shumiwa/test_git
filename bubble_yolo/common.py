"""Shared helpers for the bubble-detection pipeline."""
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def list_images(folder):
    """Return the image files in `folder`, sorted by name (= frame order)."""
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMG_EXTS)


def load_frame(path):
    """Read a high-speed camera frame as an 8-bit, 3-channel BGR image.

    High-speed cameras often save 10/12/16-bit grayscale TIFFs. YOLO expects
    8-bit images, so deeper images are contrast-stretched (0.5-99.5 percentile)
    to 8 bit. 8-bit images are returned unchanged apart from gray->BGR.
    """
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"Cannot read image: {path}")
    if img.dtype != np.uint8:
        lo, hi = np.percentile(img, (0.5, 99.5))
        img = np.clip((img.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0, 1)
        img = (img * 255).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img
