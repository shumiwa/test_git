"""Generate synthetic shadowgraph-like bubble images with exact YOLO labels.

Use this to test the whole pipeline (train -> detect -> size) before you have
annotated real data, and to check that the size estimate recovers known sizes.

    python make_synthetic.py --out synthetic --n 200
"""
import argparse
from pathlib import Path

import cv2
import numpy as np


def make_image(rng, w=640, h=480, n_min=5, n_max=25):
    img = np.full((h, w), 200, np.float32)
    img += rng.normal(0, 4, img.shape)                      # sensor noise
    img *= np.linspace(0.85, 1.05, w)[None, :]              # uneven backlight
    labels, truth = [], []
    for _ in range(rng.integers(n_min, n_max + 1)):
        dx = rng.uniform(8, 60)                              # horizontal axis [px]
        dy = dx * rng.uniform(0.6, 1.0)                      # oblate bubble
        cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        ang = rng.uniform(-15, 15)
        axes = (int(round(dx / 2)), int(round(dy / 2)))
        center = (int(round(cx)), int(round(cy)))
        cv2.ellipse(img, center, axes, ang, 0, 360, 40, -1)  # dark bubble
        inner = (max(1, int(axes[0] * 0.45)), max(1, int(axes[1] * 0.45)))
        cv2.ellipse(img, center, inner, ang, 0, 360, 170, -1)  # bright core
        # exact bounding box of the drawn (rotated) ellipse
        mask = np.zeros((h, w), np.uint8)
        cv2.ellipse(mask, center, axes, ang, 0, 360, 255, -1)
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            continue
        x1, x2, y1, y2 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        labels.append(f"0 {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                      f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
        truth.append((cx, cy, 2 * axes[0], 2 * axes[1]))
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return np.clip(img, 0, 255).astype(np.uint8), labels, truth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="synthetic")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    img_dir, lbl_dir = Path(a.out, "images"), Path(a.out, "labels")
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    for i in range(a.n):
        img, labels, _ = make_image(rng)
        cv2.imwrite(str(img_dir / f"frame_{i:05d}.png"), img)
        (lbl_dir / f"frame_{i:05d}.txt").write_text("\n".join(labels) + "\n")
    print(f"Wrote {a.n} images + labels to {a.out}/")


if __name__ == "__main__":
    main()
