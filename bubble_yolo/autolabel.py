"""Bootstrap YOLO labels from real frames with classical image processing.

Hand-drawing hundreds of boxes is the slowest step. This script proposes boxes
(background subtraction + Otsu threshold + hole filling + contours) and writes
them in YOLO format. Load the result into an annotation tool (CVAT, Label
Studio, Roboflow, ...), CORRECT the mistakes (merged/overlapping bubbles,
missed small bubbles, debris), then train. Do not train on uncorrected output:
the network would only learn to copy the thresholding errors.

    python autolabel.py --images raw_frames --out autolabels --min-area 20
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from common import list_images, load_frame


def estimate_background(paths, n=50):
    """Median of up to n frames = static background (bubbles move, it does not)."""
    step = max(1, len(paths) // n)
    stack = [cv2.cvtColor(load_frame(p), cv2.COLOR_BGR2GRAY) for p in paths[::step][:n]]
    return np.median(np.stack(stack), axis=0).astype(np.uint8)


def propose_boxes(gray, background, min_area):
    diff = cv2.absdiff(background, gray) if background is not None else 255 - gray
    diff = cv2.GaussianBlur(diff, (5, 5), 0)
    _, mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    # fill bright bubble cores so each bubble becomes one solid blob
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 255, -1)
    contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) >= min_area]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", default="autolabels")
    ap.add_argument("--min-area", type=float, default=20, help="pixels")
    ap.add_argument("--no-background", action="store_true",
                    help="skip median background subtraction (use for few frames)")
    a = ap.parse_args()
    paths = list_images(a.images)
    bg = None if a.no_background else estimate_background(paths)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "classes.txt").write_text("bubble\n")
    for p in paths:
        gray = cv2.cvtColor(load_frame(p), cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        lines = [f"0 {(x + bw / 2) / w:.6f} {(y + bh / 2) / h:.6f} {bw / w:.6f} {bh / h:.6f}"
                 for x, y, bw, bh in propose_boxes(gray, bg, a.min_area)]
        (out / f"{p.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
    print(f"Wrote proposals for {len(paths)} frames to {out}/ -- now review them!")


if __name__ == "__main__":
    main()
