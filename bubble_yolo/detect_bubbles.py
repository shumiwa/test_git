"""Detect bubbles with a trained YOLO26 model, ID them, and estimate their size.

    python detect_bubbles.py --weights runs/detect/bubble_yolo26/weights/best.pt \
        --source raw_frames --mm-per-px 0.02 --fps 10000 --track --save-vis

Outputs (in --out):
    bubbles.csv  one row per detected bubble per frame
    tracks.csv   one row per tracked bubble (only with --track)
    size_hist.png
    vis/         annotated frames (only with --save-vis)

Size from the bounding box (w = horizontal, h = vertical extent):
    projected-area equivalent diameter  d_A  = sqrt(w*h)
        (diameter of the circle with the area of the inscribed ellipse pi/4*w*h)
    volume equivalent diameter          d_eq = (w*w*h)^(1/3)
        (bubble treated as a spheroid with a vertical symmetry axis:
         V = pi/6 * w^2 * h)
    aspect ratio                        E    = h / w
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

from common import list_images, load_frame


def bubble_size(w_px, h_px, mm_per_px):
    w, h = w_px * mm_per_px, h_px * mm_per_px
    d_eq = np.cbrt(w * w * h)
    return {
        "w_mm": w,
        "h_mm": h,
        "d_area_mm": np.sqrt(w * h),
        "d_eq_mm": d_eq,
        "aspect_ratio": h / w if w > 0 else np.nan,
        "volume_mm3": np.pi / 6 * d_eq ** 3,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--source", required=True, help="folder of frames")
    ap.add_argument("--out", default="results")
    ap.add_argument("--mm-per-px", type=float, required=True,
                    help="spatial calibration, e.g. from an image of a ruler/target")
    ap.add_argument("--fps", type=float, default=None, help="frame rate, for velocities")
    ap.add_argument("--imgsz", type=int, default=1024, help="use the training imgsz")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--max-det", type=int, default=1000, help="raise for dense bubbly flow")
    ap.add_argument("--border", type=int, default=2,
                    help="boxes within this many px of the edge are flagged as cut off")
    ap.add_argument("--track", action="store_true",
                    help="keep the same ID for a bubble across frames (ByteTrack). The "
                         "tracker drops detections it cannot match to a track yet, so "
                         "run once WITHOUT --track for complete size statistics")
    ap.add_argument("--save-vis", action="store_true")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    model = YOLO(a.weights)
    out = Path(a.out)
    (out / "vis").mkdir(parents=True, exist_ok=True) if a.save_vis else out.mkdir(parents=True, exist_ok=True)

    rows = []
    for frame_idx, path in enumerate(list_images(a.source)):
        img = load_frame(path)
        H, W = img.shape[:2]
        kw = dict(imgsz=a.imgsz, conf=a.conf, max_det=a.max_det, device=a.device, verbose=False)
        res = (model.track(img, persist=True, tracker="bytetrack.yaml", **kw) if a.track
               else model.predict(img, **kw))[0]

        boxes = res.boxes
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        tids = boxes.id.int().cpu().numpy() if boxes.id is not None else [None] * len(xyxy)

        for i, ((x1, y1, x2, y2), c, tid) in enumerate(zip(xyxy, confs, tids)):
            w_px, h_px = x2 - x1, y2 - y1
            rows.append({
                "frame": frame_idx,
                "file": path.name,
                "time_s": frame_idx / a.fps if a.fps else np.nan,
                "bubble_id": f"{frame_idx:06d}_{i:04d}",   # unique per detection
                "track_id": tid,                          # same bubble across frames
                "conf": float(c),
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "xc_px": (x1 + x2) / 2, "yc_px": (y1 + y2) / 2,
                "w_px": w_px, "h_px": h_px,
                "touches_border": bool(x1 <= a.border or y1 <= a.border
                                       or x2 >= W - a.border or y2 >= H - a.border),
                **bubble_size(w_px, h_px, a.mm_per_px),
            })

        if a.save_vis:
            vis = img.copy()
            for r in (r for r in rows if r["frame"] == frame_idx):
                color = (0, 0, 255) if r["touches_border"] else (0, 200, 0)
                p1, p2 = (int(r["x1"]), int(r["y1"])), (int(r["x2"]), int(r["y2"]))
                cv2.rectangle(vis, p1, p2, color, 1)
                label = f"{r['track_id'] if r['track_id'] is not None else r['bubble_id'][-4:]}" \
                        f" {r['d_eq_mm']:.2f}mm"
                cv2.putText(vis, label, (p1[0], max(10, p1[1] - 3)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
            cv2.imwrite(str(out / "vis" / f"{path.stem}.png"), vis)

    df = pd.DataFrame(rows)
    df.to_csv(out / "bubbles.csv", index=False)
    print(f"{len(df)} detections in {df['frame'].nunique() if len(df) else 0} frames "
          f"-> {out / 'bubbles.csv'}")
    if df.empty:
        return

    full = df[~df["touches_border"]]   # cut-off bubbles would bias sizes low
    print(full[["d_area_mm", "d_eq_mm", "aspect_ratio"]].describe().round(3).to_string())
    if a.track and df["track_id"].notna().any():
        write_tracks(full, out, a)
    plot_histogram(full, out)


def write_tracks(df, out, a):
    """Per-bubble statistics over its lifetime in the field of view."""
    df = df.dropna(subset=["track_id"]).sort_values("frame")
    recs = []
    for tid, g in df.groupby("track_id"):
        rec = {"track_id": int(tid), "n_frames": len(g),
               "first_frame": int(g["frame"].iloc[0]), "last_frame": int(g["frame"].iloc[-1]),
               "d_eq_mm_mean": g["d_eq_mm"].mean(), "d_eq_mm_std": g["d_eq_mm"].std(),
               "aspect_ratio_mean": g["aspect_ratio"].mean()}
        if a.fps and len(g) > 1:
            dt = (g["frame"].iloc[-1] - g["frame"].iloc[0]) / a.fps
            rec["u_mm_s"] = (g["xc_px"].iloc[-1] - g["xc_px"].iloc[0]) * a.mm_per_px / dt
            # image y points down -> minus sign makes rising bubbles positive
            rec["v_mm_s"] = -(g["yc_px"].iloc[-1] - g["yc_px"].iloc[0]) * a.mm_per_px / dt
        recs.append(rec)
    tracks = pd.DataFrame(recs)
    tracks.to_csv(out / "tracks.csv", index=False)
    print(f"{len(tracks)} tracks -> {out / 'tracks.csv'}")


def plot_histogram(df, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df["d_eq_mm"], bins=40, color="#3b6ea5")
    ax.set_xlabel("Volume-equivalent diameter $d_{eq}$ [mm]")
    ax.set_ylabel("Count (detections)")
    ax.set_title(f"Bubble size distribution (n={len(df)}, border bubbles excluded)")
    fig.tight_layout()
    fig.savefig(out / "size_hist.png", dpi=150)


if __name__ == "__main__":
    main()
