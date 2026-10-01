"""Split annotated images into train/val and write the YOLO data.yaml.

Expects `--images` (frames) and `--labels` (one YOLO .txt per frame, same stem):
    <class> <x_center> <y_center> <width> <height>   (all normalised to 0-1)

    python prepare_dataset.py --images annotated/images --labels annotated/labels --out dataset

Tip: consecutive high-speed frames are nearly identical. If you split them
randomly, val ends up as copies of train and the metrics look too good.
`--block` keeps runs of consecutive frames together in the same split.
"""
import argparse
import random
import shutil
from pathlib import Path

from common import list_images


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--val", type=float, default=0.2, help="validation fraction")
    ap.add_argument("--block", type=int, default=20,
                    help="consecutive frames kept together in one split")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    imgs = [p for p in list_images(a.images) if (Path(a.labels) / f"{p.stem}.txt").exists()]
    if not imgs:
        raise SystemExit("No image/label pairs found.")
    blocks = [imgs[i:i + a.block] for i in range(0, len(imgs), a.block)]
    random.Random(a.seed).shuffle(blocks)
    n_val = max(1, round(len(blocks) * a.val))
    splits = {"val": blocks[:n_val], "train": blocks[n_val:]}

    out = Path(a.out).resolve()
    for split, bl in splits.items():
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for p in (p for b in bl for p in b):
            shutil.copy2(p, out / "images" / split / p.name)
            shutil.copy2(Path(a.labels) / f"{p.stem}.txt", out / "labels" / split / f"{p.stem}.txt")
        print(f"{split}: {sum(len(b) for b in bl)} images")

    (out / "data.yaml").write_text(
        f"path: {out}\ntrain: images/train\nval: images/val\nnames:\n  0: bubble\n")
    print(f"Wrote {out / 'data.yaml'}")


if __name__ == "__main__":
    main()
