"""Fine-tune a pretrained YOLO26 detector on the bubble dataset.

    python train.py --data dataset/data.yaml --model yolo26s.pt --imgsz 1024 --epochs 150

Model size: yolo26n (fastest) < s < m < l < x (most accurate). Start with `s`;
go to `m` if accuracy on small/overlapping bubbles is not enough.
Image size: use the native camera resolution (rounded to a multiple of 32) when
bubbles are small -- downscaling a 1024 px frame to 640 can erase 5 px bubbles.
"""
import argparse

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="yolo26s.pt")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch", type=int, default=-1, help="-1 = auto (fits GPU memory)")
    ap.add_argument("--device", default=None, help="0, 0,1, cpu, mps ...")
    ap.add_argument("--name", default="bubble_yolo26")
    a = ap.parse_args()

    model = YOLO(a.model)  # COCO-pretrained weights are downloaded automatically
    model.train(
        data=a.data,
        imgsz=a.imgsz,
        epochs=a.epochs,
        batch=a.batch,
        device=a.device,
        name=a.name,
        patience=30,        # stop early if val mAP stops improving
        single_cls=True,
        # Shadowgraph images are grayscale: colour augmentation is meaningless.
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.3,
        fliplr=0.5,
        flipud=0.0,         # bubbles rise; keep the up/down shape asymmetry
        degrees=0.0,
        mosaic=1.0,
        close_mosaic=10,
    )
    metrics = model.val()
    print(f"val mAP50={metrics.box.map50:.3f}  mAP50-95={metrics.box.map:.3f}")
    print(f"Best weights: {model.trainer.best}")


if __name__ == "__main__":
    main()
