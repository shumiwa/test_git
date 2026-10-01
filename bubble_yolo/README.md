# Bubble detection and sizing from high-speed camera images with YOLO26

This folder holds a complete pipeline. It uses **Ultralytics YOLO26**, the latest
YOLO generation (`ultralytics>=8.4`), and works like this:

1. it detects every bubble in each high-speed frame,
2. it gives each detection an **ID** (`bubble_id`, unique per frame). With
   `--track` it adds a `track_id` that stays the same for one bubble across frames,
3. it turns the **bounding-box coordinates** into bubble size: width, height,
   equivalent diameters, aspect ratio and volume, in mm.

```
raw frames ──► autolabel.py ──► correct in CVAT / Label Studio ──► prepare_dataset.py
                (proposals)        (ground truth)                    (train/val split)
                                                                            │
results/bubbles.csv ◄── detect_bubbles.py ◄── best.pt ◄── train.py ◄────────┘
results/tracks.csv       (ID + size)
```

| File | Purpose |
|---|---|
| `common.py` | Loads images and converts 10/12/16-bit TIFF frames to 8 bit |
| `make_synthetic.py` | Makes synthetic bubble images with exact labels, for testing the pipeline |
| `autolabel.py` | Proposes boxes with classical thresholding, so you don't have to annotate from scratch |
| `prepare_dataset.py` | Splits frames into train/val by blocks of frames and writes `data.yaml` |
| `train.py` | Fine-tunes a pretrained YOLO26 model on your bubbles |
| `detect_bubbles.py` | Runs inference, assigns IDs, estimates sizes and writes CSVs, a histogram and annotated frames |

---

## Step 0 – Install

```bash
python -m venv .venv && source .venv/bin/activate
# Install the CUDA build of PyTorch first if you have an NVIDIA GPU: https://pytorch.org/get-started/
pip install -r requirements.txt
```

You can train on a CPU, but it is slow. A single consumer GPU trains a few
hundred images in minutes.

**Optional smoke test.** This runs the whole pipeline on synthetic data:

```bash
python make_synthetic.py --out synthetic --n 200
python prepare_dataset.py --images synthetic/images --labels synthetic/labels --out ds_syn
python train.py --data ds_syn/data.yaml --model yolo26n.pt --imgsz 640 --epochs 30
python detect_bubbles.py --weights runs/detect/bubble_yolo26/weights/best.pt \
       --source synthetic/images --mm-per-px 0.02 --imgsz 640 --save-vis
```

Test result: `yolo26n` trained for only 15 CPU epochs on 100 synthetic frames.
It reached mAP50 = 0.97 and found 151 of 153 bubbles in unseen frames. The box
width and height were within about 0.5 px of the truth, a mean error of about
1.7 % in diameter.

## Step 1 – Get good images and a spatial calibration

* **Calibration (mm/px):** record a ruler or dot-grid target in the bubble plane
  with the same lens and focus. `mm_per_px = known length [mm] / length [px]`.
  This one number turns pixels into physical size, so measure it carefully.
* Backlit shadowgraphy (dark bubbles on a bright background) gives the
  sharpest edges. Use a short exposure so a bubble moves less than about 1 px
  during the exposure. Otherwise motion blur stretches the box along the motion.
* Keep the frames as numbered image files (`frame_00001.tif`, ...). If your
  camera software saves `.cine`, `.mraw` or a video, export it to TIFF or PNG.
  16-bit TIFFs are handled automatically.

## Step 2 – Annotate (make the ground truth)

YOLO needs examples. Plan on **about 100–300 frames** that cover the range of
conditions: flow rates, bubble sizes, dense and sparse regions, overlapping
bubbles, and lighting changes.

1. **Bootstrap with proposals.** This saves most of the annotation time:
   ```bash
   python autolabel.py --images raw_frames --out autolabels --min-area 20
   ```
2. Import the frames and `autolabels/*.txt` into an annotation tool, in YOLO
   format. Good choices are **CVAT**, **Label Studio**, **Roboflow** or
   **X-AnyLabeling**. Then **fix every frame**: split merged bubbles, add missed
   small ones, and delete debris or reflections.
3. Rules for consistent boxes. These matter because **the box is your size measurement**:
   * Draw the box tight around the **outer dark edge** of the bubble.
   * For partly hidden bubbles, box only the visible part, or skip them, but do
     the same thing every time.
   * Annotate bubbles cut off by the image edge too. The analysis flags them
     and leaves them out of the size statistics.

Label format, one `.txt` per image with one line per bubble, all values normalised to 0–1:
```
0 x_center y_center width height
```

## Step 3 – Build the dataset

```bash
python prepare_dataset.py --images annotated/images --labels annotated/labels \
       --out dataset --val 0.2 --block 20
```
Consecutive high-speed frames look almost the same. A random split would put
near-copies of training images into the validation set and make the metrics
look too good. `--block 20` keeps runs of 20 consecutive frames in the same split.
For an even fairer test, validate on a separate recording.

## Step 4 – Train YOLO26

```bash
python train.py --data dataset/data.yaml --model yolo26s.pt --imgsz 1024 --epochs 150
```
* **Model:** `yolo26n` < `s` < `m` < `l` < `x`, from fastest to most accurate.
  Start with `s`.
* **imgsz:** use your camera resolution, rounded to a multiple of 32, such as
  1024 or 1280. Small bubbles disappear when a frame is shrunk to 640.
* Augmentation is set for grayscale shadowgraphs: no hue or saturation
  changes, horizontal flips only.
* Check `runs/detect/bubble_yolo26/` for `results.png` (loss and mAP curves),
  `confusion_matrix.png` and the `val_batch*_pred.jpg` images.
  Aim for **mAP50 > 0.9**. If it is lower, annotate more frames of the cases
  that fail.

Same thing from the command line:
`yolo detect train data=dataset/data.yaml model=yolo26s.pt imgsz=1024 epochs=150`

## Step 5 – Detect, assign IDs and measure

```bash
# complete size statistics (every detection)
python detect_bubbles.py --weights runs/detect/bubble_yolo26/weights/best.pt \
       --source raw_frames --mm-per-px 0.02 --imgsz 1024 --save-vis --out results

# plus the same ID for one bubble across frames, with rise velocity
python detect_bubbles.py --weights runs/detect/bubble_yolo26/weights/best.pt \
       --source raw_frames --mm-per-px 0.02 --fps 10000 --imgsz 1024 --track --out results_track
```

`results/bubbles.csv` has one row per detected bubble per frame:

| column | meaning |
|---|---|
| `frame`, `file`, `time_s` | where and when |
| `bubble_id` | `FFFFFF_NNNN`: frame number + detection index, unique for every detection |
| `track_id` | with `--track`: the same number for the same bubble in every frame it appears in |
| `conf` | detection confidence |
| `x1,y1,x2,y2` | bounding box in pixels (top-left, bottom-right) |
| `xc_px,yc_px,w_px,h_px` | box centre and size in pixels |
| `w_mm,h_mm` | horizontal and vertical extent in mm |
| `d_area_mm` | projected-area equivalent diameter |
| `d_eq_mm` | volume-equivalent diameter |
| `aspect_ratio` | E = h / w |
| `volume_mm3` | spheroid volume |
| `touches_border` | box is cut off by the image edge, so it is left out of the statistics |

`tracks.csv` (with `--track`) gives each bubble's mean and std of `d_eq`, the
number of frames it appears in, and its velocity (`u`, `v`; `v` > 0 means rising).

### How size comes from the bounding box

With box width `w` and height `h`, both converted to mm with `mm_per_px`:

* Inscribed-ellipse area `A = π/4·w·h` → **d_A = √(w·h)**
* Bubble as a spheroid with a vertical symmetry axis, `V = π/6·w²·h`
  → **d_eq = (w²·h)^(1/3)**. This is the usual choice for bubbly flow and mass transfer.
* **E = h/w** (< 1 for oblate, flattened bubbles)

## Accuracy limits and how to improve them

* **Tilted or wobbly bubbles:** an axis-aligned box around a tilted ellipse is
  larger than the ellipse, so sizes come out too big. For large, deformed
  bubbles, consider **YOLO26-seg** (`yolo26s-seg.pt`, instance masks: compute the
  area or fit an ellipse to the mask) or **YOLO26-obb** (rotated boxes). Both are
  drop-in replacements in the same Ultralytics API.
* **Overlapping clusters:** the main source of errors in dense flow. Annotate
  many overlap examples. Keep `--max-det` high: the default here is 1000, while
  YOLO's own default is 300.
* **Very small bubbles (< ~8 px):** use a larger `--imgsz`, or tiled inference
  (SAHI: `pip install sahi`).
* **Box tightness bias:** annotation style becomes a systematic size bias.
  Check it with something of known size, such as calibration spheres or needle
  bubbles of known volume, and correct for it if needed.
* **Tracking:** ByteTrack only reports detections it has matched to a track, so
  a bubble can be missing from `--track` output in its first frames. Use the
  non-tracking run for size distributions and the tracking run for per-bubble
  trajectories and velocities.
* **Statistics:** `size_hist.png` counts *detections*. A bubble seen in 50
  frames counts 50 times. For a number-based distribution, use `tracks.csv`
  (one entry per bubble) or sample frames far enough apart that bubbles do not
  repeat.
