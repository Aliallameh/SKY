# 🛩️ SkyScouter

**Onboard visual reacquisition, tracking, and guidance overlays for small airborne targets.**

SkyScouter takes a video, finds a distant drone-like target, tracks it frame by frame, writes machine-readable target states, and renders an annotated review video. The repo now includes the trained YOLO checkpoints through **Git LFS**, so a fresh clone can run the trained detector profiles without hunting for model files.

---

## 🚀 What You Can Do Today

| Task | Status |
|---|---|
| Run a video through detector + tracker | ✅ Ready |
| Render readable overlays with bbox geometry | ✅ Ready |
| Evaluate sparse ground truth CSVs | ✅ Ready |
| Run the trained YOLO11s detector | ✅ Ready with Git LFS weights |
| Review hard-case frames in browser | ✅ Ready |
| Train a new YOLO detector | ✅ Scripted |
| Fly/control a real vehicle | ❌ Not implemented |

---

## 🧰 Fresh Machine Setup

### 1. Install prerequisites

You need:

- Python `3.10` to `3.12`
- Git
- Git LFS
- Optional but recommended: NVIDIA GPU + CUDA-capable PyTorch

Check Git LFS:

```powershell
git lfs version
```

If that command fails, install Git LFS first: [https://git-lfs.com](https://git-lfs.com)

### 2. Clone with model weights

```powershell
git lfs install
git clone https://github.com/Aliallameh/SKY.git
cd SKY
git lfs pull
```

The important model files should appear here:

```text
data/models/yolo11s_airborne_drone_vs_bird_v1/best.pt
data/models/yolo11s_airborne_drone_vs_bird_v1/last.pt
data/models/yolo11s_airborne_drone_vs_bird_v2/best.pt
data/models/yolo11s_airborne_drone_vs_bird_v2/last.pt
```

### 3. Create the Python environment

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If `py -3.12` is not available, use your installed Python:

```powershell
python -m venv .venv
```

### 4. Use a local Ultralytics settings folder

On some Windows machines, Ultralytics may try to read a blocked roaming profile path. Set this once per terminal:

```powershell
$env:YOLO_CONFIG_DIR = "$PWD"
```

---

## ▶️ Run The Trained Detector

Place your video in:

```text
data/videos/
```

Then run:

```powershell
.\.venv\Scripts\python.exe scripts/run_pipeline.py `
  --video data/videos/camera_20260423_113401.mp4 `
  --config configs/trained_yolo11s_v2_guidance_full.yaml `
  --output data/outputs/run_camera_113401
```

Main outputs:

| Output | What it is |
|---|---|
| `annotated.mp4` | Review video with overlays |
| `target_states.jsonl` | One target-state row per frame |
| `guidance_hints.jsonl` | Bearing/elevation/yaw proposal log |
| `diagnostics.csv` | Per-frame debug values |
| `manifest.json` | Reproducibility record |

---

## 🎯 Current Model Weights

The repo tracks these through Git LFS, not normal Git blobs:

| Model | Path | Use |
|---|---|---|
| YOLO11s airborne v1 | `data/models/yolo11s_airborne_drone_vs_bird_v1/best.pt` | earlier trained baseline |
| YOLO11s airborne v1 last | `data/models/yolo11s_airborne_drone_vs_bird_v1/last.pt` | resume/debug checkpoint |
| YOLO11s airborne v2 | `data/models/yolo11s_airborne_drone_vs_bird_v2/best.pt` | current trained profile |
| YOLO11s airborne v2 last | `data/models/yolo11s_airborne_drone_vs_bird_v2/last.pt` | resume/debug checkpoint |

Why Git LFS?

- Model weights are large binary files.
- Normal Git stores every binary change forever.
- Git LFS keeps the repo usable while still making clones reproducible.

---

## 🧪 Reproduce The Hard-Case Regression

The current hard case is the turn/reversal clip from `camera_20260423_113401`, frames `80-140`.

Ground truth packet:

```text
annotations/camera_20260423_113401_turn_review_strict/
```

Run the evaluator:

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_hard_case.py `
  --csv annotations/camera_20260423_113401_turn_review_strict/drone_sparse_gt_corrected.csv `
  --config configs/trained_yolo11s_v2_guidance_full.yaml `
  --output data/outputs/hard_case_camera_20260423_113401_turn_review_strict
```

Expected shape of the current result:

| Metric | Current behavior |
|---|---|
| Detector hit rate | still the main bottleneck |
| Tracker stale boxes | reduced |
| Negative false positives after exit | fixed in the latest tracker settings |

Open:

```text
data/outputs/hard_case_camera_20260423_113401_turn_review_strict/summary.md
```

---

## 🖍️ Annotate More Frames

Create a review packet:

```powershell
.\.venv\Scripts\python.exe scripts/prepare_drone_annotations.py `
  --video data/videos/camera_20260423_113401.mp4 `
  --output annotations/my_review_packet `
  --start-frame 80 `
  --end-frame 140
```

Then open:

```text
annotations/my_review_packet/bbox_annotator.html
```

The browser annotator writes CSV rows with:

```text
frame_id,image,x1,y1,x2,y2,review_status,visibility,occluded,label,notes
```

Use:

- `label=drone` for real target boxes
- `label=negative` for reviewed frames where the drone is gone
- empty box coordinates for negatives

---

## 🏋️ Train A New Detector

Build a YOLO dataset:

```powershell
.\.venv\Scripts\python.exe scripts/prepare_airborne_training_set.py `
  --manifest configs/training/airborne_dataset_manifest.yaml `
  --out-dir data/training/airborne_yolo_v3 `
  --link-mode copy
```

Train:

```powershell
.\.venv\Scripts\python.exe scripts/train_airborne_yolo.py `
  --config configs/training/airborne_yolo11.yaml
```

Training runs land under:

```text
data/training/runs/
```

Promote only curated checkpoints into:

```text
data/models/
```

Then commit them with Git LFS.

---

## 🗂️ Repo Map

| Path | Purpose |
|---|---|
| `configs/` | Runtime profiles and training configs |
| `data/models/` | Curated Git LFS model checkpoints |
| `annotations/` | Human-reviewed sparse ground truth packets |
| `skyscouter/perception/` | Detector backends |
| `skyscouter/tracking/` | Tracker logic |
| `skyscouter/output/` | Overlay, JSONL, reports |
| `scripts/run_pipeline.py` | Main replay command |
| `scripts/evaluate_hard_case.py` | Detector/tracker hard-case scorer |
| `scripts/prepare_drone_annotations.py` | Frame extraction + browser annotation UI |
| `scripts/train_airborne_yolo.py` | YOLO training wrapper |
| `tests/` | Regression tests |

---

## ✅ Run Tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests -v
```

Fast tracker-only check:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_single_target_tracker.py -v
```

---

## ⚠️ What Is Ignored

These stay out of normal Git:

| Path | Why |
|---|---|
| `data/videos/` | large local footage |
| `data/outputs/` | generated reports/videos |
| `data/training/` | rebuildable training datasets/runs |
| `Ultralytics/` | local settings/cache |
| random `*.pt` outside `data/models/` | avoid accidental checkpoint spam |

Curated `data/models/**/*.pt` files are the exception and are tracked by Git LFS.

---

## 🧭 Engineering Rule

No hardcoded boxes. No filename tricks. No fake detections. If the model misses, the report should say so plainly.

That is how this project stays useful.
