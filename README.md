# SkyScouter

SkyScouter is an onboard computer-vision pipeline for detecting, tracking, and
reviewing small airborne targets. The current product goal is clear:

```text
detect -> track -> lock review -> log evidence
```

The system currently runs in **log-only/advisory mode**. It does not send
MAVLink commands, ESP32 commands, payload commands, or actuator commands.

## Current Status

| Area | Status |
|---|---|
| Video replay pipeline | Ready |
| YOLO detector backend | Ready |
| Tracker and lock-state machine | Ready |
| Human-readable overlays | Ready |
| Target-state JSONL output | Ready |
| Guidance-hint JSONL output | Ready, advisory only |
| Jetson USB camera ingest | Implemented on live-camera branch |
| TensorRT export helper | Implemented on Jetson deployment branch |
| Real flight control | Not implemented |
| Safety claim | Not allowed yet |

The current deployable model is:

```text
data/models/yolo11s_airborne_aod4_antiuav300_v2/best.pt
```

Important caveat: v2 can geometrically detect the target but sometimes labels a
real drone as `airplane`. That means semantic-safe lock can be blocked. Do not
hide this. It is the main model-improvement target for V3/V4.

## Branches

There are three important branches right now:

| Branch | Purpose |
|---|---|
| `main` | Stable project history and current committed baseline |
| `codex/jetson-deployment-kit` | Jetson setup, TensorRT export, benchmark tooling |
| `feature/jetson-live-camera-runtime` | Full Jetson USB camera runtime, built on top of the deployment kit |

For Jetson live-camera work, use:

```bash
git checkout feature/jetson-live-camera-runtime
```

That branch includes the Jetson deployment kit plus the real camera runtime.

## What Gets Committed

Committed:

```text
source code
configs
docs
curated model weights under data/models/ through Git LFS
small annotation CSVs that are needed for reproducibility
```

Not committed:

```text
DATASETS/
data/videos/
data/outputs/
data/training/runs/
TensorRT .engine files
raw review frame dumps
random checkpoint files outside data/models/
```

TensorRT `.engine` files are built on the Jetson and stay local because they
are tied to JetPack, TensorRT, CUDA, and device details.

## Fresh Clone

Install:

- Python 3.10 to 3.12 on Windows workstation
- Git
- Git LFS
- NVIDIA CUDA/PyTorch stack for training or inference

Clone with model weights:

```powershell
git lfs install
git clone https://github.com/Aliallameh/SKY.git
cd SKY
git lfs pull
```

Confirm the important weights exist:

```text
data/models/yolo11s_airborne_aod4_antiuav300_v2/best.pt
data/models/yolo11s_airborne_aod4_antiuav300_v2/last.pt
```

## Windows Workstation Setup

Create the normal environment:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

For long training runs, use the dedicated training environment:

```powershell
py -3.12 -m venv .venv_train
.\.venv_train\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv_train\Scripts\python.exe -m pip install -r requirements.txt
.\.venv_train\Scripts\python.exe -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

Expected result:

```text
True
your NVIDIA GPU name
```

If Ultralytics tries to use a blocked Windows settings folder:

```powershell
$env:YOLO_CONFIG_DIR = "$PWD"
```

## Run A Video Replay

Place a video under:

```text
data/videos/
```

Run:

```powershell
.\.venv\Scripts\python.exe scripts/run_pipeline.py `
  --video data/videos/camera_20260423_113401.mp4 `
  --config configs/trained_yolo11s_v2_guidance_full.yaml `
  --output data/outputs/replay_camera_113401
```

Outputs:

| File | Meaning |
|---|---|
| `annotated.mp4` | Human review video |
| `target_states.jsonl` | One target-state row per frame |
| `guidance_hints.jsonl` | Bearing/elevation/yaw proposal log |
| `diagnostics.csv` | Per-frame debug values |
| `manifest.json` | Reproducibility record |

## Jetson Orin Nano Super Setup

Use the live-camera branch:

```bash
git fetch
git checkout feature/jetson-live-camera-runtime
git lfs pull
```

Use JetPack-matched NVIDIA PyTorch. Do not blindly install the Windows
CUDA 12.8 requirements file on Jetson.

Recommended Jetson environment shape:

```bash
python3 -m venv .venv_jetson --system-site-packages
source .venv_jetson/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-jetson.txt
```

Full Jetson instructions:

```text
docs/JETSON_ORIN_NANO_SETUP.md
```

## Verified Jetson Camera

Camera:

```text
4K ZOOM CAMERA
/dev/video0
OpenCV device index 0
```

Use this realtime mode first:

```text
backend: V4L2
fourcc: MJPG
width: 1280
height: 720
fps: 30
```

Use `MJPG`, not `YUYV`. YUYV is too slow at useful resolutions on this camera.

Verified direct OpenCV result:

```text
1280x720 MJPG 30 fps
300 frames
10.68 seconds
about 28 fps observed
```

## Run The Full Jetson Live Pipeline

Before TensorRT export exists, run the full PyTorch live pipeline:

```bash
python3 scripts/run_jetson_live_pipeline.py --backend pytorch
```

After exporting the TensorRT engine, run:

```bash
python3 scripts/run_jetson_live_pipeline.py --backend tensorrt
```

The live runtime writes:

| File | Meaning |
|---|---|
| `raw_camera.mp4` | Unannotated camera recording |
| `annotated.mp4` | Review video with overlays |
| `target_states.jsonl` | Track/lock state per frame |
| `guidance_hints.jsonl` | Advisory guidance output |
| `diagnostics.csv` | Per-frame debug data |
| `manifest.json` | Config, source metadata, artifacts, status |

Press `Ctrl+C` to stop. The pipeline will close writers and finalize
`manifest.json` with status `interrupted`.

Optional diagnostics:

```bash
python3 scripts/dev/jetson_preflight_check.py

python3 scripts/dev/test_live_camera_source.py \
  --device-index 0 \
  --width 1280 \
  --height 720 \
  --fps 30 \
  --fourcc MJPG \
  --backend v4l2 \
  --max-frames 300
```

## Export TensorRT On Jetson

Build the `.engine` on the Jetson:

```bash
python3 scripts/export_tensorrt.py \
  --weights data/models/yolo11s_airborne_aod4_antiuav300_v2/best.pt \
  --imgsz 1024 \
  --batch 1 \
  --device 0 \
  --half
```

Expected local artifacts:

```text
data/models/yolo11s_airborne_aod4_antiuav300_v2/best.engine
data/models/yolo11s_airborne_aod4_antiuav300_v2/best.export_manifest.json
```

Do not commit those files.

Benchmark:

```bash
python3 scripts/benchmark_detector_backend.py \
  --model data/models/yolo11s_airborne_aod4_antiuav300_v2/best.engine \
  --source 0 \
  --camera-width 1280 \
  --camera-height 720 \
  --camera-fps 30 \
  --fourcc MJPG \
  --imgsz 1024 \
  --frames 300 \
  --warmup 20
```

## Model Training Direction

Current issue:

```text
The detector often sees the airborne target, but the semantic label can be wrong.
The most harmful error is drone -> airplane.
```

Training plan:

1. Keep v2 as the current deployable baseline.
2. Do not promote Stage 1. Stage 1 is diagnostic/pretraining only.
3. Use staged V3/V4 training to reduce drone-to-airplane confusion.
4. Use local Mavic-style labelled frames before claiming sandbox readiness.
5. Evaluate semantic-safe lock separately from review-only airborne geometry.

Important docs:

```text
docs/DATASET_INTEGRATION_REPORT.md
docs/V3_EVALUATION_SUMMARY.md
docs/V3_TRAINING_RESULTS.md
docs/NEXT_DATA_ANNOTATION_PLAN.md
docs/TRAINING_RUNBOOK.md
```

Training command shape:

```powershell
.\.venv_train\Scripts\python.exe scripts/train_airborne_yolo.py `
  --config configs/training/airborne_yolo11_stage1_drone_only.yaml `
  --batch 16 `
  --workers 4
```

Training outputs stay under:

```text
data/training/runs/
```

Only promote curated checkpoints into:

```text
data/models/
```

## Annotation Workflow

Create a browser review packet:

```powershell
.\.venv\Scripts\python.exe scripts/prepare_drone_annotations.py `
  --video data/videos/camera_20260423_113401.mp4 `
  --output annotations/my_review_packet `
  --start-frame 80 `
  --end-frame 140
```

Open:

```text
annotations/my_review_packet/bbox_annotator.html
```

Labels:

| Label | Meaning |
|---|---|
| `drone` | Real target box |
| `negative` | Reviewed frame where the drone is absent |

Use empty box coordinates for negatives.

## Repo Map

| Path | Purpose |
|---|---|
| `configs/` | Runtime profiles and training configs |
| `data/models/` | Curated Git LFS model checkpoints |
| `annotations/` | Human-reviewed sparse ground truth packets |
| `skyscouter/io/` | Video, image, and live camera frame sources |
| `skyscouter/perception/` | Detector backends |
| `skyscouter/tracking/` | Tracker logic |
| `skyscouter/output/` | Overlay video, raw video, JSONL, reports |
| `scripts/run_pipeline.py` | Main pipeline command |
| `scripts/run_jetson_live_pipeline.py` | Full Jetson live runtime launcher |
| `scripts/export_tensorrt.py` | Jetson TensorRT export helper |
| `scripts/train_airborne_yolo.py` | YOLO training wrapper |
| `docs/` | Deployment, training, and evaluation notes |
| `tests/` | Regression tests |

## Next Steps

Immediate Jetson work:

1. Pull `feature/jetson-live-camera-runtime` on the Jetson.
2. Confirm `git lfs pull` downloads v2 weights.
3. Run the full PyTorch live pipeline.
4. Export TensorRT on the Jetson.
5. Run the full TensorRT live pipeline.
6. Review `raw_camera.mp4`, `annotated.mp4`, `target_states.jsonl`,
   `guidance_hints.jsonl`, and `manifest.json`.

Next product work:

1. Calibrate the real USB camera.
2. Collect log-only real-air footage.
3. Annotate 100 to 200 local Mavic-style frames before Stage 3.
4. Expand to 300 to 1,000 local labelled frames before sandbox readiness.
5. Train/evaluate V3/V4 against v2, especially drone-to-airplane confusion.
6. Keep false-lock negatives at zero.

## Safety Rules

- No autonomous chase.
- No flight-controller commands.
- No ESP32 command relay.
- No payload commands.
- No safety claims from review-only mode.
- Do not display numeric GSD unless a real range source exists.
- If the model misses or labels incorrectly, report it plainly.

SkyScouter is useful only if the logs tell the truth.
