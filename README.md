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
| Jetson SIYI A8 Mini RTSP ingest | Working — 1920×1080 @ 30 fps |
| TensorRT export helper | Working — yolov26n imgsz=1024 FP32 on R36.5.0 |
| SIYI gimbal follow (yaw+pitch) | Working — live UDP commands confirmed |
| Real flight control | Not implemented |
| Safety claim | Not allowed yet |

The current deployable model is:

```text
data/models/yolov26_lrdd_v2_img1024/best.engine   (Jetson — built on R36.5.0/TRT 10.3.0)
data/models/yolov26_lrdd_v2_img1024/best.pt       (training weights, Git LFS)
```

YOLOv26n, single class `drone`, trained on LRDDv2 + Anti-UAV300 RGB (no birds).
Inference at imgsz=1024 FP32; ~53×32 px drone bbox in model input at 1080p range.

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

Then run the single setup+launcher script — it handles everything from scratch:

```bash
chmod +x jetson.sh
./jetson.sh
```

`jetson.sh` will:
1. Detect and recreate a stale `.venv_jetson` if it was built on a different machine
2. Create `.venv_jetson` with `--system-site-packages` (inherits JetPack CUDA/TRT/cv2)
3. Install torch from Jetson AI Lab (`jp6/cu126`) — the only correct sm_87 source
4. Install `nvidia-cudss-cu12` (libcudss.so.0, required by Jetson AI Lab torch)
5. Purge any conflicting pip CUDA packages that shadow JetPack's cuBLAS
6. Install all project requirements and the editable `skyscouter` package
7. Verify the environment (sm_87, cuBLAS, torch/cv2/ultralytics/tensorrt)
8. Check the TRT engine was built on this exact Jetson version
9. Drop into an interactive menu for all pipeline operations

Do not manually install nvidia-cublas-cu12 or other pip CUDA packages — they
shadow JetPack's system CUDA and cause `CUBLAS_STATUS_ALLOC_FAILED`.

Full Jetson instructions:

```text
docs/JETSON_ORIN_NANO_SETUP.md
```

## Verified Jetson Camera

Camera: **SIYI A8 Mini** gimbal — 1/2.8" Sony CMOS, 81° H-FOV, 4K capable.

```text
Fixed IP:  192.168.144.25
RTSP URL:  rtsp://192.168.144.25:8554/main.264  (transport: tcp)
Gimbal:    UDP port 37260  (yaw+pitch commands)
```

Pipeline stream mode: **1920×1080 @ 30 fps** (not 60 — decoder load at 60 fps
competes with the GPU budget the detector needs).

Network setup: the Jetson Ethernet interface (`eno1`) must have a **different**
IP in the same subnet, e.g. `192.168.144.10/24`.  Use `jetson.sh` option 9 or:

```bash
# Run these on the Jetson as yourself (nmcli handles sudo internally)
sudo nmcli connection modify "Wired connection 1" ipv4.addresses "192.168.144.10/24"
sudo nmcli connection modify "Wired connection 1" ipv4.gateway "192.168.144.1"
sudo nmcli connection modify "Wired connection 1" ipv4.method manual
sudo nmcli connection up "Wired connection 1"
ping 192.168.144.25   # expect ~0.3 ms RTT
```

Confirmed camera MAC: `7c:75:1a:ea:c2:e7`.

## Run The Full Jetson Live Pipeline

The easiest way is the `jetson.sh` interactive menu:

```bash
./jetson.sh
```

Menu options:

| Option | Mode |
|--------|------|
| 1 | TRT pipeline — log only, no display (safest first run) |
| 2 | TRT pipeline — MJPEG stream → `http://<jetson-ip>:8090` |
| 3 | TRT pipeline — OpenCV window on connected display |
| 4 | TRT pipeline — gimbal follow **disabled** (observe-only) |
| 7 | 30-second smoke test (gimbal + display off, Ctrl+C to stop) |

Or call directly with the venv Python:

```bash
# Log-only, no display
.venv_jetson/bin/python3 scripts/run_pipeline.py \
  --config configs/deploy_jetson_yolov26_lrdd_v2_siyi_a8_mini_1080p.yaml \
  --output data/outputs/my_run \
  --no-operator-view

# MJPEG operator feed (open http://<jetson-ip>:8090 in browser)
.venv_jetson/bin/python3 scripts/run_pipeline.py \
  --config configs/deploy_jetson_yolov26_lrdd_v2_siyi_a8_mini_1080p.yaml \
  --output data/outputs/my_run \
  --operator-view-mode mjpeg

# OpenCV window on display (use --operator-view-window-backend opencv,
# NOT gstreamer — gstreamer videoparse element is not installed)
.venv_jetson/bin/python3 scripts/run_pipeline.py \
  --config configs/deploy_jetson_yolov26_lrdd_v2_siyi_a8_mini_1080p.yaml \
  --output data/outputs/my_run \
  --operator-view-window-backend opencv
```

> **Note:** always set `LD_LIBRARY_PATH` to include the cudss lib dir before
> any direct Python call, or the torch import will fail with
> `libcudss.so.0: cannot open shared object file`. `jetson.sh` does this
> automatically. If calling manually:
>
> ```bash
> export LD_LIBRARY_PATH=".venv_jetson/lib/python3.10/site-packages/nvidia/cu12/lib:$LD_LIBRARY_PATH"
> ```

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

Use `jetson.sh` option 5 (recommended — it checks git-LFS, prompts for
imgsz/precision, and optionally updates the config automatically).

Or build manually with the venv Python:

```bash
# FP32 (current baseline — confirmed working on R36.5.0/TRT 10.3.0)
.venv_jetson/bin/python3 scripts/export_tensorrt.py \
  --weights data/models/yolov26_lrdd_v2_img1024/best.pt \
  --imgsz 1024 \
  --batch 1 \
  --device 0

# FP16 (faster, ~2× throughput — rebuild if moving from FP32 baseline)
.venv_jetson/bin/python3 scripts/export_tensorrt.py \
  --weights data/models/yolov26_lrdd_v2_img1024/best.pt \
  --imgsz 1024 \
  --batch 1 \
  --device 0 \
  --half
```

Expected local artifacts:

```text
data/models/yolov26_lrdd_v2_img1024/best.engine
data/models/yolov26_lrdd_v2_img1024/best.export_manifest.json
```

**TRT engines are device-locked.** An engine built on R36.4.x will run on
R36.5.0 but produce wrong/zero detections silently. Always rebuild the engine
after a JetPack upgrade. `jetson.sh` checks the manifest at startup and warns
if the versions differ.

Do not commit `.engine` files — they are in `.gitignore`.

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
| `jetson.sh` | **Jetson entry point** — setup + interactive launch menu |
| `scripts/run_pipeline.py` | Main pipeline command |
| `scripts/run_jetson_live_pipeline.py` | Full Jetson live runtime launcher (called by jetson.sh) |
| `scripts/export_tensorrt.py` | Jetson TensorRT export helper |
| `scripts/train_airborne_yolo.py` | YOLO training wrapper |
| `docs/` | Deployment, training, and evaluation notes |
| `tests/` | Regression tests |

## Next Steps

Immediate Jetson work:

1. Run `./jetson.sh` — first run will set up the full environment automatically.
2. Use menu option 7 (smoke test) to confirm camera stream, detections, and lock.
3. Review `data/outputs/smoke_*/diagnostics.csv` — check detection rate.
   If < 60% on a visible drone, lower `detector.confidence_threshold` to 0.15.
4. Use menu option 4 (gimbal disabled) for initial airborne observation.
5. Enable gimbal follow: verify yaw/pitch sign via `scripts/dev/siyi_gimbal_bench.py`,
   then run menu option 1 (full live with gimbal).
6. Consider rebuilding the engine with `--half` (FP16) for better throughput —
   use menu option 5.

Next product work:

1. Calibrate the SIYI A8 Mini intrinsics (current config: `is_calibrated: false`).
2. Collect log-only real-air footage with a known drone target.
3. Annotate 100–200 local frames and validate recall at operational ranges.
4. Expand to 300–1,000 local labelled frames before sandbox readiness claim.
5. Evaluate whether FP16 engine degrades recall vs FP32 at target range.
6. Keep false-lock negatives at zero.

## Experimental flight-control link (opt-in, off by default)

The repo contains a Phase-1 ArduPilot MAVLink link (`skyscouter/flight/mavlink_link.py`).
It is **disabled and dry-run by default** (`flight_control.enabled: false`,
`dry_run: true`) and sends nothing on the wire unless explicitly opted in.

Yaw tracking is commanded as an **absolute heading**: the controller produces a
yaw *offset* (PID correction in degrees), and the link adds it to the current FC
heading (read from `ATTITUDE`) to form an absolute `MAV_CMD_CONDITION_YAW` target
(`param4=0`), commanded along the shortest path. This replaces the earlier
relative-yaw command (`param4=1`, "turn another X deg"), which accumulated when
streamed at `send_hz` and produced cumulative turning / precession.

Tunables (in `flight_control:`):

- `yaw_abs_min_delta_deg` (default `1.0`) — skip yaw sends when the absolute
  heading change is below this threshold.
- `yaw_abs_keepalive_s` (default `0.5`) — send anyway once this long has elapsed
  since the last yaw command, so near-steady targets still get periodic refreshes.

Together these suppress jitter from tiny target changes while keeping the command
fresh.

## Safety Rules

- No autonomous chase.
- No flight-controller commands.
- No ESP32 command relay.
- No payload commands.
- No safety claims from review-only mode.
- Do not display numeric GSD unless a real range source exists.
- If the model misses or labels incorrectly, report it plainly.

SkyScouter is useful only if the logs tell the truth.
