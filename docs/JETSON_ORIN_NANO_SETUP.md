# Jetson Orin Nano Super Setup

This is the repeatable edge-deployment path for SkyScouter. The first Jetson
goal is log-only perception: load the curated v2 model, export TensorRT on the
Jetson, benchmark it, and run the USB camera pipeline without sending any
flight-controller commands.

## Rules

- Export TensorRT `.engine` files on the Jetson that will run them.
- Do not commit `.engine`, raw videos, `DATASETS/`, `data/outputs/`, or
  `data/training/runs/`.
- Use Git LFS for curated `.pt` weights only.
- Use v2 as the current deployable model until a later Stage 3/V4 model is
  promoted.
- Keep the first flight mode log-only. No MAVLink command output, no ESP32
  command relay, no actuator authority.

## Official References

- NVIDIA JetPack install/setup:
  https://docs.nvidia.com/jetson/jetpack/install-setup/index.html
- NVIDIA PyTorch for Jetson:
  https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html
- NVIDIA TensorRT docs:
  https://docs.nvidia.com/deeplearning/tensorrt/index.html
- Ultralytics docs:
  https://docs.ultralytics.com/

## 1. Confirm Jetson System

Run on the Jetson:

```bash
cat /etc/nv_tegra_release
dpkg-query -W nvidia-jetpack || true
python3 --version
which python3
```

Install the JetPack runtime/development packages for your JetPack release if
they are missing:

```bash
sudo apt update
sudo apt install -y nvidia-jetpack git git-lfs python3-venv python3-pip python3-opencv v4l-utils
```

For more deterministic performance during benchmarks:

```bash
sudo nvpmodel -m 0
sudo jetson_clocks
```

## 2. Clone SkyScouter

```bash
git lfs install
git clone https://github.com/Aliallameh/SKY.git
cd SKY
git lfs pull
```

Confirm the current deployable checkpoint exists:

```bash
ls -lh data/models/yolo11s_airborne_aod4_antiuav300_v2/best.pt
```

## 3. Create Jetson Python Environment

Use the JetPack-matched NVIDIA PyTorch wheel or container. Do not install the
Windows workstation CUDA 12.8 requirements file on Jetson as-is.

One practical venv pattern:

```bash
python3 -m venv .venv_jetson --system-site-packages
source .venv_jetson/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

Install PyTorch and torchvision from the NVIDIA Jetson instructions for your
JetPack version, then install the project runtime:

```bash
python -m pip install -r requirements-jetson.txt
```

Verify CUDA:

```bash
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
PY
```

If `pip install -r requirements-jetson.txt` tries to replace your NVIDIA
Jetson PyTorch wheel with a generic Linux wheel, stop and install Ultralytics
with no dependency resolution:

```bash
python -m pip install --no-deps "ultralytics>=8.4,<8.5"
python -m pip install -e .
```

## 4. Export TensorRT FP16 Engine

Build the engine on the Jetson:

```bash
python3 scripts/export_tensorrt.py \
  --weights data/models/yolo11s_airborne_aod4_antiuav300_v2/best.pt \
  --imgsz 1024 \
  --batch 1 \
  --device 0 \
  --half
```

Expected output:

```text
data/models/yolo11s_airborne_aod4_antiuav300_v2/best.engine
data/models/yolo11s_airborne_aod4_antiuav300_v2/best.export_manifest.json
```

Both are local deployment artifacts and are ignored by Git.

Start with FP16. INT8 should be a later experiment only after we have a
representative calibration set and compare semantic confusion against FP16.

## 5. Benchmark `.pt` Versus `.engine`

Replay benchmark:

```bash
python3 scripts/benchmark_detector_backend.py \
  --model data/models/yolo11s_airborne_aod4_antiuav300_v2/best.pt \
  --source data/videos/Video_1.mp4 \
  --imgsz 1024 \
  --conf 0.25 \
  --iou 0.45 \
  --device 0 \
  --frames 300 \
  --warmup 20

python3 scripts/benchmark_detector_backend.py \
  --model data/models/yolo11s_airborne_aod4_antiuav300_v2/best.engine \
  --source data/videos/Video_1.mp4 \
  --imgsz 1024 \
  --conf 0.25 \
  --iou 0.45 \
  --device 0 \
  --frames 300 \
  --warmup 20
```

USB camera benchmark:

```bash
python3 scripts/benchmark_detector_backend.py \
  --model data/models/yolo11s_airborne_aod4_antiuav300_v2/best.engine \
  --source 0 \
  --camera-width 1280 \
  --camera-height 720 \
  --camera-fps 30 \
  --fourcc MJPG \
  --imgsz 1024 \
  --conf 0.25 \
  --iou 0.45 \
  --device 0 \
  --frames 300 \
  --warmup 20
```

Benchmark JSON is written under `data/outputs/benchmarks/`.

## 6. Run Live USB Camera Pipeline

Use the TensorRT deployment config:

```bash
python3 scripts/run_pipeline.py \
  --config configs/deploy_jetson_yolo11s_v2_engine.yaml \
  --output data/outputs/jetson_live_v2_engine
```

The config uses:

```yaml
source:
  type: "opencv_camera"
  device: 0
  width: 1280
  height: 720
  fps: 30
  fourcc: "MJPG"
```

If the camera is not device `0`, inspect it:

```bash
v4l2-ctl --list-devices
v4l2-ctl --device=/dev/video0 --list-formats-ext
```

Then update `configs/deploy_jetson_yolo11s_v2_engine.yaml`.

## 7. What To Record Per Jetson Run

Keep these in the output manifest or notes:

- Jetson model and JetPack/L4T version.
- PyTorch, Ultralytics, TensorRT versions.
- Model path and model SHA/export manifest.
- `.pt` FPS/latency and `.engine` FPS/latency.
- Camera device, resolution, FPS, pixel format, exposure/focus settings.
- Lock-state distribution and semantic label distribution.
- Any dropped-frame estimate or camera read failures.

## Current Deployment Caveats

- Camera calibration is still required before transport/guidance claims.
- v2 still has known local semantic confusion where the drone can be labelled
  `airplane`.
- `airborne-review` style geometry is for review only and is not flight-safe.
- The Jetson live camera path is now available through OpenCV, but the first
  real-air milestone remains log-only collection and offline review.
