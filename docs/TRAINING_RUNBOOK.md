# SkyScouter Training Runbook

Last updated: 2026-05-01

This is the practical runbook for training and resuming SkyScouter YOLO models on Windows with an NVIDIA GPU.

## V3/V4 Gate First

For the next model family, do not start a full 80-epoch run from the older
single-manifest flow. The current product blocker is semantic confusion
(`drone` GT matched as `airplane`), so V3/V4 training is gated.

Read first:

```text
docs/DATASET_INTEGRATION_REPORT.md
docs/V3_EVALUATION_SUMMARY.md
docs/V3_TRAINING_RESULTS.md
docs/NEXT_DATA_ANNOTATION_PLAN.md
```

Stage 1 is diagnostic/pretraining only and must not be promoted. Stage 3 is the
first eligible promotion stage.

## Read GPU Utilization Correctly

Task Manager often shows the wrong graph for ML training. The default GPU page usually emphasizes 3D, video encode, and copy engines. PyTorch training uses CUDA compute, so Task Manager can look low even when the card is working.

Use this instead:

```powershell
nvidia-smi -l 5
```

Healthy signs:

- `python.exe` appears in the process table.
- GPU memory is several GB, not a few hundred MB.
- GPU utilization moves during batches.
- YOLO prints `CUDA:0` and the GPU name at startup.

If VRAM is only around 7 GB on a 16 GB card and utilization is low, the GPU is probably waiting for the dataloader. The usual causes are too few dataloader workers, slow storage, or too small a batch.

## Environment Setup

Use Python 3.10 to 3.12. On Windows:

```powershell
cd "C:\Users\Ali\Desktop\SKY"
py -3.12 -m venv .venv_train
.\.venv_train\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv_train\Scripts\python.exe -m pip install -r requirements.txt
```

Verify CUDA:

```powershell
.\.venv_train\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

Expected shape:

```text
True
NVIDIA GeForce RTX 5070 Ti
```

If `torch.cuda.is_available()` is `False`, do not train yet. Fix PyTorch/CUDA first.

## Build A Training Dataset

For V3/V4, prefer the staged builder over the older single-manifest builder.
Stage 2 intentionally treats AOD-4 as a capped confuser source, not the main
drone identity source:

```powershell
.\.venv_train\Scripts\python.exe scripts/build_staged_airborne_dataset.py `
  --stage stage2 `
  --link-mode copy `
  --cap aod4=6000 `
  --cap anti_uav_rgbt=5000 `
  --cap dut_anti_uav=5000 `
  --cap visiodect=12000
```

By default, Stage 2 excludes images containing AOD-4 `drone` boxes until the
AOD-4 drone-vs-airplane audit passes. Use `--no-default-exclusions` only after
that audit is recorded.

Historical v1/v2 builder:

```powershell
.\.venv_train\Scripts\python.exe scripts/prepare_airborne_training_set.py `
  --manifest configs/training/airborne_dataset_manifest.yaml `
  --out-dir data/training/airborne_yolo_v3 `
  --link-mode copy
```

The output must contain:

```text
data/training/airborne_yolo_v3/data.yaml
data/training/airborne_yolo_v3/images/train/
data/training/airborne_yolo_v3/labels/train/
```

## Train A New Model

Default training uses `workers: 8`, `batch: 8`, `imgsz: 1024`, AMP, and CUDA auto-detection from `configs/training/airborne_yolo11.yaml`.

```powershell
.\.venv_train\Scripts\python.exe scripts/train_airborne_yolo.py `
  --config configs/training/airborne_yolo11.yaml
```

For a stronger RTX 5070 Ti run, try higher batch after the basic run is stable:

```powershell
.\.venv_train\Scripts\python.exe scripts/train_airborne_yolo.py `
  --config configs/training/airborne_yolo11.yaml `
  --batch 12 `
  --workers 8
```

If VRAM is still below about 12 GB and training is stable, try `--batch 16`. If you hit CUDA out-of-memory, lower batch back to `12` or `8`.

## Resume The Current v2 Run

Open PowerShell as Administrator and keep it open:

```powershell
cd "C:\Users\Ali\Desktop\SKY"
powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\resume_yolo_v2_training.ps1" -Workers 8
```

Why Administrator? On this machine, a non-admin/Codex-launched process hit a Windows multiprocessing permission failure when dataloader workers were enabled. Admin PowerShell is the preferred path for full-speed resume.

If `workers=8` fails with a Windows dataloader/multiprocessing error, use the slow stability fallback:

```powershell
cd "C:\Users\Ali\Desktop\SKY"
powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\resume_yolo_v2_training.ps1" -Workers 0
```

`Workers 0` is safe but slower. It keeps all image loading and augmentation in the main process, so the GPU often waits between batches.

## Monitor A Run

In a second terminal:

```powershell
nvidia-smi -l 5
```

Check YOLO progress logs:

```powershell
Get-ChildItem "C:\Users\Ali\Desktop\SKY\data\training\runs" -Recurse -Filter "resume_training_*.log" |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1 |
  Get-Content -Tail 60 -Wait
```

## Performance Tuning Order

Tune in this order:

1. Use CUDA PyTorch, not CPU PyTorch.
2. Use `workers=8` from an Administrator PowerShell.
3. Increase batch from `8` to `12`, then `16` if VRAM allows.
4. Keep `imgsz=1024` until recall plateaus; increasing to `1280` is a separate experiment.
5. Consider `cache=ram` only if the dataset fits comfortably in system RAM.

Do not chase 100% GPU utilization blindly. Small-object detection at high image size can be input-pipeline limited. The goal is stable throughput and better validation recall, not a pretty Task Manager graph.

## Promote Finished Weights

After training finishes:

```powershell
Copy-Item "path\to\run\weights\best.pt" "data\models\yolo11s_airborne_aod4_antiuav300_v2\best.pt" -Force
Copy-Item "path\to\run\weights\last.pt" "data\models\yolo11s_airborne_aod4_antiuav300_v2\last.pt" -Force
git lfs status
git status --short
```

Only curated `data/models/**/*.pt` files should be committed. Raw `data/training/runs/` remains ignored.
