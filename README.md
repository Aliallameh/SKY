# Skyscouter — Phase 1 Onboard Perception

**Sprint 1 in progress — first trained airborne detector landed.** Feed any
video → get bbox + class + lock state per frame, in the schema your flight
controller will consume.

## What this does today

- Ingests a video (or image folder) → runs detection → tracks → drives the
  PRD §12 lock state machine → publishes versioned `target_state` messages
  (PRD §14) → writes an annotated MP4 + JSONL log.
- Every threshold, model path, and policy is in `configs/*.yaml`.
- Every component is swappable behind a base interface.
- Sparse GT evaluation is available via `--gt`, producing `eval_report.json`
  and frame-level `diagnostics.csv`.
- A trained airborne YOLO11s detector ships under
  `data/training/runs/detect/yolo11s_airborne_drone_vs_bird_v1/weights/best.pt`
  (gitignored, rebuild from the manifest below) and is wired through
  `configs/trained_yolo11s_eval.yaml`.

## Current honest result on `my_drone_chase.MP4`

Two detector configurations were measured against the corrected sparse GT
(89 frames: 44 positive drone, 45 hard-negative).

| Metric                     | Heuristic (`airborne_cv`) | Trained YOLO11s (epoch 34)¹ | Acceptance gate |
| -------------------------- | ------------------------- | --------------------------- | --------------- |
| Matched positive rate      | 1.00 (bbox produced)      | **0.705** (31/44)           | ≥ 0.90          |
| Median IoU when matched    | 0.0                       | **0.81**                    | —               |
| Median center error (px)   | 417                       | **1.57**                    | ≤ 25            |
| Labeled track ID switches  | n/a                       | **0**                       | ≤ 3             |
| False locks on negatives   | 9                         | **0**                       | = 0             |
| Acceptance gate passes     | ❌                        | ❌ (recall only)            | —               |

¹ Trained 35 of 80 epochs on AOD-4 (drone / bird / airplane / helicopter,
~6.2k objects per class) before being interrupted overnight. Validation
mAP50 = 0.97, mAP50-95 = 0.67. Full breakdown in
[`docs/PHASE_1_RESULTS.md`](docs/PHASE_1_RESULTS.md).

The remaining gap is **detector recall on long-range drone frames**, not
tracker quality — see the diagnostic in
[`docs/PHASE_1_RESULTS.md`](docs/PHASE_1_RESULTS.md#miss-frame-diagnostic).

## What it does *not* do today (and what's coming)

| Capability | Status | Sprint |
|---|---|---|
| Drone-fine-tuned detector | ✅ first cut shipped (`yolo11s_airborne_drone_vs_bird_v1`) | Sprint 1 |
| Drone vs bird semantic discriminator | ✅ trained from AOD-4 four-class data; per-class recall in eval report | Sprint 1 |
| Long-range/small-pixel drone recall | ⚙️ training direction set; needs LRDDv2 or similar dataset | Sprint 1 |
| Real ego-motion compensation | ⚙️ interface only (identity stub) | Sprint 3 |
| Confidence calibration | ❌ | Sprint 1 |
| MAVLink bridge to flight controller | ❌ — JSONL only today | Sprint 3 |
| Operator HMI | ❌ | Sprint 3 |
| TensorRT export | ❌ | Sprint 2 |
| Compute bake-off harness | ❌ | Sprint 2 |
| Live camera ingest | ❌ — interface ready, plug-in TBD | Sprint 1 |

## Quickstart

The repo is OS-agnostic. The walk-throughs below use Windows + RTX 5070 Ti
(Blackwell, sm_120) because that's the current development host; equivalent
Linux/Mac steps are in the comments.

```powershell
# 1. Clone
git clone https://github.com/Aliallameh/SKY.git skyscouter
cd skyscouter

# 2. Create a Python 3.12 venv (PyTorch wheels ship for 3.12; not yet for 3.14)
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip wheel setuptools

# 3. Install PyTorch with CUDA 12.8 (required for Blackwell sm_120 kernels.
#    On older NVIDIA arch — Ada/Ampere — stable cu124 wheels also work.)
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 4. Install Skyscouter + Ultralytics
.\.venv\Scripts\python.exe -m pip install -e ".[yolo]"

# 5. Verify CUDA is live
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Bench-replay the heuristic detector on any video (no training needed):

```powershell
.\.venv\Scripts\python.exe scripts/run_pipeline.py `
    --video data/videos/your_video.mp4 `
    --config configs/default.yaml `
    --output data/outputs/run_001
```

Outputs:
```
data/outputs/run_001/annotated.mp4        — visual verification
data/outputs/run_001/target_states.jsonl  — one TargetState per frame
data/outputs/run_001/diagnostics.csv      — per-frame metrics
data/outputs/run_001/run.log              — text log
data/outputs/run_001/manifest.json        — replay manifest
```

Run with corrected sparse GT for evaluation:

```powershell
.\.venv\Scripts\python.exe scripts/run_pipeline.py `
    --video data/videos/my_drone_chase.MP4 `
    --config configs/default.yaml `
    --gt path\to\drone_sparse_gt_corrected.csv `
    --output data/outputs/run_default_eval
```

Use the **trained YOLO11s detector** profile (after training, see below):

```powershell
.\.venv\Scripts\python.exe scripts/run_pipeline.py `
    --video data/videos/my_drone_chase.MP4 `
    --config configs/trained_yolo11s_eval.yaml `
    --gt path\to\drone_sparse_gt_corrected.csv `
    --output data/outputs/run_trained_detector_eval
```

## Train the airborne detector from scratch

```powershell
# 1. Edit paths in configs/training/airborne_dataset_manifest.yaml to point
#    at your local AOD-4 root and your sparse-GT CSV.

# 2. Build the YOLO dataset (≈18k train / ≈3.5k val / ≈1.2k test).
.\.venv\Scripts\python.exe scripts/prepare_airborne_training_set.py `
    --manifest configs/training/airborne_dataset_manifest.yaml `
    --out-dir data/training/airborne_yolo_v1 `
    --link-mode copy

# 3. Edit configs/training/airborne_yolo11.yaml if you need to change
#    epochs/batch/imgsz. Defaults: 80 epochs, batch 8, imgsz 1024, AMP on,
#    GPU auto-detect.

# 4. Train (CUDA auto-detected on supported NVIDIA cards).
.\.venv\Scripts\python.exe scripts/train_airborne_yolo.py `
    --config configs/training/airborne_yolo11.yaml

# 5. Weights land at:
#    data/training/runs/yolo11s_airborne_drone_vs_bird_v1/weights/best.pt
#    Update detector.weights in configs/trained_yolo11s_eval.yaml if needed.
```

**Tip:** for long runs, launch training inside a detached terminal
(`Start-Process powershell` on Windows, `nohup` on Linux). If the parent
shell dies, the training process dies with it — `last.pt` is still saved
each epoch, so the run is recoverable via Ultralytics' `resume=True`, but
you lose elapsed wall time.

## Architecture

```
Video / Camera         CUE_STATE  (Sprint 1)
      │                    │
      ▼                    ▼
 ┌──────────┐         ┌──────────┐
 │  Source  │         │   Cue    │
 │ (frames) │         │ Adapter  │
 └────┬─────┘         └────┬─────┘
      │                    │
      └────────┬───────────┘
               ▼
        ┌────────────┐
        │  Detector  │   YOLO-class, swappable
        └─────┬──────┘
              ▼
        ┌────────────┐
        │  Tracker   │   IoU + ego-motion stub
        └─────┬──────┘
              ▼
        ┌────────────┐
        │ Lock State │   PRD §12 lifecycle
        └─────┬──────┘
              ▼
        ┌────────────┐
        │ Publisher  │   target_state JSON (PRD §14)
        └─────┬──────┘
              ▼
       Annotated MP4 + JSONL + manifest
```

## Code map

| Path | Purpose |
|---|---|
| `configs/default.yaml` | Heuristic-detector bench-replay profile (no training required) |
| `configs/trained_yolo11s_eval.yaml` | Acceptance-eval profile pointing at the trained YOLO11s |
| `configs/training/` | Airborne YOLO dataset manifest and training profile |
| `skyscouter/schemas.py` | `CueState`, `TargetState`, all enums (PRD §13–14) |
| `skyscouter/io/` | Frame sources (video, image folder, future live camera) |
| `skyscouter/perception/` | Detector base + Ultralytics YOLO + heuristic + stub backends |
| `skyscouter/tracking/` | Tracker base + Kalman/LK tracker + ego-motion seam |
| `skyscouter/lock/` | PRD §12 state machine + lock-quality scorer |
| `skyscouter/output/` | JSONL writer, video annotator, run logger |
| `skyscouter/pipeline.py` | The orchestrator |
| `scripts/run_pipeline.py` | CLI entry point |
| `scripts/prepare_drone_annotations.py` | Sparse frame export + bbox review HTML |
| `scripts/prepare_airborne_training_set.py` | Build unified YOLO train/val/test dataset |
| `scripts/train_airborne_yolo.py` | Ultralytics YOLO training entry point |
| `scripts/extract_miss_highlights.py` | Diagnostic: render detector-vs-GT for missed frames |
| `tests/` | Unit tests for the state machine and other deterministic parts |

## Layout that does **not** ship in git

The following are gitignored (large or machine-specific). Rebuild locally:

| Path | How to rebuild |
|---|---|
| `data/videos/*.MP4` | Place your test footage here |
| `data/training/airborne_yolo_v1/` | `scripts/prepare_airborne_training_set.py` |
| `data/training/runs/` | `scripts/train_airborne_yolo.py` |
| `data/outputs/` | `scripts/run_pipeline.py` |
| `*.pt` | Auto-downloaded by Ultralytics or produced by training |

## How to run tests

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest tests/ -v
```

The state machine has a thorough unit test suite — it's the safety-critical
part of the system. Tests use `StubDetector` so they run without any model
weights.

## Where to plug in your work

- **Better detector**: implement `BaseDetector` in
  `skyscouter/perception/`, register in `factory.py`, point a config at it.
  Current trained model is YOLO11s with classes
  `drone, bird, airplane, helicopter, unknown_airborne`.
- **Better tracker** (e.g. real ByteTrack): implement `BaseTracker` and
  register similarly.
- **Live camera**: subclass `BaseFrameSource`. Pipeline does not change.
- **Real IMU ego-motion**: subclass `EgoMotionCompensator` with a
  `GyroDrivenEgoMotion` and wire it into the tracker factory.
- **Cue ingest**: build the `CueAdapter` per PRD §13 and pass it to the
  pipeline (Sprint 1).

## What "no cheating" means

This codebase has a few rules — they exist because the alternative is a demo
that works on the screenshot you gave it and breaks at Suffield.

- No hardcoded bounding boxes, ever
- No "if filename contains 'demo' then ..." shortcuts
- No silent fallback to fake detections if the model is missing — fail loudly
- Every threshold is in YAML, not in code
- Every output is traceable to a config + run_id + manifest
- COCO `airplane` is **not** silently relabeled as `drone`. The trained
  YOLO11s now has explicit `drone` / `bird` / `airplane` / `helicopter`
  classes; only `drone` (plus `uas` and `airborne_candidate`) can promote
  to LOCKED in the lock state machine.

## Active ML handoff

Read [`docs/ML_TRAINING_AND_HANDOFF.md`](docs/ML_TRAINING_AND_HANDOFF.md)
before continuing the ML work. It has the dataset choice rationale, the
current measured failure mode, exact commands, and the remaining Sprint 1
plan. The latest acceptance-gate run is summarized in
[`docs/PHASE_1_RESULTS.md`](docs/PHASE_1_RESULTS.md).

## License

Internal. Do not redistribute.
