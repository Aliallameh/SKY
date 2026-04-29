# Skyscouter — Phase 1 Onboard Perception

**Sprint 0 — Bench replay foundation.** Feed any video → get bbox + lock state per frame, in the schema your flight controller will consume.

## What this does today

- Ingests a video (or image folder) → runs detection → tracks → drives the
  PRD §12 lock state machine → publishes versioned `target_state` messages
  (PRD §14) → writes an annotated MP4 + JSONL log.
- Every threshold, model path, and policy is in `configs/default.yaml`.
- Every component is swappable behind a base interface.
- Sparse GT evaluation is available via `--gt`, producing `eval_report.json`
  and frame-level `diagnostics.csv`.
- Airborne detector training scaffolding is available in `configs/training/`
  and `scripts/prepare_airborne_training_set.py`.

## Current honest result on `my_drone_chase.MP4`

The current heuristic/proposal pipeline is **not acceptance-ready**. With the
corrected sparse GT file from 2026-04-28:

- GT frames: 89
- Positive drone frames: 44
- Hard-negative frames: 45
- Median IoU: 0.0
- Median center error: about 417 px
- False-lock negative frames: 9
- Acceptance gate: failed

This is useful progress because the failure is now measurable. The current
detector often produces a usable drone proposal, but the tracker/lock stack
can stay on clutter. Next work is a trained airborne drone-vs-bird detector
plus stricter lock gates.

## What it does *not* do today (and what's coming)

| Capability | Status | Sprint |
|---|---|---|
| Drone-fine-tuned detector | ⚙️ training scaffolding ready; weights not trained | Sprint 1 |
| Drone vs bird semantic/kinematic discriminator | ⚙️ dataset plan ready; model not trained | Sprint 1 |
| Real ego-motion compensation | ⚙️ interface only (identity stub) | Sprint 3 |
| Confidence calibration | ❌ | Sprint 1 |
| MAVLink bridge to flight controller | ❌ — JSONL only today | Sprint 3 |
| Operator HMI | ❌ | Sprint 3 |
| TensorRT export | ❌ | Sprint 2 |
| Compute bake-off harness | ❌ | Sprint 2 |
| Live camera ingest | ❌ — interface ready, plug-in TBD | Sprint 1 |

The architecture for every "❌" item is in place. The work is real ML,
data, and integration — not refactoring.

## Quickstart

```bash
# Clone and install
git clone <repo>
cd skyscouter
pip install -e ".[yolo]"

# Run on any video. The first run will download yolov8n.pt from Ultralytics.
python scripts/run_pipeline.py \
    --video data/videos/my_drone_chase.mp4 \
    --config configs/default.yaml \
    --output data/outputs/run_001

# Outputs:
#   data/outputs/run_001/annotated.mp4        — visual verification
#   data/outputs/run_001/target_states.jsonl  — one TargetState per frame
#   data/outputs/run_001/run.log              — text log
#   data/outputs/run_001/manifest.json        — replay manifest
```

Run with corrected sparse GT:

```bash
python scripts/run_pipeline.py \
    --video data/videos/my_drone_chase.MP4 \
    --config configs/default.yaml \
    --gt "/Users/aliallameh/Desktop/DATASETS/drone_sparse_gt_corrected (1).csv" \
    --output data/outputs/run_corrected_gt_eval
```

Prepare a YOLO training dataset after placing external airborne datasets:

```bash
python scripts/prepare_airborne_training_set.py \
    --manifest configs/training/airborne_dataset_manifest.yaml \
    --out-dir data/training/airborne_yolo_v1 \
    --link-mode symlink
```

Train the first YOLO11s airborne detector:

```bash
python scripts/train_airborne_yolo.py \
    --config configs/training/airborne_yolo11.yaml
```

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
| `configs/default.yaml` | All thresholds, model paths, policies |
| `configs/training/` | Airborne YOLO dataset manifest and training profile |
| `skyscouter/schemas.py` | `CueState`, `TargetState`, all enums (PRD §13–14) |
| `skyscouter/io/` | Frame sources (video, image folder, future live camera) |
| `skyscouter/perception/` | Detector base + Ultralytics YOLO + stub backends |
| `skyscouter/tracking/` | Tracker base + simple IoU tracker + ego-motion seam |
| `skyscouter/lock/` | PRD §12 state machine + lock-quality scorer |
| `skyscouter/output/` | JSONL writer, video annotator, run logger |
| `skyscouter/pipeline.py` | The orchestrator |
| `scripts/run_pipeline.py` | CLI entry point |
| `scripts/prepare_drone_annotations.py` | Sparse frame export + bbox review HTML |
| `scripts/prepare_airborne_training_set.py` | Build unified YOLO train/val/test dataset |
| `scripts/train_airborne_yolo.py` | Ultralytics YOLO training entry point |
| `tests/` | Unit tests for the state machine and other deterministic parts |

## How to run tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

The state machine has a thorough unit test suite — it's the safety-critical
part of the system. Tests use `StubDetector` so they run without any model
weights.

## Where to plug in your work

- **Better detector**: implement `BaseDetector` in
  `skyscouter/perception/`, register in `factory.py`, point config at it.
  Current training direction is YOLO11s with classes
  `drone,bird,airplane,helicopter,unknown_airborne`.
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
- COCO `airplane` is **not** silently relabeled as `drone`. Until a
  fine-tuned detector ships, generic/proposal detections are treated as
  candidates and must pass tracker and lock gates honestly.

## Active ML handoff

Read `docs/ML_TRAINING_AND_HANDOFF.md` before continuing the ML work. It has
the dataset choice rationale, current measured failure, exact commands, and
the remaining Sprint 1 plan.

## License

Internal. Do not redistribute.
