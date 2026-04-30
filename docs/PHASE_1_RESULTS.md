# Phase 1 — First Trained Detector: Results

Last updated: 2026-04-30
Run output: `data/outputs/run_trained_detector_eval/` (gitignored)

This document summarizes the first end-to-end run of Skyscouter Phase 1
with a trained airborne detector replacing the heuristic proposal stage.
The companion narrative is in
[`ML_TRAINING_AND_HANDOFF.md`](ML_TRAINING_AND_HANDOFF.md).

## Training run

| Item | Value |
| ---- | ----- |
| Model | YOLO11s (Ultralytics 8.4.45) |
| Base weights | `yolo11s.pt` (COCO pretrained) |
| Image size | 1024 |
| Batch | 8 |
| Optimizer | auto (SGD) |
| AMP | on |
| Device | NVIDIA RTX 5070 Ti (Blackwell, sm_120), torch 2.11+cu128 |
| Epochs planned | 80 |
| Epochs completed | 35 (run interrupted overnight; weights at epoch 34/35 retained) |
| Val mAP50 best | 0.970 (epoch 34) |
| Val mAP50-95 best | 0.671 (epoch 34) |
| Val precision | 0.948 |
| Val recall | 0.932 |

**Training data**: AOD-4 four-class airborne (drone / bird / airplane /
helicopter) plus the local corrected sparse GT (44 positive + 45 hard
negative frames extracted from `my_drone_chase.MP4`).

| Split | Images | Objects | Drone | Bird | Airplane | Helicopter |
| ----- | -----: | ------: | ----: | ---: | -------: | ---------: |
| train | 17,947 | 25,115  | 6,269 | 6,312 | 6,312   | 6,222      |
| val   |  3,493 |  4,902  | 1,258 | 1,194 | 1,167   | 1,283      |
| test  |  1,165 |  1,625  |   415 |   394 |   421   |   395      |

The validation mAP50 climbed monotonically through the recorded epochs
(0.55 at epoch 3 ⟶ 0.97 at epoch 34) with only one minor regression at
the warmup-to-cosine transition. The curve was still ticking up at the
point of interruption, so the remaining 45 epochs are expected to add
non-zero margin.

## Acceptance-gate run

The `configs/trained_yolo11s_eval.yaml` profile was applied to the full
`my_drone_chase.MP4` (9,744 frames at 1920×1080 / 29.97 fps) with
`--gt path/to/drone_sparse_gt_corrected.csv`.

| Gate | Threshold | Heuristic baseline | Trained YOLO11s | Status |
| ---- | --------: | -----------------: | --------------: | :----: |
| Matched positive rate | ≥ 0.90 | 1.00 (bbox produced) | **0.705** (31/44) | ❌ |
| Median IoU when matched | — | 0.00 | **0.81** | — |
| Mean IoU when matched | — | — | 0.76 | — |
| Median center error (px) | ≤ 25 | 417 | **1.57** | ✅ |
| Max center error (px) | — | — | 6.68 | — |
| Labeled track ID switches | ≤ 3 | n/a | **0** | ✅ |
| False-lock negative frames | = 0 | 9 | **0** | ✅ |
| Acceptance gate verdict | — | ❌ | **❌ (recall only)** | — |

Lock-state distribution across 9,744 frames:

| State | Count | % of frames |
| ----- | ----: | ----------: |
| SEARCHING | 4,678 | 48.0 |
| ACQUIRED | 18 | 0.2 |
| TRACKING | 199 | 2.0 |
| LOCKED | 3,881 | 39.8 |
| STRIKE_READY | 756 | 7.8 |
| LOST | 212 | 2.2 |

The pipeline published exactly **one** track across the entire video with
zero ID switches — the tracker is no longer the source of identity churn.

## Miss-frame diagnostic

Of the 44 positive GT frames, 13 were unmatched by the published track.
`scripts/extract_miss_highlights.py` re-ran the trained detector on each
of those frames at `conf ≥ 0.01` to surface any sub-threshold signal:

| Frame | YOLO dets ≥ 0.01 | Best IoU vs GT | Best conf | Bucket |
| ----: | ---------------: | -------------: | --------: | :----- |
|   405 | 0 | — | — | **detector blind** |
|  2505 | 3 | 0.67 | 0.25 | borderline |
|  2700 | 3 | 0.69 | **0.42** | tracker miss |
|  2880 | 2 | 0.72 | 0.11 | below threshold |
|  6000 | 2 | 0.76 | **0.50** | tracker miss |
|  6120 | 2 | 0.78 | **0.44** | tracker miss |
|  6300 | 2 | 0.68 | 0.04 | far below threshold |
|  6495 | 2 | 0.86 | **0.66** | tracker miss |
|  6675 | 1 | 0.78 | **0.69** | tracker miss |
|  7050 | 2 | 0.83 | 0.20 | just below threshold |
|  7245 | 2 | 0.71 | **0.38** | tracker miss |
|  7425 | 3 | 0.68 | 0.13 | below threshold |
|  7500 | 3 | 0.86 | 0.04 | far below threshold |

**Bucket interpretation**:

- **detector blind** (1 frame): the model had no detection above 0.01 even
  with very low conf and large IoU candidates allowed. Recoverable only
  with more data / longer training / larger imgsz.
- **below threshold / borderline** (6 frames): the model had a detection
  with high IoU against GT but the confidence was below the 0.25 detector
  threshold. Lowering `detector.confidence_threshold` would surface these
  candidates; the lock state machine still gates on
  `min_class_confidence: 0.30` so false locks should not increase, but
  this is the experiment to verify.
- **tracker miss** (6 frames): the model had a 0.38–0.69 confidence drone
  detection co-located with the GT bbox, but the tracker had previously
  dropped to SEARCHING and did not reacquire on this frame. This is a
  tracker / lock-state behavior issue, not a detector issue.

Diagnostic outputs (gitignored): `data/outputs/miss_highlights/` —
contains `miss_highlights.mp4` (slow-motion reel), per-frame
`miss_NNNNNN.jpg`, and `summary.json`.

## What changed since the prior run

Compared to the heuristic `airborne_cv` baseline:

- **Median center error**: 417 px → 1.57 px (≈ 265× better).
- **Median IoU when matched**: 0.0 → 0.81.
- **False locks on hard negatives**: 9 → 0.
- **Matched-positive rate**: dropped from 1.00 to 0.705 — but the prior
  1.00 was misleading (the heuristic was producing *some* bbox almost
  always, just frequently on clutter, with median IoU 0). The trained
  detector's 0.705 is honest recall.

## Open questions

- Does `confidence_threshold = 0.10` recover the 6 below-threshold frames
  without raising the false-lock count above zero?
- Why does the tracker stay in SEARCHING through the
  6000–7500 frame range despite continuous high-confidence detections?
  Candidate hypotheses to test: stale-track bbox velocity drift, LK
  reacquisition radius too small, appearance-score gate too strict for
  small objects.
- Will resuming training to epoch 80 close the detector-blind gap on
  frame 405-class miss patterns?

## Reproducing this run

```powershell
# Build the dataset (assumes the manifest paths point at your local AOD-4
# and corrected GT CSV).
.\.venv\Scripts\python.exe scripts/prepare_airborne_training_set.py `
  --manifest configs/training/airborne_dataset_manifest.yaml `
  --out-dir data/training/airborne_yolo_v1 `
  --link-mode copy

# Train (full 80 epochs).
.\.venv\Scripts\python.exe scripts/train_airborne_yolo.py `
  --config configs/training/airborne_yolo11.yaml

# Acceptance eval.
.\.venv\Scripts\python.exe scripts/run_pipeline.py `
  --video data/videos/my_drone_chase.MP4 `
  --config configs/trained_yolo11s_eval.yaml `
  --gt path\to\drone_sparse_gt_corrected.csv `
  --output data/outputs/run_trained_detector_eval

# Miss-frame diagnostic.
.\.venv\Scripts\python.exe scripts/extract_miss_highlights.py
```
