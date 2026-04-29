# ML Training and Handoff Notes

Last updated: 2026-04-28

## Mission Model

The real mission is not generic object detection. The perception stack must
detect, lock, and follow a hostile drone in the sky while rejecting birds,
aircraft, clouds, trees, terrain, buildings, and other clutter.

Detector responsibility:

- find airborne objects in sky-like views;
- classify drone vs bird vs airplane/helicopter when possible;
- provide calibrated candidate confidence.

Tracker responsibility:

- preserve the selected target identity over time;
- bridge short detector gaps;
- avoid switching to distractors.

Lock-state responsibility:

- make `guidance_valid=true` only when the visual target is stable and meets
  quality gates;
- revoke lock when the track is stale, low quality, or semantically unsafe.

## Current Honest Status

Implemented so far:

- `airborne_cv` proposal detector for high-recall dark sky candidates.
- `single_target_kalman_lk` tracker with Kalman bbox state, LK optical-flow
  bridge, appearance scoring, and local reacquisition.
- Sparse GT annotation packet and browser annotator.
- Frame-level diagnostics CSV.
- `eval_report.json` sparse-GT metrics.
- Correct handling of hard-negative GT rows with empty bboxes.
- CLI override: `scripts/run_pipeline.py --gt <csv>`.
- Training dataset manifest and YOLO dataset builder.

Latest corrected-GT replay:

- Run: `data/outputs/run_corrected_gt_eval`
- GT file: `/Users/aliallameh/Desktop/DATASETS/drone_sparse_gt_corrected (1).csv`
- GT frames: 89
- Positive drone frames: 44
- Negative/hard-negative frames: 45
- Matched positive rate: 1.0, but this only means a bbox was published.
- Median IoU: 0.0
- Median center error: 417 px
- False-lock negative frames: 9
- Acceptance gate: failed

Detector-only proposal audit on the 44 positive frames:

- Best proposal IoU >= 0.1 on about 72.7% of positive frames.
- Best proposal center error <= 50 px on about 79.5% of positive frames.

Interpretation: the current detector often produces a usable drone proposal,
but the tracker/lock stack frequently stays on clutter or chooses the wrong
proposal. A trained airborne detector is still required, but lock semantics
must also become stricter.

## Chosen Training Direction

First production training profile:

- model: YOLO11s, then compare YOLO11m after acceptance improves;
- classes: `drone`, `bird`, `airplane`, `helicopter`, `unknown_airborne`;
- image size: 1280 initially, because small airborne targets are the mission;
- deployment target: exportable Ultralytics YOLO detector backend.

Dataset priority:

1. AOD-4 or equivalent airborne object dataset for drone/bird/aircraft
   multi-class separation.
2. Drone-vs-Bird Challenge-style data for hard drone vs bird discrimination.
3. LRDDv2-style long-range drone data for tiny drone recall.
4. Local `my_drone_chase.MP4` corrected sparse GT as acceptance and limited
   domain adaptation, not as the only training source.

Do not prioritize VisDrone2019-DET for this mission. It is mostly
drone-camera-looking-down detection of ground objects, not skyborne drone
discrimination.

## New Training Files

- `configs/training/airborne_dataset_manifest.yaml`
  - declares the target taxonomy and dataset sources.
  - external sources are disabled placeholders until the datasets are present
    locally.
  - local sparse GT source points to the user-corrected CSV.

- `configs/training/airborne_yolo11.yaml`
  - YOLO11s training profile.
  - stores image size, epochs, augmentation, output location, and acceptance
    thresholds.

- `scripts/prepare_airborne_training_set.py`
  - builds a standard YOLO dataset:
    `images/{train,val,test}`, `labels/{train,val,test}`, `data.yaml`.
  - supports existing YOLO detection datasets with class remapping.
  - supports Skyscouter sparse-GT CSV plus video frame extraction.
  - writes `manifest_summary.json` with class counts, negatives, and splits.

- `scripts/train_airborne_yolo.py`
  - trains Ultralytics YOLO from the training profile.
  - intentionally fails if `data.yaml` has not been prepared.

## Commands

Build only the local sparse-GT sanity dataset:

```bash
.venv/bin/python scripts/prepare_airborne_training_set.py \
  --manifest configs/training/airborne_dataset_manifest.yaml \
  --out-dir data/training/local_sparse_gt_yolo \
  --source local_my_drone_chase_sparse_gt \
  --link-mode copy
```

Observed local sanity output:

- train: 68 images, 34 drone objects, 34 negatives
- val: 12 images, 4 drone objects, 8 negatives
- test: 9 images, 6 drone objects, 3 negatives

Build the real training set after external datasets are placed locally:

```bash
.venv/bin/python scripts/prepare_airborne_training_set.py \
  --manifest configs/training/airborne_dataset_manifest.yaml \
  --out-dir data/training/airborne_yolo_v1 \
  --link-mode symlink
```

Train:

```bash
.venv/bin/python scripts/train_airborne_yolo.py \
  --config configs/training/airborne_yolo11.yaml
```

Evaluate a replay with corrected GT:

```bash
.venv/bin/python scripts/run_pipeline.py \
  --video data/videos/my_drone_chase.MP4 \
  --config configs/default.yaml \
  --gt "/Users/aliallameh/Desktop/DATASETS/drone_sparse_gt_corrected (1).csv" \
  --output data/outputs/run_corrected_gt_eval
```

## Next Engineering Steps

1. Obtain and normalize the external airborne datasets into YOLO labels.
2. Enable those sources in `configs/training/airborne_dataset_manifest.yaml`.
3. Build `data/training/airborne_yolo_v1` and inspect
   `manifest_summary.json` before training.
4. Train YOLO11s.
5. Add a config profile that points `detector.backend: yolo_ultralytics` to the
   trained weights.
6. Run corrected-GT replay and compare against the acceptance gate.
7. Tighten lock semantics so `guidance_valid=true` requires semantic drone
   confidence, stable tracking, and no active false-lock risk.

## Acceptance Gate

Do not call the video acceptable until all are true:

- matched positive rate >= 90%;
- median center error <= 25 px;
- labeled track ID switches <= 3;
- false-lock negative frames = 0.

The current pipeline fails this gate.
