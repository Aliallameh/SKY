# ML Training and Handoff Notes

Last updated: 2026-04-30

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

**Implemented to date**:

- Heuristic `airborne_cv` proposal detector for high-recall dark-sky candidates.
- Trained `yolo11s_airborne_drone_vs_bird_v1` detector (drone / bird /
  airplane / helicopter, val mAP50 0.97 at 35 epochs).
- `single_target_kalman_lk` tracker with Kalman bbox state, LK optical-flow
  bridge, appearance scoring, and local reacquisition.
- Sparse GT annotation packet and browser annotator.
- Frame-level diagnostics CSV.
- `eval_report.json` sparse-GT metrics.
- Correct handling of hard-negative GT rows with empty bboxes.
- CLI override: `scripts/run_pipeline.py --gt <csv>`.
- Training dataset manifest and YOLO dataset builder.
- `scripts/train_airborne_yolo.py` with CUDA / MPS / CPU auto-detect, AMP,
  and absolute-path resolution of the `project` arg (avoids the Ultralytics
  8.4 nested-output quirk).
- Trained-detector eval profile in `configs/trained_yolo11s_eval.yaml`.
- Diagnostic: `scripts/extract_miss_highlights.py` renders a slow-motion
  reel + JPEGs of every positive frame the trained detector missed, with
  GT in green and YOLO low-confidence detections in yellow.

**Latest corrected-GT replay (trained detector)**:

- Run: `data/outputs/run_trained_detector_eval/`
- GT file: corrected sparse GT (44 positive drone frames + 45 hard negatives)
- Frames processed: 9,744
- Matched positive rate: **0.705** (31/44)
- Median IoU when matched: **0.81** (was 0.0)
- Median center error: **1.57 px** (was 417 px)
- False-lock negative frames: **0** (was 9)
- Labeled track ID switches: **0**
- Acceptance gate: ❌ (only `min_matched_positive_rate ≥ 0.90` is failing)

For the per-frame breakdown see [`PHASE_1_RESULTS.md`](PHASE_1_RESULTS.md).

**Detector vs tracker root-cause split** (from
`scripts/extract_miss_highlights.py`):

- Of 13 missed positive frames, only **1** (frame 405) is true detector
  blindness. **6** had usable YOLO signal below the 0.25 confidence
  threshold; **6** had YOLO signal above threshold (with IoU 0.7–0.9
  versus GT) that the tracker did not propagate because it was in
  `SEARCHING` after a prior gap.

Interpretation: the trained detector is no longer the sole bottleneck.
Tracker reacquisition behavior and confidence-threshold policy are now
first-order concerns for the matched-positive-rate gate.

## Chosen Training Direction

First production training profile (shipped):

- model: YOLO11s
- classes: `drone, bird, airplane, helicopter, unknown_airborne`
- image size: 1024 (will revisit at 1280 if small-target recall plateaus)
- deployment target: exportable Ultralytics YOLO detector backend
- training device: GPU (CUDA / MPS) auto-detected, with AMP
- run output: `data/training/runs/<run_name>/weights/{best,last}.pt`

Dataset priority:

1. AOD-4 — airborne object dataset for drone/bird/aircraft separation. **In use.**
2. Drone-vs-Bird Challenge-style data for hard drone vs bird discrimination.
   *Not yet obtained; placeholder in manifest.*
3. LRDDv2-style long-range drone data for tiny-drone recall.
   *Not yet obtained; placeholder in manifest.*
4. Local `my_drone_chase.MP4` corrected sparse GT as **acceptance** and a
   small domain-adaptation set, **not** as the only training source.

Do not prioritize VisDrone2019-DET for this mission. It is mostly
drone-camera-looking-down detection of ground objects, not skyborne drone
discrimination.

## Training Files

- `configs/training/airborne_dataset_manifest.yaml`
  - declares the target taxonomy and dataset sources;
  - external sources are disabled placeholders until the datasets are present
    locally;
  - local sparse GT source points to the user-corrected CSV;
  - **paths in this file are machine-specific**; edit them for your host.

- `configs/training/airborne_yolo11.yaml`
  - YOLO11s training profile;
  - 80 epochs, batch 8, imgsz 1024, AMP on;
  - device: `auto` (CUDA → MPS → CPU);
  - run name: `yolo11s_airborne_drone_vs_bird_v1`.

- `configs/trained_yolo11s_eval.yaml`
  - acceptance-evaluation profile;
  - same tracker + lock + output settings as `configs/default.yaml`;
  - swaps detector backend to `yolo_ultralytics` pointed at the trained
    `best.pt`;
  - input size 1024 (matches training); confidence threshold 0.25 by
    default.

- `scripts/prepare_airborne_training_set.py`
  - builds `images/{train,val,test}`, `labels/{train,val,test}`, `data.yaml`;
  - supports existing YOLO detection datasets with class remapping;
  - supports Skyscouter sparse-GT CSV plus video frame extraction;
  - writes `manifest_summary.json` with class counts, negatives, and splits.

- `scripts/train_airborne_yolo.py`
  - trains Ultralytics YOLO from a training profile;
  - resolves the YAML `project` value to an absolute path before passing it
    to Ultralytics (otherwise 8.4.x nests the run under its default
    `runs/detect/`, producing surprising paths);
  - device auto-detect: CUDA → MPS → CPU.

- `scripts/extract_miss_highlights.py`
  - reads the eval report, finds every missed positive frame, and renders
    per-frame JPEGs and a slow-motion MP4 with GT (green) and YOLO
    low-conf detections (yellow). Use it to triage detector-vs-tracker
    failures.

## Commands

Build the local sparse-GT sanity dataset only:

```powershell
.\.venv\Scripts\python.exe scripts/prepare_airborne_training_set.py `
  --manifest configs/training/airborne_dataset_manifest.yaml `
  --out-dir data/training/local_sparse_gt_yolo `
  --source local_my_drone_chase_sparse_gt `
  --link-mode copy
```

Build the real training set (AOD-4 + local sparse GT):

```powershell
.\.venv\Scripts\python.exe scripts/prepare_airborne_training_set.py `
  --manifest configs/training/airborne_dataset_manifest.yaml `
  --out-dir data/training/airborne_yolo_v1 `
  --link-mode copy
```

Observed on 2026-04-29 (AOD-4 + local sparse GT):

- train: 17,947 images, 25,115 objects, 523 negatives
- val: 3,493 images, 4,902 objects, 88 negatives
- test: 1,165 images, 1,625 objects, 30 negatives
- per-class train objects: drone 6,269 / bird 6,312 / airplane 6,312 /
  helicopter 6,222

Train:

```powershell
.\.venv\Scripts\python.exe scripts/train_airborne_yolo.py `
  --config configs/training/airborne_yolo11.yaml
```

Resume an interrupted run from `last.pt`:

```powershell
.\.venv\Scripts\python.exe -c "from ultralytics import YOLO; YOLO(r'data\training\runs\yolo11s_airborne_drone_vs_bird_v1\weights\last.pt').train(resume=True)"
```

Evaluate the trained detector with corrected GT:

```powershell
.\.venv\Scripts\python.exe scripts/run_pipeline.py `
  --video data/videos/my_drone_chase.MP4 `
  --config configs/trained_yolo11s_eval.yaml `
  --gt path\to\drone_sparse_gt_corrected.csv `
  --output data/outputs/run_trained_detector_eval
```

Render a miss-frame diagnostic reel after an eval run:

```powershell
.\.venv\Scripts\python.exe scripts/extract_miss_highlights.py
```

## Next Engineering Steps

1. **Cheap recall experiment**: rerun the trained-detector eval with
   `detector.confidence_threshold: 0.10` (in
   `configs/trained_yolo11s_eval.yaml`). The miss-frame diagnostic shows
   six missed frames have YOLO IoU ≥ 0.7 against GT but conf below 0.25.
2. **Tracker reacquisition audit**: six of the missed frames had a
   ≥ 0.25-confidence detection co-located with the GT bbox, but the
   tracker was already in SEARCHING. Review
   `single_target_kalman_lk` reacquisition gates with the diagnostic
   frames as the regression set.
3. **Resume training to epoch 80**. The val curve was still climbing
   when the run was interrupted; remaining epochs may close part of
   the recall gap on small/long-range frames.
4. Obtain LRDDv2 (or equivalent long-range drone) data and enable the
   `lrddv2_long_range_drone` source in the manifest.
5. Tighten lock semantics so `guidance_valid=true` requires semantic
   drone confidence ≥ a calibrated threshold, not just label match.
6. Confidence calibration on validation set (Brier/temperature scaling).

## Acceptance Gate

Do not call the video acceptable until all are true:

- matched positive rate ≥ 0.90;
- median center error ≤ 25 px;
- labeled track ID switches ≤ 3;
- false-lock negative frames = 0.

Current trained-detector run: 1 of 4 gates failing
(matched positive rate 0.705 < 0.90).
