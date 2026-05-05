# V3 Evaluation Summary

Last updated: 2026-05-04

## Current V2 Baseline

Model:

```text
data/models/yolo11s_airborne_aod4_antiuav300_v2/best.pt
```

The current model is not failing only because of mAP. The main failure is matched-GT semantic confusion: the drone is often detected geometrically but predicted as `airplane`.

## Local Hard-Case GT Baseline

Command:

```powershell
$env:YOLO_CONFIG_DIR=(Resolve-Path data\training\.ultralytics).Path
.\.venv_train\Scripts\python.exe scripts\evaluate_hard_case.py `
  --csv annotations\camera_20260423_113401_turn_review_strict\drone_sparse_gt_corrected.csv `
  --config configs\trained_yolo11s_v2_guidance_full.yaml `
  --output data\outputs\hard_case_camera_20260423_113401_turn_review_strict_v2_baseline_v3_metrics `
  --case-name camera_20260423_113401_turn_review_strict_v2_baseline_v3_metrics `
  --confidence-threshold 0.25
```

Result:

| Metric | Value |
|---|---:|
| Frames | 61 |
| Positive drone frames | 54 |
| Negative frames | 7 |
| Detector geometric hit rate | 42.59% |
| Detector semantic drone hit rate | 1.85% |
| Tracker geometric hit rate | 53.70% |
| Tracker semantic drone hit rate | 0.00% |
| Negative false positives | 0 |

Matched-GT semantic confusion for GT drone boxes:

| Outcome | Count |
|---|---:|
| matched as drone | 1 |
| matched as airplane | 22 |
| matched as bird | 0 |
| matched as helicopter | 0 |
| matched wrong-class | 0 |
| missed | 31 |

Drone recall by bbox size:

| Size bin | GT frames | Geometric recall | Semantic drone recall |
|---|---:|---:|---:|
| tiny | 25 | 40.00% | 0.00% |
| small | 29 | 44.83% | 3.45% |

Size bins:

```text
tiny:  bbox_area / image_area < 0.0005
small: 0.0005 <= ratio < 0.0025
medium: ratio >= 0.0025
```

## Quick V3 Local Hard-Case Result

Model:

```text
data/training/runs/yolo11s_airborne_v3_finetune_quick/weights/best.pt
```

Command:

```powershell
$env:YOLO_CONFIG_DIR=(Resolve-Path data\training\.ultralytics).Path
.\.venv_train\Scripts\python.exe scripts\evaluate_hard_case.py `
  --csv annotations\camera_20260423_113401_turn_review_strict\drone_sparse_gt_corrected.csv `
  --config configs\trained_yolo11s_v2_guidance_full.yaml `
  --weights data\training\runs\yolo11s_airborne_v3_finetune_quick\weights\best.pt `
  --output data\outputs\hard_case_camera_20260423_113401_v3_finetune_quick_split_fp `
  --case-name camera_20260423_113401_v3_finetune_quick_split_fp `
  --confidence-threshold 0.25 `
  --no-copy-images
```

Result:

| Metric | V2 baseline | Quick V3 |
|---|---:|---:|
| Frames | 61 | 61 |
| Positive drone frames | 54 | 54 |
| Negative frames | 7 | 7 |
| Detector geometric hit rate | 42.59% | 100.00% |
| Detector semantic drone hit rate | 1.85% | 100.00% |
| Tracker geometric hit rate | 53.70% | 100.00% |
| Tracker semantic drone hit rate | 0.00% | 100.00% |
| Detector negative false positive frames | 0 | 0 |
| Tracker stale negative false positive frames | 0 | 1 |

Matched-GT semantic confusion for GT drone boxes:

| Outcome | V2 baseline | Quick V3 |
|---|---:|---:|
| matched as drone | 1 | 54 |
| matched as airplane | 22 | 0 |
| matched as bird | 0 | 0 |
| matched as helicopter | 0 | 0 |
| matched wrong-class | 0 | 0 |
| missed | 31 | 0 |

Drone recall by bbox size:

| Size bin | GT frames | V2 geometric | V2 semantic | Quick V3 geometric | Quick V3 semantic |
|---|---:|---:|---:|---:|---:|
| tiny | 25 | 40.00% | 0.00% | 100.00% | 100.00% |
| small | 29 | 44.83% | 3.45% | 100.00% | 100.00% |

Caveat:

- Quick V3 fixed the corrected local semantic failure on this packet.
- It also exposed one tracker-only stale publication on negative frame `134`: the detector had zero detections on that frame, but the tracker carried the previous drone track for one frame.
- This is not a promotion result until fresh Video_1-Video_5 sparse GT, held-out Mavic-like validation, and negative/false-lock review pass.

## Prototype Mavic-Like Baseline

This is a 20-sample prototype slice, not the final held-out evaluation.

Command:

```powershell
$env:YOLO_CONFIG_DIR=(Resolve-Path data\training\.ultralytics).Path
.\.venv_train\Scripts\python.exe scripts\evaluate_yolo_semantic_confusion.py `
  --model data\models\yolo11s_airborne_aod4_antiuav300_v2\best.pt `
  --data data\training\validation_slices\visiodect_mavic_like\data.yaml `
  --out data\training\validation_slices\visiodect_mavic_like\v2_semantic_confusion_conf025.json `
  --split val --conf 0.25 --imgsz 1024
```

Result:

| Outcome | Count |
|---|---:|
| matched as drone | 12 |
| matched as airplane | 8 |
| matched as bird | 0 |
| matched as helicopter | 0 |
| missed | 0 |

The v2 prototype Mavic-like drone-to-airplane confusion rate is `8 / 20 = 40%`.

## Fresh Video Proxy Baseline

Fresh videos still need sparse GT annotation. Until then, these are proxy metrics only.

| Video | Target coverage | Tracking+ | Semantic-safe locked | Drone label | Airplane label |
|---|---:|---:|---:|---:|---:|
| Video_1 | 96.30% | 93.52% | 0.91% | 17.95% | 78.32% |
| Video_2 | 0.18% | 0.00% | 0.00% | 0.00% | 0.18% |
| Video_3 | 33.24% | 31.83% | 0.00% | 1.88% | 30.42% |
| Video_4 | 14.85% | 13.11% | 0.00% | 0.03% | 14.82% |
| Video_5 | 44.48% | 39.16% | 0.55% | 2.79% | 41.64% |

Review-only airborne-target lock mode showed geometry is often better than semantic-safe lock:

| Video | Semantic-safe locked | Airborne-review locked |
|---|---:|---:|
| Video_1 | 0.91% | 64.52% |
| Video_2 | 0.00% | 0.00% |
| Video_3 | 0.00% | 22.82% |
| Video_4 | 0.00% | 8.95% |
| Video_5 | 0.55% | 29.69% |

## Mode Separation

Semantic-safe mode:

- Only approved drone labels may lock.
- This is the only mode eligible for safety-oriented claims.

Airborne-review mode:

- Allows non-drone airborne semantics only for review overlays.
- Useful for inspecting detector/tracker geometry.
- Never flight-safe.

## Promotion Metrics

Promotion priority:

1. false-lock negatives = 0
2. reduced Mavic-like drone-to-airplane confusion
3. no significant increase in bird-to-drone or airplane-to-drone false positives
4. improved local semantic drone hit rate
5. mAP

Promotion gate:

```text
matched positive rate >= 0.90
median center error <= 25 px
labelled track ID switches <= 3
false-lock negative frames = 0
Mavic-like drone->airplane confusion reduced versus v2
bird->drone and airplane->drone false positives not significantly worse than v2
```
