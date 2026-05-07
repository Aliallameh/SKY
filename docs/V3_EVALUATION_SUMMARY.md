# V3 Evaluation Summary

Last updated: 2026-05-07

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

## Full Mavic-Like Held-Out Baseline

This is the current held-out validation gate for Mavic-like VisioDECT samples.
It is larger than the original 20-sample prototype and must not be used for
training.

Dataset:

```text
data/training/validation_slices/visiodect_mavic_like_full
```

Held-out group:

```text
Mavic_Air_sunny
```

Dataset validation:

| Metric | Value |
|---|---:|
| Images | 1,191 |
| Drone boxes | 1,191 |
| Validation errors | 0 |
| Validation warnings | 0 |
| Small boxes | 606 |
| Medium boxes | 585 |

V2 command:

```powershell
$env:YOLO_CONFIG_DIR=(Resolve-Path data\training\.ultralytics).Path
.\.venv_train\Scripts\python.exe scripts\evaluate_yolo_semantic_confusion.py `
  --model data\models\yolo11s_airborne_aod4_antiuav300_v2\best.pt `
  --data data\training\validation_slices\visiodect_mavic_like_full\data.yaml `
  --out data\training\validation_slices\visiodect_mavic_like_full\v2_semantic_confusion_conf025.json `
  --split val --conf 0.25 --imgsz 1024
```

Quick V3 command:

```powershell
$env:YOLO_CONFIG_DIR=(Resolve-Path data\training\.ultralytics).Path
.\.venv_train\Scripts\python.exe scripts\evaluate_yolo_semantic_confusion.py `
  --model data\training\runs\yolo11s_airborne_v3_finetune_quick\weights\best.pt `
  --data data\training\validation_slices\visiodect_mavic_like_full\data.yaml `
  --out data\training\validation_slices\visiodect_mavic_like_full\v3_quick_semantic_confusion_conf025.json `
  --split val --conf 0.25 --imgsz 1024
```

Result:

| Metric | V2 baseline | Quick V3 |
|---|---:|---:|
| GT drone boxes | 1,191 | 1,191 |
| Matched as drone | 882 | 728 |
| Matched as airplane | 228 | 285 |
| Missed | 81 | 178 |
| Drone-to-airplane confusion | 19.14% | 23.93% |
| Geometric recall, small | 89.60% | 78.71% |
| Semantic drone recall, small | 78.22% | 76.24% |
| Geometric recall, medium | 96.92% | 91.62% |
| Semantic drone recall, medium | 69.74% | 45.47% |

Decision:

```text
Quick V3 is rejected as a base for promotion or Stage 3.
```

It fixed the tiny local hard-case packet but worsened the full Mavic-like
held-out validation slice and also produced worse fresh-video continuity under
strict overlay settings. The next promotion-eligible fine-tune should start
from V2 best weights unless a later Stage 2 model beats V2 on this same held-out
slice and on local sparse GT.

## Stage 1 Drone-Only Evaluation

Model:

```text
data/training/runs/yolo11s_airborne_stage1_drone_only_b16w4_nomix/weights/best.pt
```

Stage 1 is drone-only pretraining, not a promotion candidate. It is evaluated
here to decide whether it is useful as a Stage 2 base.

### Local Hard Case

Command outputs:

```text
data/outputs/stage1_eval_20260506/hard_case_v2
data/outputs/stage1_eval_20260506/hard_case_stage1
```

| Metric | V2 baseline | Stage 1 |
|---|---:|---:|
| Positive drone frames | 54 | 54 |
| Negative frames | 7 | 7 |
| Detector geometric hit rate | 42.59% | 98.15% |
| Detector semantic drone hit rate | 1.85% | 98.15% |
| Matched as drone | 1 | 53 |
| Matched as airplane | 22 | 0 |
| Missed | 31 | 1 |
| Tracker geometric hit rate | 53.70% | 100.00% |
| Tracker semantic drone hit rate | 0.00% | 100.00% |
| Detector negative false positives | 0 | 0 |
| Tracker negative false positives | 0 | 1 |
| Tracker stale frames | 6 | 2 |

Stage 1 fixes the local semantic failure, but it still publishes one
tracker-carried negative frame (`134`). That is not a detector false positive,
but it matters for lock safety.

### Full Mavic-Like Held-Out Slice

Command outputs:

```text
data/outputs/stage1_eval_20260506/mavic_like_v2.json
data/outputs/stage1_eval_20260506/mavic_like_stage1.json
```

| Metric | V2 baseline | Stage 1 |
|---|---:|---:|
| GT drone boxes | 1,191 | 1,191 |
| Matched as drone | 882 | 1,191 |
| Matched as airplane | 228 | 0 |
| Missed | 81 | 0 |
| Drone-to-airplane confusion | 19.14% | 0.00% |
| Small semantic drone recall | 78.22% | 100.00% |
| Medium semantic drone recall | 69.74% | 100.00% |

Exact image-hash comparison found `0` duplicate images between the Stage 1
training images and the full Mavic-like held-out slice. However, the Stage 1
training set does include related VisioDECT Mavic Air cloudy samples while the
held-out slice is Mavic Air sunny, so this is a useful domain result but not a
complete real-camera proof.

### AOD-4 Multiclass False-Positive Check

Command outputs:

```text
data/outputs/stage1_eval_20260506/aod4_val_v2.json
data/outputs/stage1_eval_20260506/aod4_val_stage1.json
```

| Metric | V2 baseline | Stage 1 |
|---|---:|---:|
| AOD-4 val images | 4,514 | 4,514 |
| GT drone boxes | 1,602 | 1,602 |
| Matched GT drones as drone | 1,563 | 1,503 |
| Missed GT drones | 36 | 99 |
| Bird-to-drone false matches | 1 | 280 |
| Airplane-to-drone false matches | 0 | 754 |
| Unmatched drone predictions | 113 | 879 |

Decision:

```text
Do not promote Stage 1.
Use it only as a candidate Stage 2 base. Stage 2 must re-teach non-drone
airborne rejection before any Stage 3 local fine-tune, but AOD-4 is only a
capped confuser source. Do not let AOD-4 define drone identity unless its
drone-vs-airplane audit passes.
```

## Stage 2 Capped Multiclass Evaluation

Model:

```text
data/training/runs/yolo11s_airborne_stage2_multiclass_capped_aod4conf_b16w4_nomix/weights/best.pt
```

Stage 2 used Stage 1 best as the base, capped AOD-4 as a confuser source, and
excluded AOD-4 images containing `drone`.

### Local Hard Case

Command:

```powershell
$env:YOLO_CONFIG_DIR=(Resolve-Path data\training\.ultralytics).Path
.\.venv_train\Scripts\python.exe scripts\evaluate_hard_case.py `
  --csv annotations\camera_20260423_113401_turn_review_strict\drone_sparse_gt_corrected.csv `
  --config configs\trained_yolo11s_v2_guidance_full.yaml `
  --weights data\training\runs\yolo11s_airborne_stage2_multiclass_capped_aod4conf_b16w4_nomix\weights\best.pt `
  --output data\outputs\stage2_eval_20260507\hard_case_stage2 `
  --case-name stage2_capped_hard_case_conf025 `
  --confidence-threshold 0.25 `
  --no-copy-images
```

| Metric | V2 baseline | Stage 1 | Stage 2 |
|---|---:|---:|---:|
| Positive drone frames | 54 | 54 | 54 |
| Negative frames | 7 | 7 | 7 |
| Detector geometric hit rate | 42.59% | 98.15% | 87.04% |
| Detector semantic drone hit rate | 1.85% | 98.15% | 22.22% |
| Matched as drone | 1 | 53 | 12 |
| Matched as airplane | 22 | 0 | 35 |
| Missed | 31 | 1 | 7 |
| Tracker geometric hit rate | 53.70% | 100.00% | 94.44% |
| Tracker semantic drone hit rate | 0.00% | 100.00% | 24.07% |
| Detector negative false positives | 0 | 0 | 0 |
| Tracker negative false positives | 0 | 1 | 1 |

Stage 2 improves geometric detection versus v2, but it does not pass the local
semantic gate. It still labels most matched local drone boxes as `airplane`.

### Full Mavic-Like Held-Out Slice

Command:

```powershell
$env:YOLO_CONFIG_DIR=(Resolve-Path data\training\.ultralytics).Path
.\.venv_train\Scripts\python.exe scripts\evaluate_yolo_semantic_confusion.py `
  --model data\training\runs\yolo11s_airborne_stage2_multiclass_capped_aod4conf_b16w4_nomix\weights\best.pt `
  --data data\training\validation_slices\visiodect_mavic_like_full\data.yaml `
  --out data\outputs\stage2_eval_20260507\mavic_like_stage2.json `
  --split val --conf 0.25 --imgsz 1024 --device 0
```

| Metric | V2 baseline | Quick V3 | Stage 1 | Stage 2 |
|---|---:|---:|---:|---:|
| GT drone boxes | 1,191 | 1,191 | 1,191 | 1,191 |
| Matched as drone | 882 | 728 | 1,191 | 1,191 |
| Matched as airplane | 228 | 285 | 0 | 0 |
| Missed | 81 | 178 | 0 | 0 |
| Drone-to-airplane confusion | 19.14% | 23.93% | 0.00% | 0.00% |
| Small semantic drone recall | 78.22% | 76.24% | 100.00% | 100.00% |
| Medium semantic drone recall | 69.74% | 45.47% | 100.00% | 100.00% |

Stage 2 preserves the Mavic-like win.

### AOD-4 Multiclass False-Positive Check

Command:

```powershell
$env:YOLO_CONFIG_DIR=(Resolve-Path data\training\.ultralytics).Path
.\.venv_train\Scripts\python.exe scripts\evaluate_yolo_semantic_confusion.py `
  --model data\training\runs\yolo11s_airborne_stage2_multiclass_capped_aod4conf_b16w4_nomix\weights\best.pt `
  --data data\training\converted\aod4\data.yaml `
  --out data\outputs\stage2_eval_20260507\aod4_val_stage2.json `
  --split val --conf 0.25 --imgsz 1024 --device 0
```

| Metric | V2 baseline | Stage 1 | Stage 2 |
|---|---:|---:|---:|
| AOD-4 val images | 4,514 | 4,514 | 4,514 |
| GT drone boxes | 1,602 | 1,602 | 1,602 |
| Matched GT drones as drone | 1,563 | 1,503 | 1,080 |
| Matched GT drones as airplane | 0 | 0 | 293 |
| Matched GT drones as bird | 2 | 0 | 15 |
| Matched GT drones as helicopter | 1 | 0 | 67 |
| Missed GT drones | 36 | 99 | 147 |
| Bird-to-drone false matches | 1 | 280 | 0 |
| Airplane-to-drone false matches | 0 | 754 | 1 |
| Unmatched drone predictions | 113 | 879 | 45 |

Stage 2 repairs the worst Stage 1 false-positive collapse, but because AOD-4
`drone` images were excluded from training, AOD-4 drone recall and semantics
regress hard. This confirms AOD-4 should not be the authority for drone identity
and that Stage 2 is not promotion-ready.

### Stage 2 Decision

```text
Do not promote Stage 2.
Do not use Stage 2 as the default Stage 3 base.
Use v2 as the Stage 3 base unless a Stage 2b run fixes local hard-case drone->airplane confusion.
```

Why:

- Mavic-like held-out gate: pass.
- Non-drone false-positive rejection: mostly pass.
- Local hard-case semantic gate: fail.
- AOD-4 drone semantics: fail, expected from excluding AOD-4 drone.

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

## Stage 2 Fresh Video Overlay Review

These are still proxy/video-review metrics. They do not replace sparse GT, but
they are useful for spotting product-level tracker failures.

Stage 2 improved `Video_5` materially versus V2:

| Metric | V2 | Stage 2 | Stage 2 + edge-start gate |
|---|---:|---:|---:|
| `drone` label frames | 126 | 2,320 | 2,312 |
| `airplane` label frames | 1,878 | 221 | 221 |
| `LOCKED` frames | 25 | 977 | 974 |
| median confidence | 0.428 | 0.437 | 0.437 |

However, Stage 2 also exposed a `Video_1` failure mode: fresh tracks could
initialize on a clipped left-edge false positive and remain visually stuck
near `cx ~= 25 px`.

Implemented mitigation:

```yaml
tracker:
  reject_edge_initial_detections: true
  edge_reject_margin_px: 2.0
```

The gate rejects only new track starts on clipped/touching-edge detections. It
does not reject an already-valid track that later continues toward the edge.

`Video_1` result:

| Metric | Stage 2 before | Stage 2 + edge-start gate |
|---|---:|---:|
| Active frames with `cx < 120 px` | 102 | 0 |
| Left-edge sticky runs | 3 | 0 |
| `LOCKED` frames | 1,117 | 1,126 |
| `drone` label frames | 2,078 | 2,022 |
| `airplane` label frames | 213 | 215 |

Output folders:

```text
data/outputs/fresh_20260507_stage2_edgegate_Video_1
data/outputs/fresh_20260507_stage2_edgegate_Video_5
```

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
