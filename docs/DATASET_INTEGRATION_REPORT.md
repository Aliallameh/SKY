# Dataset Integration Report

Last updated: 2026-05-04

## Purpose

V3/V4 training is not a random retrain. The blocker is semantic: current v2 often finds the airborne target but labels a real drone as `airplane`, so semantic-safe lock refuses `LOCKED`.

Training taxonomy:

| Stage | Classes |
|---|---|
| Stage 1 | `0 drone` |
| Stage 2/3 | `0 drone`, `1 bird`, `2 airplane`, `3 helicopter` |

`unknown_airborne` is not a YOLO training class unless a dataset has explicit labelled boxes for it. Generic/supercategory labels are skipped.

## Inspection Gate

Command:

```powershell
.\.venv_train\Scripts\python.exe scripts\inspect_airborne_datasets.py
```

Outputs:

```text
data/training/dataset_inspection_report.json
data/training/dataset_inspection_report.md
```

## Raw Dataset Findings

| Dataset | Evidence | Decision |
|---|---|---|
| Anti-UAV-RGBT | 636 videos, 639 JSON annotation files with `exist[]` and `gt_rect[]` | Stage 1 drone-only, visible/RGB only |
| DUT-Anti-UAV Detection | 34,804 images, VOC XML with `UAV` boxes in extracted Detection folders | Stage 1 drone-only |
| VisioDECT | 20,924 images, VOC/YOLO/CSV annotations | Stage 1 drone-only; Mavic-like validation held out |
| AOD-4 | 22,516 images, COCO/VOC/YOLOv8/TF annotations | Stage 2 multiclass after visual audit |
| Drone-vs-Bird | 4,106 classification images, no boxes found | Do not use for YOLO detection; later hard-negative/crop-classifier work only |

## Class Remapping

| Dataset | Source label | Target |
|---|---|---|
| Anti-UAV-RGBT | `exist=1` visible target | `0 drone` |
| Anti-UAV-RGBT | `exist=0` | empty negative label |
| DUT-Anti-UAV | `UAV` | `0 drone` |
| VisioDECT | all drone model folders/classes | `0 drone` |
| AOD-4 | `0 airplane` | `2 airplane` |
| AOD-4 | `1 bird` | `1 bird` |
| AOD-4 | `2 drone` | `0 drone` |
| AOD-4 | `3 helicopter` | `3 helicopter` |
| AOD-4 | generic COCO supercategory | skip |

## VisioDECT Mavic-Like Discovery

Discovered folder matches:

```text
Mavic_Air
Mavic_Enterprise
```

Discovered label names:

```text
MAVIC_Air_Cloudy
Mavic_Air_Cloudy
MAVIC_Air_Evening
Mavic_Air_Evening
MAVIC_Air_Sunny
Mavic_Air_Sunny
MAVIC2_Enterprise_Dual_Cloudy
Mavic_Enterprise_Cloudy
MAVIC2_Enterprise_Dual_Evening
Mavic_Enterprise_Evening
MAVIC2_Enterprise_Dual_Sunny
Mavic_Enterprise_Sunny
```

Prototype held-out validation slice:

```text
data/training/validation_slices/visiodect_mavic_like
```

Current prototype held-out group:

```text
Mavic_Air_sunny
```

This slice must not leak into Stage 1, Stage 2, or Stage 3 training.

## Implemented Scripts

```text
scripts/inspect_airborne_datasets.py
scripts/datasets/common_yolo.py
scripts/datasets/convert_anti_uav_rgbt.py
scripts/datasets/convert_dut_anti_uav.py
scripts/datasets/convert_visiodect.py
scripts/datasets/convert_aod4.py
scripts/datasets/convert_drone_vs_bird.py
scripts/validate_yolo_dataset.py
scripts/preview_yolo_labels.py
scripts/build_staged_airborne_dataset.py
scripts/evaluate_yolo_semantic_confusion.py
```

## 20-Sample Prototype Status

Commands were run with `--limit 20`.

| Prototype | Images | Objects | Classes | Validation | Preview |
|---|---:|---:|---|---|---|
| `data/training/prototypes/anti_uav_rgbt_20` | 20 | 18 | drone | PASS | rendered |
| `data/training/prototypes/dut_anti_uav_20` | 20 | 20 | drone | PASS, duplicate warning | rendered |
| `data/training/prototypes/visiodect_20` | 20 | 20 | drone | PASS | rendered |
| `data/training/prototypes/aod4_20` | 20 | 20 | drone/bird | PASS, missing airplane/helicopter in tiny sample | rendered |
| `data/training/validation_slices/visiodect_mavic_like` | 20 | 20 | drone | PASS | rendered |

Preview folders:

```text
data/training/previews/anti_uav_rgbt_20
data/training/previews/dut_anti_uav_20
data/training/previews/visiodect_20
data/training/previews/aod4_20
data/training/previews/visiodect_mavic_like_20
```

## Gates Still Open

Do not run full Stage 1/2/3 training yet.

Remaining gates:

- Human visual review of prototype previews.
- Full conversion after preview approval.
- AOD-4 drone-vs-airplane visual audit, especially tiny objects.
- Full Mavic-like held-out slice creation and v2 baseline on that full slice.
- Local Mavic-style annotation: at least 100-200 frames before Stage 3.
