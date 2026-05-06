# Dataset Integration Report

Last updated: 2026-05-05

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
| AOD-4 | 22,516 images, COCO/VOC/YOLOv8/TF annotations | Stage 2 capped rejection/confuser source after visual audit; not the authority for drone identity |
| Drone-vs-Bird | 4,106 classification images, no boxes found | Do not use for YOLO detection; later hard-negative/crop-classifier work only |

## AOD-4 Decision

AOD-4 was selected because it is the only currently local detection dataset with
explicit `drone`, `bird`, `airplane`, and `helicopter` boxes. That makes it
useful for rejection: it teaches the detector that not every airborne object is
a lockable drone.

It is not considered the best or final product dataset. For V3/V4, drone
identity should be driven by local Mavic-style annotations, VisioDECT,
Anti-UAV-RGBT, and DUT-Anti-UAV. AOD-4 should be capped and used mainly for
airplane/bird/helicopter confusers. The Stage 2 dataset builder now excludes
images containing AOD-4 `drone` boxes by default until the AOD-4
drone-vs-airplane visual audit passes.

Safer Stage 2 build shape:

```powershell
.\.venv_train\Scripts\python.exe scripts\build_staged_airborne_dataset.py `
  --stage stage2 `
  --link-mode copy `
  --cap aod4=6000 `
  --cap anti_uav_rgbt=5000 `
  --cap dut_anti_uav=5000 `
  --cap visiodect=12000
```

To intentionally include AOD-4 drone boxes after audit, add
`--no-default-exclusions` and record the audit result first.

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

Full held-out validation slice:

```text
data/training/validation_slices/visiodect_mavic_like_full
```

Current full held-out group:

```text
Mavic_Air_sunny
```

Full Mavic-like slice status:

| Metric | Value |
|---|---:|
| Images | 1,191 |
| Drone boxes | 1,191 |
| Validation | PASS |
| Errors | 0 |
| Warnings | 0 |
| Small boxes | 606 |
| Medium boxes | 585 |

Full VisioDECT converted source:

```text
data/training/converted/visiodect
```

| Metric | Value |
|---|---:|
| Images | 18,226 |
| Drone boxes | 18,227 |
| Validation | PASS |
| Errors | 0 |
| Warnings | 0 |
| Tiny boxes | 1,647 |
| Small boxes | 11,445 |
| Medium boxes | 5,135 |

## Full Conversion Status

All raw detection sources needed for Stage 1 and Stage 2 have now been
mechanically converted and validated. These converted folders are generated
artifacts and remain ignored by git.

| Dataset | Output | Images | Objects | Validation | Notes |
|---|---|---:|---:|---|---|
| Anti-UAV-RGBT | `data/training/converted/anti_uav_rgbt` | 19,250 | 18,727 | PASS | 523 empty negatives |
| DUT-Anti-UAV | `data/training/converted/dut_anti_uav` | 10,000 | 10,108 | PASS | 1 duplicate warning |
| VisioDECT | `data/training/converted/visiodect` | 18,226 | 18,227 | PASS | excludes held-out `Mavic_Air_sunny` |
| AOD-4 | `data/training/converted/aod4` | 22,516 | 31,598 | PASS | balanced multiclass conversion; Stage 2 should cap it and exclude AOD-4 drone boxes until audit |

Full preview folders:

```text
data/training/previews/anti_uav_rgbt_full
data/training/previews/dut_anti_uav_full
data/training/previews/visiodect_full
data/training/previews/aod4_full
data/training/previews/visiodect_mavic_like_full
```

## Stage 1 Dataset Status

Merged Stage 1 drone-only dataset:

```text
data/training/airborne_stage1_drone_only
```

| Metric | Value |
|---|---:|
| Images | 47,476 |
| Drone boxes | 47,062 |
| Empty labels | 526 |
| Validation | PASS |
| Errors | 0 |
| Warnings | 0 |
| Tiny boxes | 6,878 |
| Small boxes | 20,322 |
| Medium boxes | 19,862 |

Preview folder:

```text
data/training/previews/airborne_stage1_drone_only
```

## Stage 2 Dataset Status

Capped Stage 2 dataset:

```text
data/training/airborne_stage2_multiclass
```

Build policy:

- AOD-4 is capped to 6,000 images.
- Anti-UAV-RGBT is capped to 5,000 images.
- DUT-Anti-UAV is capped to 5,000 images.
- VisioDECT is capped to 12,000 images.
- AOD-4 images containing `drone` are excluded by default.

| Metric | Value |
|---|---:|
| Images | 28,000 |
| Objects | 31,093 |
| Drone boxes | 21,892 |
| Bird boxes | 2,913 |
| Airplane boxes | 3,095 |
| Helicopter boxes | 3,193 |
| Empty labels | 419 |
| Validation | PASS |
| Errors | 0 |
| Warnings | 0 |
| AOD-4 images skipped because they contained `drone` | 2,981 |

Preview folder:

```text
data/training/previews/airborne_stage2_multiclass
```

Rendered previews: 360.

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

- Human visual review of full preview folders.
- AOD-4 confuser visual audit, especially tiny bird/airplane/helicopter boxes.
- AOD-4 drone-vs-airplane visual audit before any training run includes AOD-4 drone boxes.
- Full Mavic-like held-out slice creation and v2 baseline on that full slice. **Done for VisioDECT Mavic_Air_sunny.**
- Local Mavic-style annotation: at least 100-200 frames before Stage 3.
