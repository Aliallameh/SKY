# V3 Training Results

Last updated: 2026-05-05

## Status

No full staged V3 training run has been started.

That is intentional. The gates now block uncontrolled 80-epoch training until inspection, 20-sample conversions, validation, previews, full conversion summaries, and AOD-4 visual audit pass.

A minimal V3 quick domain-adaptation fine-tune has been completed from v2 best weights for the corrected local hard-case packet. This is not a replacement for the staged V3/V4 pipeline, and it is not promoted.

## Quick V3 Fine-Tune

Purpose:

- Adapt the current v2 model to the corrected local turn/reversal hard case.
- Test whether the airplane-vs-drone semantic failure can be corrected before full Stage 1/2 training.

Inputs:

```text
base: data/models/yolo11s_airborne_aod4_antiuav300_v2/best.pt
dataset: data/training/airborne_yolo_v3_finetune_quick/data.yaml
config: configs/training/airborne_yolo11_v3_finetune_quick.yaml
```

Dataset build:

```powershell
.\.venv_train\Scripts\python.exe scripts\build_v3_finetune_quick_dataset.py `
  --out-dir data\training\airborne_yolo_v3_finetune_quick `
  --local-csv annotations\camera_20260423_113401_turn_review_strict\drone_sparse_gt_corrected.csv `
  --public-count 2000 `
  --link-mode copy
```

Dataset validation:

```powershell
.\.venv_train\Scripts\python.exe scripts\validate_yolo_dataset.py `
  --data data\training\airborne_yolo_v3_finetune_quick\data.yaml
```

Validation result:

| Metric | Value |
|---|---:|
| Images | 2061 |
| Objects | 3126 |
| Drone objects | 595 |
| Bird objects | 1179 |
| Airplane objects | 694 |
| Helicopter objects | 658 |
| Validation errors | 0 |
| Validation warnings | 0 |

Training command:

```powershell
$env:YOLO_CONFIG_DIR=(Resolve-Path data\training\.ultralytics).Path
.\.venv_train\Scripts\python.exe scripts\train_airborne_yolo.py `
  --config configs\training\airborne_yolo11_v3_finetune_quick.yaml `
  --batch 8 `
  --workers 0
```

Training result:

| Metric | Value |
|---|---:|
| Requested epochs | 40 |
| Completed epochs | 38 |
| Early-stopping best epoch | 26 |
| Runtime | 1.030 hours |
| Image size | 1024 |
| Batch | 8 |
| Workers | 0 |
| Optimizer | AdamW |
| lr0 | 0.001 |
| Freeze | 10 |
| Final all mAP50 | 0.971 |
| Final all mAP50-95 | 0.678 |
| Final drone precision | 0.932 |
| Final drone recall | 0.932 |
| Final drone mAP50 | 0.953 |

Weights:

```text
data/training/runs/yolo11s_airborne_v3_finetune_quick/weights/best.pt
data/training/runs/yolo11s_airborne_v3_finetune_quick/weights/last.pt
```

Promotion status: rejected for promotion and rejected as a Stage 3 base.

The local semantic result is strong on the corrected hard-case packet, but the
model does not generalize cleanly:

- strict `Video_1` overlay continuity regressed versus V2;
- full held-out Mavic-like validation got worse versus V2.

Full held-out Mavic-like result at `conf=0.25`:

| Metric | V2 baseline | Quick V3 |
|---|---:|---:|
| GT drone boxes | 1,191 | 1,191 |
| Matched as drone | 882 | 728 |
| Matched as airplane | 228 | 285 |
| Missed | 81 | 178 |

Do not use `data/training/runs/yolo11s_airborne_v3_finetune_quick/weights/best.pt`
as a base for the next fine-tune. Use V2 best weights unless Stage 2 beats V2
on both local sparse GT and the full Mavic-like held-out slice.

## Implemented Training Configs

```text
configs/training/airborne_yolo11_stage1_drone_only.yaml
configs/training/airborne_yolo11_stage2_multiclass.yaml
configs/training/airborne_yolo11_stage3_camhard_finetune.yaml
configs/training/airborne_dataset_manifest_v3.yaml
```

The training wrapper now supports these YAML fields:

```text
lr0
freeze
cos_lr
```

## Stage Plan

| Stage | Dataset | Base | Epochs | Promotion |
|---|---|---|---:|---|
| Stage 1 | `data/training/airborne_stage1_drone_only` | `yolo11s.pt` | 80 | never promote |
| Stage 2 | `data/training/airborne_stage2_multiclass` | Stage 1 best | 80 | not directly promoted |
| Stage 3 | `data/training/airborne_stage3_camhard_finetune` | Stage 2 best only if it beats v2 on semantic confusion; otherwise v2 best | 40 | eligible |

Stage 3 conservative default:

```text
data/models/yolo11s_airborne_aod4_antiuav300_v2/best.pt
```

## Prototype Conversion Commands

Anti-UAV-RGBT:

```powershell
.\.venv_train\Scripts\python.exe scripts\datasets\convert_anti_uav_rgbt.py `
  --root DATASETS\1_Anti-UAV-RGBT `
  --out-dir data\training\prototypes\anti_uav_rgbt_20 `
  --limit 20 --stride 5 --max-positives-per-seq 60 --max-negatives-per-seq 5
```

DUT-Anti-UAV:

```powershell
.\.venv_train\Scripts\python.exe scripts\datasets\convert_dut_anti_uav.py `
  --root DATASETS\2_DUT-Anti-UAV `
  --out-dir data\training\prototypes\dut_anti_uav_20 `
  --limit 20 `
  --extract-dir data\training\raw_cache\dut_antiuav_detection `
  --link-mode copy
```

VisioDECT:

```powershell
.\.venv_train\Scripts\python.exe scripts\datasets\convert_visiodect.py `
  --root "DATASETS\4_VisioDECT Dataset Upload" `
  --out-dir data\training\prototypes\visiodect_20 `
  --mavic-heldout-dir data\training\validation_slices\visiodect_mavic_like `
  --limit 20 --mavic-heldout-limit 20 --link-mode copy
```

AOD-4:

```powershell
.\.venv_train\Scripts\python.exe scripts\datasets\convert_aod4.py `
  --root "DATASETS\5_AOD 4 Dataset for Air Borne Object Detection" `
  --out-dir data\training\prototypes\aod4_20 `
  --limit 20 --link-mode copy
```

Drone-vs-Bird:

```powershell
.\.venv_train\Scripts\python.exe scripts\datasets\convert_drone_vs_bird.py `
  --root "DATASETS\3_Drone vs Bird Aerial Object Classification Dataset" `
  --out-dir data\training\prototypes\drone_vs_bird_report `
  --limit 20
```

Decision: Drone-vs-Bird is classification-only in the current download, so it is blocked from YOLO detection training.

## Prototype Validation

Validation commands:

```powershell
.\.venv_train\Scripts\python.exe scripts\validate_yolo_dataset.py --data data\training\prototypes\anti_uav_rgbt_20\data.yaml
.\.venv_train\Scripts\python.exe scripts\validate_yolo_dataset.py --data data\training\prototypes\dut_anti_uav_20\data.yaml
.\.venv_train\Scripts\python.exe scripts\validate_yolo_dataset.py --data data\training\prototypes\visiodect_20\data.yaml
.\.venv_train\Scripts\python.exe scripts\validate_yolo_dataset.py --data data\training\prototypes\aod4_20\data.yaml
.\.venv_train\Scripts\python.exe scripts\validate_yolo_dataset.py --data data\training\validation_slices\visiodect_mavic_like\data.yaml
```

Results:

| Dataset | Status | Note |
|---|---|---|
| Anti-UAV-RGBT 20 | PASS | 18 positives, 2 empty negative labels |
| DUT-Anti-UAV 20 | PASS | duplicate image warning in prototype sample |
| VisioDECT 20 | PASS | mostly small boxes |
| AOD-4 20 | PASS | prototype contained drone/bird only; full audit still required |
| Mavic-like held-out 20 | PASS | v2 baseline ran on this slice |

## Preview Commands

```powershell
.\.venv_train\Scripts\python.exe scripts\preview_yolo_labels.py --data data\training\prototypes\anti_uav_rgbt_20\data.yaml --out-dir data\training\previews\anti_uav_rgbt_20
.\.venv_train\Scripts\python.exe scripts\preview_yolo_labels.py --data data\training\prototypes\dut_anti_uav_20\data.yaml --out-dir data\training\previews\dut_anti_uav_20
.\.venv_train\Scripts\python.exe scripts\preview_yolo_labels.py --data data\training\prototypes\visiodect_20\data.yaml --out-dir data\training\previews\visiodect_20
.\.venv_train\Scripts\python.exe scripts\preview_yolo_labels.py --data data\training\prototypes\aod4_20\data.yaml --out-dir data\training\previews\aod4_20
.\.venv_train\Scripts\python.exe scripts\preview_yolo_labels.py --data data\training\validation_slices\visiodect_mavic_like\data.yaml --out-dir data\training\previews\visiodect_mavic_like_20
```

Previews rendered. Human review is still required before full conversion.

## Smoke Test Commands

Run only after dataset gates pass.

Stage 1:

```powershell
.\.venv_train\Scripts\python.exe scripts\train_airborne_yolo.py `
  --config configs\training\airborne_yolo11_stage1_drone_only.yaml `
  --epochs 1 --batch 8 --workers 8
```

Stage 2:

```powershell
.\.venv_train\Scripts\python.exe scripts\train_airborne_yolo.py `
  --config configs\training\airborne_yolo11_stage2_multiclass.yaml `
  --epochs 1 --batch 8 --workers 8
```

Stage 3:

```powershell
.\.venv_train\Scripts\python.exe scripts\train_airborne_yolo.py `
  --config configs\training\airborne_yolo11_stage3_camhard_finetune.yaml `
  --epochs 1 --batch 8 --workers 8
```

Use this inside scripted commands to avoid locked Ultralytics AppData settings:

```powershell
New-Item -ItemType Directory -Force data\training\.ultralytics | Out-Null
$env:YOLO_CONFIG_DIR=(Resolve-Path data\training\.ultralytics).Path
```

## Next Training Step

The immediate next safe step is still dataset-gated, not another quick fine-tune.
VisioDECT full conversion and the full Mavic-like held-out slice are complete:

```text
data/training/converted/visiodect
data/training/validation_slices/visiodect_mavic_like_full
```

Next, review the generated previews and complete the remaining full conversions:

```text
data/training/converted/anti_uav_rgbt
data/training/converted/dut_anti_uav
data/training/converted/aod4
```

Current status: these full conversions are complete and validate cleanly.

Merged Stage 1 dataset is also built and validated:

```text
data/training/airborne_stage1_drone_only
```

| Metric | Value |
|---|---:|
| Images | 47,476 |
| Drone boxes | 47,062 |
| Empty labels | 526 |
| Validation errors | 0 |
| Validation warnings | 0 |

Command used to build Stage 1:

```powershell
.\.venv_train\Scripts\python.exe scripts\build_staged_airborne_dataset.py --stage stage1
```

Stage 1 full training was launched after the full dataset validated and the
initial aggressive dataloader run exposed a Windows worker RAM failure from
`mixup`.

The Stage 1 config was changed to:

```text
mixup: 0.0
```

Active full-training command:

```powershell
$env:YOLO_CONFIG_DIR=(Resolve-Path data\training\.ultralytics).Path
.\.venv_train\Scripts\python.exe scripts\train_airborne_yolo.py `
  --config configs\training\airborne_yolo11_stage1_drone_only.yaml `
  --batch 16 --workers 4 `
  --run-name yolo11s_airborne_stage1_drone_only_b16w4_nomix
```

Run folder:

```text
data/training/runs/yolo11s_airborne_stage1_drone_only_b16w4_nomix
```

Stage 1 remains diagnostic/pretraining only and must not be promoted directly.
