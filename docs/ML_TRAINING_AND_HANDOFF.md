# ML Training and Handoff Notes

Last updated: 2026-05-02

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
- `docs/TRAINING_RUNBOOK.md` documents Windows/NVIDIA setup, resume,
  monitoring, and GPU utilization tuning.
- `docs/E80_FINAL_RESULTS.md` records the completed epoch-80 checkpoint,
  hard-case semantic failure, and next v3 fine-tuning direction.
- Trained-detector eval profile in `configs/trained_yolo11s_eval.yaml`.
- Diagnostic: `scripts/extract_miss_highlights.py` renders a slow-motion
  reel + JPEGs of every positive frame the trained detector missed, with
  GT in green and YOLO low-confidence detections in yellow.

**Latest corrected-GT replay (v2 detector + tracker stale-state fix)**:

- Run: `data/outputs/run_v2_eval_trackerfix/`
- Config: `configs/trained_yolo11s_v2_eval.yaml`
- Detector: `yolo11s_airborne_aod4_antiuav300_v2` (epoch 80 / 80 final,
  promoted from historical run `yolo11s_airborne_drone_vs_bird_v2`)
- GT file: corrected sparse GT (44 positive drone frames + 45 hard negatives)
- Frames processed: 9,744
- Matched positive rate: **0.977** (43/44)
- Median IoU when matched: **0.78**
- Median center error: **1.89 px**
- False-lock negative frames: **0**
- Labeled track ID switches: **2** (≤ 3 gate)
- Acceptance gate: **✅ PASSES** (all 4 gates clear)

For the full progression v1 → v2 → v2+tracker-fix and the per-frame
breakdown see [`PHASE_1_RESULTS.md`](PHASE_1_RESULTS.md).

**Detector vs tracker root-cause split (historical, pre-fix)** — from
`scripts/extract_miss_highlights.py` against the v1 detector:

- Of 13 missed positive frames in the v1 conf=0.25 baseline, **1**
  (frame 405) was true detector blindness, **6** had usable YOLO signal
  below the 0.25 confidence threshold, and **6** had YOLO signal above
  threshold with IoU 0.7–0.9 versus GT that the tracker did not propagate
  because it was in `SEARCHING` after a prior gap.

The 6 below-threshold cases were fixed by v2's improved calibration
(frames now sit at conf ≥ 0.25). The 6 tracker-miss cases were fixed by
the stale-state reacquisition patch in
`skyscouter/tracking/single_target_kalman_lk.py`. The single
detector-blind case (frame 405) remains; it requires additional
long-range training data (LRDDv2 or similar) — out of scope for the
gate.

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
  - forwards `workers`, `batch`, `cache`, and `amp` to Ultralytics;
  - device auto-detect: CUDA → MPS → CPU.

- `scripts/resume_yolo_v2_training.ps1`
  - resumes the historical `yolo11s_airborne_drone_vs_bird_v2` checkpoint;
  - auto-locates the training root under the repo or `.claude/worktrees`;
  - defaults to `-Workers 8` for full-speed GPU feeding;
  - use `-Workers 0` only as a slow fallback if Windows dataloader workers
    fail.

- `scripts/extract_miss_highlights.py`
  - reads the eval report, finds every missed positive frame, and renders
    per-frame JPEGs and a slow-motion MP4 with GT (green) and YOLO
    low-conf detections (yellow). Use it to triage detector-vs-tracker
    failures.

## Commands

Build the local sparse-GT sanity dataset only:

```powershell
.\.venv_train\Scripts\python.exe scripts/prepare_airborne_training_set.py `
  --manifest configs/training/airborne_dataset_manifest.yaml `
  --out-dir data/training/local_sparse_gt_yolo `
  --source local_my_drone_chase_sparse_gt `
  --link-mode copy
```

Build the real training set (AOD-4 + local sparse GT):

```powershell
.\.venv_train\Scripts\python.exe scripts/prepare_airborne_training_set.py `
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
.\.venv_train\Scripts\python.exe scripts/train_airborne_yolo.py `
  --config configs/training/airborne_yolo11.yaml
```

Train with the dedicated training environment:

```powershell
.\.venv_train\Scripts\python.exe scripts/train_airborne_yolo.py `
  --config configs/training/airborne_yolo11.yaml `
  --workers 8 `
  --batch 8
```

Resume an interrupted run from `last.pt`:

```powershell
.\.venv_train\Scripts\python.exe -c "from ultralytics import YOLO; YOLO(r'data\training\runs\yolo11s_airborne_drone_vs_bird_v1\weights\last.pt').train(resume=True)"
```

Resume the current v2 run on Ali's Windows workstation:

```powershell
cd "C:\Users\Ali\Desktop\SKY"
powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\resume_yolo_v2_training.ps1" -Workers 8
```

If Windows blocks dataloader multiprocessing, run the same command with
`-Workers 0`. That is a stability mode, not the performance target.

Monitor with:

```powershell
nvidia-smi -l 5
```

Task Manager can under-report CUDA training because its default graph often
shows 3D/video engines. For this project, trust YOLO's `CUDA:0` startup line,
the `nvidia-smi` process table, GPU memory, and batch throughput. If VRAM is
low on a 16 GB card and utilization is low, increase dataloader workers first,
then test larger batch sizes (`12`, then `16`) if memory allows.

Evaluate the trained detector with corrected GT:

```powershell
.\.venv_train\Scripts\python.exe scripts/run_pipeline.py `
  --video data/videos/my_drone_chase.MP4 `
  --config configs/trained_yolo11s_eval.yaml `
  --gt path\to\drone_sparse_gt_corrected.csv `
  --output data/outputs/run_trained_detector_eval
```

Render a miss-frame diagnostic reel after an eval run:

```powershell
.\.venv_train\Scripts\python.exe scripts/extract_miss_highlights.py
```

## Next Engineering Steps

### Immediate: V3 Domain-Adaptation Fine-tune

**Problem diagnosed (2026-05-02):**
The v2 model passes the acceptance gate on `my_drone_chase.MP4` at the
geometric level, but replay on a new video showed the drone is consistently
labelled `airplane`, not `drone`. The hard-case eval confirms this:

- Geometric detector hit rate at conf=0.25: **42.6 %** (box lands on target)
- Semantic drone hit rate at conf=0.25: **1.9 %** (box lands on target AND
  class = drone)

The model sees the object. It just calls it the wrong class. The cause is a
domain gap: at medium-to-long range, a multirotor quadcopter is a small dark
blob against sky — visually indistinguishable from small fixed-wing aircraft
in the AOD-4 training distribution. The model has never seen *this specific
drone*, in *this lighting and range*, labelled as `drone`.

**Fix: fine-tune on local annotated data.**

The corrected annotation packet already contains the evidence the model needs:

```
annotations/camera_20260423_113401_turn_review_strict/drone_sparse_gt_corrected.csv
```

#### Step 1 — Prepare the fine-tune dataset

Add a new source entry `camera_hard_case_corrected_gt` to the dataset
manifest pointing at the corrected CSV + video. The dataset builder already
supports the `skyscouter_sparse_gt_video` source type which reads this format.

Extract two groups:
- **Drone-positive frames** — rows where `label=drone`; convert pixel boxes
  to YOLO class 0 (drone).
- **Hard-negative frames** — rows where `label=negative`; write empty label
  files (background images, no objects).

This yields ~54 high-value domain-specific frames.

Mix with ~2 000 randomly sampled frames from the existing v2 training set
(AOD-4 + Anti-UAV300) to prevent catastrophic forgetting of birds,
helicopters, and generic airplanes. Target mix ratio roughly 1:40
(hard-case : existing). Output dir: `data/training/airborne_yolo_v3_finetune`.

#### Step 2 — Fine-tune config

Create `configs/training/airborne_yolo11_v3_finetune.yaml`:

| Setting | Value | Reason |
|---|---|---|
| `model` | `data/models/yolo11s_airborne_aod4_antiuav300_v2/best.pt` | Start from v2, not COCO |
| `epochs` | 40 | Small dataset; 80 risks overfitting |
| `lr0` | 0.001 | 10× lower than initial training (fine-tune rate) |
| `freeze` | 10 | Freeze first 10 backbone layers; update neck + head only |
| `imgsz` | 1024 | Match v2 training resolution |
| `batch` | 8 | Same as v2 |
| `run_name` | `yolo11s_airborne_aod4_antiuav300_camhard_v3` | Clear lineage |

Do **not** use `resume=True`. This is a new fine-tune run starting from
`best.pt`, not a continuation of the v2 run.

#### Step 3 — Run fine-tune

```powershell
cd "C:\Users\Ali\Desktop\SKY"
.\.venv_train\Scripts\python.exe scripts/train_airborne_yolo.py `
  --config configs/training/airborne_yolo11_v3_finetune.yaml `
  --workers 8 `
  --batch 8
```

Expected time on RTX 5070 Ti: **30–60 minutes** (40 epochs, small dataset).

#### Step 4 — Evaluate

Run against the hard-case packet and report **both** metrics:

```powershell
.\.venv_train\Scripts\python.exe scripts/evaluate_hard_case.py `
  --csv annotations/camera_20260423_113401_turn_review_strict/drone_sparse_gt_corrected.csv `
  --config configs/trained_yolo11s_v3_eval.yaml `
  --output data/outputs/hard_case_v3_finetune
```

Target: semantic drone hit rate ≥ 50 % at conf=0.25 (from current 1.9 %).
Also verify geometric hit rate does not regress below v2 (42.6 %).
False locks on negative frames must remain 0.

#### Step 5 — Promote and name

```
data/models/yolo11s_airborne_aod4_antiuav300_camhard_v3/best.pt
data/models/yolo11s_airborne_aod4_antiuav300_camhard_v3/last.pt
```

Config `model_version` string: `yolo11s_airborne_aod4_antiuav300_camhard_v3_e40_final`

#### Risk and fallback

If 54 frames does not move semantic hit rate meaningfully (< 20 % after
fine-tune), the next lever is annotating more frames from the same video —
especially frames 96–105 which are true detector misses even at conf=0.01.
Those frames require more varied drone appearances at that pixel size and
should be prioritised for the annotation session.

---

### Remaining Forward-Looking Items

These are not gate blockers for the current mission video but are needed
before the system is robust on arbitrary new footage:

1. **Frames 96–105 coverage** — true detector blindness even at conf ≥ 0.01.
   Requires additional long-range / small-pixel drone training data, or
   targeted annotation of those frames and nearby context.
2. **Obtain LRDDv2** (or equivalent long-range drone data) and enable the
   `lrddv2_long_range_drone` source in the manifest. Same pixel-tiny regime
   as the hard miss frames.
3. **Tracker regression set** — capture the 6 previously-failing tracker
   miss frames (6000, 6120, 6300, 6495, 6675, plus context) as a named
   regression slice and wire a unit test against them.
4. **Tighten lock semantics** so `guidance_valid=true` requires semantic
   drone confidence ≥ a calibrated threshold, not just label match against
   `acceptable_lock_labels`.
5. **Confidence calibration** on validation set (Brier score / temperature
   scaling) — tightens the link between detector confidence and downstream
   lock-quality scoring.

## Acceptance Gate

Do not call the video acceptable until all are true:

- matched positive rate ≥ 0.90;
- median center error ≤ 25 px;
- labeled track ID switches ≤ 3;
- false-lock negative frames = 0.

**Current status (v2 + tracker fix on `my_drone_chase.MP4`): all 4 gates
pass.** Matched positive rate 0.977, median center error 1.89 px, labeled
track ID switches 2, false-lock negative frames 0. See
[`PHASE_1_RESULTS.md`](PHASE_1_RESULTS.md) for the full breakdown.
