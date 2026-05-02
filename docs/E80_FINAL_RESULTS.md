# YOLO11s v2 Epoch-80 Final Results

Last updated: 2026-05-02

## Training Result

Run:

```text
yolo11s_airborne_drone_vs_bird_v2
```

Final validation metrics from the 80-epoch Ultralytics run:

| Scope | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| all classes | 0.958 | 0.955 | 0.979 | 0.693 |
| drone | 0.986 | 0.966 | 0.990 | 0.683 |
| bird | 0.965 | 0.951 | 0.979 | 0.733 |
| airplane | 0.939 | 0.949 | 0.971 | 0.709 |
| helicopter | 0.943 | 0.954 | 0.975 | 0.648 |

The final `best.pt` and `last.pt` were promoted to:

```text
data/models/yolo11s_airborne_drone_vs_bird_v2/
```

Configs now identify this checkpoint as:

```text
yolo11s_airborne_drone_vs_bird_v2_e80_final
```

## Hard-Case Result

Hard-case packet:

```text
annotations/camera_20260423_113401_turn_review_strict/drone_sparse_gt_corrected.csv
```

At mission threshold `conf=0.25`:

| Metric | Result |
|---|---:|
| geometric detector hit rate | 23 / 54 = 42.6% |
| semantic drone detector hit rate | 1 / 54 = 1.9% |
| geometric tracker hit rate | 29 / 54 = 53.7% |
| semantic drone tracker hit rate | 0 / 54 = 0.0% |
| negative false positives | 0 |

At diagnostic threshold `conf=0.01`:

| Metric | Result |
|---|---:|
| geometric detector hit rate | 43 / 54 = 79.6% |
| semantic drone detector hit rate | 2 / 54 = 3.7% |
| geometric tracker hit rate | 44 / 54 = 81.5% |
| semantic drone tracker hit rate | 0 / 54 = 0.0% |
| negative false positives | 0 |

Interpretation:

- The model often sees the object as an airborne target.
- It usually classifies this local drone as `airplane`, not `drone`.
- Frames `96-105` remain true hard misses even at very low confidence.
- The tracker stale-following bug is controlled on negative frames.
- The current final model is not yet safe as a drone-only guidance model on this clip.

## Overlay Run

Full-video replay:

```text
data/outputs/run_v2_guidance_camera_20260423_113401_e80_final/annotated.mp4
```

Run summary:

- frames: 344
- detections: 193
- tracks created: 13
- lock states: `SEARCHING=148`, `ACQUIRED=28`, `TRACKING=151`, `LOST=17`
- `guidance_valid=false` for every frame

That is the correct safety behavior for this model: the lock pipeline should not mark guidance valid when semantic drone evidence is weak or misclassified.

## Next ML Step

Do not tune the tracker first. The blocker is detector domain adaptation.

Recommended v3:

1. Add the strict `camera_20260423_113401` corrected packet as drone-positive and hard-negative training data.
2. Include nearby frames around `96-105`, not just the current sparse boxes.
3. Fine-tune from `data/models/yolo11s_airborne_drone_vs_bird_v2/best.pt`, not from COCO.
4. Keep this same packet as a named regression slice and report both geometric hit rate and semantic drone hit rate.
5. Only after semantic drone recall improves should tracker thresholds be relaxed.
