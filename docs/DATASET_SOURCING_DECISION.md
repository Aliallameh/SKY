# Dataset Sourcing Decision

Last updated: 2026-05-06

## Decision

AOD-4 is useful, but it is not the best source for SkyScouter drone identity.
It stays in the V3/V4 plan only as a capped airborne confuser/rejection source.

The highest-value data for product improvement is:

1. local Mavic-style footage from Ali's actual capture camera;
2. VisioDECT Mavic-like samples, with held-out validation preserved;
3. Anti-UAV-RGBT and DUT-Anti-UAV as drone-positive support data;
4. AOD-4 for airplane/bird/helicopter rejection, capped and audited;
5. additional public datasets only after bbox/license/domain audit.

## Why AOD-4 Was Used

The local `DATASETS/` folder currently contains only one detection dataset with
explicit `drone`, `bird`, `airplane`, and `helicopter` labels: AOD-4. That makes
it useful for teaching the model that a lockable drone is not the same thing as
every airborne object.

The Stage 1 evaluation proved why confusers are necessary:

| Check | V2 | Stage 1 drone-only |
|---|---:|---:|
| Local hard-case semantic drone hit | 1.85% | 98.15% |
| Held-out Mavic-like drone-to-airplane confusion | 19.14% | 0.00% |
| AOD-4 bird-to-drone false matches | 1 | 280 |
| AOD-4 airplane-to-drone false matches | 0 | 754 |

Stage 1 learned "drone" strongly, but it also started calling many aircraft and
birds `drone`. So the next model still needs rejection data. AOD-4 is kept for
that rejection role only.

## Local Dataset Ranking

| Source | Use Now? | Role | Caveat |
|---|---|---|---|
| Local Video_1 to Video_5 annotations | Yes | Primary product-domain truth | Needs 300-1,000 labelled frames before sandbox-readiness claims |
| VisioDECT | Yes | Mavic-like drone identity and validation | Keep `Mavic_Air_sunny` held out |
| Anti-UAV-RGBT | Yes | Drone-positive support, RGB first | Drone-only; no bird/airplane rejection labels |
| DUT-Anti-UAV | Yes | Drone-positive support | Drone-only; no bird/airplane rejection labels |
| AOD-4 | Limited | Airborne confuser/rejection source | Cap it; exclude AOD-4 drone boxes until audit |
| Drone-vs-Bird local download | No for YOLO | Later crop classifier/hard-negative mining | Current local copy has no detection boxes |

## Public Candidates To Audit Next

These are not automatically approved. Each needs download, license review,
format inspection, 20-sample conversion, validation, and preview.

| Candidate | Why It Might Help | First Use |
|---|---|---|
| Incenda Aerospace Open Dataset | Small aircraft/helicopter/drone boxes, CC BY 4.0 | Extra confusers, especially aircraft/helicopter |
| USC MCL Drone Dataset | Real drone tracking clips with user-labelled boxes | Real small-drone positives and tracker stress |
| Drone-vs-Bird Detection Challenge | Video-level drone boxes; better bird/drone domain if access is granted | Hard drone-vs-bird validation/training |
| AIcrowd Airborne Object Tracking | Huge airborne tracking geometry dataset with tiny boxes | Later geometry/tracking benchmark, not direct drone semantic fix |
| Synthetic Air-to-Air Object Detection | Large synthetic airborne classes | Only after real-data baseline; use carefully to avoid synthetic bias |

Reference pages checked:

- AOD-4: https://data.mendeley.com/datasets/cd5z895tr2/1
- VisioDECT paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC12877808/
- DUT-Anti-UAV: https://github.com/wangdongdut/DUT-Anti-UAV
- AIcrowd AOT: https://www.aicrowd.com/challenges/airborne-object-tracking-challenge
- Incenda Aerospace: https://www.incenda.ai/open-dataset/
- USC MCL Drone Dataset: https://mcl.usc.edu/mcl-drone-dataset/

## Required Stage 2 Build Policy

Build Stage 2 with AOD-4 capped and AOD-4 drone boxes excluded by default:

```powershell
.\.venv_train\Scripts\python.exe scripts\build_staged_airborne_dataset.py `
  --stage stage2 `
  --link-mode copy `
  --cap aod4=6000 `
  --cap anti_uav_rgbt=5000 `
  --cap dut_anti_uav=5000 `
  --cap visiodect=12000
```

The builder skips any image containing an excluded source class, instead of
only dropping that box. This prevents a visible AOD-4 drone from becoming
unlabelled background.

Only after the AOD-4 drone-vs-airplane audit passes may a run use:

```powershell
--no-default-exclusions
```

and the audit evidence must be recorded in `docs/V3_TRAINING_RESULTS.md`.

## Practical Next Move

Do not hunt for endless datasets before fixing the product domain gap. Label
100-200 local frames first, build the capped Stage 2 dataset, evaluate Stage 2
against v2 on local hard-case and held-out Mavic-like validation, then decide
whether Stage 3 starts from Stage 2 best or falls back to v2 best.
