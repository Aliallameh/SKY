# Next Data Annotation Plan

Last updated: 2026-05-04

## Why This Matters

The v2 model often tracks the local drone geometrically but calls it `airplane`. V3 needs local Mavic-style drone frames labelled as `drone`, plus hard negatives, before Stage 3 fine-tuning can be trusted.

## Required Local Annotation Folder

```text
annotations/mavic_style_local_v3/
```

Use YOLO detection format for training exports:

```text
images/train, images/val, images/test
labels/train, labels/val, labels/test
data.yaml
```

Classes:

```text
0 drone
1 bird
2 airplane
3 helicopter
```

For true negatives, create an empty `.txt` label file.

## Minimum Before Stage 3

Label at least 100-200 local frames before Stage 3.

Priority clips:

| Source | What to label |
|---|---|
| `Video_1` | lockable sections where v2 tracks but labels airplane |
| `Video_3` | partial tracking sections |
| `Video_5` | lock islands and nearby misses |
| `Video_2` | visible miss sections if the drone can be seen |
| `Video_4` | visible miss sections if the drone can be seen |
| hard negatives | birds, aircraft, clouds, empty sky, edge exits |

## Before Sandbox Readiness

Expand to 300-1,000 labelled local frames.

Sandbox readiness also requires:

- Fresh `Video_1` to `Video_5` sparse GT annotated and evaluated.
- Calibrated camera path.
- Target-hardware latency benchmark.
- Semantic-safe lock validation.
- No flight-safety claim from airborne-review mode.

## Labelling Rules

- Real local drone target: `0 drone`.
- Bird: `1 bird`.
- Airplane: `2 airplane`.
- Helicopter: `3 helicopter`.
- Empty sky or no visible target: empty label file.
- Do not label ambiguous generic airborne blobs as `unknown_airborne`; that class is not part of V3 YOLO training.
- Edge-exit frames should be negative if the object is no longer boxable.
- Tiny visible drones should still be boxed if a human can confidently locate the object.

## Sampling Strategy

Start balanced, not huge:

```text
60-100 local drone positives
20-40 hard negatives
20-40 confusing aircraft/bird/cloud frames
```

Then expand toward:

```text
200-600 local drone positives
50-200 hard negatives
50-200 bird/aircraft/cloud negatives
```

## Acceptance Metrics For Local Data

The next model must improve over v2 on:

```text
local semantic drone hit rate
Mavic-like drone->airplane confusion
false-lock negative frames
drone recall by bbox size
```

V2 local hard-case baseline at confidence `0.25`:

| Metric | Value |
|---|---:|
| Detector geometric hit rate | 42.59% |
| Detector semantic drone hit rate | 1.85% |
| matched as drone | 1 |
| matched as airplane | 22 |
| missed | 31 |
| negative false positives | 0 |

## Practical Review Workflow

1. Generate or open annotation packets for the fresh videos.
2. Label the drone as `drone`, not `airplane`.
3. Include negatives near the same scenes, especially when the drone exits frame or clouds/aircraft are present.
4. Export sparse GT CSVs and YOLO-format training folders.
5. Validate:

```powershell
.\.venv_train\Scripts\python.exe scripts\validate_yolo_dataset.py --data annotations\mavic_style_local_v3\data.yaml
```

6. Preview:

```powershell
.\.venv_train\Scripts\python.exe scripts\preview_yolo_labels.py `
  --data annotations\mavic_style_local_v3\data.yaml `
  --out-dir data\training\previews\mavic_style_local_v3
```

7. Only after review, build Stage 3.
