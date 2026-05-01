# Model Artifacts

This folder is for local trained/evaluation checkpoints used by Skyscouter
configs.

Current expected layout:

```text
data/models/
  yolo11s_airborne_drone_vs_bird_v1/
    best.pt
    last.pt
  yolo11s_airborne_drone_vs_bird_v2/
    best.pt
    last.pt
```

The `.pt` files are intentionally ignored by git because they are large binary
artifacts. Keep the README tracked so the folder purpose is clear after a fresh
clone.

Training runs may still write raw outputs under `data/training/runs/`; copy the
checkpoint chosen for evaluation or replay into this folder and point configs
here.
