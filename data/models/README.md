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

The curated `.pt` files under `data/models/` are tracked with Git LFS. Random
checkpoints elsewhere remain ignored so the repository does not fill up with
raw training outputs.

Training runs may still write raw outputs under `data/training/runs/`; copy the
checkpoint chosen for evaluation or replay into this folder and point configs
here.

Current v2 final checkpoint:

- run: `yolo11s_airborne_drone_vs_bird_v2`
- epochs: 80
- validation: `mAP50=0.979`, `mAP50-95=0.693`
- drone class: `P=0.986`, `R=0.966`, `mAP50=0.990`, `mAP50-95=0.683`
