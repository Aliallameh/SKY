# Model Artifacts

This folder is for local trained/evaluation checkpoints used by Skyscouter
configs.

Current expected layout:

```text
data/models/
  yolo11s_airborne_drone_vs_bird_v1/
    best.pt
    last.pt
  yolo11s_airborne_aod4_antiuav300_v2/
    best.pt
    last.pt
  yolo11s_airborne_stage2_multiclass_capped_aod4conf_b16w4_nomix/
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

- curated model: `yolo11s_airborne_aod4_antiuav300_v2`
- historical training run: `yolo11s_airborne_drone_vs_bird_v2`
- epochs: 80
- validation: `mAP50=0.979`, `mAP50-95=0.693`
- drone class: `P=0.986`, `R=0.966`, `mAP50=0.990`, `mAP50-95=0.683`

Stage 2 capped multiclass review checkpoint:

- curated model: `yolo11s_airborne_stage2_multiclass_capped_aod4conf_b16w4_nomix`
- source run: `yolo11s_airborne_stage2_multiclass_capped_aod4conf_b16w4_nomix`
- classes: `0=drone`, `1=bird`, `2=airplane`, `3=helicopter`
- epochs: 80
- validation: `mAP50=0.941`, `mAP50-95=0.625`
- status: review candidate for V2 comparison on Jetson, not promoted as a
  flight/safety model
