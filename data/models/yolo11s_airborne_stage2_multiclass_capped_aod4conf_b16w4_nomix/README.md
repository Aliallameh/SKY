# YOLO11s Airborne Stage 2 Multiclass Capped Candidate

Curated review checkpoint for Jetson comparison against the promoted V2 model.

```text
model folder: data/models/yolo11s_airborne_stage2_multiclass_capped_aod4conf_b16w4_nomix
source run:   data/training/runs/yolo11s_airborne_stage2_multiclass_capped_aod4conf_b16w4_nomix
weights:      best.pt, last.pt
classes:      0 drone, 1 bird, 2 airplane, 3 helicopter
imgsz:        1024
epochs:       80
batch:        16
workers:      4
```

Final validation from the training run:

```text
all:       P=0.93896 R=0.91687 mAP50=0.94054 mAP50-95=0.62538
```

Use this model as a review candidate, not a promoted flight model. Compare it
against V2 on the same Jetson live-camera conditions, fresh videos, local
hard-case GT, and false-lock negative clips before promotion.

Relevant configs:

```text
configs/trained_yolo11s_stage2_multiclass_capped_eval.yaml
configs/jetson_live_camera_stage2_multiclass_capped_pytorch.yaml
configs/deploy_jetson_yolo11s_stage2_multiclass_capped_engine.yaml
```
