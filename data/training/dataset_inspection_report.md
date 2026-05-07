# SkyScouter Airborne Dataset Inspection

This is the first gate for V3/V4 training. It records what the raw datasets actually contain before conversion.

## Taxonomy

- Stage 1: `0 drone`
- Stage 2/3: `0 drone`, `1 bird`, `2 airplane`, `3 helicopter`
- `unknown_airborne` is not a YOLO training class unless explicit labelled boxes are discovered.

## Baseline Requirement

- local hard-case GT with matched-GT semantic confusion and bbox-size recall
- corrected my_drone_chase sparse GT acceptance packet
- fresh Video_1-Video_5 proxy metrics; still proxy-only until sparse GT exists
- held-out VisioDECT Mavic-like validation slice after it is created

## Dataset Findings

### Anti-UAV-RGBT (`anti_uav_rgbt`)

- Root: `C:\Users\Ali\Desktop\SKY\DATASETS\1_Anti-UAV-RGBT`
- Status: `present`
- Images: 0
- Videos: 636
- Annotation files: 639
- Annotation format: JSON sequence annotations with exist[] and gt_rect[] plus visible/infrared videos
- Has boxes: `True`
- Sequence/video IDs: `True`
- Recommended converter: `scripts/datasets/convert_anti_uav_rgbt.py`

**Class Remapping**

- `visible target / UAV` -> `0 drone`

### DUT-Anti-UAV (`dut_anti_uav`)

- Root: `C:\Users\Ali\Desktop\SKY\DATASETS\2_DUT-Anti-UAV`
- Status: `present`
- Images: 34804
- Videos: 0
- Annotation files: 10040
- Annotation format: Detection VOC XML in train/val/test folders or zips; Tracking V0 frame folders with *_gt.txt rows in x y w h format.
- Has boxes: `True`
- Sequence/video IDs: `Detection image groups plus Tracking V0 sequence IDs (20 sequences).`
- Recommended converter: `Detection: scripts/datasets/convert_dut_anti_uav.py; Tracking: scripts/datasets/convert_dut_anti_uav_tracking.py`

**Class Remapping**

- `UAV` -> `0 drone`
- `uav` -> `0 drone`
- `drone` -> `0 drone`
- `Tracking V0 positive x y w h` -> `0 drone`
- `Tracking V0 -100/non-positive width-height` -> `empty negative label`

**Tracking V0 Discovery**

- Sequences: 20
- Frames / GT rows: 24804 / 24804
- Positive rows: 22218
- Empty/absent rows: 2586
- Frame/GT mismatches: `[]`

### VisioDECT (`visiodect`)

- Root: `C:\Users\Ali\Desktop\SKY\DATASETS\4_VisioDECT Dataset Upload`
- Status: `present`
- Images: 20924
- Videos: 0
- Annotation files: 38869
- Annotation format: Mixed VOC XML, YOLO TXT, CSV/XLSX. Use VOC XML/paired images first when available.
- Has boxes: `True`
- Sequence/video IDs: `True`
- Recommended converter: `scripts/datasets/convert_visiodect.py`

**Class Remapping**

- `all drone model folders/classes` -> `0 drone`

**Mavic-Like Discovery**

- Folder matches: `{'Mavic_Air': 3519, 'Mavic_Enterprise': 3429}`
- Label matches: `{'MAVIC_Air_Cloudy': 875, 'Mavic_Air_Cloudy': 290, 'MAVIC_Air_Evening': 867, 'Mavic_Air_Evening': 296, 'MAVIC_Air_Sunny': 891, 'Mavic_Air_Sunny': 300, 'MAVIC2_Enterprise_Dual_Cloudy': 1113, 'Mavic_Enterprise_Cloudy': 298, 'MAVIC2_Enterprise_Dual_Evening': 627, 'Mavic_Enterprise_Evening': 281, 'MAVIC2_Enterprise_Dual_Sunny': 819, 'Mavic_Enterprise_Sunny': 291}`
- Top-level Mavic-like folders: `['Mavic_Air', 'Mavic_Enterprise']`

### AOD-4 (`aod4`)

- Root: `C:\Users\Ali\Desktop\SKY\DATASETS\5_AOD 4 Dataset for Air Borne Object Detection`
- Status: `present`
- Images: 22516
- Videos: 0
- Annotation files: 67551
- Annotation format: COCO/VOC/YOLOv8/TF formats available; use YOLOv8 labels with explicit class map.
- Has boxes: `True`
- Sequence/video IDs: `False`
- Recommended converter: `scripts/datasets/convert_aod4.py`

**Class Remapping**

- `0` -> `airplane`
- `1` -> `bird`
- `2` -> `drone`
- `3` -> `helicopter`
- `generic supercategory` -> `skip`

- Required audit: Audit AOD-4 drone vs airplane samples before Stage 2 full training; generic category 0 in COCO is not a training class.

### Drone-vs-Bird classification dataset (`drone_vs_bird`)

- Root: `C:\Users\Ali\Desktop\SKY\DATASETS\3_Drone vs Bird Aerial Object Classification Dataset`
- Status: `present`
- Images: 4106
- Videos: 0
- Annotation files: 0
- Annotation format: Classification folders unless annotation files are discovered.
- Has boxes: `False`
- Sequence/video IDs: `False`
- Recommended converter: `scripts/datasets/convert_drone_vs_bird.py`

**Class Remapping**

- `drone folder` -> `crop-level drone only`
- `bird folder` -> `crop-level bird only`

- Decision: Do not mix directly into YOLO detection training; classification-only source.

## Gates

- 20-sample conversion must be validated and previewed before full conversion.
- Full Stage 1/2 training must wait for conversion summaries, validation reports, preview review, and the AOD-4 visual audit for Stage 2.
- Stage 1 is diagnostic/pretraining only and must not be promoted.
