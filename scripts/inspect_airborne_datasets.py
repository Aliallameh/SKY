"""Inspect raw airborne datasets before any V3/V4 conversion or training.

This script is deliberately evidence-first. It does not mutate raw datasets and
it does not silently bless a source for training. It writes a JSON + Markdown
report with detected annotation formats, class names, remapping tables, and
Mavic-like VisioDECT names discovered from actual folders/labels.
"""
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from collections import Counter, defaultdict
from itertools import islice
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.datasets.common_yolo import IMAGE_EXTS, VIDEO_EXTS, normalize_label, read_voc_boxes  # noqa: E402


RAW_DEFAULTS = {
    "anti_uav_rgbt": Path("DATASETS/1_Anti-UAV-RGBT"),
    "dut_anti_uav": Path("DATASETS/2_DUT-Anti-UAV"),
    "drone_vs_bird": Path("DATASETS/3_Drone vs Bird Aerial Object Classification Dataset"),
    "visiodect": Path("DATASETS/4_VisioDECT Dataset Upload"),
    "aod4": Path("DATASETS/5_AOD 4 Dataset for Air Borne Object Detection"),
}

FINAL_STAGE2_CLASSES = ["drone", "bird", "airplane", "helicopter"]
MAVIC_TOKENS = ("mavic", "mavic_air", "mavic_enterprise", "dji_mavic", "mavic_2", "enterprise")
MAX_FULL_XML_PARSE_PER_DATASET = 3000
MAX_YOLO_TXT_SCAN = 3000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect raw SkyScouter airborne datasets.")
    parser.add_argument("--datasets-root", default="DATASETS", help="Parent folder containing raw dataset folders.")
    parser.add_argument("--out-json", default="data/training/dataset_inspection_report.json")
    parser.add_argument("--out-md", default="data/training/dataset_inspection_report.md")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path.cwd()
    datasets_root = _resolve(repo_root, args.datasets_root)
    roots = {name: _resolve(repo_root, path) for name, path in RAW_DEFAULTS.items()}
    if datasets_root.name != "DATASETS":
        roots = {name: datasets_root / path.name for name, path in RAW_DEFAULTS.items()}

    report: Dict[str, Any] = {
        "schema_version": "skyscout.dataset_inspection.v1",
        "repo_root": str(repo_root.resolve()),
        "datasets_root": str(datasets_root),
        "target_taxonomy": {
            "stage1": ["drone"],
            "stage2_stage3": FINAL_STAGE2_CLASSES,
            "unknown_airborne_policy": "Not a YOLO training class unless explicit labelled boxes are discovered; generic/supercategory boxes are skipped.",
        },
        "datasets": {},
        "gates": {
            "converter_gate": "PASS only after this report contains remapping tables and no unsupported critical annotation formats.",
            "training_gate": "Do not run full 80-epoch training until 20-sample conversions validate and previews are visually reviewed.",
        },
        "baseline_plan": build_baseline_plan(),
    }
    report["datasets"]["anti_uav_rgbt"] = inspect_anti_uav_rgbt(roots["anti_uav_rgbt"])
    report["datasets"]["dut_anti_uav"] = inspect_dut_anti_uav(roots["dut_anti_uav"])
    report["datasets"]["visiodect"] = inspect_visiodect(roots["visiodect"])
    report["datasets"]["aod4"] = inspect_aod4(roots["aod4"])
    report["datasets"]["drone_vs_bird"] = inspect_drone_vs_bird(roots["drone_vs_bird"])

    out_json = _resolve(repo_root, args.out_json)
    out_md = _resolve(repo_root, args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    out_md.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"json": str(out_json), "markdown": str(out_md)}, indent=2))
    return 0


def _resolve(repo_root: Path, value: Path | str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def ext_counts(root: Path, limit: Optional[int] = None) -> Counter:
    counts: Counter = Counter()
    if not root.exists():
        return counts
    for idx, path in enumerate(root.rglob("*")):
        if path.is_file():
            counts[path.suffix.lower() or "<none>"] += 1
        if limit is not None and idx >= limit:
            break
    return counts


def sample_files(root: Path, patterns: Iterable[str], limit: int = 5) -> List[str]:
    samples: List[str] = []
    if not root.exists():
        return samples
    for pattern in patterns:
        for path in sorted(root.rglob(pattern)):
            if path.is_file():
                samples.append(str(path.relative_to(root)))
                if len(samples) >= limit:
                    return samples
    return samples


def inspect_anti_uav_rgbt(root: Path) -> Dict[str, Any]:
    result = base_dataset_result(root, "Anti-UAV-RGBT")
    result.update(
        {
            "annotation_format": "JSON sequence annotations with exist[] and gt_rect[] plus visible/infrared videos",
            "recommended_converter": "scripts/datasets/convert_anti_uav_rgbt.py",
            "stage_policy": "Stage 1 drone-only, RGB/visible only first. Thermal remains a later track.",
            "class_remapping": {"visible target / UAV": "0 drone"},
            "safe_to_include": {"stage1": True, "stage2": "sample/cap as drone data only", "stage3": "only via balanced sample"},
        }
    )
    if not root.exists():
        result["status"] = "missing"
        return result
    sequences: Dict[str, int] = {}
    json_examples: List[Dict[str, Any]] = []
    modality_pairs = Counter()
    for split in ("train", "val", "test"):
        split_dir = root / split
        seq_dirs = [p for p in split_dir.iterdir() if p.is_dir()] if split_dir.exists() else []
        sequences[split] = len(seq_dirs)
        for seq_dir in seq_dirs[:2]:
            for modality in ("visible", "infrared"):
                video = seq_dir / f"{modality}.mp4"
                ann = seq_dir / f"{modality}.json"
                if video.exists() and ann.exists():
                    modality_pairs[modality] += 1
                    if len(json_examples) < 3:
                        try:
                            payload = json.loads(ann.read_text(encoding="utf-8"))
                            rects = payload.get("gt_rect", [])
                            exists = payload.get("exist", [])
                            json_examples.append(
                                {
                                    "path": str(ann.relative_to(root)),
                                    "keys": sorted(payload.keys()),
                                    "frames": len(exists),
                                    "visible_frames": int(sum(1 for v in exists if int(v) == 1)),
                                    "sample_rect": rects[0] if rects else None,
                                }
                            )
                        except Exception as exc:  # pragma: no cover - report only
                            json_examples.append({"path": str(ann.relative_to(root)), "error": str(exc)})
    result["split_sequence_counts"] = sequences
    result["modality_pairs_seen_in_sample"] = dict(modality_pairs)
    result["sample_annotation_examples"] = json_examples
    result["has_bounding_boxes"] = True
    result["has_negative_frames"] = True
    result["sequence_ids_available"] = True
    return result


def inspect_dut_anti_uav(root: Path) -> Dict[str, Any]:
    result = base_dataset_result(root, "DUT-Anti-UAV")
    result.update(
        {
            "annotation_format": "Detection VOC XML in train/val/test folders or zips; Tracking V0 frame folders with *_gt.txt rows in x y w h format.",
            "recommended_converter": "Detection: scripts/datasets/convert_dut_anti_uav.py; Tracking: scripts/datasets/convert_dut_anti_uav_tracking.py",
            "stage_policy": (
                "Stage 1 drone-only from Detection VOC plus optional Tracking V0. "
                "Tracking V0 is especially useful for lock continuity and absent-target negatives."
            ),
            "class_remapping": {
                "UAV": "0 drone",
                "uav": "0 drone",
                "drone": "0 drone",
                "Tracking V0 positive x y w h": "0 drone",
                "Tracking V0 -100/non-positive width-height": "empty negative label",
            },
            "safe_to_include": {
                "stage1": True,
                "stage2": "sample/cap as drone data only",
                "stage3": "useful for temporal drone-positive/absent-target support, but not bird/airplane rejection",
            },
        }
    )
    if not root.exists():
        result["status"] = "missing"
        return result
    zips = sorted(root.rglob("*.zip"))
    result["zip_files"] = [str(p.relative_to(root)) for p in zips]
    voc_names: Counter = Counter()
    zip_examples: List[Dict[str, Any]] = []
    folder_examples: List[Dict[str, Any]] = []
    detection_dir = next((p for p in root.iterdir() if p.is_dir() and "Detection" in p.name), root)
    for split in ("train", "val", "test"):
        split_dir = detection_dir / split
        if not split_dir.exists():
            continue
        xml_paths = sorted(split_dir.rglob("*.xml"))
        image_paths = sorted(p for p in split_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
        example = {"split": split, "xml_count": len(xml_paths), "image_count": len(image_paths)}
        for xml_path in xml_paths[:20]:
            try:
                _w, _h, objects = read_voc_boxes(xml_path)
            except Exception:
                continue
            for name, _bbox in objects:
                voc_names[name] += 1
            if "sample_xml" not in example:
                example["sample_xml"] = str(xml_path.relative_to(root))
        folder_examples.append(example)
    for zip_path in zips:
        if "Detection" not in str(zip_path):
            continue
        try:
            with zipfile.ZipFile(zip_path) as zf:
                xml_names = [name for name in zf.namelist() if name.lower().endswith(".xml")]
                jpg_names = [name for name in zf.namelist() if Path(name).suffix.lower() in IMAGE_EXTS]
                example: Dict[str, Any] = {
                    "zip": str(zip_path.relative_to(root)),
                    "xml_count": len(xml_names),
                    "image_count": len(jpg_names),
                }
                for xml_name in xml_names[:20]:
                    text = zf.read(xml_name).decode("utf-8", errors="replace")
                    for name in extract_xml_object_names(text):
                        voc_names[name] += 1
                    if "sample_xml" not in example:
                        example["sample_xml"] = xml_name
                zip_examples.append(example)
        except zipfile.BadZipFile:
            zip_examples.append({"zip": str(zip_path.relative_to(root)), "error": "bad_zip"})
    result["detected_class_names"] = dict(voc_names)
    result["sample_annotation_examples"] = folder_examples + zip_examples
    result["has_bounding_boxes"] = bool(voc_names or zip_examples)
    tracking = inspect_dut_tracking_split(root)
    result["tracking_v0"] = tracking
    if tracking.get("present"):
        result["has_bounding_boxes"] = True
        result["has_negative_images"] = tracking.get("negative_rows", 0) > 0
        result["sequence_ids_available"] = f"Detection image groups plus Tracking V0 sequence IDs ({tracking.get('sequence_count', 0)} sequences)."
    else:
        result["has_negative_images"] = "unknown"
        result["sequence_ids_available"] = "Detection image groups only; Tracking V0 not discovered."
    return result


def inspect_dut_tracking_split(root: Path) -> Dict[str, Any]:
    tracking_root = root / "Tracking (IEEE-TITS)"
    frames_dir = tracking_root / "Anti-UAV-Tracking-V0" / "Anti-UAV-Tracking-V0"
    gt_dir = tracking_root / "Anti-UAV-Tracking-V0GT" / "Anti-UAV-Tracking-V0GT"
    result: Dict[str, Any] = {
        "present": frames_dir.exists() and gt_dir.exists(),
        "frames_dir": str(frames_dir),
        "gt_dir": str(gt_dir),
        "format": "one *_gt.txt file per video sequence; each row aligns to frame N and stores x y w h; -100 rows mean target absent",
    }
    if not result["present"]:
        return result
    sequence_summaries = []
    total_frames = 0
    total_rows = 0
    positive_rows = 0
    negative_rows = 0
    mismatches = []
    for gt_path in sorted(gt_dir.glob("*_gt.txt")):
        seq_name = gt_path.stem.replace("_gt", "")
        seq_dir = frames_dir / seq_name
        frame_count = len(list(seq_dir.glob("*.jpg"))) if seq_dir.exists() else 0
        rows = [line.strip() for line in gt_path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
        seq_pos = 0
        seq_neg = 0
        for row in rows:
            parts = row.split()
            if len(parts) != 4:
                seq_neg += 1
                continue
            try:
                _x, _y, w, h = (float(part) for part in parts)
            except ValueError:
                seq_neg += 1
                continue
            if w > 0 and h > 0:
                seq_pos += 1
            else:
                seq_neg += 1
        if frame_count != len(rows):
            mismatches.append({"sequence": seq_name, "frames": frame_count, "gt_rows": len(rows)})
        total_frames += frame_count
        total_rows += len(rows)
        positive_rows += seq_pos
        negative_rows += seq_neg
        sequence_summaries.append({"sequence": seq_name, "frames": frame_count, "gt_rows": len(rows), "positive_rows": seq_pos, "negative_rows": seq_neg})
    result.update(
        {
            "sequence_count": len(sequence_summaries),
            "frame_count": total_frames,
            "gt_rows": total_rows,
            "positive_rows": positive_rows,
            "negative_rows": negative_rows,
            "frame_gt_mismatches": mismatches,
            "sample_sequences": sequence_summaries[:5],
        }
    )
    return result


def inspect_visiodect(root: Path) -> Dict[str, Any]:
    result = base_dataset_result(root, "VisioDECT")
    result.update(
        {
            "annotation_format": "Mixed VOC XML, YOLO TXT, CSV/XLSX. Use VOC XML/paired images first when available.",
            "recommended_converter": "scripts/datasets/convert_visiodect.py",
            "stage_policy": "Stage 1 drone-only. Collapse drone model classes to 0 drone. Keep Mavic-like validation held out.",
            "class_remapping": {"all drone model folders/classes": "0 drone"},
            "safe_to_include": {"stage1": "non-held-out drone samples only", "stage2": "sample/cap as drone data only", "stage3": "only via balanced sample"},
        }
    )
    if not root.exists():
        result["status"] = "missing"
        return result
    top_dirs = [p.name for p in sorted(root.iterdir()) if p.is_dir()]
    result["top_level_folders"] = top_dirs
    xml_names: Counter = Counter()
    yolo_ids: Counter = Counter()
    csv_labels: Counter = Counter()
    mavic_paths: Counter = Counter()
    mavic_labels: Counter = Counter()
    xml_examples: List[Dict[str, Any]] = []
    parsed_xml = 0
    for xml_path in sorted(root.rglob("*.xml")):
        path_is_mavic = is_mavic_like(str(xml_path.relative_to(root)))
        if not path_is_mavic and parsed_xml >= MAX_FULL_XML_PARSE_PER_DATASET:
            continue
        try:
            _width, _height, objects = read_voc_boxes(xml_path)
        except Exception:
            continue
        parsed_xml += 1
        for label, _bbox in objects:
            xml_names[label] += 1
            if is_mavic_like(label):
                mavic_labels[label] += 1
        rel = str(xml_path.relative_to(root))
        if path_is_mavic:
            mavic_paths[rel.split("\\")[0].split("/")[0]] += 1
        if len(xml_examples) < 5 and objects:
            xml_examples.append({"path": rel, "objects": [{"name": objects[0][0], "bbox": objects[0][1]}]})
    result["xml_files_parsed_for_class_discovery"] = parsed_xml
    for txt_path in islice(sorted(root.rglob("*.txt")), MAX_YOLO_TXT_SCAN):
        try:
            for line in txt_path.read_text(encoding="utf-8", errors="replace").splitlines():
                parts = line.split()
                if len(parts) >= 5:
                    yolo_ids[parts[0]] += 1
        except OSError:
            continue
    for csv_path in sorted(root.rglob("*.csv"))[:30]:
        try:
            for idx, line in enumerate(csv_path.read_text(encoding="utf-8", errors="replace").splitlines()):
                if idx == 0:
                    continue
                parts = [p.strip() for p in line.split(",")]
                if parts:
                    csv_labels[parts[0]] += 1
        except OSError:
            continue
    discovered_mavic = {
        "folder_matches": dict(mavic_paths),
        "label_matches": dict(mavic_labels),
        "all_path_tokens": sorted({p for p in top_dirs if is_mavic_like(p)}),
    }
    result["detected_class_names"] = {
        "voc_xml_names": dict(xml_names),
        "yolo_class_ids_sample": dict(yolo_ids),
        "csv_labels_sample": dict(csv_labels),
    }
    result["mavic_like_discovery"] = discovered_mavic
    result["sample_annotation_examples"] = xml_examples
    result["has_bounding_boxes"] = bool(xml_names or yolo_ids or csv_labels)
    result["has_negative_images"] = "unknown"
    result["sequence_ids_available"] = True
    result["mavic_validation_policy"] = (
        "If enough Mavic-like scenario folders/images exist, split by scenario into train and held-out validation; "
        "otherwise keep all Mavic-like VisioDECT samples held out and train from local Mavic-style annotations."
    )
    return result


def inspect_aod4(root: Path) -> Dict[str, Any]:
    result = base_dataset_result(root, "AOD-4")
    result.update(
        {
            "annotation_format": "COCO/VOC/YOLOv8/TF formats available; use YOLOv8 labels with explicit class map.",
            "recommended_converter": "scripts/datasets/convert_aod4.py",
            "stage_policy": "Stage 2 multiclass after visual AOD-4 drone-vs-airplane audit, especially tiny boxes.",
            "class_remapping": {"0": "airplane", "1": "bird", "2": "drone", "3": "helicopter", "generic supercategory": "skip"},
            "safe_to_include": {"stage1": False, "stage2": "after preview/audit", "stage3": "balanced sample only"},
        }
    )
    if not root.exists():
        result["status"] = "missing"
        return result
    aod_root = find_aod_inner_root(root)
    result["discovered_root"] = str(aod_root) if aod_root else str(root)
    coco_categories: Dict[str, Any] = {}
    for coco_path in sorted(root.rglob("*.json")):
        if "coco" not in coco_path.name.lower() and "_annotations" not in coco_path.name.lower():
            continue
        try:
            payload = json.loads(coco_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        cats = payload.get("categories", [])
        if cats:
            coco_categories[str(coco_path.relative_to(root))] = cats[:10]
    yolo_ids: Counter = Counter()
    for txt_path in islice(sorted(root.rglob("*.txt")), MAX_YOLO_TXT_SCAN):
        try:
            for line in txt_path.read_text(encoding="utf-8", errors="replace").splitlines():
                parts = line.split()
                if len(parts) >= 5:
                    yolo_ids[parts[0]] += 1
        except OSError:
            continue
    result["detected_class_names"] = {
        "coco_categories_sample": coco_categories,
        "yolo_class_ids_sample": dict(yolo_ids),
    }
    result["sample_annotation_examples"] = sample_files(root, ["*.txt", "*.xml", "*.json"], limit=8)
    result["has_bounding_boxes"] = bool(yolo_ids or coco_categories)
    result["has_negative_images"] = "unknown"
    result["sequence_ids_available"] = False
    result["required_visual_audit"] = "Audit AOD-4 drone vs airplane samples before Stage 2 full training; generic category 0 in COCO is not a training class."
    return result


def inspect_drone_vs_bird(root: Path) -> Dict[str, Any]:
    result = base_dataset_result(root, "Drone-vs-Bird classification dataset")
    result.update(
        {
            "annotation_format": "Classification folders unless annotation files are discovered.",
            "recommended_converter": "scripts/datasets/convert_drone_vs_bird.py",
            "stage_policy": "Later hard-negative mining/crop classifier only if no bounding boxes exist.",
            "class_remapping": {"drone folder": "crop-level drone only", "bird folder": "crop-level bird only"},
            "safe_to_include": {"stage1": False, "stage2": False, "stage3": False},
        }
    )
    if not root.exists():
        result["status"] = "missing"
        return result
    folder_counts: Dict[str, int] = {}
    for child in sorted(root.iterdir()):
        if child.is_dir():
            folder_counts[child.name] = sum(1 for p in child.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    annotation_files = sample_files(root, ["*.xml", "*.json", "*.txt", "*.csv"], limit=10)
    result["detected_class_names"] = {"classification_folders": folder_counts}
    result["sample_annotation_examples"] = annotation_files
    result["has_bounding_boxes"] = bool(annotation_files)
    if not annotation_files:
        result["direct_yolo_training_decision"] = "Do not mix directly into YOLO detection training; classification-only source."
    result["has_negative_images"] = "classification-only"
    result["sequence_ids_available"] = False
    return result


def base_dataset_result(root: Path, display_name: str) -> Dict[str, Any]:
    counts = ext_counts(root)
    return {
        "display_name": display_name,
        "root": str(root),
        "status": "present" if root.exists() else "missing",
        "file_extension_counts": dict(counts),
        "image_count": int(sum(counts[ext] for ext in IMAGE_EXTS)),
        "video_count": int(sum(counts[ext] for ext in VIDEO_EXTS)),
        "annotation_file_count": int(sum(counts[ext] for ext in (".xml", ".json", ".txt", ".csv", ".mat"))),
    }


def extract_xml_object_names(xml_text: str) -> List[str]:
    import xml.etree.ElementTree as ET

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []
    names = []
    for node in root.findall(".//object/name"):
        if node.text:
            names.append(node.text.strip())
    return names


def find_aod_inner_root(root: Path) -> Optional[Path]:
    candidates = [root / "AOD 4", root]
    for candidate in candidates:
        if (candidate / "Images").exists() or (candidate / "Annotations").exists():
            return candidate
    return None


def is_mavic_like(value: object) -> bool:
    normalized = normalize_label(value)
    return any(token in normalized for token in MAVIC_TOKENS)


def build_baseline_plan() -> Dict[str, Any]:
    return {
        "current_v2_model": "data/models/yolo11s_airborne_aod4_antiuav300_v2/best.pt",
        "must_run_before_new_training": [
            "local hard-case GT with matched-GT semantic confusion and bbox-size recall",
            "corrected my_drone_chase sparse GT acceptance packet",
            "fresh Video_1-Video_5 proxy metrics; still proxy-only until sparse GT exists",
            "held-out VisioDECT Mavic-like validation slice after it is created",
        ],
        "safety_note": "Semantic-safe lock metrics and airborne-review geometry metrics must remain separate.",
    }


def render_markdown(report: Dict[str, Any]) -> str:
    lines = [
        "# SkyScouter Airborne Dataset Inspection",
        "",
        "This is the first gate for V3/V4 training. It records what the raw datasets actually contain before conversion.",
        "",
        "## Taxonomy",
        "",
        "- Stage 1: `0 drone`",
        "- Stage 2/3: `0 drone`, `1 bird`, `2 airplane`, `3 helicopter`",
        "- `unknown_airborne` is not a YOLO training class unless explicit labelled boxes are discovered.",
        "",
        "## Baseline Requirement",
        "",
    ]
    for item in report["baseline_plan"]["must_run_before_new_training"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Dataset Findings", ""])
    for name, data in report["datasets"].items():
        lines.extend(
            [
                f"### {data['display_name']} (`{name}`)",
                "",
                f"- Root: `{data['root']}`",
                f"- Status: `{data['status']}`",
                f"- Images: {data.get('image_count', 0)}",
                f"- Videos: {data.get('video_count', 0)}",
                f"- Annotation files: {data.get('annotation_file_count', 0)}",
                f"- Annotation format: {data.get('annotation_format', 'unknown')}",
                f"- Has boxes: `{data.get('has_bounding_boxes', 'unknown')}`",
                f"- Sequence/video IDs: `{data.get('sequence_ids_available', 'unknown')}`",
                f"- Recommended converter: `{data.get('recommended_converter', 'none')}`",
                "",
                "**Class Remapping**",
                "",
            ]
        )
        remap = data.get("class_remapping", {})
        if isinstance(remap, dict):
            for source, target in remap.items():
                lines.append(f"- `{source}` -> `{target}`")
        else:
            lines.append(f"- {remap}")
        if data.get("mavic_like_discovery"):
            lines.extend(["", "**Mavic-Like Discovery**", ""])
            mavic = data["mavic_like_discovery"]
            lines.append(f"- Folder matches: `{mavic.get('folder_matches', {})}`")
            lines.append(f"- Label matches: `{mavic.get('label_matches', {})}`")
            lines.append(f"- Top-level Mavic-like folders: `{mavic.get('all_path_tokens', [])}`")
        if data.get("tracking_v0", {}).get("present"):
            tracking = data["tracking_v0"]
            lines.extend(["", "**Tracking V0 Discovery**", ""])
            lines.append(f"- Sequences: {tracking.get('sequence_count', 0)}")
            lines.append(f"- Frames / GT rows: {tracking.get('frame_count', 0)} / {tracking.get('gt_rows', 0)}")
            lines.append(f"- Positive rows: {tracking.get('positive_rows', 0)}")
            lines.append(f"- Empty/absent rows: {tracking.get('negative_rows', 0)}")
            lines.append(f"- Frame/GT mismatches: `{tracking.get('frame_gt_mismatches', [])}`")
        if data.get("direct_yolo_training_decision"):
            lines.extend(["", f"- Decision: {data['direct_yolo_training_decision']}"])
        if data.get("required_visual_audit"):
            lines.extend(["", f"- Required audit: {data['required_visual_audit']}"])
        lines.append("")
    lines.extend(
        [
            "## Gates",
            "",
            "- 20-sample conversion must be validated and previewed before full conversion.",
            "- Full Stage 1/2 training must wait for conversion summaries, validation reports, preview review, and the AOD-4 visual audit for Stage 2.",
            "- Stage 1 is diagnostic/pretraining only and must not be promoted.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
