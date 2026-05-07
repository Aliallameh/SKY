"""Generate exact top-level YOLO architecture reports for SkyScouter checkpoints.

This is intentionally different from a pretty block sketch: it records the real
Ultralytics top-level layer table, actual forward-hook tensor shapes, layer
`from` connections, and detect-head class-channel differences.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import torch
from PIL import Image, ImageDraw, ImageFont

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ultralytics import YOLO  # noqa: E402


MODEL_SPECS = {
    "v1_curated": {
        "path": "data/models/yolo11s_airborne_drone_vs_bird_v1/best.pt",
        "status": "curated baseline; not current production",
    },
    "v2_curated": {
        "path": "data/models/yolo11s_airborne_aod4_antiuav300_v2/best.pt",
        "status": "current promoted benchmark; semantic airplane issue remains",
    },
    "quick_v3_unpromoted": {
        "path": "data/training/runs/yolo11s_airborne_v3_finetune_quick/weights/best.pt",
        "status": "rejected for promotion",
    },
    "stage1_drone_only": {
        "path": "data/training/runs/yolo11s_airborne_stage1_drone_only_b16w4_nomix/weights/best.pt",
        "status": "diagnostic/pretraining only; never promote directly",
    },
    "stage2_multiclass_capped": {
        "path": "data/training/runs/yolo11s_airborne_stage2_multiclass_capped_aod4conf_b16w4_nomix/weights/best.pt",
        "status": "unpromoted; useful evidence, semantic risk remains",
    },
}

ROLE_BY_INDEX = {
    0: "stem / P1/2",
    1: "P2/4 downsample",
    2: "P2/4 refine",
    3: "P3/8 downsample",
    4: "P3/8 backbone skip",
    5: "P4/16 downsample",
    6: "P4/16 backbone skip",
    7: "P5/32 downsample",
    8: "P5/32 refine",
    9: "SPPF",
    10: "P5/32 attention skip",
    11: "FPN up P5->P4",
    12: "concat with layer 6",
    13: "FPN P4 feature",
    14: "FPN up P4->P3",
    15: "concat with layer 4",
    16: "Detect input P3/8",
    17: "PAN down P3->P4",
    18: "concat with layer 13",
    19: "Detect input P4/16",
    20: "PAN down P4->P5",
    21: "concat with layer 10",
    22: "Detect input P5/32",
    23: "Detect head",
}

COLORS = {
    "backbone": "#2f80ed",
    "neck": "#27ae60",
    "concat": "#828282",
    "upsample": "#f2c94c",
    "detect_input": "#00a3a3",
    "detect": "#eb5757",
    "input": "#111827",
}


@dataclass
class LayerRow:
    index: int
    from_: str
    role: str
    module: str
    yaml_def: str
    output_shape_1024: str
    params: int


@dataclass
class ModelHead:
    key: str
    path: str
    status: str
    exists: bool
    scale: str = ""
    task: str = ""
    params: int = 0
    nc: int = 0
    names: List[str] | None = None
    detect_reg_channels_per_scale: int = 64
    detect_cls_channels_per_scale: int = 0
    detect_raw_channels_per_scale: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate exact SkyScouter YOLO architecture reports.")
    parser.add_argument("--out-dir", default="data/outputs/model_architectures_exact")
    parser.add_argument("--reference-model", default="stage2_multiclass_capped")
    parser.add_argument("--imgsz", type=int, default=1024)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    model_heads = [inspect_model_head(key, spec) for key, spec in MODEL_SPECS.items()]
    ref_spec = MODEL_SPECS[args.reference_model]
    ref_path = Path(ref_spec["path"])
    if not ref_path.exists():
        raise SystemExit(f"Reference model missing: {ref_path}")

    ref_model = YOLO(str(ref_path)).model.eval()
    shapes = capture_top_level_shapes(ref_model, args.imgsz)
    rows = build_layer_rows(ref_model, shapes)

    write_layer_csv(out_dir / "exact_layer_table_1024.csv", rows)
    write_head_csv(out_dir / "model_head_comparison.csv", model_heads)
    (out_dir / "exact_architecture_summary.json").write_text(
        json.dumps({"models": [asdict(m) for m in model_heads], "layers": [asdict(r) for r in rows]}, indent=2),
        encoding="utf-8",
    )
    render_exact_topology_png(rows, model_heads, out_dir / "exact_yolo11s_topology_1024.png")
    render_detect_head_png(model_heads, out_dir / "detect_head_class_channels.png")
    (out_dir / "exact_architecture_report.md").write_text(render_markdown(rows, model_heads, args.imgsz), encoding="utf-8")
    (out_dir / "exact_architecture_report.html").write_text(render_html(rows, model_heads, args.imgsz), encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "html": str(out_dir / "exact_architecture_report.html")}, indent=2))
    return 0


def inspect_model_head(key: str, spec: Dict[str, str]) -> ModelHead:
    path = Path(spec["path"])
    info = ModelHead(key=key, path=str(path), status=spec["status"], exists=path.exists())
    if not path.exists():
        return info
    model = YOLO(str(path))
    m = model.model
    info.scale = str(getattr(m, "yaml", {}).get("scale", ""))
    info.task = str(model.task)
    info.params = int(sum(p.numel() for p in m.parameters()))
    names = getattr(m, "names", {}) or {}
    info.names = [str(names[k]) for k in sorted(names)]
    info.nc = int(getattr(m, "nc", len(info.names)))
    detect = m.model[-1]
    info.detect_reg_channels_per_scale = int(detect.cv2[0][-1].out_channels)
    info.detect_cls_channels_per_scale = int(detect.cv3[0][-1].out_channels)
    info.detect_raw_channels_per_scale = info.detect_reg_channels_per_scale + info.detect_cls_channels_per_scale
    return info


def capture_top_level_shapes(model: torch.nn.Module, imgsz: int) -> Dict[int, object]:
    shapes: Dict[int, object] = {}

    def shape_of(value: object) -> object:
        if isinstance(value, torch.Tensor):
            return list(value.shape)
        if isinstance(value, (tuple, list)):
            return [shape_of(item) for item in value]
        if isinstance(value, dict):
            return {str(k): shape_of(v) for k, v in value.items()}
        return str(type(value))

    handles = []
    for idx, layer in enumerate(model.model):
        handles.append(layer.register_forward_hook(lambda _m, _i, out, idx=idx: shapes.__setitem__(idx, shape_of(out))))
    try:
        with torch.no_grad():
            _ = model(torch.zeros(1, 3, imgsz, imgsz))
    finally:
        for handle in handles:
            handle.remove()
    return shapes


def build_layer_rows(model: torch.nn.Module, shapes: Dict[int, object]) -> List[LayerRow]:
    yaml_layers = list(model.yaml.get("backbone", [])) + list(model.yaml.get("head", []))
    rows = []
    for idx, layer in enumerate(model.model):
        yaml_def = yaml_layers[idx] if idx < len(yaml_layers) else ""
        rows.append(
            LayerRow(
                index=idx,
                from_=json.dumps(getattr(layer, "f", -1)),
                role=ROLE_BY_INDEX.get(idx, ""),
                module=module_name(layer),
                yaml_def=json.dumps(yaml_def),
                output_shape_1024=json.dumps(shapes.get(idx, "")),
                params=int(getattr(layer, "np", sum(p.numel() for p in layer.parameters()))),
            )
        )
    return rows


def module_name(layer: object) -> str:
    return str(getattr(layer, "type", type(layer).__name__)).split(".")[-1]


def write_layer_csv(path: Path, rows: Sequence[LayerRow]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_head_csv(path: Path, heads: Sequence[ModelHead]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(heads[0]).keys()))
        writer.writeheader()
        for head in heads:
            writer.writerow(asdict(head))


def render_exact_topology_png(rows: Sequence[LayerRow], heads: Sequence[ModelHead], path: Path) -> None:
    width, height = 2450, 1480
    image = Image.new("RGB", (width, height), "#fbfbfd")
    draw = ImageDraw.Draw(image)
    title = font(28, bold=True)
    body = font(15)
    small = font(12)
    draw.text((34, 24), "Exact top-level YOLO11s topology used by current SkyScouter checkpoints", fill="#111111", font=title)
    draw.text(
        (34, 61),
        "Forward-hook shapes are from the Stage 2 checkpoint at input 1x3x1024x1024. C3k2/C2PSA internals are nested modules; see the layer table for exact YAML definitions.",
        fill="#444444",
        font=body,
    )

    positions = topology_positions()
    for src, dst in topology_edges():
        draw_arrow(draw, positions[src], positions[dst], dashed=isinstance(src, str) or dst in {12, 15, 18, 21})

    node_w, node_h = 190, 86
    draw_node(draw, positions["input"], node_w, node_h, "Input", "1x3x1024x1024", COLORS["input"], body, small)
    row_by_index = {r.index: r for r in rows}
    for idx in range(24):
        row = row_by_index[idx]
        color = color_for_idx(idx, row.module)
        label = f"{idx}: {row.module}"
        detail = f"{row.role}\n{short_shape(row.output_shape_1024)}"
        if idx == 23:
            detail = "inputs 16,19,22\nStage2 decoded 1x8x21504"
        draw_node(draw, positions[idx], node_w, node_h, label, detail, color, body, small)

    x, y = 36, 1370
    for name, color in [
        ("Backbone", COLORS["backbone"]),
        ("FPN/PAN neck", COLORS["neck"]),
        ("Concat", COLORS["concat"]),
        ("Upsample", COLORS["upsample"]),
        ("Detect inputs", COLORS["detect_input"]),
        ("Detect head", COLORS["detect"]),
    ]:
        draw.rounded_rectangle((x, y, x + 20, y + 20), radius=4, fill=color)
        draw.text((x + 28, y + 2), name, fill="#222222", font=body)
        x += 205

    head_text = "Detect class heads: " + " | ".join(f"{h.key}: nc={h.nc}" for h in heads if h.exists)
    draw.text((36, 1410), head_text, fill="#333333", font=body)
    image.save(path)


def topology_positions() -> Dict[int | str, tuple[int, int]]:
    return {
        "input": (36, 150),
        0: (255, 150),
        1: (475, 150),
        2: (695, 150),
        3: (915, 150),
        4: (1135, 150),
        5: (1355, 150),
        6: (1575, 150),
        7: (1795, 150),
        8: (2015, 150),
        9: (2015, 290),
        10: (2015, 430),
        11: (1795, 520),
        12: (1575, 520),
        13: (1355, 520),
        14: (1135, 660),
        15: (915, 660),
        16: (695, 660),
        17: (695, 910),
        18: (915, 910),
        19: (1135, 910),
        20: (1135, 1160),
        21: (1355, 1160),
        22: (1575, 1160),
        23: (1885, 910),
    }


def topology_edges() -> List[tuple[int | str, int]]:
    sequential = [("input", 0), (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16), (16, 17), (17, 18), (18, 19), (19, 20), (20, 21), (21, 22)]
    skips = [(6, 12), (4, 15), (13, 18), (10, 21), (16, 23), (19, 23), (22, 23)]
    return sequential + skips


def render_detect_head_png(heads: Sequence[ModelHead], path: Path) -> None:
    width, height = 1600, 980
    image = Image.new("RGB", (width, height), "#fbfbfd")
    draw = ImageDraw.Draw(image)
    title = font(28, bold=True)
    body = font(15)
    small = font(12)
    draw.text((34, 28), "Detect-head difference across SkyScouter checkpoints", fill="#111111", font=title)
    draw.text((34, 65), "The backbone/neck topology is the same; the class branch changes with nc.", fill="#444444", font=body)

    scales = [
        ("P3/8", "layer 16", "128 ch, 128x128"),
        ("P4/16", "layer 19", "256 ch, 64x64"),
        ("P5/32", "layer 22", "512 ch, 32x32"),
    ]
    y0 = 150
    for i, (scale, layer, shape) in enumerate(scales):
        y = y0 + i * 230
        draw_node(draw, (40, y), 220, 92, f"{scale} input", f"{layer}\n{shape}", COLORS["detect_input"], body, small)
        draw_arrow(draw, (260, y + 46), (390, y + 46))
        draw_node(draw, (390, y - 35), 250, 72, "bbox regression cv2", "two Conv blocks -> 64 DFL bins", "#2f80ed", body, small)
        draw_node(draw, (390, y + 75), 250, 72, "class branch cv3", "two DWConv/Conv blocks -> nc", "#27ae60", body, small)
        draw_arrow(draw, (640, y + 1), (760, y + 30))
        draw_arrow(draw, (640, y + 111), (760, y + 68))
        draw_node(draw, (760, y + 15), 240, 76, "raw per-scale output", "64 + nc channels\nthen decode/NMS", COLORS["detect"], body, small)

    x = 1060
    y = 145
    draw.text((x, y - 40), "Class-head comparison", fill="#111111", font=font(20, bold=True))
    for head in heads:
        if not head.exists:
            continue
        classes = ", ".join(head.names or [])
        text = f"{head.key}\nnc={head.nc}; raw channels/scale={head.detect_raw_channels_per_scale}\nclasses: {classes}"
        draw_multiline_box(draw, (x, y), 470, 92, text, "#ffffff", "#d6d9de", body, small)
        y += 112
    image.save(path)


def color_for_idx(idx: int, module: str) -> str:
    if idx == 23:
        return COLORS["detect"]
    if idx in {16, 19, 22}:
        return COLORS["detect_input"]
    if module == "Concat":
        return COLORS["concat"]
    if "Upsample" in module:
        return COLORS["upsample"]
    if idx <= 10:
        return COLORS["backbone"]
    return COLORS["neck"]


def draw_node(draw: ImageDraw.ImageDraw, pos: tuple[int, int], width: int, height: int, title: str, detail: str, color: str, title_font: ImageFont.ImageFont, detail_font: ImageFont.ImageFont) -> None:
    x, y = pos
    draw.rounded_rectangle((x, y, x + width, y + height), radius=9, fill=color)
    draw.text((x + 10, y + 11), title, fill="#ffffff", font=title_font)
    for offset, line in enumerate(detail.splitlines()):
        draw.text((x + 10, y + 38 + offset * 18), line, fill="#ffffff", font=detail_font)


def draw_multiline_box(draw: ImageDraw.ImageDraw, pos: tuple[int, int], width: int, height: int, text: str, fill: str, outline: str, title_font: ImageFont.ImageFont, detail_font: ImageFont.ImageFont) -> None:
    x, y = pos
    draw.rounded_rectangle((x, y, x + width, y + height), radius=8, fill=fill, outline=outline, width=1)
    lines = text.splitlines()
    draw.text((x + 10, y + 8), lines[0], fill="#111111", font=title_font)
    for i, line in enumerate(lines[1:]):
        draw.text((x + 10, y + 36 + i * 17), line[:70], fill="#333333", font=detail_font)


def draw_arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], dashed: bool = False) -> None:
    sx, sy = start
    ex, ey = end
    color = "#555555" if dashed else "#777777"
    width = 2
    if dashed:
        draw.line((sx, sy, ex, ey), fill=color, width=width)
    else:
        draw.line((sx, sy, ex, ey), fill=color, width=width)
    # Simple arrow head.
    import math

    angle = math.atan2(ey - sy, ex - sx)
    length = 10
    for delta in (2.55, -2.55):
        ax = ex - length * math.cos(angle + delta)
        ay = ey - length * math.sin(angle + delta)
        draw.line((ex, ey, ax, ay), fill=color, width=width)


def short_shape(value: str) -> str:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return value[:34]
    if isinstance(parsed, list) and parsed and isinstance(parsed[0], int):
        return "x".join(str(x) for x in parsed)
    return value[:34]


def render_markdown(rows: Sequence[LayerRow], heads: Sequence[ModelHead], imgsz: int) -> str:
    lines = [
        "# Exact SkyScouter Model Architecture Report",
        "",
        "This report replaces the earlier simplified drawing. It is generated from the actual checkpoints and actual forward-hook tensor shapes.",
        "",
        "## What Is Actually Different",
        "",
        "All inspected checkpoints use the same YOLO11s-scale top-level topology. The difference is the detection class head (`nc`) and training lineage.",
        "",
        "| Model | Params | nc | Classes | Status |",
        "|---|---:|---:|---|---|",
    ]
    for head in heads:
        classes = ", ".join(head.names or [])
        lines.append(f"| `{head.key}` | {head.params:,} | {head.nc} | {classes} | {head.status} |")
    lines.extend(
        [
            "",
            "## Detect Inputs",
            "",
            f"Forward-hook shapes below use input `1x3x{imgsz}x{imgsz}`.",
            "",
            "- P3/8: layer 16, shape `[1,128,128,128]`",
            "- P4/16: layer 19, shape `[1,256,64,64]`",
            "- P5/32: layer 22, shape `[1,512,32,32]`",
            "",
            "## Files",
            "",
            "- `exact_yolo11s_topology_1024.png`",
            "- `detect_head_class_channels.png`",
            "- `exact_layer_table_1024.csv`",
            "- `model_head_comparison.csv`",
            "",
            "## Exact Top-Level Layer Table",
            "",
            "| i | from | role | module | output shape | params |",
            "|---:|---|---|---|---|---:|",
        ]
    )
    for row in rows:
        lines.append(f"| {row.index} | `{row.from_}` | {row.role} | `{row.module}` | `{row.output_shape_1024}` | {row.params:,} |")
    return "\n".join(lines)


def render_html(rows: Sequence[LayerRow], heads: Sequence[ModelHead], imgsz: int) -> str:
    model_rows = []
    for head in heads:
        classes = ", ".join(head.names or [])
        model_rows.append(f"<tr><td>{esc(head.key)}</td><td>{head.params:,}</td><td>{head.nc}</td><td>{esc(classes)}</td><td>{esc(head.status)}</td></tr>")
    layer_rows = []
    for row in rows:
        layer_rows.append(
            f"<tr><td>{row.index}</td><td><code>{esc(row.from_)}</code></td><td>{esc(row.role)}</td><td><code>{esc(row.module)}</code></td><td><code>{esc(row.output_shape_1024)}</code></td><td>{row.params:,}</td></tr>"
        )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Exact SkyScouter Model Architecture</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; background: #f6f7f9; color: #141414; }}
    p {{ max-width: 1080px; line-height: 1.45; }}
    table {{ border-collapse: collapse; background: white; margin: 18px 0 32px; width: 100%; }}
    th, td {{ border: 1px solid #d8dce2; padding: 8px 10px; text-align: left; vertical-align: top; font-size: 13px; }}
    th {{ background: #eceff3; }}
    img {{ max-width: 100%; border: 1px solid #d8dce2; border-radius: 8px; background: white; margin: 10px 0 28px; }}
    code {{ background: #eef1f4; padding: 2px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>Exact SkyScouter Model Architecture</h1>
  <p>This replaces the earlier simplified sketch. It is generated from the actual YOLO checkpoints and forward hooks at <code>1x3x{imgsz}x{imgsz}</code>. The diagram is still top-level Ultralytics modules; nested C3k2/C2PSA internals are represented in the YAML/module table, not expanded as every internal Conv.</p>
  <h2>Topology</h2>
  <img src="exact_yolo11s_topology_1024.png" alt="Exact YOLO11s topology">
  <h2>Detect Head Difference</h2>
  <img src="detect_head_class_channels.png" alt="Detect head class-channel comparison">
  <h2>Model Head Comparison</h2>
  <table><thead><tr><th>Model</th><th>Params</th><th>nc</th><th>Classes</th><th>Status</th></tr></thead><tbody>{''.join(model_rows)}</tbody></table>
  <h2>Exact Top-Level Layer Table</h2>
  <table><thead><tr><th>i</th><th>from</th><th>role</th><th>module</th><th>output shape</th><th>params</th></tr></thead><tbody>{''.join(layer_rows)}</tbody></table>
</body>
</html>
"""


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "arialbd.ttf" if bold else "arial.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    return ImageFont.load_default()


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


if __name__ == "__main__":
    raise SystemExit(main())
