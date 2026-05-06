"""Export a curated YOLO checkpoint to TensorRT on the deployment Jetson.

TensorRT engines are hardware, TensorRT, CUDA, and JetPack sensitive. Build the
`.engine` on the Jetson that will run it; do not export on the Windows
workstation and expect the result to be portable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a SkyScouter YOLO .pt checkpoint to TensorRT .engine."
    )
    parser.add_argument("--weights", required=True, help="Path to source .pt weights.")
    parser.add_argument("--imgsz", type=int, default=1024, help="Export input size.")
    parser.add_argument("--device", default="0", help="CUDA device, usually 0 on Jetson.")
    parser.add_argument("--batch", type=int, default=1, help="Static TensorRT batch size.")
    parser.add_argument("--half", action="store_true", help="Export FP16 engine.")
    parser.add_argument("--int8", action="store_true", help="Export INT8 engine; requires --data.")
    parser.add_argument("--data", default=None, help="YOLO data.yaml for INT8 calibration.")
    parser.add_argument("--dynamic", action="store_true", help="Export dynamic-shape engine.")
    parser.add_argument("--nms", action="store_true", help="Embed NMS if supported by installed Ultralytics.")
    parser.add_argument("--workspace", type=float, default=None, help="TensorRT workspace in GiB.")
    parser.add_argument("--output-dir", default=None, help="Optional directory to copy the engine into.")
    parser.add_argument("--name", default=None, help="Optional output engine filename.")
    parser.add_argument(
        "--no-load-check",
        action="store_true",
        help="Skip loading the exported engine and running one dummy prediction.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def command_output(cmd: list[str]) -> Optional[str]:
    try:
        result = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=10)
    except Exception:
        return None
    text = (result.stdout or result.stderr or "").strip()
    return text or None


def read_text(path: str) -> Optional[str]:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace").strip()
    except Exception:
        return None


def platform_report() -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": sys.version.split()[0],
        "uname": " ".join(platform.uname()),
        "jetson_l4t_release": read_text("/etc/nv_tegra_release"),
        "nvidia_jetpack_package": command_output(
            ["bash", "-lc", "dpkg-query -W nvidia-jetpack 2>/dev/null || true"]
        ),
        "trtexec_version": command_output(["bash", "-lc", "trtexec --version 2>&1 | head -n 5"]),
    }
    return report


def dependency_report() -> Dict[str, Any]:
    report: Dict[str, Any] = {}
    try:
        import ultralytics  # type: ignore

        report["ultralytics"] = getattr(ultralytics, "__version__", "unknown")
    except Exception as exc:
        report["ultralytics_error"] = str(exc)

    try:
        import torch  # type: ignore

        report["torch"] = getattr(torch, "__version__", "unknown")
        report["cuda_available"] = bool(torch.cuda.is_available())
        report["cuda_device_count"] = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
        if torch.cuda.is_available():
            report["cuda_device_0"] = torch.cuda.get_device_name(0)
            report["torch_cuda"] = getattr(torch.version, "cuda", None)
    except Exception as exc:
        report["torch_error"] = str(exc)

    return report


def copy_engine(exported: Path, args: argparse.Namespace) -> Path:
    if not args.output_dir and not args.name:
        return exported

    output_dir = Path(args.output_dir) if args.output_dir else exported.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = args.name or exported.name
    if not filename.endswith(".engine"):
        filename += ".engine"
    target = output_dir / filename
    if exported.resolve() != target.resolve():
        shutil.copy2(exported, target)
    return target


def main() -> int:
    args = parse_args()
    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")
    if args.int8 and not args.data:
        raise ValueError("INT8 export needs --data for calibration. Use --half for the first Jetson MVP export.")

    if platform.machine().lower() not in {"aarch64", "arm64"}:
        print(
            "WARNING: this does not look like a Jetson/aarch64 host. "
            "TensorRT engines should be exported on the target Jetson.",
            file=sys.stderr,
        )

    from ultralytics import YOLO  # type: ignore

    started = time.perf_counter()
    model = YOLO(str(weights))

    export_args: Dict[str, Any] = {
        "format": "engine",
        "imgsz": args.imgsz,
        "device": args.device,
        "batch": args.batch,
        "half": bool(args.half),
        "int8": bool(args.int8),
        "dynamic": bool(args.dynamic),
        "nms": bool(args.nms),
    }
    if args.data:
        export_args["data"] = args.data
    if args.workspace is not None:
        export_args["workspace"] = args.workspace

    exported_raw = model.export(**export_args)
    exported = Path(str(exported_raw))
    if not exported.exists():
        candidate = weights.with_suffix(".engine")
        if candidate.exists():
            exported = candidate
        else:
            raise FileNotFoundError(f"Ultralytics reported export path {exported_raw!r}, but no engine was found.")

    engine_path = copy_engine(exported, args)

    load_check: Dict[str, Any] = {"enabled": not args.no_load_check}
    if not args.no_load_check:
        try:
            engine_model = YOLO(str(engine_path))
            dummy = np.zeros((args.imgsz, args.imgsz, 3), dtype=np.uint8)
            pred_start = time.perf_counter()
            result = engine_model.predict(
                source=dummy,
                imgsz=args.imgsz,
                conf=0.25,
                iou=0.45,
                device=args.device,
                verbose=False,
            )
            load_check.update(
                {
                    "ok": True,
                    "dummy_predict_ms": round((time.perf_counter() - pred_start) * 1000.0, 3),
                    "result_count": len(result) if result is not None else 0,
                }
            )
        except Exception as exc:
            load_check.update({"ok": False, "error": str(exc)})

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_weights": str(weights),
        "source_weights_sha256": sha256_file(weights),
        "engine_path": str(engine_path),
        "engine_sha256": sha256_file(engine_path),
        "engine_bytes": engine_path.stat().st_size,
        "export_seconds": round(time.perf_counter() - started, 3),
        "export_args": export_args,
        "load_check": load_check,
        "platform": platform_report(),
        "dependencies": dependency_report(),
    }
    manifest_path = engine_path.with_suffix(".export_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Exported TensorRT engine: {engine_path}")
    print(f"Wrote export manifest: {manifest_path}")
    if load_check.get("enabled") and not load_check.get("ok"):
        print(f"WARNING: engine load check failed: {load_check.get('error')}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
