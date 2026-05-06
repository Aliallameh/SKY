"""Jetson preflight diagnostics for SkyScouter deployment.

Run this on the Jetson before the live-camera smoke. It does not require a
camera open by default; pass --probe-camera if you want it to try one OpenCV
read from the configured camera.
"""
from __future__ import annotations

import argparse
import importlib
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SkyScouter Jetson preflight diagnostics.")
    parser.add_argument("--camera-device", default="/dev/video0")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--config", default="configs/jetson_live_camera_pytorch.yaml")
    parser.add_argument("--weights", default="data/models/yolo11s_airborne_aod4_antiuav300_v2/best.pt")
    parser.add_argument("--engine", default="data/models/yolo11s_airborne_aod4_antiuav300_v2/best.engine")
    parser.add_argument("--probe-camera", action="store_true", help="Try opening the camera through OpenCV.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Defaults to data/outputs/jetson_preflight_<timestamp>.",
    )
    return parser.parse_args()


def run_cmd(cmd: list[str], timeout: int = 10) -> Dict[str, Any]:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
        return {
            "ok": result.returncode == 0,
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def read_text(path: str) -> Optional[str]:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace").strip()
    except Exception:
        return None


def import_check(module_name: str) -> Dict[str, Any]:
    try:
        module = importlib.import_module(module_name)
        return {"ok": True, "version": getattr(module, "__version__", None)}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def torch_check() -> Dict[str, Any]:
    try:
        import torch  # type: ignore

        report: Dict[str, Any] = {
            "ok": True,
            "version": getattr(torch, "__version__", None),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
            "torch_cuda": getattr(torch.version, "cuda", None),
        }
        if torch.cuda.is_available():
            report["device_0"] = torch.cuda.get_device_name(0)
        return report
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def config_check(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        return {"ok": False, "error": f"missing config: {config_path}"}
    try:
        import yaml  # type: ignore

        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        source = cfg.get("source", {})
        detector = cfg.get("detector", {})
        return {
            "ok": True,
            "profile_name": cfg.get("profile_name"),
            "source": source,
            "detector_weights": detector.get("weights"),
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def camera_probe(camera_index: int) -> Dict[str, Any]:
    try:
        import cv2  # type: ignore

        cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 30.0)
            opened = bool(cap.isOpened())
            ok, frame = cap.read() if opened else (False, None)
            fourcc_raw = int(cap.get(cv2.CAP_PROP_FOURCC)) if opened else 0
            fourcc = "".join(chr((fourcc_raw >> 8 * i) & 0xFF) for i in range(4))
            return {
                "ok": opened and ok and frame is not None,
                "opened": opened,
                "read_ok": bool(ok),
                "shape": list(frame.shape) if frame is not None else None,
                "fourcc": fourcc,
                "width": cap.get(cv2.CAP_PROP_FRAME_WIDTH) if opened else None,
                "height": cap.get(cv2.CAP_PROP_FRAME_HEIGHT) if opened else None,
                "fps": cap.get(cv2.CAP_PROP_FPS) if opened else None,
            }
        finally:
            cap.release()
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def default_output_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("data") / "outputs" / f"jetson_preflight_{stamp}"


def write_markdown(report: Dict[str, Any], path: Path) -> None:
    lines = [
        "# Jetson Preflight Report",
        "",
        f"- Created UTC: `{report['created_utc']}`",
        f"- Overall status: `{'PASS' if report['overall_ok'] else 'CHECK'}`",
        "",
        "## Required Checks",
    ]
    for name, item in report["required"].items():
        status = "PASS" if item.get("ok") else "CHECK"
        lines.append(f"- `{name}`: **{status}**")
        detail = item.get("error") or item.get("stdout") or item.get("version")
        if detail:
            lines.append(f"  - `{str(detail).splitlines()[0][:180]}`")
    lines.extend(["", "## Camera", ""])
    lines.append(f"- Device exists: `{report['camera']['device_exists']}`")
    lines.append(f"- v4l2 formats captured: `{bool(report['camera'].get('formats', {}).get('stdout'))}`")
    if "opencv_probe" in report["camera"]:
        lines.append(f"- OpenCV probe ok: `{report['camera']['opencv_probe'].get('ok')}`")
    lines.extend(["", "See `preflight_report.json` for full command output.", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir) if args.output_dir else default_output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    required = {
        "git": run_cmd(["git", "--version"]),
        "git_lfs": run_cmd(["git", "lfs", "version"]),
        "python": {"ok": True, "version": sys.version.split()[0]},
        "cv2": import_check("cv2"),
        "numpy": import_check("numpy"),
        "yaml": import_check("yaml"),
        "ultralytics": import_check("ultralytics"),
        "torch": torch_check(),
        "config": config_check(Path(args.config)),
        "weights": {"ok": Path(args.weights).exists(), "path": args.weights},
    }

    optional = {
        "tensorrt": import_check("tensorrt"),
        "engine": {"ok": Path(args.engine).exists(), "path": args.engine},
        "nvpmodel": run_cmd(["bash", "-lc", "sudo -n nvpmodel -q"], timeout=5),
        "jetson_clocks": run_cmd(["bash", "-lc", "which jetson_clocks && echo present"], timeout=5),
    }

    camera = {
        "camera_device": args.camera_device,
        "camera_index": args.camera_index,
        "device_exists": Path(args.camera_device).exists(),
        "v4l2_ctl_path": shutil.which("v4l2-ctl"),
        "devices": run_cmd(["v4l2-ctl", "--list-devices"]) if shutil.which("v4l2-ctl") else {},
        "formats": run_cmd(["v4l2-ctl", "-d", args.camera_device, "--list-formats-ext"], timeout=20)
        if shutil.which("v4l2-ctl")
        else {},
    }
    if args.probe_camera:
        camera["opencv_probe"] = camera_probe(args.camera_index)

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": sys.version,
            "nv_tegra_release": read_text("/etc/nv_tegra_release"),
            "home": str(Path.home()),
        },
        "required": required,
        "optional": optional,
        "camera": camera,
    }
    report["overall_ok"] = all(item.get("ok") for item in required.values())

    json_path = out_dir / "preflight_report.json"
    md_path = out_dir / "preflight_report.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, md_path)

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print("PASS" if report["overall_ok"] else "CHECK REQUIRED ITEMS")
    return 0 if report["overall_ok"] else 2


if __name__ == "__main__":
    sys.exit(main())
