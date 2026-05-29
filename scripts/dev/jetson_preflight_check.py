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
    parser.add_argument("--rtsp-url", default=None)
    parser.add_argument("--config", default="configs/jetson_live_camera_pytorch.yaml")
    parser.add_argument("--weights", default="data/models/yolo11s_airborne_aod4_antiuav300_v2/best.pt")
    parser.add_argument("--engine", default="data/models/yolo11s_airborne_aod4_antiuav300_v2/best.engine")
    parser.add_argument("--probe-camera", action="store_true", help="Try opening the camera through OpenCV.")
    parser.add_argument("--probe-rtsp", action="store_true", help="Try opening an RTSP/IP camera stream.")
    parser.add_argument(
        "--probe-fc",
        action="store_true",
        help="Passively check the flight controller link (heartbeat only — never arms or takes off).",
    )
    parser.add_argument("--fc-serial", default=None, help="Override flight_control.serial_port for the FC probe.")
    parser.add_argument("--fc-baud", type=int, default=None, help="Override flight_control.baud for the FC probe.")
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


def rtsp_probe(url: str, timeout_s: float = 15.0) -> Dict[str, Any]:
    try:
        import cv2  # type: ignore

        deadline = datetime.now(timezone.utc).timestamp() + timeout_s
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        try:
            opened = bool(cap.isOpened())
            ok = False
            frame = None
            while opened and datetime.now(timezone.utc).timestamp() < deadline:
                ok, frame = cap.read()
                if ok and frame is not None:
                    break
            return {
                "ok": opened and ok and frame is not None,
                "opened": opened,
                "read_ok": bool(ok),
                "shape": list(frame.shape) if frame is not None else None,
                "width": cap.get(cv2.CAP_PROP_FRAME_WIDTH) if opened else None,
                "height": cap.get(cv2.CAP_PROP_FRAME_HEIGHT) if opened else None,
                "fps": cap.get(cv2.CAP_PROP_FPS) if opened else None,
                "url": url,
            }
        finally:
            cap.release()
    except Exception as exc:
        return {"ok": False, "url": url, "error": str(exc)}


def fc_probe(
    config_path: Path,
    serial_override: Optional[str],
    baud_override: Optional[int],
    timeout_s: float = 8.0,
) -> Dict[str, Any]:
    """Passive flight-controller readiness check.

    Opens a MAVLink connection, waits for a heartbeat, and reports the FC's
    current state. This is STRICTLY read-only: it never arms, never changes
    mode, and never sends a takeoff command. The engines get no power from
    this check — it only confirms we can talk to the FC and that GUIDED mode
    is available, so we know we are ready to fly.
    """
    port = serial_override
    baud = baud_override
    if port is None or baud is None:
        try:
            import yaml  # type: ignore

            cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            fc = cfg.get("flight_control", {}) or {}
            if port is None:
                port = fc.get("serial_port", "/dev/ttyACM0")
            if baud is None:
                baud = int(fc.get("baud", 115200))
        except Exception as exc:
            return {"ok": False, "error": f"could not read flight_control config: {exc}"}
    port = port or "/dev/ttyACM0"
    baud = int(baud or 115200)

    if not Path(port).exists():
        return {
            "ok": False,
            "serial_port": port,
            "baud": baud,
            "error": f"serial port {port} does not exist (FC not connected / not powered?)",
        }

    try:
        from pymavlink import mavutil  # type: ignore
    except Exception as exc:
        return {"ok": False, "serial_port": port, "baud": baud, "error": f"pymavlink import failed: {exc}"}

    conn = None
    try:
        conn = mavutil.mavlink_connection(port, baud=baud)
        hb = conn.wait_heartbeat(timeout=timeout_s)
        if hb is None:
            return {
                "ok": False,
                "serial_port": port,
                "baud": baud,
                "heartbeat": False,
                "error": f"no MAVLink heartbeat within {timeout_s:.0f}s on {port}@{baud}",
            }

        flight_mode = None
        guided_available: Optional[bool] = None
        armed = None
        try:
            flight_mode = conn.flightmode
        except Exception:
            flight_mode = None
        try:
            mapping = conn.mode_mapping() or {}
            guided_available = "GUIDED" in mapping
        except Exception:
            guided_available = None
        try:
            armed = bool(conn.motors_armed())
        except Exception:
            armed = None

        return {
            "ok": bool(hb is not None) and (guided_available is not False),
            "serial_port": port,
            "baud": baud,
            "heartbeat": True,
            "target_system": getattr(conn, "target_system", None),
            "target_component": getattr(conn, "target_component", None),
            "flight_mode": flight_mode,
            "armed": armed,
            "guided_available": guided_available,
        }
    except Exception as exc:
        return {"ok": False, "serial_port": port, "baud": baud, "error": str(exc)}
    finally:
        try:
            if conn is not None:
                conn.close()
        except Exception:
            pass


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
    if "rtsp_probe" in report["camera"]:
        lines.append(f"- RTSP probe ok: `{report['camera']['rtsp_probe'].get('ok')}`")
    fc = report.get("flight", {}).get("fc_probe")
    if fc is not None:
        lines.extend(["", "## Flight Controller (passive — never arms)", ""])
        lines.append(f"- FC link ready: `{fc.get('ok')}`")
        lines.append(f"- Heartbeat: `{fc.get('heartbeat')}`")
        lines.append(f"- Flight mode: `{fc.get('flight_mode')}`")
        lines.append(f"- Armed: `{fc.get('armed')}`")
        lines.append(f"- GUIDED available: `{fc.get('guided_available')}`")
        if fc.get("error"):
            lines.append(f"  - `{str(fc['error']).splitlines()[0][:180]}`")
    if report.get("ready_to_fly") is not None:
        verdict = "GO" if report["ready_to_fly"] else "NO-GO"
        lines.extend(["", f"## Flight Readiness: **{verdict}**", ""])
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
    if args.probe_rtsp:
        rtsp_url = args.rtsp_url
        if rtsp_url is None:
            try:
                import yaml  # type: ignore

                cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
                rtsp_url = cfg.get("source", {}).get("url")
            except Exception:
                rtsp_url = None
        camera["rtsp_probe"] = rtsp_probe(str(rtsp_url)) if rtsp_url else {
            "ok": False,
            "error": "No RTSP URL provided and source.url missing from config.",
        }

    flight: Dict[str, Any] = {}
    if args.probe_fc:
        flight["fc_probe"] = fc_probe(Path(args.config), args.fc_serial, args.fc_baud)

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
        "flight": flight,
    }
    report["overall_ok"] = all(item.get("ok") for item in required.values())

    # GO/NO-GO flight verdict: only meaningful when both the camera stream and
    # the FC link were actually probed. ready_to_fly stays None if either probe
    # was skipped, so a plain preflight (no --probe-* flags) does not claim GO.
    rtsp_ok = camera.get("rtsp_probe", {}).get("ok") if "rtsp_probe" in camera else None
    fc_ok = flight.get("fc_probe", {}).get("ok") if "fc_probe" in flight else None
    if rtsp_ok is None and fc_ok is None:
        report["ready_to_fly"] = None
    else:
        report["ready_to_fly"] = bool(
            report["overall_ok"]
            and (rtsp_ok is True)
            and (fc_ok is True)
        )

    json_path = out_dir / "preflight_report.json"
    md_path = out_dir / "preflight_report.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, md_path)

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print("PASS" if report["overall_ok"] else "CHECK REQUIRED ITEMS")

    if args.probe_fc:
        fc = report["flight"].get("fc_probe", {})
        if fc.get("ok"):
            print(
                f"FC link READY (heartbeat from sys {fc.get('target_system')}, "
                f"mode={fc.get('flight_mode')}, armed={fc.get('armed')}, "
                f"guided_available={fc.get('guided_available')})"
            )
        else:
            print(f"FC link NOT READY: {fc.get('error', 'unknown')}")

    if report["ready_to_fly"] is not None:
        print("FLIGHT READINESS: GO" if report["ready_to_fly"] else "FLIGHT READINESS: NO-GO")

    return 0 if report["overall_ok"] else 2


if __name__ == "__main__":
    sys.exit(main())
