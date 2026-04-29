"""Train the airborne drone-vs-bird YOLO detector from a training profile."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train airborne YOLO detector")
    p.add_argument("--config", default="configs/training/airborne_yolo11.yaml")
    p.add_argument("--data", default=None, help="Override prepared data.yaml")
    p.add_argument("--weights", default=None, help="Override base weights")
    p.add_argument("--device", default=None, help="Override device, e.g. cpu, 0, mps")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--imgsz", type=int, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Ultralytics is not installed. Run: pip install -e '.[yolo]'") from exc

    data_yaml = Path(args.data or cfg["dataset"]["data_yaml"])
    if not data_yaml.exists():
        raise SystemExit(
            f"Prepared data.yaml not found: {data_yaml}\n"
            "Run scripts/prepare_airborne_training_set.py first."
        )

    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})
    aug_cfg = cfg.get("augmentation", {})

    weights = args.weights or model_cfg.get("base_weights", "yolo11s.pt")
    model = YOLO(weights)
    train_args = {
        "data": str(data_yaml),
        "imgsz": args.imgsz or int(model_cfg.get("image_size", 1280)),
        "epochs": args.epochs or int(train_cfg.get("epochs", 80)),
        "batch": args.batch or int(train_cfg.get("batch", 8)),
        "patience": int(train_cfg.get("patience", 20)),
        "workers": int(train_cfg.get("workers", 4)),
        "project": str(train_cfg.get("project", "data/training/runs")),
        "name": str(train_cfg.get("run_name", "yolo11s_airborne_drone_vs_bird_v1")),
        "optimizer": train_cfg.get("optimizer", "auto"),
        "close_mosaic": int(train_cfg.get("close_mosaic", 10)),
        "seed": int(train_cfg.get("seed", 42)),
    }
    device = args.device or train_cfg.get("device")
    if device and device != "auto":
        train_args["device"] = device
    else:
        # Auto-detect: prefer GPU (Metal on Mac, CUDA on Linux/Windows)
        try:
            import torch
            if torch.backends.mps.is_available():
                device = "mps"
                print("[INFO] Metal GPU (MPS) detected and enabled for training")
            elif torch.cuda.is_available():
                device = 0  # default CUDA device
                print(f"[INFO] CUDA GPU detected: {torch.cuda.get_device_name(0)}")
            else:
                device = "cpu"
                print("[WARNING] No GPU detected; training on CPU (slow)")
            train_args["device"] = device
        except ImportError:
            pass
    train_args.update(aug_cfg)
    model.train(**train_args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
