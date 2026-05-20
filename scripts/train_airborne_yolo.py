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
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--cache", choices=["ram", "disk", "false"], default=None)
    p.add_argument("--amp", choices=["true", "false"], default=None)
    p.add_argument("--lr0", type=float, default=None, help="Override initial learning rate")
    p.add_argument("--freeze", type=int, default=None, help="Override number of layers to freeze")
    p.add_argument("--cos-lr", choices=["true", "false"], default=None, help="Override cosine LR schedule")
    p.add_argument("--run-name", default=None, help="Override Ultralytics run name")
    p.add_argument("--resume", default=None, help="Path to last.pt checkpoint to resume training")
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

    if args.resume:
        resume_ckpt = Path(args.resume)
        if not resume_ckpt.exists():
            raise SystemExit(f"Resume checkpoint not found: {resume_ckpt}")
        model = YOLO(str(resume_ckpt))
        model.train(resume=True)
        return 0

    weights = args.weights or model_cfg.get("base_weights", "yolo11s.pt")
    model = YOLO(weights)

    # Resolve "project" to an absolute path. Ultralytics 8.4.x will otherwise
    # nest a relative project dir under its own default `runs/detect/`,
    # producing surprising paths like
    # `runs/detect/data/training/runs/<run_name>/`. We always want the run to
    # land at `<repo_root>/<project>/<run_name>/` exactly.
    project_cfg = str(train_cfg.get("project", "data/training/runs"))
    project_path = Path(project_cfg)
    if not project_path.is_absolute():
        project_path = (Path.cwd() / project_path).resolve()
    project_path.mkdir(parents=True, exist_ok=True)

    train_args = {
        "data": str(data_yaml),
        "imgsz": args.imgsz or int(model_cfg.get("image_size", 1280)),
        "epochs": args.epochs or int(train_cfg.get("epochs", 80)),
        "batch": args.batch or int(train_cfg.get("batch", 8)),
        "patience": int(train_cfg.get("patience", 20)),
        "workers": args.workers if args.workers is not None else int(train_cfg.get("workers", 4)),
        "project": str(project_path),
        "name": str(args.run_name or train_cfg.get("run_name", "yolo11s_airborne_drone_vs_bird_v1")),
        "optimizer": train_cfg.get("optimizer", "auto"),
        "close_mosaic": int(train_cfg.get("close_mosaic", 10)),
        "seed": int(train_cfg.get("seed", 42)),
        "cache": train_cfg.get("cache", False),
        "amp": bool(train_cfg.get("amp", True)),
    }
    if "lr0" in train_cfg or args.lr0 is not None:
        train_args["lr0"] = float(args.lr0 if args.lr0 is not None else train_cfg["lr0"])
    if "freeze" in train_cfg or args.freeze is not None:
        train_args["freeze"] = int(args.freeze if args.freeze is not None else train_cfg["freeze"])
    if "cos_lr" in train_cfg or args.cos_lr is not None:
        train_args["cos_lr"] = (args.cos_lr == "true") if args.cos_lr is not None else bool(train_cfg["cos_lr"])
    if args.cache is not None:
        train_args["cache"] = False if args.cache == "false" else args.cache
    if args.amp is not None:
        train_args["amp"] = args.amp == "true"
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
