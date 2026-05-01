"""
Skyscouter pipeline CLI.

Usage:
    python scripts/run_pipeline.py \
        --video data/videos/your_video.mp4 \
        --config configs/default.yaml \
        --output data/outputs/run_001
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

# Allow `python scripts/run_pipeline.py ...` from repo root without install
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from skyscouter.io.frame_source import build_frame_source
from skyscouter.perception.factory import build_detector
from skyscouter.tracking.factory import build_tracker
from skyscouter.output.run_logger import RunLogger
from skyscouter.output.target_state_writer import TargetStateJsonlWriter
from skyscouter.output.guidance_writer import GuidanceHintJsonlWriter
from skyscouter.output.bridge_writer import BridgeProposalJsonlWriter
from skyscouter.output.annotator import VideoAnnotator
from skyscouter.output.evaluation import DiagnosticsWriter, EvaluationCollector
from skyscouter.pipeline import Pipeline
from skyscouter.utils.config_loader import load_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Skyscouter Phase 1 pipeline runner")
    p.add_argument("--video", required=True,
                   help="Path to input video file (or image folder if source.type=image_folder)")
    p.add_argument("--config", required=True, help="Path to YAML config")
    p.add_argument("--output", required=True, help="Output directory for this run")
    p.add_argument("--run-id", default=None, help="Optional run identifier")
    p.add_argument("--gt", default=None, help="Optional sparse GT CSV override for evaluation")
    p.add_argument("--guidance-enabled", action="store_true",
                   help="Override config to enable visual bearing guidance hints")
    p.add_argument("--no-guidance", action="store_true",
                   help="Override config to disable visual bearing guidance hints")
    p.add_argument("--camera-hfov-deg", type=float, default=None,
                   help="Override guidance.camera.horizontal_fov_deg")
    p.add_argument("--mock-bridge-enabled", action="store_true",
                   help="Override config to enable mock GuidanceHint bridge JSONL output")
    p.add_argument("--no-mock-bridge", action="store_true",
                   help="Override config to disable mock GuidanceHint bridge output")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # Build run logger first so we capture everything that follows
    logger = RunLogger(output_dir=args.output, run_id=args.run_id)
    try:
        cfg = load_config(args.config)
        if args.gt:
            cfg.setdefault("evaluation", {})
            cfg["evaluation"]["enabled"] = True
            cfg["evaluation"]["gt_path"] = args.gt
        if args.guidance_enabled and args.no_guidance:
            raise ValueError("Use only one of --guidance-enabled or --no-guidance")
        if args.guidance_enabled:
            cfg.setdefault("guidance", {})
            cfg["guidance"]["enabled"] = True
        if args.no_guidance:
            cfg.setdefault("guidance", {})
            cfg["guidance"]["enabled"] = False
        if args.camera_hfov_deg is not None:
            cfg.setdefault("guidance", {}).setdefault("camera", {})
            cfg["guidance"]["camera"]["horizontal_fov_deg"] = args.camera_hfov_deg
        if args.mock_bridge_enabled and args.no_mock_bridge:
            raise ValueError("Use only one of --mock-bridge-enabled or --no-mock-bridge")
        if args.mock_bridge_enabled:
            cfg.setdefault("mock_bridge", {})
            cfg["mock_bridge"]["enabled"] = True
        if args.no_mock_bridge:
            cfg.setdefault("mock_bridge", {})
            cfg["mock_bridge"]["enabled"] = False
        if cfg.get("mock_bridge", {}).get("enabled", False) and not cfg.get("guidance", {}).get("enabled", False):
            raise ValueError("mock_bridge.enabled=true requires guidance.enabled=true")
        logger.set_config(cfg)
        logger.set_video_path(args.video)
        logger.info(f"Loaded config: {args.config}")

        # Build components
        source = build_frame_source(cfg["source"], video_path=args.video)
        w, h = source.get_resolution()
        fps = source.get_fps() or 30.0
        logger.info(f"Frame source: {source.get_source_id()}  resolution={w}x{h}  fps={fps:.2f}")

        detector = build_detector(cfg["detector"])
        logger.info(f"Detector: {detector.get_model_version()} input_size={detector.get_input_size()}")

        tracker = build_tracker(cfg["tracker"])
        logger.info("Tracker built")

        # Output writers
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        annotated_path = out_dir / "annotated.mp4"
        jsonl_path = out_dir / "target_states.jsonl"
        diagnostics_path = out_dir / "diagnostics.csv"
        eval_path = out_dir / "eval_report.json"

        out_cfg = cfg.get("output", {})
        annot_fps = out_cfg.get("annotated_video_fps") or fps
        annotator = (
            VideoAnnotator(
                str(annotated_path),
                w,
                h,
                fps=annot_fps,
                guidance_overlay_cfg=cfg.get("guidance", {}).get("overlay", {"enabled": False}),
            )
            if out_cfg.get("save_annotated_video", True) else None
        )
        writer_ctx = TargetStateJsonlWriter(str(jsonl_path)) \
            if out_cfg.get("save_target_states_jsonl", True) else None
        guidance_cfg = cfg.get("guidance", {})
        guidance_path = out_dir / "guidance_hints.jsonl"
        guidance_writer_ctx = GuidanceHintJsonlWriter(str(guidance_path)) \
            if guidance_cfg.get("enabled", False) and guidance_cfg.get("output_jsonl", True) else None
        bridge_cfg = cfg.get("mock_bridge", {})
        bridge_path = out_dir / "mock_bridge_proposals.jsonl"
        bridge_writer_ctx = BridgeProposalJsonlWriter(str(bridge_path)) \
            if bridge_cfg.get("enabled", False) and bridge_cfg.get("output_jsonl", True) else None
        diagnostics = DiagnosticsWriter(str(diagnostics_path)) \
            if out_cfg.get("save_diagnostics_csv", True) else None
        eval_cfg = cfg.get("evaluation", {})
        evaluator = EvaluationCollector(eval_cfg.get("gt_path")) \
            if eval_cfg.get("enabled", True) else None

        try:
            if writer_ctx is not None:
                writer_ctx.__enter__()
            if guidance_writer_ctx is not None:
                guidance_writer_ctx.__enter__()
                logger.set_artifact("guidance_hints_jsonl", str(guidance_path))
            if bridge_writer_ctx is not None:
                bridge_writer_ctx.__enter__()
                logger.set_artifact("mock_bridge_proposals_jsonl", str(bridge_path))

            pipeline = Pipeline(
                config=cfg,
                frame_source=source,
                detector=detector,
                tracker=tracker,
                run_logger=logger,
                target_state_writer=writer_ctx,
                guidance_hint_writer=guidance_writer_ctx,
                bridge_proposal_writer=bridge_writer_ctx,
                annotator=annotator,
                diagnostics_writer=diagnostics,
                evaluation_collector=evaluator,
            )
            pipeline.run()
        finally:
            if writer_ctx is not None:
                writer_ctx.__exit__(None, None, None)
            if guidance_writer_ctx is not None:
                guidance_writer_ctx.__exit__(None, None, None)
            if bridge_writer_ctx is not None:
                bridge_writer_ctx.__exit__(None, None, None)
            if annotator is not None:
                annotator.close()
            if diagnostics is not None:
                diagnostics.close()
            if evaluator is not None:
                evaluator.write_report(str(eval_path))
            source.close()

        logger.finalize(status="completed")
        print(f"\nRun complete. Outputs in: {out_dir}")
        return 0

    except Exception as e:
        tb = traceback.format_exc()
        try:
            logger.error(f"Pipeline failed: {e}\n{tb}")
            logger.finalize(status="failed", error=str(e))
        except Exception:
            print(tb, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
