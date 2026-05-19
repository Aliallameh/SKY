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
from skyscouter.output.operator_view import LiveOperatorView
from skyscouter.output.raw_video_recorder import RawVideoRecorder
from skyscouter.output.evaluation import DiagnosticsWriter, EvaluationCollector
from skyscouter.gimbal.factory import build_gimbal_follow_controller
from skyscouter.pipeline import Pipeline
from skyscouter.utils.config_loader import load_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Skyscouter Phase 1 pipeline runner")
    p.add_argument("--video", default=None,
                   help="Path to input video file or image folder. Not needed for source.type=opencv_camera.")
    p.add_argument("--config", required=True, help="Path to YAML config")
    p.add_argument("--output", required=True, help="Output directory for this run")
    p.add_argument("--source-url", default=None,
                   help="Override source.url for RTSP/IP camera configs")
    p.add_argument("--run-id", default=None, help="Optional run identifier")
    p.add_argument("--gt", default=None, help="Optional sparse GT CSV override for evaluation")
    p.add_argument("--guidance-enabled", action="store_true",
                   help="Override config to enable visual bearing guidance hints")
    p.add_argument("--no-guidance", action="store_true",
                   help="Override config to disable visual bearing guidance hints")
    p.add_argument("--camera-hfov-deg", type=float, default=None,
                   help="Override guidance.camera.horizontal_fov_deg")
    p.add_argument("--detector-conf", type=float, default=None,
                   help="Override detector.confidence_threshold for diagnostic sweeps")
    p.add_argument("--tracker-max-prediction-only-frames", type=int, default=None,
                   help="Override tracker.max_prediction_only_frames for diagnostic sweeps")
    p.add_argument("--tracker-min-prediction-confidence", type=float, default=None,
                   help="Override tracker.min_prediction_confidence for diagnostic sweeps")
    p.add_argument("--tracker-reacquisition-radius-px", type=float, default=None,
                   help="Override tracker.reacquisition_radius_px for diagnostic sweeps")
    p.add_argument("--tracker-center-match-threshold-px", type=float, default=None,
                   help="Override tracker.center_match_threshold_px for diagnostic sweeps")
    p.add_argument("--lock-accept-labels", default=None,
                   help="Comma-separated override for lock.acceptable_lock_labels")
    p.add_argument("--mock-bridge-enabled", action="store_true",
                   help="Override config to enable mock GuidanceHint bridge JSONL output")
    p.add_argument("--no-mock-bridge", action="store_true",
                   help="Override config to disable mock GuidanceHint bridge output")
    p.add_argument("--operator-view", action="store_true",
                   help="Enable live operator view for annotated ML frames")
    p.add_argument("--no-operator-view", action="store_true",
                   help="Disable live operator view even if config enables it")
    p.add_argument("--operator-view-mode", choices=["mjpeg", "window", "both"], default=None,
                   help="Live operator view mode")
    p.add_argument("--operator-view-host", default=None,
                   help="MJPEG bind host, for example 0.0.0.0")
    p.add_argument("--operator-view-port", type=int, default=None,
                   help="MJPEG bind port")
    p.add_argument("--operator-view-max-width", type=int, default=None,
                   help="Downscale live view frames to this max width before streaming")
    p.add_argument("--operator-view-jpeg-quality", type=int, default=None,
                   help="MJPEG JPEG quality from 1 to 100")
    p.add_argument("--operator-view-fullscreen", action="store_true",
                   help="Open the operator window fullscreen for HDMI/video transmitters")
    p.add_argument("--operator-view-windowed", action="store_true",
                   help="Force the operator window to non-fullscreen")
    p.add_argument("--operator-view-display-width", type=int, default=None,
                   help="Letterbox the operator window frame to this display width")
    p.add_argument("--operator-view-display-height", type=int, default=None,
                   help="Letterbox the operator window frame to this display height")
    p.add_argument("--operator-view-window-x", type=int, default=None,
                   help="Move the operator window to this X position before fullscreen")
    p.add_argument("--operator-view-window-y", type=int, default=None,
                   help="Move the operator window to this Y position before fullscreen")
    p.add_argument("--operator-view-window-backend", choices=["opencv", "gstreamer"], default=None,
                   help="Window display backend. Use gstreamer on Jetson headless OpenCV builds.")
    p.add_argument("--operator-view-display-fps", type=int, default=None,
                   help="Display framerate hint for the GStreamer window backend")
    p.add_argument("--gimbal-follow-enabled", action="store_true",
                   help="Override config to enable SIYI gimbal follow control")
    p.add_argument("--no-gimbal-follow", action="store_true",
                   help="Override config to disable SIYI gimbal follow control")
    p.add_argument("--gimbal-follow-dry-run", action="store_true",
                   help="Force gimbal follow to dry-run (log commands but do not send UDP)")
    p.add_argument("--gimbal-follow-live", action="store_true",
                   help="Disable dry-run: actually send UDP rotation commands to the SIYI gimbal")
    p.add_argument("--gimbal-host", default=None, help="Override gimbal_follow.host")
    p.add_argument("--gimbal-port", type=int, default=None, help="Override gimbal_follow.port")
    p.add_argument("--gimbal-invert-yaw", action="store_true",
                   help="Override gimbal_follow.invert_yaw=true")
    p.add_argument("--gimbal-invert-pitch", action="store_true",
                   help="Override gimbal_follow.invert_pitch=true")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # Build run logger first so we capture everything that follows
    logger = RunLogger(output_dir=args.output, run_id=args.run_id)
    try:
        cfg = load_config(args.config)
        if args.source_url is not None:
            cfg.setdefault("source", {})
            cfg["source"]["url"] = args.source_url
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
        if args.detector_conf is not None:
            cfg.setdefault("detector", {})
            cfg["detector"]["confidence_threshold"] = args.detector_conf
        if args.tracker_max_prediction_only_frames is not None:
            cfg.setdefault("tracker", {})
            cfg["tracker"]["max_prediction_only_frames"] = args.tracker_max_prediction_only_frames
        if args.tracker_min_prediction_confidence is not None:
            cfg.setdefault("tracker", {})
            cfg["tracker"]["min_prediction_confidence"] = args.tracker_min_prediction_confidence
        if args.tracker_reacquisition_radius_px is not None:
            cfg.setdefault("tracker", {})
            cfg["tracker"]["reacquisition_radius_px"] = args.tracker_reacquisition_radius_px
        if args.tracker_center_match_threshold_px is not None:
            cfg.setdefault("tracker", {})
            cfg["tracker"]["center_match_threshold_px"] = args.tracker_center_match_threshold_px
        if args.lock_accept_labels is not None:
            labels = [x.strip() for x in args.lock_accept_labels.split(",") if x.strip()]
            cfg.setdefault("lock", {})
            cfg["lock"]["acceptable_lock_labels"] = labels
        if args.mock_bridge_enabled and args.no_mock_bridge:
            raise ValueError("Use only one of --mock-bridge-enabled or --no-mock-bridge")
        if args.mock_bridge_enabled:
            cfg.setdefault("mock_bridge", {})
            cfg["mock_bridge"]["enabled"] = True
        if args.no_mock_bridge:
            cfg.setdefault("mock_bridge", {})
            cfg["mock_bridge"]["enabled"] = False
        if args.operator_view and args.no_operator_view:
            raise ValueError("Use only one of --operator-view or --no-operator-view")
        if args.operator_view:
            cfg.setdefault("operator_view", {})
            cfg["operator_view"]["enabled"] = True
        if args.no_operator_view:
            cfg.setdefault("operator_view", {})
            cfg["operator_view"]["enabled"] = False
        if args.operator_view_mode is not None:
            cfg.setdefault("operator_view", {})
            cfg["operator_view"]["mode"] = args.operator_view_mode
        if args.operator_view_host is not None:
            cfg.setdefault("operator_view", {})
            cfg["operator_view"]["host"] = args.operator_view_host
        if args.operator_view_port is not None:
            cfg.setdefault("operator_view", {})
            cfg["operator_view"]["port"] = args.operator_view_port
        if args.operator_view_max_width is not None:
            cfg.setdefault("operator_view", {})
            cfg["operator_view"]["max_width"] = args.operator_view_max_width
        if args.operator_view_jpeg_quality is not None:
            cfg.setdefault("operator_view", {})
            cfg["operator_view"]["jpeg_quality"] = args.operator_view_jpeg_quality
        if args.operator_view_fullscreen and args.operator_view_windowed:
            raise ValueError("Use only one of --operator-view-fullscreen or --operator-view-windowed")
        if args.operator_view_fullscreen:
            cfg.setdefault("operator_view", {})
            cfg["operator_view"]["fullscreen"] = True
        if args.operator_view_windowed:
            cfg.setdefault("operator_view", {})
            cfg["operator_view"]["fullscreen"] = False
        if args.operator_view_display_width is not None:
            cfg.setdefault("operator_view", {})
            cfg["operator_view"]["display_width"] = args.operator_view_display_width
        if args.operator_view_display_height is not None:
            cfg.setdefault("operator_view", {})
            cfg["operator_view"]["display_height"] = args.operator_view_display_height
        if args.operator_view_window_x is not None:
            cfg.setdefault("operator_view", {})
            cfg["operator_view"]["window_x"] = args.operator_view_window_x
        if args.operator_view_window_y is not None:
            cfg.setdefault("operator_view", {})
            cfg["operator_view"]["window_y"] = args.operator_view_window_y
        if args.operator_view_window_backend is not None:
            cfg.setdefault("operator_view", {})
            cfg["operator_view"]["window_backend"] = args.operator_view_window_backend
        if args.operator_view_display_fps is not None:
            cfg.setdefault("operator_view", {})
            cfg["operator_view"]["display_fps"] = args.operator_view_display_fps
        if cfg.get("mock_bridge", {}).get("enabled", False) and not cfg.get("guidance", {}).get("enabled", False):
            raise ValueError("mock_bridge.enabled=true requires guidance.enabled=true")
        if args.gimbal_follow_enabled and args.no_gimbal_follow:
            raise ValueError("Use only one of --gimbal-follow-enabled or --no-gimbal-follow")
        if args.gimbal_follow_dry_run and args.gimbal_follow_live:
            raise ValueError("Use only one of --gimbal-follow-dry-run or --gimbal-follow-live")
        if args.gimbal_follow_enabled:
            cfg.setdefault("gimbal_follow", {})
            cfg["gimbal_follow"]["enabled"] = True
        if args.no_gimbal_follow:
            cfg.setdefault("gimbal_follow", {})
            cfg["gimbal_follow"]["enabled"] = False
        if args.gimbal_follow_dry_run:
            cfg.setdefault("gimbal_follow", {})
            cfg["gimbal_follow"]["dry_run"] = True
        if args.gimbal_follow_live:
            cfg.setdefault("gimbal_follow", {})
            cfg["gimbal_follow"]["dry_run"] = False
        if args.gimbal_host is not None:
            cfg.setdefault("gimbal_follow", {})
            cfg["gimbal_follow"]["host"] = args.gimbal_host
        if args.gimbal_port is not None:
            cfg.setdefault("gimbal_follow", {})
            cfg["gimbal_follow"]["port"] = args.gimbal_port
        if args.gimbal_invert_yaw:
            cfg.setdefault("gimbal_follow", {})
            cfg["gimbal_follow"]["invert_yaw"] = True
        if args.gimbal_invert_pitch:
            cfg.setdefault("gimbal_follow", {})
            cfg["gimbal_follow"]["invert_pitch"] = True
        if cfg.get("gimbal_follow", {}).get("enabled", False) and not cfg.get("guidance", {}).get("enabled", False):
            raise ValueError("gimbal_follow.enabled=true requires guidance.enabled=true")
        logger.set_config(cfg)
        logger.set_video_path(args.video)
        logger.info(f"Loaded config: {args.config}")

        # Build components
        source = build_frame_source(cfg["source"], video_path=args.video)
        w, h = source.get_resolution()
        fps = source.get_fps() or 30.0
        logger.info(f"Frame source: {source.get_source_id()}  resolution={w}x{h}  fps={fps:.2f}")
        if hasattr(source, "get_negotiated_settings"):
            source_metadata = source.get_negotiated_settings()
            logger.set_source_metadata(source_metadata)
            logger.info(f"Frame source negotiated settings: {source_metadata}")

        detector = build_detector(cfg["detector"])
        logger.info(f"Detector: {detector.get_model_version()} input_size={detector.get_input_size()}")

        tracker = build_tracker(cfg["tracker"])
        logger.info("Tracker built")

        # Output writers
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        annotated_path = out_dir / "annotated.mp4"
        raw_video_path = out_dir / "raw_camera.mp4"
        jsonl_path = out_dir / "target_states.jsonl"
        diagnostics_path = out_dir / "diagnostics.csv"
        eval_path = out_dir / "eval_report.json"

        out_cfg = cfg.get("output", {})
        annot_fps = out_cfg.get("annotated_video_fps") or fps
        operator_view_cfg = cfg.get("operator_view", {})
        operator_view_enabled = bool(operator_view_cfg.get("enabled", False))
        operator_view = None
        save_annotated_video = bool(out_cfg.get("save_annotated_video", True))
        needs_annotated_frames = save_annotated_video or operator_view_enabled
        annotator = (
            VideoAnnotator(
                str(annotated_path) if save_annotated_video else None,
                w,
                h,
                fps=annot_fps,
                guidance_overlay_cfg=cfg.get("guidance", {}).get("overlay", {"enabled": False}),
                write_video=save_annotated_video,
            )
            if needs_annotated_frames else None
        )
        raw_recorder = (
            RawVideoRecorder(str(raw_video_path), w, h, fps=annot_fps)
            if out_cfg.get("save_raw_video", False) else None
        )
        if raw_recorder is not None:
            logger.set_artifact("raw_camera_video", str(raw_video_path))
        if save_annotated_video and annotator is not None:
            logger.set_artifact("annotated_video", str(annotated_path))
        writer_ctx = TargetStateJsonlWriter(str(jsonl_path)) \
            if out_cfg.get("save_target_states_jsonl", True) else None
        if writer_ctx is not None:
            logger.set_artifact("target_states_jsonl", str(jsonl_path))
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
        if diagnostics is not None:
            logger.set_artifact("diagnostics_csv", str(diagnostics_path))
        eval_cfg = cfg.get("evaluation", {})
        evaluator = EvaluationCollector(eval_cfg.get("gt_path")) \
            if eval_cfg.get("enabled", True) else None
        if evaluator is not None:
            logger.set_artifact("eval_report_json", str(eval_path))

        gimbal_follow = build_gimbal_follow_controller(
            cfg, output_dir=out_dir, run_id=logger.run_id,
        )
        if gimbal_follow is not None:
            mode = "DRY_RUN" if gimbal_follow.dry_run else "LIVE_UDP"
            logger.info(f"Gimbal follow controller: enabled mode={mode}")
            if gimbal_follow.log_path is not None:
                logger.set_artifact("gimbal_follow_commands_jsonl", str(gimbal_follow.log_path))

        try:
            if operator_view_enabled:
                operator_view = LiveOperatorView.from_config(operator_view_cfg)
                if operator_view.remote_url_hint:
                    logger.info(f"Operator view: {operator_view.remote_url_hint}")
                    logger.set_artifact("operator_view_url", operator_view.remote_url_hint)
                elif operator_view.url:
                    logger.info(f"Operator view: {operator_view.url}")
                    logger.set_artifact("operator_view_url", operator_view.url)
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
                operator_view=operator_view,
                raw_video_recorder=raw_recorder,
                diagnostics_writer=diagnostics,
                evaluation_collector=evaluator,
                gimbal_follow_controller=gimbal_follow,
            )
            status = "completed"
            try:
                pipeline.run()
            except KeyboardInterrupt:
                status = "interrupted"
                logger.warning("Pipeline interrupted by user; closing outputs cleanly")
        finally:
            if writer_ctx is not None:
                writer_ctx.__exit__(None, None, None)
            if guidance_writer_ctx is not None:
                guidance_writer_ctx.__exit__(None, None, None)
            if bridge_writer_ctx is not None:
                bridge_writer_ctx.__exit__(None, None, None)
            if annotator is not None:
                annotator.close()
            if operator_view is not None:
                operator_view.close()
            if raw_recorder is not None:
                raw_recorder.close()
            if diagnostics is not None:
                diagnostics.close()
            if evaluator is not None:
                evaluator.write_report(str(eval_path))
            if gimbal_follow is not None:
                gimbal_follow.close()
            source.close()

        logger.finalize(status=status)
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
