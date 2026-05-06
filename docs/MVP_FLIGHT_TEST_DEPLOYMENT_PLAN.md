# MVP Flight-Test Deployment Plan

Last updated: 2026-05-05

This is the handoff plan for moving SkyScouter from bench replay to Jetson
Orin Nano Super shadow testing on the MVP drone.

The first real-air milestone is **log-only evaluation**, not autonomous chase.
The current model can collect real flight evidence, but it is not yet
guidance-ready: the known hard-case still shows severe semantic confusion
where the drone is often boxed as `airplane`.

## Current Ground Truth

Implemented today:

- Video-file replay with detector, tracker, lock state, overlays, JSONL logs,
  diagnostics, and manifests.
- Visual bearing/elevation guidance hints from bbox center.
- Mock bridge JSONL rows behind lock-state, guidance, yaw-limit, and reviewed
  calibration gates.
- Final v2 model:
  `data/models/yolo11s_airborne_aod4_antiuav300_v2/best.pt`.
- V3 fine-tune plan for the local hard-case semantic failure.

Implemented on the Jetson deployment/live-camera branches:

- Live USB camera frame source through OpenCV/V4L2 (`source.type: live_camera`).
- Jetson/TensorRT export script: `scripts/export_tensorrt.py`.
- Detector backend benchmark script: `scripts/benchmark_detector_backend.py`.
- Jetson setup runbook: `docs/JETSON_ORIN_NANO_SETUP.md`.

Not implemented today:

- Camera calibration capture/calibrate/review tooling.
- MAVLink or ESP32 command transport.
- Real range estimation or meter-per-pixel GSD.

Known MVP choices:

- Camera connection: **USB**.
- Camera: **4K ZOOM CAMERA** at `/dev/video0`.
- First live capture mode: **MJPG 1280x720 30 fps** through V4L2.
- First flight mode: **log-only shadow mode**.
- Range/GSD: **calibration-only for now**. Show angular scale; show GSD as
  `n/a` until a real range source exists.

## Open Questions To Close Before Coding

Ask the hardware/flight teammate:

```text
1. Exact USB camera model? **Closed: 4K ZOOM CAMERA.**
2. Does it appear on Jetson as /dev/video0, /dev/video1, etc.? **Closed: `/dev/video0`.**
3. Supported resolution/FPS modes and pixel formats? **Closed for first runtime:
   use MJPG 1280x720 30 fps; avoid YUYV for realtime.**
4. Can exposure, gain, white balance, and focus be locked manually?
5. Lens FOV and whether the lens is fixed-focus or autofocus?
6. Is the camera rigidly mounted to the drone body or on a gimbal?
7. Flight controller model and firmware: ArduPilot or PX4?
8. Jetson-to-flight-controller physical link: USB serial, UART, Ethernet, or ESP32?
9. What exactly does the ESP32 do: relay, safety switch, sensor hub, or command bridge?
10. Jetson storage, power regulator, cooling, and JetPack version?
```

Do not design a control bridge until items 7-9 are answered.

## Phase 1: Calibrate The Actual USB Camera

Why this is first:

- `guidance.camera.calibration_reviewed: false` suppresses
  `valid_for_transport=true` in the mock bridge.
- The default 70 degree FOV is a bench assumption, not flight calibration.
- Bearing/elevation/yaw proposals are only credible after real intrinsics are
  measured on the exact capture camera and lens.

Implementation tasks:

- Add `scripts/calibrate_camera.py` with three modes:
  - `capture`: collect 15-30 chessboard images from the USB camera.
  - `calibrate`: run `cv2.calibrateCamera`.
  - `review`: save undistorted samples and a YAML calibration block.
- Use OpenCV chessboard size `9x6` inner corners by default.
- Emit `fx_px`, `fy_px`, `cx_px`, `cy_px`, distortion coefficients, image
  width/height, reprojection error, camera model/name, and
  `calibration_reviewed: false` by default.
- Human review sets `calibration_reviewed: true` only after checking
  reprojection error and undistorted samples.

Acceptance:

- Calibration YAML loads into `PinholeCameraModel`.
- Center pixel produces near-zero bearing/elevation.
- Mock bridge suppression no longer includes `calibration_not_reviewed` after
  reviewed calibration is intentionally enabled.

## Phase 2: Jetson Environment And TensorRT

Implementation tasks:

- Add `docs/JETSON_ORIN_NANO_SETUP.md`.
- Add `scripts/export_tensorrt.py`.
- Export on the Jetson, not Windows:

```bash
python3 scripts/export_tensorrt.py \
  --weights data/models/yolo11s_airborne_aod4_antiuav300_v2/best.pt \
  --imgsz 1024 \
  --half
```

- Add a benchmark command comparing PyTorch `.pt` and TensorRT `.engine` on
  the same input size, confidence threshold, and USB stream or replay clip.

Acceptance:

- TensorRT engine loads on Jetson.
- Benchmark reports average FPS, p50/p95 latency, GPU memory, CPU load, and
  dropped-frame count.
- Run manifest records Jetson platform, JetPack version, model path, and
  detector backend.

## Phase 3: Live USB Camera Source

Implementation tasks:

- Add a USB camera frame source.
- Preferred config shape:

```yaml
source:
  type: "opencv_camera"
  device: 0
  width: 1280
  height: 720
  fps: 30
  fourcc: "MJPG"
  strict: true
```

- Support `/dev/videoN` or integer device index.
- Timestamp frames from monotonic time.
- Log actual FPS, capture latency, dropped frames, resolution, and pixel
  format when available.
- Add record-only output so every flight can save raw video before detection
  failures are debugged.

Acceptance:

- USB camera runs for 10 minutes on Jetson without crashing.
- Output includes raw video, annotated video, target states, guidance hints,
  diagnostics, and manifest.
- Existing video-file replay remains unchanged.

## Phase 4: Log-Only Flight Test

This is the first real-air test mode.

Rules:

- Pilot/manual control only.
- Jetson sends no commands to the flight controller or ESP32.
- No MAVLink command messages.
- No actuator authority.
- Save logs even when no target is detected.

Flight outputs:

- raw camera video
- annotated video
- `target_states.jsonl`
- `guidance_hints.jsonl`
- optional `mock_bridge_proposals.jsonl`
- `diagnostics.csv`
- `manifest.json`
- Jetson performance summary

Post-flight report must include:

- detector geometric hit rate
- semantic drone hit rate
- false positives on clouds/birds/aircraft/background
- lock-state distribution
- stale tracker frames
- guidance-valid frame count
- latency p50/p95
- FPS and dropped frames

Acceptance:

- The flight can be replayed offline from saved video.
- The run can be annotated into a new hard-case packet.
- No command transport was active.

## Phase 5: Read-Only Flight Controller Telemetry

Only after Phase 4 is stable.

Implementation tasks:

- Add read-only telemetry logging from MAVLink if the FC path is known.
- Log attitude, altitude, GPS/local position, velocity, mode, armed state, and
  timestamp.
- Do not send commands.

Acceptance:

- Telemetry aligns with camera timestamps well enough for review.
- The logs can explain camera motion, target motion, and lock failures.
- No code path can arm, change mode, set velocity, or send yaw commands.

## Explicit Non-Goals For This Milestone

- Autonomous chase.
- Strike-ready or actuation testing.
- MAVLink control output.
- ESP32 command relay.
- Real GSD in meters without range.
- Relaxing lock/guidance gates to make demos look better.

## GSD And Range Policy

Camera calibration gives angular scale, not range.

Allowed in overlays now:

- bbox size in pixels
- bbox angular width/height in degrees
- bearing/elevation error
- `range: n/a`
- `GSD: n/a`

Allowed later only when explicitly configured:

- fixed debug range for bench review
- size-prior range estimate from known drone dimensions
- radar/lidar/GPS-derived range

Meter-per-pixel GSD must show its source. If range source is `NONE`, GSD must
not be displayed as a numeric value.

## Decision Gate Before Any Control Work

Do not begin MAVLink/ESP32 command output until all are true:

- Camera calibration reviewed and committed in config.
- Jetson TensorRT or PyTorch path meets latency target on live USB camera.
- Log-only flight data has been collected and reviewed.
- Semantic drone hit rate is acceptable on real flight footage.
- False positive behavior is understood.
- Manual override and abort rules are written down.
- Flight controller link and ESP32 role are fully specified.
