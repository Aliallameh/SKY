# Visual Bearing Guidance And Mock Bridge Implementation

Last updated: 2026-05-01

Branch: `feature/visual-guidance-mock-bridge`

Commit: `155e4e3 Add visual bearing guidance and mock bridge`

## Why This Work Was Done

SkyScouter already produced tracked target state, lock-state decisions, JSONL
logs, and annotated video. The next deterministic perception/mechatronics layer
was to convert the tracked target bbox into camera-relative line-of-sight error
and a bounded yaw-alignment proposal.

This was implemented as a log-only capability because the project is not ready
for live flight-controller integration. The work intentionally does not send
MAVLink, open a control transport, actuate a drone, estimate intercept geometry,
or bypass existing lock/fault safety gates.

The mocked bridge was added as the next integration step: it consumes
`GuidanceHint` in-pipeline and writes auditable JSONL proposal/suppression rows
for bench review. It requires reviewed calibration before a row can be marked
`valid_for_transport`.

## Original Plan

1. Add a guidance package that converts tracked bbox center into bearing and
   elevation error using a pinhole camera model.
2. Add filtering, optional short-horizon center prediction, and a yaw-only
   bounded P-controller.
3. Keep the existing `TargetState` contract backward-compatible by adding a
   separate versioned `GuidanceHint` schema and JSONL output.
4. Integrate guidance after tracking and lock-state evaluation, then write
   `guidance_hints.jsonl` and optionally draw guidance overlay on annotated
   video.
5. Add config-driven camera, validity, filtering, controller, and overlay
   settings.
6. Add deterministic tests that do not require model weights.
7. Add a mocked bridge that consumes `GuidanceHint` and writes JSONL only,
   behind lock-state, fault, command-limit, and reviewed-calibration gates.

## What Was Implemented

### Visual Bearing Guidance

New package: `skyscouter/guidance/`

Implemented:

- `PinholeCameraModel` with explicit-intrinsics and FOV-derived modes.
- Bbox center conversion from tracked xyxy bbox.
- Bearing/elevation calculation:
  - positive bearing means target is right of optical center;
  - positive elevation means target is above optical center.
- EMA filtering over bearing/elevation.
- Optional center prediction from recent tracker center history.
- Yaw-only bounded P-controller:
  - deadband;
  - saturation;
  - optional per-frame command delta limit.

Output schema:

- `GuidanceHint`
- `schema_version: skyscout.guidance_hint.v1`
- JSONL path: `data/outputs/<run>/guidance_hints.jsonl`

Safety behavior:

- `valid_for_actuation` is always false.
- Invalid hints zero yaw command proposals.
- Guidance validity depends on tracked target, valid bbox, allowed lock state,
  `TargetState.guidance_valid`, confidence, staleness, class label, and fault
  flags.

### Pipeline Integration

Guidance is computed after `TargetState` creation:

```text
frame -> detector -> tracker -> lock state -> TargetState -> GuidanceHint -> outputs
```

The pipeline writes `guidance_hints.jsonl` when `guidance.enabled` and
`guidance.output_jsonl` are true.

The annotator can draw:

- optical-center crosshair;
- target-center dot;
- error vector;
- bearing/elevation/yaw-proposal text.

### Mock Bridge

New package: `skyscouter/bridge/`

Implemented:

- `MockGuidanceBridge`
- `BridgeProposal`
- `schema_version: skyscout.mock_bridge_proposal.v1`
- JSONL path: `data/outputs/<run>/mock_bridge_proposals.jsonl`

The mock bridge consumes in-memory `GuidanceHint` after guidance computation:

```text
TargetState -> GuidanceHint -> MockGuidanceBridge -> BridgeProposal JSONL
```

It writes both valid and suppressed rows for auditability.

A row can be `valid_for_transport=true` only when:

- source `GuidanceHint.valid` is true;
- source lock state is allowed;
- reviewed calibration is present when required;
- no source fault reason is present;
- yaw command is finite and within the bridge limit.

Suppressed rows have zero command values and explicit reason codes such as:

- `calibration_not_reviewed`
- `guidance_hint_not_valid`
- `lock_state_not_allowed:<STATE>`
- `fault_flags_active`
- `command_exceeds_bridge_limit`

The mock bridge does not send MAVLink, open UDP, open sockets, or actuate
anything.

## Config Added

Added to both `configs/default.yaml` and `configs/trained_yolo11s_eval.yaml`:

- `guidance.enabled`
- `guidance.output_jsonl`
- `guidance.camera`
- `guidance.validity`
- `guidance.filtering`
- `guidance.controller`
- `guidance.overlay`
- `mock_bridge.enabled`
- `mock_bridge.output_jsonl`
- `mock_bridge.require_reviewed_calibration`
- `mock_bridge.allowed_lock_states`
- `mock_bridge.require_guidance_hint_valid`
- `mock_bridge.max_abs_yaw_rate_deg_s`

Default safety posture:

- `guidance.enabled: true`
- `guidance.camera.calibration_reviewed: false`
- `mock_bridge.enabled: false`
- `mock_bridge.require_reviewed_calibration: true`

This means normal bench runs produce guidance hints and video overlay, while
the mock bridge must be explicitly enabled and will suppress transport-valid
rows until calibration is marked reviewed.

## CLI Added

Guidance overrides:

```powershell
--guidance-enabled
--no-guidance
--camera-hfov-deg <degrees>
```

Mock bridge overrides:

```powershell
--mock-bridge-enabled
--no-mock-bridge
```

## How To Run

Guidance overlay and guidance JSONL:

```powershell
.\.venv\Scripts\python.exe scripts/run_pipeline.py `
  --video data/videos/your_video.mp4 `
  --config configs/default.yaml `
  --output data/outputs/run_guidance_check
```

Guidance plus mock bridge:

```powershell
.\.venv\Scripts\python.exe scripts/run_pipeline.py `
  --video data/videos/your_video.mp4 `
  --config configs/default.yaml `
  --mock-bridge-enabled `
  --output data/outputs/run_mock_bridge_check
```

Expected outputs:

- `annotated.mp4`
- `target_states.jsonl`
- `guidance_hints.jsonl`
- `mock_bridge_proposals.jsonl` when the bridge is enabled
- `diagnostics.csv`
- `manifest.json`

## Tests Added

Guidance tests:

- camera center pixel gives zero bearing/elevation;
- target right of center gives positive bearing;
- FOV-derived focal length is correct;
- bbox center calculation;
- invalid bbox produces invalid guidance;
- yaw controller deadband and saturation;
- invalid guidance zeros command;
- EMA smoothing preserves sign;
- center prediction moves in expected direction;
- guidance JSONL writer writes valid rows.

Mock bridge tests:

- reviewed calibration plus valid hint produces `valid_for_transport=true`;
- unreviewed calibration suppresses proposal;
- invalid guidance suppresses and zeros command;
- unsafe lock state suppresses;
- fault reason suppresses;
- yaw command above bridge limit is rejected with
  `command_exceeds_bridge_limit`.

Pipeline smoke tests:

- guidance hints are written from synthetic tracked targets;
- mock bridge writes suppressed rows when calibration is unreviewed;
- mock bridge writes transport-valid rows when calibration is reviewed;
- bridge enabled while guidance disabled fails loudly.

Latest verification:

```text
53 passed
```

## Remaining Work

1. Run a short real bench replay and visually inspect `annotated.mp4`.
2. Replace bench FOV defaults with measured camera intrinsics or reviewed HFOV.
3. Set `guidance.camera.calibration_reviewed: true` only after human review.
4. Add a replay summary tool that reports valid/suppressed guidance and bridge
   counts, max yaw proposal, and suppression reason distribution.
5. Only after bench evidence exists, design a separate mocked MAVLink adapter.
   Live MAVLink remains intentionally unimplemented.
