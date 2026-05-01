# Visual Bearing Guidance

Visual bearing guidance converts the tracked target bounding box into a
camera-relative line-of-sight error and a bounded yaw-rate command proposal.
It is a log-only bench/simulation artifact. The current implementation does
not send MAVLink, actuate a vehicle, or authorize any autonomous action.

## Math

The guidance package uses a pinhole camera model. For a target center pixel
`(u, v)` and camera intrinsics `(fx, fy, cx, cy)`:

```text
bearing_rad   = atan((u - cx) / fx)
elevation_rad = -atan((v - cy) / fy)
```

The negative sign on elevation makes the convention explicit: positive
elevation error means the target is above optical center. Positive bearing
means the target is to the right of optical center. Positive yaw-rate proposal
therefore means yaw right.

The package supports two camera modes:

- `intrinsics`: use explicit `fx_px`, `fy_px`, `cx_px`, `cy_px`.
- `fov`: derive `fx` from horizontal FOV and frame width. If vertical FOV is
  missing, `fy = fx` is used as a bench-replay approximation. Replace this
  with calibrated intrinsics before flight integration.

## Output

When enabled, the pipeline writes:

```text
data/outputs/<run>/guidance_hints.jsonl
```

Each row has schema version `skyscout.guidance_hint.v1` and includes target
center, optional predicted center, frame center, pixel/normalized error,
bearing/elevation in radians and degrees, filtered angles, and yaw-rate
proposal. `valid_for_actuation` is always false.

If `mock_bridge.enabled: true`, the pipeline also writes:

```text
data/outputs/<run>/mock_bridge_proposals.jsonl
```

Those rows are transport-neutral audit records. They are not MAVLink messages
and are not sent over any transport.

## Validity Gates

A hint is valid only when configured gates pass:

- a tracked target bbox exists and is finite;
- frame dimensions are known;
- lock state is in `guidance.validity.allowed_lock_states`;
- `TargetState.guidance_valid` is true unless advisory bench mode is enabled;
- class label is allowed;
- confidence is above threshold;
- track staleness is below threshold;
- no fault flags are active.

`advisory_before_lock: true` can be used for bench replay. It permits logging
pre-lock line-of-sight estimates but still leaves `valid_for_actuation=false`.

## Filtering And Prediction

The default filter is an exponential moving average over bearing/elevation.
Optional center prediction uses recent tracker center history:

```text
predicted_center = current_center + velocity_px_per_s * lead_time_s
```

The prediction is bounded by `max_prediction_px` and is intended only to make
bench review more stable. It does not estimate range or intercept geometry.

## Controller

The first controller is yaw-only proportional control:

```text
yaw_rate_cmd_deg_s = kp_yaw * filtered_bearing_error_deg
```

It applies a deadband, saturation, and optional per-frame rate change limit.
Pitch command is disabled by default; elevation error is calculated for
review but not converted to a pitch command unless a future reviewed design
adds that path.

## Limitations

- Bounding box center gives line-of-sight bearing only, not range.
- Bench FOV defaults are not flight calibration.
- The proposal is not sent to an autopilot.
- No forward velocity, pursuit guidance, intercept geometry, or action
  authorization is implemented.

## Mock Bridge

The mock bridge consumes each in-memory `GuidanceHint` after the pipeline
publishes the hint. It emits one `skyscout.mock_bridge_proposal.v1` JSONL row
per hint when enabled. Rows can be `valid_for_transport=true` only when:

- the source hint is valid;
- the source lock state is allowed by `mock_bridge.allowed_lock_states`;
- no source fault reason is present;
- the yaw-rate proposal is finite and within `max_abs_yaw_rate_deg_s`;
- reviewed calibration is present when `require_reviewed_calibration` is true.

Default configs set `mock_bridge.enabled: false` and
`guidance.camera.calibration_reviewed: false`, so enabling the bridge on bench
FOV defaults produces explicit suppression rows with
`calibration_not_reviewed`.

## Future MAVLink Integration

A later live integration can consume reviewed mock bridge output or the
in-memory proposal object in a separate MAVLink bridge. Any live bridge must
remain behind the existing lock-state, calibration, and fault gates and should
require bench replay evidence before flight testing.
