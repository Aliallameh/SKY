# Skyscouter Phase 1 — Engineering Plan

This document is the engineering counterpart to PRD v3.2. It breaks the PRD into
concrete code tasks, sequenced so each delivers a working slice.

## Architecture (mirrors PRD §9)

```
Video / Camera         CUE_STATE
      │                    │
      ▼                    ▼
 ┌──────────┐         ┌──────────┐
 │  Source  │         │   Cue    │
 │ (frames) │         │ Adapter  │
 └────┬─────┘         └────┬─────┘
      │                    │
      └────────┬───────────┘
               ▼
        ┌────────────┐
        │  Detector  │   YOLO-class, swappable
        └─────┬──────┘
              ▼
        ┌────────────┐
        │  Tracker   │   ByteTrack + ego-motion stub
        └─────┬──────┘
              ▼
        ┌────────────┐
        │ Discrim.   │   Drone vs bird (kinematic)
        └─────┬──────┘
              ▼
        ┌────────────┐
        │ Lock State │   Lifecycle state machine (PRD §12)
        └─────┬──────┘
              ▼
        ┌────────────┐
        │ Publisher  │   target_state JSON (PRD §14)
        └─────┬──────┘
              ▼
        Logs + annotated video + JSONL stream
```

Each box is a Python module with a clean interface. Any box can be swapped
without touching the others.

## Task breakdown

### Sprint 0 — Bench replay foundation (THIS WORK)
The goal is: feed any video, get bbox + lock state + target_state JSON out.
Exactly the PRD G3 deliverable, minus live cue (cue is simulated for now).

- [x] T0.1: Project skeleton (configs, modules, tests, scripts)
- [x] T0.2: Frame source abstraction (video file, image folder, future: live)
- [x] T0.3: Detector wrapper (YOLO-class, config-driven, swappable backend)
- [x] T0.4: Tracker wrapper (ByteTrack-class, with ego-motion stub interface)
- [x] T0.5: Lock state machine (PRD §12 lifecycle)
- [x] T0.6: Target-state schema + publisher (PRD §14)
- [x] T0.7: CLI entry-point: `scripts/run_pipeline.py`
- [x] T0.8: Annotated video output with bbox + state overlay
- [x] T0.9: Replayable JSONL log per session
- [x] T0.10: Unit tests for state machine (deterministic, no model needed)
- [x] T0.11: README with quickstart

### Sprint 1 — Airborne discrimination + cue (ACTIVE NEXT)
- [x] T1.0: Sparse GT evaluation harness with IoU, center error, false-lock
  negatives, lock counts, and per-frame diagnostics.
- [x] T1.0a: Browser annotation packet for correcting sparse drone boxes and
  marking hard negatives.
- [x] T1.0b: Dataset manifest and YOLO training-set builder for airborne
  drone-vs-bird detector training.
- [ ] T1.1: Obtain/normalize airborne datasets: AOD-4 first, then
  Drone-vs-Bird, then LRDDv2 if access is available.
- [ ] T1.2: Train YOLO11s airborne detector with classes
  `drone,bird,airplane,helicopter,unknown_airborne`.
- [ ] T1.3: Wire trained detector weights into `yolo_ultralytics` config and
  replay against corrected sparse GT.
- [ ] T1.4: Tighten lock semantics so `guidance_valid=true` requires semantic
  drone confidence and cannot stay true on known hard-negative clutter.
- [ ] T1.5: Cue adapter + cue simulator (PRD FR-CUE-005)
- [ ] T1.6: Kinematic feature extractor on tracks (PRD FR-DISC-001)
- [ ] T1.7: Confidence calibration (FR-DET-002)

### Sprint 2 — Edge readiness
- T2.1: TensorRT export path
- T2.2: Latency stack-up measurement harness
- T2.3: Compute bake-off scripts (Orin / Orange Pi / RDK X5)

### Sprint 3 — Integration
- T3.1: MAVLink bridge (mock + real)
- T3.2: Operator HMI (minimum, web-based)
- T3.3: IMU ingestion path (real ego-motion compensation)

### Sprint 4 — Hardening
- T4.1: Fault injection test suite
- T4.2: STRIKE_READY gate validation
- T4.3: Drift detection (FR-CAM-007)

## Decisions baked into the code

These are committed; see PRD v3.2 §21.

- **Detector:** YOLO-family default. Backend abstracted via `BaseDetector`.
- **Tracker:** ByteTrack-class. Ego-motion compensation is an interface today
  (returns identity transform), wired to real IMU later.
- **Schema versions:** `skyscout.cue_state.v1`, `skyscout.onboard_target.v1`.
- **Lock states:** exactly the 9 states from PRD §12, no shortcuts.
- **Config:** all thresholds in YAML; no magic numbers in code.

## Honest limitations

- **Current pipeline fails corrected sparse GT.** On the 2026-04-28 corrected
  CSV for `my_drone_chase.MP4`, median IoU is 0.0, median center error is
  about 417 px, and there are 9 false-lock negative frames. This is now
  measured, not guessed.
- **Pretrained/heuristic detector is not acceptance-ready.** It produces some
  usable drone proposals, but the tracker/lock stack often stays on clutter.
  Fine-tuning a skyborne drone-vs-bird detector is required for credible Phase
  1 acceptance.
- **Range estimation is a stub.** Class-conditioned size-prior is implemented
  as an interface but disabled by default until target-size priors are
  committed in config.
- **Ego-motion compensation returns identity.** Wired to a real IMU later.
- **No GPU required to run.** Everything works on CPU; just slower.

## What "no cheating" means in this codebase

- No hardcoded bounding boxes
- No hardcoded "drone is at this position" priors
- No "if filename contains 'demo' then return x" shortcuts
- No silent fallback to fake detections
- Every threshold and parameter externalized in config
- Every output traceable via run_id and replayable from logs

## Quickstart for Ali & Erfan

```bash
# install
pip install -e .

# run on any video
python scripts/run_pipeline.py \
    --video data/videos/your_video.mp4 \
    --config configs/default.yaml \
    --output data/outputs/run_001

# outputs:
#   run_001/annotated.mp4       — video with bbox overlay
#   run_001/target_states.jsonl — one target_state per frame
#   run_001/run.log             — pipeline log
#   run_001/manifest.json       — run metadata for reproducibility
```

## Current handoff docs

See `docs/ML_TRAINING_AND_HANDOFF.md` for the active ML plan, corrected-GT
metrics, dataset selection rationale, training commands, and remaining work.
