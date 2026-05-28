# SkyScouter Flight Control — The Plan, in Three Phases

*Last updated: 2026-05-28 · branch `feature/jetson-live-camera-runtime`*

This is the shared plan for turning SkyScouter from "a camera that locks onto a
drone" into "an aircraft that flies itself at the drone." We're doing it in
three steps so that each one can be flown and trusted before we add the next.

If you read nothing else, read this:

- **Phase 1 — turn to face the target. DONE and flown.**
- **Phase 2 — match its height (climb / descend).** ← colleague owns this
- **Phase 3 — close the distance (fly forward).** ← Ali owns this

Each phase is a small change in **one file** (`skyscouter/flight/mavlink_link.py`)
plus a couple of config knobs. The vision side — detection, tracking, lock — is
already done and doesn't change. All three phases just consume the same per-frame
`GuidanceHint` the pipeline already produces.

---

## How the whole thing fits together

The vision pipeline watches the camera, finds the drone, and every frame it
hands out one small message — a `GuidanceHint`. That message says, in plain
terms, *"the target is this many degrees to the right and this many degrees
up from the middle of the frame, and here's how confident/locked I am."*

Our job on the flight side is to read that message and move the aircraft to
shrink those numbers to zero.

```
   camera ─► detector ─► tracker ─► lock FSM ─► GuidanceHint ─► MavlinkFlightLink ─► Pixhawk
   (vision: already built, doesn't change)        │                (our 3 phases live here)
                                                   │
        bearing_error_deg   (left/right)  ─────────┼─► Phase 1: yaw  (turn to face)   ✅
        elevation_error_deg (up/down)     ─────────┼─► Phase 2: climb (match height)  ⬅ next
        "I am STRIKE_READY"               ─────────┴─► Phase 3: forward velocity       ⬅ after
```

The three phases stack. Phase 2 keeps Phase 1's yaw running and adds vertical
motion. Phase 3 keeps both and adds forward motion, but only once the target is
properly locked and centered (STRIKE_READY).

**One important note about how the aircraft is controlled.** Everything goes
over MAVLink to an ArduPilot autopilot (Pixhawk 6X) on a USB serial cable. We
put the aircraft in GUIDED mode and stream it setpoints. We do **not** send
land, return-to-home, or disarm from code — those stay on the pilot's RC switch
as the hardware safety net. That was true in the engineer's old `hand_track.py`
and it's still true here.

---

## Phase 1 — Turn to face the target  ✅ DONE

### What it does

The aircraft takes off to a fixed height, holds that height, and **yaws (rotates
its nose) to point at the locked drone.** It doesn't go up, down, forward, or
sideways. It just turns to face. That's the whole job of Phase 1, and it's
flying.

This is the same thing the engineer's `hand_track.py` did on the bench (turn to
face a hand), except the target now comes from the autonomous drone detector
instead of a person waving, and there's no mouse click anywhere.

### How it works under the hood

The flight link runs a background thread at a steady 30 Hz. When valid guidance
starts arriving, it walks through this sequence once:

1. **Switch to GUIDED mode** (`DO_SET_MODE`)
2. **Force-arm the motors** (`COMPONENT_ARM_DISARM`, param2 = `2989.0` — the
   ArduPilot "skip pre-arm checks" magic number)
3. **Take off** to the configured height (`NAV_TAKEOFF`, default 2 m)
4. **Settle** — hold a zero-yaw for a couple of seconds so it stabilizes

Then, every tick after that, it streams two messages:

- **Hold altitude** — `SET_POSITION_TARGET_LOCAL_NED` with type_mask `1531`.
  That mask means "I'm only commanding a Z position; ignore all X/Y velocity."
  In plain terms: stay at this height, don't drift forward or sideways.
- **Point at the target** — `CONDITION_YAW` with a relative heading. We take the
  target's left/right angle (`filtered_bearing_error_deg`), apply a 1° deadband
  so it doesn't twitch when basically centered, and clamp it to ±15° per command
  so it turns smoothly instead of snapping.

If guidance goes stale (no valid hint for more than 0.5 s) or the lock is lost,
it commands zero yaw and holds — it does not keep spinning toward a target it
can no longer see.

### The seven MAVLink messages (this is the entire vocabulary)

| Message | When | Why |
|---|---|---|
| `DO_SET_MODE` → GUIDED | once, at start | put the FC under our control |
| `COMPONENT_ARM_DISARM` (param2 = 2989.0) | once | force-arm the motors |
| `NAV_TAKEOFF` | once | climb to the fixed start height |
| `SET_POSITION_TARGET_LOCAL_NED` (mask 1531) | every tick | hold altitude, no XY motion |
| `CONDITION_YAW` (param4 = 1.0, relative) | every tick | turn the nose toward the target |
| `HEARTBEAT` | continuous | keep the link alive |
| `REQUEST_DATA_STREAM` | at start | ask the FC for telemetry back |

Those magic numbers (`2989.0`, `1531`) came straight from the proven
`hand_track.py` and are commented **"do not change."** Phases 2 and 3 will
*relax* the `1531` mask to allow vertical and forward motion — but only
deliberately, and only in the file where it's defined.

### The tests we ran (and what went wrong on the way)

Phase 1 wasn't a clean first try, so here's the honest record — it's useful for
Phases 2 and 3, because the same test discipline applies.

**Test 1 — Sign check on the bench (dry-run, 765 frames).**
We ran the full pipeline against recorded footage with the flight link in
**dry-run** (commands computed and logged, but the serial port never opened).
We then read `flight_commands.jsonl` and confirmed: when the target is to the
**right** of center, the commanded heading is **positive**; the deadband fired
correctly when the target crossed the middle. Conclusion: the sign convention is
correct, so we keep `invert_yaw: false`. **No aircraft involved — this is how you
prove direction safely before anything spins.**

**Test 2 — First live attempts FAILED (and why).**
On the first real `option 6` runs the drone did nothing — only heartbeats went
out, no arm, no takeoff. The autopilot was sitting in STABILIZE mode, and our
sender thread had a logic bug: it checked "am I in GUIDED?" and bailed out
*before* it ever sent the command to switch into GUIDED. Classic chicken-and-egg.
Fixed by sending `SET_MODE → GUIDED` **before** that gate, and we added verbose
`[flight]` logging that decodes every COMMAND_ACK to plain words
(ACCEPTED / DENIED / etc.) so we can see exactly what the FC is doing.

**Test 3 — Live flight SUCCESS.**
Next run: force-arm ACCEPTED, NAV_TAKEOFF to 2 m ACCEPTED, CONDITION_YAW
ACCEPTED. The aircraft lifted off to 2 m, held, and yawed toward the target. The
"unusual movements" we saw were exactly the intended yaw-to-target. Pilot landed
it on the RC switch. That's a working Phase 1.

**The takeaway for everyone:** dry-run first, read the JSONL, *then* fly. The
verbose `[flight]` log telling you ACCEPTED vs DENIED is your best friend.

### Where Phase 1 lives

- Code: `skyscouter/flight/mavlink_link.py` (the `MavlinkFlightLink` class)
- Wiring: `skyscouter/flight/factory.py` (builds it from config, lazy-imports pymavlink)
- Config: the `flight_control:` block in
  `configs/deploy_jetson_yolov26_lrdd_v2_siyi_a8_mini_1080p.yaml`
- Log: `flight_commands.jsonl` (one row per decision, dry-run or live flagged)

---

## Phase 2 — Match the target's height  ⬅ colleague owns this

### The goal

Right now the aircraft holds a fixed altitude. Phase 2 makes it **climb or
descend to line up vertically with the drone** — the same idea as Phase 1's yaw,
but for up/down instead of left/right. When you're done, the aircraft should keep
the target centered both horizontally (Phase 1) and vertically (Phase 2).

### The good news: the vision side is already done

You do **not** need to touch detection, tracking, or guidance. The pipeline
already computes the vertical angle for you and puts it in every `GuidanceHint`:

- `elevation_error_deg` — degrees the target is above (+) or below (−) center
- `filtered_elevation_error_deg` — the smoothed version (use this one, like
  Phase 1 uses `filtered_bearing_error_deg`)
- `pitch_rate_cmd_deg_s` — a pre-computed rate if you'd rather drive a rate

So Phase 2 is almost entirely a flight-side change. Your work happens in
`MavlinkFlightLink`.

### What you actually change

Today the altitude-hold message is built in `_send_altitude_hold_only()`:

```python
target_ned_z_m = -float(self._guided_alt_m)   # fixed height, NED: down is +
```

That's a constant. Phase 2 makes it move with the target. Two reasonable ways to
do it (pick one, document which):

- **Move the altitude setpoint (position).** Keep the same `1531` mask, but ramp
  `target_ned_z_m` up or down based on `filtered_elevation_error_deg`. More
  conservative — you're still commanding a position, just a changing one.
- **Command a climb rate (velocity).** Relax the `1531` type_mask to let the Z
  *velocity* (vz) term through, and set vz proportional to the elevation error,
  clamped to a safe rate. Smoother to control, but you're changing the mask, so
  be deliberate about it.

Either way: add a deadband (don't chase tiny errors), clamp hard (a max climb
rate / max altitude band), and make both limits **config values**, not constants.

### New config knobs to add (suggested names)

In the `flight_control:` block:

```yaml
  pitch_follow_enabled: true     # master switch for Phase 2
  max_climb_rate_m_s: 1.0        # clamp on vertical speed
  alt_min_m: 1.5                 # never command below this
  alt_max_m: 30.0                # never command above this
  elevation_deadband_deg: 1.0    # ignore tiny vertical errors
```

### How "done" looks

- Bench dry-run shows sane vz/altitude in `flight_commands.jsonl`: target high in
  frame → command up; target low → command down; centered → no vertical command.
- Climb rate and altitude band are clamped and come from config.
- A guarded live test (see "How to run"): the aircraft climbs/descends to keep the
  drone vertically centered while still yawing to face it.

### Test it the same way Phase 1 was tested

1. Dry-run (`option 5`), read the JSONL, confirm vertical **direction and clamps**
   before anything flies. Vertical motion near the ground is less forgiving than
   yaw, so be extra strict here.
2. Then a guarded live test with conservative clamps.

---

## Phase 3 — Close the distance  ⬅ Ali owns this

### The goal

With Phase 1 (facing) and Phase 2 (height) holding the target centered, Phase 3
adds **forward velocity** — the aircraft actually flies *at* the drone to close
the range. This is the business end, so it has the strictest gate.

### The key rule: only when STRIKE_READY

Forward motion only happens when the lock FSM reports **STRIKE_READY** — meaning
the target is locked, centered, big enough, and stable. In any other state
(TRACKING, LOCKED-but-not-centered, LOST), forward velocity is **zero**. We don't
fly forward at something we're not sure about.

### What you change

Forward motion means letting the **body-frame X velocity (vx)** through the
setpoint. Today the `1531` mask explicitly forbids all X/Y velocity. Phase 3
relaxes that mask (when STRIKE_READY) so vx is allowed, and sets vx to a
controlled closing speed.

- Build a STRIKE_READY-only path in the sender. Outside that state, vx = 0 and
  the aircraft behaves exactly like Phase 2.
- Closing speed should ramp, be clamped low at first, and be a config value.
- Keep yaw (Phase 1) and altitude/climb (Phase 2) running underneath — forward
  motion is added on top, not instead.

### New config knobs to add (suggested names)

```yaml
  forward_velocity_enabled: false   # master switch, OFF by default
  max_forward_speed_m_s: 1.0        # start slow; clamp hard
  strike_ready_only: true           # never move forward outside STRIKE_READY
  max_close_range_time_s: 5.0       # safety: stop closing after N seconds
```

### How "done" looks

- vx is **zero** in every state except STRIKE_READY (prove this in dry-run JSONL).
- Closing speed is clamped and configurable, starts conservative.
- A guarded live test shows the aircraft only moves forward when fully locked and
  centered, and stops the moment lock is lost.

---

## Rules we all follow

These kept Phase 1 safe; they apply to all three phases.

1. **Dry-run before you fly. Every time.** Run `option 5`, open
   `flight_commands.jsonl`, and confirm direction, signs, and clamps look right.
   Direction bugs are cheap to find on the bench and very expensive in the air.
2. **Edit only `mavlink_link.py`** for flight behavior. The way data comes in is
   `consume(hint)`. Everything else is config. You should almost never need to
   touch the vision pipeline.
3. **Tune through YAML, not code.** Limits, deadbands, switches — all live in the
   `flight_control:` block of the deploy config. Hard-coded numbers are how you
   forget what a build was doing.
4. **Don't change the magic numbers** (`2989.0`, `1531`) casually. Phase 2/3 will
   *relax* the `1531` mask on purpose — that's fine — but do it in the one place
   it's defined, with a comment saying why.
5. **The failsafe is "zero and hold."** Lost lock or stale guidance → stop
   commanding motion (zero yaw, hold altitude, zero forward). Land/RTL/disarm stay
   on the pilot's RC switch.
6. **Geo-fence is ArduPilot's job.** We rely on the autopilot's onboard geo-fence,
   not software in our pipeline.
7. **Clamp first, tune later.** Always ship a new phase with conservative limits.
   It's easy to loosen a clamp; it's hard to un-crash a drone.

---

## How to run and test

Everything launches from `jetson.sh`:

| What | Command | Notes |
|---|---|---|
| FC **dry-run** (safe) | `./jetson.sh run 5` | Full pipeline + flight logic; commands computed and logged but **never sent**. This is where you check signs and clamps. |
| FC **LIVE** | `./jetson.sh run 6` | Commands actually sent. Asks you to type `fly` to confirm; refuses to run without a terminal. |
| Live, higher takeoff | `./jetson.sh run 6 --fc-alt-m 5` | Sets a 5 m takeoff/hold height. |
| See the video while running | add `--operator-view-mode mjpeg` | View over WiFi in a browser; raw + annotated video are always saved anyway. |
| Headless field autostart | `./jetson.sh autostart` (option 14) | Installs a systemd user service so it runs on boot with no display. Use `SKYSCOUTER_MODEL=…` to skip the model picker. |

**One-time setup** (already done on the field Jetson, noted here for a fresh box):

```bash
pip install pymavlink pyserial        # not in requirements-jetson.txt by design
sudo usermod -aG dialout $USER        # serial port access; re-login after
```

### Where things are

| Thing | Path |
|---|---|
| The flight link (edit here) | `skyscouter/flight/mavlink_link.py` |
| Builds it from config | `skyscouter/flight/factory.py` |
| Config block to tune | `flight_control:` in `configs/deploy_jetson_yolov26_lrdd_v2_siyi_a8_mini_1080p.yaml` |
| The contract from vision | `GuidanceHint` in `skyscouter/schemas.py` |
| Per-flight decision log | `flight_commands.jsonl` (in the run's output folder) |
| Old reference code | `~/Documents/drone-control-jetson-main/hand_track.py` (reference only — new work lands in SkyScouter) |

---

## Housekeeping note

The docstring at the top of `mavlink_link.py` still labels the later stages
"Phase 4 / Phase 5." That's leftover numbering from an earlier draft — it should
read **Phase 2 (yaw + climb)** and **Phase 3 (yaw + climb + forward velocity)** to
match this plan. Whoever touches that file first, please fix the header so we're
all using the same numbers.
