"""MAVLink flight-controller link for SkyScouter (ArduPilot copter, Phase 1).

Phase 1 scope (verbatim parity with the drone-control bench code):
    * Yaw-only tracking via MAV_CMD_CONDITION_YAW (relative heading).
    * Altitude hold via SET_POSITION_TARGET_LOCAL_NED with type_mask 1531
      (Z position only; X/Y velocity always zero on the wire).
    * Auto sequence: enter GUIDED -> force-arm -> NAV_TAKEOFF to fixed alt
      -> post-takeoff settle (yaw held at 0).
    * Pilot/GCS override detection: stop streaming setpoints when the FC
      reports any non-GUIDED mode (LAND/RTL/STABILIZE/AUTO/etc).
    * Stale-guidance failsafe: command zero yaw if no valid GuidanceHint
      arrives within stale_guidance_timeout_s (default 0.5s).
    * Keyboard inhibit hook (Phase 3 wiring): consume() obeys the
      keyboard_inhibit flag but operator-view binding is added later.
    * Dry-run default: when cfg.dry_run=True the serial port is never
      opened.  Commands are computed and logged exactly as they would be
      sent, with a `dry_run=true` flag in each JSONL row.

The ArduPilot-specific magic numbers (force-arm param2=2989.0, alt-hold
type_mask=1531) are imported verbatim from the proven hand_track.py
implementation -- DO NOT change them without checking ArduPilot source.

This module imports pymavlink lazily inside connect() so the pipeline
can run without pymavlink installed when flight_control.enabled=false.

Interception model (full plan, not Phase 1):
    Phase 1 (here):   yaw + alt-hold
    Phase 4 (later):  yaw + pitch
    Phase 5 (later):  yaw + pitch + body-frame forward velocity on
                      STRIKE_READY (drop the no-XY-velocity type-mask bit)
"""
from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO

from ..schemas import GuidanceHint

# ---------------------------------------------------------------------------
# ArduPilot-specific magic numbers (hard-won; do not change).
# ---------------------------------------------------------------------------

# SET_POSITION_TARGET_LOCAL_NED type_mask: Z position only, X/Y velocity and
# yaw_rate always zero on the wire.  This is altitude hold with no XY motion.
# Bit decoding: ignore vx, vy, ax, ay, az, force, yaw, yaw_rate.  Z position
# active.  Equivalent to bits: 0x05FB.
ARDUPILOT_ALT_HOLD_ONLY_TYPE_MASK = 1531

# MAV_CMD_COMPONENT_ARM_DISARM param2 magic for ArduPilot "force arm"
# (bypasses pre-arm checks).  Without this, the FC rejects ARM commands
# whenever any pre-arm check is failing (GPS lock, compass, etc).
ARDUPILOT_FORCE_ARM_PARAM2 = 2989.0


# ---------------------------------------------------------------------------
# Command record (one per MAVLink-yaw command attempt; written to JSONL)
# ---------------------------------------------------------------------------

@dataclass
class FlightCommand:
    frame_id: int
    timestamp_utc: str
    run_id: Optional[str]
    track_id: Optional[int]
    lock_state: Optional[str]
    guidance_valid: bool

    # Computed command (what we WOULD send; for dry_run=True, never reaches FC)
    target_heading_deg: float
    yaw_slew_deg_s: float

    # State observed from FC at command time
    fc_connected: bool
    fc_flight_mode: str
    fc_is_armed: bool
    fc_takeoff_complete: bool
    fc_pilot_override: bool
    fc_relative_alt_m: Optional[float]

    # Why this command (or its suppression) happened
    reason: List[str] = field(default_factory=list)

    # Was this actually transmitted on the wire?
    sent: bool = False
    dry_run: bool = True

    def to_jsonl(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))


# ---------------------------------------------------------------------------
# MavlinkFlightLink
# ---------------------------------------------------------------------------

class MavlinkFlightLink:
    """ArduPilot GUIDED-mode flight-controller link consuming GuidanceHints.

    Lifecycle:
        link = MavlinkFlightLink(cfg=..., output_dir=..., run_id=...)
        link.connect()                # opens serial unless dry_run=True
        for guidance_hint in pipeline_loop:
            link.consume(guidance_hint)
        link.close()

    Thread-safety:
        Public consume() is called from the main pipeline thread.
        Background sender thread runs at send_hz (default 30Hz).
        Both threads access shared state under self._target_lock.
    """

    def __init__(
        self,
        *,
        cfg: Dict[str, Any],
        output_dir: Path,
        run_id: Optional[str],
    ) -> None:
        self._cfg = dict(cfg)
        self._run_id = run_id

        # ---- top-level enable / dry-run ----
        self._enabled = bool(self._cfg.get("enabled", False))
        self._dry_run = bool(self._cfg.get("dry_run", True))

        # ---- serial / link parameters ----
        self._serial_port = str(self._cfg.get("serial_port", "/dev/ttyACM0"))
        self._baud = int(self._cfg.get("baud", 115200))
        self._heartbeat_timeout_s = float(self._cfg.get("heartbeat_timeout_s", 8.0))

        # ---- GUIDED / takeoff parameters ----
        self._guided_alt_m = max(0.5, float(self._cfg.get("guided_alt_m", 2.0)))
        self._post_takeoff_settle_s = max(0.0, float(self._cfg.get("post_takeoff_settle_s", 4.0)))
        self._no_alt_takeoff_confirm_s = max(0.5, float(self._cfg.get("no_alt_takeoff_confirm_s", 3.0)))

        # ---- yaw command parameters ----
        self._max_heading_deg = max(1.0, float(self._cfg.get("max_heading_deg", 15.0)))
        self._yaw_slew_deg_s = max(1.0, float(self._cfg.get("yaw_slew_deg_s", 15.0)))
        self._yaw_deadband_deg = max(0.0, float(self._cfg.get("yaw_deadband_deg", 1.0)))
        self._invert_yaw = bool(self._cfg.get("invert_yaw", False))
        self._send_hz = max(1.0, float(self._cfg.get("send_hz", 30.0)))

        # ---- failsafe parameters ----
        self._stale_guidance_timeout_s = float(self._cfg.get("stale_guidance_timeout_s", 0.5))
        self._command_delay_s = max(0.0, float(self._cfg.get("command_delay_s", 3.0)))

        # ---- lock-state gating ----
        # Only command yaw when the lock state is in this set.  Matches the
        # default GuidanceController allowed_lock_states.
        allowed = self._cfg.get("allowed_lock_states", ["TRACKING", "LOCKED", "STRIKE_READY"])
        self._allowed_lock_states = {str(s).upper().strip() for s in allowed}

        # ---- I/O ----
        self._log_path = Path(output_dir) / str(self._cfg.get("log_filename", "flight_commands.jsonl"))
        self._log_fh: Optional[TextIO] = None
        if self._enabled and bool(self._cfg.get("log_jsonl", True)):
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_fh = open(self._log_path, "a", buffering=1)

        # ---- shared state ----
        self._target_lock = threading.Lock()
        self._target_yaw_heading_deg = 0.0
        self._last_guidance_at = 0.0
        self._keyboard_inhibit = False

        # FC link / state machine state
        self._master = None
        self._mavutil = None
        self._target_system = 1
        self._target_component = 1
        self._connected = False
        self._flight_mode = ""
        self._guided = False
        self._fc_is_armed = False
        self._have_fc_armed_state = False
        self._takeoff_complete = False
        self._takeoff_complete_at = 0.0
        self._guided_takeoff_sent = False
        self._last_takeoff_cmd_time = 0.0
        self._first_takeoff_cmd_time = 0.0
        self._relative_alt_m: Optional[float] = None
        self._seen_global_position = False
        self._user_override = False
        self._last_guided_req_time = 0.0
        self._last_force_arm_req_time = 0.0
        self._last_takeoff_ack_result: Optional[int] = None
        self._last_error = ""
        self._last_yaw_block_reason = ""

        # tx accounting
        self._tx_count = 0
        self._takeoff_tx_count = 0
        self._last_send_time = 0.0
        self._last_hb_send_time = 0.0
        self._last_tx_time = 0.0

        # threading primitives
        self._mav_send_lock = threading.Lock()
        self._sender_stop = threading.Event()
        self._sender_thread: Optional[threading.Thread] = None

        # connect() is deferred to start() / first consume() so the pipeline
        # can build the link object even when flight_control is fully disabled.
        self._activate_at_monotonic = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def dry_run(self) -> bool:
        return self._dry_run

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def takeoff_complete(self) -> bool:
        return self._takeoff_complete

    @property
    def flight_mode(self) -> str:
        return self._flight_mode

    def set_keyboard_inhibit(self, active: bool) -> None:
        """Operator emergency inhibit (Phase 3 wiring point).

        Sets a flag that immediately zeros the commanded yaw heading on the
        next sender-thread cycle.  Does NOT disarm or change mode -- that's
        the pilot's job via RC.  This is a soft inhibit for vision-stack
        confidence loss, not a kill switch.
        """
        with self._target_lock:
            self._keyboard_inhibit = bool(active)
            if active:
                self._target_yaw_heading_deg = 0.0

    def start(self) -> bool:
        """Connect to the FC and start the background sender thread.

        Returns True if the link is ready to consume guidance hints.  In
        dry-run mode this is always True (no serial connection attempted).
        """
        if not self._enabled:
            return False

        self._activate_at_monotonic = time.monotonic() + self._command_delay_s

        if self._dry_run:
            # No FC connection in dry-run.  Start a stub sender thread that
            # just logs what *would* be sent.  Useful for verifying our
            # GuidanceHint -> heading mapping against recorded runs.
            self._sender_thread = threading.Thread(
                target=self._dry_run_logger_loop,
                name="skyscouter-flight-link-dry",
                daemon=True,
            )
            self._sender_thread.start()
            return True

        if not self._connect():
            return False

        self._sender_thread = threading.Thread(
            target=self._sender_loop,
            name="skyscouter-flight-link",
            daemon=True,
        )
        self._sender_thread.start()
        return True

    def consume(self, hint: Optional[GuidanceHint]) -> None:
        """Main entry point called once per pipeline frame.

        The pipeline already produces a fully-processed GuidanceHint with
        validity flags, lock-state context, and (optionally) a filtered
        bearing error in degrees.  We map it to a relative-yaw command and
        hand it to the background sender thread for transmission.
        """
        if not self._enabled or hint is None:
            return

        now_m = time.monotonic()
        reasons: List[str] = []

        # ---- gating ----
        guidance_valid = bool(hint.valid)
        if not guidance_valid:
            reasons.append("hint.invalid")

        lock_state = (hint.source_lock_state or "").upper().strip()
        if lock_state and lock_state not in self._allowed_lock_states:
            reasons.append(f"lock_state_not_allowed:{lock_state}")
            guidance_valid = False

        # ---- compute target heading ----
        # Use filtered bearing error if available, otherwise raw.  Positive
        # bearing_error_deg = target right of optical center (our schema).
        bearing_deg = hint.filtered_bearing_error_deg
        if bearing_deg is None:
            bearing_deg = hint.bearing_error_deg
        target_heading = 0.0
        if guidance_valid and bearing_deg is not None:
            target_heading = float(bearing_deg)
            if self._invert_yaw:
                target_heading = -target_heading
            # Deadband
            if abs(target_heading) <= self._yaw_deadband_deg:
                reasons.append("deadband")
                target_heading = 0.0
            else:
                # Clamp to per-command max
                target_heading = max(
                    -self._max_heading_deg,
                    min(self._max_heading_deg, target_heading),
                )

        # ---- push to sender ----
        with self._target_lock:
            self._target_yaw_heading_deg = target_heading
            self._last_guidance_at = now_m
            if self._keyboard_inhibit:
                self._target_yaw_heading_deg = 0.0
                reasons.append("keyboard_inhibit")

        # ---- log row ----
        if self._log_fh is not None:
            cmd = FlightCommand(
                frame_id=int(hint.frame_id),
                timestamp_utc=hint.timestamp_utc,
                run_id=self._run_id,
                track_id=hint.track_id,
                lock_state=lock_state or None,
                guidance_valid=guidance_valid,
                target_heading_deg=target_heading,
                yaw_slew_deg_s=self._yaw_slew_deg_s,
                fc_connected=self._connected,
                fc_flight_mode=self._flight_mode,
                fc_is_armed=self._fc_is_armed,
                fc_takeoff_complete=self._takeoff_complete,
                fc_pilot_override=self._user_override,
                fc_relative_alt_m=self._relative_alt_m,
                reason=reasons,
                sent=False,  # updated by sender thread for live; False for dry_run
                dry_run=self._dry_run,
            )
            try:
                self._log_fh.write(cmd.to_jsonl() + "\n")
            except Exception:
                pass

    def close(self) -> None:
        self._sender_stop.set()
        if self._sender_thread is not None and self._sender_thread.is_alive():
            self._sender_thread.join(timeout=0.5)
        if self._master is not None:
            try:
                # final zero-yaw command for safety
                self._send_condition_yaw(0.0)
            except Exception:
                pass
            try:
                self._master.close()
            except Exception:
                pass
        if self._log_fh is not None:
            try:
                self._log_fh.flush()
                self._log_fh.close()
            except Exception:
                pass
            self._log_fh = None
        self._connected = False

    # ------------------------------------------------------------------
    # Background sender (DRY-RUN variant)
    # ------------------------------------------------------------------

    def _dry_run_logger_loop(self) -> None:
        """In dry-run mode: no serial, but tick at send_hz and observe stale
        guidance.  Useful for verifying our GuidanceHint -> heading mapping
        against recorded runs."""
        send_interval = 1.0 / self._send_hz
        while not self._sender_stop.wait(send_interval):
            now_m = time.monotonic()
            with self._target_lock:
                stale = (now_m - self._last_guidance_at) > self._stale_guidance_timeout_s
                if stale:
                    self._target_yaw_heading_deg = 0.0

    # ------------------------------------------------------------------
    # Background sender (LIVE variant) -- ports MavlinkYawLink._sender_loop
    # ------------------------------------------------------------------

    def _sender_loop(self) -> None:
        send_interval = 1.0 / self._send_hz
        while not self._sender_stop.wait(send_interval):
            try:
                self._sender_tick()
            except Exception as exc:
                self._last_error = f"sender tick error: {exc}"

    def _sender_tick(self) -> None:
        if not self._connected or self._master is None or self._mavutil is None:
            return

        now_m = time.monotonic()

        # ---- 1Hz companion heartbeat (so GCS can see us) ----
        if (now_m - self._last_hb_send_time) >= 1.0:
            self._send_companion_heartbeat()
            self._last_hb_send_time = now_m

        # ---- drain incoming messages (flight_mode, armed, relative_alt) ----
        self._poll_fc_messages()

        # ---- pilot/GCS explicitly took us out of GUIDED -> back off ----
        # If _user_override is set (we WERE in GUIDED and got kicked out), do
        # not fight the pilot.  But if we've simply never been in GUIDED yet
        # (e.g. FC boots in STABILIZE), we should TRY to enter GUIDED — that's
        # what option 6 is for.
        if self._user_override:
            return

        # ---- command-start delay (gives operator time to abort) ----
        if now_m < self._activate_at_monotonic:
            return

        # ---- request GUIDED if not there yet ----
        # This MUST happen before the "not guided -> return" gate, otherwise
        # we just sit forever sending heartbeats while the FC stays in whatever
        # mode it booted in (STABILIZE / LOITER / etc).  Throttled to once
        # every 0.7s so we don't spam.
        if not self._guided:
            if (now_m - self._last_guided_req_time) >= 0.7:
                self._send_set_guided()
                self._last_guided_req_time = now_m
                import sys as _sys
                if self._last_yaw_block_reason != "requesting_guided":
                    msg = (
                        f"FC: requesting GUIDED (currently '{self._flight_mode or '?'}'); "
                        "if FC stays in this mode, the pilot's RC mode-switch is forcing it. "
                        "Map one switch position to GUIDED in Mission Planner."
                    )
                    self._last_error = msg
                    print(f"[flight] {msg}", file=_sys.stderr, flush=True)
                    self._last_yaw_block_reason = "requesting_guided"
                else:
                    print(
                        f"[flight] SET_MODE GUIDED sent (FC still in '{self._flight_mode or '?'}')",
                        file=_sys.stderr, flush=True,
                    )
            return   # wait for next HEARTBEAT to confirm GUIDED

        # Reached GUIDED for the first time — announce it
        if self._last_yaw_block_reason == "requesting_guided":
            import sys as _sys
            print(f"[flight] FC entered GUIDED mode  armed={self.is_armed()}",
                  file=_sys.stderr, flush=True)
            self._last_yaw_block_reason = ""

        # ---- now in GUIDED -> ARM -> TAKEOFF state machine ----
        if not self.is_armed():
            self._force_arm_step()
            return
        self._mark_takeoff_complete_if_airborne()
        if not self._takeoff_complete:
            self._ensure_guided_takeoff()
            return
        if self._post_takeoff_settle_remaining_s() > 0.0:
            self._send_altitude_hold_only()
            self._send_condition_yaw(0.0)
            return

        # ---- failsafe: stale guidance -> zero yaw ----
        with self._target_lock:
            stale = (now_m - self._last_guidance_at) > self._stale_guidance_timeout_s
            target_heading = 0.0 if stale else self._target_yaw_heading_deg

        # ---- transmit ----
        self._send_altitude_hold_only()
        self._send_condition_yaw(target_heading)
        self._tx_count += 1
        self._last_tx_time = now_m

    # ------------------------------------------------------------------
    # MAVLink connection / heartbeat / message handling
    # ------------------------------------------------------------------

    def _connect(self) -> bool:
        try:
            from pymavlink import mavutil  # type: ignore[import-not-found]
        except ImportError:
            self._last_error = (
                "pymavlink is not installed.  Run: pip install pymavlink "
                "(or use menu option 11 to sync requirements)."
            )
            return False

        self._mavutil = mavutil
        try:
            self._master = mavutil.mavlink_connection(
                self._serial_port,
                baud=self._baud,
                autoreconnect=True,
                source_system=250,
            )
            deadline = time.monotonic() + self._heartbeat_timeout_s
            while time.monotonic() < deadline:
                hb = self._master.recv_match(type="HEARTBEAT", blocking=True, timeout=0.5)
                if hb is None:
                    continue
                autopilot = int(getattr(hb, "autopilot", mavutil.mavlink.MAV_AUTOPILOT_INVALID))
                if autopilot == mavutil.mavlink.MAV_AUTOPILOT_INVALID:
                    continue
                # Real autopilot heartbeat received; use its system id.
                self._target_system = int(getattr(hb, "get_srcSystem", lambda: 1)())
                self._target_component = int(getattr(hb, "get_srcComponent", lambda: 1)())
                self._connected = True
                # Request streams we'll need.
                try:
                    self._master.mav.request_data_stream_send(
                        self._target_system,
                        self._target_component,
                        mavutil.mavlink.MAV_DATA_STREAM_POSITION,
                        4,  # Hz
                        1,
                    )
                    self._master.mav.request_data_stream_send(
                        self._target_system,
                        self._target_component,
                        mavutil.mavlink.MAV_DATA_STREAM_EXTRA2,
                        4,
                        1,
                    )
                except Exception:
                    pass
                self._refresh_flight_mode()
                return True
        except Exception as exc:
            self._last_error = f"MAVLink connect failed: {exc}"
            return False

        self._last_error = "MAVLink heartbeat timeout"
        return False

    def _send_companion_heartbeat(self) -> None:
        if self._master is None or self._mavutil is None:
            return
        try:
            with self._mav_send_lock:
                self._master.mav.heartbeat_send(
                    self._mavutil.mavlink.MAV_TYPE_ONBOARD_CONTROLLER,
                    self._mavutil.mavlink.MAV_AUTOPILOT_INVALID,
                    0,
                    0,
                    self._mavutil.mavlink.MAV_STATE_ACTIVE,
                )
        except Exception as exc:
            self._last_error = f"heartbeat send failed: {exc}"

    def _refresh_flight_mode(self) -> None:
        if self._master is None or self._mavutil is None:
            return
        try:
            mode = self._master.flightmode
        except Exception:
            mode = ""
        prev = self._flight_mode
        self._flight_mode = str(mode or "")
        guided_modes = ("GUIDED", "GUIDED_NOGPS")
        prev_was_guided = prev in guided_modes
        now_is_guided = self._flight_mode in guided_modes
        self._guided = now_is_guided
        if prev_was_guided and not now_is_guided:
            # Pilot/GCS took over -- stop commanding.
            self._user_override = True
        elif now_is_guided and not prev_was_guided:
            # Returned to GUIDED.
            self._user_override = False

    def _poll_fc_messages(self) -> None:
        if self._master is None or self._mavutil is None:
            return
        try:
            while True:
                msg = self._master.recv_match(blocking=False)
                if msg is None:
                    break
                self._handle_message(msg)
        except Exception:
            pass

    def _handle_message(self, msg: Any) -> None:
        assert self._mavutil is not None
        mtype = msg.get_type()
        if mtype == "HEARTBEAT":
            try:
                self._refresh_flight_mode()
                base_mode = int(getattr(msg, "base_mode", 0))
                new_armed = bool(base_mode & self._mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
                self._fc_is_armed = new_armed
                self._have_fc_armed_state = True
            except Exception:
                pass
        elif mtype == "GLOBAL_POSITION_INT":
            try:
                self._relative_alt_m = float(getattr(msg, "relative_alt", 0)) / 1000.0
                self._seen_global_position = True
                self._mark_takeoff_complete_if_airborne()
            except Exception:
                pass
        elif mtype == "COMMAND_ACK":
            try:
                cmd = int(getattr(msg, "command", -1))
                result = int(getattr(msg, "result", -1))
                import sys as _sys
                # Decode common results for human-readable logging
                mav = self._mavutil.mavlink
                result_name = {
                    mav.MAV_RESULT_ACCEPTED: "ACCEPTED",
                    mav.MAV_RESULT_TEMPORARILY_REJECTED: "TEMP_REJECTED",
                    mav.MAV_RESULT_DENIED: "DENIED",
                    mav.MAV_RESULT_UNSUPPORTED: "UNSUPPORTED",
                    mav.MAV_RESULT_FAILED: "FAILED",
                    mav.MAV_RESULT_IN_PROGRESS: "IN_PROGRESS",
                }.get(result, f"result={result}")
                cmd_name = {
                    mav.MAV_CMD_DO_SET_MODE: "SET_MODE",
                    mav.MAV_CMD_COMPONENT_ARM_DISARM: "ARM_DISARM",
                    mav.MAV_CMD_NAV_TAKEOFF: "NAV_TAKEOFF",
                    mav.MAV_CMD_CONDITION_YAW: "CONDITION_YAW",
                }.get(cmd, f"cmd={cmd}")
                print(f"[flight] FC ACK {cmd_name} -> {result_name}",
                      file=_sys.stderr, flush=True)
                if cmd == mav.MAV_CMD_NAV_TAKEOFF:
                    self._last_takeoff_ack_result = result
            except Exception:
                pass
        elif mtype == "STATUSTEXT":
            text = getattr(msg, "text", "")
            if text:
                self._last_error = str(text)

    # ------------------------------------------------------------------
    # GUIDED + ARM + TAKEOFF state machine
    # ------------------------------------------------------------------

    def is_armed(self) -> bool:
        if self._master is not None:
            try:
                if self._master.motors_armed():
                    return True
            except Exception:
                pass
        return bool(self._fc_is_armed) if self._have_fc_armed_state else False

    def _send_set_guided(self) -> None:
        if self._master is None or self._mavutil is None:
            return
        mapping = self._master.mode_mapping() or {}
        if "GUIDED" not in mapping:
            return
        with self._mav_send_lock:
            self._master.mav.command_long_send(
                self._target_system,
                self._target_component,
                self._mavutil.mavlink.MAV_CMD_DO_SET_MODE,
                0,
                float(self._mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED),
                float(mapping["GUIDED"]),
                0.0, 0.0, 0.0, 0.0, 0.0,
            )

    def _force_arm_step(self, interval_s: float = 0.35) -> None:
        # Caller (_sender_tick) ensures we're already in GUIDED before calling this.
        if self._master is None or self._mavutil is None or not self._guided:
            return
        now_m = time.monotonic()
        if (now_m - self._last_force_arm_req_time) < interval_s:
            return
        try:
            with self._mav_send_lock:
                self._master.mav.command_long_send(
                    self._target_system,
                    self._target_component,
                    self._mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                    0,
                    1.0,
                    ARDUPILOT_FORCE_ARM_PARAM2,
                    0.0, 0.0, 0.0, 0.0, 0.0,
                )
            self._last_force_arm_req_time = now_m
            import sys as _sys
            print("[flight] sent force-ARM (param2=2989)", file=_sys.stderr, flush=True)
        except Exception as exc:
            self._last_error = f"force arm failed: {exc}"

    def _ensure_guided_takeoff(self) -> None:
        if self._master is None or self._mavutil is None:
            return
        if self._takeoff_complete:
            return
        now_m = time.monotonic()
        # Don't spam NAV_TAKEOFF -- ArduPilot ignores duplicates anyway.
        if self._guided_takeoff_sent and (now_m - self._last_takeoff_cmd_time) < 5.0:
            return
        try:
            with self._mav_send_lock:
                self._master.mav.command_long_send(
                    self._target_system,
                    self._target_component,
                    self._mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
                    0,
                    0.0,   # min pitch
                    0.0, 0.0, 0.0,
                    0.0,   # lat (unused)
                    0.0,   # lon (unused)
                    float(self._guided_alt_m),
                )
            self._guided_takeoff_sent = True
            self._last_takeoff_cmd_time = now_m
            if self._first_takeoff_cmd_time <= 0.0:
                self._first_takeoff_cmd_time = now_m
            self._takeoff_tx_count += 1
            import sys as _sys
            print(
                f"[flight] sent NAV_TAKEOFF #{self._takeoff_tx_count} "
                f"to alt={self._guided_alt_m:.1f}m AGL",
                file=_sys.stderr, flush=True,
            )
        except Exception as exc:
            self._last_error = f"NAV_TAKEOFF failed: {exc}"

    def _mark_takeoff_complete_if_airborne(self) -> bool:
        if self._takeoff_complete or not self.is_armed():
            return self._takeoff_complete
        rel_alt = self._relative_alt_m
        if rel_alt is None:
            # No telemetry yet.  Fall back to "in GUIDED+ARM for N seconds"
            # confirmation, same as their code.
            if (
                self._guided_takeoff_sent
                and self._first_takeoff_cmd_time > 0.0
                and self._guided
                and (time.monotonic() - self._first_takeoff_cmd_time)
                >= self._no_alt_takeoff_confirm_s
            ):
                self._set_takeoff_complete()
                return True
            return False
        # 70% of target altitude is "airborne enough" (their threshold)
        threshold_m = max(0.35, min(1.0, 0.35 * self._guided_alt_m))
        threshold_m = max(threshold_m, 0.7 * self._guided_alt_m)
        if rel_alt >= threshold_m:
            self._set_takeoff_complete()
            return True
        return False

    def _set_takeoff_complete(self) -> None:
        if self._takeoff_complete:
            return
        self._takeoff_complete = True
        self._takeoff_complete_at = time.monotonic()

    def _post_takeoff_settle_remaining_s(self) -> float:
        if not self._takeoff_complete or self._post_takeoff_settle_s <= 0.0:
            return 0.0
        elapsed = time.monotonic() - self._takeoff_complete_at
        return max(0.0, self._post_takeoff_settle_s - elapsed)

    # ------------------------------------------------------------------
    # MAVLink command emitters
    # ------------------------------------------------------------------

    def _send_altitude_hold_only(self) -> None:
        """Z position only; X/Y velocity always zero (no forward/strafe)."""
        if self._master is None or self._mavutil is None:
            return
        target_ned_z_m = -float(self._guided_alt_m)  # NED: down is positive
        with self._mav_send_lock:
            self._master.mav.set_position_target_local_ned_send(
                0,
                self._target_system,
                self._target_component,
                self._mavutil.mavlink.MAV_FRAME_LOCAL_NED,
                ARDUPILOT_ALT_HOLD_ONLY_TYPE_MASK,
                0.0, 0.0, target_ned_z_m,
                0.0, 0.0, 0.0,
                0.0, 0.0, 0.0,
                0.0, 0.0,
            )

    def _send_condition_yaw(self, heading_deg: float) -> None:
        """MAV_CMD_CONDITION_YAW with relative heading (param4=1.0)."""
        if self._master is None or self._mavutil is None:
            return
        heading = float(heading_deg)
        speed = self._yaw_slew_deg_s
        direction = 1
        if heading < 0.0:
            heading = abs(heading)
            direction = -1
        if heading == 0.0:
            speed = 0.0
        with self._mav_send_lock:
            self._master.mav.command_long_send(
                self._target_system,
                self._target_component,
                self._mavutil.mavlink.MAV_CMD_CONDITION_YAW,
                0,
                heading,
                speed,
                float(direction),
                1.0,  # relative yaw
                0.0, 0.0, 0.0,
            )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def status_text(self) -> str:
        if not self._enabled:
            return "FC: disabled"
        if self._dry_run:
            return f"FC: DRY-RUN (alt={self._guided_alt_m:.1f}m max_yaw={self._max_heading_deg:.0f}deg)"
        if not self._connected:
            return f"FC: not connected ({self._last_error})"
        mode = self._flight_mode or "?"
        armed = "ARMED" if self.is_armed() else "DISARMED"
        ovr = " [PILOT]" if self._user_override else ""
        rel = f" alt={self._relative_alt_m:.1f}m" if self._relative_alt_m is not None else ""
        return f"FC: {mode} {armed}{rel}{ovr} tx={self._tx_count}"
