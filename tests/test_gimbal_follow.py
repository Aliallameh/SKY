"""Tests for SIYI gimbal client packet build and the follow controller logic."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from skyscouter.gimbal.follow_controller import GimbalFollowController
from skyscouter.gimbal.siyi_client import (
    _clamp_int,
    _crc16_ccitt,
    build_rotation_packet,
)
from skyscouter.schemas import GuidanceHint


# ---------------------------------------------------------------- siyi_client


def test_crc16_ccitt_matches_siyi_heartbeat_example():
    # SDK heartbeat reference packet (header..payload, no CRC):
    # 55 66 01 01 00 00 00 00 00  -> CRC = 0x8B59
    body = bytes.fromhex("55 66 01 01 00 00 00 00 00".replace(" ", ""))
    assert _crc16_ccitt(body) == 0x8B59


def test_build_rotation_packet_structure():
    pkt = build_rotation_packet(10, -5, seq=0)
    # STX(2) + ctrl(1) + len(2) + seq(2) + cmd(1) + payload(2) + CRC(2) = 12
    assert len(pkt) == 12
    assert pkt[:2] == b"\x55\x66"        # STX
    assert pkt[2] == 0x01                 # ctrl: need ACK
    assert pkt[3:5] == b"\x02\x00"        # data length = 2 (little endian)
    assert pkt[5:7] == b"\x00\x00"        # seq = 0
    assert pkt[7] == 0x07                 # CMD_ID gimbal rotation
    assert pkt[8] == 10                   # yaw signed byte
    assert pkt[9] == (256 - 5)            # pitch signed byte (-5 -> 251)
    # CRC of body matches what's appended
    expected = _crc16_ccitt(pkt[:-2])
    assert pkt[-2] + (pkt[-1] << 8) == expected


def test_build_rotation_packet_clamps_to_speed_range():
    pkt = build_rotation_packet(500, -500)
    # signed bytes: 100 and -100 (= 156)
    assert pkt[8] == 100
    assert pkt[9] == (256 - 100)


def test_clamp_int_helper():
    assert _clamp_int(0, -100, 100) == 0
    assert _clamp_int(150, -100, 100) == 100
    assert _clamp_int(-150, -100, 100) == -100


# -------------------------------------------------------- follow_controller


def _hint(
    *,
    frame_id=1,
    valid=True,
    lock_state="LOCKED",
    confidence=0.9,
    pixel_error=(120.0, -60.0),
    normalized_error=(0.5, -0.3),
    track_id=42,
) -> GuidanceHint:
    return GuidanceHint(
        frame_id=frame_id,
        timestamp_s=float(frame_id),
        timestamp_utc=f"2026-05-11T00:00:00.{frame_id:03d}Z",
        track_id=track_id,
        valid=valid,
        source_lock_state=lock_state,
        guidance_valid_from_lock_state=lock_state in ("LOCKED", "STRIKE_READY"),
        target_center_px=[960.0 + pixel_error[0], 540.0 + pixel_error[1]],
        frame_center_px=[960.0, 540.0],
        pixel_error_px=[pixel_error[0], pixel_error[1]],
        normalized_error=[normalized_error[0], normalized_error[1]],
        confidence=confidence,
    )


def _ctrl(tmp_path, **overrides) -> GimbalFollowController:
    cfg = {
        "enabled": True,
        "dry_run": True,
        "min_confidence": 0.30,
        "command_hz": 0,  # disable rate limiting in tests
        "deadband_px_x": 40.0,
        "deadband_px_y": 30.0,
        "kp_yaw": 35.0,
        "kp_pitch": 28.0,
        "max_yaw_cmd": 25,
        "max_pitch_cmd": 20,
        "stop_after_lost_s": 0.0,  # immediate stop on invalid
    }
    cfg.update(overrides)
    return GimbalFollowController(cfg=cfg, output_dir=tmp_path, run_id="t")


def test_controller_disabled_returns_none(tmp_path: Path):
    ctrl = GimbalFollowController(cfg={"enabled": False}, output_dir=tmp_path, run_id="t")
    assert ctrl.consume(_hint()) is None
    ctrl.close()


def test_controller_valid_hint_produces_clamped_command(tmp_path: Path):
    ctrl = _ctrl(tmp_path)
    cmd = ctrl.consume(_hint(pixel_error=(200.0, -200.0), normalized_error=(1.0, -1.0)))
    assert cmd is not None and cmd.valid
    # kp_yaw=35 * 1.0 = 35 -> clamped to max_yaw_cmd=25
    # kp_pitch=28 * -1.0 = -28 -> clamped to -max_pitch_cmd=-20
    assert cmd.yaw_cmd == 25
    assert cmd.pitch_cmd == -20
    assert cmd.dry_run is True
    ctrl.close()


def test_controller_respects_deadband(tmp_path: Path):
    ctrl = _ctrl(tmp_path)
    # |pixel_error_x|=10 < deadband 40, |pixel_error_y|=10 < deadband 30 -> zero
    cmd = ctrl.consume(_hint(pixel_error=(10.0, 10.0), normalized_error=(0.01, 0.01)))
    assert cmd is not None and cmd.valid
    assert cmd.yaw_cmd == 0
    assert cmd.pitch_cmd == 0
    ctrl.close()


def test_controller_rejects_low_confidence(tmp_path: Path):
    ctrl = _ctrl(tmp_path)
    cmd = ctrl.consume(_hint(confidence=0.05))
    assert cmd is not None
    assert cmd.valid is False
    assert cmd.yaw_cmd == 0 and cmd.pitch_cmd == 0
    assert any("confidence" in r for r in cmd.reason)
    ctrl.close()


def test_controller_rejects_disallowed_lock_state(tmp_path: Path):
    ctrl = _ctrl(tmp_path)
    cmd = ctrl.consume(_hint(lock_state="SEARCHING"))
    assert cmd is not None and cmd.valid is False
    assert any("lock_state_not_allowed" in r for r in cmd.reason)
    ctrl.close()


def test_controller_invalid_hint_zeros_command_after_lost_window(tmp_path: Path):
    ctrl = _ctrl(tmp_path, stop_after_lost_s=0.0)
    ok = ctrl.consume(_hint(pixel_error=(200.0, 0.0), normalized_error=(1.0, 0.0)))
    assert ok.yaw_cmd == 25
    lost = ctrl.consume(None)
    assert lost is not None and lost.valid is False
    assert lost.yaw_cmd == 0 and lost.pitch_cmd == 0
    ctrl.close()


def test_controller_invert_flags(tmp_path: Path):
    ctrl = _ctrl(tmp_path, invert_yaw=True, invert_pitch=True)
    cmd = ctrl.consume(_hint(pixel_error=(200.0, 200.0), normalized_error=(1.0, 1.0)))
    assert cmd.yaw_cmd == -25
    assert cmd.pitch_cmd == -20
    ctrl.close()


def test_controller_writes_jsonl(tmp_path: Path):
    ctrl = _ctrl(tmp_path)
    ctrl.consume(_hint())
    ctrl.consume(_hint(frame_id=2, lock_state="SEARCHING"))
    ctrl.close()
    log = tmp_path / "gimbal_follow_commands.jsonl"
    lines = [json.loads(x) for x in log.read_text().splitlines() if x.strip()]
    assert len(lines) == 2
    assert lines[0]["valid"] is True
    assert lines[1]["valid"] is False
    assert lines[0]["schema_version"] == "skyscout.gimbal_follow_command.v1"
    assert lines[0]["dry_run"] is True


def test_controller_rate_limits_commands(tmp_path: Path):
    ctrl = _ctrl(tmp_path, command_hz=1.0)  # 1 Hz: 1 second between sends
    first = ctrl.consume(_hint())
    second = ctrl.consume(_hint(frame_id=2))
    assert first is not None
    # Second hint arrives well within 1s window -> rate-limited out
    assert second is None
    ctrl.close()


def test_controller_detection_only_bypasses_validity_and_lock_gate(tmp_path: Path):
    ctrl = _ctrl(tmp_path, detection_only=True, min_confidence=0.20)
    hint = _hint(lock_state="SEARCHING", confidence=0.40,
                 pixel_error=(200.0, -200.0), normalized_error=(1.0, -1.0))
    # Prove the bypass by explicitly marking the hint invalid.
    hint.valid = False
    cmd = ctrl.consume(hint)
    assert cmd is not None and cmd.valid is True
    # kp_yaw=35 * 1.0 = 35 clamped to max_yaw_cmd=25; kp_pitch=28 * -1.0 = -28 clamped to -20
    assert cmd.yaw_cmd == 25
    assert cmd.pitch_cmd == -20
    ctrl.close()


def test_controller_detection_only_still_enforces_confidence(tmp_path: Path):
    ctrl = _ctrl(tmp_path, detection_only=True, min_confidence=0.50)
    cmd = ctrl.consume(_hint(lock_state="SEARCHING", confidence=0.10))
    assert cmd is not None and cmd.valid is False
    assert any("confidence" in r for r in cmd.reason)
    ctrl.close()


def test_controller_rejects_unknown_backend_when_live(tmp_path: Path):
    with pytest.raises(ValueError):
        GimbalFollowController(
            cfg={"enabled": True, "dry_run": False, "backend": "made_up"},
            output_dir=tmp_path,
            run_id="t",
        )
