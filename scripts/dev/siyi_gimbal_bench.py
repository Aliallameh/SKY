"""
SIYI A8 Mini gimbal bench tool.

Sends a small, fixed sequence of yaw/pitch speed commands (CMD_ID 0x07) to the
gimbal over UDP and prints what is being sent. Intended for two things only:

  1. Verifying the Jetson can reach the gimbal control endpoint at
     192.168.144.25:37260.
  2. Validating the sign convention: object right of optical center should map
     to positive yaw, object above to positive pitch. Reversed signs at this
     stage mean the autonomous follow controller would run AWAY from the
     target, so this must pass before flipping gimbal_follow.dry_run to false.

By default the script is dry-run (no packets sent). Pass --send to actually
transmit. Always run on the bench with the airframe disarmed and the gimbal
mechanically free to move.

Usage:

  # 1) Inspect what the packets would look like:
  python scripts/dev/siyi_gimbal_bench.py

  # 2) Send the standard sign-check sequence:
  python scripts/dev/siyi_gimbal_bench.py --send

  # 3) Send a single custom command:
  python scripts/dev/siyi_gimbal_bench.py --send --yaw 15 --pitch 0 --duration 1.0
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from skyscouter.gimbal.siyi_client import SiyiGimbalClient, build_rotation_packet


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SIYI A8 Mini gimbal bench tool")
    p.add_argument("--host", default="192.168.144.25", help="Gimbal control IP")
    p.add_argument("--port", type=int, default=37260, help="Gimbal control UDP port")
    p.add_argument("--send", action="store_true",
                   help="Actually transmit UDP packets. Without this flag, dry-run only.")
    p.add_argument("--yaw", type=int, default=None, help="Yaw speed -100..100 (single command mode)")
    p.add_argument("--pitch", type=int, default=None, help="Pitch speed -100..100 (single command mode)")
    p.add_argument("--duration", type=float, default=1.0,
                   help="Duration to hold the command (seconds) in single command mode")
    p.add_argument("--rate-hz", type=float, default=10.0,
                   help="Resend rate while a non-zero command is held (Hz)")
    p.add_argument("--center", action="store_true",
                   help="Send only the SIYI 'center gimbal' command (CMD_ID 0x08) "
                        "and exit. Use this to recover when a prior command drove "
                        "the gimbal to a mechanical limit.")
    return p.parse_args()


SIGN_CHECK_SEQUENCE = [
    # (yaw, pitch, duration_s, label)
    (0, 0, 0.5, "stop"),
    (15, 0, 1.5, "yaw RIGHT (+15): camera should pan right"),
    (0, 0, 0.5, "stop"),
    (-15, 0, 1.5, "yaw LEFT (-15): camera should pan left"),
    (0, 0, 0.5, "stop"),
    (0, 15, 1.5, "pitch UP (+15): camera should tilt up"),
    (0, 0, 0.5, "stop"),
    (0, -15, 1.5, "pitch DOWN (-15): camera should tilt down"),
    (0, 0, 0.5, "final stop"),
]


def _print_packet(yaw: int, pitch: int) -> None:
    pkt = build_rotation_packet(yaw, pitch)
    print(f"  packet ({len(pkt)} bytes): {pkt.hex(' ')}  yaw={yaw} pitch={pitch}")


def main() -> int:
    args = parse_args()

    if args.center:
        mode = "LIVE_UDP" if args.send else "DRY_RUN"
        print(f"SIYI gimbal CENTER  target={args.host}:{args.port}  mode={mode}")
        if not args.send:
            print("Dry run — would send CMD_ID 0x08 with payload 0x01. Re-run with --send.")
            return 0
        client = SiyiGimbalClient(host=args.host, port=args.port)
        try:
            client.center()
            print("Center command sent. Gimbal should return to neutral attitude.")
        finally:
            client.close()
        return 0

    if args.send and (args.yaw is not None or args.pitch is not None):
        if args.yaw is None or args.pitch is None:
            print("--yaw and --pitch must be set together for single command mode", file=sys.stderr)
            return 2
        sequence = [
            (0, 0, 0.2, "stop"),
            (int(args.yaw), int(args.pitch), float(args.duration),
             f"hold yaw={args.yaw} pitch={args.pitch}"),
            (0, 0, 0.2, "final stop"),
        ]
    else:
        sequence = list(SIGN_CHECK_SEQUENCE)

    mode = "LIVE_UDP" if args.send else "DRY_RUN"
    print(f"SIYI gimbal bench  target={args.host}:{args.port}  mode={mode}")
    print("Sequence:")
    for yaw, pitch, dur, label in sequence:
        print(f"  - {label}  yaw={yaw} pitch={pitch}  hold={dur}s")
    if not args.send:
        print("\nDry run — no UDP traffic. Inspect packets below, then re-run with --send.\n")
        for yaw, pitch, _, _ in sequence:
            _print_packet(yaw, pitch)
        return 0

    period = 1.0 / max(1.0, args.rate_hz)
    client = SiyiGimbalClient(host=args.host, port=args.port)
    try:
        for yaw, pitch, dur, label in sequence:
            print(f"\n>>> {label}")
            t_end = time.monotonic() + max(0.0, dur)
            while True:
                client.rotate(yaw, pitch)
                if time.monotonic() >= t_end:
                    break
                time.sleep(period)
        print("\nDone. Verify direction of motion against labels above.")
        print("If any axis moves opposite to the label, enable invert_yaw or invert_pitch")
        print("in configs/deploy_jetson_siyi_a8_mini_stage2_engine_1080p.yaml.")
    finally:
        try:
            client.stop()
        finally:
            client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
