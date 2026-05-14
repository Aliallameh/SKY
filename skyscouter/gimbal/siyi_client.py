"""SIYI gimbal UDP client.

Implements the small subset of the SIYI gimbal SDK needed for camera follow:
CMD_ID 0x07, yaw/pitch speed control. Packet CRC matches the SDK heartbeat
example: CRC-16/CCITT, initial value 0, little-endian CRC bytes.
"""
from __future__ import annotations

import socket
import struct
import threading
from dataclasses import dataclass
from typing import Optional


_STX = b"\x55\x66"
_CTRL_NEED_ACK = 0x01
_CMD_GIMBAL_ROTATION = 0x07
_CMD_GIMBAL_CENTER = 0x08


@dataclass(frozen=True)
class SiyiRotationCommand:
    """SIYI signed speed command in SDK units, each clamped to [-100, 100]."""

    yaw: int
    pitch: int


class SiyiGimbalClient:
    """UDP sender for SIYI A8 Mini gimbal rotation commands."""

    def __init__(
        self,
        *,
        host: str = "192.168.144.25",
        port: int = 37260,
        timeout_s: float = 0.20,
    ):
        self._addr = (str(host), int(port))
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.settimeout(float(timeout_s))
        self._seq = 0
        self._lock = threading.Lock()
        self._closed = False

    @property
    def address(self) -> tuple[str, int]:
        return self._addr

    def rotate(self, yaw: int, pitch: int) -> bytes:
        """Send yaw/pitch speed command and return the packet bytes sent."""

        cmd = SiyiRotationCommand(
            yaw=_clamp_int(yaw, -100, 100),
            pitch=_clamp_int(pitch, -100, 100),
        )
        payload = struct.pack("<bb", cmd.yaw, cmd.pitch)
        packet = self._build_packet(_CMD_GIMBAL_ROTATION, payload)
        with self._lock:
            if self._closed:
                raise RuntimeError("SIYI gimbal client is closed")
            self._sock.sendto(packet, self._addr)
        return packet

    def stop(self) -> bytes:
        return self.rotate(0, 0)

    def center(self) -> bytes:
        """Send the SIYI 'center gimbal' command (CMD_ID 0x08).

        The gimbal returns to its neutral (level forward) attitude. Useful
        when a prior speed command has driven it to a mechanical limit.
        """
        payload = bytes([0x01])
        packet = self._build_packet(_CMD_GIMBAL_CENTER, payload)
        with self._lock:
            if self._closed:
                raise RuntimeError("SIYI gimbal client is closed")
            self._sock.sendto(packet, self._addr)
        return packet

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            try:
                self._sock.close()
            finally:
                self._closed = True

    def _build_packet(self, cmd_id: int, payload: bytes) -> bytes:
        if len(payload) > 0xFFFF:
            raise ValueError("SIYI payload too large")
        with self._lock:
            seq = self._seq
            self._seq = (self._seq + 1) & 0xFFFF
        body = (
            _STX
            + bytes([_CTRL_NEED_ACK])
            + struct.pack("<H", len(payload))
            + struct.pack("<H", seq)
            + bytes([int(cmd_id) & 0xFF])
            + payload
        )
        crc = _crc16_ccitt(body)
        return body + struct.pack("<H", crc)


def build_rotation_packet(yaw: int, pitch: int, *, seq: int = 0) -> bytes:
    """Pure helper used by tests/tools without opening a UDP socket."""

    payload = struct.pack("<bb", _clamp_int(yaw, -100, 100), _clamp_int(pitch, -100, 100))
    body = (
        _STX
        + bytes([_CTRL_NEED_ACK])
        + struct.pack("<H", len(payload))
        + struct.pack("<H", int(seq) & 0xFFFF)
        + bytes([_CMD_GIMBAL_ROTATION])
        + payload
    )
    return body + struct.pack("<H", _crc16_ccitt(body))


def _crc16_ccitt(data: bytes) -> int:
    crc = 0
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc & 0xFFFF


def _clamp_int(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(round(value))))

