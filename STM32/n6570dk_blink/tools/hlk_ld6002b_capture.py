#!/usr/bin/env python3
"""Capture and decode HLK-LD6002B TinyFrame reports on a host computer.

This tool never transmits bytes.  It reads either a saved byte capture or a
serial device and decodes only documented radar-to-host reports: 0x0A04,
0x0A08, and 0x0A0A.

Safety: the HLK-LD6002B protocol PDF does not specify supply voltage, UART I/O
voltage, or the STM32 wiring.  Verify those facts and the physical wiring
before opening any serial port.  In particular, this tool does not assume that
STM32 PE5/PE6 are connected to the radar.
"""

from __future__ import annotations

import argparse
import csv
import json
import struct
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Iterable, TextIO


SYNC = 0x01
MAX_PAYLOAD = 1024
HEADER_SIZE = 8  # SOF, ID, LEN, TYPE, HEAD_CKSUM
FRAME_OVERHEAD = 9  # HEADER_SIZE + DATA_CKSUM
REPORT_TYPES = {0x0A04, 0x0A08, 0x0A0A}


def tf_checksum(data: bytes) -> int:
    """HLK TinyFrame checksum: bitwise complement of the bytewise XOR."""
    value = 0
    for byte in data:
        value ^= byte
    return (~value) & 0xFF


@dataclass(frozen=True)
class TinyFrame:
    offset: int
    frame_id: int
    message_type: int
    payload: bytes


@dataclass
class ParserStats:
    bytes_received: int = 0
    valid_frames: int = 0
    invalid_checksums: int = 0
    oversized_frames: int = 0
    timeouts: int = 0
    discarded_bytes: int = 0


class TinyFrameParser:
    """Incremental, bounded TinyFrame parser with corruption recovery."""

    def __init__(self, frame_timeout_s: float = 1.0) -> None:
        if frame_timeout_s <= 0:
            raise ValueError("frame_timeout_s must be positive")
        self.frame_timeout_s = frame_timeout_s
        self.stats = ParserStats()
        self._buffer = bytearray()
        self._buffer_offset = 0
        self._last_activity: float | None = None

    def _discard(self, count: int) -> None:
        if count <= 0:
            return
        del self._buffer[:count]
        self._buffer_offset += count
        self.stats.discarded_bytes += count

    def feed(self, data: bytes, now: float | None = None) -> list[TinyFrame]:
        """Feed arbitrary chunks and return every complete valid frame."""
        if now is None:
            now = time.monotonic()
        if data:
            if not self._buffer:
                self._buffer_offset = self.stats.bytes_received
            self._buffer.extend(data)
            self.stats.bytes_received += len(data)
            self._last_activity = now

        frames: list[TinyFrame] = []
        while self._buffer:
            try:
                sync_index = self._buffer.index(SYNC)
            except ValueError:
                self._discard(len(self._buffer))
                break
            if sync_index:
                self._discard(sync_index)

            if len(self._buffer) < HEADER_SIZE:
                break

            header = bytes(self._buffer[:HEADER_SIZE])
            payload_length = int.from_bytes(header[3:5], "big")
            if payload_length > MAX_PAYLOAD:
                self.stats.oversized_frames += 1
                # Header is complete but its declared length is unsafe. It is
                # not a partial frame, so skip the candidate header as one
                # malformed frame rather than treating header bytes as sync.
                self._discard(HEADER_SIZE)
                continue
            if tf_checksum(header[:7]) != header[7]:
                self.stats.invalid_checksums += 1
                self._discard(1)
                continue

            frame_length = payload_length + FRAME_OVERHEAD
            if len(self._buffer) < frame_length:
                break

            payload = bytes(self._buffer[HEADER_SIZE : HEADER_SIZE + payload_length])
            if tf_checksum(payload) != self._buffer[frame_length - 1]:
                self.stats.invalid_checksums += 1
                # A complete candidate with a valid header has a bounded,
                # trusted length. Skip it whole to avoid interpreting a 0x01
                # within corrupt DATA as another frame header.
                self._discard(frame_length)
                continue

            frames.append(
                TinyFrame(
                    offset=self._buffer_offset,
                    frame_id=int.from_bytes(header[1:3], "big"),
                    message_type=int.from_bytes(header[5:7], "big"),
                    payload=payload,
                )
            )
            self.stats.valid_frames += 1
            self._discard(frame_length)

        return frames

    def expire_partial(self, now: float | None = None) -> bool:
        """Expire an incomplete candidate frame after serial inactivity.

        A saved capture is never timed out. A serial loop calls this after an
        empty read. Dropping only the leading sync preserves later candidates.
        """
        if now is None:
            now = time.monotonic()
        if (
            self._buffer
            and self._last_activity is not None
            and now - self._last_activity >= self.frame_timeout_s
        ):
            self.stats.timeouts += 1
            self._discard(1)
            self._last_activity = now
            self.feed(b"", now)
            return True
        return False


def _expect_length(payload: bytes, expected: int, label: str) -> None:
    if len(payload) != expected:
        raise ValueError(f"{label}: LEN={len(payload)}, expected {expected}")


def decode_report(frame: TinyFrame) -> dict | None:
    """Decode only documented inbound report messages; ignore all other types."""
    if frame.message_type not in REPORT_TYPES:
        return None

    result: dict = {
        "stream_offset": frame.offset,
        "frame_id": frame.frame_id,
        "message_type": f"0x{frame.message_type:04X}",
        "payload_length": len(frame.payload),
    }
    payload = frame.payload

    if frame.message_type == 0x0A04:
        if len(payload) < 4:
            raise ValueError("0x0A04: missing target_num")
        target_num = struct.unpack_from("<i", payload, 0)[0]
        if target_num < 0:
            raise ValueError(f"0x0A04: negative target_num {target_num}")
        _expect_length(payload, 4 + target_num * 20, "0x0A04")
        targets = []
        for index in range(target_num):
            x_m, y_m, z_m, dop_idx, cluster_id = struct.unpack_from(
                "<fffii", payload, 4 + index * 20
            )
            targets.append(
                {
                    "index": index,
                    "x_m": x_m,
                    "y_m": y_m,
                    "z_m": z_m,
                    "dop_idx": dop_idx,
                    "cluster_id": cluster_id,
                }
            )
        return result | {"report": "target_list", "target_num": target_num, "targets": targets}

    if frame.message_type == 0x0A08:
        if len(payload) < 4:
            raise ValueError("0x0A08: missing target_num")
        target_num = struct.unpack_from("<i", payload, 0)[0]
        if target_num < 0:
            raise ValueError(f"0x0A08: negative target_num {target_num}")
        _expect_length(payload, 4 + target_num * 20, "0x0A08")
        points = []
        for index in range(target_num):
            cluster_index, x_m, y_m, z_m, speed_m_s = struct.unpack_from(
                "<iffff", payload, 4 + index * 20
            )
            points.append(
                {
                    "index": index,
                    "cluster_index": cluster_index,
                    "x_m": x_m,
                    "y_m": y_m,
                    "z_m": z_m,
                    "speed_m_s": speed_m_s,
                }
            )
        return result | {"report": "point_cloud", "target_num": target_num, "points": points}

    _expect_length(payload, 16, "0x0A0A")
    area_states = list(struct.unpack("<IIII", payload))
    return result | {
        "report": "area_presence",
        "area_states": area_states,
        "detected": int(any(area_states)),
    }


CSV_FIELDS = [
    "stream_offset",
    "frame_id",
    "message_type",
    "payload_length",
    "report",
    "record_index",
    "target_num",
    "cluster_id",
    "x_m",
    "y_m",
    "z_m",
    "dop_idx",
    "speed_m_s",
    "area0",
    "area1",
    "area2",
    "area3",
    "detected",
]


def csv_rows(decoded: dict) -> list[dict]:
    """Flatten a decoded report into deterministic, one-record-per-row CSV."""
    base = {key: decoded.get(key, "") for key in CSV_FIELDS}
    report = decoded["report"]
    if report == "target_list":
        targets = decoded["targets"]
        if not targets:
            return [base]
        return [base | {"record_index": item["index"], **item} for item in targets]
    if report == "point_cloud":
        points = decoded["points"]
        if not points:
            return [base]
        return [
            base
            | {
                "record_index": item["index"],
                "cluster_id": item["cluster_index"],
                "x_m": item["x_m"],
                "y_m": item["y_m"],
                "z_m": item["z_m"],
                "speed_m_s": item["speed_m_s"],
            }
            for item in points
        ]
    states = decoded["area_states"]
    return [base | {f"area{index}": state for index, state in enumerate(states)}]


class OutputWriter:
    def __init__(self, output_prefix: Path) -> None:
        output_prefix.parent.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = output_prefix.with_suffix(".jsonl")
        self.csv_path = output_prefix.with_suffix(".csv")
        self._jsonl: TextIO = self.jsonl_path.open("w", encoding="utf-8", newline="\n")
        self._csv: TextIO = self.csv_path.open("w", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(self._csv, fieldnames=CSV_FIELDS, extrasaction="ignore")
        self._writer.writeheader()
        self.decoded_frames = 0

    def write(self, decoded: dict) -> None:
        self._jsonl.write(json.dumps(decoded, sort_keys=True, separators=(",", ":")) + "\n")
        self._writer.writerows(csv_rows(decoded))
        self.decoded_frames += 1

    def close(self) -> None:
        self._jsonl.close()
        self._csv.close()


def diagnostic(frame: TinyFrame, decoded: dict | None, error: Exception | None = None) -> str:
    prefix = (
        f"offset={frame.offset} id=0x{frame.frame_id:04X} "
        f"type=0x{frame.message_type:04X} len={len(frame.payload)}"
    )
    if error is not None:
        return f"FRAME_DECODE_ERROR {prefix} reason={error}"
    if decoded is None:
        return f"FRAME_IGNORED {prefix}"
    if decoded["report"] == "area_presence":
        return f"FRAME_OK {prefix} report=area_presence detect={decoded['detected']}"
    return f"FRAME_OK {prefix} report={decoded['report']} count={decoded['target_num']}"


def process_chunks(
    chunks: Iterable[bytes], parser: TinyFrameParser, writer: OutputWriter, diagnostics: TextIO
) -> None:
    for chunk in chunks:
        for frame in parser.feed(chunk):
            try:
                decoded = decode_report(frame)
            except ValueError as exc:
                print(diagnostic(frame, None, exc), file=diagnostics)
                continue
            print(diagnostic(frame, decoded), file=diagnostics)
            if decoded is not None:
                writer.write(decoded)


def file_chunks(path: Path, chunk_size: int) -> Iterable[bytes]:
    with path.open("rb") as capture:
        while chunk := capture.read(chunk_size):
            yield chunk


def serial_chunks(
    device: str, baudrate: int, read_size: int, read_timeout_s: float, duration_s: float | None,
    parser: TinyFrameParser,
) -> Iterable[bytes]:
    try:
        import serial  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("serial mode requires pyserial: python3 -m pip install pyserial") from exc

    deadline = None if duration_s is None else time.monotonic() + duration_s
    with serial.Serial(device, baudrate=baudrate, bytesize=8, parity="N", stopbits=1,
                       timeout=read_timeout_s, xonxoff=False, rtscts=False, dsrdtr=False) as port:
        while deadline is None or time.monotonic() < deadline:
            chunk = port.read(read_size)
            if chunk:
                yield chunk
            else:
                parser.expire_partial()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--serial", metavar="DEVICE", help="host serial device; read-only")
    source.add_argument("--input", type=Path, metavar="CAPTURE.bin", help="saved binary capture")
    parser.add_argument("--baud", type=int, default=115200, help="serial baud rate (default: 115200)")
    parser.add_argument("--duration", type=float, help="serial capture duration in seconds")
    parser.add_argument("--read-timeout", type=float, default=0.1, help="serial read timeout seconds")
    parser.add_argument("--frame-timeout", type=float, default=1.0, help="partial-frame timeout seconds")
    parser.add_argument("--chunk-size", type=int, default=64, help="read chunk size in bytes")
    parser.add_argument("--output-prefix", type=Path, default=Path("hlk_ld6002b_decoded"),
                        help="writes <prefix>.jsonl and <prefix>.csv")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.chunk_size <= 0 or args.read_timeout <= 0 or args.baud <= 0:
        raise SystemExit("--chunk-size, --read-timeout, and --baud must be positive")
    parser = TinyFrameParser(args.frame_timeout)
    writer = OutputWriter(args.output_prefix)
    try:
        if args.input:
            process_chunks(file_chunks(args.input, args.chunk_size), parser, writer, sys.stderr)
        else:
            process_chunks(
                serial_chunks(args.serial, args.baud, args.chunk_size, args.read_timeout, args.duration, parser),
                parser,
                writer,
                sys.stderr,
            )
    except (OSError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    finally:
        writer.close()

    stats = parser.stats
    print(
        "SUMMARY "
        f"bytes={stats.bytes_received} valid_frames={stats.valid_frames} "
        f"decoded_reports={writer.decoded_frames} invalid_checksums={stats.invalid_checksums} "
        f"oversized_frames={stats.oversized_frames} timeouts={stats.timeouts} "
        f"discarded_bytes={stats.discarded_bytes} jsonl={writer.jsonl_path} csv={writer.csv_path}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
