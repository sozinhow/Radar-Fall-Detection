#!/usr/bin/env python3
"""Collect, decode, and export HLK-LD6002B radar training data.

The program is intentionally receive-only.  It can read direct HLK TinyFrame
bytes from a serial device, or unwrap the RCAP v1 records emitted by the
STM32 raw-capture overlay.  A collection keeps the host input bytes as a raw
binary file, decodes all supported reports to JSONL/CSV, and exports the
first target of each 0x0A04 report in the project's 20 Hz training schema.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TextIO

# The script is normally launched directly from tools/.  Keep imports usable
# both as a script and from the host-only unit tests.
TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from decoded_csv_to_edge_impulse import (  # noqa: E402
    OUTPUT_FIELDS as TRAINING_FIELDS,
    REQUIRED_INPUT_FIELDS,
    convert_rows,
    parse_start_time,
)
from hlk_ld6002b_capture import (  # noqa: E402
    OutputWriter,
    TinyFrameParser,
    decode_report,
    diagnostic,
)


RCAP_MAGIC = b"\xA5\x5A"
RCAP_VERSION = 1
RCAP_HEADER_SIZE = 14
RCAP_MAX_PAYLOAD = 1024
RCAP_RECORD_OVERHEAD = RCAP_HEADER_SIZE + 1
RCAP_TYPE_RAW = 1
RCAP_TYPE_DIAG = 2
RCAP_TYPE_STATUS = 3
RCAP_TYPES = {RCAP_TYPE_RAW, RCAP_TYPE_DIAG, RCAP_TYPE_STATUS}


@dataclass(frozen=True)
class RcapRecord:
    """One valid RCAP v1 record from the STM32 VCP stream."""

    offset: int
    sequence: int
    tick_ms: int
    record_type: int
    payload: bytes


@dataclass
class RcapStats:
    records: int = 0
    raw_records: int = 0
    raw_bytes: int = 0
    invalid_records: int = 0
    oversized_records: int = 0
    discarded_bytes: int = 0


class RcapParser:
    """Incrementally validate and unwrap STM32 RCAP v1 records."""

    def __init__(self) -> None:
        self.stats = RcapStats()
        self._buffer = bytearray()
        self._buffer_offset = 0
        self._bytes_received = 0

    def _discard(self, count: int) -> None:
        if count <= 0:
            return
        del self._buffer[:count]
        self._buffer_offset += count
        self.stats.discarded_bytes += count

    def feed(self, data: bytes) -> list[RcapRecord]:
        if data:
            if not self._buffer:
                self._buffer_offset = self._bytes_received
            self._buffer.extend(data)
            self._bytes_received += len(data)

        records: list[RcapRecord] = []
        while self._buffer:
            magic_index = self._buffer.find(RCAP_MAGIC)
            if magic_index < 0:
                # Keep a possible first magic byte for the next serial read.
                keep = 1 if self._buffer[-1:] == RCAP_MAGIC[:1] else 0
                self._discard(len(self._buffer) - keep)
                break
            if magic_index:
                self._discard(magic_index)
            if len(self._buffer) < RCAP_HEADER_SIZE:
                break

            version = self._buffer[2]
            record_type = self._buffer[3]
            payload_length = int.from_bytes(self._buffer[4:6], "little")
            if version != RCAP_VERSION or record_type not in RCAP_TYPES:
                self.stats.invalid_records += 1
                self._discard(1)
                continue
            if payload_length > RCAP_MAX_PAYLOAD:
                self.stats.oversized_records += 1
                self._discard(RCAP_HEADER_SIZE)
                continue

            record_length = RCAP_RECORD_OVERHEAD + payload_length
            if len(self._buffer) < record_length:
                break
            candidate = bytes(self._buffer[:record_length])
            expected_checksum = candidate[-1]
            checksum = _checksum(candidate[:-1])
            if checksum != expected_checksum:
                self.stats.invalid_records += 1
                self._discard(1)
                continue

            records.append(
                RcapRecord(
                    offset=self._buffer_offset,
                    sequence=int.from_bytes(candidate[6:10], "little"),
                    tick_ms=int.from_bytes(candidate[10:14], "little"),
                    record_type=record_type,
                    payload=candidate[14:-1],
                )
            )
            self.stats.records += 1
            if record_type == RCAP_TYPE_RAW:
                self.stats.raw_records += 1
                self.stats.raw_bytes += payload_length
            self._discard(record_length)
        return records


def _checksum(data: bytes) -> int:
    value = 0
    for byte in data:
        value ^= byte
    return (~value) & 0xFF


class StreamDecoder:
    """Route direct or RCAP input into the existing TinyFrame decoder."""

    def __init__(self, input_format: str, output: OutputWriter, diagnostics: TextIO) -> None:
        self.input_format = input_format
        self.output = output
        self.diagnostics = diagnostics
        self.tf_parser = TinyFrameParser()
        self.rcap_parser = RcapParser() if input_format == "rcap" else None

    def _decode_radar_bytes(self, data: bytes) -> None:
        for frame in self.tf_parser.feed(data):
            try:
                decoded = decode_report(frame)
            except ValueError as exc:
                print(diagnostic(frame, None, exc), file=self.diagnostics)
                continue
            print(diagnostic(frame, decoded), file=self.diagnostics)
            if decoded is not None:
                self.output.write(decoded)

    def feed(self, data: bytes) -> None:
        if not data:
            return
        if self.rcap_parser is None:
            self._decode_radar_bytes(data)
            return
        for record in self.rcap_parser.feed(data):
            if record.record_type == RCAP_TYPE_RAW:
                self._decode_radar_bytes(record.payload)

    def expire_partial(self) -> None:
        # RCAP is a saved envelope and has no wall-clock timeout requirement;
        # the TinyFrame parser does need timeout handling for serial gaps.
        self.tf_parser.expire_partial()


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _write_training_csv(
    decoded_csv: Path, output_csv: Path, activity: str, start_time: datetime
) -> tuple[int, int]:
    with decoded_csv.open("r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise ValueError("decoded CSV is missing a header row")
        missing = REQUIRED_INPUT_FIELDS - set(reader.fieldnames)
        if missing:
            raise ValueError(
                "decoded CSV is missing required columns: " + ", ".join(sorted(missing))
            )
        rows = list(reader)

    converted, skipped_empty = convert_rows(rows, activity.strip(), start_time)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=TRAINING_FIELDS)
        writer.writeheader()
        writer.writerows(converted)
    return len(converted), skipped_empty


def _decode_file(
    input_path: Path,
    output_prefix: Path,
    input_format: str,
    chunk_size: int,
) -> StreamDecoder:
    output = OutputWriter(output_prefix)
    decoder = StreamDecoder(input_format, output, sys.stderr)
    try:
        with input_path.open("rb") as source:
            while chunk := source.read(chunk_size):
                decoder.feed(chunk)
    finally:
        output.close()
    return decoder


def _print_summary(decoder: StreamDecoder) -> None:
    stats = decoder.tf_parser.stats
    message = (
        "SUMMARY "
        f"bytes={stats.bytes_received} valid_frames={stats.valid_frames} "
        f"decoded_reports={decoder.output.decoded_frames} "
        f"invalid_checksums={stats.invalid_checksums} "
        f"oversized_frames={stats.oversized_frames} timeouts={stats.timeouts} "
        f"discarded_bytes={stats.discarded_bytes}"
    )
    if decoder.rcap_parser is not None:
        rcap = decoder.rcap_parser.stats
        message += (
            f" rcap_records={rcap.records} rcap_raw_records={rcap.raw_records}"
            f" rcap_raw_bytes={rcap.raw_bytes}"
            f" rcap_invalid={rcap.invalid_records}"
            f" rcap_discarded_bytes={rcap.discarded_bytes}"
        )
    print(message, file=sys.stderr)


def _make_collection_paths(output_dir: Path, activity: str, started_at: datetime) -> tuple[Path, Path, Path]:
    stem = f"{activity}_{started_at.strftime('%Y%m%d_%H%M%S')}"
    raw = output_dir / f"{stem}_raw.bin"
    decoded_prefix = output_dir / f"{stem}_decoded"
    training = output_dir / f"{stem}_20hz.csv"
    return raw, decoded_prefix, training


def _check_new_paths(paths: list[Path]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError("output already exists: " + ", ".join(existing))


def collect(args: argparse.Namespace) -> int:
    activity = args.activity.strip()
    if not activity:
        raise SystemExit("--activity must not be empty")

    try:
        import serial  # type: ignore[import-not-found]
    except ImportError as exc:
        print("ERROR: serial mode requires pyserial: python3 -m pip install pyserial", file=sys.stderr)
        return 2

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Opening {args.serial} at {args.baud} baud (receive-only).", file=sys.stderr)
    try:
        with serial.Serial(
            args.serial,
            baudrate=args.baud,
            bytesize=8,
            parity="N",
            stopbits=1,
            timeout=args.read_timeout,
            xonxoff=False,
            rtscts=False,
            dsrdtr=False,
        ) as port:
            if args.countdown:
                print(f"Starting collection in {args.countdown:.1f} seconds...", file=sys.stderr)
                time.sleep(args.countdown)
                port.reset_input_buffer()
            started_at = datetime.now().astimezone()
            raw_path, decoded_prefix, training_path = _make_collection_paths(
                output_dir, activity, started_at
            )
            _check_new_paths([
                raw_path,
                decoded_prefix.with_suffix(".jsonl"),
                decoded_prefix.with_suffix(".csv"),
                training_path,
            ])
            with raw_path.open("wb") as raw_file:
                output = OutputWriter(decoded_prefix)
                decoder = StreamDecoder(args.input_format, output, sys.stderr)
                deadline = None if args.duration is None else time.monotonic() + args.duration
                print("Collecting; press Ctrl-C to stop.", file=sys.stderr)
                try:
                    while deadline is None or time.monotonic() < deadline:
                        chunk = port.read(args.chunk_size)
                        if chunk:
                            raw_file.write(chunk)
                            raw_file.flush()
                            decoder.feed(chunk)
                        else:
                            decoder.expire_partial()
                except KeyboardInterrupt:
                    print("\nCollection stopped by user.", file=sys.stderr)
                finally:
                    output.close()
    except (OSError, RuntimeError, FileExistsError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    try:
        converted, skipped_empty = _write_training_csv(
            decoded_prefix.with_suffix(".csv"), training_path, activity, started_at
        )
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    _print_summary(decoder)
    print(
        f"Wrote raw={raw_path} decoded={decoded_prefix.with_suffix('.csv')} "
        f"training={training_path} rows={converted} skipped_empty={skipped_empty}",
        file=sys.stderr,
    )
    return 0


def decode(args: argparse.Namespace) -> int:
    output_prefix = args.output_prefix or args.input.with_name(f"{args.input.stem}_decoded")
    try:
        decoder = _decode_file(args.input, output_prefix, args.input_format, args.chunk_size)
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    _print_summary(decoder)
    print(f"Wrote {output_prefix.with_suffix('.jsonl')} and {output_prefix.with_suffix('.csv')}", file=sys.stderr)
    return 0


def convert(args: argparse.Namespace) -> int:
    if not args.activity.strip():
        raise SystemExit("--activity must not be empty")
    try:
        converted, skipped_empty = _write_training_csv(
            args.input, args.output, args.activity, args.start_time
        )
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(
        f"Wrote {converted} rows at fixed 20 Hz to {args.output} "
        f"(skipped empty target reports: {skipped_empty})",
        file=sys.stderr,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect_parser = subparsers.add_parser("collect", help="capture a serial stream and export all dataset files")
    collect_parser.add_argument("--serial", required=True, metavar="DEVICE")
    collect_parser.add_argument("--activity", required=True, help="label, e.g. falling, walking, standing")
    collect_parser.add_argument("--duration", type=_positive_float, help="seconds; omit to collect until Ctrl-C")
    collect_parser.add_argument("--output-dir", type=Path, default=Path("training_data"))
    collect_parser.add_argument("--input-format", choices=("direct", "rcap"), default="direct",
                                help="direct HLK TinyFrame or STM32 RCAP v1 envelope")
    collect_parser.add_argument("--baud", type=_positive_int, default=115200)
    collect_parser.add_argument("--read-timeout", type=_positive_float, default=0.1)
    collect_parser.add_argument("--chunk-size", type=_positive_int, default=64)
    collect_parser.add_argument("--countdown", type=float, default=3.0,
                                help="seconds to wait before clearing stale serial bytes (default: 3)")
    collect_parser.set_defaults(handler=collect)

    decode_parser = subparsers.add_parser("decode", help="decode an existing raw binary capture")
    decode_parser.add_argument("--input", type=Path, required=True, metavar="RAW.bin")
    decode_parser.add_argument("--output-prefix", type=Path)
    decode_parser.add_argument("--input-format", choices=("direct", "rcap"), default="direct")
    decode_parser.add_argument("--chunk-size", type=_positive_int, default=64)
    decode_parser.set_defaults(handler=decode)

    convert_parser = subparsers.add_parser("convert", help="convert decoded CSV to the 20 Hz training schema")
    convert_parser.add_argument("--input", type=Path, required=True, metavar="DECODED.csv")
    convert_parser.add_argument("--output", type=Path, required=True, metavar="TRAINING.csv")
    convert_parser.add_argument("--activity", required=True)
    convert_parser.add_argument("--start-time", type=parse_start_time, required=True,
                                 help="ISO-8601 timestamp assigned to output frame 0")
    convert_parser.set_defaults(handler=convert)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    countdown = getattr(args, "countdown", 0.0)
    if not math.isfinite(countdown) or countdown < 0:
        parser.error("--countdown must be a finite, non-negative number")
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
