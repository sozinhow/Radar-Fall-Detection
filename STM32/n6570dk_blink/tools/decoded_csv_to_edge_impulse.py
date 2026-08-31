#!/usr/bin/env python3
"""Convert decoded LD6002B target-list CSV rows to the teammate CSV schema."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path


OUTPUT_FIELDS = [
    "timestamp",
    "activity",
    "frame",
    "cluster_id",
    "x",
    "y",
    "z",
    "dop_idx",
]

TEAMMATE_SAMPLE_RATE_HZ = Decimal("20")

REQUIRED_INPUT_FIELDS = {
    "stream_offset",
    "frame_id",
    "message_type",
    "report",
    "x_m",
    "y_m",
    "z_m",
    "dop_idx",
}


def parse_start_time(value: str) -> datetime:
    """Accept ISO-8601, including a trailing Z, without guessing a timezone."""
    try:
        return datetime.fromisoformat(value.removesuffix("Z") + ("+00:00" if value.endswith("Z") else ""))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--start-time must be ISO-8601, e.g. 2026-07-28T13:45:00"
        ) from exc


def timestamp_for_frame(start_time: datetime, frame: int) -> str:
    """Generate deterministic, microsecond-resolution timestamps per report."""
    offset_us = (Decimal(frame) * Decimal(1_000_000) / TEAMMATE_SAMPLE_RATE_HZ).to_integral_value(
        rounding=ROUND_HALF_UP
    )
    value = start_time + timedelta(microseconds=int(offset_us))
    return value.isoformat(timespec="microseconds")


def report_key(row: dict[str, str]) -> tuple[str, str, str]:
    """Fields shared by every flattened row of one decoded TinyFrame report."""
    return (row["stream_offset"], row["frame_id"], row["message_type"])


def is_empty_target_report(row: dict[str, str]) -> bool:
    return not any(row[field].strip() for field in ("x_m", "y_m", "z_m", "dop_idx"))


def validate_target_row(row: dict[str, str], row_number: int) -> None:
    for field in ("dop_idx",):
        try:
            int(row[field])
        except (KeyError, ValueError) as exc:
            raise ValueError(f"input row {row_number}: {field} must be an integer") from exc
    for field in ("x_m", "y_m", "z_m"):
        try:
            value = float(row[field])
        except (KeyError, ValueError) as exc:
            raise ValueError(f"input row {row_number}: {field} must be numeric") from exc
        if not math.isfinite(value):
            raise ValueError(f"input row {row_number}: {field} must be finite")


def convert_rows(
    rows: list[dict[str, str]], activity: str, start_time: datetime
) -> tuple[list[dict[str, str | int]], int]:
    """Export the first target in each target-list report at fixed 20 Hz."""
    output: list[dict[str, str | int]] = []
    skipped_empty_reports = 0
    current_key: tuple[str, str, str] | None = None
    next_frame = 0

    for row_number, row in enumerate(rows, start=2):
        if row["report"] != "target_list":
            continue

        key = report_key(row)
        if key == current_key:
            # The decoder flattens a multi-target report into consecutive rows.
            # The teammate collection format keeps its first target only.
            continue
        current_key = key
        if is_empty_target_report(row):
            skipped_empty_reports += 1
            continue
        validate_target_row(row, row_number)
        output.append(
            {
                "timestamp": timestamp_for_frame(start_time, next_frame),
                "activity": activity,
                "frame": next_frame,
                "cluster_id": 1,
                "x": row["x_m"],
                "y": row["y_m"],
                "z": row["z_m"],
                "dop_idx": row["dop_idx"],
            }
        )
        next_frame += 1

    return output, skipped_empty_reports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="decoded CSV from hlk_ld6002b_capture.py")
    parser.add_argument("--output", type=Path, help="output CSV (default: <input>_edge_impulse.csv)")
    parser.add_argument("--activity", required=True, help="label applied to every output row, e.g. falling")
    parser.add_argument("--start-time", type=parse_start_time, required=True,
                        help="first target-list frame time, ISO-8601")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.activity.strip():
        raise SystemExit("--activity must not be empty")
    output_path = args.output or args.input.with_name(f"{args.input.stem}_edge_impulse.csv")

    with args.input.open("r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise SystemExit("input CSV is missing a header row")
        missing = REQUIRED_INPUT_FIELDS - set(reader.fieldnames)
        if missing:
            raise SystemExit(f"input CSV is missing required columns: {', '.join(sorted(missing))}")
        rows = list(reader)

    try:
        converted, skipped_empty = convert_rows(rows, args.activity, args.start_time)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(converted)

    print(
        f"Wrote {len(converted)} first-target rows at fixed 20 Hz to {output_path} "
        f"(skipped empty target reports: {skipped_empty})",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
