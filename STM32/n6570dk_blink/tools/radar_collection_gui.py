#!/usr/bin/env python3
"""Tkinter collector for teammate-compatible, single-target 20 Hz CSV files."""

from __future__ import annotations

import csv
import math
import os
import struct
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Sequence, TextIO

from hlk_ld6002b_capture import TinyFrameParser, decode_report

# This script is normally launched directly from tools/, so make the host
# inference package importable without requiring an STM32-side change.
RADAR_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(RADAR_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(RADAR_PROJECT_ROOT))

try:
    from radar_pipeline.live_inference import (  # noqa: E402
        CLASS_NAMES,
        CausalHostInference,
        FRESH_FRAME_HANDOFF_GRACE_S,
        InferenceUpdate,
        LiveRadarRecord,
        MISSING_SUBREASON_LABELS,
        RESET_REASON_LABELS,
        load_demo_binding,
        should_wait_for_fresh_frame,
    )
    LIVE_INFERENCE_IMPORT_ERROR: str | None = None
except Exception as exc:  # Preserve the established collection-only workflow.
    CLASS_NAMES = ("walking", "standing", "sitting", "fall")
    CausalHostInference = None  # type: ignore[assignment,misc]
    InferenceUpdate = None  # type: ignore[assignment,misc]
    LiveRadarRecord = None  # type: ignore[assignment,misc]
    load_demo_binding = None  # type: ignore[assignment,misc]
    RESET_REASON_LABELS = {
        "missing_record": "missing record", "duplicate_frame_id": "duplicate frame ID",
        "invalid_value": "invalid value", "timestamp_gap": "timestamp gap", "other": "other reset",
    }
    MISSING_SUBREASON_LABELS = {
        "no_fresh_frame": "no fresh target frame", "queue_empty": "target queue empty",
        "late_frame": "fresh frame arrived late", "tracking_inactive": "tracking inactive",
        "other_missing": "other missing record",
    }
    FRESH_FRAME_HANDOFF_GRACE_S = 0.020

    def should_wait_for_fresh_frame(**_: object) -> bool:
        return False
    LIVE_INFERENCE_IMPORT_ERROR = str(exc)

try:
    import tkinter as tk
    from tkinter import messagebox, ttk
except ModuleNotFoundError:
    # Keep the pure CSV writer importable for headless tests and give operators
    # a direct launch-time explanation instead of an import traceback.
    tk = None  # type: ignore[assignment]
    messagebox = None  # type: ignore[assignment]
    ttk = None  # type: ignore[assignment]


SAMPLE_RATE_HZ = 20
SAMPLE_PERIOD_S = 1.0 / SAMPLE_RATE_HZ
DEFAULT_DURATION_S = 5.0
TRACKING_TIMEOUT_S = 0.5
# These are Tk logical pixels. The final minimum is also clamped to the
# widgets' requested size after construction, so macOS display scaling cannot
# start the diagnostics in a clipped window.
DEFAULT_WINDOW_WIDTH = 1120
DEFAULT_WINDOW_HEIGHT = 840
MIN_WINDOW_WIDTH = 980
MIN_WINDOW_HEIGHT = 700
QUALITY_DISTANCE_MIN_M = 0.5
QUALITY_DISTANCE_MAX_M = 6.0
QUALITY_Z_MIN_M = -2.5
QUALITY_Z_MAX_M = 1.0
QUALITY_POSITION_JUMP_M = 0.5
QUALITY_HISTORY_FRAMES = 60
TARGET_SELECTION_AUTO = "auto"
TARGET_SELECTION_TEAMMATE = "teammate_compatible"
TARGET_SELECTION_FORCE_INDEX_0 = "force_index_0"
TARGET_SELECTION_FORCE_INDEX_1 = "force_index_1"
DEFAULT_TARGET_SELECTION_MODE = TARGET_SELECTION_TEAMMATE
TARGET_SELECTION_LABELS = {
    TARGET_SELECTION_TEAMMATE: "Teammate-compatible (first target + legacy gates)",
    TARGET_SELECTION_AUTO: "Smart continuity (explicit opt-in)",
    TARGET_SELECTION_FORCE_INDEX_0: "Force index 0 (debug)",
    TARGET_SELECTION_FORCE_INDEX_1: "Force index 1 (debug)",
}
ACTIVITIES = ("sitting", "standing", "walking", "falling", "laying")
CSV_FIELDS = ["timestamp", "activity", "frame", "cluster_id", "x", "y", "z", "dop_idx"]
HOST_DEMO_MANIFEST = (
    RADAR_PROJECT_ROOT / "outputs/deployment/host_demo_20260728_run01/host_demo_manifest.json"
)


@dataclass(frozen=True)
class RadarTarget:
    x: float
    y: float
    z: float
    dop_idx: int
    # Keep the established writer/test positional constructor compatible. The
    # live 0x0A04 path always supplies the actual protocol values by keyword.
    target_num: int = 1
    cluster_id: int = -1

    @property
    def distance(self) -> float:
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)


@dataclass(frozen=True)
class TargetSelection:
    index: int
    target: RadarTarget
    reason: str


def is_strict_quality_target(target: RadarTarget) -> bool:
    """Return whether a raw target is eligible for automatic live selection."""
    return (
        all(math.isfinite(value) for value in (target.x, target.y, target.z, target.distance))
        and QUALITY_DISTANCE_MIN_M <= target.distance <= QUALITY_DISTANCE_MAX_M
        and QUALITY_Z_MIN_M <= target.z <= QUALITY_Z_MAX_M
    )


def is_teammate_compatible_target(target: RadarTarget) -> bool:
    """Mirror collect_fixed_20hz.py's only coordinate gates, including endpoints."""
    return not (
        target.distance < 0.3 or target.distance > 8.0
        or target.z < -3.0 or target.z > 2.0
    )


def teammate_next_sample_time(next_sample_at: float, now: float) -> float:
    """Advance one legacy 20 Hz tick without generating catch-up rows."""
    next_sample_at += SAMPLE_PERIOD_S
    if next_sample_at < now:
        return now + SAMPLE_PERIOD_S
    return next_sample_at


def parse_teammate_target_chunk(data: bytes) -> RadarTarget | None:
    """Mirror the old collector's stateless 0x0A04 chunk scan literally.

    This intentionally does not validate TinyFrame checksums or carry partial
    frames between reads. Those limitations are part of the historical data
    collection behavior being reproduced for regression comparisons.
    """
    for index in range(len(data) - 8):
        if data[index] != 0x01:
            continue
        try:
            frame_type = struct.unpack(">H", data[index + 5:index + 7])[0]
            if frame_type != 0x0A04:
                continue
            data_len = struct.unpack(">H", data[index + 3:index + 5])[0]
            if index + 8 + data_len > len(data) or data_len < 24:
                continue
            frame_data = data[index + 8:index + 8 + data_len]
            target_num = struct.unpack("<I", frame_data[0:4])[0]
            if target_num == 0 or target_num > 10:
                continue
            target = RadarTarget(
                target_num=target_num,
                cluster_id=1,
                x=struct.unpack("<f", frame_data[4:8])[0],
                y=struct.unpack("<f", frame_data[8:12])[0],
                z=struct.unpack("<f", frame_data[12:16])[0],
                dop_idx=struct.unpack("<i", frame_data[16:20])[0],
            )
            if not is_teammate_compatible_target(target):
                continue
            return target
        except Exception:
            continue
    return None


def _xyz_jump(target: RadarTarget, previous_target: RadarTarget | None) -> float:
    if previous_target is None:
        return 0.0
    return math.sqrt(
        (target.x - previous_target.x) ** 2
        + (target.y - previous_target.y) ** 2
        + (target.z - previous_target.z) ** 2
    )


def rank_target(
    targets: Sequence[RadarTarget], *, mode: str, previous_target: RadarTarget | None,
) -> TargetSelection | None:
    """Select a stable target; manual force modes exist only for comparison."""
    if mode not in TARGET_SELECTION_LABELS:
        raise ValueError(f"unknown target-selection mode: {mode}")

    if mode == TARGET_SELECTION_TEAMMATE:
        if not targets or not is_teammate_compatible_target(targets[0]):
            return None
        return TargetSelection(0, targets[0], "teammate first target + legacy gates")

    if mode != TARGET_SELECTION_AUTO:
        forced_index = 0 if mode == TARGET_SELECTION_FORCE_INDEX_0 else 1
        if forced_index >= len(targets):
            return None
        forced_target = targets[forced_index]
        if not all(math.isfinite(value) for value in (
            forced_target.x, forced_target.y, forced_target.z, forced_target.distance,
        )):
            return None
        return TargetSelection(forced_index, forced_target, f"forced index {forced_index} (debug)")

    valid = [(index, target) for index, target in enumerate(targets) if is_strict_quality_target(target)]
    if not valid:
        return None

    if previous_target is not None:
        continuity = [
            (index, target) for index, target in valid
            if target.cluster_id == previous_target.cluster_id
        ]
        if continuity:
            index, target = min(
                continuity,
                key=lambda item: (_xyz_jump(item[1], previous_target), item[1].distance, item[0]),
            )
            return TargetSelection(index, target, "continuity")

    minimum_distance = min(target.distance for _, target in valid)
    nearest = [
        (index, target) for index, target in valid
        if math.isclose(target.distance, minimum_distance, rel_tol=0.0, abs_tol=1e-9)
    ]
    if len(nearest) == 1:
        index, target = nearest[0]
        return TargetSelection(index, target, "nearest valid")

    if previous_target is not None:
        index, target = min(nearest, key=lambda item: (_xyz_jump(item[1], previous_target), item[0]))
        return TargetSelection(index, target, "nearest valid")

    index, target = min(nearest, key=lambda item: item[0])
    return TargetSelection(index, target, "fallback to first valid")


def format_target_table_row(index: int, target: RadarTarget, *, used: bool) -> tuple[str, ...]:
    """Format one of the latest 0x0A04 targets for the diagnostic table."""
    return (
        "USED" if used else "",
        str(index),
        str(target.cluster_id),
        f"{target.x:.3f}",
        f"{target.y:.3f}",
        f"{target.z:.3f}",
        f"{target.distance:.3f}",
        str(target.dop_idx),
    )


@dataclass(frozen=True)
class LiveDataQuality:
    """Current and rolling checks ported from validate_data_quality.py."""

    level: str
    reason: str
    sample_count: int
    current_distance_in_range: bool
    current_z_in_range: bool
    distance_min_m: float | None
    distance_max_m: float | None
    distance_mean_m: float | None
    z_min_m: float | None
    z_max_m: float | None
    z_mean_m: float | None
    jump_percent: float | None
    average_cv: float | None


def _mean(values: Sequence[float]) -> float:
    return math.fsum(values) / len(values)


def _sample_std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    return math.sqrt(math.fsum((value - mean) ** 2 for value in values) / (len(values) - 1))


def summarize_live_data_quality(targets: Sequence[RadarTarget]) -> LiveDataQuality:
    """Apply the Downloads validator's range, jump, and CV rules to recent targets."""
    # The live GUI owns a bounded deque; normalize it to a sliceable immutable
    # sequence so the same quality path works in the GUI and direct tests.
    targets = tuple(targets)
    if not targets:
        return LiveDataQuality(
            level="WAIT", reason="waiting for first target", sample_count=0,
            current_distance_in_range=False, current_z_in_range=False,
            distance_min_m=None, distance_max_m=None, distance_mean_m=None,
            z_min_m=None, z_max_m=None, z_mean_m=None,
            jump_percent=None, average_cv=None,
        )

    latest = targets[-1]
    latest_values = (latest.x, latest.y, latest.z, latest.distance)
    latest_finite = all(math.isfinite(value) for value in latest_values)
    distance_in_range = latest_finite and QUALITY_DISTANCE_MIN_M <= latest.distance <= QUALITY_DISTANCE_MAX_M
    z_in_range = latest_finite and QUALITY_Z_MIN_M <= latest.z <= QUALITY_Z_MAX_M

    valid_targets = [
        target for target in targets
        if all(math.isfinite(value) for value in (target.x, target.y, target.z, target.distance))
    ]
    distances = [target.distance for target in valid_targets]
    z_values = [target.z for target in valid_targets]
    x_values = [target.x for target in valid_targets]
    y_values = [target.y for target in valid_targets]
    nonfinite_count = len(targets) - len(valid_targets)

    position_changes = []
    for previous, current in zip(targets, targets[1:]):
        values = (previous.x, previous.y, previous.z, current.x, current.y, current.z)
        if all(math.isfinite(value) for value in values):
            position_changes.append(math.sqrt(
                (current.x - previous.x) ** 2
                + (current.y - previous.y) ** 2
                + (current.z - previous.z) ** 2
            ))
    jump_percent = (
        100.0 * sum(change > QUALITY_POSITION_JUMP_M for change in position_changes) / len(position_changes)
        if position_changes else None
    )

    def coefficient_of_variation(values: Sequence[float]) -> float:
        mean = _mean(values)
        return _sample_std(values) / abs(mean) if mean != 0 else 0.0

    average_cv = (
        (coefficient_of_variation(x_values) + coefficient_of_variation(y_values) + coefficient_of_variation(z_values)) / 3
        if valid_targets else None
    )

    if not latest_finite:
        level, reason = "ERROR", "current coordinate is non-finite"
    elif not distance_in_range:
        level, reason = "ERROR", "distance outside 0.5–6.0 m"
    elif not z_in_range:
        level, reason = "ERROR", "Z outside −2.5–1.0 m"
    elif jump_percent is not None and jump_percent >= 15.0:
        level, reason = "ERROR", "frequent >0.5 m position jumps"
    elif nonfinite_count:
        level, reason = "WARN", "non-finite sample in recent window"
    elif (jump_percent is not None and jump_percent >= 5.0) or (
        average_cv is not None and average_cv >= 0.3
    ):
        level, reason = "WARN", "recent motion/jitter exceeds stable-target guideline"
    else:
        level, reason = "GOOD", "distance and Z are in range"

    return LiveDataQuality(
        level=level, reason=reason, sample_count=len(targets),
        current_distance_in_range=distance_in_range, current_z_in_range=z_in_range,
        distance_min_m=min(distances) if distances else None,
        distance_max_m=max(distances) if distances else None,
        distance_mean_m=_mean(distances) if distances else None,
        z_min_m=min(z_values) if z_values else None,
        z_max_m=max(z_values) if z_values else None,
        z_mean_m=_mean(z_values) if z_values else None,
        jump_percent=jump_percent, average_cv=average_cv,
    )


def format_raw_target_diagnostic(target: RadarTarget, *, age_s: float) -> str:
    """Show the exact selected-target fields and host distance derivation."""
    return (
        f"targetnum={target.target_num} · cluster_id={target.cluster_id} · "
        f"x={target.x:.3f} m · y={target.y:.3f} m · z={target.z:.3f} m · "
        f"host distance=sqrt(x²+y²+z²)={target.distance:.3f} m · "
        f"dopidx={target.dop_idx} · age={age_s * 1000:.0f} ms"
    )


class SingleTarget20HzWriter:
    """Write the historical teammate schema: first target, one row per 20 Hz frame."""

    def __init__(
        self, output_dir: Path, activity: str, started_at: datetime, *,
        teammate_compatible: bool = False,
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        self.path = output_dir / f"{activity}_{started_at.strftime('%Y%m%d_%H%M%S')}_20hz.csv"
        self._file: TextIO | None = None
        self._writer: csv.DictWriter | None = None
        self.started_at = started_at
        self.teammate_compatible = teammate_compatible
        self.frame = 0

        # The old collector creates no file when it receives no valid target.
        # Retain the existing eager behavior outside compatibility mode.
        if not teammate_compatible:
            self._open()

    def _open(self) -> None:
        if self._file is not None:
            return
        self._file = self.path.open("x", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=CSV_FIELDS)
        self._writer.writeheader()

    def write(
        self, target: RadarTarget, activity: str, *, sampled_at: datetime | None = None,
    ) -> int:
        self._open()
        assert self._writer is not None
        assert self._file is not None
        frame = self.frame
        if self.teammate_compatible:
            timestamp = sampled_at or datetime.now()
        else:
            timestamp = self.started_at + timedelta(seconds=frame / SAMPLE_RATE_HZ)
        self._writer.writerow(
            {
                "timestamp": (
                    timestamp.isoformat() if self.teammate_compatible
                    else timestamp.isoformat(timespec="microseconds")
                ),
                "activity": activity,
                "frame": frame,
                "cluster_id": 1,
                "x": target.x,
                "y": target.y,
                "z": target.z,
                "dop_idx": target.dop_idx,
            }
        )
        self._file.flush()
        self.frame += 1
        return frame

    def close(self) -> None:
        if self._file is not None:
            self._file.close()


class RadarCollectionApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("LD6002B 20 Hz Collector and Diagnostic Inference")
        self.root.resizable(True, True)
        self.root.columnconfigure(0, weight=1)
        # The inference panel is the largest section. Giving it the spare
        # height keeps the full layout usable when a window is enlarged.
        self.root.rowconfigure(3, weight=1)

        self.port_var = tk.StringVar()
        self.baud_var = tk.StringVar(value="115200")
        self.activity_var = tk.StringVar(value="sitting")
        self.duration_var = tk.StringVar(value=str(int(DEFAULT_DURATION_S)))
        self.status_var = tk.StringVar(value="disconnected")
        self.x_var = tk.StringVar(value="—")
        self.y_var = tk.StringVar(value="—")
        self.z_var = tk.StringVar(value="—")
        self.distance_var = tk.StringVar(value="—")
        self.dop_var = tk.StringVar(value="—")
        self.targetnum_var = tk.StringVar(value="—")
        self.cluster_id_var = tk.StringVar(value="—")
        self.raw_target_var = tk.StringVar(value="waiting for 0x0A04 target-list")
        self.continuity_var = tk.StringVar(value="cluster switches=0 · targetnum=0 reports=0")
        self.target_selection_mode_var = tk.StringVar(value=DEFAULT_TARGET_SELECTION_MODE)
        self.used_target_var = tk.StringVar(
            value="selection policy: Teammate-compatible (first target + legacy gates) · waiting for 0x0A04"
        )
        self.sample_freshness_var = tk.StringVar(value="sample age: —")
        self.selection_change_var = tk.StringVar(value="selection changes: 0")
        self.target_table_vars = [
            [tk.StringVar(value="—") for _ in range(8)] for _ in range(3)
        ]
        self.quality_var = tk.StringVar(value="WAIT · waiting for first target")
        self.quality_window_var = tk.StringVar(value="last 0/60 frames: waiting for coordinates")
        self.frames_var = tk.StringVar(value="0")
        self.rate_var = tk.StringVar(value="0.0 Hz")
        self.parser_var = tk.StringVar(value="waiting for serial data")
        self.inference_buffer_var = tk.StringVar(value="0 / 60")
        self.inference_class_var = tk.StringVar(value="—")
        self.inference_health_var = tk.StringVar(value="initializing")
        self.inference_reset_var = tk.StringVar(value="none")
        self.inference_reset_counts_var = tk.StringVar(value="resets: missing=0 · duplicate=0 · invalid=0 · ts_gap=0 · other=0")
        self.inference_missing_var = tk.StringVar(value="missing detail: none")
        self.inference_timing_var = tk.StringVar(value="timing: waiting for record")
        self.inference_mode_var = tk.StringVar(value="diagnostic tolerance: OFF")
        self.diagnostic_tolerance_var = tk.BooleanVar(value=False)
        self.probability_vars = {name: tk.StringVar(value="—") for name in CLASS_NAMES}

        self.port: object | None = None
        self.parser = TinyFrameParser()
        self.latest_target: RadarTarget | None = None
        self.last_target_at = 0.0
        self.latest_frame_targetnum: int | None = None
        self.latest_frame_targets: list[RadarTarget] = []
        self.latest_selected_target_index: int | None = None
        self.last_selected_target_index: int | None = None
        self.last_selected_cluster_id: int | None = None
        self.previous_selected_target: RadarTarget | None = None
        self.selection_reason: str | None = None
        self.selected_target_is_current = False
        self.selected_cluster_switches = 0
        self.selection_change_count = 0
        self.targetnum_dropouts = 0
        self.quality_history: deque[RadarTarget] = deque(maxlen=QUALITY_HISTORY_FRAMES)
        self.inference_queue: deque[tuple[RadarTarget, int, float]] = deque()
        # The literal teammate parser intentionally does not retain the
        # TinyFrame frame ID. Use a connection-local monotonic ID for duplicate
        # detection; it is metadata only and does not change model features.
        self.next_teammate_inference_frame_id = 0
        self.next_inference_at = 0.0
        self.last_target_list_at = 0.0
        self.last_empty_target_list_at = 0.0
        self.last_inference_update: InferenceUpdate | None = None
        self.inference: CausalHostInference | None = None
        self.inference_startup_error: str | None = None
        try:
            if LIVE_INFERENCE_IMPORT_ERROR is not None or CausalHostInference is None or load_demo_binding is None:
                raise RuntimeError(LIVE_INFERENCE_IMPORT_ERROR or "host inference dependencies are unavailable")
            self.inference = CausalHostInference(load_demo_binding(HOST_DEMO_MANIFEST))
        except Exception as exc:
            # Collection remains usable when the local Python environment lacks
            # PyTorch or the pinned artifact cannot be loaded.
            self.inference_startup_error = str(exc)
            self.inference_health_var.set("unavailable")
        self.writer: SingleTarget20HzWriter | None = None
        self.collecting = False
        self.collection_deadline = 0.0
        self.collection_starts_at = 0.0
        self.next_sample_at = 0.0
        self.sample_times: deque[float] = deque()
        self._build_ui()
        self._configure_window_geometry()
        self.refresh_ports()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(10, self.poll)

    def _configure_window_geometry(self) -> None:
        """Set a macOS-safe initial size after all diagnostics request space."""
        self.root.update_idletasks()
        min_width = max(MIN_WINDOW_WIDTH, self.root.winfo_reqwidth())
        min_height = max(MIN_WINDOW_HEIGHT, self.root.winfo_reqheight())
        self.root.minsize(min_width, min_height)
        self.root.geometry(
            f"{max(DEFAULT_WINDOW_WIDTH, min_width)}x{max(DEFAULT_WINDOW_HEIGHT, min_height)}"
        )

    def _build_ui(self) -> None:
        root = self.root
        controls = ttk.LabelFrame(root, text="Connection", padding=10)
        controls.grid(row=0, column=0, padx=10, pady=(10, 5), sticky="nsew")
        controls.columnconfigure(1, weight=1)
        ttk.Label(controls, text="Port").grid(row=0, column=0, sticky="w")
        self.port_combo = ttk.Combobox(controls, textvariable=self.port_var, width=28)
        self.port_combo.grid(row=0, column=1, padx=5)
        ttk.Button(controls, text="Refresh", command=self.refresh_ports).grid(row=0, column=2)
        ttk.Label(controls, text="Baud").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(controls, textvariable=self.baud_var, width=12).grid(row=1, column=1, sticky="w", padx=5, pady=(6, 0))
        ttk.Button(controls, text="Connect", command=self.connect).grid(row=1, column=2, pady=(6, 0))
        ttk.Button(controls, text="Disconnect", command=self.disconnect).grid(row=1, column=3, padx=(5, 0), pady=(6, 0))

        collection = ttk.LabelFrame(root, text="Collection", padding=10)
        collection.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")
        collection.columnconfigure(1, weight=1)
        ttk.Label(collection, text="Activity").grid(row=0, column=0, sticky="w")
        ttk.Combobox(collection, textvariable=self.activity_var, values=ACTIVITIES,
                     state="readonly", width=15).grid(row=0, column=1, padx=5)
        ttk.Label(collection, text="Duration (s)").grid(row=0, column=2, padx=(10, 0), sticky="w")
        ttk.Entry(collection, textvariable=self.duration_var, width=8).grid(row=0, column=3, padx=5)
        ttk.Button(collection, text="Start collection", command=self.start_collection).grid(row=1, column=0, columnspan=2, pady=(8, 0), sticky="ew")
        ttk.Button(collection, text="Stop", command=self.stop_collection).grid(row=1, column=2, pady=(8, 0), sticky="ew")
        ttk.Button(collection, text="Open output folder", command=self.open_output_folder).grid(row=1, column=3, pady=(8, 0), sticky="ew")

        live = ttk.LabelFrame(root, text="Live selected target", padding=10)
        live.grid(row=2, column=0, padx=10, pady=5, sticky="nsew")
        live.columnconfigure(1, weight=1)
        fields = (("last 0x0A04 targetnum", self.targetnum_var), ("used cluster_id", self.cluster_id_var),
                  ("x", self.x_var), ("y", self.y_var), ("z", self.z_var),
                  ("host distance √(x²+y²+z²)", self.distance_var), ("dop_idx", self.dop_var),
                  ("frame count", self.frames_var), ("actual rate", self.rate_var))
        for index, (label, value) in enumerate(fields):
            ttk.Label(live, text=f"{label}:").grid(row=index, column=0, sticky="w")
            ttk.Label(live, textvariable=value, width=22).grid(row=index, column=1, sticky="w", padx=(12, 0))
        ttk.Label(live, text="raw selected-target diagnostic:").grid(row=len(fields), column=0, sticky="w")
        ttk.Label(live, textvariable=self.raw_target_var, width=98).grid(row=len(fields), column=1, sticky="w", padx=(12, 0))
        ttk.Label(live, text="selected-target continuity:").grid(row=len(fields) + 1, column=0, sticky="w")
        ttk.Label(live, textvariable=self.continuity_var, width=98).grid(row=len(fields) + 1, column=1, sticky="w", padx=(12, 0))
        ttk.Label(live, text="live data quality:").grid(row=len(fields) + 2, column=0, sticky="w")
        ttk.Label(live, textvariable=self.quality_var, width=98).grid(row=len(fields) + 2, column=1, sticky="w", padx=(12, 0))
        ttk.Label(live, text="quality window:").grid(row=len(fields) + 3, column=0, sticky="w")
        ttk.Label(live, textvariable=self.quality_window_var, width=98).grid(row=len(fields) + 3, column=1, sticky="w", padx=(12, 0))
        selection = ttk.LabelFrame(live, text="Temporary target-selection diagnostic", padding=5)
        selection.grid(row=len(fields) + 4, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        ttk.Radiobutton(
            selection, text="Teammate-compatible", value=TARGET_SELECTION_TEAMMATE,
            variable=self.target_selection_mode_var, command=self._set_target_selection_mode,
        ).grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(
            selection, text="Smart continuity (explicit opt-in)", value=TARGET_SELECTION_AUTO,
            variable=self.target_selection_mode_var, command=self._set_target_selection_mode,
        ).grid(row=0, column=1, sticky="w", padx=(12, 0))
        ttk.Radiobutton(
            selection, text="Force index 0 (debug)", value=TARGET_SELECTION_FORCE_INDEX_0,
            variable=self.target_selection_mode_var, command=self._set_target_selection_mode,
        ).grid(row=0, column=2, sticky="w", padx=(12, 0))
        ttk.Radiobutton(
            selection, text="Force index 1 (debug)", value=TARGET_SELECTION_FORCE_INDEX_1,
            variable=self.target_selection_mode_var, command=self._set_target_selection_mode,
        ).grid(row=0, column=3, sticky="w", padx=(12, 0))
        ttk.Label(selection, textvariable=self.used_target_var, width=100).grid(row=1, column=0, columnspan=2, sticky="w")
        ttk.Label(selection, textvariable=self.sample_freshness_var, width=100).grid(row=2, column=0, columnspan=2, sticky="w")
        ttk.Label(selection, textvariable=self.selection_change_var, width=100).grid(row=3, column=0, columnspan=2, sticky="w")

        table = ttk.LabelFrame(live, text="Latest 0x0A04 targets (first three)", padding=5)
        table.grid(row=len(fields) + 5, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        headings = ("used", "index", "cluster_id", "x (m)", "y (m)", "z (m)", "host distance (m)", "dop_idx")
        for column, heading in enumerate(headings):
            ttk.Label(table, text=heading).grid(row=0, column=column, sticky="w", padx=(0, 10))
        for row, values in enumerate(self.target_table_vars, start=1):
            for column, value in enumerate(values):
                ttk.Label(table, textvariable=value, width=14 if column == 6 else 10).grid(
                    row=row, column=column, sticky="w", padx=(0, 10)
                )

        inference = ttk.LabelFrame(root, text="Diagnostic host inference — alerts disabled", padding=10)
        inference.grid(row=3, column=0, padx=10, pady=5, sticky="nsew")
        inference.columnconfigure(1, weight=1)
        ttk.Label(inference, text="buffer:").grid(row=0, column=0, sticky="w")
        ttk.Label(inference, textvariable=self.inference_buffer_var, width=22).grid(row=0, column=1, sticky="w", padx=(12, 0))
        ttk.Label(inference, text="prediction:").grid(row=1, column=0, sticky="w")
        ttk.Label(inference, textvariable=self.inference_class_var, width=22).grid(row=1, column=1, sticky="w", padx=(12, 0))
        for index, name in enumerate(CLASS_NAMES, start=2):
            ttk.Label(inference, text=f"P({name}):").grid(row=index, column=0, sticky="w")
            ttk.Label(inference, textvariable=self.probability_vars[name], width=22).grid(row=index, column=1, sticky="w", padx=(12, 0))
        ttk.Label(inference, text="inference health:").grid(row=6, column=0, sticky="w")
        ttk.Label(inference, textvariable=self.inference_health_var, width=42).grid(row=6, column=1, sticky="w", padx=(12, 0))
        ttk.Label(inference, text="last reset:").grid(row=7, column=0, sticky="w")
        ttk.Label(inference, textvariable=self.inference_reset_var, width=42).grid(row=7, column=1, sticky="w", padx=(12, 0))
        ttk.Label(inference, text="reset counters:").grid(row=8, column=0, sticky="w")
        ttk.Label(inference, textvariable=self.inference_reset_counts_var, width=42).grid(row=8, column=1, sticky="w", padx=(12, 0))
        ttk.Label(inference, text="missing detail:").grid(row=9, column=0, sticky="w")
        ttk.Label(inference, textvariable=self.inference_missing_var, width=42).grid(row=9, column=1, sticky="w", padx=(12, 0))
        ttk.Label(inference, text="tick timing:").grid(row=10, column=0, sticky="w")
        ttk.Label(inference, textvariable=self.inference_timing_var, width=42).grid(row=10, column=1, sticky="w", padx=(12, 0))
        ttk.Checkbutton(
            inference,
            text="Allow one held tick (diagnostic only)",
            variable=self.diagnostic_tolerance_var,
            command=self._set_diagnostic_tolerance_mode,
        ).grid(row=11, column=0, columnspan=2, sticky="w", pady=(4, 0))
        ttk.Button(inference, text="Copy session stats", command=self._copy_session_stats).grid(row=12, column=0, sticky="w", pady=(4, 0))
        ttk.Label(inference, textvariable=self.inference_mode_var, width=42).grid(row=12, column=1, sticky="w", padx=(12, 0))
        ttk.Label(inference, text="parser health:").grid(row=13, column=0, sticky="w")
        ttk.Label(inference, textvariable=self.parser_var, width=42).grid(row=13, column=1, sticky="w", padx=(12, 0))

        ttk.Label(root, textvariable=self.status_var, anchor="w", padding=10).grid(row=4, column=0, padx=10, pady=(0, 10), sticky="ew")

    @property
    def output_dir(self) -> Path:
        return Path(__file__).resolve().parent.parent / "training_data"

    def _update_target_table(self) -> None:
        """Render the first three raw targets from the latest 0x0A04 frame."""
        for row, variables in enumerate(self.target_table_vars):
            if row < len(self.latest_frame_targets):
                values = format_target_table_row(
                    row, self.latest_frame_targets[row],
                    used=row == self.latest_selected_target_index,
                )
            else:
                values = ("—",) * 8
            for variable, value in zip(variables, values):
                variable.set(value)

    def _record_target_selection_change(self, index: int, target: RadarTarget) -> None:
        index_changed = self.last_selected_target_index is not None and index != self.last_selected_target_index
        cluster_changed = (
            self.last_selected_cluster_id is not None and target.cluster_id != self.last_selected_cluster_id
        )
        if index_changed or cluster_changed:
            reasons = []
            if index_changed:
                reasons.append(f"index {self.last_selected_target_index}→{index}")
            if cluster_changed:
                reasons.append(f"cluster {self.last_selected_cluster_id}→{target.cluster_id}")
            self.selection_change_count += 1
            message = (
                f"[target-selection] {TARGET_SELECTION_LABELS[self.target_selection_mode_var.get()]}: "
                f"{' · '.join(reasons)}"
            )
            print(message)
            self.selection_change_var.set(f"selection changes: {self.selection_change_count} · last: {' · '.join(reasons)}")
        self.last_selected_target_index = index
        self.last_selected_cluster_id = target.cluster_id
        self.previous_selected_target = target

    def _accept_selected_target(
        self, selection: TargetSelection, *, inference_frame_id: int, arrival_at: float,
    ) -> None:
        """Route one selected object to display, quality, CSV state, and inference."""
        selected_target = selection.target
        self.latest_selected_target_index = selection.index
        self.selected_target_is_current = True
        self.selection_reason = selection.reason
        if (
            self.last_selected_cluster_id is not None
            and selected_target.cluster_id != self.last_selected_cluster_id
        ):
            self.selected_cluster_switches += 1
        self._record_target_selection_change(selection.index, selected_target)

        # These consumers deliberately share the same immutable RadarTarget.
        # _collect_due writes latest_target; the quality display reads the
        # history entry; and _advance_inference consumes the queued entry.
        self.latest_target = selected_target
        self._update_target_table()
        self.quality_history.append(selected_target)
        self.last_target_at = arrival_at
        if len(self.inference_queue) >= 120:
            self.inference_queue.clear()
            if self.inference is not None:
                self.last_inference_update = self.inference.reset(
                    "other", detail="inference_queue_overflow"
                )
        self.inference_queue.append((selected_target, inference_frame_id, arrival_at))

    def _set_target_selection_mode(self) -> None:
        """Switch the debug policy; clear inference state so policies never mix."""
        if self.collecting:
            self.stop_collection()
        mode = self.target_selection_mode_var.get()
        if mode not in TARGET_SELECTION_LABELS:
            self.target_selection_mode_var.set(DEFAULT_TARGET_SELECTION_MODE)
            mode = DEFAULT_TARGET_SELECTION_MODE
        self.inference_queue.clear()
        self.next_teammate_inference_frame_id = 0
        self.quality_history.clear()
        self.last_selected_target_index = None
        self.last_selected_cluster_id = None
        self.previous_selected_target = None
        self.latest_selected_target_index = None
        self.selection_reason = None
        self.selected_target_is_current = False
        self.latest_target = None
        if self.port is not None:
            self.port.timeout = 0.01 if mode == TARGET_SELECTION_TEAMMATE else 0  # type: ignore[union-attr]
            self.parser = TinyFrameParser()
        if self.inference is not None:
            self.last_inference_update = self.inference.reset("other", detail="target_selection_mode_changed")
            self._update_inference_values()
        self.used_target_var.set(
            f"selection policy: {TARGET_SELECTION_LABELS[mode]} · applies to next decoded 0x0A04 frame "
            "for CSV / quality / inference"
        )
        self.selection_change_var.set(
            f"selection changes: {self.selection_change_count} · selection policy changed; inference buffer cleared"
        )
        self._update_target_table()

    def refresh_ports(self) -> None:
        try:
            from serial.tools import list_ports  # type: ignore[import-not-found]
        except ImportError:
            self.status_var.set("pyserial is required: python3 -m pip install pyserial")
            return
        ports = [item.device for item in list_ports.comports()]
        self.port_combo["values"] = ports
        if not self.port_var.get() and ports:
            self.port_var.set(ports[0])

    def connect(self) -> None:
        if self.port is not None:
            return
        try:
            import serial  # type: ignore[import-not-found]
            baud = int(self.baud_var.get())
            if baud <= 0:
                raise ValueError("baudrate must be positive")
            if not self.port_var.get().strip():
                raise ValueError("select or enter a serial port")
            if self.target_selection_mode_var.get() == TARGET_SELECTION_TEAMMATE:
                # Match collect_fixed_20hz.py. PySerial defaults supply 8N1 and
                # no flow control; no bytes are ever written by this GUI.
                self.port = serial.Serial(self.port_var.get().strip(), baud, timeout=0.01)
                time.sleep(0.5)
            else:
                self.port = serial.Serial(self.port_var.get().strip(), baudrate=baud, bytesize=8,
                                          parity="N", stopbits=1, timeout=0,
                                          xonxoff=False, rtscts=False, dsrdtr=False)
        except Exception as exc:
            self.port = None
            messagebox.showerror("Connection failed", str(exc))
            return
        self.parser = TinyFrameParser()
        self.latest_target = None
        self.latest_frame_targetnum = None
        self.latest_frame_targets = []
        self.latest_selected_target_index = None
        self.last_selected_target_index = None
        self.last_selected_cluster_id = None
        self.previous_selected_target = None
        self.selection_reason = None
        self.selected_target_is_current = False
        self.selected_cluster_switches = 0
        self.selection_change_count = 0
        self.targetnum_dropouts = 0
        self.quality_history.clear()
        self.raw_target_var.set("waiting for 0x0A04 target-list")
        self.continuity_var.set("cluster switches=0 · targetnum=0 reports=0")
        self.used_target_var.set(
            f"selection policy: {TARGET_SELECTION_LABELS[self.target_selection_mode_var.get()]} · waiting for 0x0A04"
        )
        self.sample_freshness_var.set("sample age: —")
        self.selection_change_var.set("selection changes: 0")
        self._update_target_table()
        self.quality_var.set("WAIT · waiting for first target")
        self.quality_window_var.set(f"last 0/{QUALITY_HISTORY_FRAMES} frames: waiting for coordinates")
        self.inference_queue.clear()
        self.next_teammate_inference_frame_id = 0
        self.next_inference_at = time.monotonic()
        self.last_target_list_at = 0.0
        self.last_empty_target_list_at = 0.0
        if self.inference is not None:
            self.last_inference_update = self.inference.reset("other", detail="connected")
            self._update_inference_values()
        self.status_var.set("connected · waiting for target")

    def disconnect(self) -> None:
        self.stop_collection()
        if self.port is not None:
            try:
                self.port.close()  # type: ignore[union-attr]
            finally:
                self.port = None
        self.latest_target = None
        self.inference_queue.clear()
        if self.inference is not None:
            self.last_inference_update = self.inference.reset("other", detail="disconnected")
            self._update_inference_values()
        self.status_var.set("disconnected")

    def start_collection(self) -> None:
        if self.port is None:
            messagebox.showerror("Not connected", "Connect to the radar VCP first.")
            return
        try:
            duration = float(self.duration_var.get())
            if not math.isfinite(duration) or duration <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid duration", "Duration must be a positive number of seconds.")
            return
        if not self.activity_var.get() in ACTIVITIES:
            messagebox.showerror("Invalid activity", "Choose one of the listed activity labels.")
            return

        # Historical filenames and CSV timestamps use local wall-clock time.
        started_at = datetime.now()
        try:
            teammate_compatible = self.target_selection_mode_var.get() == TARGET_SELECTION_TEAMMATE
            self.writer = SingleTarget20HzWriter(
                self.output_dir, self.activity_var.get(), started_at,
                teammate_compatible=teammate_compatible,
            )
        except FileExistsError:
            messagebox.showerror("Output exists", "A file with this second-level name already exists; try again next second.")
            return
        now = time.monotonic()
        self.collecting = True
        countdown_s = 3.0 if teammate_compatible else 0.0
        self.collection_starts_at = now + countdown_s
        self.collection_deadline = self.collection_starts_at + duration
        self.next_sample_at = self.collection_starts_at
        self.sample_times.clear()
        self.frames_var.set("0")
        self.status_var.set(
            "connected · teammate countdown 3 s" if teammate_compatible
            else "connected · collecting · waiting for target"
        )

    def stop_collection(self) -> None:
        if self.writer is not None:
            path = self.writer.path
            wrote_rows = self.writer.frame > 0
            self.writer.close()
            self.writer = None
            if wrote_rows and path.exists():
                self.status_var.set(
                    f"connected · collection saved: {path.name}" if self.port else f"saved: {path.name}"
                )
            else:
                self.status_var.set("connected · no valid target; no CSV created" if self.port else "no CSV created")
            if self.inference is not None:
                print(self.inference.format_session_stats())
        self.collecting = False

    def _decode_available(self) -> None:
        if self.port is None:
            return
        if self.target_selection_mode_var.get() == TARGET_SELECTION_TEAMMATE:
            waiting = int(self.port.in_waiting)  # type: ignore[union-attr]
            if waiting <= 0:
                return
            target = parse_teammate_target_chunk(
                self.port.read(waiting)  # type: ignore[union-attr]
            )
            if target is None:
                return
            arrival_at = time.monotonic()
            self.latest_frame_targetnum = target.target_num
            self.latest_frame_targets = [target]
            self._accept_selected_target(
                TargetSelection(0, target, "teammate first target + legacy gates"),
                inference_frame_id=self.next_teammate_inference_frame_id,
                arrival_at=arrival_at,
            )
            self.next_teammate_inference_frame_id += 1
            return
        while True:
            data = self.port.read(512)  # type: ignore[union-attr]
            if not data:
                break
            for frame in self.parser.feed(data):
                try:
                    decoded = decode_report(frame)
                except ValueError:
                    continue
                if decoded is None or decoded["report"] != "target_list":
                    continue
                arrival_at = time.monotonic()
                self.last_target_list_at = arrival_at
                self.latest_frame_targetnum = int(decoded["target_num"])
                self.latest_frame_targets = [
                    RadarTarget(
                        target_num=self.latest_frame_targetnum,
                        cluster_id=int(item["cluster_id"]),
                        x=float(item["x_m"]), y=float(item["y_m"]), z=float(item["z_m"]),
                        dop_idx=int(item["dop_idx"]),
                    )
                    for item in decoded["targets"]
                ]
                if not self.latest_frame_targets:
                    self.selected_target_is_current = False
                    self.selection_reason = "no target in frame"
                    self._update_target_table()
                    self.last_empty_target_list_at = arrival_at
                    self.targetnum_dropouts += 1
                    self.continuity_var.set(
                        f"cluster switches={self.selected_cluster_switches} · "
                        f"targetnum=0 reports={self.targetnum_dropouts}"
                    )
                    continue
                selection = rank_target(
                    self.latest_frame_targets,
                    mode=self.target_selection_mode_var.get(),
                    previous_target=self.previous_selected_target,
                )
                if selection is None:
                    self.latest_selected_target_index = None
                    self.selected_target_is_current = False
                    self.selection_reason = "no strict-quality target"
                    self._update_target_table()
                    self.used_target_var.set(
                        "no target selected for CSV / quality / inference: no finite strict-quality target"
                    )
                    self.sample_freshness_var.set("sample age: — · no current selected target")
                    continue
                self._accept_selected_target(
                    selection,
                    inference_frame_id=int(decoded["frame_id"]),
                    arrival_at=arrival_at,
                )

    def _update_live_values(self, now: float) -> bool:
        if self.target_selection_mode_var.get() == TARGET_SELECTION_TEAMMATE:
            # The old collector holds the last accepted target indefinitely,
            # including across empty/invalid reports.
            tracking = self.latest_target is not None
        else:
            tracking = (
                self.selected_target_is_current
                and self.latest_target is not None
                and now - self.last_target_at <= TRACKING_TIMEOUT_S
            )
        self.targetnum_var.set("—" if self.latest_frame_targetnum is None else str(self.latest_frame_targetnum))
        if tracking and self.latest_target is not None:
            target = self.latest_target
            age_s = now - self.last_target_at
            freshness = "fresh" if age_s <= SAMPLE_PERIOD_S else "stale"
            self.cluster_id_var.set(str(target.cluster_id))
            self.x_var.set(f"{target.x:.3f} m")
            self.y_var.set(f"{target.y:.3f} m")
            self.z_var.set(f"{target.z:.3f} m")
            self.distance_var.set(f"{target.distance:.3f} m")
            self.dop_var.set(str(target.dop_idx))
            self.raw_target_var.set(format_raw_target_diagnostic(target, age_s=age_s))
            self.used_target_var.set(
                f"USED by CSV / quality / inference: {TARGET_SELECTION_LABELS[self.target_selection_mode_var.get()]} · "
                f"index={self.latest_selected_target_index} · cluster_id={target.cluster_id} · "
                f"reason={self.selection_reason or 'unknown'}"
            )
            self.sample_freshness_var.set(f"sample age: {age_s * 1000:.0f} ms · {freshness}")
            self.continuity_var.set(
                f"cluster switches={self.selected_cluster_switches} · "
                f"targetnum=0 reports={self.targetnum_dropouts}"
            )
            quality = summarize_live_data_quality(self.quality_history)
            self.quality_var.set(
                f"{quality.level} · {quality.reason} · "
                f"distance {'in' if quality.current_distance_in_range else 'outside'} 0.5–6.0 m · "
                f"Z {'in' if quality.current_z_in_range else 'outside'} −2.5–1.0 m"
            )
            if quality.distance_mean_m is None or quality.z_mean_m is None:
                self.quality_window_var.set(
                    f"last {quality.sample_count}/{QUALITY_HISTORY_FRAMES} frames: no finite coordinate summary"
                )
            else:
                jump = "n/a" if quality.jump_percent is None else f"{quality.jump_percent:.1f}%"
                cv = "n/a" if quality.average_cv is None else f"{quality.average_cv:.3f}"
                self.quality_window_var.set(
                    f"last {quality.sample_count}/{QUALITY_HISTORY_FRAMES}: "
                    f"distance min/max/mean={quality.distance_min_m:.2f}/{quality.distance_max_m:.2f}/{quality.distance_mean_m:.2f} m · "
                    f"Z min/max/mean={quality.z_min_m:.2f}/{quality.z_max_m:.2f}/{quality.z_mean_m:.2f} m · "
                    f"jumps={jump} · avg CV={cv}"
                )
        else:
            self.cluster_id_var.set("—")
            self.x_var.set("—")
            self.y_var.set("—")
            self.z_var.set("—")
            self.distance_var.set("—")
            self.dop_var.set("—")
            self.raw_target_var.set("no target within the 0.5 s tracking timeout")
            if self.latest_target is not None:
                age_s = now - self.last_target_at
                self.used_target_var.set(
                    f"last selected target: index={self.latest_selected_target_index} · "
                    f"cluster_id={self.latest_target.cluster_id} · {self.selection_reason or 'no current selection'}"
                )
                self.sample_freshness_var.set(f"sample age: {age_s * 1000:.0f} ms · stale")
            else:
                self.used_target_var.set(
                    f"selection policy: {TARGET_SELECTION_LABELS[self.target_selection_mode_var.get()]} · no selected target"
                )
                self.sample_freshness_var.set("sample age: — · stale")
            self.quality_var.set("WAIT · no target within tracking timeout")
            self.quality_window_var.set(f"last {len(self.quality_history)}/{QUALITY_HISTORY_FRAMES} frames: waiting for a fresh target")
        return tracking

    def _update_inference_values(self) -> None:
        if self.inference is None:
            detail = self.inference_startup_error or "pinned host model unavailable"
            self.inference_buffer_var.set("unavailable")
            self.inference_class_var.set("—")
            self.inference_health_var.set(f"unavailable: {detail}")
            self.inference_reset_var.set("unavailable")
            self.inference_reset_counts_var.set("resets: unavailable")
            self.inference_missing_var.set("missing detail: unavailable")
            self.inference_timing_var.set("timing: unavailable")
            self.inference_mode_var.set("diagnostic tolerance: unavailable")
            for value in self.probability_vars.values():
                value.set("—")
            return
        snapshot = self.inference.diagnostic_snapshot()
        self.inference_buffer_var.set(f"{snapshot.buffer_length} / 60")
        self.inference_health_var.set(snapshot.state)
        if snapshot.last_reset_reason is None:
            self.inference_reset_var.set("none")
        else:
            label = RESET_REASON_LABELS[snapshot.last_reset_reason]
            detail = f" ({snapshot.last_reset_detail})" if snapshot.last_reset_detail else ""
            self.inference_reset_var.set(f"{label}{detail}")
        counts = snapshot.reset_counts
        self.inference_reset_counts_var.set(
            "resets: "
            f"missing={counts['missing_record']} · duplicate={counts['duplicate_frame_id']} · "
            f"invalid={counts['invalid_value']} · ts_gap={counts['timestamp_gap']} · other={counts['other']}"
        )
        missing_counts = snapshot.missing_subreason_counts
        if snapshot.last_missing_subreason is None:
            self.inference_missing_var.set("missing detail: none")
        else:
            self.inference_missing_var.set(
                f"{MISSING_SUBREASON_LABELS[snapshot.last_missing_subreason]} · "
                f"n={missing_counts['no_fresh_frame']} q={missing_counts['queue_empty']} "
                f"late={missing_counts['late_frame']} inactive={missing_counts['tracking_inactive']} "
                f"other={missing_counts['other_missing']} held={snapshot.held_missed_ticks}"
            )
        if snapshot.last_arrival_jitter_s is None:
            self.inference_timing_var.set(
                f"timing: next={snapshot.expected_next_tick_s!s} · last=— · queue={snapshot.queue_depth}/{snapshot.max_queue_depth}"
            )
        else:
            self.inference_timing_var.set(
                f"jitter={snapshot.last_arrival_jitter_s * 1000:+.1f} ms · "
                f"next={snapshot.expected_next_tick_s:.3f} · last={snapshot.last_accepted_record_timestamp_s:.3f} · "
                f"queue={snapshot.queue_depth}/{snapshot.max_queue_depth}"
            )
        mode = "ON — one held tick; invalid windows disabled" if snapshot.diagnostic_tolerance_mode else "OFF"
        self.inference_mode_var.set(f"diagnostic tolerance: {mode}")
        prediction = self.inference.last_prediction
        if prediction is None or prediction.probabilities is None:
            self.inference_class_var.set("—")
            for value in self.probability_vars.values():
                value.set("—")
            return
        self.inference_class_var.set(prediction.predicted_class or "—")
        for index, name in enumerate(CLASS_NAMES):
            self.probability_vars[name].set(f"{float(prediction.probabilities[index]):.3f}")

    def _set_diagnostic_tolerance_mode(self) -> None:
        if self.inference is not None:
            self.inference.diagnostic_tolerance_mode = bool(self.diagnostic_tolerance_var.get())
        self._update_inference_values()

    def _copy_session_stats(self) -> None:
        if self.inference is None:
            return
        summary = self.inference.format_session_stats()
        self.root.clipboard_clear()
        self.root.clipboard_append(summary)
        self.status_var.set("host inference session stats copied to clipboard")

    def _advance_inference(self, now: float, tracking: bool) -> None:
        if self.inference is None or self.port is None:
            self._update_inference_values()
            return
        if not tracking:
            if self.inference.buffer_progress:
                self.last_inference_update = self.inference.mark_missing_record(
                    "tracking_inactive",
                    expected_tick_s=self.next_inference_at or now,
                    queue_depth=len(self.inference_queue),
                    allow_diagnostic_hold=False,
                )
            self.inference_queue.clear()
            self.next_inference_at = now
            self.inference.update_transport_diagnostics(
                expected_next_tick_s=self.next_inference_at,
                queue_depth=0,
            )
            self._update_inference_values()
            return
        if self.next_inference_at == 0.0:
            self.next_inference_at = now
        while now >= self.next_inference_at:
            if not self.inference_queue:
                # Primary stability fix: the former strict-tick consumer reset
                # before a just-arriving decoded frame could be queued. Wait at
                # most 20 ms; no input is fabricated during this handoff grace.
                if should_wait_for_fresh_frame(
                    now_s=now,
                    expected_tick_s=self.next_inference_at,
                    queue_depth=0,
                    handoff_grace_s=FRESH_FRAME_HANDOFF_GRACE_S,
                ):
                    self.inference.update_transport_diagnostics(
                        expected_next_tick_s=self.next_inference_at,
                        queue_depth=0,
                    )
                    break
                recent_empty_report = self.last_empty_target_list_at >= (
                    self.next_inference_at - SAMPLE_PERIOD_S - self.inference.jitter_tolerance_s
                )
                self.last_inference_update = self.inference.mark_missing_record(
                    "no_fresh_frame" if recent_empty_report else "queue_empty",
                    expected_tick_s=self.next_inference_at,
                    queue_depth=0,
                )
            else:
                target, frame_id, arrival_at = self.inference_queue.popleft()
                if arrival_at - self.next_inference_at > FRESH_FRAME_HANDOFF_GRACE_S:
                    self.inference.record_frame_lateness(
                        expected_tick_s=self.next_inference_at,
                        arrival_timestamp_s=arrival_at,
                    )
                    self.last_inference_update = self.inference.mark_missing_record(
                        "late_frame",
                        expected_tick_s=self.next_inference_at,
                        queue_depth=len(self.inference_queue),
                    )
                else:
                    record = LiveRadarRecord.from_first_target(
                        timestamp_s=self.next_inference_at,
                        frame_id=frame_id,
                        x=target.x, y=target.y, z=target.z, dop_idx=target.dop_idx,
                        arrival_timestamp_s=arrival_at,
                    )
                    self.last_inference_update = self.inference.ingest(record)
            self.next_inference_at += SAMPLE_PERIOD_S
        self.inference.update_transport_diagnostics(
            expected_next_tick_s=self.next_inference_at,
            queue_depth=len(self.inference_queue),
        )
        self._update_inference_values()

    def _collect_due(self, now: float, tracking: bool) -> None:
        if not self.collecting or self.writer is None:
            return
        if now < self.collection_starts_at:
            remaining = max(1, math.ceil(self.collection_starts_at - now))
            self.status_var.set(f"connected · teammate countdown {remaining} s")
            return
        if now >= self.collection_deadline:
            self.stop_collection()
            return
        if self.writer.teammate_compatible:
            if now >= self.next_sample_at:
                if self.latest_target is not None:
                    self.writer.write(
                        self.latest_target, self.activity_var.get(), sampled_at=datetime.now()
                    )
                    self.sample_times.append(now)
                    self.frames_var.set(str(self.writer.frame))
                self.next_sample_at = teammate_next_sample_time(self.next_sample_at, now)
            return
        while now >= self.next_sample_at:
            if tracking and self.latest_target is not None:
                self.writer.write(self.latest_target, self.activity_var.get())
                self.sample_times.append(now)
                self.frames_var.set(str(self.writer.frame))
            self.next_sample_at += SAMPLE_PERIOD_S

    def _update_status(self, now: float, tracking: bool) -> None:
        while self.sample_times and now - self.sample_times[0] > 1.0:
            self.sample_times.popleft()
        if len(self.sample_times) >= 2:
            elapsed = self.sample_times[-1] - self.sample_times[0]
            rate = (len(self.sample_times) - 1) / elapsed if elapsed else 0.0
        else:
            rate = 0.0
        self.rate_var.set(f"{rate:.1f} Hz")
        stats = self.parser.stats
        errors = stats.invalid_checksums + stats.oversized_frames + stats.timeouts
        self.parser_var.set(
            f"frames={stats.valid_frames} · errors={errors} · "
            f"discarded={stats.discarded_bytes} · queue={len(self.inference_queue)}"
        )
        if self.port is None:
            return
        if self.collecting and now < self.collection_starts_at:
            remaining = max(1, math.ceil(self.collection_starts_at - now))
            self.status_var.set(f"connected · teammate countdown {remaining} s")
            return
        mode = "collecting" if self.collecting else "connected"
        state = "tracking" if tracking else "waiting for target"
        self.status_var.set(f"{mode} · {state}")

    def poll(self) -> None:
        try:
            self._decode_available()
            now = time.monotonic()
            tracking = self._update_live_values(now)
            self._advance_inference(now, tracking)
            self._collect_due(now, tracking)
            self._update_status(now, tracking)
        except Exception as exc:
            self.stop_collection()
            self.status_var.set(f"serial error: {exc}")
        finally:
            self.root.after(10, self.poll)

    def open_output_folder(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if sys.platform == "darwin":
            subprocess.run(["open", str(self.output_dir)], check=False)
        elif os.name == "nt":
            os.startfile(self.output_dir)  # type: ignore[attr-defined]
        else:
            subprocess.run(["xdg-open", str(self.output_dir)], check=False)

    def close(self) -> None:
        self.disconnect()
        self.root.destroy()


def main() -> None:
    if tk is None:
        raise SystemExit(
            "Tkinter is unavailable in this Python installation. Use a Python build with Tk support."
        )
    root = tk.Tk()
    RadarCollectionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
