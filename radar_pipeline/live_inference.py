"""Hash-pinned causal host inference for the diagnostic LD6002B demo.

This module deliberately implements only the host demonstration path.  It does
not export a model, alter STM32 firmware, or turn a fall probability into an
alert.  A caller must supply one fresh first-target record at each scheduled
20 Hz tick; repeated targets and missing ticks are invalid rather than padded.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn


BASE_FEATURES = (
    "x",
    "y",
    "z",
    "dop_idx",
    "range_m",
    "azimuth_deg",
    "elevation_deg",
)
DERIVED_FEATURES = (
    "xyz_delta_mag",
    "x_roll_std",
    "y_roll_std",
    "z_roll_std",
    "range_roll_std",
    "range_centered",
)
CLASS_NAMES = ("walking", "standing", "sitting", "fall")
CLIP_LENGTH = 60
STRIDE_FRAMES = 15
SAMPLE_RATE_HZ = 20.0
SAMPLE_PERIOD_S = 1.0 / SAMPLE_RATE_HZ
FRESH_FRAME_HANDOFF_GRACE_S = 0.020

# Keep the externally visible taxonomy small and stable. More specific source
# details are carried separately so GUI counters remain readable.
RESET_REASON_MISSING_RECORD = "missing_record"
RESET_REASON_DUPLICATE_FRAME_ID = "duplicate_frame_id"
RESET_REASON_INVALID_VALUE = "invalid_value"
RESET_REASON_TIMESTAMP_GAP = "timestamp_gap"
RESET_REASON_OTHER = "other"
RESET_REASON_CODES = (
    RESET_REASON_MISSING_RECORD,
    RESET_REASON_DUPLICATE_FRAME_ID,
    RESET_REASON_INVALID_VALUE,
    RESET_REASON_TIMESTAMP_GAP,
    RESET_REASON_OTHER,
)
RESET_REASON_LABELS = {
    RESET_REASON_MISSING_RECORD: "missing record",
    RESET_REASON_DUPLICATE_FRAME_ID: "duplicate frame ID",
    RESET_REASON_INVALID_VALUE: "invalid value",
    RESET_REASON_TIMESTAMP_GAP: "timestamp gap",
    RESET_REASON_OTHER: "other reset",
}
MISSING_SUBREASON_NO_FRESH_FRAME = "no_fresh_frame"
MISSING_SUBREASON_QUEUE_EMPTY = "queue_empty"
MISSING_SUBREASON_LATE_FRAME = "late_frame"
MISSING_SUBREASON_TRACKING_INACTIVE = "tracking_inactive"
MISSING_SUBREASON_OTHER = "other_missing"
MISSING_SUBREASON_CODES = (
    MISSING_SUBREASON_NO_FRESH_FRAME,
    MISSING_SUBREASON_QUEUE_EMPTY,
    MISSING_SUBREASON_LATE_FRAME,
    MISSING_SUBREASON_TRACKING_INACTIVE,
    MISSING_SUBREASON_OTHER,
)
MISSING_SUBREASON_LABELS = {
    MISSING_SUBREASON_NO_FRESH_FRAME: "no fresh target frame",
    MISSING_SUBREASON_QUEUE_EMPTY: "target queue empty",
    MISSING_SUBREASON_LATE_FRAME: "fresh frame arrived late",
    MISSING_SUBREASON_TRACKING_INACTIVE: "tracking inactive",
    MISSING_SUBREASON_OTHER: "other missing record",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def should_wait_for_fresh_frame(
    *, now_s: float, expected_tick_s: float, queue_depth: int, handoff_grace_s: float = FRESH_FRAME_HANDOFF_GRACE_S
) -> bool:
    """Allow a bounded post-tick handoff delay before declaring a real miss.

    This solves a host scheduling race only: no record is appended while
    waiting, and a queue-empty tick still fails closed once its short grace
    deadline expires.
    """
    if handoff_grace_s < 0:
        raise ValueError("handoff_grace_s must be non-negative")
    return queue_depth == 0 and now_s < expected_tick_s + handoff_grace_s


class CNNBiLSTM(nn.Module):
    """The frozen checkpoint architecture, kept local to avoid training imports."""

    def __init__(self, dropout_input: float, dropout_hidden: float) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(13, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(64, 96, kernel_size=3, padding=1),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.lstm = nn.LSTM(96, 64, num_layers=1, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_input), nn.Linear(128, 64), nn.ReLU(),
            nn.Dropout(dropout_hidden), nn.Linear(64, 4),
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.transpose(1, 2)
        tensor = self.conv(tensor)
        tensor = tensor.transpose(1, 2)
        output, _ = self.lstm(tensor)
        return self.classifier(output[:, -1, :])


@dataclass(frozen=True)
class LiveRadarRecord:
    """One first-target tracker record, with its host 20 Hz tick and frame ID."""

    timestamp_s: float
    frame_id: int
    x: float
    y: float
    z: float
    dop_idx: int
    range_m: float
    azimuth_deg: float
    elevation_deg: float
    arrival_timestamp_s: float | None = None

    @classmethod
    def from_first_target(
        cls, *, timestamp_s: float, frame_id: int, x: float, y: float, z: float, dop_idx: int,
        arrival_timestamp_s: float | None = None,
    ) -> "LiveRadarRecord":
        horizontal = math.hypot(x, y)
        return cls(
            timestamp_s=timestamp_s,
            frame_id=frame_id,
            x=x,
            y=y,
            z=z,
            dop_idx=dop_idx,
            range_m=math.sqrt(x * x + y * y + z * z),
            azimuth_deg=math.degrees(math.atan2(x, y)),
            elevation_deg=math.degrees(math.atan2(z, horizontal)),
            arrival_timestamp_s=arrival_timestamp_s,
        )

    def base_values(self) -> np.ndarray:
        return np.asarray(
            [self.x, self.y, self.z, self.dop_idx, self.range_m, self.azimuth_deg, self.elevation_deg],
            dtype=np.float32,
        )


@dataclass(frozen=True)
class DemoBinding:
    manifest_path: Path
    checkpoint_path: Path
    normalization_path: Path
    mean: np.ndarray
    std: np.ndarray
    model: CNNBiLSTM


@dataclass(frozen=True)
class InferenceUpdate:
    buffer_progress: int
    state: str
    reset_reason: str | None
    tensor: np.ndarray | None
    predicted_class: str | None
    probabilities: np.ndarray | None


@dataclass(frozen=True)
class InferenceDiagnostics:
    """Small, UI-safe state snapshot for diagnosing fail-closed resets."""

    buffer_length: int
    state: str
    last_reset_reason: str | None
    last_reset_detail: str | None
    reset_counts: Mapping[str, int]
    last_missing_subreason: str | None
    missing_subreason_counts: Mapping[str, int]
    expected_next_tick_s: float | None
    last_accepted_record_timestamp_s: float | None
    last_record_arrival_s: float | None
    last_arrival_jitter_s: float | None
    queue_depth: int
    max_queue_depth: int
    diagnostic_tolerance_mode: bool
    held_missed_ticks: int
    total_held_tick_events: int
    total_resets: int
    positive_lateness_samples: int
    mean_positive_lateness_s: float | None
    max_positive_lateness_s: float | None


def _resolve(manifest_path: Path, relative_path: str) -> Path:
    return (manifest_path.parent / relative_path).resolve()


def load_demo_binding(manifest_path: Path) -> DemoBinding:
    """Load only the selected host-demo pair after validating all file hashes."""
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("purpose") != "diagnostic_host_demo" or manifest.get("alert_enabled") is not False:
        raise ValueError("Host inference requires the diagnostic-only demo manifest")
    checkpoint_path = _resolve(manifest_path, manifest["checkpoint"]["path"])
    normalization_path = _resolve(manifest_path, manifest["normalization"]["path"])
    for path, expected in (
        (checkpoint_path, manifest["checkpoint"]["sha256"]),
        (normalization_path, manifest["normalization"]["sha256"]),
    ):
        if sha256_file(path) != expected:
            raise ValueError(f"SHA-256 mismatch for {path}")
    normalizer = json.loads(normalization_path.read_text(encoding="utf-8"))
    if tuple(normalizer.get("feature_order", ())) != BASE_FEATURES:
        raise ValueError("Normalization feature order does not match the host contract")
    mean = np.asarray(normalizer.get("mean"), dtype=np.float32)
    std = np.asarray(normalizer.get("std"), dtype=np.float32)
    if mean.shape != (7,) or std.shape != (7,) or not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError("Normalization arrays must be seven finite values")
    if np.any(std <= 0):
        raise ValueError("The selected demo normalizer does not permit non-positive standard deviations")

    checkpoint: dict[str, Any] = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if tuple(checkpoint.get("base_feature_names", ())) != BASE_FEATURES:
        raise ValueError("Checkpoint base feature order does not match the host contract")
    checkpoint_norm = checkpoint.get("normalization", {})
    if not (
        np.array_equal(mean, np.asarray(checkpoint_norm.get("mean"), dtype=np.float32))
        and np.array_equal(std, np.asarray(checkpoint_norm.get("std"), dtype=np.float32))
    ):
        raise ValueError("Pinned normalization file does not exactly match the checkpoint")
    config = checkpoint["train_config"]
    model = CNNBiLSTM(float(config["dropout_input"]), float(config["dropout_hidden"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return DemoBinding(manifest_path, checkpoint_path, normalization_path, mean, std, model)


class CausalHostInference:
    """Fail-closed 20 Hz buffer and inference cadence for the host demo."""

    def __init__(
        self,
        binding: DemoBinding,
        *,
        jitter_tolerance_s: float = 0.020,
        diagnostic_tolerance_mode: bool = False,
    ) -> None:
        if jitter_tolerance_s < 0:
            raise ValueError("jitter_tolerance_s must be non-negative")
        self.binding = binding
        self.jitter_tolerance_s = jitter_tolerance_s
        self.diagnostic_tolerance_mode = diagnostic_tolerance_mode
        self._records: deque[LiveRadarRecord] = deque(maxlen=CLIP_LENGTH)
        self._last_timestamp_s: float | None = None
        self._last_frame_id: int | None = None
        self._valid_since_reset = 0
        self._state = "buffering"
        self.last_reset_reason: str | None = None
        self.last_reset_detail: str | None = None
        self._reset_counts = {reason: 0 for reason in RESET_REASON_CODES}
        self.last_missing_subreason: str | None = None
        self._missing_subreason_counts = {reason: 0 for reason in MISSING_SUBREASON_CODES}
        self._expected_next_tick_s: float | None = None
        self._last_accepted_record_timestamp_s: float | None = None
        self._last_record_arrival_s: float | None = None
        self._last_arrival_jitter_s: float | None = None
        self._queue_depth = 0
        self._max_queue_depth = 0
        self._held_missing_tick = False
        self._held_missed_ticks = 0
        self._total_held_tick_events = 0
        self._positive_lateness_count = 0
        self._positive_lateness_total_s = 0.0
        self._max_positive_lateness_s: float | None = None
        self.last_prediction: InferenceUpdate | None = None

    @property
    def buffer_progress(self) -> int:
        return len(self._records)

    @property
    def state(self) -> str:
        return self._state

    def diagnostic_snapshot(self) -> InferenceDiagnostics:
        return InferenceDiagnostics(
            buffer_length=self.buffer_progress,
            state=self.state,
            last_reset_reason=self.last_reset_reason,
            last_reset_detail=self.last_reset_detail,
            reset_counts=dict(self._reset_counts),
            last_missing_subreason=self.last_missing_subreason,
            missing_subreason_counts=dict(self._missing_subreason_counts),
            expected_next_tick_s=self._expected_next_tick_s,
            last_accepted_record_timestamp_s=self._last_accepted_record_timestamp_s,
            last_record_arrival_s=self._last_record_arrival_s,
            last_arrival_jitter_s=self._last_arrival_jitter_s,
            queue_depth=self._queue_depth,
            max_queue_depth=self._max_queue_depth,
            diagnostic_tolerance_mode=self.diagnostic_tolerance_mode,
            held_missed_ticks=self._held_missed_ticks,
            total_held_tick_events=self._total_held_tick_events,
            total_resets=sum(self._reset_counts.values()),
            positive_lateness_samples=self._positive_lateness_count,
            mean_positive_lateness_s=(
                self._positive_lateness_total_s / self._positive_lateness_count
                if self._positive_lateness_count else None
            ),
            max_positive_lateness_s=self._max_positive_lateness_s,
        )

    def format_session_stats(self) -> str:
        """Return one copy/paste-safe host-session summary for a live demo log."""
        snapshot = self.diagnostic_snapshot()
        resets = snapshot.reset_counts
        missing = snapshot.missing_subreason_counts
        mean_late = "n/a" if snapshot.mean_positive_lateness_s is None else f"{snapshot.mean_positive_lateness_s * 1000:.1f}ms"
        max_late = "n/a" if snapshot.max_positive_lateness_s is None else f"{snapshot.max_positive_lateness_s * 1000:.1f}ms"
        return (
            "host inference session stats | "
            f"resets_total={snapshot.total_resets} "
            f"(missing={resets[RESET_REASON_MISSING_RECORD]}, duplicate={resets[RESET_REASON_DUPLICATE_FRAME_ID]}, "
            f"invalid={resets[RESET_REASON_INVALID_VALUE]}, ts_gap={resets[RESET_REASON_TIMESTAMP_GAP]}, other={resets[RESET_REASON_OTHER]}) | "
            f"missing_subcauses=(no_fresh={missing[MISSING_SUBREASON_NO_FRESH_FRAME]}, "
            f"queue_empty={missing[MISSING_SUBREASON_QUEUE_EMPTY]}, late={missing[MISSING_SUBREASON_LATE_FRAME]}, "
            f"tracking_inactive={missing[MISSING_SUBREASON_TRACKING_INACTIVE]}, other={missing[MISSING_SUBREASON_OTHER]}) | "
            f"queue_max={snapshot.max_queue_depth} | positive_lateness_mean={mean_late} "
            f"positive_lateness_max={max_late} samples={snapshot.positive_lateness_samples} | "
            f"held_tick_events={snapshot.total_held_tick_events}"
        )

    def update_transport_diagnostics(self, *, expected_next_tick_s: float, queue_depth: int) -> None:
        """Record GUI scheduler/queue state without changing inference validity."""
        self._expected_next_tick_s = expected_next_tick_s
        self._queue_depth = max(0, queue_depth)
        self._max_queue_depth = max(self._max_queue_depth, self._queue_depth)

    def record_frame_lateness(self, *, expected_tick_s: float, arrival_timestamp_s: float) -> None:
        """Record an observed arrival offset, including a frame rejected as late."""
        self._last_record_arrival_s = arrival_timestamp_s
        self._last_arrival_jitter_s = arrival_timestamp_s - expected_tick_s
        if self._last_arrival_jitter_s > 0:
            self._positive_lateness_count += 1
            self._positive_lateness_total_s += self._last_arrival_jitter_s
            self._max_positive_lateness_s = max(
                self._max_positive_lateness_s or 0.0,
                self._last_arrival_jitter_s,
            )

    def mark_missing_record(
        self,
        subreason: str,
        *,
        expected_tick_s: float,
        queue_depth: int,
        allow_diagnostic_hold: bool = True,
    ) -> InferenceUpdate:
        """Account for a missing tick and optionally hold exactly one invalid window.

        The optional hold is diagnostic-only. It appends no replacement sample,
        exposes the state as disabled, and blocks inference until 60 later
        fresh records have displaced every pre-gap record from the buffer.
        """
        if subreason not in MISSING_SUBREASON_CODES:
            subreason = MISSING_SUBREASON_OTHER
        self.update_transport_diagnostics(expected_next_tick_s=expected_tick_s, queue_depth=queue_depth)
        self.last_missing_subreason = subreason
        self._missing_subreason_counts[subreason] += 1
        if (
            self.diagnostic_tolerance_mode
            and allow_diagnostic_hold
            and not self._held_missing_tick
            and not self._held_missed_ticks
        ):
            self._held_missing_tick = True
            self._held_missed_ticks += 1
            self._total_held_tick_events += 1
            # A retained buffer may contain a timing hole. Start a fresh valid
            # record count and prohibit model calls until it has rolled out.
            self._valid_since_reset = 0
            self._state = "disabled"
            self.last_prediction = None
            return InferenceUpdate(self.buffer_progress, "disabled", None, None, None, None)
        return self.reset(RESET_REASON_MISSING_RECORD, detail=subreason)

    def reset(self, reason: str, *, detail: str | None = None) -> InferenceUpdate:
        """Clear the causal buffer and account for the structured reason code."""
        if reason not in RESET_REASON_CODES:
            detail = detail or reason
            reason = RESET_REASON_OTHER
        self._records.clear()
        self._last_timestamp_s = None
        self._last_frame_id = None
        self._valid_since_reset = 0
        self._state = "buffering"
        self._held_missing_tick = False
        self._held_missed_ticks = 0
        self.last_reset_reason = reason
        self.last_reset_detail = detail
        self._reset_counts[reason] += 1
        self.last_prediction = None
        return InferenceUpdate(0, "buffering", reason, None, None, None)

    def _invalid(self, reason: str, detail: str) -> InferenceUpdate:
        return self.reset(reason, detail=detail)

    def _validate(self, record: LiveRadarRecord) -> tuple[str, str] | None:
        values = record.base_values()
        if not np.isfinite(values).all() or not math.isfinite(record.timestamp_s):
            return RESET_REASON_INVALID_VALUE, "non_finite_record"
        if int(record.dop_idx) != record.dop_idx:
            return RESET_REASON_INVALID_VALUE, "non_integer_dop_idx"
        if self._last_frame_id is not None and record.frame_id == self._last_frame_id:
            return RESET_REASON_DUPLICATE_FRAME_ID, "duplicate_frame_id"
        if self._last_timestamp_s is not None:
            delta = record.timestamp_s - self._last_timestamp_s
            if delta <= 0:
                return RESET_REASON_TIMESTAMP_GAP, "non_monotonic_timestamp"
            expected_delta = SAMPLE_PERIOD_S * (2 if self._held_missing_tick else 1)
            if abs(delta - expected_delta) > self.jitter_tolerance_s:
                return RESET_REASON_TIMESTAMP_GAP, "timestamp_discontinuity"
        return None

    def current_tensor(self) -> np.ndarray:
        if len(self._records) != CLIP_LENGTH:
            raise RuntimeError("A 60-record buffer is required before tensor construction")
        base = np.stack([record.base_values() for record in self._records]).astype(np.float32)
        normalized = ((base - self.binding.mean) / self.binding.std).astype(np.float32)
        xyz = normalized[:, :3]
        delta = np.diff(xyz, axis=0, prepend=xyz[:1])
        xyz_delta_mag = np.linalg.norm(delta, axis=1, keepdims=True).astype(np.float32)
        roll_std = np.empty((CLIP_LENGTH, 4), dtype=np.float32)
        for index in range(CLIP_LENGTH):
            lo, hi = max(0, index - 2), min(CLIP_LENGTH, index + 3)
            roll_std[index] = normalized[lo:hi, [0, 1, 2, 4]].std(axis=0)
        range_centered = (normalized[:, 4:5] - normalized[:, 4:5].mean(axis=0, keepdims=True)).astype(np.float32)
        tensor = np.concatenate((normalized, xyz_delta_mag, roll_std, range_centered), axis=1).astype(np.float32)
        if tensor.shape != (CLIP_LENGTH, 13) or not np.isfinite(tensor).all():
            raise RuntimeError("Host preprocessing did not produce a finite [60,13] tensor")
        return tensor[np.newaxis, :, :]

    def ingest(self, record: LiveRadarRecord) -> InferenceUpdate:
        invalid_reason = self._validate(record)
        if invalid_reason:
            return self._invalid(*invalid_reason)
        self._records.append(record)
        self._last_timestamp_s, self._last_frame_id = record.timestamp_s, record.frame_id
        self._last_accepted_record_timestamp_s = record.timestamp_s
        if record.arrival_timestamp_s is not None:
            self.record_frame_lateness(
                expected_tick_s=record.timestamp_s,
                arrival_timestamp_s=record.arrival_timestamp_s,
            )
        self._valid_since_reset += 1
        # Preserve the most recent failure through recovery so an operator can
        # still see why a partially refilled buffer was previously cleared.
        if self._held_missing_tick:
            # This accepted record bridges a known omitted tick. Keep the
            # window disabled until 60 fresh records remove that hole.
            self._held_missing_tick = False
        if self._valid_since_reset < CLIP_LENGTH:
            self._state = "disabled" if self._held_missed_ticks else "buffering"
            return InferenceUpdate(self.buffer_progress, self._state, None, None, None, None)
        if self.buffer_progress < CLIP_LENGTH:
            self._state = "buffering"
            return InferenceUpdate(self.buffer_progress, "buffering", None, None, None, None)
        self._held_missed_ticks = 0
        tensor = self.current_tensor()
        if (self._valid_since_reset - CLIP_LENGTH) % STRIDE_FRAMES:
            self._state = "ready"
            return InferenceUpdate(self.buffer_progress, "ready", None, tensor, None, None)
        with torch.no_grad():
            logits = self.binding.model(torch.from_numpy(tensor)).numpy()[0]
        shifted = logits - logits.max()
        probabilities = (np.exp(shifted) / np.exp(shifted).sum()).astype(np.float32)
        update = InferenceUpdate(
            self.buffer_progress,
            "running",
            None,
            tensor,
            CLASS_NAMES[int(np.argmax(probabilities))],
            probabilities,
        )
        self._state = "running"
        self.last_prediction = update
        return update
