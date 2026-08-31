from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from radar_pipeline.live_inference import (
    BASE_FEATURES,
    CLASS_NAMES,
    CausalHostInference,
    FRESH_FRAME_HANDOFF_GRACE_S,
    LiveRadarRecord,
    MISSING_SUBREASON_LATE_FRAME,
    MISSING_SUBREASON_NO_FRESH_FRAME,
    MISSING_SUBREASON_OTHER,
    MISSING_SUBREASON_QUEUE_EMPTY,
    MISSING_SUBREASON_TRACKING_INACTIVE,
    RESET_REASON_DUPLICATE_FRAME_ID,
    RESET_REASON_INVALID_VALUE,
    RESET_REASON_MISSING_RECORD,
    RESET_REASON_OTHER,
    RESET_REASON_TIMESTAMP_GAP,
    load_demo_binding,
    sha256_file,
    should_wait_for_fresh_frame,
)


RADAR_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = RADAR_ROOT / "outputs/deployment/host_demo_20260728_run01/host_demo_manifest.json"
GOLDEN = RADAR_ROOT / "outputs/deployment/host_demo_20260728_run01/golden_corpus/real_walking_60_reference_v2.npz"
GUI_TOOLS = RADAR_ROOT / "STM32/n6570dk_blink/tools"
if str(GUI_TOOLS) not in sys.path:
    sys.path.insert(0, str(GUI_TOOLS))

from radar_collection_gui import RadarTarget, format_raw_target_diagnostic  # noqa: E402


def _engine() -> CausalHostInference:
    return CausalHostInference(load_demo_binding(MANIFEST))


def _record(row: np.ndarray, index: int) -> LiveRadarRecord:
    return LiveRadarRecord(
        timestamp_s=index / 20.0,
        frame_id=index + 1,
        x=float(row[0]), y=float(row[1]), z=float(row[2]), dop_idx=int(row[3]),
        range_m=float(row[4]), azimuth_deg=float(row[5]), elevation_deg=float(row[6]),
    )


def test_selected_pair_is_hash_pinned_and_loadable() -> None:
    binding = load_demo_binding(MANIFEST)
    assert sha256_file(binding.checkpoint_path) == "5c9fd69a3e1de77ae46950d34322bc2df56043f93834ad5e43c66024c4baea4b"
    assert sha256_file(binding.normalization_path) == "0b66e0eae7aab2ff0a268af856504022980ee8d73b00cf85ef67461b62ababd3"
    assert binding.mean.shape == binding.std.shape == (len(BASE_FEATURES),)


def test_real_record_golden_corpus_preprocessing_and_model_parity() -> None:
    corpus = np.load(GOLDEN, allow_pickle=False)
    engine = _engine()
    update = None
    for index, row in enumerate(corpus["base_records"]):
        update = engine.ingest(_record(row, index))
    assert update is not None
    assert update.tensor is not None and update.tensor.dtype == np.float32
    assert update.tensor.shape == (1, 60, 13)
    np.testing.assert_allclose(update.tensor, corpus["expected_tensor"], rtol=0, atol=1e-6)
    np.testing.assert_allclose(update.probabilities, corpus["expected_probabilities"], rtol=0, atol=1e-6)
    assert update.predicted_class == str(corpus["expected_class"])
    assert update.predicted_class in CLASS_NAMES


def test_first_target_geometry_and_causal_stride() -> None:
    first = LiveRadarRecord.from_first_target(timestamp_s=0.0, frame_id=1, x=3.0, y=4.0, z=12.0, dop_idx=-2)
    assert first.range_m == 13.0
    engine = _engine()
    updates = []
    for index in range(75):
        updates.append(engine.ingest(LiveRadarRecord.from_first_target(
            timestamp_s=index / 20.0, frame_id=index + 1, x=1.0 + index / 1000, y=2.0, z=-0.5, dop_idx=-2,
        )))
    assert updates[58].buffer_progress == 59
    assert updates[59].predicted_class is not None
    assert updates[60].predicted_class is None
    assert updates[74].predicted_class is not None


def test_gui_raw_target_diagnostic_uses_protocol_coordinates_for_host_distance() -> None:
    """0x0A04 has no distance field: the GUI must display sqrt(x²+y²+z²)."""
    target = RadarTarget(target_num=2, cluster_id=17, x=3.0, y=4.0, z=12.0, dop_idx=-2)

    assert target.distance == 13.0
    diagnostic = format_raw_target_diagnostic(target, age_s=0.025)
    assert "targetnum=2" in diagnostic
    assert "cluster_id=17" in diagnostic
    assert "x=3.000 m" in diagnostic
    assert "y=4.000 m" in diagnostic
    assert "z=12.000 m" in diagnostic
    assert "host distance=sqrt(x²+y²+z²)=13.000 m" in diagnostic
    assert "dopidx=-2" in diagnostic


def _fresh_record(index: int, *, frame_id: int | None = None, timestamp_s: float | None = None) -> LiveRadarRecord:
    return LiveRadarRecord.from_first_target(
        timestamp_s=index / 20.0 if timestamp_s is None else timestamp_s,
        frame_id=index + 1 if frame_id is None else frame_id,
        x=1.0, y=2.0, z=0.0, dop_idx=1,
    )


def _seed_buffer(engine: CausalHostInference, count: int = 4) -> None:
    for index in range(count):
        engine.ingest(_fresh_record(index))
    assert engine.buffer_progress == count


def test_duplicate_frame_id_reset_is_cleared_and_counted() -> None:
    engine = _engine()
    _seed_buffer(engine)
    duplicate = engine.ingest(_fresh_record(4, frame_id=4))
    assert duplicate.state == "buffering"
    assert duplicate.reset_reason == RESET_REASON_DUPLICATE_FRAME_ID
    assert duplicate.buffer_progress == 0
    snapshot = engine.diagnostic_snapshot()
    assert snapshot.last_reset_reason == RESET_REASON_DUPLICATE_FRAME_ID
    assert snapshot.last_reset_detail == "duplicate_frame_id"
    assert snapshot.reset_counts[RESET_REASON_DUPLICATE_FRAME_ID] == 1


def test_missing_record_reset_is_cleared_and_counted() -> None:
    engine = _engine()
    _seed_buffer(engine)
    missing = engine.reset(RESET_REASON_MISSING_RECORD, detail="missing_scheduled_record")
    assert missing.buffer_progress == 0
    assert missing.reset_reason == RESET_REASON_MISSING_RECORD
    snapshot = engine.diagnostic_snapshot()
    assert snapshot.last_reset_detail == "missing_scheduled_record"
    assert snapshot.reset_counts[RESET_REASON_MISSING_RECORD] == 1
    engine.ingest(_fresh_record(0))
    assert engine.diagnostic_snapshot().last_reset_reason == RESET_REASON_MISSING_RECORD


def test_invalid_value_reset_is_cleared_and_counted() -> None:
    engine = _engine()
    _seed_buffer(engine)
    invalid = engine.ingest(LiveRadarRecord.from_first_target(
        timestamp_s=4 / 20.0, frame_id=5, x=float("nan"), y=2.0, z=0.0, dop_idx=1,
    ))
    assert invalid.buffer_progress == 0
    assert invalid.reset_reason == RESET_REASON_INVALID_VALUE
    snapshot = engine.diagnostic_snapshot()
    assert snapshot.last_reset_detail == "non_finite_record"
    assert snapshot.reset_counts[RESET_REASON_INVALID_VALUE] == 1


def test_timestamp_gap_reset_is_cleared_and_counted() -> None:
    engine = _engine()
    _seed_buffer(engine)
    timestamp_gap = engine.ingest(_fresh_record(4, timestamp_s=0.50))
    assert timestamp_gap.buffer_progress == 0
    assert timestamp_gap.reset_reason == RESET_REASON_TIMESTAMP_GAP
    snapshot = engine.diagnostic_snapshot()
    assert snapshot.last_reset_detail == "timestamp_discontinuity"
    assert snapshot.reset_counts[RESET_REASON_TIMESTAMP_GAP] == 1


def test_other_reset_tracks_lifecycle_detail_and_all_counters_are_exposed() -> None:
    engine = _engine()
    lifecycle = engine.reset("tracking_lost")
    assert lifecycle.reset_reason == RESET_REASON_OTHER
    snapshot = engine.diagnostic_snapshot()
    assert snapshot.last_reset_detail == "tracking_lost"
    assert snapshot.reset_counts[RESET_REASON_OTHER] == 1
    assert set(snapshot.reset_counts) == {
        RESET_REASON_MISSING_RECORD,
        RESET_REASON_DUPLICATE_FRAME_ID,
        RESET_REASON_INVALID_VALUE,
        RESET_REASON_TIMESTAMP_GAP,
        RESET_REASON_OTHER,
    }


def test_missing_subreasons_are_recorded_under_the_high_level_reset_category() -> None:
    engine = _engine()
    subreasons = (
        MISSING_SUBREASON_NO_FRESH_FRAME,
        MISSING_SUBREASON_QUEUE_EMPTY,
        MISSING_SUBREASON_LATE_FRAME,
        MISSING_SUBREASON_TRACKING_INACTIVE,
        MISSING_SUBREASON_OTHER,
    )
    for index, subreason in enumerate(subreasons):
        _seed_buffer(engine, count=2)
        update = engine.mark_missing_record(
            subreason,
            expected_tick_s=0.10 + index / 20.0,
            queue_depth=index,
            allow_diagnostic_hold=False,
        )
        assert update.buffer_progress == 0
        assert update.reset_reason == RESET_REASON_MISSING_RECORD
        snapshot = engine.diagnostic_snapshot()
        assert snapshot.last_missing_subreason == subreason
        assert snapshot.missing_subreason_counts[subreason] == 1
    snapshot = engine.diagnostic_snapshot()
    assert snapshot.reset_counts[RESET_REASON_MISSING_RECORD] == len(subreasons)


def test_timing_and_queue_snapshot_tracks_scheduled_and_actual_arrival_times() -> None:
    engine = _engine()
    record = LiveRadarRecord.from_first_target(
        timestamp_s=10.0, arrival_timestamp_s=10.012, frame_id=1, x=1.0, y=2.0, z=0.0, dop_idx=1,
    )
    engine.ingest(record)
    engine.update_transport_diagnostics(expected_next_tick_s=10.05, queue_depth=3)
    snapshot = engine.diagnostic_snapshot()
    assert snapshot.last_accepted_record_timestamp_s == 10.0
    assert snapshot.last_record_arrival_s == 10.012
    assert np.isclose(snapshot.last_arrival_jitter_s, 0.012)
    assert snapshot.expected_next_tick_s == 10.05
    assert snapshot.queue_depth == snapshot.max_queue_depth == 3
    assert snapshot.positive_lateness_samples == 1
    assert np.isclose(snapshot.mean_positive_lateness_s, 0.012)
    assert np.isclose(snapshot.max_positive_lateness_s, 0.012)
    assert "queue_max=3" in engine.format_session_stats()
    assert "positive_lateness_max=12.0ms" in engine.format_session_stats()
    engine.record_frame_lateness(expected_tick_s=11.0, arrival_timestamp_s=11.030)
    late_snapshot = engine.diagnostic_snapshot()
    assert late_snapshot.positive_lateness_samples == 2
    assert np.isclose(late_snapshot.mean_positive_lateness_s, 0.021)
    assert np.isclose(late_snapshot.max_positive_lateness_s, 0.030)


def test_bounded_handoff_grace_prevents_premature_queue_empty_reset() -> None:
    expected_tick = 10.0
    # The target may be enqueued in the next GUI poll without appending a
    # fabricated record. It becomes a true missing record after this bound.
    assert should_wait_for_fresh_frame(
        now_s=expected_tick + FRESH_FRAME_HANDOFF_GRACE_S - 0.001,
        expected_tick_s=expected_tick,
        queue_depth=0,
    )
    assert not should_wait_for_fresh_frame(
        now_s=expected_tick + FRESH_FRAME_HANDOFF_GRACE_S,
        expected_tick_s=expected_tick,
        queue_depth=0,
    )
    assert not should_wait_for_fresh_frame(
        now_s=expected_tick,
        expected_tick_s=expected_tick,
        queue_depth=1,
    )


def test_diagnostic_tolerance_holds_one_missing_tick_without_invalid_inference() -> None:
    engine = CausalHostInference(load_demo_binding(MANIFEST), diagnostic_tolerance_mode=True)
    _seed_buffer(engine)
    held = engine.mark_missing_record(
        MISSING_SUBREASON_QUEUE_EMPTY,
        expected_tick_s=4 / 20.0,
        queue_depth=0,
    )
    assert held.state == "disabled"
    assert held.tensor is None and held.probabilities is None
    assert held.buffer_progress == 4
    snapshot = engine.diagnostic_snapshot()
    assert snapshot.reset_counts[RESET_REASON_MISSING_RECORD] == 0
    assert snapshot.held_missed_ticks == 1

    # The following 59 fresh records cannot produce a valid window: the held
    # tick remains inside the rolling history until sixty new records arrive.
    for index in range(5, 64):
        update = engine.ingest(_fresh_record(index))
        assert update.tensor is None and update.probabilities is None
    assert engine.buffer_progress == 60
    assert engine.state == "disabled"

    recovered = engine.ingest(_fresh_record(64))
    assert recovered.tensor is not None
    assert recovered.probabilities is not None
    assert recovered.state == "running"


def test_second_missing_tick_after_a_diagnostic_hold_fails_closed() -> None:
    engine = CausalHostInference(load_demo_binding(MANIFEST), diagnostic_tolerance_mode=True)
    _seed_buffer(engine)
    engine.mark_missing_record(MISSING_SUBREASON_QUEUE_EMPTY, expected_tick_s=4 / 20.0, queue_depth=0)
    engine.ingest(_fresh_record(5))
    second_missing = engine.mark_missing_record(MISSING_SUBREASON_LATE_FRAME, expected_tick_s=6 / 20.0, queue_depth=1)
    assert second_missing.buffer_progress == 0
    assert second_missing.reset_reason == RESET_REASON_MISSING_RECORD
    snapshot = engine.diagnostic_snapshot()
    assert snapshot.last_missing_subreason == MISSING_SUBREASON_LATE_FRAME
    assert snapshot.reset_counts[RESET_REASON_MISSING_RECORD] == 1
