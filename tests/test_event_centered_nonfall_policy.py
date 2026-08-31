from __future__ import annotations

import numpy as np

import pytest

from radar_pipeline.run_event_centered_clips import parse_clip_lengths, select_steady_state_nonfall_start


def test_steady_state_selector_avoids_an_edge_transition_for_sitting() -> None:
    values = np.zeros((100, 7), dtype=np.float32)
    # A large transition at the start should not be part of the selected
    # guarded 60-frame candidate when an interior candidate exists.
    values[:15, 0] = np.linspace(0.0, 3.0, 15, dtype=np.float32)
    values[15:, 0] = 3.0

    start, details = select_steady_state_nonfall_start(
        session_id="sitting_example",
        activity="sitting",
        values=values,
        length=60,
        stride_frames=15,
        seed=42,
    )

    assert start == 15
    assert details["nonfall_interior_candidate_count"] == 1
    assert details["nonfall_selection_reason"] == "lowest_nonfall_motion_interior"


def test_steady_state_selector_is_deterministic_and_keeps_short_session_padding() -> None:
    values = np.zeros((40, 7), dtype=np.float32)
    first = select_steady_state_nonfall_start(
        session_id="walking_example",
        activity="walking",
        values=values,
        length=60,
        stride_frames=15,
        seed=42,
    )
    second = select_steady_state_nonfall_start(
        session_id="walking_example",
        activity="walking",
        values=values,
        length=60,
        stride_frames=15,
        seed=42,
    )

    assert first == second
    assert first[0] == -10
    assert first[1]["nonfall_selection_reason"] == "short_session_edge_padding"


def test_clip_length_ablation_accepts_only_versioned_lengths() -> None:
    assert parse_clip_lengths("60,100") == (60, 100)
    with pytest.raises(ValueError, match="Unsupported clip lengths"):
        parse_clip_lengths("60,90")
