from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from radar_pipeline.build_risky_fall_excluded_dataset import FILTER_REASON, build_risky_fall_excluded_dataset
from radar_pipeline.common import load_config


def _session(activity: str, session_id: str, *, n: int = 80) -> pd.DataFrame:
    t = np.arange(n, dtype=float) / 20.0
    return pd.DataFrame(
        {
            "timestamp": t,
            "x": 0.02 * np.sin(t),
            "y": 2.0 + 0.02 * np.cos(t),
            "z": 0.5 + 0.01 * np.sin(t),
            "dop_idx": np.ones(n, dtype=np.int32),
            "range_m": np.full(n, 2.0),
            "azimuth_deg": np.zeros(n),
            "elevation_deg": np.full(n, -10.0),
            "session_id": session_id,
        }
    )


def test_build_risky_dataset_excludes_only_documented_ambiguous_auto_fall_windows(tmp_path: Path) -> None:
    cleaned = tmp_path / "cleaned"
    cleaned.mkdir()
    _session("walking", "walking_a").to_csv(cleaned / "walking_a.csv", index=False)
    _session("standing", "standing_a").to_csv(cleaned / "standing_a.csv", index=False)
    _session("sitting", "sitting_a").to_csv(cleaned / "sitting_a.csv", index=False)
    _session("fall", "falling_a").to_csv(cleaned / "falling_a.csv", index=False)
    annotations = tmp_path / "annotations.csv"
    pd.DataFrame(
        [
            {
                "session_id": "falling_a",
                "activity": "fall",
                "start_state": "standing",
                "event_start_s": 1.0,
                "impact_s": 1.5,
                "event_end_s": 2.5,
                "end_state": "lying",
                "distance_m_inferred": 2.0,
                "distance_band_inferred": "middle",
                "direction_inferred": "forward",
                "confidence": "medium",
                "auto_include_in_training": True,
                "quality_flags": "geometry_edge_warning;weak_height_evidence",
                "method_version": "auto_event_heuristic_v1",
                "method_summary": "test",
            }
        ]
    ).to_csv(annotations, index=False)
    cfg_path = tmp_path / "config.yaml"
    cfg = load_config()
    cfg["windowing"]["window_len_sec"] = 1.0
    cfg["windowing"]["overlap_pct"] = 0.5
    import yaml

    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    dataset_dir = tmp_path / "dataset"
    result = build_risky_fall_excluded_dataset(
        cleaned_dir=cleaned,
        annotations_csv=annotations,
        windowed_output_dir=tmp_path / "windowed",
        dataset_output_dir=dataset_dir,
        config_path=cfg_path,
    )

    removed = pd.read_csv(dataset_dir / "removed_risky_fall_windows.csv")
    assert result["removed_window_count"] == len(removed) > 0
    assert set(removed["exclude_reason"]) == {FILTER_REASON}
    data = np.load(dataset_dir / "radar_dataset.npz", allow_pickle=True)
    labels = np.concatenate([data[f"y_{split}"] for split in ("train", "val", "test")])
    assert 3 not in labels


def test_build_risky_dataset_refuses_existing_outputs(tmp_path: Path) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()
    annotations = tmp_path / "annotations.csv"
    annotations.write_text("session_id\n", encoding="utf-8")
    import pytest

    with pytest.raises(FileExistsError, match="Windowed output directory already exists"):
        build_risky_fall_excluded_dataset(
            cleaned_dir=tmp_path,
            annotations_csv=annotations,
            windowed_output_dir=existing,
            dataset_output_dir=tmp_path / "dataset",
            config_path=tmp_path / "missing.yaml",
        )
