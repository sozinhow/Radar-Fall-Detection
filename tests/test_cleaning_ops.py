from __future__ import annotations

import numpy as np
import pandas as pd

from radar_pipeline.cleaning_ops import (
    apply_range_velocity_filters,
    drop_long_missing_segments,
    interpolate_short_gaps,
    remove_zscore_outliers,
)


def test_interpolate_short_nan_gap() -> None:
    df = pd.DataFrame({"range_m": [1.0, np.nan, 3.0], "velocity_mps": [0.0, 0.1, 0.2]})
    out = interpolate_short_gaps(df, max_gap=1)
    assert out.loc[1, "range_m"] == 2.0


def test_drop_long_missing_segment() -> None:
    df = pd.DataFrame({"range_m": [1.0, np.nan, np.nan, np.nan, 2.0], "velocity_mps": [0, 1, 1, 1, 0]})
    out = drop_long_missing_segments(df, max_gap=2)
    assert len(out) == 2
    assert out["range_m"].tolist() == [1.0, 2.0]


def test_velocity_and_range_filter_remove_invalid_rows() -> None:
    df = pd.DataFrame(
        {
            "range_m": [1.0, 9.0, 1.0],
            "velocity_mps": [0.5, 0.5, 5.0],
            "elevation_deg": [-10.0, -10.0, -10.0],
        }
    )
    out = apply_range_velocity_filters(df, mount_height_m=1.8, max_range_m=5.0, velocity_max_mps=3.0)
    assert len(out) == 1
    assert out.loc[0, "range_m"] == 1.0


def test_zscore_outlier_removed() -> None:
    df = pd.DataFrame({"range_m": [1.0, 1.1, 1.2, 1.1, 20.0], "velocity_mps": [0, 0, 0, 0, 0]})
    out = remove_zscore_outliers(df, thresh=1.5)
    assert len(out) == 4
    assert 20.0 not in out["range_m"].tolist()

