from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.signal import medfilt, savgol_filter

from .common import feature_columns


@dataclass
class StageLog:
    rows: dict[str, int] = field(default_factory=dict)
    metrics: dict[str, int | float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def mark(self, stage: str, df: pd.DataFrame) -> None:
        self.rows[stage] = int(len(df))


def remove_static_clutter(df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    out = df.copy()
    cols = cols or [c for c in feature_columns(out) if c in {"energy", "snr", "power", "magnitude", "amplitude"}]
    for col in cols:
        out[col] = out[col] - out[col].mean(skipna=True)
    return out


def interpolate_short_gaps(df: pd.DataFrame, max_gap: int, cols: list[str] | None = None) -> pd.DataFrame:
    out = df.copy()
    cols = cols or feature_columns(out)
    for col in cols:
        out[col] = out[col].interpolate(method="linear", limit=max_gap, limit_direction="both")
    return out


def drop_long_missing_segments(df: pd.DataFrame, max_gap: int, cols: list[str] | None = None) -> pd.DataFrame:
    cols = cols or feature_columns(df)
    if not cols:
        return df.copy()
    invalid = df[cols].isna().any(axis=1)
    if not invalid.any():
        return df.copy()
    groups = invalid.ne(invalid.shift(fill_value=False)).cumsum()
    long_invalid = invalid & invalid.groupby(groups).transform("sum").gt(max_gap)
    return df.loc[~long_invalid].reset_index(drop=True)


def drop_frozen_duplicate_frames(df: pd.DataFrame, min_run: int = 3, cols: list[str] | None = None) -> pd.DataFrame:
    out = df.copy()
    cols = cols or feature_columns(out)
    if min_run < 2 or len(out) < min_run or not cols:
        return out
    same_as_previous = out[cols].eq(out[cols].shift()).all(axis=1)
    run_id = (~same_as_previous).cumsum()
    run_lengths = run_id.map(run_id.value_counts())
    keep = ~(same_as_previous & run_lengths.ge(min_run))
    return out.loc[keep].reset_index(drop=True)


def count_frozen_duplicate_frames(df: pd.DataFrame, min_run: int = 3, cols: list[str] | None = None) -> int:
    cols = cols or [c for c in ("x", "y", "z", "dop_idx", "velocity_mps") if c in df.columns]
    if min_run < 2 or len(df) < min_run or not cols:
        return 0
    same_as_previous = df[cols].eq(df[cols].shift()).all(axis=1)
    run_id = (~same_as_previous).cumsum()
    run_lengths = run_id.map(run_id.value_counts())
    return int((same_as_previous & run_lengths.ge(min_run)).sum())


def apply_range_velocity_filters(
    df: pd.DataFrame,
    mount_height_m: float,
    max_range_m: float,
    velocity_max_mps: float,
) -> pd.DataFrame:
    out = df.copy()
    mask = pd.Series(True, index=out.index)
    if "range_m" in out.columns:
        mask &= out["range_m"].between(0, max_range_m, inclusive="both") | out["range_m"].isna()
    if "velocity_mps" in out.columns:
        mask &= out["velocity_mps"].abs().le(velocity_max_mps) | out["velocity_mps"].isna()
    if {"range_m", "elevation_deg"}.issubset(out.columns):
        height = mount_height_m + out["range_m"] * np.sin(np.deg2rad(out["elevation_deg"]))
        mask &= height.between(0, mount_height_m + 0.4, inclusive="both") | height.isna()
    return out.loc[mask].reset_index(drop=True)


def remove_zscore_outliers(df: pd.DataFrame, thresh: float, cols: list[str] | None = None) -> pd.DataFrame:
    out = df.copy()
    cols = cols or feature_columns(out)
    mask = pd.Series(True, index=out.index)
    for col in cols:
        series = out[col]
        std = series.std(skipna=True)
        if not std or np.isnan(std):
            continue
        z = (series - series.mean(skipna=True)).abs() / std
        mask &= z.le(thresh) | z.isna()
    return out.loc[mask].reset_index(drop=True)


def smooth_features(df: pd.DataFrame, window: int, method: str = "median", cols: list[str] | None = None) -> pd.DataFrame:
    out = df.copy()
    cols = cols or feature_columns(out)
    if window < 3:
        return out
    if window % 2 == 0:
        window += 1
    for col in cols:
        values = out[col].to_numpy(dtype=float)
        if np.isfinite(values).sum() < window:
            continue
        filled = pd.Series(values).interpolate(limit_direction="both").to_numpy(dtype=float)
        if method == "savgol" and len(filled) >= window:
            out[col] = savgol_filter(filled, window_length=window, polyorder=min(2, window - 1))
        else:
            out[col] = medfilt(filled, kernel_size=window)
    return out


def angle_elevation_gate(
    df: pd.DataFrame,
    azimuth_beamwidth_deg: float,
    elevation_beamwidth_deg: float,
    tilt_deg: float = 0.0,
    margin_deg: float = 5.0,
) -> pd.DataFrame:
    out = df.copy()
    mask = pd.Series(True, index=out.index)
    if "azimuth_deg" in out.columns:
        mask &= out["azimuth_deg"].abs().le(azimuth_beamwidth_deg / 2.0) | out["azimuth_deg"].isna()
    if "elevation_deg" in out.columns:
        center = -float(tilt_deg)
        half_width = elevation_beamwidth_deg / 2.0 + float(margin_deg)
        mask &= out["elevation_deg"].between(center - half_width, center + half_width, inclusive="both") | out["elevation_deg"].isna()
    out = out.loc[mask].copy()
    subset = [c for c in ("timestamp", "range_m", "velocity_mps", "azimuth_deg", "elevation_deg") if c in out.columns]
    if subset:
        out = out.drop_duplicates(subset=subset)
    return out.reset_index(drop=True)


def trim_idle_segments(
    df: pd.DataFrame,
    thresh: float,
    cols: list[str] | None = None,
    min_active_run: int = 5,
    edge_padding: int = 2,
    min_active_fraction: float = 0.15,
    min_keep_fraction: float = 0.5,
) -> pd.DataFrame:
    out = df.copy()
    cols = cols or feature_columns(out)
    if len(out) < 3 or not cols:
        return out
    delta = out[cols].diff().abs().mean(axis=1).fillna(0)
    max_delta = delta.max()
    if max_delta and np.isfinite(max_delta):
        score = delta / max_delta
    else:
        return out
    active = score.gt(thresh)
    if not active.any():
        return out
    if float(active.mean()) < min_active_fraction:
        return out

    # Only use sustained activity to trim low-motion edges. A single transient jump
    # inside a mostly still sitting/standing recording is valid signal, not an event
    # boundary, and must not collapse the session down to one frame.
    sustained = active.rolling(min_active_run, min_periods=min_active_run).sum().eq(min_active_run)
    if not sustained.any():
        return out
    start = max(int(sustained.idxmax()) - min_active_run + 1 - edge_padding, 0)
    end = min(int(sustained.iloc[::-1].idxmax()) + edge_padding, len(out) - 1)
    if end <= start:
        return out
    keep_fraction = (end - start + 1) / len(out)
    if keep_fraction < min_keep_fraction:
        return out
    return out.loc[start:end].reset_index(drop=True)


def fit_normalizer(frames: list[pd.DataFrame], method: str = "zscore") -> dict[str, dict[str, float]]:
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    params: dict[str, dict[str, float]] = {}
    for col in feature_columns(combined):
        s = combined[col]
        if method == "minmax":
            params[col] = {"min": float(s.min(skipna=True)), "max": float(s.max(skipna=True))}
        else:
            std = float(s.std(skipna=True) or 1.0)
            params[col] = {"mean": float(s.mean(skipna=True)), "std": std if std > 0 else 1.0}
    return params


def apply_normalizer(df: pd.DataFrame, params: dict[str, dict[str, float]], method: str = "zscore") -> pd.DataFrame:
    out = df.copy()
    for col, p in params.items():
        if col not in out.columns:
            continue
        if method == "minmax":
            denom = p["max"] - p["min"] or 1.0
            out[col] = (out[col] - p["min"]) / denom
        else:
            out[col] = (out[col] - p["mean"]) / (p["std"] or 1.0)
    return out


def clean_frame(df: pd.DataFrame, cfg: dict, normalize: bool = False, normalizer: dict | None = None) -> tuple[pd.DataFrame, StageLog]:
    log = StageLog()
    radar = cfg["radar"]
    cleaning = cfg["cleaning"]
    out = df.copy()
    log.mark("raw", out)
    out = remove_static_clutter(out)
    log.mark("static_clutter_removed", out)
    log.metrics["frozen_duplicate_frames_detected"] = count_frozen_duplicate_frames(
        out,
        int(cleaning.get("frozen_duplicate_min_run", 3)),
    )
    out = interpolate_short_gaps(out, int(cleaning["max_gap_frames_interp"]))
    out = drop_long_missing_segments(out, int(cleaning["max_gap_frames_interp"]))
    log.mark("missing_handled", out)
    out = apply_range_velocity_filters(
        out,
        float(radar["mount_height_m"]),
        float(radar["max_range_m"]),
        float(cleaning["velocity_max_mps"]),
    )
    log.mark("range_velocity_filtered", out)
    out = remove_zscore_outliers(out, float(cleaning["outlier_zscore_thresh"]))
    log.mark("zscore_filtered", out)
    out = smooth_features(out, int(cleaning["smoothing_window"]))
    log.mark("smoothed", out)
    out = angle_elevation_gate(
        out,
        float(radar["azimuth_beamwidth_deg"]),
        float(radar["elevation_beamwidth_deg"]),
        float(radar.get("tilt_deg", 0.0)),
        float(cleaning.get("angle_gate_margin_deg", 5.0)),
    )
    log.mark("angle_gated", out)
    activity = str(out["activity"].iloc[0]).lower() if "activity" in out.columns and len(out) else ""
    static_activity = activity in {"sitting", "standing"}
    if static_activity and not bool(cleaning.get("idle_trim_static_activities", False)):
        pass
    else:
        out = trim_idle_segments(
            out,
            float(cleaning["idle_entropy_thresh"]),
            min_active_run=int(cleaning.get("idle_min_active_run", 5)),
            edge_padding=int(cleaning.get("idle_edge_padding", 2)),
            min_active_fraction=float(cleaning.get("idle_min_active_fraction", 0.15)),
            min_keep_fraction=float(cleaning.get("idle_min_keep_fraction", 0.5)),
        )
    log.mark("idle_trimmed", out)
    out = out.dropna(subset=feature_columns(out)).reset_index(drop=True)
    log.mark("final_non_null", out)
    if normalize and normalizer:
        out = apply_normalizer(out, normalizer, cleaning.get("normalization", "zscore"))
        log.mark("normalized", out)
    return out, log
