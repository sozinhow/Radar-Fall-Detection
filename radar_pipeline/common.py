from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = Path(__file__).resolve().parent
CLASS_NAMES = ["walking", "standing", "sitting", "fall"]
CLASS_LABELS = list(range(len(CLASS_NAMES)))
MODEL_FEATURES = ["x", "y", "z", "dop_idx", "range_m", "azimuth_deg", "elevation_deg"]
DISTANCE_BANDS = {"near", "middle", "far"}


COLUMN_ALIASES = {
    "timestamp": ("timestamp", "time", "datetime", "date", "frame_time", "seconds", "sec", "t"),
    "range_m": ("range", "range_m", "distance", "distance_m", "r"),
    "velocity_mps": ("velocity", "velocity_mps", "speed", "v_mps"),
    "dop_idx": ("dop_idx", "doppler_idx"),
    "azimuth_deg": ("azimuth", "angle", "angle_deg", "azimuth_deg", "theta", "bearing"),
    "elevation_deg": ("elevation", "elevation_deg", "phi", "pitch"),
    "energy": ("energy", "snr", "power", "magnitude", "amplitude", "signal", "intensity"),
}


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else PIPELINE_DIR / "config.yaml"
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dirs() -> None:
    for rel in (
        "data/raw_csv",
        "outputs/validation/before_cleaning",
        "outputs/validation/after_cleaning",
    ):
        (ROOT / rel).mkdir(parents=True, exist_ok=True)


def canonical_activity_from_name(name: str) -> str:
    stem = Path(name).stem.lower()
    first_token = re.split(r"[_\W]+", stem, maxsplit=1)[0]
    if first_token == "falling" or first_token == "fall":
        return "fall"
    if first_token in {"walking", "standing", "sitting"}:
        return first_token
    for activity in CLASS_NAMES:
        if stem == activity or stem.startswith(f"{activity}_"):
            return activity
    if stem.startswith("falling_"):
        return "fall"
    raise ValueError(f"Cannot infer canonical activity from filename prefix: {name}")


def distance_band_from_path(path: str | Path) -> str:
    parent = Path(path).parent.name.lower()
    tokens = parent.split("_")
    for token in tokens:
        if token in DISTANCE_BANDS:
            return token
    stem_tokens = Path(path).stem.lower().split("_")
    for token in stem_tokens:
        if token in DISTANCE_BANDS:
            return token
    return ""


def activity_from_path(path: str | Path) -> str:
    parsed = Path(path)
    parent = parsed.parent.name.lower()
    if parent in CLASS_NAMES or parent == "falling":
        return "fall" if parent == "falling" else parent
    for activity in CLASS_NAMES:
        if parent == activity or parent.startswith(f"{activity}_"):
            return activity
    return canonical_activity_from_name(parsed.name)


def discover_csv_files(input_dir: str | Path) -> list[Path]:
    root = Path(input_dir)
    paths = sorted(p for p in root.rglob("*.csv") if p.is_file())
    return [p for p in paths if not p.name.endswith("conversion_log.json") and not p.name.endswith("_log.csv")]


def session_id_from_df_or_path(df: pd.DataFrame, path: str | Path, activity: str | None = None) -> str:
    if "session_id" in df.columns and len(df):
        value = str(df["session_id"].iloc[0])
        if "__" in value:
            return value.split("__", 1)[1]
        return value
    stem = Path(path).stem
    if activity and stem.startswith(f"{activity}_"):
        return stem
    return stem


def normalize_column_name(name: Any) -> str:
    text = str(name).strip().lower()
    text = re.sub(r"\([^)]*\)", "", text)
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text


def canonicalize_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    original_cols = list(df.columns)
    normalized = {col: normalize_column_name(col) for col in original_cols}
    assigned: dict[str, str] = {}
    rename: dict[str, str] = {}

    for canonical, aliases in COLUMN_ALIASES.items():
        for col, norm in normalized.items():
            if canonical in assigned:
                break
            if norm == canonical or norm in aliases or any(alias in norm for alias in aliases if len(alias) > 2):
                assigned[canonical] = col
                rename[col] = canonical

    result = df.rename(columns=rename).copy()
    non_numeric_metadata = {"timestamp", "activity", "session_id", "recording_id", "source_file", "split"}
    for col in result.columns:
        if col not in non_numeric_metadata:
            result[col] = pd.to_numeric(result[col], errors="coerce")

    lower_cols = {normalize_column_name(c): c for c in result.columns}
    if "range_m" not in result.columns and {"x", "y", "z"}.issubset(lower_cols):
        x, y, z = lower_cols["x"], lower_cols["y"], lower_cols["z"]
        result["range_m"] = np.sqrt(result[x] ** 2 + result[y] ** 2 + result[z] ** 2)
        assigned["range_m"] = "derived from x/y/z"
    if "azimuth_deg" not in result.columns and {"x", "y"}.issubset(lower_cols):
        x, y = lower_cols["x"], lower_cols["y"]
        result["azimuth_deg"] = np.rad2deg(np.arctan2(result[x], result[y]))
        assigned["azimuth_deg"] = "derived from x/y"
    if "elevation_deg" not in result.columns and {"x", "y", "z"}.issubset(lower_cols):
        x, y, z = lower_cols["x"], lower_cols["y"], lower_cols["z"]
        horizontal = np.sqrt(result[x] ** 2 + result[y] ** 2)
        result["elevation_deg"] = np.rad2deg(np.arctan2(result[z], horizontal))
        assigned["elevation_deg"] = "derived from x/y/z"

    return result, assigned


def feature_columns(df: pd.DataFrame) -> list[str]:
    ignored = {"timestamp", "activity", "label", "frame", "cluster_id", "split", "session_id", "recording_id", "source_file"}
    return [c for c in df.columns if c not in ignored and pd.api.types.is_numeric_dtype(df[c])]


def model_feature_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in MODEL_FEATURES if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]


def elapsed_axis(df: pd.DataFrame, fallback_rate_hz: float | None = None) -> tuple[np.ndarray, str]:
    if len(df) == 0:
        return np.asarray([], dtype=float), "Frame"
    if "timestamp" in df.columns:
        numeric = pd.to_numeric(df["timestamp"], errors="coerce")
        if numeric.notna().sum() >= 2:
            values = numeric.to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            if len(finite) >= 2:
                diffs = np.diff(finite)
                diffs = diffs[np.isfinite(diffs) & (diffs != 0)]
                if len(diffs):
                    if float(np.nanmedian(np.abs(diffs))) > 10:
                        values = values / 1000.0
                    return values - np.nanmin(values), "Elapsed time (s)"
        dt = pd.to_datetime(df["timestamp"], errors="coerce")
        if dt.notna().sum() >= 2:
            first = dt.dropna().iloc[0]
            seconds = (dt - first).dt.total_seconds()
            fallback = pd.Series(np.arange(len(df), dtype=float), index=df.index)
            return seconds.fillna(fallback).to_numpy(dtype=float), "Elapsed time (s)"
    if fallback_rate_hz and fallback_rate_hz > 0:
        return np.arange(len(df), dtype=float) / float(fallback_rate_hz), "Elapsed time (s)"
    return np.arange(len(df), dtype=float), "Frame"


def read_csv_canonical(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df, _ = canonicalize_columns(df)
    return df


def estimate_sampling_rate_hz(df: pd.DataFrame) -> float | None:
    if "timestamp" not in df.columns:
        return None
    ts = df["timestamp"]
    numeric = pd.to_numeric(ts, errors="coerce")
    if numeric.notna().sum() >= 3:
        values = numeric.dropna().to_numpy(dtype=float)
        diffs = np.diff(values)
    else:
        dt = pd.to_datetime(ts, errors="coerce")
        if dt.notna().sum() < 3:
            return None
        diffs = dt.dropna().diff().dt.total_seconds().dropna().to_numpy(dtype=float)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if len(diffs) == 0:
        return None
    # Treat millisecond timestamps as ms if the median step is too large.
    median = float(np.median(diffs))
    if median > 10:
        median /= 1000.0
    return 1.0 / median if median > 0 else None


def write_json(path: str | Path, payload: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def load_json(path: str | Path, default: Any = None) -> Any:
    p = Path(path)
    if not p.exists():
        return default
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def estimated_height_m(range_m: pd.Series, elevation_deg: pd.Series, mount_height_m: float) -> pd.Series:
    elevation_rad = np.deg2rad(pd.to_numeric(elevation_deg, errors="coerce"))
    # Positive elevation points upward; radar is tilted downward in config, so detections below mount reduce height.
    return mount_height_m + pd.to_numeric(range_m, errors="coerce") * np.sin(elevation_rad)


def finite_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    return float(value) if math.isfinite(float(value)) else None
