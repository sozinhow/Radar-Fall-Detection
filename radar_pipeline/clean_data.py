from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from radar_pipeline.cleaning_ops import clean_frame
from radar_pipeline.common import (
    ROOT,
    activity_from_path,
    canonical_activity_from_name,
    discover_csv_files,
    distance_band_from_path,
    ensure_dirs,
    feature_columns,
    load_config,
    read_csv_canonical,
    write_json,
)


def _ensure_session_columns(df: pd.DataFrame, activity: str, source_name: str) -> pd.DataFrame:
    out = df.copy()
    fallback = f"{activity}__{Path(source_name).stem.lower()}"
    if "session_id" not in out.columns:
        out["session_id"] = fallback
    out["session_id"] = out["session_id"].fillna(fallback).astype(str)
    if "recording_id" not in out.columns:
        out["recording_id"] = out["session_id"]
    out["recording_id"] = out["recording_id"].fillna(out["session_id"]).astype(str)
    if "source_file" not in out.columns:
        out["source_file"] = source_name
    out["source_file"] = out["source_file"].fillna(source_name).astype(str)
    return out


def _clear_derived_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for child in output_dir.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        elif child.suffix.lower() == ".csv":
            child.unlink()


def _preflight_flat_destinations(items: list[tuple[str, Path, pd.DataFrame]], output_dir: Path) -> None:
    destinations: dict[Path, list[Path]] = {}
    invalid: list[str] = []
    for activity, path, df in items:
        try:
            file_activity = canonical_activity_from_name(path.name)
        except ValueError as exc:
            invalid.append(str(exc))
            continue
        if file_activity != activity:
            invalid.append(f"{path}: filename resolves to {file_activity}, path resolves to {activity}")
        for session_id, _ in df.groupby("session_id", sort=False):
            stem = str(session_id)
            if "__" in stem:
                stem = stem.split("__", 1)[1]
            destination = output_dir / f"{Path(stem).stem}.csv"
            destinations.setdefault(destination, []).append(path)
    collisions = {dest: sources for dest, sources in destinations.items() if len(set(sources)) > 1}
    if invalid or collisions:
        messages = []
        messages.extend(invalid)
        for dest, sources in sorted(collisions.items()):
            source_text = ", ".join(str(src) for src in sorted(set(sources)))
            messages.append(f"flattened destination collision: {dest} <- {source_text}")
        raise ValueError("Cleaned CSV flattening preflight failed:\n" + "\n".join(messages))


def clean_directory(input_dir: Path, output_dir: Path, cfg: dict) -> dict:
    ensure_dirs()
    output_dir.mkdir(parents=True, exist_ok=True)
    excluded_sessions = {str(k): str(v) for k, v in cfg.get("excluded_sessions", {}).items()}
    paths = discover_csv_files(input_dir)
    raw_frames = []
    for path in paths:
        df = read_csv_canonical(path)
        activity = activity_from_path(path)
        df = _ensure_session_columns(df, activity, path.name)
        if feature_columns(df):
            raw_frames.append((activity, path, df))
    if not raw_frames:
        raise FileNotFoundError(f"No usable CSV files found under {input_dir}")
    _preflight_flat_destinations(raw_frames, output_dir)
    _clear_derived_output_dir(output_dir)

    logs = {
        "activities": {},
        "normalization": "fit on training windows in dataset_builder.py",
        "layout": "flat",
        "output_dir": str(output_dir),
        "warnings": [],
    }
    schema = None
    for activity, path, df in raw_frames:
        activity_log = logs["activities"].setdefault(
            activity,
            {
                "rows": {"raw": 0, "final_non_null": 0},
                "sessions": {},
                "session_count": 0,
                "warnings": [],
            },
        )
        for session_id, session_df in df.groupby("session_id", sort=False):
            session_df = session_df.reset_index(drop=True)
            raw_rows = int(len(session_df))
            cols = feature_columns(session_df)
            missing_rows = int(session_df[cols].isna().any(axis=1).sum()) if cols else 0
            source_rel_path = path.relative_to(input_dir) if path.is_relative_to(input_dir) else path
            distance_band = distance_band_from_path(path)
            output_stem = str(session_id)
            if "__" in output_stem:
                output_stem = output_stem.split("__", 1)[1]
            output_path = output_dir / f"{Path(output_stem).stem}.csv"
            if str(session_id) in excluded_sessions:
                activity_log["sessions"][str(session_id)] = {
                    "raw_rows": raw_rows,
                    "final_rows": 0,
                    "excluded": True,
                    "exclusion_reason": excluded_sessions[str(session_id)],
                    "recommendation": "re-record",
                    "missing_rows_before_interpolation": missing_rows,
                    "missing_dropped_rows": 0,
                    "frozen_duplicate_detected_rows": 0,
                    "frozen_duplicate_dropped_rows": 0,
                    "out_of_range_dropped_rows": 0,
                    "zscore_dropped_rows": 0,
                    "elevation_gated_dropped_rows": 0,
                    "idle_trimmed_dropped_rows": 0,
                    "final_nan_dropped_rows": 0,
                    "non_frozen_dropped_rows": 0,
                    "total_dropped_rows": raw_rows,
                    "non_frozen_drop_fraction": 0.0,
                    "flag_high_non_frozen_drop_fraction": False,
                    "rows_by_stage": {"raw": raw_rows, "excluded": 0},
                    "raw_input": str(path),
                    "source_path": str(path),
                    "source_relative_path": str(source_rel_path),
                    "distance_band": distance_band,
                    "canonical_activity": activity,
                    "final_output": None,
                    "warnings": [f"excluded: {excluded_sessions[str(session_id)]}"],
                }
                activity_log["rows"]["raw"] += raw_rows
                activity_log.setdefault("excluded_sessions", {})[str(session_id)] = excluded_sessions[str(session_id)]
                logs["warnings"].append(f"{activity}/{session_id}: excluded: {excluded_sessions[str(session_id)]}")
                continue
            cleaned_session, stage_log = clean_frame(session_df, cfg, normalize=False)
            final_rows = int(len(cleaned_session))
            missing_dropped = stage_log.rows.get("static_clutter_removed", raw_rows) - stage_log.rows.get("missing_handled", raw_rows)
            out_of_range_dropped = stage_log.rows.get("missing_handled", raw_rows) - stage_log.rows.get("range_velocity_filtered", raw_rows)
            zscore_dropped = stage_log.rows.get("range_velocity_filtered", raw_rows) - stage_log.rows.get("zscore_filtered", raw_rows)
            elevation_gated_dropped = stage_log.rows.get("zscore_filtered", raw_rows) - stage_log.rows.get("angle_gated", raw_rows)
            idle_trimmed_dropped = stage_log.rows.get("angle_gated", raw_rows) - stage_log.rows.get("idle_trimmed", raw_rows)
            final_nan_dropped = stage_log.rows.get("idle_trimmed", raw_rows) - stage_log.rows.get("final_non_null", raw_rows)
            total_dropped = raw_rows - final_rows
            non_frozen_dropped = max(total_dropped, 0)
            non_frozen_drop_fraction = non_frozen_dropped / raw_rows if raw_rows else 0.0
            warning = non_frozen_drop_fraction > 0.20
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cleaned_session.to_csv(output_path, index=False)
            if schema is None:
                schema = feature_columns(cleaned_session)
            elif feature_columns(cleaned_session) != schema:
                logs["warnings"].append(f"{activity}/{session_id} feature schema differs from first session")
            activity_log["sessions"][str(session_id)] = {
                "raw_rows": raw_rows,
                "final_rows": final_rows,
                "missing_rows_before_interpolation": missing_rows,
                "missing_dropped_rows": int(max(missing_dropped, 0)),
                "frozen_duplicate_detected_rows": int(stage_log.metrics.get("frozen_duplicate_frames_detected", 0)),
                "frozen_duplicate_dropped_rows": 0,
                "out_of_range_dropped_rows": int(max(out_of_range_dropped, 0)),
                "zscore_dropped_rows": int(max(zscore_dropped, 0)),
                "elevation_gated_dropped_rows": int(max(elevation_gated_dropped, 0)),
                "idle_trimmed_dropped_rows": int(max(idle_trimmed_dropped, 0)),
                "final_nan_dropped_rows": int(max(final_nan_dropped, 0)),
                "non_frozen_dropped_rows": int(non_frozen_dropped),
                "total_dropped_rows": int(max(total_dropped, 0)),
                "non_frozen_drop_fraction": float(non_frozen_drop_fraction),
                "flag_high_non_frozen_drop_fraction": bool(warning),
                "rows_by_stage": stage_log.rows,
                "raw_input": str(path),
                "source_path": str(path),
                "source_relative_path": str(source_rel_path),
                "distance_band": distance_band,
                "canonical_activity": activity,
                "final_output": str(output_path),
                "warnings": stage_log.warnings,
            }
            activity_log["rows"]["raw"] += raw_rows
            activity_log["rows"]["final_non_null"] += final_rows
            if warning:
                logs["warnings"].append(f"{activity}/{session_id}: non-frozen dropped frame fraction {non_frozen_drop_fraction:.1%} exceeds 20%")

    for activity, activity_log in logs["activities"].items():
        activity_log["session_count"] = len(activity_log["sessions"])
        first_session = next(iter(activity_log["sessions"].values()), {})
        first_output = first_session.get("final_output")
        if first_output:
            activity_log["feature_columns"] = feature_columns(pd.read_csv(first_output))

    write_json(output_dir.parent / "cleaning_log.json", logs)
    return logs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=str(ROOT / "data/raw_csv"))
    parser.add_argument("--output-dir", default=str(ROOT / "data/cleaned_csv"))
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    args = parser.parse_args()
    logs = clean_directory(Path(args.input_dir), Path(args.output_dir), load_config(args.config))
    print(f"Cleaned {len(logs['activities'])} activities.")


if __name__ == "__main__":
    main()
