from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from radar_pipeline.common import (
    CLASS_LABELS,
    CLASS_NAMES,
    ROOT,
    activity_from_path,
    discover_csv_files,
    ensure_dirs,
    feature_columns,
    load_config,
    read_csv_canonical,
    write_json,
)


LABELS = dict(zip(CLASS_NAMES, CLASS_LABELS))
ANNOTATION_COLUMNS = [
    "session_id",
    "activity",
    "start_state",
    "event_start_s",
    "impact_s",
    "event_end_s",
    "end_state",
    "distance_band",
    "distance_m",
    "direction",
    "notes",
]
AUTO_ANNOTATION_COLUMNS = [
    "session_id",
    "activity",
    "start_state",
    "event_start_s",
    "impact_s",
    "event_end_s",
    "end_state",
    "distance_m_inferred",
    "distance_band_inferred",
    "direction_inferred",
    "confidence",
    "auto_include_in_training",
    "quality_flags",
    "method_version",
    "method_summary",
]
EXCLUDED_LABEL = -1
AMBIGUOUS_FALL_WEAKNESS_FLAGS = {
    "event_near_recording_boundary",
    "missing_post_event_stability",
    "weak_height_evidence",
    "weak_motion_peak",
}


def _event_cfg(cfg: dict) -> dict:
    return cfg.get("windowing", {}).get("event_aware", {}) or {}


def _exclude_ambiguous_fall_windows(cfg: dict) -> bool:
    return bool(cfg.get("auto_event_annotation", {}).get("exclude_ambiguous_fall_windows", False))


def _is_ambiguous_auto_fall(annotation: dict[str, Any]) -> bool:
    if annotation.get("annotation_kind") != "auto_event_annotations":
        return False
    flags = {flag.strip() for flag in str(annotation.get("quality_flags", "")).split(";") if flag.strip()}
    return "geometry_edge_warning" in flags and bool(flags & AMBIGUOUS_FALL_WEAKNESS_FLAGS)


def _metadata_path(cfg: dict) -> Path:
    configured = _event_cfg(cfg).get("metadata_csv", "data/metadata/session_annotations.csv")
    path = Path(configured)
    return path if path.is_absolute() else ROOT / path


def load_event_annotations(cfg: dict) -> dict[str, dict[str, Any]]:
    """Load optional event annotations keyed by session_id.

    If event-aware mode is disabled, callers should not use this function.
    If enabled, metadata must exist and match the fall sessions being staged.
    """

    path = _metadata_path(cfg)
    if not path.exists():
        raise FileNotFoundError(f"Event-aware metadata_csv does not exist: {path}")
    df = pd.read_csv(path)
    is_auto = set(AUTO_ANNOTATION_COLUMNS).issubset(df.columns)
    required_columns = AUTO_ANNOTATION_COLUMNS if is_auto else ANNOTATION_COLUMNS
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Annotation file {path} is missing required columns: {missing_cols}")
    if df.empty:
        return {}
    df = df.copy()
    df["session_id"] = df["session_id"].astype(str).str.strip()
    blank_sessions = df["session_id"].eq("")
    if blank_sessions.any():
        rows = (df.index[blank_sessions] + 2).tolist()
        raise ValueError(f"Annotation file {path} has blank session_id at CSV rows {rows}")
    duplicates = sorted(df.loc[df["session_id"].duplicated(keep=False), "session_id"].unique().tolist())
    if duplicates:
        raise ValueError(f"Annotation file {path} has duplicate session_id values: {duplicates}")

    annotations: dict[str, dict[str, Any]] = {}
    for row_idx, row in df.iterrows():
        session_id = str(row["session_id"]).strip()
        activity = str(row["activity"]).strip().lower()
        if activity not in LABELS:
            raise ValueError(
                f"Annotation row {row_idx + 2} for {session_id} has invalid activity '{activity}'. "
                f"Expected one of {sorted(LABELS)}"
            )
        item: dict[str, Any] = {col: row[col] for col in required_columns}
        item["session_id"] = session_id
        item["activity"] = activity
        item["start_state"] = str(row["start_state"]).strip().lower() if pd.notna(row["start_state"]) else ""
        item["end_state"] = str(row["end_state"]).strip().lower() if pd.notna(row["end_state"]) else ""
        item["annotation_kind"] = "auto_event_annotations" if is_auto else "manual_event_metadata"
        if is_auto:
            item["distance_m"] = pd.to_numeric(row.get("distance_m_inferred"), errors="coerce")
            item["distance_band"] = row.get("distance_band_inferred", "")
            item["direction"] = row.get("direction_inferred", "")
            item["confidence"] = str(row.get("confidence", "")).strip().lower()
            item["auto_include_in_training"] = bool(row.get("auto_include_in_training"))
            if isinstance(row.get("auto_include_in_training"), str):
                item["auto_include_in_training"] = row.get("auto_include_in_training", "").strip().lower() == "true"
            item["quality_flags"] = str(row.get("quality_flags", "")).strip()
            item["method_version"] = str(row.get("method_version", "")).strip()
            item["method_summary"] = str(row.get("method_summary", "")).strip()
        for col in ("event_start_s", "impact_s", "event_end_s"):
            item[col] = pd.to_numeric(row[col], errors="coerce")
            item[col] = None if pd.isna(item[col]) else float(item[col])
        if not is_auto:
            item["distance_m"] = pd.to_numeric(row["distance_m"], errors="coerce")
        item["distance_m"] = None if pd.isna(item.get("distance_m")) else float(item["distance_m"])
        if activity == "fall":
            missing = [
                col
                for col in ("start_state", "event_start_s", "impact_s", "event_end_s", "end_state")
                if item[col] in {"", None}
            ]
            if missing:
                raise ValueError(f"Annotation row {row_idx + 2} for {session_id} is missing required fall fields: {missing}")
            if not (0.0 <= item["event_start_s"] <= item["impact_s"] <= item["event_end_s"]):
                raise ValueError(
                    f"Annotation row {row_idx + 2} for {session_id} has invalid time ordering: "
                    "expected 0 <= event_start_s <= impact_s <= event_end_s"
                )
        annotations[session_id] = item
    return annotations


def _session_lookup_ids(session_id: str, source_csv: str) -> list[str]:
    ids = [str(session_id)]
    if "__" in str(session_id):
        ids.append(str(session_id).split("__", 1)[1])
    if source_csv:
        ids.append(Path(source_csv).stem)
    deduped: list[str] = []
    for item in ids:
        if item and item not in deduped:
            deduped.append(item)
    return deduped


def _annotation_for_session(
    annotations: dict[str, dict[str, Any]],
    session_id: str,
    source_csv: str,
) -> dict[str, Any] | None:
    matches = [key for key in _session_lookup_ids(session_id, source_csv) if key in annotations]
    if len(matches) > 1:
        raise ValueError(f"Multiple annotations match session '{session_id}' ({source_csv}): {matches}")
    return annotations[matches[0]] if matches else None


def _validate_annotation_for_session(annotation: dict[str, Any], session_id: str, n_frames: int, rate: float) -> None:
    duration_s = n_frames / rate if rate > 0 else 0.0
    for col in ("event_start_s", "impact_s", "event_end_s"):
        value = annotation.get(col)
        if value is None or not np.isfinite(float(value)):
            raise ValueError(f"Annotation for {session_id} has invalid {col}: {value}")
        if float(value) > duration_s:
            raise ValueError(
                f"Annotation for {session_id} has {col}={value:.3f}s outside session duration {duration_s:.3f}s"
            )


def _assert_event_staging_output(output_dir: Path, cfg: dict) -> None:
    if not bool(_event_cfg(cfg).get("enabled", False)):
        return
    resolved = output_dir.resolve()
    forbidden = {(ROOT / "data/windowed").resolve()}
    if resolved in forbidden:
        raise ValueError(
            "Event-aware windowing is staging-only and cannot write to data/windowed. "
            "Use a staging directory such as data/windowed_auto_event_staging or data/windowed_manual_event_staging."
        )


def _event_window_decision(
    activity: str,
    annotation: dict[str, Any] | None,
    start_frame: int,
    win_len: int,
    rate: float,
    cfg: dict,
) -> dict[str, Any]:
    start_s = start_frame / rate
    end_s = (start_frame + win_len) / rate
    duration_s = win_len / rate
    base = {
        "label_source": "legacy_folder",
        "event_phase": "legacy",
        "include_in_training": True,
        "exclude_reason": "",
        "annotation_confidence": "",
        "quality_flags": "",
        "method_version": "",
        "overlap_seconds": np.nan,
        "overlap_fraction": np.nan,
        "window_start_s": float(start_s),
        "window_end_s": float(end_s),
    }
    if annotation is None or activity != "fall":
        base["label"] = LABELS[activity]
        return base

    event_start = float(annotation["event_start_s"])
    event_end = float(annotation["event_end_s"])
    overlap = max(0.0, min(end_s, event_end) - max(start_s, event_start))
    overlap_fraction = overlap / duration_s if duration_s > 0 else 0.0
    threshold = float(_event_cfg(cfg).get("fall_overlap_threshold", 0.50))
    base.update(
        {
            "label_source": "event_metadata",
            "annotation_confidence": str(annotation.get("confidence", "")),
            "quality_flags": str(annotation.get("quality_flags", "")),
            "method_version": str(annotation.get("method_version", "")),
            "overlap_seconds": float(overlap),
            "overlap_fraction": float(overlap_fraction),
        }
    )
    annotation_allows_training = bool(annotation.get("auto_include_in_training", True))
    annotation_exclude_reason = "auto_annotation_excluded" if not annotation_allows_training else ""

    if overlap_fraction >= threshold:
        base["event_phase"] = "fall_event"
        base["include_in_training"] = annotation_allows_training
        base["exclude_reason"] = annotation_exclude_reason
        base["label"] = LABELS["fall"]
        if annotation_allows_training and _exclude_ambiguous_fall_windows(cfg) and _is_ambiguous_auto_fall(annotation):
            base["include_in_training"] = False
            base["exclude_reason"] = "ambiguous_auto_fall_window"
    elif end_s <= event_start:
        start_state = str(annotation.get("start_state", "")).strip().lower()
        base["event_phase"] = "pre_event"
        if start_state in LABELS and start_state != "fall":
            base["include_in_training"] = annotation_allows_training
            base["exclude_reason"] = annotation_exclude_reason
            base["label"] = LABELS[start_state]
        else:
            base["include_in_training"] = False
            base["exclude_reason"] = "unsupported_start_state"
            base["label"] = EXCLUDED_LABEL
    elif start_s >= event_end:
        base["event_phase"] = "post_event"
        base["include_in_training"] = False
        base["exclude_reason"] = "post_event_excluded"
        base["label"] = EXCLUDED_LABEL
    else:
        base["event_phase"] = "transition_excluded"
        base["include_in_training"] = False
        base["exclude_reason"] = "transition_below_overlap_threshold"
        base["label"] = EXCLUDED_LABEL
    return base


def make_windows_for_session(
    df: pd.DataFrame,
    activity: str,
    cfg: dict,
    source_csv: str = "",
    annotation: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str], list[dict]]:
    cols = feature_columns(df)
    rate = float(cfg["sampling"]["expected_rate_hz"])
    win_len = max(1, int(round(float(cfg["windowing"]["window_len_sec"]) * rate)))
    step = max(1, int(round(win_len * (1.0 - float(cfg["windowing"]["overlap_pct"])))))
    min_valid = float(cfg["windowing"]["min_valid_ratio"])
    values = df[cols].to_numpy(dtype=np.float32)
    windows = []
    labels = []
    metadata = []
    session_id = str(df["session_id"].iloc[0]) if "session_id" in df.columns and len(df) else ""
    if activity not in LABELS:
        raise ValueError(f"Unsupported activity '{activity}'. Expected one of {sorted(LABELS)}")
    if annotation is not None:
        if annotation.get("activity") != activity:
            raise ValueError(
                f"Annotation for session '{session_id}' has activity '{annotation.get('activity')}', "
                f"but source folder resolves to '{activity}'"
            )
        _validate_annotation_for_session(annotation, session_id, len(df), rate)
    for start in range(0, max(len(values) - win_len + 1, 0), step):
        chunk = values[start : start + win_len]
        valid_ratio = np.isfinite(chunk).mean()
        if valid_ratio < min_valid:
            continue
        if not np.isfinite(chunk).all():
            continue
        decision = _event_window_decision(activity, annotation, start, win_len, rate, cfg)
        windows.append(chunk)
        labels.append(int(decision["label"]))
        metadata.append(
            {
                "session_id": session_id,
                "start_frame": int(start),
                "end_frame": int(start + win_len - 1),
                "source_csv": source_csv,
                "label_source": decision["label_source"],
                "event_phase": decision["event_phase"],
                "include_in_training": bool(decision["include_in_training"]),
                "exclude_reason": decision["exclude_reason"],
                "annotation_confidence": decision["annotation_confidence"],
                "quality_flags": decision["quality_flags"],
                "method_version": decision["method_version"],
                "overlap_seconds": float(decision["overlap_seconds"]),
                "overlap_fraction": float(decision["overlap_fraction"]),
                "window_start_s": float(decision["window_start_s"]),
                "window_end_s": float(decision["window_end_s"]),
            }
        )
    if not windows:
        return np.empty((0, win_len, len(cols)), dtype=np.float32), np.empty((0,), dtype=np.int64), cols, metadata
    return np.stack(windows).astype(np.float32), np.asarray(labels, dtype=np.int64), cols, metadata


def make_windows(df: pd.DataFrame, activity: str, cfg: dict) -> tuple[np.ndarray, np.ndarray, list[str]]:
    X, y, cols, _ = make_windows_for_session(df, activity, cfg)
    return X, y, cols


def window_directory(input_dir: Path, output_dir: Path, cfg: dict) -> dict:
    ensure_dirs()
    _assert_event_staging_output(output_dir, cfg)
    output_dir.mkdir(parents=True, exist_ok=True)
    event_enabled = bool(_event_cfg(cfg).get("enabled", False))
    metadata_path = _metadata_path(cfg)
    annotations = load_event_annotations(cfg) if event_enabled else {}
    matched_annotations: set[str] = set()
    summary = {
        "activities": {},
        "labels": LABELS,
        "event_aware": {
            "enabled": event_enabled,
            "metadata_csv": str(metadata_path),
            "metadata_path_resolved": str(metadata_path.resolve()) if metadata_path.exists() else str(metadata_path),
            "fall_overlap_threshold": float(_event_cfg(cfg).get("fall_overlap_threshold", 0.50)),
            "exclude_ambiguous_fall_windows": _exclude_ambiguous_fall_windows(cfg),
            "label_source": "",
            "included_excluded_by_activity_phase": {},
        },
    }
    if event_enabled:
        print(f"Using event metadata: {metadata_path.resolve()}")
        if metadata_path.name == "auto_event_annotations.csv":
            summary["event_aware"]["label_source"] = "auto_event_annotations"
        else:
            summary["event_aware"]["label_source"] = "manual_event_metadata"
    paths = discover_csv_files(input_dir)
    paths_by_activity: dict[str, list[Path]] = {}
    for path in paths:
        paths_by_activity.setdefault(activity_from_path(path), []).append(path)

    for activity, activity_paths in sorted(paths_by_activity.items()):
        session_arrays = []
        session_labels = []
        all_meta = []
        session_summaries = {}
        total_frames = 0
        cols: list[str] = []
        session_summaries = {}
        for path in activity_paths:
            df = read_csv_canonical(path)
            if "session_id" not in df.columns:
                df["session_id"] = f"{activity}__{path.stem.lower()}"
            total_frames += int(len(df))
            if not cols:
                cols = feature_columns(df)
            for session_id, session_df in df.groupby("session_id", sort=False):
                session_df = session_df.reset_index(drop=True)
                annotation = _annotation_for_session(annotations, str(session_id), str(path))
                if event_enabled and activity == "fall" and annotation is None:
                    raise ValueError(f"Missing event-aware metadata row for fall session '{session_id}' from {path}")
                if annotation is not None:
                    matched_annotations.add(str(annotation["session_id"]))
                X_session, y_session, cols, meta = make_windows_for_session(
                    session_df,
                    activity,
                    cfg,
                    str(path),
                    annotation=annotation,
                )
                session_arrays.append(X_session)
                session_labels.append(y_session)
                all_meta.extend(meta)
                included = sum(1 for item in meta if item["include_in_training"])
                excluded = len(meta) - included
                phase_counts = pd.Series([item["event_phase"] for item in meta]).value_counts().to_dict() if meta else {}
                reason_counts = pd.Series([item["exclude_reason"] or "included" for item in meta]).value_counts().to_dict() if meta else {}
                overlap_values = [float(item["overlap_seconds"]) for item in meta if np.isfinite(float(item["overlap_seconds"]))]
                overlap_fraction_values = [
                    float(item["overlap_fraction"]) for item in meta if np.isfinite(float(item["overlap_fraction"]))
                ]
                session_summaries[str(session_id)] = {
                    "frames": int(len(session_df)),
                    "windows": int(X_session.shape[0]),
                    "included_windows": int(included),
                    "excluded_windows": int(excluded),
                    "label_source": "event_metadata" if annotation is not None and activity == "fall" else "legacy_folder",
                    "event_phase_counts": {str(k): int(v) for k, v in phase_counts.items()},
                    "exclude_reason_counts": {str(k): int(v) for k, v in reason_counts.items()},
                    "overlap_seconds_mean": float(np.mean(overlap_values)) if overlap_values else None,
                    "overlap_fraction_mean": float(np.mean(overlap_fraction_values)) if overlap_fraction_values else None,
                    "source_csv": str(path),
                }
        X = np.concatenate(session_arrays, axis=0) if session_arrays else np.empty((0, 0, len(cols)), dtype=np.float32)
        y = np.concatenate(session_labels, axis=0) if session_labels else np.empty((0,), dtype=np.int64)
        session_ids = np.asarray([m["session_id"] for m in all_meta], dtype=str)
        start_frames = np.asarray([m["start_frame"] for m in all_meta], dtype=np.int64)
        end_frames = np.asarray([m["end_frame"] for m in all_meta], dtype=np.int64)
        source_csvs = np.asarray([m["source_csv"] for m in all_meta], dtype=str)
        label_sources = np.asarray([m["label_source"] for m in all_meta], dtype=str)
        event_phases = np.asarray([m["event_phase"] for m in all_meta], dtype=str)
        include_in_training = np.asarray([m["include_in_training"] for m in all_meta], dtype=bool)
        overlap_seconds = np.asarray([m["overlap_seconds"] for m in all_meta], dtype=np.float32)
        overlap_fraction = np.asarray([m["overlap_fraction"] for m in all_meta], dtype=np.float32)
        window_start_s = np.asarray([m["window_start_s"] for m in all_meta], dtype=np.float32)
        window_end_s = np.asarray([m["window_end_s"] for m in all_meta], dtype=np.float32)
        exclude_reasons = np.asarray([m["exclude_reason"] for m in all_meta], dtype=str)
        annotation_confidences = np.asarray([m["annotation_confidence"] for m in all_meta], dtype=str)
        quality_flags = np.asarray([m["quality_flags"] for m in all_meta], dtype=str)
        method_versions = np.asarray([m["method_version"] for m in all_meta], dtype=str)
        out_path = output_dir / f"{activity}_windows.npz"
        np.savez_compressed(
            out_path,
            X=X,
            y=y,
            feature_names=np.asarray(cols),
            activity=activity,
            session_id=session_ids,
            start_frame=start_frames,
            end_frame=end_frames,
            source_csv=source_csvs,
            label_source=label_sources,
            event_phase=event_phases,
            include_in_training=include_in_training,
            overlap_seconds=overlap_seconds,
            overlap_fraction=overlap_fraction,
            window_start_s=window_start_s,
            window_end_s=window_end_s,
            exclude_reason=exclude_reasons,
            annotation_confidence=annotation_confidences,
            quality_flags=quality_flags,
            method_version=method_versions,
        )
        included_total = int(include_in_training.sum()) if len(include_in_training) else 0
        phase_counts = pd.Series(event_phases).value_counts().to_dict() if len(event_phases) else {}
        reason_counts = pd.Series([x or "included" for x in exclude_reasons]).value_counts().to_dict() if len(exclude_reasons) else {}
        summary["activities"][activity] = {
            "windows": int(X.shape[0]),
            "included_windows": included_total,
            "excluded_windows": int(X.shape[0] - included_total),
            "sessions": int(len(session_summaries)),
            "frames": int(total_frames),
            "window_shape": list(X.shape[1:]),
            "output": str(out_path),
            "features": cols,
            "event_phase_counts": {str(k): int(v) for k, v in phase_counts.items()},
            "exclude_reason_counts": {str(k): int(v) for k, v in reason_counts.items()},
            "session_summary": session_summaries,
        }
    if event_enabled:
        unmatched = sorted(set(annotations) - matched_annotations)
        if unmatched:
            raise ValueError(f"Event-aware metadata has session_id values not found in cleaned input: {unmatched}")
        phase_rows = []
        for activity, activity_summary in summary["activities"].items():
            for session in activity_summary["session_summary"].values():
                for phase, count in session.get("event_phase_counts", {}).items():
                    phase_rows.append(
                        {
                            "activity": activity,
                            "phase": phase,
                            "windows": int(count),
                            "included": int(session["included_windows"]) if len(session.get("event_phase_counts", {})) == 1 else None,
                        }
                    )
        grouped: dict[str, dict[str, dict[str, int]]] = {}
        for activity, activity_summary in summary["activities"].items():
            grouped[activity] = {}
            for session in activity_summary["session_summary"].values():
                for phase, count in session.get("event_phase_counts", {}).items():
                    item = grouped[activity].setdefault(phase, {"windows": 0})
                    item["windows"] += int(count)
            grouped[activity]["__included_excluded__"] = {
                "included_windows": int(activity_summary["included_windows"]),
                "excluded_windows": int(activity_summary["excluded_windows"]),
            }
        summary["event_aware"]["included_excluded_by_activity_phase"] = grouped
    write_json(output_dir / "window_log.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=str(ROOT / "data/cleaned_csv"))
    parser.add_argument("--output-dir", default=str(ROOT / "data/windowed"))
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    args = parser.parse_args()
    summary = window_directory(Path(args.input_dir), Path(args.output_dir), load_config(args.config))
    if summary["event_aware"]["enabled"]:
        print(f"Resolved metadata path: {summary['event_aware']['metadata_path_resolved']}")
    for activity, item in summary["activities"].items():
        print(f"{activity}: {item['windows']} windows")


if __name__ == "__main__":
    main()
