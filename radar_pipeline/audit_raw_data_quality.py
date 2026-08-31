"""Audit raw radar-session timing, motion transitions, and label-risk metadata."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from radar_pipeline.common import ROOT, activity_from_path, discover_csv_files, read_csv_canonical


NONFALL = {"walking", "standing", "sitting"}


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for row in frame.itertuples(index=False, name=None):
        values = ["" if pd.isna(value) else str(value) for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _timestamp_deltas_seconds(frame: pd.DataFrame) -> np.ndarray:
    if "timestamp" not in frame:
        return np.asarray([], dtype=float)
    numeric = pd.to_numeric(frame["timestamp"], errors="coerce")
    if numeric.notna().sum() >= 2:
        values = numeric.dropna().to_numpy(dtype=float)
        delta = np.diff(values)
        if len(delta) and np.nanmedian(np.abs(delta)) > 10:
            delta /= 1000.0
    else:
        values = pd.to_datetime(frame["timestamp"], errors="coerce").dropna()
        delta = values.diff().dt.total_seconds().dropna().to_numpy(dtype=float)
    return delta[np.isfinite(delta)]


def _motion(frame: pd.DataFrame) -> np.ndarray:
    if not {"x", "y", "z"}.issubset(frame.columns) or len(frame) < 2:
        return np.asarray([], dtype=float)
    values = frame[["x", "y", "z"]].to_numpy(dtype=float)
    return np.linalg.norm(np.diff(values, axis=0), axis=1)


def _quantile(values: np.ndarray, q: float) -> float:
    return float(np.quantile(values, q)) if len(values) else float("nan")


def audit_raw_data(
    input_dir: Path,
    output_dir: Path,
    annotations_csv: Path | None,
    manifest_csv: Path | None,
    expected_rate_hz: float = 20.0,
    transition_motion_threshold: float = 0.15,
) -> dict[str, object]:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing audit output: {output_dir}")
    paths = discover_csv_files(input_dir)
    if not paths:
        raise FileNotFoundError(f"No raw CSV files found under {input_dir}")
    nominal_dt = 1.0 / expected_rate_hz
    rows: list[dict[str, object]] = []
    for path in paths:
        activity = activity_from_path(path)
        frame = read_csv_canonical(path)
        delta = _timestamp_deltas_seconds(frame)
        motion = _motion(frame)
        edge = max(1, len(motion) // 5)
        frame_index = pd.to_numeric(frame.get("frame", pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=float)
        frame_steps = np.diff(frame_index[np.isfinite(frame_index)]) if len(frame_index) else np.asarray([], dtype=float)
        rows.append(
            {
                "activity": activity,
                "source_session_id": path.stem,
                "source_csv": str(path),
                "frames": int(len(frame)),
                "timestamp_valid": int(len(delta) + 1) if len(delta) else 0,
                "sampling_rate_hz": float(1.0 / np.median(delta)) if len(delta) and np.median(delta) > 0 else np.nan,
                "dt_median_s": float(np.median(delta)) if len(delta) else np.nan,
                "dt_p95_s": _quantile(delta, 0.95),
                "dt_max_s": float(np.max(delta)) if len(delta) else np.nan,
                "jitter_p95_s": _quantile(np.abs(delta - nominal_dt), 0.95),
                "timestamp_nonmonotonic_steps": int((delta <= 0).sum()),
                "timestamp_gaps_gt_1p5x": int((delta > nominal_dt * 1.5).sum()),
                "timestamp_gaps_gt_2x": int((delta > nominal_dt * 2.0).sum()),
                "frame_index_nonunit_steps": int((frame_steps != 1).sum()),
                "frame_index_max_step": float(np.max(frame_steps)) if len(frame_steps) else np.nan,
                "motion_median_m_per_frame": float(np.median(motion)) if len(motion) else np.nan,
                "motion_p95_m_per_frame": _quantile(motion, 0.95),
                "motion_max_m_per_frame": float(np.max(motion)) if len(motion) else np.nan,
                "early_motion_p95_m_per_frame": _quantile(motion[:edge], 0.95),
                "late_motion_p95_m_per_frame": _quantile(motion[-edge:], 0.95),
                "active_motion_fraction": float(np.mean(motion > transition_motion_threshold)) if len(motion) else np.nan,
            }
        )
    sessions = pd.DataFrame(rows)
    sessions["transition_candidate"] = (
        sessions["activity"].isin(NONFALL)
        & (
            (sessions["early_motion_p95_m_per_frame"] > transition_motion_threshold)
            | (sessions["late_motion_p95_m_per_frame"] > transition_motion_threshold)
        )
    )
    sessions["timing_warning"] = (
        (sessions["timestamp_nonmonotonic_steps"] > 0)
        | (sessions["frame_index_nonunit_steps"] > 0)
        | (sessions["timestamp_gaps_gt_2x"] > 0)
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    sessions.to_csv(output_dir / "session_timing_motion_quality.csv", index=False)
    sessions.loc[sessions["transition_candidate"]].sort_values(
        ["activity", "motion_p95_m_per_frame"], ascending=[True, False]
    ).to_csv(output_dir / "nonfall_transition_candidates.csv", index=False)
    class_balance = sessions.groupby("activity", as_index=False).agg(
        raw_sessions=("source_session_id", "nunique"), raw_frames=("frames", "sum"),
        median_frames=("frames", "median"), min_frames=("frames", "min"), max_frames=("frames", "max"),
        timing_warning_sessions=("timing_warning", "sum"),
        transition_candidate_sessions=("transition_candidate", "sum"),
    )
    if manifest_csv and manifest_csv.is_file():
        manifest = pd.read_csv(manifest_csv)
        frozen = manifest.groupby("source_activity", as_index=False).agg(
            frozen_source_sessions=("source_session_id", "nunique")
        ).rename(columns={"source_activity": "activity"})
        class_balance = class_balance.merge(frozen, on="activity", how="outer")
    class_balance.to_csv(output_dir / "class_balance.csv", index=False)
    fall_flags = pd.DataFrame()
    if annotations_csv and annotations_csv.is_file():
        annotations = pd.read_csv(annotations_csv)
        fall_flags = annotations.loc[annotations["activity"].astype(str).eq("fall")].copy()
        if not fall_flags.empty:
            flags = fall_flags.get("quality_flags", pd.Series("", index=fall_flags.index)).astype(str)
            fall_flags["event_near_recording_boundary"] = flags.str.contains("event_near_recording_boundary")
            fall_flags["geometry_edge_warning"] = flags.str.contains("geometry_edge_warning")
            fall_flags.to_csv(output_dir / "fall_annotation_quality_flags.csv", index=False)
    timing_ok = bool((sessions["timestamp_nonmonotonic_steps"] == 0).all() and (sessions["frame_index_nonunit_steps"] == 0).all())
    gap_count = int(sessions["timestamp_gaps_gt_2x"].sum())
    transition_count = int(sessions["transition_candidate"].sum())
    status = "STABLE_WITH_NOTES" if timing_ok else "REVIEW_REQUIRED"
    lines = [
        "# Raw Data Quality and Transition Audit",
        "",
        f"- Status: **{status}**.",
        f"- Raw sessions: {len(sessions)}; expected timing: {expected_rate_hz:.1f} Hz.",
        f"- Frame-index discontinuities: {int(sessions['frame_index_nonunit_steps'].sum())}.",
        f"- Non-monotonic timestamp steps: {int(sessions['timestamp_nonmonotonic_steps'].sum())}.",
        f"- Timestamp gaps >2x nominal interval: {gap_count}.",
        f"- Non-fall transition candidates: {transition_count}; these are motion-risk flags, not relabelled samples.",
        "",
        "## Class Support",
        "",
        _markdown_table(class_balance),
        "",
        "## Policy Gate",
        "",
        "- Fall clips remain event-centred.",
        "- Non-fall clip-policy changes must use these candidate flags to avoid edge/transition-heavy selections.",
        "- No label is changed by this audit. A transition class requires separately annotated transition recordings and a new documented experiment.",
    ]
    if not fall_flags.empty:
        lines.extend(
            [
                "",
                "## Fall Annotation Risks",
                "",
                f"- Boundary-flagged fall annotations: {int(fall_flags['event_near_recording_boundary'].sum())}.",
                f"- Geometry-edge-warning fall annotations: {int(fall_flags['geometry_edge_warning'].sum())}.",
            ]
        )
    (output_dir / "RAW_DATA_QUALITY_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"status": status, "sessions": len(sessions), "transition_candidates": transition_count, "gap_gt_2x": gap_count}


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit raw session timing, motion transitions, class balance, and fall-boundary risk.")
    parser.add_argument("--input-dir", default=str(ROOT / "data/raw_csv"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--annotations-csv", default=str(ROOT / "data/metadata/auto_event_annotations.csv"))
    parser.add_argument("--manifest-csv", default=str(ROOT / "data/metadata/auto_event_exclude_risky_fall_20260731_rebuild01_source_session_folds.csv"))
    parser.add_argument("--expected-rate-hz", type=float, default=20.0)
    parser.add_argument("--transition-motion-threshold", type=float, default=0.15)
    args = parser.parse_args()
    result = audit_raw_data(
        Path(args.input_dir), Path(args.output_dir), Path(args.annotations_csv), Path(args.manifest_csv),
        args.expected_rate_hz, args.transition_motion_threshold,
    )
    print(result)


if __name__ == "__main__":
    main()
