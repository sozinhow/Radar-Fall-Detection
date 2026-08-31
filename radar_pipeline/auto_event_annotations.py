from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from radar_pipeline.common import ROOT, activity_from_path, discover_csv_files, estimated_height_m, load_config, read_csv_canonical
from radar_pipeline.dataset_builder import build_dataset
from radar_pipeline.event_annotation_review import _elapsed_seconds, _motion_proxy, _session_id_for
from radar_pipeline.windowing import window_directory


METHOD_VERSION = "auto_event_heuristic_v1"
METHOD_SUMMARY = (
    "Heuristic radar-only event inference using robust normalized xyz displacement, "
    "height/range/elevation derivatives, peak prominence, post-impact low-motion "
    "stability, estimated-height change, and cleaning-drop quality gates. "
    "Not manually verified ground truth."
)

AUTO_COLUMNS = [
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


def _robust_z(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    med = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - med))
    scale = 1.4826 * mad if mad > 1e-9 else np.nanstd(values)
    if not np.isfinite(scale) or scale <= 1e-9:
        return np.zeros_like(values, dtype=float)
    return (values - med) / scale


def _smooth(values: np.ndarray, width: int = 5) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) < width:
        return values
    kernel = np.ones(width, dtype=float) / width
    return np.convolve(values, kernel, mode="same")


def _cleaning_session_info(cleaning_log: dict[str, Any], session_id: str) -> dict[str, Any]:
    sessions = cleaning_log.get("activities", {}).get("fall", {}).get("sessions", {})
    return sessions.get(f"fall__{session_id}", sessions.get(session_id, {}))


def _band(distance: float, near_max: float, middle_max: float) -> str:
    if not np.isfinite(distance):
        return "unknown"
    if distance <= near_max:
        return "near"
    if distance <= middle_max:
        return "middle"
    return "far"


def _direction(df: pd.DataFrame, start_i: int, end_i: int) -> str:
    if not {"x", "y", "z"}.issubset(df.columns) or end_i <= start_i:
        return "unknown"
    before = df[["x", "y", "z"]].iloc[max(0, start_i - 3) : start_i + 1].median()
    after = df[["x", "y", "z"]].iloc[end_i : min(len(df), end_i + 4)].median()
    delta = after - before
    if not np.isfinite(delta).all() or float(np.linalg.norm(delta)) < 0.15:
        return "unknown"
    horizontal_axis = "y" if abs(float(delta["y"])) >= abs(float(delta["x"])) else "x"
    value = float(delta[horizontal_axis])
    if horizontal_axis == "y":
        return "forward" if value > 0 else "backward"
    return "right" if value > 0 else "left"


def infer_session_annotation(
    cleaned_df: pd.DataFrame,
    session_id: str,
    cfg: dict,
    cleaning_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rate = float(cfg["sampling"]["expected_rate_hz"])
    mount_height = float(cfg["radar"]["mount_height_m"])
    auto_cfg = cfg.get("auto_event_annotation", {})
    near_max = float(auto_cfg.get("near_max_m", 2.0))
    middle_max = float(auto_cfg.get("middle_max_m", 3.2))
    min_event_s = float(auto_cfg.get("min_event_duration_s", 0.8))
    max_event_s = float(auto_cfg.get("max_event_duration_s", 2.2))
    high_drop_threshold = float(auto_cfg.get("high_drop_threshold", 0.40))
    low_motion_z = float(auto_cfg.get("low_motion_z", 0.8))
    peak_z_threshold = float(auto_cfg.get("peak_z_threshold", 2.5))

    df = cleaned_df.reset_index(drop=True).copy()
    n = len(df)
    t = _elapsed_seconds(df, rate)
    duration = n / rate
    flags: list[str] = []

    motion = _motion_proxy(df).to_numpy(dtype=float)
    range_v = pd.to_numeric(df["range_m"], errors="coerce").to_numpy(dtype=float) if "range_m" in df else np.full(n, np.nan)
    elev_v = (
        pd.to_numeric(df["elevation_deg"], errors="coerce").to_numpy(dtype=float)
        if "elevation_deg" in df
        else np.full(n, np.nan)
    )
    if {"range_m", "elevation_deg"}.issubset(df.columns):
        height = estimated_height_m(df["range_m"], df["elevation_deg"], mount_height).to_numpy(dtype=float)
    else:
        height = np.full(n, np.nan)
        flags.append("weak_height_evidence")

    if n < max(8, int(rate * min_event_s)):
        flags.append("too_short_for_reliable_inference")

    h_slope = np.r_[0.0, np.abs(np.diff(height))] if np.isfinite(height).any() else np.zeros(n)
    r_slope = np.r_[0.0, np.abs(np.diff(range_v))] if np.isfinite(range_v).any() else np.zeros(n)
    e_slope = np.r_[0.0, np.abs(np.diff(elev_v))] if np.isfinite(elev_v).any() else np.zeros(n)
    score = _smooth(
        np.maximum(_robust_z(motion), 0)
        + 0.7 * np.maximum(_robust_z(h_slope), 0)
        + 0.4 * np.maximum(_robust_z(r_slope), 0)
        + 0.3 * np.maximum(_robust_z(e_slope), 0),
        width=5,
    )

    if len(score) == 0 or not np.isfinite(score).any():
        impact_i = max(1, n // 2)
        flags.append("no_finite_event_score")
    else:
        search_start = min(max(2, int(0.2 * rate)), max(0, n - 1))
        search_end = max(search_start + 1, n - min(2, n))
        search_slice = score[search_start:search_end] if search_end > search_start else score
        impact_i = int(np.nanargmax(search_slice) + search_start) if len(search_slice) else int(np.nanargmax(score))

    peak_score = float(score[impact_i]) if len(score) else 0.0
    if peak_score < peak_z_threshold:
        flags.append("weak_motion_peak")
    peak_count = int((score > max(peak_z_threshold, 0.75 * peak_score)).sum()) if len(score) else 0
    if peak_count > max(8, int(0.4 * rate)):
        flags.append("multiple_or_broad_peaks")

    pre_limit = max(0, impact_i - int(max_event_s * rate))
    onset_threshold = max(0.8, 0.35 * peak_score)
    onset_candidates = np.where(score[pre_limit : impact_i + 1] >= onset_threshold)[0] if len(score) else np.array([])
    start_i = int(pre_limit + onset_candidates[0]) if len(onset_candidates) else max(0, impact_i - int(1.0 * rate))

    post_limit = min(n - 1, impact_i + int(max_event_s * rate))
    low_candidates = []
    run = max(3, int(0.25 * rate))
    for idx in range(impact_i + 1, max(impact_i + 2, post_limit - run + 1)):
        segment = score[idx : idx + run]
        if len(segment) == run and float(np.nanmedian(segment)) <= low_motion_z:
            low_candidates.append(idx)
            break
    end_i = int(low_candidates[0]) if low_candidates else min(n - 1, impact_i + int(1.0 * rate))

    min_frames = max(2, int(min_event_s * rate))
    max_frames = max(min_frames + 1, int(max_event_s * rate))
    if impact_i - start_i < int(0.25 * rate):
        start_i = max(0, impact_i - int(0.5 * rate))
    if end_i - impact_i < int(0.25 * rate):
        end_i = min(n - 1, impact_i + int(0.5 * rate))
    if end_i - start_i < min_frames:
        pad = (min_frames - (end_i - start_i)) // 2 + 1
        start_i = max(0, start_i - pad)
        end_i = min(n - 1, end_i + pad)
    if end_i - start_i > max_frames:
        start_i = max(0, impact_i - max_frames // 2)
        end_i = min(n - 1, start_i + max_frames)
    if not (start_i < impact_i < end_i):
        impact_i = min(max(1, impact_i), max(1, n - 2))
        start_i = max(0, impact_i - int(0.5 * rate))
        end_i = min(n - 1, impact_i + int(0.5 * rate))

    event_start_s = start_i / rate
    impact_s = impact_i / rate
    event_end_s = max(end_i / rate, impact_s + 1.0 / rate)

    if event_start_s <= 0.15 or event_end_s >= duration - 0.15:
        flags.append("event_near_recording_boundary")

    post = slice(min(n, end_i + 1), min(n, end_i + 1 + int(0.8 * rate)))
    pre = slice(max(0, start_i - int(0.8 * rate)), max(1, start_i))
    post_motion = float(np.nanmedian(motion[post])) if len(motion[post]) else np.inf
    pre_height = float(np.nanmedian(height[pre])) if np.isfinite(height[pre]).any() else np.nan
    post_height = float(np.nanmedian(height[post])) if np.isfinite(height[post]).any() else np.nan
    height_drop = pre_height - post_height if np.isfinite(pre_height) and np.isfinite(post_height) else np.nan
    if not np.isfinite(height_drop) or height_drop < 0.15:
        flags.append("weak_height_evidence")
    if not np.isfinite(post_motion) or post_motion > max(0.08, np.nanmedian(motion) + np.nanstd(motion)):
        flags.append("missing_post_event_stability")

    edge_ratio = float(np.nanmean(np.abs(elev_v) >= float(cfg["radar"]["elevation_beamwidth_deg"]) * 0.5 * 0.95)) if np.isfinite(elev_v).any() else 0
    if edge_ratio > 0.30:
        flags.append("geometry_edge_warning")

    cleaning_info = cleaning_info or {}
    drop = cleaning_info.get("non_frozen_drop_fraction")
    if drop is not None and float(drop) >= high_drop_threshold:
        flags.append(f"high_cleaning_drop={float(drop):.1%}")

    event_duration = event_end_s - event_start_s
    if event_duration < min_event_s or event_duration > max_event_s:
        flags.append("event_duration_outside_target")

    pre_range = float(np.nanmedian(range_v[pre])) if np.isfinite(range_v[pre]).any() else float(np.nanmedian(range_v))
    start_state = "standing"
    # Walking is only defensible if pre-event motion is sustained and larger than static jitter.
    if len(motion[pre]) >= 5 and float(np.nanmedian(motion[pre])) > max(0.08, 2.0 * float(np.nanmedian(motion[: max(1, start_i)]))):
        start_state = "walking"
    if "weak_motion_peak" in flags and "weak_height_evidence" in flags:
        start_state = "unknown"

    end_state = "lying" if np.isfinite(post_height) and post_height < max(1.05, pre_height - 0.15) and "missing_post_event_stability" not in flags else "unknown"
    direction = _direction(df, start_i, end_i)

    hard_flags = {
        "too_short_for_reliable_inference",
        "no_finite_event_score",
        "weak_motion_peak",
        "event_duration_outside_target",
    }
    if any(flag.startswith("high_cleaning_drop") for flag in flags) and not (peak_score >= peak_z_threshold * 1.5 and np.isfinite(height_drop) and height_drop >= 0.25):
        hard_flags.add("high_cleaning_drop")
    hard_fail = any(flag in hard_flags or (flag.startswith("high_cleaning_drop") and "high_cleaning_drop" in hard_flags) for flag in flags)
    if hard_fail:
        confidence = "low"
    elif any(flag in {"weak_height_evidence", "multiple_or_broad_peaks", "event_near_recording_boundary", "missing_post_event_stability", "geometry_edge_warning"} for flag in flags):
        confidence = "medium"
    else:
        confidence = "high"
    auto_include = confidence in {"high", "medium"}

    return {
        "session_id": session_id,
        "activity": "fall",
        "start_state": start_state,
        "event_start_s": round(float(event_start_s), 3),
        "impact_s": round(float(impact_s), 3),
        "event_end_s": round(float(event_end_s), 3),
        "end_state": end_state,
        "distance_m_inferred": round(float(pre_range), 3) if np.isfinite(pre_range) else "",
        "distance_band_inferred": _band(pre_range, near_max, middle_max),
        "direction_inferred": direction,
        "confidence": confidence,
        "auto_include_in_training": bool(auto_include),
        "quality_flags": ";".join(sorted(set(flags))) if flags else "none",
        "method_version": METHOD_VERSION,
        "method_summary": METHOD_SUMMARY,
        "_score": score,
        "_motion": motion,
        "_height": height,
        "_range": range_v,
        "_t": t,
        "_cleaning_drop": float(drop) if drop is not None else np.nan,
    }


def _plot_auto(row: dict[str, Any], out_path: Path) -> None:
    t = row["_t"]
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    axes[0].plot(t, row["_motion"], label="xyz_delta_mag", color="tab:blue")
    axes[0].plot(t, row["_score"], label="combined event score", color="tab:orange", alpha=0.8)
    axes[0].set_ylabel("motion / score")
    axes[1].plot(t, row["_height"], label="estimated_height_m", color="tab:green")
    axes[1].set_ylabel("height (m)")
    axes[2].plot(t, row["_range"], label="range_m", color="tab:purple")
    axes[2].plot(t, np.r_[0.0, np.diff(row["_range"])], label="range_delta", color="tab:red", alpha=0.6)
    axes[2].set_ylabel("range")
    axes[2].set_xlabel("Elapsed time (s)")
    for ax in axes:
        for value, color, label in [
            (row["event_start_s"], "tab:green", "event_start"),
            (row["impact_s"], "tab:red", "impact"),
            (row["event_end_s"], "tab:purple", "event_end"),
        ]:
            ax.axvline(float(value), color=color, linestyle="--", linewidth=1.2, label=label)
        ax.grid(alpha=0.25)
        handles, labels = ax.get_legend_handles_labels()
        dedup = dict(zip(labels, handles))
        ax.legend(dedup.values(), dedup.keys(), fontsize=8, loc="upper right")
    fig.suptitle(
        f"{row['session_id']} auto event annotation\n"
        f"confidence={row['confidence']}; include={row['auto_include_in_training']}; flags={row['quality_flags']}"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def generate_auto_event_annotations(
    cleaned_fall_dir: Path,
    output_csv: Path,
    plot_dir: Path,
    index_csv: Path,
    report_path: Path,
    cfg: dict,
    raw_fall_dir: Path | None = None,
    cleaning_log_path: Path | None = None,
) -> dict[str, Any]:
    cleaning_log = {}
    if cleaning_log_path and cleaning_log_path.exists():
        cleaning_log = json.load(open(cleaning_log_path))
    rows = []
    index_rows = []
    for stale in (plot_dir / "fall").glob("*.png"):
        stale.unlink()
    cleaned_paths = [path for path in discover_csv_files(cleaned_fall_dir) if activity_from_path(path) == "fall"]
    raw_lookup = {path.name: path for path in discover_csv_files(raw_fall_dir)} if raw_fall_dir else {}
    for path in cleaned_paths:
        df = read_csv_canonical(path)
        session_id = _session_id_for(path, df)
        row = infer_session_annotation(df, session_id, cfg, _cleaning_session_info(cleaning_log, session_id))
        plot_path = plot_dir / "fall" / f"{session_id}.png"
        _plot_auto(row, plot_path)
        public = {col: row[col] for col in AUTO_COLUMNS}
        rows.append(public)
        raw_path = raw_lookup.get(path.name, Path("")) if raw_fall_dir else Path("")
        index_rows.append(
            {
                **public,
                "cleaned_csv": str(path),
                "raw_csv": str(raw_path) if raw_fall_dir and raw_path.exists() else "",
                "plot_path": str(plot_path),
                "duration_s": round(len(df) / float(cfg["sampling"]["expected_rate_hz"]), 3),
                "cleaned_frames": len(df),
                "cleaning_drop": row["_cleaning_drop"],
            }
        )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=AUTO_COLUMNS).to_csv(output_csv, index=False)
    index_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(index_rows).to_csv(index_csv, index=False)
    _write_report(pd.DataFrame(index_rows), report_path, cfg)
    return {
        "annotations_csv": str(output_csv),
        "plot_dir": str(plot_dir),
        "index_csv": str(index_csv),
        "report_path": str(report_path),
        "sessions": len(rows),
        "confidence_counts": pd.Series([r["confidence"] for r in rows]).value_counts().to_dict(),
        "included_sessions": int(sum(bool(r["auto_include_in_training"]) for r in rows)),
        "excluded_sessions": int(sum(not bool(r["auto_include_in_training"]) for r in rows)),
    }


def _write_report(index: pd.DataFrame, report_path: Path, cfg: dict) -> None:
    conf = index["confidence"].value_counts().to_dict() if not index.empty else {}
    include = index["auto_include_in_training"].value_counts().to_dict() if not index.empty else {}
    durations = index["event_end_s"].astype(float) - index["event_start_s"].astype(float) if not index.empty else pd.Series(dtype=float)
    excluded = index.loc[~index["auto_include_in_training"].astype(bool), ["session_id", "confidence", "quality_flags"]]
    lines = [
        "# Automatic Event Annotation Report",
        "",
        "These annotations are `auto_event_annotations`: algorithmically inferred from radar time series.",
        "They are not manually verified ground truth and are not commercially validated fall-event labels.",
        "",
        "## Method",
        "",
        f"- Method version: `{METHOD_VERSION}`",
        f"- Summary: {METHOD_SUMMARY}",
        "- Impact point: maximum smoothed combined score from xyz displacement, absolute height/range/elevation derivatives.",
        "- Event start: first sustained pre-impact score rise above 35% of peak, bounded to a short event window.",
        "- Event end: first post-impact low-motion run, bounded to the target event duration.",
        "- Confidence: downgraded for weak motion peak, weak height evidence, broad/multiple peaks, boundary events, missing post-event stability, geometry-edge warnings, and high cleaning drop.",
        "- Doppler index is not treated as calibrated velocity.",
        "",
        "## Thresholds",
        "",
        "```yaml",
        yaml.safe_dump(cfg.get("auto_event_annotation", {}), sort_keys=True).strip(),
        "```",
        "",
        "## Counts",
        "",
        f"- Sessions: {len(index)}",
        f"- Confidence counts: `{conf}`",
        f"- Include counts: `{include}`",
        f"- Event duration seconds: min={durations.min():.3f}, median={durations.median():.3f}, max={durations.max():.3f}" if len(durations) else "- Event duration seconds: none",
        "",
        "## Excluded sessions",
        "",
    ]
    if excluded.empty:
        lines.append("- None")
    else:
        for _, row in excluded.iterrows():
            lines.append(f"- `{row['session_id']}`: confidence={row['confidence']}; flags={row['quality_flags']}")
    lines.extend(["", "## Cleaning drop summary", ""])
    for _, row in index.iterrows():
        drop = row.get("cleaning_drop")
        if pd.notna(drop):
            lines.append(f"- `{row['session_id']}`: non-frozen drop {float(drop):.1%}")
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- The heuristic may select the largest radar motion artifact rather than the clinical fall event.",
            "- Low-height postures can overlap with sitting or lying non-fall activities.",
            "- Cleaning may remove or distort parts of the transition.",
            "- These labels are suitable only for staging experiments and audit, not for commercial validation claims.",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_auto_event_staging(cfg: dict, annotations_csv: Path, window_dir: Path, final_dir: Path) -> dict[str, Any]:
    staging_cfg = dict(cfg)
    staging_cfg["windowing"] = dict(cfg["windowing"])
    staging_cfg["windowing"]["event_aware"] = {
        "enabled": True,
        "metadata_csv": str(annotations_csv),
        "fall_overlap_threshold": float(cfg["windowing"].get("event_aware", {}).get("fall_overlap_threshold", 0.50)),
    }
    window_summary = window_directory(ROOT / "data/cleaned_csv", window_dir, staging_cfg)
    dataset_summary = build_dataset(window_dir, final_dir, staging_cfg)
    return {"window_summary": window_summary, "dataset_summary": dataset_summary}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cleaned-fall-dir", default=str(ROOT / "data/cleaned_csv"))
    parser.add_argument("--raw-fall-dir", default=str(ROOT / "data/raw_csv/fall"))
    parser.add_argument("--annotations-csv", default=str(ROOT / "data/metadata/auto_event_annotations.csv"))
    parser.add_argument("--plot-dir", default=str(ROOT / "outputs/validation/auto_event_annotation"))
    parser.add_argument("--index-csv", default=str(ROOT / "outputs/validation/auto_event_annotation/fall/index.csv"))
    parser.add_argument("--report", default=str(ROOT / "outputs/auto_event_annotation_report.md"))
    parser.add_argument("--window-output-dir", default=str(ROOT / "data/windowed_auto_event_staging"))
    parser.add_argument("--final-output-dir", default=str(ROOT / "data/final_dataset_auto_event_staging"))
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    parser.add_argument("--cleaning-log", default=str(ROOT / "data/cleaning_log.json"))
    parser.add_argument("--run-staging", action="store_true", help="Deprecated; use explicit windowing/dataset_builder staging commands instead.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    result = generate_auto_event_annotations(
        Path(args.cleaned_fall_dir),
        Path(args.annotations_csv),
        Path(args.plot_dir),
        Path(args.index_csv),
        Path(args.report),
        cfg,
        Path(args.raw_fall_dir),
        Path(args.cleaning_log),
    )
    if args.run_staging:
        result["staging"] = run_auto_event_staging(
            cfg,
            Path(args.annotations_csv),
            Path(args.window_output_dir),
            Path(args.final_output_dir),
        )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
