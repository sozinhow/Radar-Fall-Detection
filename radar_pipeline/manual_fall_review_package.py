from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "radar_mpl_cache"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from radar_pipeline.common import ROOT, elapsed_axis, read_csv_canonical


DEFAULT_RAW_FALL_DIR = ROOT / "data/raw_csv/fall"
DEFAULT_ANNOTATIONS = ROOT / "data/metadata/auto_event_annotations.csv"
DEFAULT_MANIFEST = ROOT / "data/metadata/auto_event_aware_v1_source_session_folds.csv"
DEFAULT_WINDOWED_FALL = ROOT / "data/windowed_auto_event_staging_20260716/fall_windows.npz"
DEFAULT_ALERT_DIR = ROOT / "outputs/experiments/causal_streaming_alert_budget005_sgkf4_20260716"
DEFAULT_PROB_DIR = ROOT / "outputs/experiments/causal_streaming_alert_sgkf4_20260716"
DEFAULT_OUTPUT_PARENT = ROOT / "outputs/review_packages"


def _unique_output_dir(parent: Path, stem: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d")
    base = parent / f"{stem}_{stamp}"
    if not base.exists():
        return base
    suffix = 2
    while True:
        candidate = parent / f"{stem}_{stamp}_{suffix}"
        if not candidate.exists():
            return candidate
        suffix += 1


def _read_cleaning_log(path: Path) -> dict[str, dict[str, object]]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(session_id): dict(row)
        for activity in data.get("activities", {}).values()
        for session_id, row in activity.get("sessions", {}).items()
    }


def _load_fall_windows(path: Path) -> pd.DataFrame:
    data = np.load(path, allow_pickle=True)
    rows = []
    for i in range(len(data["session_id"])):
        rows.append(
            {
                "source_session_id": str(data["source_csv"][i]).split("/")[-1].replace(".csv", ""),
                "session_id": str(data["session_id"][i]),
                "window_start_s": float(data["window_start_s"][i]),
                "window_end_s": float(data["window_end_s"][i]),
                "event_phase": str(data["event_phase"][i]),
                "include_in_training": bool(data["include_in_training"][i]),
                "label": int(data["y"][i]),
                "exclude_reason": str(data["exclude_reason"][i]),
                "overlap_fraction": float(data["overlap_fraction"][i]),
                "quality_flags": str(data["quality_flags"][i]),
            }
        )
    return pd.DataFrame(rows)


def _load_test_probabilities(prob_dir: Path) -> pd.DataFrame:
    parts = []
    for fold_dir in sorted(prob_dir.glob("fold_*")):
        path = fold_dir / "test_window_probabilities.csv"
        if path.exists():
            parts.append(pd.read_csv(path))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _load_alerts(alert_dir: Path) -> pd.DataFrame:
    parts = []
    for fold_dir in sorted(alert_dir.glob("fold_*")):
        path = fold_dir / "test_alerts_budget_policy.csv"
        if path.exists():
            fold = int(fold_dir.name.split("_")[-1])
            df = pd.read_csv(path)
            df["fold"] = fold
            parts.append(df)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _raw_duration(path: Path) -> tuple[float, pd.DataFrame, np.ndarray]:
    df = read_csv_canonical(path)
    elapsed, _ = elapsed_axis(df, fallback_rate_hz=20.0)
    duration = float(np.nanmax(elapsed) - np.nanmin(elapsed)) if len(elapsed) else math.nan
    return duration, df, elapsed


def _motion_trace(df: pd.DataFrame) -> np.ndarray:
    cols = [c for c in ("x", "y", "z") if c in df.columns]
    if len(cols) != 3 or len(df) == 0:
        return np.asarray([], dtype=float)
    xyz = df[cols].to_numpy(dtype=float)
    return np.r_[0.0, np.linalg.norm(np.diff(xyz, axis=0), axis=1)]


def _priority_reason(row: dict[str, object]) -> tuple[tuple[int, int, int, int, int, int], str]:
    reasons = []
    missed = bool(row.get("false_negative", False))
    low_conf = str(row.get("auto_confidence", "")).lower() in {"low", "medium"}
    flags = str(row.get("quality_flags", ""))
    cleaning_warning = bool(row.get("flag_high_non_frozen_drop_fraction", False)) or float(row.get("non_frozen_drop_fraction", 0) or 0) >= 0.25
    boundary = "boundary" in flags
    few_windows = int(row.get("fall_event_windows_retained", 0) or 0) <= 1
    disagreement = int(row.get("fall_event_model_disagreement_windows", 0) or 0) > 0
    excluded = not bool(row.get("auto_include_in_training", True))
    if missed:
        reasons.append("missed fall in outer-test causal evaluation")
    if low_conf or flags not in {"", "none", "nan"}:
        reasons.append("low-confidence or warning-flagged auto annotation")
    if boundary:
        reasons.append("boundary-truncated event")
    if few_windows:
        reasons.append("very few retained fall-event windows")
    if disagreement:
        reasons.append("fall-event windows disagree with model prediction")
    if cleaning_warning:
        reasons.append("large cleaning/drop warning")
    if excluded:
        reasons.append("auto annotation excluded from training/evaluation")
    if not reasons:
        reasons.append("spot check auto timing")
    key = (
        0 if missed else 1,
        0 if (low_conf or flags not in {"", "none", "nan"}) else 1,
        0 if boundary else 1,
        0 if few_windows else 1,
        0 if disagreement else 1,
        0 if cleaning_warning or excluded else 1,
    )
    return key, "; ".join(reasons[:4])


def build_review_table(
    raw_fall_dir: Path,
    annotations_path: Path,
    manifest_path: Path,
    windowed_fall_path: Path,
    alert_dir: Path,
    prob_dir: Path,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    annotations = pd.read_csv(annotations_path)
    manifest = pd.read_csv(manifest_path)
    windows = _load_fall_windows(windowed_fall_path)
    alerts = _load_alerts(alert_dir)
    probs = _load_test_probabilities(prob_dir)
    cleaning = _read_cleaning_log(ROOT / "data/cleaning_log.json")
    rows = []
    raw_paths = sorted(p for p in raw_fall_dir.glob("*.csv") if p.is_file())
    for raw_path in raw_paths:
        sid = raw_path.stem
        ann_match = annotations[annotations["session_id"].astype(str) == sid]
        ann = ann_match.iloc[0].to_dict() if len(ann_match) else {}
        fold_match = manifest[manifest["source_session_id"].astype(str) == sid]
        fold = int(fold_match["outer_fold"].iloc[0]) if len(fold_match) else math.nan
        session_windows = windows[windows["source_session_id"].astype(str) == sid]
        retained_fall = session_windows[
            (session_windows["include_in_training"].astype(bool))
            & (session_windows["event_phase"].astype(str) == "fall_event")
            & (session_windows["label"].astype(int) == 3)
        ]
        session_probs = probs[probs["source_session_id"].astype(str) == sid] if len(probs) else pd.DataFrame()
        fall_event_probs = session_probs[session_probs["true_label"].astype(str) == "fall"] if len(session_probs) else pd.DataFrame()
        disagreement = int((fall_event_probs["predicted_label"].astype(str) != "fall").sum()) if len(fall_event_probs) else 0
        session_alerts = alerts[alerts["source_session_id"].astype(str) == sid] if len(alerts) else pd.DataFrame()
        event_start = float(ann.get("event_start_s", math.nan)) if ann else math.nan
        valid_alerts = (
            session_alerts[session_alerts["alert_time_s"].astype(float) >= event_start]
            if len(session_alerts) and np.isfinite(event_start)
            else pd.DataFrame()
        )
        first_alert = float(session_alerts["alert_time_s"].min()) if len(session_alerts) else math.nan
        first_valid_alert = float(valid_alerts["alert_time_s"].min()) if len(valid_alerts) else math.nan
        false_negative = bool(len(fold_match) and np.isfinite(event_start) and len(valid_alerts) == 0)
        duration, _, _ = _raw_duration(raw_path)
        clean = cleaning.get(f"fall__{sid}", {})
        row = {
            "source_session_id": sid,
            "raw_source_csv": str(raw_path),
            "cleaned_source_csv": f"data/cleaned_csv/{sid}.csv",
            "auto_event_start_s": event_start,
            "auto_impact_s": ann.get("impact_s", math.nan),
            "auto_event_end_s": ann.get("event_end_s", math.nan),
            "auto_confidence": ann.get("confidence", ""),
            "quality_flags": ann.get("quality_flags", ""),
            "auto_include_in_training": bool(ann.get("auto_include_in_training", False)) if ann else False,
            "outer_fold": fold,
            "causal_test_policy_alerted": bool(len(session_alerts) > 0),
            "causal_test_policy_alerted_after_event_start": bool(len(valid_alerts) > 0),
            "first_alert_time_s": first_alert,
            "first_event_alert_time_s": first_valid_alert,
            "alert_delay_s": first_valid_alert - event_start if np.isfinite(first_valid_alert) and np.isfinite(event_start) else math.nan,
            "false_negative": false_negative,
            "fall_event_windows_retained": int(len(retained_fall)),
            "fall_event_windows_total": int((session_windows["event_phase"].astype(str) == "fall_event").sum()) if len(session_windows) else 0,
            "excluded_windows": int((~session_windows["include_in_training"].astype(bool)).sum()) if len(session_windows) else 0,
            "recording_duration_s": duration,
            "non_frozen_drop_fraction": clean.get("non_frozen_drop_fraction", math.nan),
            "total_dropped_rows": clean.get("total_dropped_rows", math.nan),
            "out_of_range_dropped_rows": clean.get("out_of_range_dropped_rows", math.nan),
            "zscore_dropped_rows": clean.get("zscore_dropped_rows", math.nan),
            "elevation_gated_dropped_rows": clean.get("elevation_gated_dropped_rows", math.nan),
            "flag_high_non_frozen_drop_fraction": clean.get("flag_high_non_frozen_drop_fraction", False),
            "cleaning_warnings": ";".join(clean.get("warnings", [])) if isinstance(clean.get("warnings", []), list) else clean.get("warnings", ""),
            "distance_m_inferred": ann.get("distance_m_inferred", math.nan),
            "distance_band_inferred": ann.get("distance_band_inferred", ""),
            "direction_inferred": ann.get("direction_inferred", ""),
            "fall_event_model_disagreement_windows": disagreement,
            "fall_event_model_disagreement_fraction": float(disagreement / len(fall_event_probs)) if len(fall_event_probs) else math.nan,
            "max_fall_probability": float(session_probs["prob_fall"].max()) if len(session_probs) else math.nan,
            "mean_fall_event_probability": float(fall_event_probs["prob_fall"].mean()) if len(fall_event_probs) else math.nan,
        }
        priority_key, reason = _priority_reason(row)
        row["_priority_key"] = priority_key
        row["review_priority_reason"] = reason
        rows.append(row)
    review = pd.DataFrame(rows).sort_values("_priority_key", kind="stable").reset_index(drop=True)
    review.insert(0, "priority_rank", np.arange(1, len(review) + 1))
    review = review.drop(columns=["_priority_key"])
    return review, {"annotations": annotations, "manifest": manifest, "windows": windows, "alerts": alerts, "probabilities": probs}


def _plot_session(row: pd.Series, tables: dict[str, pd.DataFrame], output_dir: Path) -> None:
    sid = str(row["source_session_id"])
    raw_path = Path(str(row["raw_source_csv"]))
    df = read_csv_canonical(raw_path)
    t, _ = elapsed_axis(df, fallback_rate_hz=20.0)
    motion = _motion_trace(df)
    windows = tables["windows"][tables["windows"]["source_session_id"].astype(str) == sid]
    probs = tables["probabilities"][tables["probabilities"]["source_session_id"].astype(str) == sid]
    alerts = tables["alerts"][tables["alerts"]["source_session_id"].astype(str) == sid]

    fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True, gridspec_kw={"height_ratios": [1.2, 1.0, 1.0, 1.0]})
    if "range_m" in df.columns:
        axes[0].plot(t, df["range_m"], label="range_m", color="tab:blue")
        axes[0].set_ylabel("Range (m)")
    if "z" in df.columns:
        axes[1].plot(t, df["z"], label="z", color="tab:green")
        axes[1].set_ylabel("z")
    if len(motion):
        axes[2].plot(t, motion, label="xyz delta", color="tab:orange")
        axes[2].set_ylabel("Motion")
    if len(probs):
        axes[3].plot(probs["window_end"], probs["prob_fall"], marker="o", label="P(fall)", color="tab:red")
    axes[3].set_ylabel("P(fall)")
    axes[3].set_ylim(-0.03, 1.03)
    axes[3].set_xlabel("Elapsed time (s)")

    for ax in axes:
        for _, win in windows.iterrows():
            color = "seagreen" if bool(win["include_in_training"]) else "lightgray"
            alpha = 0.12 if bool(win["include_in_training"]) else 0.25
            ax.axvspan(float(win["window_start_s"]), float(win["window_end_s"]), color=color, alpha=alpha)
        for x, color, label in (
            (row["auto_event_start_s"], "purple", "event_start"),
            (row["auto_impact_s"], "black", "impact"),
            (row["auto_event_end_s"], "purple", "event_end"),
        ):
            if not pd.isna(x) and np.isfinite(float(x)):
                ax.axvline(float(x), color=color, linestyle="--" if label != "impact" else ":", linewidth=1.4)
        for _, alert in alerts.iterrows():
            ax.axvline(float(alert["alert_time_s"]), color="crimson", linestyle="-.", linewidth=1.2)
        ax.grid(alpha=0.25)
    axes[0].set_title(f"{sid} | rank {int(row['priority_rank'])}: {row['review_priority_reason']}")
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_dir / "plots" / f"{int(row['priority_rank']):02d}_{sid}.png", dpi=150)
    plt.close(fig)


def write_review_guide(output_dir: Path, review: pd.DataFrame) -> None:
    lines = [
        "# Manual Fall-Event Review Guide",
        "",
        "This package is read-only support material. It does not modify raw CSVs, cleaned data, datasets, manifests, models, checkpoints, or existing experiment outputs.",
        "",
        "Reviewer task:",
        "",
        "1. Open `fall_event_review_queue.csv` in priority order.",
        "2. For each session, inspect the matching plot in `plots/`.",
        "3. Confirm or correct only `event_start_s`, `impact_s`, and `event_end_s`.",
        "4. If the recording is ambiguous, truncated, or the person/event cannot be located confidently, mark it as `uncertain` or `exclude` in a separate manual annotation file.",
        "5. Do not use model predictions as ground truth. Treat P(fall) and alert markers only as cues for where the frozen model struggled.",
        "",
        "Plot legend:",
        "",
        "- Purple dashed lines: auto event start/end.",
        "- Black dotted line: auto impact time, when available.",
        "- Green shaded windows: retained event-aware windows.",
        "- Gray shaded windows: excluded windows.",
        "- Red probability trace: frozen SGKF4 P(fall) on that session's outer-test fold, when available.",
        "- Crimson vertical lines: budget-constrained causal alert times.",
        "",
        f"Package contains {len(review)} raw fall sessions.",
        f"Top priority count, missed causal falls: {int(review['false_negative'].sum())}.",
    ]
    (output_dir / "REVIEW_GUIDE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_manual_review_package(
    raw_fall_dir: Path = DEFAULT_RAW_FALL_DIR,
    annotations_path: Path = DEFAULT_ANNOTATIONS,
    manifest_path: Path = DEFAULT_MANIFEST,
    windowed_fall_path: Path = DEFAULT_WINDOWED_FALL,
    alert_dir: Path = DEFAULT_ALERT_DIR,
    prob_dir: Path = DEFAULT_PROB_DIR,
    output_dir: Path | None = None,
) -> dict[str, object]:
    if output_dir is None:
        output_dir = _unique_output_dir(DEFAULT_OUTPUT_PARENT, "manual_fall_event_review")
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    (output_dir / "plots").mkdir(parents=True, exist_ok=False)
    review, tables = build_review_table(raw_fall_dir, annotations_path, manifest_path, windowed_fall_path, alert_dir, prob_dir)
    review.to_csv(output_dir / "fall_event_review_queue.csv", index=False)
    template_cols = [
        "priority_rank",
        "source_session_id",
        "raw_source_csv",
        "auto_event_start_s",
        "auto_impact_s",
        "auto_event_end_s",
        "review_event_start_s",
        "review_impact_s",
        "review_event_end_s",
        "review_status",
        "review_notes",
    ]
    template = review[[c for c in template_cols if c in review.columns]].copy()
    for col in ("review_event_start_s", "review_impact_s", "review_event_end_s", "review_status", "review_notes"):
        if col not in template:
            template[col] = ""
    template[template_cols].to_csv(output_dir / "manual_review_template.csv", index=False)
    for _, row in review.iterrows():
        _plot_session(row, tables, output_dir)
    write_review_guide(output_dir, review)
    summary = {
        "output_dir": str(output_dir),
        "raw_fall_sessions": int(len(review)),
        "auto_include_in_training": int(review["auto_include_in_training"].sum()),
        "excluded_by_auto_annotation": int((~review["auto_include_in_training"].astype(bool)).sum()),
        "false_negative_outer_test_sessions": int(review["false_negative"].sum()),
        "low_or_medium_confidence": int(review["auto_confidence"].astype(str).isin(["low", "medium"]).sum()),
        "warning_flagged_annotations": int((~review["quality_flags"].astype(str).isin(["none", "", "nan"])).sum()),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-fall-dir", default=str(DEFAULT_RAW_FALL_DIR))
    parser.add_argument("--annotations", default=str(DEFAULT_ANNOTATIONS))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--windowed-fall", default=str(DEFAULT_WINDOWED_FALL))
    parser.add_argument("--alert-dir", default=str(DEFAULT_ALERT_DIR))
    parser.add_argument("--prob-dir", default=str(DEFAULT_PROB_DIR))
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)
    summary = run_manual_review_package(
        raw_fall_dir=Path(args.raw_fall_dir),
        annotations_path=Path(args.annotations),
        manifest_path=Path(args.manifest),
        windowed_fall_path=Path(args.windowed_fall),
        alert_dir=Path(args.alert_dir),
        prob_dir=Path(args.prob_dir),
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )
    print(f"saved_summary={Path(summary['output_dir']) / 'summary.json'}")


if __name__ == "__main__":
    main()
