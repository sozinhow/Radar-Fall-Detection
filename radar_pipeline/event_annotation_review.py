from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from radar_pipeline.common import ROOT, discover_csv_files, estimated_height_m, load_config, read_csv_canonical


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


def _elapsed_seconds(df: pd.DataFrame, fallback_rate_hz: float) -> np.ndarray:
    if "timestamp" not in df.columns or df.empty:
        return np.arange(len(df), dtype=float) / fallback_rate_hz
    numeric = pd.to_numeric(df["timestamp"], errors="coerce")
    if numeric.notna().sum() >= 2:
        values = numeric.to_numpy(dtype=float)
        if np.nanmedian(np.diff(values[np.isfinite(values)])) > 10:
            values = values / 1000.0
        return values - np.nanmin(values)
    dt = pd.to_datetime(df["timestamp"], errors="coerce")
    if dt.notna().sum() >= 2:
        seconds = (dt - dt.dropna().iloc[0]).dt.total_seconds()
        fallback = pd.Series(np.arange(len(df), dtype=float) / fallback_rate_hz, index=df.index)
        return seconds.fillna(fallback).to_numpy(dtype=float)
    return np.arange(len(df), dtype=float) / fallback_rate_hz


def _motion_proxy(df: pd.DataFrame) -> pd.Series:
    if {"x", "y", "z"}.issubset(df.columns):
        delta = df[["x", "y", "z"]].diff()
        return np.sqrt((delta**2).sum(axis=1)).fillna(0.0)
    return pd.Series(np.zeros(len(df)), index=df.index)


def _session_id_for(path: Path, cleaned_df: pd.DataFrame | None) -> str:
    if cleaned_df is not None and "session_id" in cleaned_df.columns and len(cleaned_df):
        session_id = str(cleaned_df["session_id"].iloc[0])
        if "__" in session_id:
            return session_id.split("__", 1)[1]
        return session_id
    return path.stem


def _cleaning_warning(cleaning_log: dict[str, Any], session_key: str, high_drop_threshold: float) -> str:
    sessions = cleaning_log.get("activities", {}).get("fall", {}).get("sessions", {})
    candidates = [session_key, f"fall__{session_key}"]
    warnings: list[str] = []
    for candidate in candidates:
        item = sessions.get(candidate)
        if not item:
            continue
        drop = item.get("non_frozen_drop_fraction")
        if drop is not None and float(drop) >= high_drop_threshold:
            warnings.append(f"high_non_frozen_drop_fraction={float(drop):.1%}")
        if item.get("warnings"):
            warnings.extend(str(w) for w in item["warnings"])
        break
    return "; ".join(warnings)


def _plot_session(
    raw_df: pd.DataFrame,
    cleaned_df: pd.DataFrame | None,
    session_id: str,
    plot_path: Path,
    cfg: dict,
) -> list[str]:
    rate = float(cfg["sampling"]["expected_rate_hz"])
    mount_height = float(cfg["radar"]["mount_height_m"])
    review_df = cleaned_df if cleaned_df is not None and len(cleaned_df) else raw_df
    warnings: list[str] = []

    required = ["x", "y", "z", "range_m", "elevation_deg"]
    missing = [col for col in required if col not in review_df.columns]
    if missing:
        warnings.append(f"missing_required_columns={','.join(missing)}")

    t = _elapsed_seconds(review_df, rate)
    raw_t = _elapsed_seconds(raw_df, rate)
    signals: list[tuple[str, pd.Series | np.ndarray]] = []
    for col in ["x", "y", "z", "range_m", "elevation_deg"]:
        if col in review_df.columns:
            signals.append((col, pd.to_numeric(review_df[col], errors="coerce")))
    if {"range_m", "elevation_deg"}.issubset(review_df.columns):
        signals.append(("estimated_height_m", estimated_height_m(review_df["range_m"], review_df["elevation_deg"], mount_height)))
    signals.append(("xyz_delta_mag", _motion_proxy(review_df)))

    if not signals:
        signals.append(("row_index", np.arange(len(review_df), dtype=float)))

    fig, axes = plt.subplots(len(signals), 1, figsize=(14, max(7, 2.0 * len(signals))), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (name, values) in zip(axes, signals):
        if name in raw_df.columns:
            raw_values = pd.to_numeric(raw_df[name], errors="coerce")
            ax.plot(raw_t, raw_values, color="0.78", linewidth=0.8, label=f"raw {name}")
        ax.plot(t, values, color="tab:blue", linewidth=1.1, label=f"cleaned {name}" if cleaned_df is not None else name)
        ax.set_ylabel(name)
        ax.grid(alpha=0.25)

    win_len = max(1, int(round(float(cfg["windowing"]["window_len_sec"]) * rate)))
    step = max(1, int(round(win_len * (1.0 - float(cfg["windowing"]["overlap_pct"])))))
    window_rows: list[tuple[float, float]] = []
    for start in range(0, max(len(review_df) - win_len + 1, 0), step):
        end = start + win_len
        start_s = start / rate
        end_s = end / rate
        window_rows.append((start_s, end_s))
        for ax in axes:
            ax.axvspan(start_s, end_s, color="tab:orange", alpha=0.08)
            ax.axvline(start_s, color="tab:orange", alpha=0.22, linewidth=0.6)
    for ax in axes:
        ax.legend(loc="upper right", fontsize=8)

    if len(window_rows) == 0:
        warnings.append("too_short_for_one_window")

    duration_s = float(t[-1]) if len(t) else 0.0
    axes[-1].set_xlabel("Elapsed time (s)")
    fig.suptitle(
        f"Fall event annotation review: {session_id}\n"
        f"Duration={duration_s:.2f}s, frames={len(review_df)}, windows={len(window_rows)}; "
        "orange spans show 30-frame windows"
    )
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)
    return warnings


def generate_event_annotation_review(
    raw_fall_dir: Path,
    cleaned_fall_dir: Path | None,
    output_dir: Path,
    template_path: Path,
    cfg: dict,
    cleaning_log_path: Path | None = None,
) -> dict[str, Any]:
    raw_paths = discover_csv_files(raw_fall_dir)
    cleaned_lookup = {path.stem: path for path in discover_csv_files(cleaned_fall_dir)} if cleaned_fall_dir else {}
    output_dir.mkdir(parents=True, exist_ok=True)
    template_path.parent.mkdir(parents=True, exist_ok=True)
    cleaning_log: dict[str, Any] = {}
    if cleaning_log_path and cleaning_log_path.exists():
        with cleaning_log_path.open("r", encoding="utf-8") as f:
            cleaning_log = json.load(f)

    rows: list[dict[str, Any]] = []
    template_rows: list[dict[str, Any]] = []
    high_drop_threshold = 0.20

    for raw_path in raw_paths:
        raw_df = read_csv_canonical(raw_path)
        cleaned_path = cleaned_lookup.get(raw_path.stem) if cleaned_fall_dir else None
        cleaned_df = read_csv_canonical(cleaned_path) if cleaned_path and cleaned_path.exists() else None
        session_id = _session_id_for(raw_path, cleaned_df)
        review_df = cleaned_df if cleaned_df is not None and len(cleaned_df) else raw_df
        rate = float(cfg["sampling"]["expected_rate_hz"])
        duration_s = len(review_df) / rate
        plot_path = output_dir / f"{session_id}_review.png"
        warnings = _plot_session(raw_df, cleaned_df, session_id, plot_path, cfg)
        cleaning_warning = _cleaning_warning(cleaning_log, session_id, high_drop_threshold)
        if cleaning_warning:
            warnings.append(cleaning_warning)

        row = {
            "session_id": session_id,
            "raw_csv": str(raw_path),
            "cleaned_csv": str(cleaned_path) if cleaned_path and cleaned_path.exists() else "",
            "duration_s": round(float(duration_s), 3),
            "raw_frame_count": int(len(raw_df)),
            "cleaned_frame_count": int(len(cleaned_df)) if cleaned_df is not None else "",
            "window_len_frames": int(round(float(cfg["windowing"]["window_len_sec"]) * rate)),
            "plot_path": str(plot_path),
            "warnings": "; ".join(warnings),
            "start_state": "",
            "event_start_s": "",
            "impact_s": "",
            "event_end_s": "",
            "end_state": "",
            "distance_band": "",
            "distance_m": "",
            "direction": "",
            "notes": "",
        }
        rows.append(row)
        template_rows.append({col: "" for col in ANNOTATION_COLUMNS})
        template_rows[-1]["session_id"] = session_id
        template_rows[-1]["activity"] = "fall"

    index_path = output_dir / "index.csv"
    pd.DataFrame(rows).to_csv(index_path, index=False)
    pd.DataFrame(template_rows, columns=ANNOTATION_COLUMNS).to_csv(template_path, index=False)
    return {
        "sessions_reviewed": len(rows),
        "plot_dir": str(output_dir),
        "index_csv": str(index_path),
        "template_csv": str(template_path),
        "warnings": {row["session_id"]: row["warnings"] for row in rows if row["warnings"]},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-fall-dir", default=str(ROOT / "data/raw_csv/fall"))
    parser.add_argument("--cleaned-fall-dir", default=str(ROOT / "data/staging/20260731_rebuild01/cleaned_csv"))
    parser.add_argument("--output-dir", default=str(ROOT / "outputs/validation/event_annotation_review/fall"))
    parser.add_argument(
        "--template-path",
        default=str(ROOT / "outputs/validation/event_annotation_review/session_annotations_template.csv"),
    )
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    parser.add_argument("--cleaning-log", default=str(ROOT / "data/staging/20260731_rebuild01/cleaning_log.json"))
    args = parser.parse_args()
    summary = generate_event_annotation_review(
        Path(args.raw_fall_dir),
        Path(args.cleaned_fall_dir),
        Path(args.output_dir),
        Path(args.template_path),
        load_config(args.config),
        Path(args.cleaning_log),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
