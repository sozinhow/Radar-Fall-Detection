from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from radar_pipeline.common import (
    MODEL_FEATURES,
    ROOT,
    activity_from_path,
    discover_csv_files,
    elapsed_axis,
    ensure_dirs,
    model_feature_columns,
    read_csv_canonical,
)


EXPECTED_CURRENT_ACTIVITIES = {"sitting", "standing", "walking", "fall"}


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


UNITS = {
    "x": "m",
    "y": "m",
    "z": "m",
    "dop_idx": "index",
    "range_m": "m",
    "azimuth_deg": "deg",
    "elevation_deg": "deg",
}


def _clean_stage_dir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in out_dir.iterdir():
        if path.is_dir():
            shutil.rmtree(path)
        elif path.suffix.lower() == ".png":
            path.unlink()
    for path in out_dir.glob("*.png"):
        path.unlink()


def plot_activity(df: pd.DataFrame, session_id: str, out_dir: Path, activity: str | None = None) -> None:
    cols = model_feature_columns(df)
    if not cols:
        return
    x, xlabel = elapsed_axis(df)
    out_path = out_dir / (activity or "unknown") / f"{session_id}.png"

    fig, axes = plt.subplots(len(cols), 1, figsize=(10, max(3, 2.2 * len(cols))), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, col in zip(axes, cols):
        ax.plot(x, df[col], linewidth=1)
        ax.set_ylabel(f"{col} ({UNITS.get(col, 'value')})")
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel(xlabel)
    missing = [col for col in MODEL_FEATURES if col not in cols]
    subtitle = f"; missing: {', '.join(missing)}" if missing else ""
    fig.suptitle(f"{activity or 'unknown'} / {session_id}: model feature time series{subtitle}")
    _save(fig, out_path)


def overlay_plot(frames: dict[str, pd.DataFrame], out_dir: Path, suffix: str = "overlay") -> None:
    common = [col for col in MODEL_FEATURES if frames and all(col in model_feature_columns(df) for df in frames.values())]
    if not common:
        return
    fig, axes = plt.subplots(len(common), 1, figsize=(10, max(3, 2.2 * len(common))), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, col in zip(axes, common):
        for activity, df in frames.items():
            if len(df) < 2:
                continue
            x = np.linspace(0, 1, len(df))
            y = df[col]
            y = (y - y.mean()) / (y.std() or 1.0)
            ax.plot(x, y, linewidth=1, label=activity)
        ax.set_ylabel(col)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("Normalized duration")
    axes[0].legend()
    fig.suptitle("Activity overlay comparison")
    _save(fig, out_dir / f"all_activities_{suffix}.png")


def inspect_dir(input_dir: Path, out_dir: Path, write_aggregate: bool = True) -> dict[str, pd.DataFrame]:
    ensure_dirs()
    _clean_stage_dir(out_dir)
    activity_frames: dict[str, list[pd.DataFrame]] = {}
    for csv_path in discover_csv_files(input_dir):
        activity = activity_from_path(csv_path)
        df = read_csv_canonical(csv_path)
        activity_frames.setdefault(activity, []).append(df)
        plot_activity(df, csv_path.stem, out_dir, activity)
    frames = {activity: pd.concat(items, ignore_index=True) for activity, items in activity_frames.items()}
    # Batch plotting is often run on a single activity. Never let such a
    # partial run overwrite the canonical all-activities aggregate image.
    if write_aggregate and EXPECTED_CURRENT_ACTIVITIES.issubset(frames):
        overlay_plot(frames, out_dir)
    return frames


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=str(ROOT / "data/raw_csv"))
    parser.add_argument("--output-dir", default=str(ROOT / "outputs/validation/before_cleaning"))
    args = parser.parse_args()
    frames = inspect_dir(Path(args.input_dir), Path(args.output_dir))
    print(f"Generated inspection plots for {len(frames)} activities.")


if __name__ == "__main__":
    main()
