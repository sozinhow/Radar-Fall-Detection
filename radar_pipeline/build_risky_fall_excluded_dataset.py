"""Build the versioned risky-fall-excluded event-aware dataset.

This is the reproducible replacement for the one-off historical filtering
step.  It deliberately reuses :mod:`radar_pipeline.windowing` rather than
post-hoc editing an NPZ: the exclusion is made while event windows are labelled
and is therefore captured in the window log and the final dataset index.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from radar_pipeline.common import ROOT, load_config, write_json
from radar_pipeline.create_grouped_manifest import ManifestConfig, create_grouped_manifest
from radar_pipeline.dataset_builder import build_dataset
from radar_pipeline.windowing import AMBIGUOUS_FALL_WEAKNESS_FLAGS, window_directory


FILTER_REASON = "ambiguous_auto_fall_window"


def _require_new_directory(path: Path, label: str) -> None:
    """Refuse both existing directories and existing non-directory targets."""

    if path.exists():
        raise FileExistsError(f"{label} already exists: {path}. Use a new dated output path.")


def _risky_rows(window_dir: Path) -> pd.DataFrame:
    """Return an audit table for exactly the windows excluded by the rule."""

    rows: list[dict[str, object]] = []
    for path in sorted(window_dir.glob("*_windows.npz")):
        data = np.load(path, allow_pickle=True)
        reasons = np.asarray(data["exclude_reason"]).astype(str)
        for index in np.flatnonzero(reasons == FILTER_REASON):
            rows.append(
                {
                    "source_window_file": str(path),
                    "source_session_id": str(np.asarray(data["session_id"])[index]),
                    "source_csv": str(np.asarray(data["source_csv"])[index]),
                    "label": int(np.asarray(data["y"])[index]),
                    "event_phase": str(np.asarray(data["event_phase"])[index]),
                    "start_frame": int(np.asarray(data["start_frame"])[index]),
                    "end_frame": int(np.asarray(data["end_frame"])[index]),
                    "window_start_s": float(np.asarray(data["window_start_s"])[index]),
                    "window_end_s": float(np.asarray(data["window_end_s"])[index]),
                    "annotation_confidence": str(np.asarray(data["annotation_confidence"])[index]),
                    "quality_flags": str(np.asarray(data["quality_flags"])[index]),
                    "method_version": str(np.asarray(data["method_version"])[index]),
                    "exclude_reason": FILTER_REASON,
                }
            )
    columns = [
        "source_window_file",
        "source_session_id",
        "source_csv",
        "label",
        "event_phase",
        "start_frame",
        "end_frame",
        "window_start_s",
        "window_end_s",
        "annotation_confidence",
        "quality_flags",
        "method_version",
        "exclude_reason",
    ]
    return pd.DataFrame(rows, columns=columns)


def build_risky_fall_excluded_dataset(
    *,
    cleaned_dir: Path,
    annotations_csv: Path,
    windowed_output_dir: Path,
    dataset_output_dir: Path,
    config_path: Path,
    manifest_output: Path | None = None,
    manifest_readme: Path | None = None,
    seed: int = 42,
    folds: int = 4,
) -> dict[str, object]:
    """Build an event-aware dataset with the documented risky-fall exclusion.

    A risky window is an *auto-annotated fall-event window* whose annotation
    contains ``geometry_edge_warning`` and at least one of the weakness flags
    implemented by ``windowing._is_ambiguous_auto_fall``.  No physical-speed
    interpretation or threshold is applied to ``dop_idx``.
    """

    if not cleaned_dir.is_dir():
        raise FileNotFoundError(f"Cleaned CSV directory does not exist: {cleaned_dir}")
    if not annotations_csv.is_file():
        raise FileNotFoundError(f"Auto-event annotation CSV does not exist: {annotations_csv}")
    if manifest_output is None and manifest_readme is not None:
        raise ValueError("--manifest-readme requires --manifest-output")
    if manifest_output is not None and manifest_readme is None:
        raise ValueError("--manifest-output requires --manifest-readme")

    _require_new_directory(windowed_output_dir, "Windowed output directory")
    _require_new_directory(dataset_output_dir, "Dataset output directory")
    if manifest_output is not None:
        if manifest_output.exists():
            raise FileExistsError(f"Manifest already exists: {manifest_output}. Use a new dated output path.")
        if manifest_readme is not None and manifest_readme.exists():
            raise FileExistsError(f"Manifest README already exists: {manifest_readme}. Use a new dated output path.")

    cfg = copy.deepcopy(load_config(config_path))
    cfg.setdefault("windowing", {}).setdefault("event_aware", {})
    cfg["windowing"]["event_aware"].update(
        {
            "enabled": True,
            "metadata_csv": str(annotations_csv),
            "exclude_post_event": True,
            "exclude_transition": True,
        }
    )
    cfg.setdefault("auto_event_annotation", {})["exclude_ambiguous_fall_windows"] = True

    window_summary = window_directory(cleaned_dir, windowed_output_dir, cfg)
    removed = _risky_rows(windowed_output_dir)
    dataset_summary = build_dataset(windowed_output_dir, dataset_output_dir, cfg)
    removed_path = dataset_output_dir / "removed_risky_fall_windows.csv"
    removed.to_csv(removed_path, index=False)

    provenance = {
        "cleaned_dir": str(cleaned_dir),
        "annotations_csv": str(annotations_csv),
        "windowed_output_dir": str(windowed_output_dir),
        "dataset_output_dir": str(dataset_output_dir),
        "config_path": str(config_path),
        "filter_reason": FILTER_REASON,
        "rule": {
            "annotation_kind": "auto_event_annotations",
            "event_phase": "fall_event",
            "required_quality_flag": "geometry_edge_warning",
            "one_or_more_weakness_flags": sorted(AMBIGUOUS_FALL_WEAKNESS_FLAGS),
        },
        "removed_window_count": int(len(removed)),
        "dop_idx_semantics": "integer Doppler-related classification feature; no m/s threshold applied",
        "window_summary": window_summary,
        "dataset_summary": dataset_summary,
    }
    write_json(dataset_output_dir / "risky_fall_exclusion_provenance.json", provenance)

    if manifest_output is not None and manifest_readme is not None:
        manifest = create_grouped_manifest(
            dataset_path=dataset_output_dir / "radar_dataset.npz",
            output_path=manifest_output,
            readme_path=manifest_readme,
            cfg=ManifestConfig(folds=folds, seed=seed),
        )
        provenance["manifest_output"] = str(manifest_output)
        provenance["manifest_readme"] = str(manifest_readme)
        provenance["manifest_source_sessions"] = int(len(manifest))
        write_json(dataset_output_dir / "risky_fall_exclusion_provenance.json", provenance)

    return provenance


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build a new event-aware dataset with documented risky auto-fall windows excluded."
    )
    parser.add_argument("--cleaned-dir", required=True, help="Flat cleaned CSV directory from clean_data.")
    parser.add_argument("--annotations-csv", required=True, help="Auto-event annotation CSV to use.")
    parser.add_argument("--windowed-output-dir", required=True, help="New output directory for event-aware window NPZ files.")
    parser.add_argument("--dataset-output-dir", required=True, help="New output directory for radar_dataset.npz and audits.")
    parser.add_argument("--config", default=str(Path(__file__).with_name("config_event_aware_auto_20260716.yaml")))
    parser.add_argument("--manifest-output", help="Optional new frozen SGKF manifest CSV path.")
    parser.add_argument("--manifest-readme", help="Optional new frozen SGKF manifest README path.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=4)
    args = parser.parse_args(list(argv) if argv is not None else None)
    result = build_risky_fall_excluded_dataset(
        cleaned_dir=Path(args.cleaned_dir),
        annotations_csv=Path(args.annotations_csv),
        windowed_output_dir=Path(args.windowed_output_dir),
        dataset_output_dir=Path(args.dataset_output_dir),
        config_path=Path(args.config),
        manifest_output=Path(args.manifest_output) if args.manifest_output else None,
        manifest_readme=Path(args.manifest_readme) if args.manifest_readme else None,
        seed=args.seed,
        folds=args.folds,
    )
    print(json.dumps({"removed_risky_fall_windows": result["removed_window_count"]}, sort_keys=True))


if __name__ == "__main__":
    main()
