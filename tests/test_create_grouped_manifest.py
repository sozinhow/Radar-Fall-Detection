from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from radar_pipeline.common import CLASS_NAMES
from radar_pipeline.create_grouped_manifest import (
    ManifestConfig,
    create_grouped_manifest,
    load_manifest_payload,
    verify_zero_leakage,
)
from radar_pipeline.evaluate_grouped_cv import GroupedCVConfig, validate_manifest as validate_training_manifest


def _write_fixture_dataset(path: Path, *, blank_source: bool = False, drop_source_key: bool = False) -> None:
    rows = []
    for label, name in enumerate(CLASS_NAMES):
        for session_num in range(12):
            sid = f"{name}_session_{session_num:02d}"
            for window_num in range(2):
                rows.append(
                    {
                        "X": np.full((6, 3), label + window_num / 10.0, dtype=np.float32),
                        "y": label,
                        "activity": name,
                        "source_activity": name,
                        "session_id": f"{name}__{sid}",
                        "source_csv": "" if blank_source and not rows else f"data/cleaned_csv/{sid}.csv",
                        "start_frame": window_num * 10,
                        "end_frame": window_num * 10 + 5,
                        "window_start_s": float(window_num),
                        "window_end_s": float(window_num + 1),
                    }
                )
    payload: dict[str, np.ndarray] = {
        "feature_names": np.asarray(["x", "y", "z"]),
        "label_names": np.asarray(CLASS_NAMES),
    }
    split_names = ["train", "val", "test"]
    split_indices = {
        "train": np.arange(0, len(rows), 3),
        "val": np.arange(1, len(rows), 3),
        "test": np.arange(2, len(rows), 3),
    }
    for split in split_names:
        idx = split_indices[split]
        split_rows = [rows[i] for i in idx]
        payload[f"X_{split}"] = np.stack([r["X"] for r in split_rows])
        payload[f"y_{split}"] = np.asarray([r["y"] for r in split_rows], dtype=np.int64)
        payload[f"activity_{split}"] = np.asarray([r["activity"] for r in split_rows])
        payload[f"source_activity_{split}"] = np.asarray([r["source_activity"] for r in split_rows])
        payload[f"session_id_{split}"] = np.asarray([r["session_id"] for r in split_rows])
        if not (drop_source_key and split == "train"):
            payload[f"source_csv_{split}"] = np.asarray([r["source_csv"] for r in split_rows])
        payload[f"start_frame_{split}"] = np.asarray([r["start_frame"] for r in split_rows], dtype=np.int64)
        payload[f"end_frame_{split}"] = np.asarray([r["end_frame"] for r in split_rows], dtype=np.int64)
        payload[f"window_start_s_{split}"] = np.asarray([r["window_start_s"] for r in split_rows], dtype=np.float32)
        payload[f"window_end_s_{split}"] = np.asarray([r["window_end_s"] for r in split_rows], dtype=np.float32)
    np.savez_compressed(path, **payload)


def test_manifest_generation_is_deterministic_for_same_dataset_and_seed(tmp_path: Path) -> None:
    dataset = tmp_path / "radar_dataset.npz"
    _write_fixture_dataset(dataset)
    first = tmp_path / "auto_event_aware_20260717_source_session_folds.csv"
    first_readme = tmp_path / "first_README.md"
    second = tmp_path / "copy_auto_event_aware_20260717_source_session_folds.csv"
    second_readme = tmp_path / "second_README.md"

    create_grouped_manifest(dataset, first, first_readme, ManifestConfig(folds=4, seed=42))
    create_grouped_manifest(dataset, second, second_readme, ManifestConfig(folds=4, seed=42))

    pd.testing.assert_frame_equal(pd.read_csv(first), pd.read_csv(second))
    assert "sgkf_grouped_20260717_seed42_k4" in first_readme.read_text(encoding="utf-8")


def test_all_source_sessions_represented_once_and_no_leakage(tmp_path: Path) -> None:
    dataset = tmp_path / "radar_dataset.npz"
    _write_fixture_dataset(dataset)
    output = tmp_path / "auto_event_aware_20260717_source_session_folds.csv"
    readme = tmp_path / "auto_event_aware_20260717_source_session_folds_README.md"

    manifest = create_grouped_manifest(dataset, output, readme, ManifestConfig(folds=4, seed=42))
    payload = load_manifest_payload(dataset)
    leakage = verify_zero_leakage(payload, manifest, ManifestConfig(folds=4, seed=42))

    expected = set(payload["source_session_id"].astype(str))
    assert set(manifest["source_session_id"].astype(str)) == expected
    assert manifest["source_session_id"].is_unique
    assert len(manifest) == len(expected)
    assert leakage[["train_val_leakage", "train_test_leakage", "val_test_leakage"]].to_numpy().sum() == 0
    validate_training_manifest(payload, manifest, GroupedCVConfig(folds=4, seed=42))


def test_missing_or_blank_session_metadata_fails(tmp_path: Path) -> None:
    blank_dataset = tmp_path / "blank_source.npz"
    _write_fixture_dataset(blank_dataset, blank_source=True)
    with pytest.raises(ValueError, match="missing source_csv"):
        load_manifest_payload(blank_dataset)

    missing_key_dataset = tmp_path / "missing_key.npz"
    _write_fixture_dataset(missing_key_dataset, drop_source_key=True)
    with pytest.raises(ValueError, match="missing required grouped-manifest metadata"):
        load_manifest_payload(missing_key_dataset)


def test_refuses_to_overwrite_outputs_by_default(tmp_path: Path) -> None:
    dataset = tmp_path / "radar_dataset.npz"
    _write_fixture_dataset(dataset)
    output = tmp_path / "auto_event_aware_20260717_source_session_folds.csv"
    readme = tmp_path / "auto_event_aware_20260717_source_session_folds_README.md"
    create_grouped_manifest(dataset, output, readme, ManifestConfig(folds=4, seed=42))

    with pytest.raises(FileExistsError, match="Manifest already exists"):
        create_grouped_manifest(dataset, output, tmp_path / "other_README.md", ManifestConfig(folds=4, seed=42))
    with pytest.raises(FileExistsError, match="README already exists"):
        create_grouped_manifest(dataset, tmp_path / "other.csv", readme, ManifestConfig(folds=4, seed=42))
