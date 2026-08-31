from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from radar_pipeline.clean_data import clean_directory
from radar_pipeline.common import load_config
from radar_pipeline.auto_event_annotations import infer_session_annotation
from radar_pipeline.auto_event_experiment import feasible_folds, make_group_folds
from radar_pipeline.dataset_builder import build_dataset
from radar_pipeline.event_annotation_review import generate_event_annotation_review
from radar_pipeline.train_model import (
    TrainConfig,
    confirm_temporal_fall_predictions,
    select_temporal_confirmation,
    train,
    validation_checkpoint_metrics,
)
from radar_pipeline.windowing import window_directory


def _sample(activity: str, velocity: float) -> pd.DataFrame:
    n = 80
    t = np.arange(n) / 20.0
    return pd.DataFrame(
        {
            "timestamp": t,
            "range_m": 2.0 + 0.05 * np.sin(t),
            "velocity_mps": velocity + 0.02 * np.sin(t * 3),
            "azimuth_deg": 5.0 * np.sin(t),
            "elevation_deg": -12.0 + 2.0 * np.cos(t),
            "energy": 10.0 + np.cos(t),
            "activity": activity,
        }
    )


def _event_cfg(tmp_path: Path, metadata: Path, threshold: float = 0.50) -> dict:
    cfg = load_config()
    cfg["sampling"]["expected_rate_hz"] = 20
    cfg["windowing"]["window_len_sec"] = 1.0
    cfg["windowing"]["overlap_pct"] = 0.5
    cfg["windowing"]["event_aware"] = {
        "enabled": True,
        "metadata_csv": str(metadata),
        "fall_overlap_threshold": threshold,
        "exclude_post_event": True,
        "exclude_transition": True,
    }
    return cfg


def _clean_like_sample(activity: str, n: int = 100) -> pd.DataFrame:
    t = np.arange(n) / 20.0
    return pd.DataFrame(
        {
            "timestamp": t,
            "x": 0.1 * np.sin(t),
            "y": 2.0 + 0.05 * np.cos(t),
            "z": 0.6 + 0.1 * np.sin(t * 2),
            "dop_idx": 1.0 + 0.1 * np.cos(t),
            "range_m": 2.0 + 0.02 * np.sin(t),
            "azimuth_deg": 3.0 * np.sin(t),
            "elevation_deg": -10.0 + np.cos(t),
            "activity": activity,
        }
    )


def _write_metadata(path: Path, rows: list[dict]) -> None:
    columns = [
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
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)


def test_temporal_fall_confirmation_is_session_local_and_validation_selected() -> None:
    y_pred = np.asarray([3, 0, 3, 3, 3, 3])
    y_prob = np.asarray(
        [
            [0.35, 0.05, 0.10, 0.50],
            [0.80, 0.05, 0.05, 0.10],
            [0.10, 0.05, 0.05, 0.80],
            [0.10, 0.05, 0.05, 0.80],
            [0.20, 0.10, 0.15, 0.55],
            [0.15, 0.10, 0.20, 0.55],
        ]
    )
    sessions = np.asarray(["a", "a", "a", "a", "b", "b"])
    starts = np.asarray([0, 15, 30, 45, 0, 30])
    ends = np.asarray([30, 45, 60, 75, 30, 60])

    confirmed = confirm_temporal_fall_predictions(y_pred, y_prob, sessions, starts, ends)
    assert confirmed.tolist() == [0, 0, 3, 3, 0, 2]

    selected = select_temporal_confirmation(
        np.asarray([0, 3, 3]),
        np.asarray([3, 3, 3]),
        np.asarray([0, 3, 3]),
    )
    assert selected["selected"] is True
    assert selected["temporal_confirmation"]["fall_recall"] == selected["baseline"]["fall_recall"]

    rejected = select_temporal_confirmation(
        np.asarray([0, 3, 3]),
        np.asarray([0, 3, 3]),
        np.asarray([0, 0, 3]),
    )
    assert rejected["selected"] is False


def test_validation_checkpoint_metrics_are_explicit_and_from_best_checkpoint() -> None:
    report = {
        "fall": {"precision": 0.75, "recall": 0.60, "f1-score": 2 * 0.75 * 0.60 / (0.75 + 0.60)}
    }
    metrics = validation_checkpoint_metrics(
        {
            "best_epoch": 7,
            "best_val_loss": 0.42,
            "best_val_accuracy": 0.81,
            "best_train_loss": 0.31,
            "best_train_accuracy": 0.89,
            "validation_report": report,
            "validation_confusion_matrix": [[1, 0, 0, 0]] * 4,
        }
    )

    assert metrics["selection_metric"] == "minimum_validation_loss"
    assert metrics["best_epoch"] == 7
    assert metrics["best_val_loss"] == 0.42
    assert metrics["accuracy"] == 0.81
    assert metrics["fall_precision"] == 0.75
    assert metrics["fall_recall"] == 0.60
    assert metrics["fall_f1"] == report["fall"]["f1-score"]


def test_training_reporting_serializes_consistent_best_validation_metrics(tmp_path: Path, monkeypatch) -> None:
    import radar_pipeline.train_model as train_module

    class DummyModel:
        def state_dict(self) -> dict:
            return {"weight": torch.tensor([1.0])}

    validation_report = {
        "fall": {"precision": 0.75, "recall": 0.60, "f1-score": 2 * 0.75 * 0.60 / (0.75 + 0.60)}
    }
    fit = {
        "model": DummyModel(),
        "device": torch.device("cpu"),
        "data": {"X_test": np.zeros((2, 3, 1), dtype=np.float32), "y_test": np.asarray([0, 3])},
        "dataset_info": {"feature_names": ["x"], "path": "synthetic"},
        "weights": torch.ones(4),
        "loss_fn": object(),
        "architecture_summary": [],
        "history": [{"epoch": 7, "train_loss": 0.31, "train_acc": 0.89, "val_loss": 0.42, "val_acc": 0.81}],
        "best_epoch": 7,
        "best_train_loss": 0.31,
        "best_train_accuracy": 0.89,
        "best_val_loss": 0.42,
        "best_val_accuracy": 0.81,
        "validation_report": validation_report,
        "validation_confusion_matrix": [[1, 0, 0, 0]] * 4,
    }
    checkpoint_payload: dict = {}
    metrics_payload: dict = {}

    monkeypatch.setattr(train_module, "fit_validation_only", lambda *_args, **_kwargs: fit)
    monkeypatch.setattr(train_module, "make_loader", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        train_module,
        "evaluate",
        lambda *_args, **_kwargs: (
            0.5,
            0.5,
            np.asarray([0, 3]),
            np.asarray([0, 3]),
            np.asarray([[0.9, 0.05, 0.03, 0.02], [0.02, 0.03, 0.05, 0.9]]),
        ),
    )
    monkeypatch.setattr(train_module, "plot_history", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(train_module.torch, "save", lambda payload, _path: checkpoint_payload.update(payload))

    def capture_metrics(_path: Path, text: str, **_kwargs) -> int:
        metrics_payload.update(json.loads(text))
        return len(text)

    monkeypatch.setattr(Path, "write_text", capture_metrics)
    returned = train(
        tmp_path / "unused_dataset.npz",
        tmp_path / "models",
        TrainConfig(feature_mode="raw"),
        detailed_evaluation_outputs=False,
    )

    for payload in (checkpoint_payload, metrics_payload, returned):
        validation = payload["validation_metrics"]
        assert payload["best_epoch"] == validation["best_epoch"] == 7
        assert payload["best_val_loss"] == validation["best_val_loss"] == 0.42
        assert validation["selection_metric"] == "minimum_validation_loss"
        for field in ("accuracy", "fall_precision", "fall_recall", "fall_f1"):
            assert isinstance(validation[field], (int, float))
            assert np.isfinite(validation[field])
        assert np.isfinite(payload["best_epoch"])
        assert np.isfinite(payload["best_val_loss"])


def test_synthetic_pipeline_end_to_end(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    clean = tmp_path / "clean"
    windows = tmp_path / "windows"
    final = tmp_path / "final"
    raw.mkdir()
    for activity, velocity in {"walking": 1.0, "standing": 0.05, "sitting": 0.0, "fall": -0.8}.items():
        _sample(activity, velocity).to_csv(raw / f"{activity}.csv", index=False)

    cfg = load_config()
    clean_log = clean_directory(raw, clean, cfg)
    assert set(clean_log["activities"]) == {"walking", "standing", "sitting", "fall"}

    window_log = window_directory(clean, windows, cfg)
    assert all(item["windows"] > 0 for item in window_log["activities"].values())

    stats = build_dataset(windows, final, cfg)
    assert stats["total_windows"] > 0
    data = np.load(final / "radar_dataset.npz")
    assert data["X_train"].ndim == 3
    assert data["X_train"].shape[-1] > 0
    assert data["label_names"].tolist() == ["walking", "standing", "sitting", "fall"]

    walking_mean = pd.read_csv(clean / "walking.csv")["velocity_mps"].mean()
    sitting_mean = pd.read_csv(clean / "sitting.csv")["velocity_mps"].mean()
    assert walking_mean > sitting_mean


def test_event_aware_disabled_preserves_legacy_labels(tmp_path: Path) -> None:
    clean = tmp_path / "clean"
    legacy = tmp_path / "legacy"
    staged = tmp_path / "staged"
    clean.mkdir()
    _clean_like_sample("fall").to_csv(clean / "fall_case.csv", index=False)

    cfg = load_config()
    cfg["windowing"]["window_len_sec"] = 1.0
    cfg["windowing"]["overlap_pct"] = 0.5
    legacy_log = window_directory(clean, legacy, cfg)

    cfg_with_disabled_metadata = load_config()
    cfg_with_disabled_metadata["windowing"]["window_len_sec"] = 1.0
    cfg_with_disabled_metadata["windowing"]["overlap_pct"] = 0.5
    cfg_with_disabled_metadata["windowing"]["event_aware"] = {
        "enabled": False,
        "metadata_csv": str(tmp_path / "missing.csv"),
        "fall_overlap_threshold": 0.50,
    }
    staged_log = window_directory(clean, staged, cfg_with_disabled_metadata)

    y_legacy = np.load(legacy / "fall_windows.npz")["y"]
    y_staged = np.load(staged / "fall_windows.npz")["y"]
    assert np.array_equal(y_legacy, y_staged)
    assert legacy_log["activities"]["fall"]["windows"] == staged_log["activities"]["fall"]["windows"]


def test_event_aware_fall_metadata_labels_and_excludes_phases(tmp_path: Path) -> None:
    clean = tmp_path / "clean"
    windows = tmp_path / "windows"
    metadata = tmp_path / "annotations.csv"
    clean.mkdir()
    _clean_like_sample("fall").to_csv(clean / "fall_case.csv", index=False)
    _write_metadata(
        metadata,
        [
            {
                "session_id": "fall_case",
                "activity": "fall",
                "start_state": "standing",
                "event_start_s": 2.0,
                "impact_s": 2.5,
                "event_end_s": 3.0,
                "end_state": "lying",
                "distance_band": "mid",
                "distance_m": 2.5,
                "direction": "forward",
                "notes": "unit test",
            }
        ],
    )

    log = window_directory(clean, windows, _event_cfg(tmp_path, metadata))
    data = np.load(windows / "fall_windows.npz", allow_pickle=True)
    phases = data["event_phase"].astype(str).tolist()
    labels = data["y"].tolist()
    include = data["include_in_training"].tolist()

    assert log["activities"]["fall"]["event_phase_counts"] == {
        "post_event": 3,
        "pre_event": 3,
        "fall_event": 3,
    }
    assert phases[:3] == ["pre_event", "pre_event", "pre_event"]
    assert labels[:3] == [1, 1, 1]
    assert include[:6] == [True, True, True, True, True, True]
    assert phases[3:6] == ["fall_event", "fall_event", "fall_event"]
    assert labels[3:6] == [3, 3, 3]
    assert phases[6:] == ["post_event", "post_event", "post_event"]
    assert include[6:] == [False, False, False]


def test_event_overlap_below_threshold_is_excluded(tmp_path: Path) -> None:
    clean = tmp_path / "clean"
    windows = tmp_path / "windows"
    metadata = tmp_path / "annotations.csv"
    clean.mkdir()
    _clean_like_sample("fall").to_csv(clean / "fall_case.csv", index=False)
    _write_metadata(
        metadata,
        [
            {
                "session_id": "fall_case",
                "activity": "fall",
                "start_state": "standing",
                "event_start_s": 2.0,
                "impact_s": 2.5,
                "event_end_s": 3.0,
                "end_state": "lying",
                "distance_band": "mid",
                "distance_m": 2.5,
                "direction": "forward",
                "notes": "unit test",
            }
        ],
    )

    window_directory(clean, windows, _event_cfg(tmp_path, metadata, threshold=0.75))
    data = np.load(windows / "fall_windows.npz", allow_pickle=True)
    phases = data["event_phase"].astype(str).tolist()
    assert phases[3:6] == ["transition_excluded", "fall_event", "transition_excluded"]
    assert data["y"][3] == -1
    assert data["include_in_training"][3] == np.False_


def test_invalid_event_annotation_fails_clearly(tmp_path: Path) -> None:
    clean = tmp_path / "clean"
    windows = tmp_path / "windows"
    metadata = tmp_path / "annotations.csv"
    clean.mkdir()
    _clean_like_sample("fall").to_csv(clean / "fall_case.csv", index=False)
    _write_metadata(
        metadata,
        [
            {
                "session_id": "fall_case",
                "activity": "fall",
                "start_state": "standing",
                "event_start_s": 3.0,
                "impact_s": 2.5,
                "event_end_s": 2.0,
                "end_state": "lying",
                "distance_band": "mid",
                "distance_m": 2.5,
                "direction": "forward",
                "notes": "bad ordering",
            }
        ],
    )

    try:
        window_directory(clean, windows, _event_cfg(tmp_path, metadata))
    except ValueError as exc:
        assert "invalid time ordering" in str(exc)
    else:
        raise AssertionError("invalid annotation did not fail")


def test_dataset_builder_excludes_event_aware_excluded_windows_and_preserves_session_grouping(tmp_path: Path) -> None:
    clean = tmp_path / "clean"
    windows = tmp_path / "windows"
    final = tmp_path / "final"
    metadata = tmp_path / "annotations.csv"
    clean.mkdir()
    for session in ("fall_case_a", "fall_case_b"):
        _clean_like_sample("fall").to_csv(clean / f"{session}.csv", index=False)
    rows = []
    for session in ("fall_case_a", "fall_case_b"):
        rows.append(
            {
                "session_id": session,
                "activity": "fall",
                "start_state": "standing",
                "event_start_s": 2.0,
                "impact_s": 2.5,
                "event_end_s": 3.0,
                "end_state": "lying",
                "distance_band": "mid",
                "distance_m": 2.5,
                "direction": "forward",
                "notes": "unit test",
            }
        )
    _write_metadata(metadata, rows)
    cfg = _event_cfg(tmp_path, metadata)
    cfg["split"] = {"train": 0.7, "val": 0.15, "test": 0.15, "random_seed": 42}
    window_directory(clean, windows, cfg)
    stats = build_dataset(windows, final, cfg)
    index = pd.read_csv(final / "dataset_index.csv")

    assert stats["excluded_windows_filtered"] == 6
    assert set(index["event_phase"]) == {"pre_event", "fall_event"}
    assert index["include_in_training"].all()
    assert stats["class_balance"] == {"standing": 6, "fall": 6}

    split_sessions = index.groupby("session_id")["split"].nunique()
    assert split_sessions.max() == 1


def test_event_annotation_review_generates_plots_index_and_blank_template(tmp_path: Path) -> None:
    raw = tmp_path / "raw" / "fall"
    clean = tmp_path / "clean" / "fall"
    out = tmp_path / "review"
    template = tmp_path / "metadata" / "session_annotations_template_from_review.csv"
    raw.mkdir(parents=True)
    clean.mkdir(parents=True)
    _clean_like_sample("fall", n=60).to_csv(raw / "fall_case.csv", index=False)
    clean_df = _clean_like_sample("fall", n=60)
    clean_df["session_id"] = "fall__fall_case"
    clean_df.to_csv(clean / "fall_case.csv", index=False)

    cfg = load_config()
    cfg["windowing"]["window_len_sec"] = 1.0
    cfg["windowing"]["overlap_pct"] = 0.5
    summary = generate_event_annotation_review(raw, clean, out, template, cfg)

    assert summary["sessions_reviewed"] == 1
    index = pd.read_csv(out / "index.csv")
    generated_template = pd.read_csv(template)
    assert (out / "fall_case_review.png").exists()
    assert index.loc[0, "session_id"] == "fall_case"
    assert index.loc[0, "plot_path"].endswith("fall_case_review.png")
    assert generated_template.loc[0, "session_id"] == "fall_case"
    assert generated_template.loc[0, "activity"] == "fall"
    assert pd.isna(generated_template.loc[0, "event_start_s"])


def test_event_annotation_review_warns_for_too_short_sessions(tmp_path: Path) -> None:
    raw = tmp_path / "raw" / "fall"
    out = tmp_path / "review"
    template = tmp_path / "metadata" / "session_annotations_template_from_review.csv"
    raw.mkdir(parents=True)
    _clean_like_sample("fall", n=10).to_csv(raw / "short_fall.csv", index=False)

    cfg = load_config()
    cfg["windowing"]["window_len_sec"] = 1.5
    summary = generate_event_annotation_review(raw, None, out, template, cfg)

    assert summary["sessions_reviewed"] == 1
    assert "too_short_for_one_window" in summary["warnings"]["short_fall"]


def test_auto_event_annotation_has_strict_timestamp_ordering() -> None:
    cfg = load_config()
    df = _clean_like_sample("fall", n=100)
    df.loc[35:45, "x"] += np.linspace(0, 1.0, 11)
    df.loc[35:45, "z"] -= np.linspace(0, 0.8, 11)
    df.loc[46:, "z"] -= 0.8
    row = infer_session_annotation(df, "auto_case", cfg, {"non_frozen_drop_fraction": 0.0})
    assert row["event_start_s"] < row["impact_s"] < row["event_end_s"]
    assert 0 <= row["event_start_s"]
    assert row["event_end_s"] <= len(df) / cfg["sampling"]["expected_rate_hz"]


def test_auto_excluded_annotation_does_not_enter_staging_training(tmp_path: Path) -> None:
    clean = tmp_path / "clean"
    windows = tmp_path / "windows"
    final = tmp_path / "final"
    metadata = tmp_path / "auto_event_annotations.csv"
    (clean / "fall").mkdir(parents=True)
    df = _clean_like_sample("fall", n=100)
    df["session_id"] = "fall__low_conf"
    df.to_csv(clean / "fall" / "low_conf.csv", index=False)
    pd.DataFrame(
        [
            {
                "session_id": "low_conf",
                "activity": "fall",
                "start_state": "standing",
                "event_start_s": 1.5,
                "impact_s": 2.0,
                "event_end_s": 2.8,
                "end_state": "unknown",
                "distance_m_inferred": 2.0,
                "distance_band_inferred": "near",
                "direction_inferred": "unknown",
                "confidence": "low",
                "auto_include_in_training": False,
                "quality_flags": "weak_motion_peak",
                "method_version": "test",
                "method_summary": "test",
            }
        ]
    ).to_csv(metadata, index=False)
    cfg = _event_cfg(tmp_path, metadata)
    window_directory(clean, windows, cfg)
    data = np.load(windows / "fall_windows.npz", allow_pickle=True)
    assert not data["include_in_training"].any()
    try:
        build_dataset(windows, final, cfg)
    except FileNotFoundError as exc:
        assert "No non-empty window files" in str(exc)
    else:
        raise AssertionError("low-confidence windows entered dataset")


def test_auto_event_staging_dataset_construction(tmp_path: Path) -> None:
    clean = tmp_path / "clean"
    windows = tmp_path / "windows"
    final = tmp_path / "final"
    metadata = tmp_path / "auto_event_annotations.csv"
    (clean / "fall").mkdir(parents=True)
    for sid in ("auto_a", "auto_b"):
        df = _clean_like_sample("fall", n=100)
        df["session_id"] = f"fall__{sid}"
        df.to_csv(clean / "fall" / f"{sid}.csv", index=False)
    rows = []
    for sid in ("auto_a", "auto_b"):
        rows.append(
            {
                "session_id": sid,
                "activity": "fall",
                "start_state": "standing",
                "event_start_s": 2.0,
                "impact_s": 2.5,
                "event_end_s": 3.0,
                "end_state": "lying",
                "distance_m_inferred": 2.2,
                "distance_band_inferred": "middle",
                "direction_inferred": "unknown",
                "confidence": "medium",
                "auto_include_in_training": True,
                "quality_flags": "weak_height_evidence",
                "method_version": "test",
                "method_summary": "test",
            }
        )
    pd.DataFrame(rows).to_csv(metadata, index=False)
    cfg = _event_cfg(tmp_path, metadata)
    window_directory(clean, windows, cfg)
    stats = build_dataset(windows, final, cfg)
    index = pd.read_csv(final / "dataset_index.csv")
    assert stats["total_windows"] > 0
    assert set(index["annotation_confidence"]) == {"medium"}
    assert index["include_in_training"].all()


def test_conservative_auto_fall_filter_is_opt_in_and_preserves_pre_event_and_splits(tmp_path: Path) -> None:
    clean = tmp_path / "clean"
    metadata = tmp_path / "auto_event_annotations.csv"
    baseline_windows = tmp_path / "baseline_windows"
    conservative_windows = tmp_path / "conservative_windows"
    baseline_final = tmp_path / "baseline_final"
    conservative_final = tmp_path / "conservative_final"
    (clean / "fall").mkdir(parents=True)
    rows = []
    for sid in ("ambiguous_a", "ambiguous_b", "ambiguous_c", "ambiguous_d"):
        df = _clean_like_sample("fall", n=100)
        df["session_id"] = f"fall__{sid}"
        df.to_csv(clean / "fall" / f"{sid}.csv", index=False)
        rows.append(
            {
                "session_id": sid,
                "activity": "fall",
                "start_state": "standing",
                "event_start_s": 2.0,
                "impact_s": 2.5,
                "event_end_s": 3.0,
                "end_state": "lying",
                "distance_m_inferred": 2.2,
                "distance_band_inferred": "middle",
                "direction_inferred": "unknown",
                "confidence": "medium",
                "auto_include_in_training": True,
                "quality_flags": "geometry_edge_warning;weak_height_evidence",
                "method_version": "test",
                "method_summary": "test",
            }
        )
    pd.DataFrame(rows).to_csv(metadata, index=False)

    baseline_cfg = _event_cfg(tmp_path, metadata)
    baseline_cfg["split"] = {"train": 0.5, "val": 0.25, "test": 0.25, "random_seed": 42}
    conservative_cfg = _event_cfg(tmp_path, metadata)
    conservative_cfg["split"] = baseline_cfg["split"].copy()
    conservative_cfg["auto_event_annotation"]["exclude_ambiguous_fall_windows"] = True

    baseline_log = window_directory(clean, baseline_windows, baseline_cfg)
    conservative_log = window_directory(clean, conservative_windows, conservative_cfg)
    baseline = np.load(baseline_windows / "fall_windows.npz", allow_pickle=True)
    conservative = np.load(conservative_windows / "fall_windows.npz", allow_pickle=True)
    baseline_phases = baseline["event_phase"].astype(str)
    conservative_phases = conservative["event_phase"].astype(str)

    assert baseline_log["event_aware"]["exclude_ambiguous_fall_windows"] is False
    assert conservative_log["event_aware"]["exclude_ambiguous_fall_windows"] is True
    assert baseline["include_in_training"][baseline_phases == "fall_event"].all()
    assert not conservative["include_in_training"][conservative_phases == "fall_event"].any()
    assert conservative["include_in_training"][conservative_phases == "pre_event"].all()
    assert set(conservative["y"][conservative_phases == "pre_event"].tolist()) == {1}
    assert set(conservative["exclude_reason"][conservative_phases == "fall_event"].astype(str)) == {
        "ambiguous_auto_fall_window"
    }

    build_dataset(baseline_windows, baseline_final, baseline_cfg)
    build_dataset(conservative_windows, conservative_final, conservative_cfg)
    baseline_index = pd.read_csv(baseline_final / "dataset_index.csv")
    conservative_index = pd.read_csv(conservative_final / "dataset_index.csv")
    assert baseline_index.groupby("session_id")["split"].nunique().max() == 1
    assert conservative_index.groupby("session_id")["split"].nunique().max() == 1
    baseline_splits = baseline_index.groupby("session_id")["split"].first().to_dict()
    conservative_splits = conservative_index.groupby("session_id")["split"].first().to_dict()
    assert conservative_splits == baseline_splits


def test_auto_event_experiment_group_folds_keep_fall_and_no_session_leakage() -> None:
    y = np.asarray([3, 3, 1, 1, 2, 2, 0, 0, 3, 1, 2, 0])
    groups = np.asarray(["fall_a", "fall_a", "stand_a", "stand_a", "sit_a", "sit_a", "walk_a", "walk_a", "fall_b", "stand_b", "sit_b", "walk_b"])
    folds, reason = feasible_folds(y, groups, requested=5)
    assert folds == 2
    assert "fall_sessions=2" in reason
    for train_idx, test_idx in make_group_folds(y, groups, folds, seed=42):
        assert set(groups[train_idx]).isdisjoint(set(groups[test_idx]))
        assert int((y[test_idx] == 3).sum()) > 0
