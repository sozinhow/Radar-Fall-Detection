from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import runpy
import sys
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch import nn

from radar_pipeline.common import CLASS_LABELS, CLASS_NAMES, MODEL_FEATURES
from radar_pipeline.evaluate_grouped_cv import GroupedCVConfig, apply_augmented_features
from radar_pipeline.run_synthetic_fall_pilot import _train_model
from radar_pipeline.train_model import evaluate, make_loader, set_seed


ROOT = Path.cwd()
OUTPUT = ROOT / "outputs/experiments/demo_candidate_60frame_80_20_20260721"
REFERENCE = ROOT / "outputs/experiments/event_centered_clips_sgkf4_20260721"
DATASET = REFERENCE / "staging_clip_dataset_60.npz"
METADATA = REFERENCE / "staging_clip_metadata_60.csv"
REFERENCE_RUNNER = REFERENCE / "run_event_centered_clips.py"
CHECKPOINT = OUTPUT / "model/cnn_bilstm_60frame_demo_candidate.pt"
CLASS_ORDER = ["walking", "standing", "sitting", "fall"]
BASE_FEATURES = ["x", "y", "z", "dop_idx", "range_m", "azimuth_deg", "elevation_deg"]
MODEL_FEATURES_EXPECTED = BASE_FEATURES + [
    "xyz_delta_mag",
    "x_roll_std",
    "y_roll_std",
    "z_roll_std",
    "range_roll_std",
    "range_centered",
]
CLIP_LENGTH = 60
RATE_HZ = 20.0
STRIDE_FRAMES = 15
TEST_SEED = 42
VALIDATION_SEED = 43
TEST_FRACTION = 0.20
VALIDATION_SESSION_COUNT = 61
MIN_TEST_FALL_SESSIONS = 5
EXPECTED_SPLIT_COUNTS = {
    "train": {"walking": 92, "standing": 115, "sitting": 91, "fall": 29},
    "validation": {"walking": 17, "standing": 22, "sitting": 17, "fall": 5},
    "test": {"walking": 27, "standing": 34, "sitting": 27, "fall": 9},
}
CONFIG = GroupedCVConfig(
    epochs=80,
    batch_size=32,
    learning_rate=1e-3,
    weight_decay=1e-4,
    patience=12,
    seed=42,
    folds=4,
    val_folds=6,
    feature_mode="augmented",
    dropout_input=0.25,
    dropout_hidden=0.20,
)


PROTECTED_PATHS = (
    ROOT / "outputs/PROJECT_STATUS.md",
    ROOT / "outputs/pipeline_operation.md",
    ROOT / "outputs/GUIDANCE.md",
    ROOT / "outputs/experiments/auto_event_aware_sgkf4_baseline",
    ROOT / "outputs/experiments/event_centered_clips_sgkf4_20260721",
    ROOT / "outputs/experiments/event_centered_transformer_ablation_sgkf4_20260721",
    ROOT / "data/metadata/auto_event_aware_v1_source_session_folds.csv",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_digest(path: Path) -> str:
    if path.is_file():
        return sha256(path)
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(str(child.relative_to(path)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256(child)))
    return digest.hexdigest()


def safe_text(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.write_text(text, encoding="utf-8")


def safe_json(path: Path, payload: dict[str, object]) -> None:
    safe_text(path, json.dumps(payload, indent=2, allow_nan=False) + "\n")


def safe_csv(path: Path, frame: pd.DataFrame) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    frame.to_csv(path, index=False)


def safe_plot(path: Path, fig: plt.Figure) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def output_is_clean() -> None:
    allowed_existing = {OUTPUT / "logs/run_demo_training.py"}
    runtime_cache = OUTPUT / "logs/.runtime_cache"
    unexpected = [
        path
        for path in OUTPUT.rglob("*")
        if path.is_file() and path not in allowed_existing and runtime_cache not in path.parents
    ]
    if unexpected:
        raise FileExistsError(f"Output contains pre-existing files: {unexpected[:3]}")
    for directory in (OUTPUT / "model", OUTPUT / "metrics", OUTPUT / "plots", OUTPUT / "logs"):
        if not directory.is_dir():
            raise FileNotFoundError(directory)


def load_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    source = np.load(DATASET, allow_pickle=True)
    X = source["X"].astype(np.float32)
    y = source["y"].astype(np.int64)
    masks = source["mask"].astype(bool)
    metadata = pd.read_csv(METADATA)
    feature_names = [str(value) for value in source["feature_names"]]
    if X.shape != (485, CLIP_LENGTH, 7) or masks.shape != (485, CLIP_LENGTH):
        raise ValueError(f"Unexpected canonical clip shape: {X.shape}, {masks.shape}")
    if feature_names != BASE_FEATURES or [str(value) for value in CLASS_NAMES] != CLASS_ORDER:
        raise ValueError("Feature or class order differs from the accepted experiment")
    if not np.array_equal(source["source_session_id"].astype(str), metadata["source_session_id"].astype(str)):
        raise ValueError("Dataset and metadata source-session order differ")
    if not np.array_equal(y, metadata["label"].astype(int).to_numpy()):
        raise ValueError("Dataset and metadata labels differ")
    if len(metadata) != metadata["source_session_id"].astype(str).nunique():
        raise ValueError("Expected one deterministic canonical clip per source session")
    if set(y.tolist()) != set(CLASS_LABELS):
        raise ValueError("All four classes are required")
    return X, y, masks, metadata


def make_splits(y: np.ndarray) -> dict[str, np.ndarray]:
    indices = np.arange(len(y))
    training_pool, test_idx = train_test_split(
        indices,
        test_size=TEST_FRACTION,
        random_state=TEST_SEED,
        stratify=y,
    )
    train_idx, validation_idx = train_test_split(
        training_pool,
        test_size=VALIDATION_SESSION_COUNT,
        random_state=VALIDATION_SEED,
        stratify=y[training_pool],
    )
    splits = {
        "train": np.sort(train_idx),
        "validation": np.sort(validation_idx),
        "test": np.sort(test_idx),
    }
    sets = {name: set(values.tolist()) for name, values in splits.items()}
    if (sets["train"] & sets["validation"]) or (sets["train"] & sets["test"]) or (
        sets["validation"] & sets["test"]
    ):
        raise ValueError("Index overlap in deterministic split")
    if set.union(*sets.values()) != set(indices.tolist()):
        raise ValueError("Split does not cover every canonical session")
    for split, expected in EXPECTED_SPLIT_COUNTS.items():
        observed = {name: int((y[splits[split]] == label).sum()) for label, name in enumerate(CLASS_ORDER)}
        if observed != expected:
            raise ValueError(f"Unexpected {split} class counts: {observed} != {expected}")
        if any(value == 0 for value in observed.values()):
            raise ValueError(f"{split} is missing a class: {observed}")
    if int((y[splits["test"]] == CLASS_ORDER.index("fall")).sum()) < MIN_TEST_FALL_SESSIONS:
        raise ValueError("Held-out test does not contain enough fall sessions")
    return splits


def split_manifest(metadata: pd.DataFrame, y: np.ndarray, masks: np.ndarray, splits: dict[str, np.ndarray]) -> pd.DataFrame:
    split_for_index = {
        int(index): split
        for split, indices in splits.items()
        for index in indices
    }
    frame = metadata.copy()
    frame.insert(0, "split", [split_for_index[index] for index in range(len(frame))])
    frame.insert(1, "split_seed", [TEST_SEED if value == "test" else VALIDATION_SEED for value in frame["split"]])
    frame.insert(2, "class", [CLASS_ORDER[int(value)] for value in y])
    frame.insert(3, "class_label", y)
    frame["canonical_clip_count"] = 1
    frame["class_support_contribution"] = 1
    frame["mask_valid_frames"] = masks.sum(axis=1).astype(int)
    frame["mask_padding_frames"] = (~masks).sum(axis=1).astype(int)
    frame = frame.rename(columns={"outer_fold": "source_sgkf4_outer_fold"})
    ordered = [
        "split",
        "split_seed",
        "source_session_id",
        "source_csv",
        "class",
        "class_label",
        "canonical_clip_count",
        "class_support_contribution",
        "source_sgkf4_outer_fold",
        "clip_length_frames",
        "clip_start_row",
        "clip_end_row_exclusive",
        "clip_start_s",
        "clip_end_s",
        "mask_valid_frames",
        "mask_padding_frames",
        "padding_left_frames",
        "padding_right_frames",
        "event_start_s",
        "impact_s",
        "event_end_s",
        "annotation_confidence",
        "quality_flags",
        "sampling_rule",
    ]
    return frame[ordered].sort_values(["split", "class_label", "source_session_id"]).reset_index(drop=True)


def dataset_audit(manifest: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for split in ("train", "validation", "test"):
        subset = manifest.loc[manifest["split"].eq(split)]
        for label, class_name in enumerate(CLASS_ORDER):
            group = subset.loc[subset["class_label"].eq(label)]
            rows.append(
                {
                    "split": split,
                    "class": class_name,
                    "label": label,
                    "source_sessions": int(group["source_session_id"].nunique()),
                    "canonical_clips": int(group["canonical_clip_count"].sum()),
                    "padded_clips": int((group["mask_padding_frames"] > 0).sum()),
                    "minimum_valid_frames": int(group["mask_valid_frames"].min()),
                    "maximum_valid_frames": int(group["mask_valid_frames"].max()),
                }
            )
    return pd.DataFrame(rows)


def normalization_from_train(X: np.ndarray, train_idx: np.ndarray) -> dict[str, list[float]]:
    flat = X[train_idx].reshape(-1, X.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    return {"mean": mean.tolist(), "std": std.tolist()}


def normalize(X: np.ndarray, norm: dict[str, list[float]]) -> np.ndarray:
    mean = np.asarray(norm["mean"], dtype=np.float32)
    std = np.asarray(norm["std"], dtype=np.float32)
    return ((X - mean) / std).astype(np.float32)


def canonical_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[dict[str, object], pd.DataFrame, np.ndarray]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=CLASS_LABELS, zero_division=0
    )
    macro = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    weighted = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    metrics = {
        "task": "canonical_clip_four_class_argmax",
        "class_order": CLASS_ORDER,
        "confusion_matrix_orientation": "rows=true_class, columns=predicted_class",
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro[0]),
        "macro_recall": float(macro[1]),
        "macro_f1": float(macro[2]),
        "weighted_precision": float(weighted[0]),
        "weighted_recall": float(weighted[1]),
        "weighted_f1": float(weighted[2]),
        "heldout_sessions": int(len(y_true)),
    }
    per_class = pd.DataFrame(
        {
            "class": CLASS_ORDER,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support.astype(int),
        }
    )
    matrix = confusion_matrix(y_true, y_pred, labels=CLASS_LABELS)
    return metrics, per_class, matrix


def named_matrix(matrix: np.ndarray, row_names: list[str], column_names: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(matrix.astype(int), columns=column_names)
    frame.insert(0, "true_class", row_names)
    return frame


def plot_confusion(matrix: np.ndarray, class_names: list[str], title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.3))
    image = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted class",
        ylabel="True class",
        title=title,
    )
    threshold = matrix.max() / 2.0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(
                column,
                row,
                str(int(matrix[row, column])),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
            )
    fig.tight_layout()
    safe_plot(path, fig)


def plot_training(history: list[dict[str, float]], path: Path) -> None:
    frame = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    axes[0].plot(frame["epoch"], frame["train_loss"], label="train")
    axes[0].plot(frame["epoch"], frame["val_loss"], label="validation")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Cross-entropy")
    axes[0].legend()
    axes[1].plot(frame["epoch"], frame["train_acc"], label="train")
    axes[1].plot(frame["epoch"], frame["val_acc"], label="validation")
    axes[1].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1))
    axes[1].legend()
    fig.suptitle("60-frame CNN-BiLSTM demo candidate training")
    fig.tight_layout()
    safe_plot(path, fig)


def session_diagnostics(stream: pd.DataFrame, sessions: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, info in sessions.sort_values("source_session_id").iterrows():
        sid = str(info["source_session_id"])
        activity = str(info["activity"])
        group = stream.loc[stream["source_session_id"].astype(str).eq(sid)].sort_values("clip_end_s")
        probabilities = group["fall_probability"].to_numpy(dtype=float)
        support = probabilities >= threshold
        previous = np.concatenate([[False], support[:-1]]) if len(support) else np.asarray([], dtype=bool)
        alert_indices = np.flatnonzero(support & ~previous)
        alert_times = group.iloc[alert_indices]["clip_end_s"].to_numpy(dtype=float) if len(alert_indices) else np.asarray([])
        is_fall = activity == "fall"
        event_start = float(info["event_start_s"]) if is_fall else np.nan
        impact = float(info["impact_s"]) if is_fall else np.nan
        valid_alerts = alert_times[alert_times >= event_start] if is_fall else alert_times
        first_alert = float(valid_alerts[0]) if len(valid_alerts) else np.nan
        causal = bool(
            (group["max_source_row_exclusive"] <= group["causal_available_row_exclusive"]).all()
        )
        rows.append(
            {
                "source_session_id": sid,
                "source_csv": str(info["source_csv"]),
                "activity": activity,
                "is_fall_session": is_fall,
                "has_alert": bool(len(valid_alerts)),
                "alert_count": int(len(valid_alerts)),
                "repeated_alerts": int(max(len(valid_alerts) - 1, 0)),
                "first_alert_s": first_alert,
                "event_start_s": event_start,
                "impact_s": impact,
                "delay_event_start_s": first_alert - event_start if is_fall and len(valid_alerts) else np.nan,
                "delay_impact_s": first_alert - impact if is_fall and len(valid_alerts) else np.nan,
                "maximum_fall_probability": float(probabilities.max()),
                "selected_threshold": threshold,
                "streaming_predictions": int(len(group)),
                "causal_current_or_past_only": causal,
                "alert_time_equals_trailing_clip_end": True,
            }
        )
    return pd.DataFrame(rows)


def causal_metrics_from_diagnostics(diagnostics: pd.DataFrame, threshold: float) -> tuple[dict[str, object], np.ndarray]:
    fall = diagnostics.loc[diagnostics["is_fall_session"]]
    nonfall = diagnostics.loc[~diagnostics["is_fall_session"]]
    tp = int(fall["has_alert"].sum())
    fn = int((~fall["has_alert"]).sum())
    fp = int(nonfall["has_alert"].sum())
    tn = int((~nonfall["has_alert"]).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    metrics = {
        "task": "causal_streaming_binary_fall_alert",
        "selected_validation_threshold": threshold,
        "threshold_selected_from_test": False,
        "alert_rule": "rising edge of fall_probability >= selected validation threshold",
        "alert_time_policy": "alert time equals trailing 60-frame clip end time",
        "fall_precision": precision,
        "fall_recall": recall,
        "fall_f1": f1,
        "nonfall_sessions_alerted": fp,
        "false_alerts_per_nonfall_session": float(nonfall["alert_count"].sum() / len(nonfall)),
        "fall_sessions_no_alert": fn,
        "fall_sessions_multiple_alerts": int((fall["alert_count"] > 1).sum()),
        "repeated_alerts_total_fall_sessions": int(fall["repeated_alerts"].sum()),
        "repeated_alerts_per_fall_session": float(fall["repeated_alerts"].sum() / len(fall)),
        "all_sessions_multiple_alerts": int((diagnostics["alert_count"] > 1).sum()),
        "mean_delay_event_start_s": float(fall["delay_event_start_s"].mean()),
        "mean_delay_impact_s": float(fall["delay_impact_s"].mean()),
        "fall_sessions": int(len(fall)),
        "nonfall_sessions": int(len(nonfall)),
        "causal_current_or_past_only": bool(diagnostics["causal_current_or_past_only"].all()),
        "all_alert_times_equal_trailing_clip_end": bool(
            diagnostics["alert_time_equals_trailing_clip_end"].all()
        ),
        "confusion_matrix_orientation": "rows=true [non_fall, fall], columns=predicted [no_alert, alert]",
    }
    return metrics, np.asarray([[tn, fp], [fn, tp]], dtype=int)


def matrix_markdown(matrix: np.ndarray, row_names: list[str], column_names: list[str]) -> list[str]:
    lines = ["| True \\ Predicted | " + " | ".join(column_names) + " |", "|---|" + "---:|" * len(column_names)]
    for name, row in zip(row_names, matrix):
        lines.append(f"| {name} | " + " | ".join(str(int(value)) for value in row) + " |")
    return lines


def build_report(
    manifest: pd.DataFrame,
    training_config: dict[str, object],
    four_class: dict[str, object],
    per_class: pd.DataFrame,
    four_matrix: np.ndarray,
    causal: dict[str, object],
    causal_matrix: np.ndarray,
) -> str:
    counts = manifest.groupby(["split", "class"]).size().unstack(fill_value=0).reindex(
        index=["train", "validation", "test"], columns=CLASS_ORDER
    )
    lines = [
        "# 60-Frame Event-Centred CNN-BiLSTM Demo Candidate",
        "",
        "**Single deterministic 80/20 source-session split; prototype demonstration candidate.**",
        "",
        "The split is session-grouped, not random window-level splitting. Validation was derived only from the 80% training pool. Test signal values and predictions were untouched until training, early stopping, validation-only threshold selection, checkpoint selection, and checkpoint saving were complete.",
        "",
        "This run is not directly comparable to the SGKF4 mean +/- standard deviation. It does not supersede the frozen SGKF4 staging evidence. This is suitable for a single-model demo candidate, not a deployment, clinical-reliability, or final-product validation claim.",
        "",
        "## Split Audit",
        "",
        "| Split | Sessions/clips | Walking | Standing | Sitting | Fall |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for split in ("train", "validation", "test"):
        lines.append(
            f"| {split} | {int(counts.loc[split].sum())} | "
            + " | ".join(str(int(counts.loc[split, name])) for name in CLASS_ORDER)
            + " |"
        )
    lines.extend(
        [
            "",
            "Every class is present in every split. Held-out test contains nine fall sessions, above the predeclared minimum of five. Source-session overlap is zero.",
            "",
            "## Selected Training State",
            "",
            f"- Selected epoch: {training_config['selected_epoch']} of {training_config['epochs_ran']} epochs run.",
            f"- Best validation loss: {training_config['best_validation_loss']:.6f}.",
            f"- Validation-selected fall-alert threshold: {training_config['selected_fall_alert_threshold']:.2f}.",
            "- Architecture: accepted CNN-BiLSTM; seven normalized base inputs plus the accepted deterministic six-feature derivation (13 model channels).",
            "- Padding/mask behavior: saved edge-padded 60-frame clips are consumed exactly as in the accepted CNN-BiLSTM, whose architecture has no padding-mask input; masks remain recorded for audit.",
            "",
            "## Held-Out Canonical Four-Class Argmax",
            "",
            "| Accuracy | Macro precision | Macro recall | Macro F1 | Weighted precision | Weighted recall | Weighted F1 |",
            "|---:|---:|---:|---:|---:|---:|---:|",
            f"| {four_class['accuracy']:.4f} | {four_class['macro_precision']:.4f} | {four_class['macro_recall']:.4f} | {four_class['macro_f1']:.4f} | {four_class['weighted_precision']:.4f} | {four_class['weighted_recall']:.4f} | {four_class['weighted_f1']:.4f} |",
            "",
            "| Class | Precision | Recall | F1 | Support |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in per_class.to_dict(orient="records"):
        lines.append(
            f"| {row['class']} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} | {row['support']} |"
        )
    lines.extend(["", "Rows are true classes and columns are predicted classes.", ""])
    lines.extend(matrix_markdown(four_matrix, CLASS_ORDER, CLASS_ORDER))
    lines.extend(
        [
            "",
            "## Held-Out Causal Streaming Fall Alerts",
            "",
            "Only trailing 60-frame clips ending at the current alert timestamp were used; no future frame or full-session aggregation was available to an earlier decision.",
            "",
            "| Fall precision | Fall recall | Fall F1 | Non-fall sessions alerted | False alerts/non-fall | No-alert falls | Fall sessions with repeated alerts | Event-start delay s | Impact delay s |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            f"| {causal['fall_precision']:.4f} | {causal['fall_recall']:.4f} | {causal['fall_f1']:.4f} | {causal['nonfall_sessions_alerted']} | {causal['false_alerts_per_nonfall_session']:.4f} | {causal['fall_sessions_no_alert']} | {causal['fall_sessions_multiple_alerts']} | {causal['mean_delay_event_start_s']:.4f} | {causal['mean_delay_impact_s']:.4f} |",
            "",
            "Causal session confusion matrix: rows are true `non_fall, fall`; columns are predicted `no_alert, alert`.",
            "",
        ]
    )
    lines.extend(matrix_markdown(causal_matrix, ["non_fall", "fall"], ["no_alert", "alert"]))
    lines.extend(
        [
            "",
            "## Artifact Locations",
            "",
            "- Checkpoint: `outputs/experiments/demo_candidate_60frame_80_20_20260721/model/cnn_bilstm_60frame_demo_candidate.pt`",
            "- Split manifest: `outputs/experiments/demo_candidate_60frame_80_20_20260721/split_manifest.csv`",
            "- Training configuration: `outputs/experiments/demo_candidate_60frame_80_20_20260721/training_config.json`",
            "- Normalization: `outputs/experiments/demo_candidate_60frame_80_20_20260721/normalization_stats.json`",
            "- Four-class metrics: `outputs/experiments/demo_candidate_60frame_80_20_20260721/metrics/test_four_class_metrics.json`",
            "- Causal metrics: `outputs/experiments/demo_candidate_60frame_80_20_20260721/metrics/test_causal_fall_alert_metrics.json`",
            "- Session diagnostics: `outputs/experiments/demo_candidate_60frame_80_20_20260721/metrics/test_causal_session_diagnostics.csv`",
            "- Training curve: `outputs/experiments/demo_candidate_60frame_80_20_20260721/plots/training_curve.png`",
            "",
            "## Integrity Checks",
            "",
            "- Exactly one new checkpoint was saved; no ensemble and no existing fold model was selected.",
            "- Normalization was fitted on train sessions only and then frozen.",
            "- Early stopping and fall-alert threshold selection used validation only.",
            "- The test split was scored once after model and threshold freeze.",
            "- Every causal current/past-only and alert-time-equals-clip-end assertion passed.",
            "- Protected paths were verified unchanged after the run.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    output_is_clean()
    if sys.executable != "/Users/stella/miniconda3/bin/python":
        raise RuntimeError(f"Unverified Python environment: {sys.executable}")
    protected_before = {str(path.relative_to(ROOT)): tree_digest(path) for path in PROTECTED_PATHS}
    X, y, masks, metadata = load_inputs()
    splits = make_splits(y)
    manifest = split_manifest(metadata, y, masks, splits)
    audit = dataset_audit(manifest)
    safe_csv(OUTPUT / "split_manifest.csv", manifest)
    safe_csv(OUTPUT / "dataset_audit.csv", audit)
    for split, filename in (
        ("train", "train_session_ids.txt"),
        ("validation", "validation_session_ids.txt"),
        ("test", "test_session_ids.txt"),
    ):
        values = sorted(manifest.loc[manifest["split"].eq(split), "source_session_id"].astype(str))
        safe_text(OUTPUT / filename, "\n".join(values) + "\n")
    safe_text(
        OUTPUT / "split_manifest_README.md",
        "# Demo Candidate Split Manifest\n\n"
        "Deterministic source-session split with seed 42 for the 80/20 training-pool/test split and seed 43 for a 61-session validation subset drawn only from the training pool. One accepted deterministic 60-frame canonical clip exists per source session. All four classes are present in train, validation, and test; session overlap is zero. The original SGKF4 assignment is recorded only as provenance and is not used as this demo split.\n",
    )
    print("pretraining_split", audit[["split", "class", "source_sessions"]].to_dict(orient="records"), flush=True)

    norm = normalization_from_train(X, splits["train"])
    safe_json(
        OUTPUT / "normalization_stats.json",
        {
            "format": "mean_std_by_base_feature",
            "fit_scope": "train sessions only",
            "feature_order": BASE_FEATURES,
            **norm,
        },
    )
    X_train = normalize(X[splits["train"]], norm)
    X_validation = normalize(X[splits["validation"]], norm)
    X_train_aug, X_validation_aug, _, augmented_names = apply_augmented_features(
        X_train,
        X_validation,
        X_validation[:1],
        np.asarray(BASE_FEATURES),
        "augmented",
    )
    if augmented_names != MODEL_FEATURES_EXPECTED:
        raise ValueError("Accepted augmented feature order changed")
    set_seed(CONFIG.seed)
    random.seed(CONFIG.seed)
    np.random.seed(CONFIG.seed)
    start_time = time.perf_counter()
    model, history, best_epoch, best_validation_loss = _train_model(
        X_train_aug,
        y[splits["train"]],
        X_validation_aug,
        y[splits["validation"]],
        CONFIG,
        torch.device("cpu"),
    )
    training_time = time.perf_counter() - start_time
    history_frame = pd.DataFrame(history)
    safe_csv(OUTPUT / "logs/training_history.csv", history_frame)
    plot_training(history, OUTPUT / "plots/training_curve.png")

    validation_loader = make_loader(
        X_validation_aug, y[splits["validation"]], CONFIG.batch_size, shuffle=False
    )
    _, _, val_true, val_pred, _ = evaluate(
        model, validation_loader, nn.CrossEntropyLoss(), torch.device("cpu")
    )
    validation_metrics, _, _ = canonical_metrics(val_true, val_pred)
    helpers = runpy.run_path(str(REFERENCE_RUNNER), run_name="demo_candidate_reference_helpers")
    validation_sessions = metadata.iloc[splits["validation"]].copy()
    validation_raw, validation_stream = helpers["sliding_clips"](CLIP_LENGTH, validation_sessions)
    validation_probabilities = helpers["infer_probabilities"](
        model, validation_raw, norm, CLIP_LENGTH, CONFIG.batch_size
    )
    validation_stream["fall_probability"] = validation_probabilities[:, CLASS_ORDER.index("fall")]
    selected_threshold, threshold_sweep = helpers["choose_threshold"](
        validation_stream, validation_sessions
    )
    safe_csv(OUTPUT / "logs/validation_threshold_sweep.csv", threshold_sweep)

    checkpoint_payload = {
        "model_state_dict": model.state_dict(),
        "model_class": "CNNLSTM",
        "prototype_status": "single_model_demo_candidate_not_deployment_ready",
        "class_order": CLASS_ORDER,
        "base_feature_names": BASE_FEATURES,
        "model_feature_names": augmented_names,
        "clip_length_frames": CLIP_LENGTH,
        "clip_length_seconds": CLIP_LENGTH / RATE_HZ,
        "sampling_rate_hz": RATE_HZ,
        "causal_stride_frames": STRIDE_FRAMES,
        "normalization": norm,
        "normalization_fit": "train source sessions only",
        "padding_mask_handling": "edge-padded clips as accepted CNN inference; masks recorded but CNN has no mask input",
        "train_config": asdict(CONFIG),
        "test_split_seed": TEST_SEED,
        "validation_split_seed": VALIDATION_SEED,
        "selected_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "validation_four_class_metrics": validation_metrics,
        "selected_fall_alert_threshold": selected_threshold,
        "threshold_selection_scope": "validation sessions only",
        "threshold_policy": "0.10..0.90 step 0.05; maximize fall F1, then recall, then minimize nonfall sessions alerted, false alerts/nonfall, and impact delay",
        "split_manifest_sha256": sha256(OUTPUT / "split_manifest.csv"),
        "source_dataset_sha256": sha256(DATASET),
    }
    if CHECKPOINT.exists():
        raise FileExistsError(f"Refusing to overwrite {CHECKPOINT}")
    torch.save(checkpoint_payload, CHECKPOINT)

    # The held-out test signal values are first transformed and scored only after
    # the selected model checkpoint and validation-only threshold are frozen above.
    X_test = normalize(X[splits["test"]], norm)
    _, _, X_test_aug, test_feature_names = apply_augmented_features(
        X_train[:1], X_validation[:1], X_test, np.asarray(BASE_FEATURES), "augmented"
    )
    if test_feature_names != augmented_names:
        raise ValueError("Test feature order changed")
    test_loader = make_loader(X_test_aug, y[splits["test"]], CONFIG.batch_size, shuffle=False)
    _, _, test_true, test_pred, _ = evaluate(
        model, test_loader, nn.CrossEntropyLoss(), torch.device("cpu")
    )
    four_metrics, per_class, four_matrix = canonical_metrics(test_true, test_pred)
    four_metrics.update(
        {
            "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
            "model_frozen_before_test_scoring": True,
            "thresholds_applied": False,
        }
    )
    safe_json(OUTPUT / "metrics/test_four_class_metrics.json", four_metrics)
    safe_csv(OUTPUT / "metrics/test_per_class_metrics.csv", per_class)
    safe_csv(
        OUTPUT / "metrics/test_four_class_confusion_matrix.csv",
        named_matrix(four_matrix, CLASS_ORDER, CLASS_ORDER),
    )
    plot_confusion(
        four_matrix,
        CLASS_ORDER,
        "Demo Candidate — Held-Out Canonical Four-Class",
        OUTPUT / "metrics/test_four_class_confusion_matrix.png",
    )

    test_sessions = metadata.iloc[splits["test"]].copy()
    test_raw, test_stream = helpers["sliding_clips"](CLIP_LENGTH, test_sessions)
    test_probabilities = helpers["infer_probabilities"](
        model, test_raw, norm, CLIP_LENGTH, CONFIG.batch_size
    )
    test_stream["fall_probability"] = test_probabilities[:, CLASS_ORDER.index("fall")]
    diagnostics = session_diagnostics(test_stream, test_sessions, selected_threshold)
    causal_metrics, causal_matrix = causal_metrics_from_diagnostics(diagnostics, selected_threshold)
    reference_causal = helpers["event_metrics"](test_stream, test_sessions, selected_threshold)
    checks = {
        "fall_precision": "session_fall_precision",
        "fall_recall": "session_fall_recall",
        "fall_f1": "session_fall_f1",
        "nonfall_sessions_alerted": "nonfall_sessions_alerted",
        "fall_sessions_no_alert": "fall_sessions_no_alert",
        "mean_delay_event_start_s": "mean_delay_event_start_s",
        "mean_delay_impact_s": "mean_delay_impact_s",
    }
    for local_key, reference_key in checks.items():
        if not np.isclose(causal_metrics[local_key], reference_causal[reference_key], equal_nan=True):
            raise ValueError(f"Causal reconstruction differs for {local_key}")
    if not causal_metrics["causal_current_or_past_only"] or not causal_metrics[
        "all_alert_times_equal_trailing_clip_end"
    ]:
        raise ValueError("Causal no-future or alert-time assertion failed")
    causal_metrics.update(
        {
            "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
            "model_and_threshold_frozen_before_test_scoring": True,
        }
    )
    safe_json(OUTPUT / "metrics/test_causal_fall_alert_metrics.json", causal_metrics)
    safe_csv(
        OUTPUT / "metrics/test_causal_fall_alert_confusion_matrix.csv",
        named_matrix(causal_matrix, ["non_fall", "fall"], ["no_alert", "alert"]),
    )
    safe_csv(OUTPUT / "metrics/test_causal_session_diagnostics.csv", diagnostics)

    training_config = {
        "purpose": "single deterministic 80/20 source-session prototype demonstration candidate",
        "deployment_ready": False,
        "clinical_reliability_claim": False,
        "class_order": CLASS_ORDER,
        "base_feature_order": BASE_FEATURES,
        "model_feature_order": augmented_names,
        "architecture": "CNN1D(32,64,96)+bidirectional_LSTM(hidden=64)+MLP(128,64,4)",
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "sampling_rate_hz": RATE_HZ,
        "clip_length_frames": CLIP_LENGTH,
        "clip_length_seconds": CLIP_LENGTH / RATE_HZ,
        "causal_stride_frames": STRIDE_FRAMES,
        "causal_stride_seconds": STRIDE_FRAMES / RATE_HZ,
        "padding_mask_handling": "same accepted edge padding; mask recorded for audit; CNN has no mask input",
        "normalization_fit": "train source sessions only",
        "test_split": {"fraction": TEST_FRACTION, "seed": TEST_SEED},
        "validation_split": {
            "sessions": VALIDATION_SESSION_COUNT,
            "fraction_of_all_sessions": VALIDATION_SESSION_COUNT / len(metadata),
            "seed": VALIDATION_SEED,
            "source": "80 percent training pool only",
        },
        "minimum_test_fall_sessions": MIN_TEST_FALL_SESSIONS,
        "training": asdict(CONFIG),
        "device": "cpu",
        "selected_epoch": best_epoch,
        "epochs_ran": len(history),
        "best_validation_loss": best_validation_loss,
        "validation_four_class_metrics": validation_metrics,
        "selected_fall_alert_threshold": selected_threshold,
        "threshold_selection_data": "validation sessions only",
        "threshold_grid": [float(value) for value in helpers["THRESHOLDS"]],
        "threshold_policy": "maximize validation session fall F1; tie-break recall, fewer nonfall sessions alerted, fewer false alerts/nonfall, shorter impact delay",
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "training_time_seconds": training_time,
        "test_access_policy": "test signal values and predictions first accessed after checkpoint and threshold freeze",
    }
    safe_json(OUTPUT / "training_config.json", training_config)
    protected_after = {str(path.relative_to(ROOT)): tree_digest(path) for path in PROTECTED_PATHS}
    if protected_before != protected_after:
        changed = [key for key in protected_before if protected_before[key] != protected_after[key]]
        raise RuntimeError(f"Protected path changed during run: {changed}")
    report = build_report(manifest, training_config, four_metrics, per_class, four_matrix, causal_metrics, causal_matrix)
    safe_text(OUTPUT / "REPORT.md", report)
    provenance = {
        "created_utc": pd.Timestamp.now(tz="Asia/Hong_Kong").isoformat(),
        "project_root": str(ROOT),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "package_versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "source_dataset": str(DATASET.relative_to(ROOT)),
        "source_dataset_sha256": sha256(DATASET),
        "source_metadata": str(METADATA.relative_to(ROOT)),
        "source_metadata_sha256": sha256(METADATA),
        "accepted_runner": str(REFERENCE_RUNNER.relative_to(ROOT)),
        "accepted_runner_sha256": sha256(REFERENCE_RUNNER),
        "checkpoint_sha256": sha256(CHECKPOINT),
        "split_manifest_sha256": sha256(OUTPUT / "split_manifest.csv"),
        "protected_paths_unchanged": True,
        "protected_path_digests": protected_after,
        "retraining_existing_model": False,
        "new_models_trained": 1,
        "ensemble": False,
        "threshold_selected_on_test": False,
        "files_outside_output_created": False,
    }
    safe_json(OUTPUT / "provenance.json", provenance)
    print(
        json.dumps(
            {
                "split_sessions": {name: len(indices) for name, indices in splits.items()},
                "selected_epoch": best_epoch,
                "test_accuracy": four_metrics["accuracy"],
                "test_macro_f1": four_metrics["macro_f1"],
                "causal_fall_precision": causal_metrics["fall_precision"],
                "causal_fall_recall": causal_metrics["fall_recall"],
                "causal_fall_f1": causal_metrics["fall_f1"],
                "session_overlap": 0,
                "protected_paths_unchanged": True,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
