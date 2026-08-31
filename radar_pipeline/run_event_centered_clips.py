"""Reusable launcher for the versioned event-centred SGKF4 runner.

The implementation is kept beside the pipeline so the runnable workflow does
not depend on an old experiment-output directory. This launcher supplies new
paths and either reuses saved source-session inventories or derives
session-grouped train/validation/test roles from a supplied frozen manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterable

import matplotlib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from radar_pipeline.common import CLASS_LABELS, CLASS_NAMES, MODEL_FEATURES, ROOT
from radar_pipeline.evaluate_grouped_cv import GroupedCVConfig, grouped_validation_split, load_all_windows, validate_manifest
from radar_pipeline.evaluate_grouped_cv import apply_augmented_features
from radar_pipeline.train_model import CNNLSTM


RECOVERED_REFERENCE_RUNNER = ROOT / "radar_pipeline/reference_event_centered_runner.py"
CANONICAL_EVALUATION_DIR = "evaluation_artifacts"
NONFALL_CLIP_POLICIES = ("legacy_hash", "steady_state_v1")
SUPPORTED_CLIP_LENGTHS = {50: 15, 60: 20, 100: 20}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_clip_lengths(value: str) -> tuple[int, ...]:
    """Parse explicitly supported event-centred lengths in a stable order."""

    try:
        lengths = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise ValueError("--clip-lengths must be a comma-separated integer list") from exc
    if not lengths or len(set(lengths)) != len(lengths):
        raise ValueError("--clip-lengths must contain one or more unique lengths")
    unsupported = [length for length in lengths if length not in SUPPORTED_CLIP_LENGTHS]
    if unsupported:
        raise ValueError(
            f"Unsupported clip lengths {unsupported}; supported lengths are {sorted(SUPPORTED_CLIP_LENGTHS)}."
        )
    return lengths


def load_recovered_runner(path: Path) -> ModuleType:
    if not path.is_file():
        raise FileNotFoundError(
            f"Recovered event-centred implementation is unavailable: {path}. "
            "Restore the historical runner or provide --reference-runner."
        )
    spec = importlib.util.spec_from_file_location("_recovered_event_centered_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load recovered runner: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_reconstruction_inputs(
    dataset: Path, manifest_path: Path, *, seed: int, val_folds: int
) -> tuple[dict[str, np.ndarray], pd.DataFrame, GroupedCVConfig]:
    """Load and validate a supplied reconstruction dataset and frozen manifest.

    This intentionally has no historical inventory-size requirement.  A new
    reconstruction is valid when its dataset metadata and manifest pass the
    normal grouped-CV validation rules.
    """

    payload = load_all_windows(dataset)
    manifest = pd.read_csv(manifest_path)
    cfg = GroupedCVConfig(seed=seed, folds=4, val_folds=val_folds)
    validate_manifest(payload, manifest, cfg)
    return payload, manifest, cfg


def reconstruction_inventory(payload: dict[str, np.ndarray], manifest: pd.DataFrame) -> pd.DataFrame:
    """Build the session table consumed by the recovered training logic.

    Unlike the historical runner's ``inventory()``, this only requires a
    one-to-one, manifest-validated source-session mapping.  It deliberately
    does not require exactly 485 sessions or historical cleaned CSV paths.
    """

    groups = payload["source_session_id"].astype(str)
    source_csv = payload["source_csv"].astype(str)
    source_activity = payload["source_activity"].astype(str)
    manifest_folds = manifest.set_index("source_session_id")["outer_fold"].astype(int).to_dict()
    rows: list[dict[str, object]] = []
    for session_id in sorted(np.unique(groups).tolist()):
        idx = np.flatnonzero(groups == session_id)
        csv_values = sorted(set(source_csv[idx].tolist()))
        activity_values = sorted(set(value for value in source_activity[idx].tolist() if value))
        if len(csv_values) != 1:
            raise ValueError(
                f"Reconstruction session {session_id} maps to {len(csv_values)} source CSV paths; expected one."
            )
        if len(activity_values) != 1:
            raise ValueError(
                f"Reconstruction session {session_id} has inconsistent source_activity values: {activity_values}"
            )
        if session_id not in manifest_folds:
            raise ValueError(f"Reconstruction session {session_id} is absent from the supplied manifest")
        rows.append(
            {
                "source_session_id": session_id,
                "source_csv": csv_values[0],
                "activity": activity_values[0],
                "outer_fold": manifest_folds[session_id],
            }
        )
    inventory = pd.DataFrame(rows)
    if set(inventory["source_session_id"].astype(str)) != set(manifest_folds):
        raise ValueError("Reconstruction inventory/session manifest mismatch after validation")
    return inventory


def derived_split_inventory(
    payload: dict[str, np.ndarray], manifest: pd.DataFrame, cfg: GroupedCVConfig
) -> Callable[[int], dict[str, set[str]]]:
    """Derive fold-local roles using the same grouped validation splitter.

    This is for a new reconstruction only.  When an existing E0 split
    inventory is available, use it instead to reproduce those exact roles.
    """

    groups = payload["source_session_id"].astype(str)
    y = payload["y"].astype(np.int64)
    fold_map = dict(zip(manifest["source_session_id"].astype(str), manifest["outer_fold"].astype(int)))

    def split_for_fold(fold: int) -> dict[str, set[str]]:
        test_sessions = {sid for sid, assigned in fold_map.items() if assigned == fold}
        indices = np.arange(len(y))
        test_idx = indices[np.asarray([sid in test_sessions for sid in groups])]
        train_pool_idx = indices[np.asarray([sid not in test_sessions for sid in groups])]
        train_idx, val_idx = grouped_validation_split(train_pool_idx, y, groups, cfg)
        result = {
            "train": set(groups[train_idx].tolist()),
            "val": set(groups[val_idx].tolist()),
            "test": set(groups[test_idx].tolist()),
        }
        overlap = (result["train"] & result["val"]) | (result["train"] & result["test"]) | (
            result["val"] & result["test"]
        )
        if overlap:
            raise ValueError(f"Derived source-session leakage in fold {fold}: {sorted(overlap)[:5]}")
        return result

    return split_for_fold


def saved_split_inventory(directory: Path) -> Callable[[int], dict[str, set[str]]]:
    def split_for_fold(fold: int) -> dict[str, set[str]]:
        path = directory / f"fold_{fold}" / "source_session_ids.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing saved E0 source-session inventory: {path}")
        frame = pd.read_csv(path)
        required = {"split", "source_session_id"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Saved inventory {path} is missing columns: {sorted(missing)}")
        result = {
            split: set(frame.loc[frame["split"].astype(str).eq(split), "source_session_id"].astype(str))
            for split in ("train", "val", "test")
        }
        overlap = (result["train"] & result["val"]) | (result["train"] & result["test"]) | (
            result["val"] & result["test"]
        )
        if overlap:
            raise ValueError(f"Saved source-session leakage in fold {fold}: {sorted(overlap)[:5]}")
        return result

    return split_for_fold


def _stable_hash_rank(*parts: object) -> int:
    """Return a deterministic tie-break rank without using model outputs."""

    text = ":".join(str(part) for part in parts)
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def select_steady_state_nonfall_start(
    *,
    session_id: str,
    activity: str,
    values: np.ndarray,
    length: int,
    stride_frames: int,
    seed: int,
) -> tuple[int, dict[str, object]]:
    """Select an interior, low-transition non-fall clip deterministically.

    The selector is intentionally clip-local: it uses only `x/y/z` frame
    differences from candidate clips in the same non-fall source session.
    It never changes features or labels.  Sitting and standing candidates are
    ranked for low movement; walking candidates retain the session's typical
    movement while avoiding a high-burst candidate.  If a session is too
    short for a full clip, edge padding remains the existing fallback.
    """

    if activity not in {"walking", "standing", "sitting"}:
        raise ValueError(f"steady_state_v1 is only defined for non-fall clips, got {activity!r}")
    if values.ndim != 2 or values.shape[1] < 3:
        raise ValueError(f"{session_id} must provide x/y/z for steady-state selection")
    n_frames = int(len(values))
    if n_frames < length:
        start = -(length - n_frames) // 2
        return start, {
            "nonfall_candidate_count": 0,
            "nonfall_interior_candidate_count": 0,
            "nonfall_selection_motion_median": np.nan,
            "nonfall_selection_motion_p95": np.nan,
            "nonfall_selection_active_fraction": np.nan,
            "nonfall_selection_reason": "short_session_edge_padding",
        }

    candidates = list(range(0, n_frames - length + 1, stride_frames))
    if not candidates:
        raise AssertionError("A full-length session must have at least one aligned candidate")
    # A 0.75-second guard removes start/stop portions where the recorded
    # session is long enough.  If there is no guarded candidate, retain all
    # candidates rather than silently dropping a source session.
    guard = min(stride_frames, max(0, (n_frames - length) // 2))
    interior = [start for start in candidates if start >= guard and start + length <= n_frames - guard]
    pool = interior or candidates

    xyz = values[:, :3].astype(np.float64, copy=False)
    session_motion = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    typical_walking_motion = float(np.median(session_motion)) if len(session_motion) else 0.0
    scored: list[tuple[tuple[float, ...], int, dict[str, object]]] = []
    for start in pool:
        motion = np.linalg.norm(np.diff(xyz[start : start + length], axis=0), axis=1)
        median = float(np.median(motion)) if len(motion) else 0.0
        p95 = float(np.quantile(motion, 0.95)) if len(motion) else 0.0
        active_fraction = float(np.mean(motion > 0.15)) if len(motion) else 0.0
        burst = max(0.0, p95 - median)
        tie = float(_stable_hash_rank(seed, length, session_id, start))
        if activity == "walking":
            # Preserve characteristic walking activity while preferring a
            # stable segment over an acceleration/deceleration edge.
            score = (abs(median - typical_walking_motion), burst, active_fraction, tie)
            reason = "typical_walking_motion_low_burst"
        else:
            score = (p95, active_fraction, median, tie)
            reason = "lowest_nonfall_motion_interior"
        scored.append(
            (
                score,
                start,
                {
                    "nonfall_candidate_count": len(candidates),
                    "nonfall_interior_candidate_count": len(interior),
                    "nonfall_selection_motion_median": median,
                    "nonfall_selection_motion_p95": p95,
                    "nonfall_selection_active_fraction": active_fraction,
                    "nonfall_selection_reason": reason,
                },
            )
        )
    _, start, details = min(scored, key=lambda item: item[0])
    return int(start), details


def install_nonfall_clip_policy(module: ModuleType, policy: str, *, seed: int) -> None:
    """Install an optional non-fall sampler without modifying recovered code."""

    if policy == "legacy_hash":
        return
    if policy != "steady_state_v1":
        raise ValueError(f"Unsupported non-fall clip policy: {policy}")

    def build_clip_dataset(
        length: int, sessions: pd.DataFrame, annotations: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
        annotation_map = annotations.set_index("source_session_id")
        clips: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        labels: list[int] = []
        metadata: list[dict[str, object]] = []
        for _, row in sessions.iterrows():
            sid = str(row["source_session_id"])
            activity = str(row["activity"])
            values = module.load_session(row)
            event_start = impact = event_end = np.nan
            confidence = quality_flags = ""
            nonfall_details: dict[str, object] = {
                "nonfall_candidate_count": np.nan,
                "nonfall_interior_candidate_count": np.nan,
                "nonfall_selection_motion_median": np.nan,
                "nonfall_selection_motion_p95": np.nan,
                "nonfall_selection_active_fraction": np.nan,
                "nonfall_selection_reason": "not_applicable_fall",
            }
            if activity == "fall":
                if sid not in annotation_map.index:
                    raise ValueError(f"Fall session lacks annotation: {sid}")
                ann = annotation_map.loc[sid]
                event_start = float(ann["event_start_s"])
                impact = float(ann["impact_s"])
                event_end = float(ann["event_end_s"])
                confidence = str(ann["confidence"])
                quality_flags = str(ann["quality_flags"])
                impact_frame = int(round(impact * module.RATE_HZ))
                start = impact_frame + module.POST_IMPACT_FRAMES[length] - length
                sampling_rule = "impact_plus_1s"
            else:
                start, nonfall_details = select_steady_state_nonfall_start(
                    session_id=sid,
                    activity=activity,
                    values=values,
                    length=length,
                    stride_frames=int(module.STRIDE_FRAMES),
                    seed=seed,
                )
                sampling_rule = f"steady_state_v1_{activity}"
            clip, mask, left, right = module.extract_clip(values, start, length)
            clips.append(clip)
            masks.append(mask)
            labels.append(module.CLASS_NAMES.index(activity))
            metadata.append(
                {
                    "source_session_id": sid,
                    "source_csv": str(row["source_csv"]),
                    "activity": activity,
                    "label": module.CLASS_NAMES.index(activity),
                    "outer_fold": int(row["outer_fold"]),
                    "clip_length_frames": length,
                    "clip_start_row": start,
                    "clip_end_row_exclusive": start + length,
                    "clip_start_s": start / module.RATE_HZ,
                    "clip_end_s": (start + length) / module.RATE_HZ,
                    "event_start_s": event_start,
                    "impact_s": impact,
                    "event_end_s": event_end,
                    "pre_event_context_s": event_start - start / module.RATE_HZ if activity == "fall" else np.nan,
                    "requested_post_impact_s": module.POST_IMPACT_FRAMES[length] / module.RATE_HZ if activity == "fall" else np.nan,
                    "physical_post_impact_s": max(
                        0.0, min(len(values) / module.RATE_HZ - impact, module.POST_IMPACT_FRAMES[length] / module.RATE_HZ)
                    )
                    if activity == "fall"
                    else np.nan,
                    "padding_left_frames": left,
                    "padding_right_frames": right,
                    "valid_frames": int(mask.sum()),
                    "annotation_confidence": confidence,
                    "quality_flags": quality_flags,
                    "sampling_rule": sampling_rule,
                    **nonfall_details,
                }
            )
        return (
            np.stack(clips).astype(np.float32),
            np.asarray(labels, dtype=np.int64),
            np.stack(masks),
            pd.DataFrame(metadata),
        )

    module.build_clip_dataset = build_clip_dataset


def _new_path(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing canonical evaluation artifact: {path}")


def _named_confusion_matrix(matrix: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame(matrix.astype(int), columns=CLASS_NAMES)
    frame.insert(0, "true_class", CLASS_NAMES)
    return frame


def _write_confusion_matrix_plot(matrix: np.ndarray, title: str, path: Path) -> None:
    _new_path(path)
    fig, ax = plt.subplots(figsize=(6.2, 5.3))
    image = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(len(CLASS_NAMES)),
        yticks=np.arange(len(CLASS_NAMES)),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        xlabel="Predicted class",
        ylabel="True class",
        title=title,
    )
    threshold = matrix.max() / 2.0 if matrix.size else 0.0
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
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[pd.DataFrame, dict[str, float]]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=CLASS_LABELS, zero_division=0
    )
    per_class = pd.DataFrame(
        {
            "class": CLASS_NAMES,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support.astype(int),
        }
    )
    macro = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    weighted = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    return per_class, {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro[0]),
        "macro_recall": float(macro[1]),
        "macro_f1": float(macro[2]),
        "weighted_precision": float(weighted[0]),
        "weighted_recall": float(weighted[1]),
        "weighted_f1": float(weighted[2]),
    }


def _canonical_predictions(checkpoint: dict, X: np.ndarray, test_idx: np.ndarray) -> np.ndarray:
    feature_names = [str(value) for value in checkpoint["base_feature_names"]]
    if feature_names != MODEL_FEATURES:
        raise ValueError(f"Canonical checkpoint base feature order differs from {MODEL_FEATURES}: {feature_names}")
    mean = np.asarray(checkpoint["normalization"]["mean"], dtype=np.float32)
    std = np.asarray(checkpoint["normalization"]["std"], dtype=np.float32)
    if mean.shape != (len(MODEL_FEATURES),) or std.shape != (len(MODEL_FEATURES),):
        raise ValueError("Checkpoint normalization has unexpected feature dimensions")
    normalized = ((X[test_idx] - mean) / std).astype(np.float32)
    dummy = normalized[:1]
    _, augmented, _, augmented_names = apply_augmented_features(
        dummy, normalized, dummy, np.asarray(MODEL_FEATURES), "augmented"
    )
    if list(augmented_names) != [str(value) for value in checkpoint["feature_names"]]:
        raise ValueError("Canonical augmented feature order differs from checkpoint")
    train_config = checkpoint["train_config"]
    model = CNNLSTM(
        n_features=augmented.shape[-1],
        n_classes=len(CLASS_NAMES),
        dropout_input=float(train_config["dropout_input"]),
        dropout_hidden=float(train_config["dropout_hidden"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    probabilities: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(augmented), 32):
            logits = model(torch.from_numpy(augmented[start : start + 32]).float())
            probabilities.append(torch.softmax(logits, dim=1).numpy())
    return np.concatenate(probabilities, axis=0).argmax(axis=1).astype(np.int64)


def _format_mean_std(frame: pd.DataFrame, metric: str) -> str:
    return f"{frame[metric].mean():.4f} +/- {frame[metric].std(ddof=1):.4f}"


def _canonical_report(length: int, summary: pd.DataFrame, aggregate: pd.DataFrame, matrix: np.ndarray) -> str:
    lines = [
        f"# {length}-Frame CNN-BiLSTM: Canonical Four-Class Evaluation Artifacts",
        "",
        f"Canonical clip classification only. Each held-out source session contributes one deterministic {length}-frame clip. The model prediction is four-class argmax in the fixed order `walking, standing, sitting, fall`. No causal threshold, alert rule, temporal smoothing, binary conversion, future clip, or full-session aggregation is used.",
        "",
        "## Mean +/- Std Across Outer Folds",
        "",
        "| Accuracy | Macro precision | Macro recall | Macro F1 | Weighted precision | Weighted recall | Weighted F1 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        f"| {_format_mean_std(summary, 'accuracy')} | {_format_mean_std(summary, 'macro_precision')} | {_format_mean_std(summary, 'macro_recall')} | {_format_mean_std(summary, 'macro_f1')} | {_format_mean_std(summary, 'weighted_precision')} | {_format_mean_std(summary, 'weighted_recall')} | {_format_mean_std(summary, 'weighted_f1')} |",
        "",
        "## Aggregate Per-Class Metrics",
        "",
        "| Class | Precision | Recall | F1 | Pooled support |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in aggregate.to_dict(orient="records"):
        lines.append(
            f"| {row['class']} | {row['precision_mean']:.4f} +/- {row['precision_std']:.4f} | "
            f"{row['recall_mean']:.4f} +/- {row['recall_std']:.4f} | "
            f"{row['f1_mean']:.4f} +/- {row['f1_std']:.4f} | {row['total_support']} |"
        )
    lines.extend(
        [
            "",
            "## Pooled Confusion Matrix",
            "",
            "Rows are true class; columns are predicted class.",
            "",
            "| True \\ Predicted | walking | standing | sitting | fall |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for class_name, row in zip(CLASS_NAMES, matrix):
        lines.append(f"| {class_name} | " + " | ".join(str(int(value)) for value in row) + " |")
    lines.extend(
        [
            "",
            "## Protocol Checks",
            "",
            "- Every source session is evaluated once as an outer-fold held-out canonical clip.",
            "- Train/validation/test source-session overlap is zero in every fold.",
            "- Each checkpoint's fold-local training normalization is used unchanged for held-out inference.",
            "- Edge-padded clips are consumed exactly as by the CNN training/inference implementation; padding counts are recorded per fold.",
            "- These matrices are supplementary to, and must not be confused with, causal trailing-clip fall-alert metrics in `fold_metrics.csv` and `aggregate_metrics.csv`.",
            "",
        ]
    )
    return "\n".join(lines)


def _generate_canonical_evaluation_for_length(
    output_dir: Path,
    split_for_fold: Callable[[int], dict[str, set[str]]],
    length: int,
    model_root: Path,
) -> Path:
    """Write canonical four-class matrices for one trained clip length."""

    source = output_dir / f"staging_clip_dataset_{length}.npz"
    metadata_path = output_dir / f"staging_clip_metadata_{length}.csv"
    if not source.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"{length}-frame canonical dataset/metadata is missing after training")
    data = np.load(source, allow_pickle=True)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)
    mask = data["mask"].astype(bool)
    source_ids = data["source_session_id"].astype(str)
    feature_names = [str(value) for value in data["feature_names"]]
    metadata = pd.read_csv(metadata_path)
    if X.ndim != 3 or X.shape[1:] != (length, len(MODEL_FEATURES)):
        raise ValueError(f"Unexpected {length}-frame canonical tensor shape: {X.shape}")
    if mask.shape != X.shape[:2] or feature_names != MODEL_FEATURES:
        raise ValueError(f"{length}-frame canonical dataset mask or feature order is invalid")
    if len(metadata) != len(X) or not np.array_equal(source_ids, metadata["source_session_id"].astype(str).to_numpy()):
        raise ValueError(f"{length}-frame canonical metadata order differs from the saved dataset")
    if metadata["source_session_id"].astype(str).duplicated().any():
        raise ValueError("Canonical dataset must contain exactly one clip per source session")

    model_root.mkdir(exist_ok=False)
    per_fold_root = model_root / "per_fold"
    per_fold_root.mkdir(exist_ok=False)
    fold_rows: list[dict[str, object]] = []
    class_rows: list[pd.DataFrame] = []
    matrices: list[np.ndarray] = []
    seen_test_sessions: set[str] = set()
    for fold in range(1, 5):
        splits = split_for_fold(fold)
        overlap = (splits["train"] & splits["val"]) | (splits["train"] & splits["test"]) | (
            splits["val"] & splits["test"]
        )
        if overlap:
            raise ValueError(f"Canonical fold {fold} has source-session leakage: {sorted(overlap)[:5]}")
        test_ids = splits["test"]
        if seen_test_sessions & test_ids:
            raise ValueError(f"Canonical test sessions repeat across outer folds: {sorted(seen_test_sessions & test_ids)[:5]}")
        seen_test_sessions |= test_ids
        test_idx = np.flatnonzero(metadata["source_session_id"].astype(str).isin(test_ids).to_numpy())
        if set(metadata.iloc[test_idx]["source_session_id"].astype(str)) != test_ids:
            raise ValueError(f"Canonical metadata and fold {fold} held-out session inventory differ")
        if not metadata.iloc[test_idx]["outer_fold"].astype(int).eq(fold).all():
            raise ValueError(f"Canonical metadata outer-fold assignment differs in fold {fold}")
        checkpoint_path = output_dir / f"{length}_frame" / f"fold_{fold}" / "cnn_lstm_event_centered.pt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"{length}-frame checkpoint is missing: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if int(checkpoint["outer_fold"]) != fold or int(checkpoint["clip_length_frames"]) != length:
            raise ValueError(f"Checkpoint fold/length mismatch: {checkpoint_path}")
        train_idx = np.flatnonzero(metadata["source_session_id"].astype(str).isin(splits["train"]).to_numpy())
        train_flat = X[train_idx].reshape(-1, X.shape[-1])
        mean = train_flat.mean(axis=0)
        std = np.where(train_flat.std(axis=0) == 0, 1.0, train_flat.std(axis=0))
        normalization_error = max(
            float(np.max(np.abs(mean - np.asarray(checkpoint["normalization"]["mean"], dtype=float)))),
            float(np.max(np.abs(std - np.asarray(checkpoint["normalization"]["std"], dtype=float)))),
        )
        if normalization_error > 1e-7:
            raise ValueError(f"Fold {fold} normalization is not training-only reproducible: {normalization_error}")
        y_true = y[test_idx]
        y_pred = _canonical_predictions(checkpoint, X, test_idx)
        per_class, summary = _per_class_metrics(y_true, y_pred)
        matrix = confusion_matrix(y_true, y_pred, labels=CLASS_LABELS)
        fold_dir = per_fold_root / f"fold_{fold}"
        fold_dir.mkdir(exist_ok=False)
        _named_confusion_matrix(matrix).to_csv(fold_dir / "confusion_matrix.csv", index=False)
        _write_confusion_matrix_plot(matrix, f"{length}-frame CNN-BiLSTM — Fold {fold}", fold_dir / "confusion_matrix.png")
        per_class.to_csv(fold_dir / "per_class_metrics.csv", index=False)
        class_rows.append(per_class.assign(fold=fold))
        matrices.append(matrix)
        fold_rows.append(
            {
                "fold": fold,
                "heldout_sessions": int(len(test_ids)),
                "source_session_overlap_count": 0,
                "checkpoint": str(checkpoint_path),
                "normalization_source": f"{checkpoint_path}::normalization",
                "normalization_fit": str(checkpoint["normalization_fit"]),
                "normalization_reproduction_max_abs_error": normalization_error,
                "padding_mask_handling": "saved edge-padded canonical clips used exactly as CNN inference; mask audited",
                "padded_heldout_clips": int((~mask[test_idx]).any(axis=1).sum()),
                "minimum_valid_frames": int(mask[test_idx].sum(axis=1).min()),
                **summary,
            }
        )
    expected_sessions = set(metadata["source_session_id"].astype(str))
    if seen_test_sessions != expected_sessions:
        raise ValueError("Canonical held-out fold union does not cover every source session exactly once")
    summary = pd.DataFrame(fold_rows)
    per_class_all = pd.concat(class_rows, ignore_index=True)
    aggregate = pd.DataFrame(
        [
            {
                "class": class_name,
                "precision_mean": float(group["precision"].mean()),
                "precision_std": float(group["precision"].std(ddof=1)),
                "recall_mean": float(group["recall"].mean()),
                "recall_std": float(group["recall"].std(ddof=1)),
                "f1_mean": float(group["f1"].mean()),
                "f1_std": float(group["f1"].std(ddof=1)),
                "total_support": int(group["support"].sum()),
            }
            for class_name, group in per_class_all.groupby("class", sort=False)
        ]
    )
    aggregate["class"] = pd.Categorical(aggregate["class"], categories=CLASS_NAMES, ordered=True)
    aggregate = aggregate.sort_values("class").reset_index(drop=True)
    pooled = np.sum(matrices, axis=0)
    _named_confusion_matrix(pooled).to_csv(model_root / "pooled_confusion_matrix.csv", index=False)
    _write_confusion_matrix_plot(pooled, f"{length}-frame CNN-BiLSTM — Pooled Held-Out Folds", model_root / "pooled_confusion_matrix.png")
    summary.to_csv(model_root / "per_fold_summary.csv", index=False)
    aggregate.to_csv(model_root / "aggregate_per_class_metrics.csv", index=False)
    (model_root / "EVALUATION_ARTIFACTS_REPORT.md").write_text(
        _canonical_report(length, summary, aggregate, pooled), encoding="utf-8"
    )
    return model_root


def generate_canonical_evaluation_artifacts(
    output_dir: Path, split_for_fold: Callable[[int], dict[str, set[str]]], lengths: Iterable[int]
) -> dict[int, Path]:
    """Write canonical four-class bundles for all requested trained branches."""

    artifacts_root = output_dir / CANONICAL_EVALUATION_DIR
    _new_path(artifacts_root)
    artifacts_root.mkdir(parents=True, exist_ok=False)
    outputs: dict[int, Path] = {}
    for length in lengths:
        numeric_length = int(length)
        outputs[numeric_length] = _generate_canonical_evaluation_for_length(
            output_dir,
            split_for_fold,
            numeric_length,
            artifacts_root / f"cnn_bilstm_{numeric_length}_frame",
        )
    return outputs


def reconcile_recovered_report(
    output_dir: Path,
    *,
    derive_splits: bool,
    nonfall_clip_policy: str,
    clip_lengths: tuple[int, ...],
) -> None:
    """Replace recovered-runner provenance prose with the actual wrapper protocol."""

    report_path = output_dir / "REPORT.md"
    if not report_path.is_file():
        raise FileNotFoundError("Recovered runner did not write REPORT.md")
    text = report_path.read_text(encoding="utf-8")
    split_text = (
        "The supplied frozen source-session SGKF4 outer assignments were used; "
        "grouped train/validation roles were derived from the supplied manifest."
        if derive_splits
        else "The supplied frozen source-session SGKF4 outer assignments and saved train/validation/test inventories were reused."
    )
    old_intro = (
        "This is a staging-only experiment. The frozen source-session SGKF4 outer assignments and E0 "
        "train/validation/test session inventories were reused without change. Normalization was fitted "
        "separately on each fold's real training clips. No synthetic, height-cluster, or session-height "
        "features were used."
    )
    new_intro = (
        f"This is a staging-only experiment. {split_text} Normalization was fitted separately on each "
        "fold's real training clips. No synthetic, height-cluster, or session-height features were used."
    )
    if old_intro not in text:
        raise ValueError("Recovered report introduction did not match the expected provenance text")
    text = text.replace(old_intro, new_intro, 1)
    requested = ", ".join(f"{length}-frame" for length in clip_lengths)
    old_sampling = (
        "The 50-frame fall clips end 0.75 seconds after impact and the 60-frame clips end 1.0 second after "
        "impact when available; missing boundary context is edge padded and recorded by a mask. Non-fall starts "
        "are selected by SHA-256 of seed 42, clip length, and source-session ID from 15-frame-aligned candidates. "
        "No clip crosses a session boundary."
    )
    if nonfall_clip_policy == "legacy_hash":
        nonfall_text = (
            "Non-fall starts use the historical SHA-256 seed-42, clip-length, source-session selection "
            "from 15-frame-aligned candidates."
        )
    else:
        nonfall_text = (
            "Non-fall starts use `steady_state_v1`: same-session 15-frame-aligned guarded candidates are "
            "ranked using clip-local x/y/z motion; fall anchoring is unchanged."
        )
    new_sampling = (
        f"The requested branches are {requested}. Fall clips request one second of post-impact context when "
        "available; missing boundary context is edge padded and recorded by a mask. "
        f"{nonfall_text} No clip crosses a session boundary."
    )
    if old_sampling not in text:
        raise ValueError("Recovered report sampling description did not match the expected text")
    report_path.write_text(text.replace(old_sampling, new_sampling, 1), encoding="utf-8")


def run(
    *,
    dataset: Path,
    manifest: Path,
    annotations: Path,
    output_dir: Path,
    reference_runner: Path,
    split_inventory_dir: Path | None,
    derive_splits: bool,
    expected_manifest_sha256: str | None,
    seed: int,
    val_folds: int,
    nonfall_clip_policy: str = "legacy_hash",
    clip_lengths: tuple[int, ...] = (50, 60),
) -> None:
    # The recovered implementation records provenance with
    # ``Path.relative_to(ROOT)``.  Canonicalize caller-supplied paths before
    # injecting them so both relative and absolute CLI arguments work.
    dataset = dataset.resolve()
    manifest = manifest.resolve()
    annotations = annotations.resolve()
    output_dir = output_dir.resolve()
    reference_runner = reference_runner.resolve()
    if split_inventory_dir is not None:
        split_inventory_dir = split_inventory_dir.resolve()
    if seed != 42 or val_folds != 6:
        raise ValueError(
            "The recovered implementation fixes the grouped protocol at seed=42 and val_folds=6; "
            f"got seed={seed}, val_folds={val_folds}."
        )
    for path, label in ((dataset, "Dataset"), (manifest, "Manifest"), (annotations, "Annotation CSV")):
        if not path.is_file():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}. Use a new dated output directory.")
    if bool(split_inventory_dir) == bool(derive_splits):
        raise ValueError("Specify exactly one of --split-inventory-dir or --derive-splits.")
    if nonfall_clip_policy not in NONFALL_CLIP_POLICIES:
        raise ValueError(
            f"Unsupported --nonfall-clip-policy {nonfall_clip_policy!r}; "
            f"choose one of {list(NONFALL_CLIP_POLICIES)}."
        )
    if not clip_lengths or any(length not in SUPPORTED_CLIP_LENGTHS for length in clip_lengths):
        raise ValueError(f"Unsupported clip_lengths: {clip_lengths}")
    actual_manifest_sha256 = sha256_file(manifest)
    if expected_manifest_sha256 and actual_manifest_sha256 != expected_manifest_sha256:
        raise ValueError(
            f"Manifest SHA-256 mismatch: expected {expected_manifest_sha256}, got {actual_manifest_sha256}"
        )

    payload, manifest_frame, grouped_cfg = validate_reconstruction_inputs(
        dataset, manifest, seed=seed, val_folds=val_folds
    )
    module = load_recovered_runner(reference_runner)
    module.ROOT = ROOT
    module.OUTPUT = output_dir
    module.DATASET = dataset
    module.MANIFEST = manifest
    module.ANNOTATIONS = annotations
    module.CLIP_LENGTHS = tuple(clip_lengths)
    module.POST_IMPACT_FRAMES = {length: SUPPORTED_CLIP_LENGTHS[length] for length in clip_lengths}
    # The recovered implementation validates this global before running.  For a
    # new, explicitly supplied manifest, pin it to the supplied file's hash.
    module.EXPECTED_MANIFEST_SHA256 = actual_manifest_sha256
    if split_inventory_dir is not None:
        module.E0 = split_inventory_dir
        split_for_fold = saved_split_inventory(split_inventory_dir)
        module.frozen_split_sessions = split_for_fold
    else:
        # The recovered main() calls inventory() before training.  Replace only
        # that historical 485-session/path assertion with the already
        # validated supplied reconstruction inventory; all downstream training
        # and evaluation code remains the recovered implementation.
        module.inventory = lambda: reconstruction_inventory(payload, manifest_frame)
        split_for_fold = derived_split_inventory(payload, manifest_frame, grouped_cfg)
        module.frozen_split_sessions = split_for_fold
    install_nonfall_clip_policy(module, nonfall_clip_policy, seed=seed)
    output_dir.mkdir(parents=True, exist_ok=False)
    try:
        module.main()
        provenance_path = output_dir / "provenance.json"
        if not provenance_path.is_file():
            raise FileNotFoundError("Recovered runner did not write provenance.json")
        provenance: dict[str, Any] = json.loads(provenance_path.read_text(encoding="utf-8"))
        provenance["nonfall_clip_policy"] = nonfall_clip_policy
        provenance["nonfall_clip_policy_definition"] = (
            "historical SHA-256 seed-42 stride-15 selection"
            if nonfall_clip_policy == "legacy_hash"
            else "interior candidate selection with clip-local x/y/z motion stability; fall anchoring unchanged"
        )
        provenance["wrapper_requested_clip_lengths"] = list(clip_lengths)
        provenance["wrapper_post_impact_frames"] = {
            str(length): SUPPORTED_CLIP_LENGTHS[length] for length in clip_lengths
        }
        provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
        reconcile_recovered_report(
            output_dir,
            derive_splits=derive_splits,
            nonfall_clip_policy=nonfall_clip_policy,
            clip_lengths=clip_lengths,
        )
        canonical_outputs = generate_canonical_evaluation_artifacts(output_dir, split_for_fold, clip_lengths)
        for length, canonical_output in canonical_outputs.items():
            print(f"canonical_evaluation_artifacts_{length}_frame={canonical_output}", flush=True)
    except Exception:
        # Preserve partial output for diagnosis; never conceal a failed run by
        # deleting artifacts that may explain the error.
        raise


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run the recovered event-centred 50/60-frame SGKF4 implementation with explicit inputs."
    )
    parser.add_argument("--dataset", required=True, help="Filtered radar_dataset.npz produced by the risky-fall builder.")
    parser.add_argument("--manifest", required=True, help="Frozen source-session SGKF4 manifest CSV.")
    parser.add_argument("--annotations", required=True, help="Auto-event annotation CSV used for the dataset.")
    parser.add_argument("--output-dir", required=True, help="New dated experiment output directory.")
    parser.add_argument("--reference-runner", default=str(RECOVERED_REFERENCE_RUNNER))
    inventory = parser.add_mutually_exclusive_group(required=True)
    inventory.add_argument("--split-inventory-dir", help="E0 output directory containing fold_N/source_session_ids.csv.")
    inventory.add_argument("--derive-splits", action="store_true", help="Derive grouped train/validation/test session roles from the manifest.")
    parser.add_argument("--expected-manifest-sha256", help="Optional SHA-256 pin for the supplied manifest.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for --derive-splits only.")
    parser.add_argument("--val-folds", type=int, default=6, help="Inner grouped folds for --derive-splits only.")
    parser.add_argument(
        "--nonfall-clip-policy",
        choices=NONFALL_CLIP_POLICIES,
        default="legacy_hash",
        help="Non-fall canonical clip selection. legacy_hash preserves historical replay; steady_state_v1 avoids edge transitions.",
    )
    parser.add_argument(
        "--clip-lengths",
        default="50,60",
        help="Comma-separated supported frame lengths. Default 50,60 preserves the recovered comparison; use 60,100 for the fixed-policy ablation.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    run(
        dataset=Path(args.dataset),
        manifest=Path(args.manifest),
        annotations=Path(args.annotations),
        output_dir=Path(args.output_dir),
        reference_runner=Path(args.reference_runner),
        split_inventory_dir=Path(args.split_inventory_dir) if args.split_inventory_dir else None,
        derive_splits=bool(args.derive_splits),
        expected_manifest_sha256=args.expected_manifest_sha256,
        seed=args.seed,
        val_folds=args.val_folds,
        nonfall_clip_policy=args.nonfall_clip_policy,
        clip_lengths=parse_clip_lengths(args.clip_lengths),
    )


if __name__ == "__main__":
    main()
