from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "radar_mpl_cache"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from radar_pipeline.common import CLASS_LABELS, CLASS_NAMES, ROOT, activity_from_path
from radar_pipeline.train_model import CNNLSTM, TrainConfig, augment_window_features, class_weights, device, evaluate, make_loader, plot_history, set_seed


SPLIT_PROTOCOL = "sgkf_grouped_v1_seed42_k4"
DATED_SPLIT_PROTOCOL_RE = re.compile(r"^sgkf_grouped_(20\d{6})_seed(?P<seed>\d+)_k(?P<folds>\d+)$")
DEFAULT_DATASET = ROOT / "data/final_dataset_auto_event_staging/radar_dataset.npz"
DEFAULT_MANIFEST = ROOT / "data/metadata/auto_event_aware_v1_source_session_folds.csv"
DEFAULT_OUTPUT = ROOT / "outputs/experiments/auto_event_aware_sgkf4_baseline"


@dataclass
class GroupedCVConfig:
    epochs: int = 80
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 12
    seed: int = 42
    folds: int = 4
    val_folds: int = 6
    feature_mode: str = "augmented"
    dropout_input: float = 0.25
    dropout_hidden: float = 0.20


def source_session_id(path: str) -> str:
    return Path(str(path)).stem


def load_all_windows(dataset_path: Path) -> dict[str, np.ndarray]:
    data = np.load(dataset_path, allow_pickle=True)
    payload: dict[str, np.ndarray] = {
        "feature_names": data["feature_names"],
        "label_names": data["label_names"],
    }
    keys = [
        "X",
        "y",
        "activity",
        "source_activity",
        "session_id",
        "source_csv",
        "start_frame",
        "end_frame",
        "window_start_s",
        "window_end_s",
    ]
    for key in keys:
        payload[key] = np.concatenate([data[f"{key}_{split}"] for split in ("train", "val", "test")], axis=0)
    payload["source_session_id"] = np.asarray([source_session_id(x) for x in payload["source_csv"]], dtype=str)
    return payload


def normalize_by_train(X: np.ndarray, train_idx: np.ndarray) -> tuple[np.ndarray, dict[str, list[float]]]:
    train = X[train_idx]
    flat = train.reshape(-1, train.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    return ((X - mean) / std).astype(np.float32), {"mean": mean.tolist(), "std": std.tolist()}


def apply_augmented_features(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    feature_names: np.ndarray,
    mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    names = [str(x) for x in feature_names]
    if mode == "raw":
        return X_train, X_val, X_test, names
    if mode != "augmented":
        raise ValueError(f"Unsupported feature mode for this baseline: {mode}")
    augmented, augmented_names = augment_window_features(
        {
            "feature_names": feature_names,
            "X_train": X_train,
            "X_val": X_val,
            "X_test": X_test,
        }
    )
    return augmented["X_train"], augmented["X_val"], augmented["X_test"], augmented_names


def _session_activity(values: np.ndarray, source_csv: np.ndarray) -> str:
    nonempty = [str(v) for v in values if str(v)]
    if nonempty:
        return str(pd.Series(nonempty).mode().iloc[0])
    return activity_from_path(str(source_csv[0]))


def generate_manifest(payload: dict[str, np.ndarray], manifest_path: Path, cfg: GroupedCVConfig) -> pd.DataFrame:
    groups = payload["source_session_id"]
    y = payload["y"].astype(int)
    splitter = StratifiedGroupKFold(n_splits=cfg.folds, shuffle=True, random_state=cfg.seed)
    rows: list[dict[str, object]] = []
    for fold_idx, (_, test_idx) in enumerate(splitter.split(np.arange(len(y)), y, groups), start=1):
        for sid in sorted(np.unique(groups[test_idx]).tolist()):
            sid_idx = np.flatnonzero(groups == sid)
            rows.append(
                {
                    "source_session_id": sid,
                    "source_activity": _session_activity(payload["source_activity"][sid_idx], payload["source_csv"][sid_idx]),
                    "outer_fold": fold_idx,
                    "split_protocol": SPLIT_PROTOCOL,
                }
            )
    manifest = pd.DataFrame(rows).sort_values("source_session_id").reset_index(drop=True)
    validate_manifest(payload, manifest, cfg)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)
    return manifest


def load_or_create_manifest(dataset_path: Path, manifest_path: Path, cfg: GroupedCVConfig, force: bool = False) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    payload = load_all_windows(dataset_path)
    if manifest_path.exists() and not force:
        manifest = pd.read_csv(manifest_path)
        validate_manifest(payload, manifest, cfg)
        return payload, manifest
    manifest = generate_manifest(payload, manifest_path, cfg)
    return payload, manifest


def validate_manifest(payload: dict[str, np.ndarray], manifest: pd.DataFrame, cfg: GroupedCVConfig) -> None:
    required = {"source_session_id", "source_activity", "outer_fold", "split_protocol"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")
    expected_sessions = set(np.unique(payload["source_session_id"]).tolist())
    manifest_sessions = manifest["source_session_id"].astype(str).tolist()
    if len(manifest_sessions) != len(set(manifest_sessions)):
        raise ValueError("Manifest has duplicate source_session_id values")
    if set(manifest_sessions) != expected_sessions:
        missing_sessions = sorted(expected_sessions - set(manifest_sessions))
        extra_sessions = sorted(set(manifest_sessions) - expected_sessions)
        raise ValueError(f"Manifest/session mismatch. missing={missing_sessions[:5]} extra={extra_sessions[:5]}")
    _manifest_protocol(manifest, cfg)
    if sorted(manifest["outer_fold"].astype(int).unique().tolist()) != list(range(1, cfg.folds + 1)):
        raise ValueError("Manifest outer_fold values must be 1..4")
    groups = payload["source_session_id"].astype(str)
    y = payload["y"].astype(int)
    fold_map = dict(zip(manifest["source_session_id"].astype(str), manifest["outer_fold"].astype(int)))
    for fold in range(1, cfg.folds + 1):
        test_sessions = {sid for sid, assigned in fold_map.items() if assigned == fold}
        train_sessions = set(fold_map) - test_sessions
        if train_sessions & test_sessions:
            raise ValueError(f"Leakage in fold {fold}")
        test_idx = np.asarray([sid in test_sessions for sid in groups])
        present = set(y[test_idx].tolist())
        if present != set(CLASS_LABELS):
            raise ValueError(f"Fold {fold} lacks effective classes: {sorted(set(CLASS_LABELS) - present)}")


def _manifest_protocol(manifest: pd.DataFrame, cfg: GroupedCVConfig) -> str:
    protocols = set(manifest["split_protocol"].astype(str))
    if len(protocols) != 1:
        raise ValueError(f"Manifest must contain exactly one split_protocol, found {sorted(protocols)}")
    protocol = next(iter(protocols))
    if protocol == SPLIT_PROTOCOL:
        if cfg.seed != 42 or cfg.folds != 4:
            raise ValueError(f"Frozen v1 protocol requires seed=42 and folds=4, got seed={cfg.seed} folds={cfg.folds}")
        return protocol
    match = DATED_SPLIT_PROTOCOL_RE.match(protocol)
    if not match:
        raise ValueError(
            "Manifest split_protocol must be the frozen v1 protocol or a date-versioned "
            "`sgkf_grouped_YYYYMMDD_seed{seed}_k{folds}` protocol"
        )
    seed = int(match.group("seed"))
    folds = int(match.group("folds"))
    if seed != cfg.seed or folds != cfg.folds:
        raise ValueError(
            f"Manifest protocol {protocol} does not match evaluator configuration seed={cfg.seed}, folds={cfg.folds}"
        )
    return protocol


def manifest_checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fold_count_table(payload: dict[str, np.ndarray], manifest: pd.DataFrame) -> pd.DataFrame:
    fold_map = dict(zip(manifest["source_session_id"].astype(str), manifest["outer_fold"].astype(int)))
    rows = []
    groups = payload["source_session_id"].astype(str)
    for fold in sorted(manifest["outer_fold"].astype(int).unique()):
        mask = np.asarray([fold_map[sid] == fold for sid in groups])
        for label, name in zip(CLASS_LABELS, CLASS_NAMES):
            label_mask = mask & (payload["y"].astype(int) == label)
            rows.append(
                {
                    "outer_fold": int(fold),
                    "class": name,
                    "effective_session_count": int(len(np.unique(groups[label_mask]))),
                    "effective_window_count": int(label_mask.sum()),
                }
            )
    return pd.DataFrame(rows)


def write_manifest_readme(dataset_path: Path, manifest_path: Path, payload: dict[str, np.ndarray], manifest: pd.DataFrame) -> Path:
    protocol = _manifest_protocol(manifest, GroupedCVConfig())
    counts = fold_count_table(payload, manifest)
    source_inventory = (
        pd.DataFrame({"source_session_id": payload["source_session_id"], "source_activity": payload["source_activity"]})
        .drop_duplicates()
        .groupby("source_activity")
        .size()
        .to_dict()
    )
    checksum = manifest_checksum(manifest_path)
    lines = [
        "# Auto-Event-Aware V1 Source Session Folds",
        "",
        f"- Dataset: `{dataset_path}`",
        f"- Manifest: `{manifest_path}`",
        f"- Split protocol: `{protocol}`",
        "- Splitter: `StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)`",
        "- Group key: `Path(source_csv).stem`",
        f"- Source sessions: {len(manifest)}",
        f"- SHA-256: `{checksum}`",
        "",
        "All windows from a source recording are assigned to one outer fold, including pre-event standing and fall-event windows from the same fall recording.",
        "",
        "## Source Inventory",
        "",
        "| Source activity | Sessions |",
        "|---|---:|",
    ]
    for activity in CLASS_NAMES:
        lines.append(f"| {activity} | {int(source_inventory.get(activity, 0))} |")
    lines.extend(["", "## Fold Counts", "", "| Fold | Class | Effective sessions | Windows |", "|---:|---|---:|---:|"])
    for _, row in counts.iterrows():
        lines.append(
            f"| {int(row['outer_fold'])} | {row['class']} | {int(row['effective_session_count'])} | {int(row['effective_window_count'])} |"
        )
    readme = manifest_path.with_name(f"{manifest_path.stem}_README.md")
    readme.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return readme


def grouped_validation_split(train_pool_idx: np.ndarray, y: np.ndarray, groups: np.ndarray, cfg: GroupedCVConfig) -> tuple[np.ndarray, np.ndarray]:
    splitter = StratifiedGroupKFold(n_splits=cfg.val_folds, shuffle=True, random_state=cfg.seed)
    rel = np.arange(len(train_pool_idx))
    train_rel, val_rel = next(splitter.split(rel, y[train_pool_idx], groups[train_pool_idx]))
    return train_pool_idx[train_rel], train_pool_idx[val_rel]


def save_confusion_matrix_plot(cm: np.ndarray, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(len(CLASS_NAMES)), labels=CLASS_NAMES, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(CLASS_NAMES)), labels=CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", color="black")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def error_counts(cm: np.ndarray) -> dict[str, int]:
    fall_idx = CLASS_NAMES.index("fall")
    standing_idx = CLASS_NAMES.index("standing")
    sitting_idx = CLASS_NAMES.index("sitting")
    return {
        "fall_false_positives": int(cm[:, fall_idx].sum() - cm[fall_idx, fall_idx]),
        "fall_false_negatives": int(cm[fall_idx, :].sum() - cm[fall_idx, fall_idx]),
        "sitting_to_standing": int(cm[sitting_idx, standing_idx]),
        "standing_to_sitting": int(cm[standing_idx, sitting_idx]),
    }


def metrics_from_predictions(y_true: np.ndarray, y_pred: np.ndarray, prefix: str = "") -> dict[str, object]:
    report = classification_report(y_true, y_pred, labels=CLASS_LABELS, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_LABELS)
    out: dict[str, object] = {
        f"{prefix}accuracy": float(accuracy_score(y_true, y_pred)),
        f"{prefix}macro_f1": float(report["macro avg"]["f1-score"]),
        f"{prefix}weighted_f1": float(report["weighted avg"]["f1-score"]),
        f"{prefix}confusion_matrix": cm.tolist(),
        **{f"{prefix}{k}": v for k, v in error_counts(cm).items()},
    }
    for name in CLASS_NAMES:
        metrics = report[name]
        out[f"{prefix}{name}_precision"] = float(metrics["precision"])
        out[f"{prefix}{name}_recall"] = float(metrics["recall"])
        out[f"{prefix}{name}_f1"] = float(metrics["f1-score"])
        out[f"{prefix}{name}_support"] = int(metrics["support"])
    return out


def session_level_predictions(y_true: np.ndarray, y_prob: np.ndarray, groups: np.ndarray, source_activity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    true_by_session: list[int] = []
    pred_by_session: list[int] = []
    source_label = {name: label for label, name in zip(CLASS_LABELS, CLASS_NAMES)}
    for sid in sorted(np.unique(groups).tolist()):
        idx = np.flatnonzero(groups == sid)
        activity = _session_activity(source_activity[idx], np.asarray([sid] * len(idx)))
        true_by_session.append(source_label[activity])
        pred_by_session.append(int(np.mean(y_prob[idx], axis=0).argmax()))
    return np.asarray(true_by_session, dtype=np.int64), np.asarray(pred_by_session, dtype=np.int64)


def train_one_fold(
    fold: int,
    payload: dict[str, np.ndarray],
    manifest: pd.DataFrame,
    cfg: GroupedCVConfig,
    output_root: Path,
    dev: torch.device,
) -> dict[str, object]:
    protocol = _manifest_protocol(manifest, cfg)
    set_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    groups = payload["source_session_id"].astype(str)
    y = payload["y"].astype(np.int64)
    fold_sessions = set(manifest.loc[manifest["outer_fold"].astype(int) == fold, "source_session_id"].astype(str))
    all_idx = np.arange(len(y))
    test_idx = all_idx[np.asarray([sid in fold_sessions for sid in groups])]
    train_pool_idx = all_idx[np.asarray([sid not in fold_sessions for sid in groups])]
    train_idx, val_idx = grouped_validation_split(train_pool_idx, y, groups, cfg)

    leakage = sorted((set(groups[train_idx]) | set(groups[val_idx])) & set(groups[test_idx]))
    if leakage:
        raise ValueError(f"Fold {fold} source-session leakage: {leakage[:5]}")

    X_norm, norm_params = normalize_by_train(payload["X"].astype(np.float32), train_idx)
    X_train, X_val, X_test, feature_names = apply_augmented_features(
        X_norm[train_idx],
        X_norm[val_idx],
        X_norm[test_idx],
        payload["feature_names"],
        cfg.feature_mode,
    )
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

    train_loader = make_loader(X_train, y_train, cfg.batch_size, shuffle=True)
    val_loader = make_loader(X_val, y_val, cfg.batch_size, shuffle=False)
    test_loader = make_loader(X_test, y_test, cfg.batch_size, shuffle=False)
    model = CNNLSTM(
        n_features=X_train.shape[-1],
        n_classes=len(CLASS_LABELS),
        dropout_input=cfg.dropout_input,
        dropout_hidden=cfg.dropout_hidden,
    ).to(dev)
    weights = class_weights(y_train, dev)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    fold_dir = output_root / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    history: list[dict[str, float]] = []
    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    stale = 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0
        train_true: list[int] = []
        train_pred: list[int] = []
        for xb, yb in train_loader:
            xb = xb.to(dev)
            yb = yb.to(dev)
            optim.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optim.step()
            total_loss += float(loss.item()) * len(yb)
            total += len(yb)
            train_true.extend(yb.detach().cpu().numpy().tolist())
            train_pred.extend(logits.argmax(dim=1).detach().cpu().numpy().tolist())
        train_loss = total_loss / max(total, 1)
        train_acc = float(accuracy_score(train_true, train_pred))
        val_loss, val_acc, _, _, _ = evaluate(model, val_loader, loss_fn, dev)
        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc})
        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= cfg.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, _, y_true, y_pred, y_prob = evaluate(model, test_loader, loss_fn, dev)
    window_metrics = metrics_from_predictions(y_true, y_pred, prefix="window_")
    session_true, session_pred = session_level_predictions(
        y_true,
        y_prob,
        groups[test_idx],
        payload["source_activity"][test_idx],
    )
    session_metrics = metrics_from_predictions(session_true, session_pred, prefix="session_")
    cm = np.asarray(window_metrics["window_confusion_matrix"], dtype=int)
    pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(fold_dir / "confusion_matrix.csv")
    save_confusion_matrix_plot(cm, fold_dir / "confusion_matrix.png", f"Fold {fold} Window Confusion Matrix")
    session_cm = np.asarray(session_metrics["session_confusion_matrix"], dtype=int)
    pd.DataFrame(session_cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(fold_dir / "session_confusion_matrix.csv")
    save_confusion_matrix_plot(session_cm, fold_dir / "session_confusion_matrix.png", f"Fold {fold} Session Confusion Matrix")
    per_class_rows = []
    for name in CLASS_NAMES:
        per_class_rows.append(
            {
                "class": name,
                "precision": window_metrics[f"window_{name}_precision"],
                "recall": window_metrics[f"window_{name}_recall"],
                "f1": window_metrics[f"window_{name}_f1"],
                "support": window_metrics[f"window_{name}_support"],
            }
        )
    pd.DataFrame(per_class_rows).to_csv(fold_dir / "per_class_metrics.csv", index=False)
    plot_history(history, fold_dir / "training_curve.png")

    split_rows = []
    for split_name, idx in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        for sid in sorted(np.unique(groups[idx]).tolist()):
            split_rows.append({"split": split_name, "source_session_id": sid})
    pd.DataFrame(split_rows).to_csv(fold_dir / "source_session_ids.csv", index=False)

    train_cfg = TrainConfig(
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        patience=cfg.patience,
        seed=cfg.seed,
        feature_mode=cfg.feature_mode,
        dropout_input=cfg.dropout_input,
        dropout_hidden=cfg.dropout_hidden,
    )
    checkpoint_path = fold_dir / "cnn_lstm_radar_augmented_sgkf4.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_class": "CNNLSTM",
            "feature_names": feature_names,
            "class_names": CLASS_NAMES,
            "label_mapping": {name: label for name, label in zip(CLASS_NAMES, CLASS_LABELS)},
            "train_config": asdict(train_cfg),
            "grouped_cv_config": asdict(cfg),
            "outer_fold": fold,
            "split_protocol": protocol,
            "normalization": norm_params,
            "history": history,
            "best_epoch": best_epoch,
        },
        checkpoint_path,
    )

    metrics = {
        "fold": fold,
        "split_protocol": protocol,
        "best_epoch": int(best_epoch),
        "epochs_ran": int(len(history)),
        "runtime_sec": float(time.perf_counter() - start),
        "test_loss": float(test_loss),
        "train_windows": int(len(train_idx)),
        "val_windows": int(len(val_idx)),
        "test_windows": int(len(test_idx)),
        "train_sessions": int(len(np.unique(groups[train_idx]))),
        "val_sessions": int(len(np.unique(groups[val_idx]))),
        "test_sessions": int(len(np.unique(groups[test_idx]))),
        "leakage_count": int(len(leakage)),
        "window_class_counts": {name: int((y_test == label).sum()) for label, name in zip(CLASS_LABELS, CLASS_NAMES)},
        "session_class_counts": {name: int((session_true == label).sum()) for label, name in zip(CLASS_LABELS, CLASS_NAMES)},
        "checkpoint": str(checkpoint_path),
        "training_curve": str(fold_dir / "training_curve.png"),
        "confusion_matrix": str(fold_dir / "confusion_matrix.csv"),
        "per_class_metrics": str(fold_dir / "per_class_metrics.csv"),
        **window_metrics,
        **session_metrics,
    }
    (fold_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def aggregate_metrics(fold_metrics: list[dict[str, object]]) -> dict[str, object]:
    metric_names = [
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "fall_false_positives",
        "fall_false_negatives",
        "sitting_to_standing",
        "standing_to_sitting",
    ]
    for class_name in CLASS_NAMES:
        metric_names.extend([f"{class_name}_precision", f"{class_name}_recall", f"{class_name}_f1"])
    summary: dict[str, object] = {"folds": len(fold_metrics), "window": {}, "session": {}}
    for level in ("window", "session"):
        level_summary = {}
        for metric in metric_names:
            key = f"{level}_{metric}"
            values = np.asarray([float(m[key]) for m in fold_metrics], dtype=float)
            level_summary[metric] = {"mean": float(values.mean()), "std": float(values.std(ddof=1))}
        summary[level] = level_summary
    return summary


def write_summary_md(output_root: Path, fold_metrics: list[dict[str, object]], aggregate: dict[str, object], checksum: str, protocol: str) -> None:
    lines = [
        "# Auto-Event-Aware SGKF4 Baseline",
        "",
        f"- Split protocol: `{protocol}`",
        f"- Manifest SHA-256: `{checksum}`",
        "- Group key: `Path(source_csv).stem`",
        "- Validation: grouped split from the training pool only, seed 42",
        "- Feature mode: existing augmented 13-feature CNN-BiLSTM",
        "- Normalization: fit on each fold's final training subset only",
        "",
        "## Fold Counts",
        "",
        "| Fold | Train sessions | Val sessions | Test sessions | Train windows | Val windows | Test windows | Leakage |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for m in fold_metrics:
        lines.append(
            f"| {int(m['fold'])} | {int(m['train_sessions'])} | {int(m['val_sessions'])} | {int(m['test_sessions'])} | "
            f"{int(m['train_windows'])} | {int(m['val_windows'])} | {int(m['test_windows'])} | {int(m['leakage_count'])} |"
        )
    lines.extend(["", "## Aggregate Mean +/- Std", "", "| Metric | Window | Session |", "|---|---:|---:|"])
    show_metrics = [
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "fall_precision",
        "fall_recall",
        "fall_f1",
        "fall_false_positives",
        "fall_false_negatives",
        "sitting_to_standing",
        "standing_to_sitting",
    ]
    for metric in show_metrics:
        w = aggregate["window"][metric]
        s = aggregate["session"][metric]
        lines.append(f"| {metric} | {w['mean']:.4f} +/- {w['std']:.4f} | {s['mean']:.4f} +/- {s['std']:.4f} |")
    output_root.joinpath("cv_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_grouped_cv(dataset_path: Path, manifest_path: Path, output_root: Path, cfg: GroupedCVConfig, force_manifest: bool = False, overwrite_output: bool = False) -> dict[str, object]:
    payload, manifest = load_or_create_manifest(dataset_path, manifest_path, cfg, force=force_manifest)
    protocol = _manifest_protocol(manifest, cfg)
    readme = manifest_path.with_name(f"{manifest_path.stem}_README.md")
    if not readme.exists():
        readme = write_manifest_readme(dataset_path, manifest_path, payload, manifest)
    if output_root.exists():
        if not overwrite_output:
            raise FileExistsError(f"Output directory already exists: {output_root}. Use --overwrite-output to replace it.")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    dev = device()
    fold_metrics = []
    for fold in range(1, cfg.folds + 1):
        metrics = train_one_fold(fold, payload, manifest, cfg, output_root, dev)
        fold_metrics.append(metrics)
        print(
            f"fold {fold}/{cfg.folds}: "
            f"window_acc={metrics['window_accuracy']:.4f} "
            f"window_macro_f1={metrics['window_macro_f1']:.4f} "
            f"session_acc={metrics['session_accuracy']:.4f} "
            f"epochs={metrics['epochs_ran']}"
        )
    aggregate = aggregate_metrics(fold_metrics)
    checksum = manifest_checksum(manifest_path)
    summary = {
        "dataset": str(dataset_path),
        "manifest": str(manifest_path),
        "manifest_readme": str(readme),
        "manifest_sha256": checksum,
        "config": asdict(cfg),
        "split_protocol": protocol,
        "leakage_checks": {"max_leakage_count": int(max(m["leakage_count"] for m in fold_metrics))},
        "folds": fold_metrics,
        "aggregate": aggregate,
    }
    output_root.joinpath("cv_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame(fold_metrics).drop(columns=["window_confusion_matrix", "session_confusion_matrix"]).to_csv(output_root / "fold_metrics.csv", index=False)
    write_summary_md(output_root, fold_metrics, aggregate, checksum, protocol)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DEFAULT_DATASET))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-folds", type=int, default=6)
    parser.add_argument("--force-manifest", action="store_true")
    parser.add_argument("--overwrite-output", action="store_true")
    args = parser.parse_args()
    cfg = GroupedCVConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        seed=args.seed,
        val_folds=args.val_folds,
    )
    summary = run_grouped_cv(
        Path(args.data),
        Path(args.manifest),
        Path(args.output_dir),
        cfg,
        force_manifest=args.force_manifest,
        overwrite_output=args.overwrite_output,
    )
    print(f"manifest_sha256={summary['manifest_sha256']}")
    print(f"saved_summary={Path(args.output_dir) / 'cv_summary.md'}")


if __name__ == "__main__":
    main()
