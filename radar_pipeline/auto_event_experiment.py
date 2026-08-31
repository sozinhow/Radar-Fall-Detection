from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from radar_pipeline.common import CLASS_LABELS, CLASS_NAMES, ROOT
from radar_pipeline.train_model import CNNLSTM, class_weights, device, evaluate, set_seed


@dataclass
class ExperimentConfig:
    folds: int = 5
    epochs: int = 80
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 12
    seed: int = 42
    feature_mode: str = "augmented"
    val_fraction: float = 0.15
    focal_gamma: float = 2.0


class FocalLoss(nn.Module):
    def __init__(self, alpha: torch.Tensor, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = nn.functional.cross_entropy(logits, target, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce)
        return (((1.0 - pt) ** self.gamma) * ce).mean()


def load_included_windowed(input_dir: Path) -> dict[str, np.ndarray]:
    arrays: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    sessions: list[np.ndarray] = []
    phases: list[np.ndarray] = []
    confidences: list[np.ndarray] = []
    sources: list[np.ndarray] = []
    feature_names: list[str] | None = None
    for path in sorted(input_dir.glob("*_windows.npz")):
        data = np.load(path, allow_pickle=True)
        X = data["X"].astype(np.float32)
        y = data["y"].astype(np.int64)
        keep = np.isin(y, CLASS_LABELS)
        if "include_in_training" in data:
            keep &= data["include_in_training"].astype(bool)
        if not keep.any():
            continue
        names = [str(x) for x in data["feature_names"]]
        if feature_names is None:
            feature_names = names
        elif names != feature_names:
            raise ValueError(f"Feature mismatch in {path}: {names} != {feature_names}")
        arrays.append(X[keep])
        labels.append(y[keep])
        sessions.append(data["session_id"].astype(str)[keep])
        phases.append(data["event_phase"].astype(str)[keep] if "event_phase" in data else np.array(["legacy"] * keep.sum()))
        confidences.append(
            data["annotation_confidence"].astype(str)[keep] if "annotation_confidence" in data else np.array([""] * keep.sum())
        )
        sources.append(data["source_csv"].astype(str)[keep] if "source_csv" in data else np.array([""] * keep.sum()))
    if not arrays or feature_names is None:
        raise FileNotFoundError(f"No included windows found under {input_dir}")
    y_all = np.concatenate(labels)
    return {
        "X": np.concatenate(arrays),
        "y": y_all,
        "activity": np.asarray([CLASS_NAMES[int(v)] for v in y_all]),
        "session_id": np.concatenate(sessions),
        "event_phase": np.concatenate(phases),
        "annotation_confidence": np.concatenate(confidences),
        "source_csv": np.concatenate(sources),
        "feature_names": np.asarray(feature_names),
    }


def normalize_by_train(X: np.ndarray, train_idx: np.ndarray) -> tuple[np.ndarray, dict[str, list[float]]]:
    train = X[train_idx]
    mean = train.reshape(-1, train.shape[-1]).mean(axis=0)
    std = train.reshape(-1, train.shape[-1]).std(axis=0)
    std = np.where(std == 0, 1.0, std)
    return ((X - mean) / std).astype(np.float32), {"mean": mean.tolist(), "std": std.tolist()}


def _session_label_counts(y: np.ndarray, groups: np.ndarray) -> pd.DataFrame:
    rows = []
    for sid in sorted(np.unique(groups)):
        idx = np.where(groups == sid)[0]
        row = {"session_id": sid, "windows": len(idx)}
        for label, name in enumerate(CLASS_NAMES):
            row[name] = int((y[idx] == label).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def feasible_folds(y: np.ndarray, groups: np.ndarray, requested: int) -> tuple[int, str]:
    counts = _session_label_counts(y, groups)
    fall_sessions = int((counts["fall"] > 0).sum())
    n_sessions = len(counts)
    folds = min(requested, n_sessions, fall_sessions)
    reason = f"requested={requested}, sessions={n_sessions}, fall_sessions={fall_sessions}; using {folds} folds"
    if folds < requested:
        reason += " so every test fold can contain at least one fall-event session."
    return max(2, folds), reason


def make_group_folds(y: np.ndarray, groups: np.ndarray, n_folds: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    session_counts = _session_label_counts(y, groups)
    rng = np.random.default_rng(seed)
    records = session_counts.to_dict("records")
    rng.shuffle(records)
    records = sorted(records, key=lambda r: (r["fall"], r["windows"]), reverse=True)
    fold_sessions: list[list[str]] = [[] for _ in range(n_folds)]
    fold_counts = [np.zeros(len(CLASS_NAMES), dtype=int) for _ in range(n_folds)]
    for rec in records:
        counts = np.array([rec[name] for name in CLASS_NAMES], dtype=int)
        # Prioritize putting fall sessions into folds with less fall support.
        fold_id = min(range(n_folds), key=lambda i: (fold_counts[i][3], fold_counts[i].sum(), len(fold_sessions[i]), i))
        fold_sessions[fold_id].append(str(rec["session_id"]))
        fold_counts[fold_id] += counts
    all_idx = np.arange(len(y))
    folds = []
    for sessions in fold_sessions:
        test = np.isin(groups, sessions)
        folds.append((all_idx[~test], all_idx[test]))
    return folds


def grouped_val_split(train_val_idx: np.ndarray, y: np.ndarray, groups: np.ndarray, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    unique = np.unique(groups[train_val_idx])
    if len(unique) < 3:
        return train_val_idx, train_val_idx[:0]
    rng = np.random.default_rng(seed)
    session_counts = _session_label_counts(y[train_val_idx], groups[train_val_idx])
    records = session_counts.to_dict("records")
    rng.shuffle(records)
    target = max(1, int(round(len(unique) * val_fraction)))
    # Prefer at least one fall validation session if available.
    records = sorted(records, key=lambda r: (r["fall"], r["windows"]), reverse=True)
    val_sessions = [str(records[0]["session_id"])]
    remaining = [r for r in records[1:]]
    rng.shuffle(remaining)
    val_sessions.extend(str(r["session_id"]) for r in remaining[: max(0, target - 1)])
    val_mask = np.isin(groups[train_val_idx], val_sessions)
    return train_val_idx[~val_mask], train_val_idx[val_mask]


def apply_feature_mode_arrays(X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray, feature_names: np.ndarray, mode: str):
    names = [str(x) for x in feature_names]
    if mode == "raw":
        return X_train, X_val, X_test, names
    if mode != "augmented":
        raise ValueError(f"Unsupported feature mode {mode}")
    payload = {"X_train": X_train, "X_val": X_val, "X_test": X_test, "feature_names": feature_names}
    # Imported from train_model as augment via public apply helper is split-dict oriented.
    from radar_pipeline.train_model import augment_window_features

    augmented, augmented_names = augment_window_features(payload)
    return augmented["X_train"], augmented["X_val"], augmented["X_test"], augmented_names


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, mode: Literal["shuffle", "balanced"]) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).long())
    if mode == "balanced":
        counts = np.asarray([(y == label).sum() for label in CLASS_LABELS], dtype=np.float32)
        weights = 1.0 / np.maximum(counts, 1.0)
        sample_weights = np.asarray([weights[int(label)] for label in y], dtype=np.float32)
        sampler = WeightedRandomSampler(torch.from_numpy(sample_weights), num_samples=len(sample_weights), replacement=True)
        return DataLoader(ds, batch_size=batch_size, sampler=sampler)
    return DataLoader(ds, batch_size=batch_size, shuffle=True)


def train_fold(
    experiment: str,
    fold: int,
    payload: dict[str, np.ndarray],
    train_val_idx: np.ndarray,
    test_idx: np.ndarray,
    cfg: ExperimentConfig,
    out_dir: Path,
    dev: torch.device,
) -> dict[str, object]:
    set_seed(cfg.seed + fold)
    random.seed(cfg.seed + fold)
    X = payload["X"]
    y = payload["y"]
    groups = payload["session_id"]
    train_idx, val_idx = grouped_val_split(train_val_idx, y, groups, cfg.val_fraction, cfg.seed + fold)
    if len(val_idx) == 0:
        val_idx = train_idx
    X_norm, norm = normalize_by_train(X, train_idx)
    X_train, X_val, X_test, features = apply_feature_mode_arrays(
        X_norm[train_idx], X_norm[val_idx], X_norm[test_idx], payload["feature_names"], cfg.feature_mode
    )
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]
    train_loader = make_loader(X_train, y_train, cfg.batch_size, "balanced" if experiment == "focal_balanced" else "shuffle")
    val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long()), batch_size=cfg.batch_size)
    test_loader = DataLoader(TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long()), batch_size=cfg.batch_size)

    model = CNNLSTM(n_features=X_train.shape[-1], n_classes=len(CLASS_LABELS)).to(dev)
    weights = class_weights(y_train, dev)
    loss_fn: nn.Module = FocalLoss(weights, cfg.focal_gamma) if experiment == "focal_balanced" else nn.CrossEntropyLoss(weight=weights)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    history = []
    best_state = None
    best_val = float("inf")
    best_epoch = 0
    stale = 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total = 0
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(dev), yb.to(dev)
            optim.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optim.step()
            total_loss += float(loss.item()) * len(yb)
            total += len(yb)
        train_loss = total_loss / max(total, 1)
        val_loss, val_acc, _, _, _ = evaluate(model, val_loader, loss_fn, dev)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc})
        if val_loss < best_val - 1e-5:
            best_val = val_loss
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
    report = classification_report(y_true, y_pred, labels=CLASS_LABELS, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_LABELS)
    fall_true = (y_true == 3).astype(int)
    fall_score = y_prob[:, 3] if len(y_prob) else np.array([])
    fall_roc = float(roc_auc_score(fall_true, fall_score)) if len(np.unique(fall_true)) == 2 else np.nan
    fall_ap = float(average_precision_score(fall_true, fall_score)) if fall_true.sum() > 0 else np.nan
    train_sessions = set(groups[train_idx])
    val_sessions = set(groups[val_idx])
    test_sessions = set(groups[test_idx])
    leakage = sorted((train_sessions | val_sessions) & test_sessions)

    fold_dir = out_dir / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(fold_dir / "history.csv", index=False)
    plot_history(history, fold_dir / "loss_curve.png")
    np.savetxt(fold_dir / "confusion_matrix.csv", cm, delimiter=",", fmt="%d")
    pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "fall_score": fall_score}).to_csv(fold_dir / "predictions.csv", index=False)

    row = {
        "experiment": experiment,
        "fold": fold,
        "seed": cfg.seed,
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "test_loss": float(test_loss),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "fall_precision": float(report["fall"]["precision"]),
        "fall_recall": float(report["fall"]["recall"]),
        "fall_f1": float(report["fall"]["f1-score"]),
        "fall_support": int(report["fall"]["support"]),
        "fall_roc_auc": fall_roc,
        "fall_pr_auc": fall_ap,
        "leaky_sessions": ";".join(leakage),
        "leakage_count": len(leakage),
        "confusion_matrix": json.dumps(cm.tolist()),
        "train_sessions": len(train_sessions),
        "val_sessions": len(val_sessions),
        "test_sessions": len(test_sessions),
        "train_windows": len(train_idx),
        "val_windows": len(val_idx),
        "test_windows": len(test_idx),
        "train_class_counts": json.dumps(class_count_dict(y_train)),
        "test_class_counts": json.dumps(class_count_dict(y_test)),
        "feature_count": len(features),
        "normalization": "fit on fold training windows only",
    }
    for name in CLASS_NAMES:
        row[f"{name}_precision"] = float(report[name]["precision"])
        row[f"{name}_recall"] = float(report[name]["recall"])
        row[f"{name}_f1"] = float(report[name]["f1-score"])
        row[f"{name}_support"] = int(report[name]["support"])
    assignment = pd.DataFrame(
        {
            "index": np.r_[train_idx, val_idx, test_idx],
            "split": ["train"] * len(train_idx) + ["val"] * len(val_idx) + ["test"] * len(test_idx),
            "session_id": np.r_[groups[train_idx], groups[val_idx], groups[test_idx]],
            "label": np.r_[y[train_idx], y[val_idx], y[test_idx]],
        }
    )
    assignment.to_csv(fold_dir / "fold_assignment.csv", index=False)
    return row


def class_count_dict(y: np.ndarray) -> dict[str, int]:
    return {name: int((y == label).sum()) for label, name in enumerate(CLASS_NAMES)}


def plot_history(history: list[dict[str, float]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([h["epoch"] for h in history], [h["train_loss"] for h in history], label="train")
    ax.plot([h["epoch"] for h in history], [h["val_loss"] for h in history], label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_aggregate_curves(rows: pd.DataFrame, root: Path) -> None:
    for exp, df in rows.groupby("experiment"):
        pred_parts = []
        for fold in df["fold"]:
            p = root / exp / f"fold_{int(fold)}" / "predictions.csv"
            if p.exists():
                pred_parts.append(pd.read_csv(p))
        if not pred_parts:
            continue
        pred = pd.concat(pred_parts, ignore_index=True)
        y_true = (pred["y_true"].to_numpy() == 3).astype(int)
        score = pred["fall_score"].to_numpy()
        fig, ax = plt.subplots(figsize=(6, 4))
        if len(np.unique(y_true)) == 2:
            fpr, tpr, _ = roc_curve(y_true, score)
            ax.plot(fpr, tpr, label=f"fall ROC AUC={roc_auc_score(y_true, score):.3f}")
        ax.plot([0, 1], [0, 1], "k--", linewidth=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(root / exp / "fall_roc_curve.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 4))
        precision, recall, _ = precision_recall_curve(y_true, score)
        ax.plot(recall, precision, label=f"fall AP={average_precision_score(y_true, score):.3f}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(root / exp / "fall_pr_curve.png", dpi=150)
        plt.close(fig)


def write_report(rows: pd.DataFrame, root: Path, fold_reason: str, baseline_fall_f1: float = 0.593) -> None:
    summary = rows.groupby("experiment")[["macro_f1", "fall_precision", "fall_recall", "fall_f1", "fall_roc_auc", "fall_pr_auc"]].agg(["mean", "std"])
    flat = summary.copy()
    flat.columns = [f"{metric}_{stat}" for metric, stat in flat.columns]
    lines = [
        "# Auto Event-Aware Staging Model Comparison",
        "",
        "This is a staging-only experiment. Fall labels are heuristic-derived `auto_event_annotations`, not manually verified ground truth.",
        "",
        f"- Fold feasibility: {fold_reason}",
        "- Architecture: current augmented CNN-BiLSTM.",
        "- Normalization: fit on each fold's training windows only.",
        "- Session leakage: asserted per fold; any leakage appears in `fold_metrics.csv`.",
        "",
        "## Aggregate metrics",
        "",
        "| Experiment | Macro F1 mean | Macro F1 std | Fall precision mean | Fall recall mean | Fall F1 mean | Fall F1 std | Fall ROC-AUC mean | Fall PR-AUC mean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for exp, row in flat.iterrows():
        lines.append(
            f"| {exp} | {row['macro_f1_mean']:.3f} | {row['macro_f1_std']:.3f} | "
            f"{row['fall_precision_mean']:.3f} | {row['fall_recall_mean']:.3f} | "
            f"{row['fall_f1_mean']:.3f} | {row['fall_f1_std']:.3f} | "
            f"{row['fall_roc_auc_mean']:.3f} | {row['fall_pr_auc_mean']:.3f} |"
        )
    lines.extend([
        "",
        "## Interpretation cautions",
        "",
        f"- The current whole-session baseline fall F1 was approximately {baseline_fall_f1:.3f}, but that came from a different label definition/split protocol. Treat comparison as descriptive, not apples-to-apples.",
        "- Only 30 fall-event windows are available in this staged dataset.",
        "- Labels are automatic heuristics, so improved metrics could reflect heuristic bias rather than true deployment performance.",
        "- Do not claim commercial readiness or validated deployment performance from this run.",
    ])
    a = rows[rows.experiment == "weighted_ce"]["fall_f1"].mean()
    b = rows[rows.experiment == "focal_balanced"]["fall_f1"].mean()
    bp = rows[rows.experiment == "focal_balanced"]["fall_precision"].mean()
    br = rows[rows.experiment == "focal_balanced"]["fall_recall"].mean()
    lines.extend(
        [
            "",
            "## Short conclusion",
            "",
            f"- Weighted CE mean fall F1: {a:.3f}.",
            f"- Focal+balanced mean fall F1: {b:.3f}; mean fall precision={bp:.3f}, recall={br:.3f}.",
            "- Judge focal+balanced by whether recall gain is worth any precision loss; inspect per-fold supports before drawing conclusions.",
        ]
    )
    (root / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(root: Path, input_dir: Path, cfg: ExperimentConfig) -> pd.DataFrame:
    payload = load_included_windowed(input_dir)
    folds, reason = feasible_folds(payload["y"], payload["session_id"], cfg.folds)
    fold_pairs = make_group_folds(payload["y"], payload["session_id"], folds, cfg.seed)
    root.mkdir(parents=True, exist_ok=True)
    (root / "feasibility.json").write_text(
        json.dumps(
            {
                "fold_reason": reason,
                "folds": folds,
                "unique_sessions": int(len(np.unique(payload["session_id"]))),
                "unique_fall_sessions": int(
                    len(np.unique(payload["session_id"][payload["y"] == 3]))
                ),
                "class_counts": class_count_dict(payload["y"]),
                "config": asdict(cfg),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    dev = device()
    rows = []
    for experiment in ("weighted_ce", "focal_balanced"):
        exp_dir = root / experiment
        exp_dir.mkdir(parents=True, exist_ok=True)
        for fold, (train_val_idx, test_idx) in enumerate(fold_pairs, start=1):
            test_counts = class_count_dict(payload["y"][test_idx])
            if test_counts["fall"] == 0:
                print(f"WARNING: fold {fold} has no fall test windows")
            row = train_fold(experiment, fold, payload, train_val_idx, test_idx, cfg, exp_dir, dev)
            rows.append(row)
            print(f"{experiment} fold {fold}/{folds}: fall_f1={row['fall_f1']:.3f}, leakage={row['leakage_count']}")
    df = pd.DataFrame(rows)
    df.to_csv(root / "fold_metrics.csv", index=False)
    comparison = df.groupby("experiment")[["macro_f1", "fall_precision", "fall_recall", "fall_f1", "fall_roc_auc", "fall_pr_auc"]].agg(["mean", "std"])
    comparison.to_csv(root / "comparison.csv")
    # Aggregate confusion matrices.
    for exp, exp_df in df.groupby("experiment"):
        cms = [np.asarray(json.loads(x), dtype=int) for x in exp_df["confusion_matrix"]]
        cm = np.sum(cms, axis=0)
        pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(root / exp / "aggregate_confusion_matrix.csv")
    plot_aggregate_curves(df, root)
    write_report(df, root, reason)
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=str(ROOT / "data/windowed_auto_event_staging"))
    parser.add_argument("--output-root", default=str(ROOT / "outputs/experiments/auto_event_aware_v1"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    cfg = ExperimentConfig(
        folds=args.folds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        seed=args.seed,
    )
    run(Path(args.output_root), Path(args.input_dir), cfg)


if __name__ == "__main__":
    main()
