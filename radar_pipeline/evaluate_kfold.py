from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "radar_matplotlib"))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from radar_pipeline.common import ROOT
from radar_pipeline.train_model import CLASS_LABELS, CLASS_NAMES, CNNLSTM, augment_window_features, class_weights, device, evaluate, make_loader, set_seed


@dataclass
class KFoldConfig:
    folds: int = 5
    epochs: int = 80
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 12
    seed: int = 42
    feature_mode: str = "augmented"
    val_fraction: float = 0.15


def load_windowed_dataset(input_dir: Path) -> dict[str, np.ndarray]:
    arrays: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    activities: list[str] = []
    sessions: list[str] = []
    sources: list[str] = []
    feature_names: list[str] | None = None

    for path in sorted(input_dir.glob("*_windows.npz")):
        data = np.load(path, allow_pickle=True)
        X = data["X"].astype(np.float32)
        y = data["y"].astype(np.int64)
        if X.size == 0:
            continue
        names = [str(x) for x in data["feature_names"]]
        if feature_names is None:
            feature_names = names
        elif names != feature_names:
            raise ValueError(f"Feature mismatch in {path}: {names} != {feature_names}")
        arrays.append(X)
        labels.append(y)
        activity = str(data["activity"])
        activities.extend([activity] * len(y))
        sessions.extend([str(x) for x in data["session_id"]])
        sources.extend([str(x) for x in data["source_csv"]])

    if not arrays or feature_names is None:
        raise FileNotFoundError(f"No window files found under {input_dir}")

    X_all = np.concatenate(arrays, axis=0)
    y_all = np.concatenate(labels, axis=0)
    keep = np.isin(y_all, CLASS_LABELS)
    return {
        "X": X_all[keep],
        "y": y_all[keep],
        "activity": np.asarray(activities, dtype=str)[keep],
        "session_id": np.asarray(sessions, dtype=str)[keep],
        "source_csv": np.asarray(sources, dtype=str)[keep],
        "feature_names": np.asarray(feature_names, dtype=str),
    }


def normalize_by_train(X: np.ndarray, train_idx: np.ndarray) -> tuple[np.ndarray, dict[str, list[float]]]:
    train = X[train_idx]
    means = train.reshape(-1, train.shape[-1]).mean(axis=0)
    stds = train.reshape(-1, train.shape[-1]).std(axis=0)
    stds = np.where(stds == 0, 1.0, stds)
    X_norm = ((X - means) / stds).astype(np.float32)
    return X_norm, {"mean": means.tolist(), "std": stds.tolist()}


def apply_feature_mode_to_arrays(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    feature_names: np.ndarray,
    feature_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    names = [str(x) for x in feature_names]
    if feature_mode == "raw":
        return X_train, X_val, X_test, names
    if feature_mode != "augmented":
        raise ValueError(f"Unknown feature_mode: {feature_mode}")
    payload = {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "feature_names": feature_names,
    }
    augmented, augmented_names = augment_window_features(payload)
    return augmented["X_train"], augmented["X_val"], augmented["X_test"], augmented_names


def grouped_train_val_split(
    train_val_idx: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    val_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    unique_groups = np.unique(groups[train_val_idx])
    if len(unique_groups) < 3 or val_fraction <= 0:
        return train_val_idx, train_val_idx[:0]
    n_splits = max(2, min(len(unique_groups), round(1.0 / val_fraction)))
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rel = np.arange(len(train_val_idx))
    first_train_rel, first_val_rel = next(splitter.split(rel, y[train_val_idx], groups[train_val_idx]))
    return train_val_idx[first_train_rel], train_val_idx[first_val_rel]


def activity_balanced_group_folds(
    y: np.ndarray,
    groups: np.ndarray,
    activity: np.ndarray,
    n_splits: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    session_rows = []
    for session_id in sorted(np.unique(groups)):
        idx = np.where(groups == session_id)[0]
        session_activity = str(pd.Series(activity[idx]).mode().iloc[0])
        session_label = int(pd.Series(y[idx]).mode().iloc[0])
        session_rows.append(
            {
                "session_id": session_id,
                "activity": session_activity,
                "label": session_label,
                "windows": int(len(idx)),
            }
        )
    sessions_df = pd.DataFrame(session_rows)
    fold_sessions: list[list[str]] = [[] for _ in range(n_splits)]
    fold_activity_windows: list[dict[str, int]] = [dict() for _ in range(n_splits)]
    rng = np.random.default_rng(seed)

    for activity_name, act_df in sessions_df.groupby("activity", sort=True):
        records = act_df.to_dict("records")
        rng.shuffle(records)
        records = sorted(records, key=lambda r: r["windows"], reverse=True)
        for record in records:
            fold_id = min(
                range(n_splits),
                key=lambda i: (
                    fold_activity_windows[i].get(activity_name, 0),
                    len(fold_sessions[i]),
                    i,
                ),
            )
            fold_sessions[fold_id].append(str(record["session_id"]))
            fold_activity_windows[fold_id][activity_name] = fold_activity_windows[fold_id].get(activity_name, 0) + int(record["windows"])

    folds = []
    all_idx = np.arange(len(y))
    for session_list in fold_sessions:
        test_mask = np.isin(groups, session_list)
        test_idx = all_idx[test_mask]
        train_idx = all_idx[~test_mask]
        folds.append((train_idx, test_idx))
    return folds


def train_one_fold(
    fold: int,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    activity: np.ndarray,
    train_val_idx: np.ndarray,
    test_idx: np.ndarray,
    feature_names: np.ndarray,
    cfg: KFoldConfig,
    dev: torch.device,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    set_seed(cfg.seed + fold)
    train_idx, val_idx = grouped_train_val_split(train_val_idx, y, groups, cfg.val_fraction, cfg.seed + fold)
    if len(val_idx) == 0:
        val_idx = train_idx

    X_norm, norm_params = normalize_by_train(X, train_idx)
    X_train, X_val, X_test, used_features = apply_feature_mode_to_arrays(
        X_norm[train_idx],
        X_norm[val_idx],
        X_norm[test_idx],
        feature_names,
        cfg.feature_mode,
    )
    y_train = y[train_idx]
    y_val = y[val_idx]
    y_test = y[test_idx]

    train_loader = make_loader(X_train, y_train, cfg.batch_size, shuffle=True)
    val_loader = make_loader(X_val, y_val, cfg.batch_size, shuffle=False)
    test_loader = make_loader(X_test, y_test, cfg.batch_size, shuffle=False)

    model = CNNLSTM(n_features=X_train.shape[-1], n_classes=len(CLASS_LABELS)).to(dev)
    weights = class_weights(y_train, dev)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    start = time.perf_counter()
    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    stale = 0
    history = []
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
    test_loss, test_acc, y_true, y_pred, _ = evaluate(model, test_loader, loss_fn, dev)
    runtime_sec = time.perf_counter() - start
    report = classification_report(y_true, y_pred, labels=CLASS_LABELS, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_LABELS)

    test_sessions = sorted(np.unique(groups[test_idx]).tolist())
    train_sessions = sorted(np.unique(groups[train_idx]).tolist())
    val_sessions = sorted(np.unique(groups[val_idx]).tolist())
    leakage = sorted((set(train_sessions) | set(val_sessions)) & set(test_sessions))
    class_counts = pd.Series(activity[test_idx]).value_counts().to_dict()

    fold_result: dict[str, object] = {
        "fold": fold,
        "accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "best_epoch": int(best_epoch),
        "epochs_ran": int(len(history)),
        "runtime_sec": float(runtime_sec),
        "train_windows": int(len(train_idx)),
        "val_windows": int(len(val_idx)),
        "test_windows": int(len(test_idx)),
        "train_sessions": int(len(train_sessions)),
        "val_sessions": int(len(val_sessions)),
        "test_sessions": int(len(test_sessions)),
        "test_session_ids": ";".join(test_sessions),
        "test_class_counts": json.dumps({str(k): int(v) for k, v in class_counts.items()}, sort_keys=True),
        "leaky_sessions": ";".join(leakage),
        "confusion_matrix": json.dumps(cm.tolist()),
        "feature_count": int(X_train.shape[-1]),
        "feature_mode": cfg.feature_mode,
        "normalization": "fit on each fold training split",
        "normalizer_mean": json.dumps(norm_params["mean"]),
        "normalizer_std": json.dumps(norm_params["std"]),
    }
    for class_name in CLASS_NAMES:
        metrics = report[class_name]
        fold_result[f"{class_name}_precision"] = float(metrics["precision"])
        fold_result[f"{class_name}_recall"] = float(metrics["recall"])
        fold_result[f"{class_name}_f1"] = float(metrics["f1-score"])
        fold_result[f"{class_name}_support"] = int(metrics["support"])
    fold_result["macro_precision"] = float(report["macro avg"]["precision"])
    fold_result["macro_recall"] = float(report["macro avg"]["recall"])
    fold_result["macro_f1"] = float(report["macro avg"]["f1-score"])
    fold_result["weighted_precision"] = float(report["weighted avg"]["precision"])
    fold_result["weighted_recall"] = float(report["weighted avg"]["recall"])
    fold_result["weighted_f1"] = float(report["weighted avg"]["f1-score"])

    detail_rows = []
    for sid, sid_df in pd.DataFrame({"session_id": groups[test_idx], "activity": activity[test_idx], "y_true": y_true, "y_pred": y_pred}).groupby("session_id"):
        detail_rows.append(
            {
                "fold": fold,
                "session_id": sid,
                "activity": str(sid_df["activity"].iloc[0]),
                "windows": int(len(sid_df)),
                "correct": int((sid_df["y_true"] == sid_df["y_pred"]).sum()),
                "accuracy": float((sid_df["y_true"] == sid_df["y_pred"]).mean()),
            }
        )
    return fold_result, detail_rows


def summarize_results(results: pd.DataFrame, cfg: KFoldConfig, n_sessions: int, n_windows: int, output_path: Path) -> None:
    metric_cols = [
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_precision",
        "weighted_recall",
        "weighted_f1",
    ]
    means = results[metric_cols].mean()
    stds = results[metric_cols].std(ddof=1)
    total_runtime = float(results["runtime_sec"].sum())
    avg_runtime = float(results["runtime_sec"].mean())
    loso_estimate = avg_runtime * n_sessions
    speedup = loso_estimate / total_runtime if total_runtime else 0.0

    lines = [
        "# Group K-Fold Evaluation Summary",
        "",
        "This is a robustness validation run only. It does not replace the official fixed train/val/test model.",
        "",
        "## Configuration",
        "",
        f"- Splitter: activity-balanced Group K-Fold grouped by `session_id`",
        f"- Folds: {cfg.folds}",
        f"- Feature mode: {cfg.feature_mode}",
        f"- Windows: {n_windows}",
        f"- Sessions: {n_sessions}",
        f"- Epochs max: {cfg.epochs}",
        f"- Early-stopping patience: {cfg.patience}",
        f"- Batch size: {cfg.batch_size}",
        f"- Normalization: fit separately on each fold's training windows only",
        "",
        "## Mean +/- Std Across Folds",
        "",
        "| Metric | Mean | Std |",
        "|---|---:|---:|",
    ]
    for col in metric_cols:
        lines.append(f"| {col} | {means[col]:.4f} | {stds[col]:.4f} |")

    lines.extend(
        [
            "",
            "## Runtime",
            "",
            f"- Actual 5-fold runtime: {total_runtime:.1f} sec",
            f"- Average runtime per fold: {avg_runtime:.1f} sec",
            f"- LOSO estimate at {n_sessions} sessions: {loso_estimate:.1f} sec",
            f"- Estimated speedup versus LOSO: {speedup:.2f}x",
            "",
            "The LOSO runtime is an estimate based on the average K-Fold training time multiplied by the number of sessions.",
            "",
            "## Per-Fold Results",
            "",
            "| Fold | Accuracy | Macro F1 | Weighted F1 | Test windows | Test sessions | Test class counts | Leaky sessions |",
            "|---:|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for _, row in results.iterrows():
        raw_leaky = row.get("leaky_sessions", "")
        leaky = "none" if pd.isna(raw_leaky) or not str(raw_leaky).strip() else str(raw_leaky)
        lines.append(
            f"| {int(row['fold'])} | {row['accuracy']:.4f} | {row['macro_f1']:.4f} | {row['weighted_f1']:.4f} | "
            f"{int(row['test_windows'])} | {int(row['test_sessions'])} | `{row['test_class_counts']}` | {leaky} |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_kfold(input_dir: Path, output_dir: Path, cfg: KFoldConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    set_seed(cfg.seed)
    random.seed(cfg.seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = load_windowed_dataset(input_dir)
    X = payload["X"]
    y = payload["y"]
    activity = payload["activity"]
    groups = payload["session_id"]
    feature_names = payload["feature_names"]
    n_sessions = len(np.unique(groups))
    if cfg.folds > n_sessions:
        raise ValueError(f"folds={cfg.folds} exceeds session count {n_sessions}")

    folds = activity_balanced_group_folds(y, groups, activity, cfg.folds, cfg.seed)
    dev = device()
    results = []
    detail_rows = []
    for fold, (train_val_idx, test_idx) in enumerate(folds, start=1):
        fold_result, fold_details = train_one_fold(
            fold,
            X,
            y,
            groups,
            activity,
            train_val_idx,
            test_idx,
            feature_names,
            cfg,
            dev,
        )
        results.append(fold_result)
        detail_rows.extend(fold_details)
        print(
            f"fold {fold}/{cfg.folds}: "
            f"acc={fold_result['accuracy']:.4f}, "
            f"macro_f1={fold_result['macro_f1']:.4f}, "
            f"test_sessions={fold_result['test_sessions']}, "
            f"runtime={fold_result['runtime_sec']:.1f}s"
        )

    results_df = pd.DataFrame(results)
    details_df = pd.DataFrame(detail_rows)
    results_path = output_dir / "kfold_results.csv"
    details_path = output_dir / "kfold_session_results.csv"
    summary_path = output_dir / "kfold_summary.md"
    results_df.to_csv(results_path, index=False)
    details_df.to_csv(details_path, index=False)
    summarize_results(results_df, cfg, n_sessions, len(X), summary_path)
    print(f"Saved fold results: {results_path}")
    print(f"Saved session details: {details_path}")
    print(f"Saved summary: {summary_path}")
    return results_df, details_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=str(ROOT / "data/windowed"))
    parser.add_argument("--output-dir", default=str(ROOT / "outputs/models"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--feature-mode", choices=("raw", "augmented"), default="augmented")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    args = parser.parse_args()
    cfg = KFoldConfig(
        folds=args.folds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        seed=args.seed,
        feature_mode=args.feature_mode,
        val_fraction=args.val_fraction,
    )
    run_kfold(Path(args.input_dir), Path(args.output_dir), cfg)


if __name__ == "__main__":
    main()
