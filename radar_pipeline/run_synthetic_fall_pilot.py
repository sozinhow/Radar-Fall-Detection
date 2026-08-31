from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from torch import nn

from radar_pipeline.common import CLASS_LABELS, CLASS_NAMES, ROOT
from radar_pipeline.evaluate_grouped_cv import (
    DEFAULT_MANIFEST,
    GroupedCVConfig,
    aggregate_metrics,
    apply_augmented_features,
    grouped_validation_split,
    load_or_create_manifest,
    manifest_checksum,
    metrics_from_predictions,
    normalize_by_train,
    save_confusion_matrix_plot,
    session_level_predictions,
)
from radar_pipeline.synthetic_fall_augmentation import (
    FALL_LABEL,
    SyntheticFallConfig,
    assert_synthetic_fold_safety,
    calibrate_training_variability,
    generate_synthetic_fall_windows,
    session_seed_audit,
)
from radar_pipeline.train_model import CNNLSTM, TrainConfig, class_weights, device, evaluate, make_loader, plot_history, set_seed


DEFAULT_DATASET = ROOT / "data/final_dataset_auto_event_staging_20260717/radar_dataset.npz"
DEFAULT_DATE_MATCHED_MANIFEST = ROOT / "data/metadata/auto_event_aware_20260717_source_session_folds.csv"
DEFAULT_OUTPUT = ROOT / "outputs/experiments/auto_event_aware_sgkf4_20260717_synthetic_fall_pilot"
MATCHED_BASELINE_OUTPUT = ROOT / "outputs/experiments/auto_event_aware_sgkf4_0260717_candidate"
DEFAULT_RATIOS = (0.0, 0.5, 1.0, 2.0)


def _ratio_name(ratio: float) -> str:
    return str(ratio).replace(".", "p") + "x"


def _load_extra_metadata(dataset_path: Path, payload: dict[str, np.ndarray]) -> None:
    data = np.load(dataset_path, allow_pickle=True)
    n = len(payload["y"])
    for key in (
        "event_phase",
        "quality_flags",
        "annotation_confidence",
        "include_in_training",
        "label_source",
        "overlap_seconds",
        "overlap_fraction",
    ):
        split_key = f"{key}_train"
        if split_key in data.files:
            payload[key] = np.concatenate([data[f"{key}_{split}"] for split in ("train", "val", "test")], axis=0)
        elif key not in payload:
            payload[key] = np.full(n, "", dtype=object)


def _train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: GroupedCVConfig,
    dev: torch.device,
) -> tuple[CNNLSTM, list[dict[str, float]], int, float]:
    train_loader = make_loader(X_train, y_train, cfg.batch_size, shuffle=True)
    val_loader = make_loader(X_val, y_val, cfg.batch_size, shuffle=False)
    model = CNNLSTM(
        n_features=X_train.shape[-1],
        n_classes=len(CLASS_LABELS),
        dropout_input=cfg.dropout_input,
        dropout_hidden=cfg.dropout_hidden,
    ).to(dev)
    weights = class_weights(y_train, dev)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
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
        train_acc = float(accuracy_score(train_true, train_pred)) if train_true else 0.0
        val_loss, val_acc, _, _, _ = evaluate(model, val_loader, loss_fn, dev)
        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc})
        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= cfg.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, best_epoch, best_val_loss


def _evaluate_window_and_session(
    model: CNNLSTM,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    source_activity: np.ndarray,
    cfg: GroupedCVConfig,
    dev: torch.device,
    prefix: str,
) -> dict[str, object]:
    loader = make_loader(X, y, cfg.batch_size, shuffle=False)
    loss_fn = nn.CrossEntropyLoss()
    loss, _, y_true, y_pred, y_prob = evaluate(model, loader, loss_fn, dev)
    window_metrics = metrics_from_predictions(y_true, y_pred, prefix=f"{prefix}window_")
    session_true, session_pred = session_level_predictions(y_true, y_prob, groups, source_activity)
    session_metrics = metrics_from_predictions(session_true, session_pred, prefix=f"{prefix}session_")
    return {f"{prefix}loss": float(loss), **window_metrics, **session_metrics}


def _selection_key(row: dict[str, object], baseline_recall: float) -> tuple[int, float, float, float, float]:
    recall = float(row["val_window_fall_recall"])
    eligible = int(recall >= baseline_recall - 0.05)
    return (
        eligible,
        float(row["val_window_fall_f1"]) if eligible else -1.0,
        float(row["val_window_fall_precision"]) if eligible else -1.0,
        -abs(float(row["ratio"])),
        -float(row["val_window_fall_false_positives"]),
    )


def train_fold_with_ratio_grid(
    fold: int,
    payload: dict[str, np.ndarray],
    manifest: pd.DataFrame,
    cfg: GroupedCVConfig,
    ratios: tuple[float, ...],
    output_root: Path,
    dev: torch.device,
) -> dict[str, object]:
    set_seed(cfg.seed + fold)
    random.seed(cfg.seed + fold)
    np.random.seed(cfg.seed + fold)
    groups = payload["source_session_id"].astype(str)
    y = payload["y"].astype(np.int64)
    fold_sessions = set(manifest.loc[manifest["outer_fold"].astype(int) == fold, "source_session_id"].astype(str))
    all_idx = np.arange(len(y))
    test_idx = all_idx[np.asarray([sid in fold_sessions for sid in groups])]
    train_pool_idx = all_idx[np.asarray([sid not in fold_sessions for sid in groups])]
    train_idx, val_idx = grouped_validation_split(train_pool_idx, y, groups, cfg)

    train_sessions = set(groups[train_idx])
    val_sessions = set(groups[val_idx])
    test_sessions = set(groups[test_idx])
    assert_synthetic_fold_safety(train_sessions, val_sessions, test_sessions, pd.DataFrame())

    fold_dir = output_root / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=False)
    seed_audit = session_seed_audit(payload, train_idx)
    seed_audit.to_csv(fold_dir / "real_fall_seed_audit.csv", index=False)
    eligible_seed_idx = seed_audit.loc[seed_audit["included"], "window_index"].to_numpy(dtype=int)

    X_norm, norm_params = normalize_by_train(payload["X"].astype(np.float32), train_idx)
    calibration = calibrate_training_variability(X_norm[train_idx])
    selected_by_ratio: dict[float, dict[str, object]] = {}
    ablation_rows: list[dict[str, object]] = []
    ratio_artifacts: dict[float, dict[str, object]] = {}
    start = time.perf_counter()

    for ratio in ratios:
        ratio_dir = fold_dir / f"ratio_{_ratio_name(ratio)}"
        ratio_dir.mkdir(parents=True, exist_ok=False)
        synth_X, synth_prov = generate_synthetic_fall_windows(
            X_norm,
            eligible_seed_idx,
            seed_audit,
            ratio,
            calibration,
            SyntheticFallConfig(seed=cfg.seed),
            fold=fold,
        )
        assert_synthetic_fold_safety(train_sessions, val_sessions, test_sessions, synth_prov)
        synth_prov.to_csv(ratio_dir / "synthetic_sample_provenance.csv", index=False)

        X_train_raw = X_norm[train_idx]
        y_train = y[train_idx]
        if len(synth_X):
            X_train_raw = np.concatenate([X_train_raw, synth_X], axis=0)
            y_train = np.concatenate([y_train, np.full(len(synth_X), FALL_LABEL, dtype=np.int64)], axis=0)
        X_train, X_val, X_test, feature_names = apply_augmented_features(
            X_train_raw,
            X_norm[val_idx],
            X_norm[test_idx],
            payload["feature_names"],
            cfg.feature_mode,
        )
        model, history, best_epoch, best_val_loss = _train_model(X_train, y_train, X_val, y[val_idx], cfg, dev)
        val_metrics = _evaluate_window_and_session(
            model,
            X_val,
            y[val_idx],
            groups[val_idx],
            payload["source_activity"][val_idx],
            cfg,
            dev,
            prefix="val_",
        )
        plot_history(history, ratio_dir / "training_curve.png")
        checkpoint_path = ratio_dir / "cnn_lstm_radar_augmented_sgkf4_synthetic_pilot.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_class": "CNNLSTM",
                "feature_names": feature_names,
                "class_names": CLASS_NAMES,
                "label_mapping": {name: label for name, label in zip(CLASS_NAMES, CLASS_LABELS)},
                "train_config": asdict(
                    TrainConfig(
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
                ),
                "grouped_cv_config": asdict(cfg),
                "outer_fold": fold,
                "synthetic_ratio": ratio,
                "synthetic_samples": int(len(synth_X)),
                "normalization": norm_params,
                "history": history,
                "best_epoch": int(best_epoch),
            },
            checkpoint_path,
        )
        row = {
            "fold": fold,
            "ratio": float(ratio),
            "eligible_seed_windows": int(len(eligible_seed_idx)),
            "synthetic_windows": int(len(synth_X)),
            "real_train_windows": int(len(train_idx)),
            "augmented_train_windows": int(len(y_train)),
            "best_epoch": int(best_epoch),
            "best_val_loss": float(best_val_loss),
            "checkpoint": str(checkpoint_path),
            **val_metrics,
        }
        ablation_rows.append(row)
        selected_by_ratio[ratio] = row
        ratio_artifacts[ratio] = {
            "model": model,
            "X_test": X_test,
            "feature_names": feature_names,
            "checkpoint": checkpoint_path,
        }
        (ratio_dir / "validation_metrics.json").write_text(json.dumps(row, indent=2), encoding="utf-8")

    baseline_recall = float(selected_by_ratio[0.0]["val_window_fall_recall"])
    selected_ratio = max(ratios, key=lambda ratio: _selection_key(selected_by_ratio[ratio], baseline_recall))
    selected = selected_by_ratio[selected_ratio]
    selected_model = ratio_artifacts[selected_ratio]["model"]
    X_selected_test = ratio_artifacts[selected_ratio]["X_test"]
    test_metrics = _evaluate_window_and_session(
        selected_model,
        X_selected_test,
        y[test_idx],
        groups[test_idx],
        payload["source_activity"][test_idx],
        cfg,
        dev,
        prefix="",
    )

    cm = np.asarray(test_metrics["window_confusion_matrix"], dtype=int)
    pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(fold_dir / "confusion_matrix.csv")
    save_confusion_matrix_plot(cm, fold_dir / "confusion_matrix.png", f"Fold {fold} Synthetic Pilot Window Confusion Matrix")
    session_cm = np.asarray(test_metrics["session_confusion_matrix"], dtype=int)
    pd.DataFrame(session_cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(fold_dir / "session_confusion_matrix.csv")
    save_confusion_matrix_plot(session_cm, fold_dir / "session_confusion_matrix.png", f"Fold {fold} Synthetic Pilot Session Confusion Matrix")

    split_rows = []
    for split_name, idx in (("train_real", train_idx), ("validation_real", val_idx), ("outer_test_real", test_idx)):
        for sid in sorted(np.unique(groups[idx]).tolist()):
            split_rows.append({"split": split_name, "source_session_id": sid, "is_synthetic": False})
    pd.DataFrame(split_rows).to_csv(fold_dir / "source_session_ids.csv", index=False)
    pd.DataFrame(ablation_rows).drop(columns=["val_window_confusion_matrix", "val_session_confusion_matrix"]).to_csv(
        fold_dir / "validation_ablation.csv", index=False
    )

    selected_prov_path = fold_dir / f"ratio_{_ratio_name(float(selected_ratio))}" / "synthetic_sample_provenance.csv"
    if int(selected["synthetic_windows"]):
        selected_prov = pd.read_csv(selected_prov_path)
    else:
        selected_prov = pd.DataFrame()
    selected_prov.to_csv(fold_dir / "selected_synthetic_sample_provenance.csv", index=False)
    assert_synthetic_fold_safety(train_sessions, val_sessions, test_sessions, selected_prov)

    metrics = {
        "fold": fold,
        "selected_ratio": float(selected_ratio),
        "selection_baseline_val_fall_recall": baseline_recall,
        "selected_validation_fall_precision": float(selected["val_window_fall_precision"]),
        "selected_validation_fall_recall": float(selected["val_window_fall_recall"]),
        "selected_validation_fall_f1": float(selected["val_window_fall_f1"]),
        "selected_validation_fall_false_positives": int(selected["val_window_fall_false_positives"]),
        "selected_validation_fall_false_negatives": int(selected["val_window_fall_false_negatives"]),
        "eligible_seed_windows": int(selected["eligible_seed_windows"]),
        "synthetic_train_windows": int(selected["synthetic_windows"]),
        "train_windows": int(len(train_idx)),
        "val_windows": int(len(val_idx)),
        "test_windows": int(len(test_idx)),
        "train_sessions": int(len(train_sessions)),
        "val_sessions": int(len(val_sessions)),
        "test_sessions": int(len(test_sessions)),
        "leakage_count": 0,
        "real_only_validation": True,
        "real_only_outer_test": True,
        "runtime_sec": float(time.perf_counter() - start),
        "checkpoint": str(ratio_artifacts[selected_ratio]["checkpoint"]),
        **test_metrics,
    }
    (fold_dir / "config_provenance.json").write_text(
        json.dumps(
            {
                "fold": fold,
                "config": asdict(cfg),
                "ratios": list(ratios),
                "selected_ratio": float(selected_ratio),
                "normalization_fit": "real final training subset only",
                "augmentation_fit": "real final training subset only",
                "seed_gate": {
                    "minimum_retained_fall_event_windows_per_session": 2,
                    "disallowed_flags": [
                        "event_near_recording_boundary",
                        "high_cleaning_drop",
                        "weak_motion_peak",
                        "geometry_edge_warning",
                    ],
                },
                "real_only_validation": True,
                "real_only_outer_test": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (fold_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def _write_report(
    output_root: Path,
    fold_metrics: list[dict[str, object]],
    aggregate: dict[str, object],
    ablation: pd.DataFrame,
    baseline_metrics_path: Path | None = None,
) -> None:
    baseline_fold_metrics = pd.read_csv(baseline_metrics_path) if baseline_metrics_path and baseline_metrics_path.exists() else pd.DataFrame()
    selected = pd.DataFrame(fold_metrics).set_index("fold")
    changed = (
        selected["window_fall_f1"].mean() - baseline_fold_metrics["window_fall_f1"].mean()
        if not baseline_fold_metrics.empty and "window_fall_f1" in baseline_fold_metrics
        else np.nan
    )
    if not np.isfinite(changed) or abs(changed) < 0.01:
        conclusion = "did not materially change"
    elif changed > 0:
        conclusion = "improved"
    else:
        conclusion = "worsened"
    lines = [
        "# Synthetic Fall Augmentation Pilot",
        "",
        "This is an internal staging-only SGKF4 benchmark, not deployment validation.",
        "Synthetic fall windows were generated only for final training subsets after grouped train/validation/test splits were fixed.",
        "Validation and outer-test windows remained exclusively real held-out source-session windows.",
        "",
        f"Conclusion: synthetic augmentation **{conclusion}** real held-out SGKF4 fall performance in this pilot.",
        f"Matched baseline comparison source: `{baseline_metrics_path}`" if baseline_metrics_path else "Matched baseline comparison source: unavailable",
        "",
        "## Selected Fold Policies",
        "",
        "| Fold | Selected ratio | Eligible seeds | Synthetic train windows | Test fall precision | Test fall recall | Test fall F1 | Test fall FP | Test fall FN |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in fold_metrics:
        lines.append(
            f"| {int(row['fold'])} | {float(row['selected_ratio']):.1f} | {int(row['eligible_seed_windows'])} | "
            f"{int(row['synthetic_train_windows'])} | {float(row['window_fall_precision']):.4f} | "
            f"{float(row['window_fall_recall']):.4f} | {float(row['window_fall_f1']):.4f} | "
            f"{int(row['window_fall_false_positives'])} | {int(row['window_fall_false_negatives'])} |"
        )
    if not baseline_fold_metrics.empty:
        lines.extend(
            [
                "",
                "## Matched 20260717 Baseline Comparison",
                "",
                "| Metric | Baseline mean | Pilot selected mean | Delta |",
                "|---|---:|---:|---:|",
            ]
        )
        for metric in ("window_fall_precision", "window_fall_recall", "window_fall_f1", "window_fall_false_positives", "window_fall_false_negatives"):
            b = float(baseline_fold_metrics[metric].mean())
            p = float(selected[metric].mean())
            lines.append(f"| {metric} | {b:.4f} | {p:.4f} | {p - b:.4f} |")
    lines.extend(["", "## Aggregate Mean +/- Std", "", "| Metric | Window | Session |", "|---|---:|---:|"])
    for metric in ("fall_precision", "fall_recall", "fall_f1", "fall_false_positives", "fall_false_negatives", "macro_f1", "weighted_f1"):
        w = aggregate["window"][metric]
        s = aggregate["session"][metric]
        lines.append(f"| {metric} | {w['mean']:.4f} +/- {w['std']:.4f} | {s['mean']:.4f} +/- {s['std']:.4f} |")
    lines.extend(
        [
            "",
            "## Validation Ablation",
            "",
            "| Ratio | Fall precision | Fall recall | Fall F1 | Fall FP | Fall FN |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for ratio, group in ablation.groupby("ratio"):
        lines.append(
            f"| {float(ratio):.1f} | {group['val_window_fall_precision'].mean():.4f} | "
            f"{group['val_window_fall_recall'].mean():.4f} | {group['val_window_fall_f1'].mean():.4f} | "
            f"{group['val_window_fall_false_positives'].mean():.2f} | {group['val_window_fall_false_negatives'].mean():.2f} |"
        )
    output_root.joinpath("FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_synthetic_fall_pilot(
    dataset_path: Path,
    manifest_path: Path,
    output_root: Path,
    cfg: GroupedCVConfig,
    ratios: tuple[float, ...] = DEFAULT_RATIOS,
    baseline_metrics_path: Path | None = MATCHED_BASELINE_OUTPUT / "fold_metrics.csv",
) -> dict[str, object]:
    if output_root.exists():
        raise FileExistsError(f"Output directory already exists: {output_root}")
    payload, manifest = load_or_create_manifest(dataset_path, manifest_path, cfg, force=False)
    _load_extra_metadata(dataset_path, payload)
    output_root.mkdir(parents=True, exist_ok=False)
    dev = device()
    fold_metrics = []
    for fold in range(1, cfg.folds + 1):
        metrics = train_fold_with_ratio_grid(fold, payload, manifest, cfg, ratios, output_root, dev)
        fold_metrics.append(metrics)
        print(
            f"fold {fold}/{cfg.folds}: selected_ratio={metrics['selected_ratio']:.1f} "
            f"test_fall_f1={metrics['window_fall_f1']:.4f} "
            f"test_fall_precision={metrics['window_fall_precision']:.4f} "
            f"test_fall_recall={metrics['window_fall_recall']:.4f}"
        )
    aggregate = aggregate_metrics(fold_metrics)
    fold_df = pd.DataFrame(fold_metrics)
    fold_df.drop(columns=["window_confusion_matrix", "session_confusion_matrix"]).to_csv(output_root / "fold_metrics.csv", index=False)
    ablations = []
    for fold in range(1, cfg.folds + 1):
        ablations.append(pd.read_csv(output_root / f"fold_{fold}" / "validation_ablation.csv"))
    ablation = pd.concat(ablations, ignore_index=True)
    ablation.to_csv(output_root / "validation_ablation_all_folds.csv", index=False)
    summary = {
        "dataset": str(dataset_path),
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_checksum(manifest_path),
        "config": asdict(cfg),
        "ratios": list(ratios),
        "matched_baseline_metrics": str(baseline_metrics_path) if baseline_metrics_path else "",
        "real_only_validation": True,
        "real_only_outer_test": True,
        "folds": fold_metrics,
        "aggregate": aggregate,
    }
    (output_root / "cv_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_report(output_root, fold_metrics, aggregate, ablation, baseline_metrics_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DEFAULT_DATASET))
    parser.add_argument("--manifest", default=str(DEFAULT_DATE_MATCHED_MANIFEST if DEFAULT_DATE_MATCHED_MANIFEST.exists() else DEFAULT_MANIFEST))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-folds", type=int, default=6)
    parser.add_argument("--ratios", default="0,0.5,1,2")
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
    ratios = tuple(float(x) for x in args.ratios.split(",") if x.strip())
    summary = run_synthetic_fall_pilot(Path(args.data), Path(args.manifest), Path(args.output_dir), cfg, ratios)
    print(f"saved_summary={Path(args.output_dir) / 'cv_summary.json'}")
    print(f"real_only_outer_test={summary['real_only_outer_test']}")


if __name__ == "__main__":
    main()
