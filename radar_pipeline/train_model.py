from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, auc, classification_report, confusion_matrix, f1_score, recall_score, roc_curve
from sklearn.preprocessing import label_binarize
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from radar_pipeline.common import CLASS_LABELS, CLASS_NAMES, ROOT


EXPECTED_FEATURES = ["x", "y", "z", "dop_idx", "range_m", "azimuth_deg", "elevation_deg"]


@dataclass
class TrainConfig:
    epochs: int = 80
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 12
    seed: int = 42
    feature_mode: str = "raw"
    dropout_input: float = 0.25
    dropout_hidden: float = 0.20


class CNNLSTM(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_classes: int,
        dropout_input: float = 0.25,
        dropout_hidden: float = 0.20,
    ) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(64, 96, kernel_size=3, padding=1),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.lstm = nn.LSTM(input_size=96, hidden_size=64, num_layers=1, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_input),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_hidden),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, time, features] -> Conv1d expects [batch, features, time].
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        out, _ = self.lstm(x)
        return self.classifier(out[:, -1, :])


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_dataset(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    data = np.load(path, allow_pickle=True)
    payload = {key: data[key] for key in data.files}
    feature_names = [str(x) for x in payload["feature_names"]]
    label_names = [str(x) for x in payload["label_names"]]
    if feature_names != EXPECTED_FEATURES:
        raise ValueError(f"Unexpected feature order: {feature_names} != {EXPECTED_FEATURES}")
    for split in ("train", "val", "test"):
        y = payload[f"y_{split}"]
        unexpected = sorted(set(int(v) for v in y) - set(CLASS_LABELS))
        if unexpected:
            raise ValueError(f"{split} contains labels outside {CLASS_LABELS}: {unexpected}")
    info = {
        "path": str(path),
        "feature_names": feature_names,
        "label_names": label_names,
        "splits": {
            split: {
                "X_shape": list(payload[f"X_{split}"].shape),
                "y_shape": list(payload[f"y_{split}"].shape),
                "class_counts": {
                    CLASS_NAMES[label]: int((payload[f"y_{split}"] == label).sum()) for label in CLASS_LABELS
                },
                "sessions": int(len(np.unique(payload[f"session_id_{split}"]))),
            }
            for split in ("train", "val", "test")
        },
    }
    return payload, info


def rolling_std_feature(values: np.ndarray, radius: int = 2) -> np.ndarray:
    out = np.zeros_like(values, dtype=np.float32)
    n_windows, n_steps, _ = values.shape
    for t in range(n_steps):
        lo = max(0, t - radius)
        hi = min(n_steps, t + radius + 1)
        out[:, t, :] = values[:, lo:hi, :].std(axis=1)
    return out


def augment_window_features(data: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], list[str]]:
    feature_names = [str(x) for x in data["feature_names"]]
    idx = {name: feature_names.index(name) for name in feature_names}
    augmented = dict(data)
    new_feature_names = feature_names + [
        "xyz_delta_mag",
        "x_roll_std",
        "y_roll_std",
        "z_roll_std",
        "range_roll_std",
        "range_centered",
    ]
    for split in ("train", "val", "test"):
        X = data[f"X_{split}"].astype(np.float32)
        xyz = X[:, :, [idx["x"], idx["y"], idx["z"]]]
        dxyz = np.diff(xyz, axis=1, prepend=xyz[:, :1, :])
        xyz_delta_mag = np.linalg.norm(dxyz, axis=2, keepdims=True).astype(np.float32)
        std_cols = X[:, :, [idx["x"], idx["y"], idx["z"], idx["range_m"]]]
        roll_std = rolling_std_feature(std_cols, radius=2)
        range_values = X[:, :, idx["range_m"] : idx["range_m"] + 1]
        range_centered = range_values - range_values.mean(axis=1, keepdims=True)
        augmented[f"X_{split}"] = np.concatenate([X, xyz_delta_mag, roll_std, range_centered], axis=2).astype(np.float32)
    augmented["feature_names"] = np.asarray(new_feature_names)
    return augmented, new_feature_names


def apply_feature_mode(data: dict[str, np.ndarray], dataset_info: dict, feature_mode: str) -> tuple[dict[str, np.ndarray], dict]:
    if feature_mode == "raw":
        return data, dataset_info
    if feature_mode != "augmented":
        raise ValueError(f"Unknown feature_mode: {feature_mode}")
    data, feature_names = augment_window_features(data)
    updated = dict(dataset_info)
    updated["feature_names"] = feature_names
    updated["feature_mode"] = feature_mode
    for split in ("train", "val", "test"):
        updated["splits"][split]["X_shape"] = list(data[f"X_{split}"].shape)
    return data, updated


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).long())
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def class_weights(y_train: np.ndarray, dev: torch.device) -> torch.Tensor:
    counts = np.asarray([(y_train == label).sum() for label in CLASS_LABELS], dtype=np.float32)
    weights = counts.sum() / (len(CLASS_LABELS) * np.maximum(counts, 1.0))
    return torch.tensor(weights, dtype=torch.float32, device=dev)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    dev: torch.device,
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    y_prob: list[list[float]] = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(dev)
            yb = yb.to(dev)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            total_loss += float(loss.item()) * len(yb)
            total += len(yb)
            pred = logits.argmax(dim=1)
            prob = torch.softmax(logits, dim=1)
            y_true.extend(yb.cpu().numpy().tolist())
            y_pred.extend(pred.cpu().numpy().tolist())
            y_prob.extend(prob.cpu().numpy().tolist())
    avg_loss = total_loss / max(total, 1)
    acc = accuracy_score(y_true, y_pred) if y_true else 0.0
    return avg_loss, float(acc), np.asarray(y_true), np.asarray(y_pred), np.asarray(y_prob)


def confirm_temporal_fall_predictions(
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    session_ids: np.ndarray,
    start_frames: np.ndarray,
    end_frames: np.ndarray,
) -> np.ndarray:
    """Suppress isolated fall predictions using session-local overlapping windows.

    A run of at least two overlapping fall predictions is retained. An isolated
    fall is replaced by its highest-probability non-fall class. This can be used
    online with one-window confirmation latency.
    """
    confirmed = np.asarray(y_pred, dtype=np.int64).copy()
    probabilities = np.asarray(y_prob)
    sessions = np.asarray(session_ids).astype(str)
    starts = np.asarray(start_frames)
    ends = np.asarray(end_frames)
    if not (len(confirmed) == len(probabilities) == len(sessions) == len(starts) == len(ends)):
        raise ValueError("Temporal confirmation inputs must have the same number of windows")
    if probabilities.ndim != 2 or probabilities.shape[1] != len(CLASS_LABELS):
        raise ValueError(f"Expected probabilities with shape [windows, {len(CLASS_LABELS)}]")

    fall_label = CLASS_NAMES.index("fall")
    for session_id in np.unique(sessions):
        indices = np.flatnonzero(sessions == session_id)
        ordered = indices[np.argsort(starts[indices], kind="stable")]
        run: list[int] = []

        def finish_run() -> None:
            if len(run) == 1:
                idx = run[0]
                non_fall = [label for label in CLASS_LABELS if label != fall_label]
                confirmed[idx] = non_fall[int(np.argmax(probabilities[idx, non_fall]))]
            run.clear()

        for idx in ordered:
            if confirmed[idx] != fall_label:
                finish_run()
                continue
            if run and starts[idx] >= ends[run[-1]]:
                finish_run()
            run.append(int(idx))
        finish_run()
    return confirmed


def prediction_summary(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, object]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=CLASS_LABELS, average="macro", zero_division=0)),
        "fall_recall": float(
            recall_score(y_true, y_pred, labels=[CLASS_NAMES.index("fall")], average="macro", zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=CLASS_LABELS).tolist(),
    }


def select_temporal_confirmation(
    y_true_val: np.ndarray,
    baseline_val_pred: np.ndarray,
    temporal_val_pred: np.ndarray,
) -> dict[str, object]:
    """Select temporal confirmation from validation predictions only."""
    baseline = prediction_summary(y_true_val, baseline_val_pred)
    temporal = prediction_summary(y_true_val, temporal_val_pred)
    selected = bool(
        temporal["macro_f1"] > baseline["macro_f1"] + 1e-12
        and temporal["fall_recall"] >= baseline["fall_recall"] - 1e-12
    )
    return {
        "selected": selected,
        "selection_rule": "validation macro_f1 must improve and validation fall_recall must not decrease",
        "baseline": baseline,
        "temporal_confirmation": temporal,
    }


def evaluate_temporal_confirmation_read_only(
    data_path: Path,
    checkpoint_path: Path,
    batch_size: int = 32,
) -> dict[str, object]:
    """Evaluate a validation-selected temporal rule without writing artifacts."""
    dev = device()
    data, dataset_info = load_dataset(data_path)
    ckpt = torch.load(checkpoint_path, map_location=dev, weights_only=False)
    train_config = ckpt.get("train_config", {})
    feature_mode = train_config.get("feature_mode", "raw")
    data, dataset_info = apply_feature_mode(data, dataset_info, feature_mode)
    model = CNNLSTM(
        n_features=data["X_train"].shape[-1],
        n_classes=len(CLASS_LABELS),
        dropout_input=float(train_config.get("dropout_input", 0.25)),
        dropout_hidden=float(train_config.get("dropout_hidden", 0.20)),
    ).to(dev)
    model.load_state_dict(ckpt["model_state_dict"])
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(data["y_train"], dev))

    predictions: dict[str, dict[str, np.ndarray]] = {}
    for split in ("val", "test"):
        loader = make_loader(data[f"X_{split}"], data[f"y_{split}"], batch_size, shuffle=False)
        _, _, y_true, y_pred, y_prob = evaluate(model, loader, loss_fn, dev)
        temporal_pred = confirm_temporal_fall_predictions(
            y_pred,
            y_prob,
            data[f"session_id_{split}"],
            data[f"start_frame_{split}"],
            data[f"end_frame_{split}"],
        )
        predictions[split] = {"true": y_true, "baseline": y_pred, "temporal": temporal_pred}

    selection = select_temporal_confirmation(
        predictions["val"]["true"],
        predictions["val"]["baseline"],
        predictions["val"]["temporal"],
    )
    frozen_test_pred = predictions["test"]["temporal"] if selection["selected"] else predictions["test"]["baseline"]
    return {
        "dataset": str(data_path),
        "checkpoint": str(checkpoint_path),
        "feature_mode": feature_mode,
        "selection": selection,
        "test_baseline": prediction_summary(predictions["test"]["true"], predictions["test"]["baseline"]),
        "test_frozen_rule": prediction_summary(predictions["test"]["true"], frozen_test_pred),
        "test_changed_windows": int(np.sum(predictions["test"]["baseline"] != frozen_test_pred)),
    }


def architecture_summary(model: nn.Module, input_shape: tuple[int, int], dev: torch.device) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    hooks = []

    def hook(name: str):
        def _hook(module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor | tuple[torch.Tensor, ...]) -> None:
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output
            rows.append(
                {
                    "layer": name,
                    "type": module.__class__.__name__,
                    "output_shape": list(out.shape),
                    "params": int(sum(p.numel() for p in module.parameters() if p.requires_grad)),
                }
            )

        return _hook

    for name, module in model.named_modules():
        if name and not any(module.children()):
            hooks.append(module.register_forward_hook(hook(name)))
    model.eval()
    with torch.no_grad():
        model(torch.zeros((1, *input_shape), device=dev))
    for h in hooks:
        h.remove()
    return rows


def plot_history(history: list[dict[str, float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, [h["train_loss"] for h in history], label="train")
    axes[0].plot(epochs, [h["val_loss"] for h in history], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].plot(epochs, [h["train_acc"] for h in history], label="train")
    axes[1].plot(epochs, [h["val_acc"] for h in history], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrices(cm: np.ndarray, output_dir: Path, prefix: str = "") -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / f"{prefix}confusion_matrix.png"
    norm_path = output_dir / f"{prefix}confusion_matrix_normalized.png"

    def draw(matrix: np.ndarray, path: Path, title: str, fmt: str) -> None:
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(matrix, cmap="Blues")
        fig.colorbar(im, ax=ax)
        ax.set_xticks(np.arange(len(CLASS_NAMES)), labels=CLASS_NAMES, rotation=30, ha="right")
        ax.set_yticks(np.arange(len(CLASS_NAMES)), labels=CLASS_NAMES)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(title)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, format(matrix[i, j], fmt), ha="center", va="center", color="black")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

    draw(cm, raw_path, "Confusion Matrix (Counts)", "d")
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0) * 100.0
    draw(cm_norm, norm_path, "Confusion Matrix (Row-Normalized %)", ".1f")
    return raw_path, norm_path


def save_classification_reports(report: dict, report_text: str, output_dir: Path, prefix: str = "") -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    per_class_path = output_dir / f"{prefix}cnn_lstm_per_class_metrics.csv"
    report_csv_path = output_dir / f"{prefix}cnn_lstm_classification_report.csv"
    report_txt_path = output_dir / f"{prefix}cnn_lstm_classification_report.txt"

    per_class_rows = []
    for name in CLASS_NAMES:
        metrics = report[name]
        per_class_rows.append(
            {
                "class": name,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1-score"],
                "support": int(metrics["support"]),
            }
        )
    pd.DataFrame(per_class_rows).to_csv(per_class_path, index=False)

    rows = []
    for key, value in report.items():
        if isinstance(value, dict):
            rows.append(
                {
                    "label": key,
                    "precision": value.get("precision"),
                    "recall": value.get("recall"),
                    "f1_score": value.get("f1-score"),
                    "support": value.get("support"),
                }
            )
        else:
            rows.append({"label": key, "precision": None, "recall": None, "f1_score": value, "support": None})
    pd.DataFrame(rows).to_csv(report_csv_path, index=False)
    report_txt_path.write_text(report_text, encoding="utf-8")
    return per_class_path, report_csv_path, report_txt_path


def plot_roc_curves(y_true: np.ndarray, y_prob: np.ndarray, output_dir: Path, prefix: str = "") -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    roc_path = output_dir / f"{prefix}roc_curves.png"
    y_bin = label_binarize(y_true, classes=CLASS_LABELS)
    fig, ax = plt.subplots(figsize=(7, 5))
    for idx, name in enumerate(CLASS_NAMES):
        if y_bin[:, idx].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, idx], y_prob[:, idx])
        ax.plot(fpr, tpr, label=f"{name} AUC={auc(fpr, tpr):.3f}")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("One-vs-Rest ROC Curves")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(roc_path, dpi=150)
    plt.close(fig)
    return roc_path


def save_evaluation_outputs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    report: dict,
    report_text: str,
    output_dir: Path,
    plot_dir: Path,
    prefix: str = "",
) -> dict[str, str]:
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_LABELS)
    cm_path, cm_norm_path = plot_confusion_matrices(cm, plot_dir, prefix)
    per_class_path, report_csv_path, report_txt_path = save_classification_reports(report, report_text, output_dir, prefix)
    roc_path = plot_roc_curves(y_true, y_prob, plot_dir, prefix)
    return {
        "confusion_matrix_plot": str(cm_path),
        "confusion_matrix_normalized_plot": str(cm_norm_path),
        "per_class_metrics_csv": str(per_class_path),
        "classification_report_csv": str(report_csv_path),
        "classification_report_txt": str(report_txt_path),
        "roc_curves_plot": str(roc_path),
    }


def fit_validation_only(data_path: Path, cfg: TrainConfig) -> dict[str, object]:
    """Fit using train/validation splits without reading or evaluating test windows."""
    set_seed(cfg.seed)
    dev = device()
    data, dataset_info = load_dataset(data_path)
    data, dataset_info = apply_feature_mode(data, dataset_info, cfg.feature_mode)
    train_loader = make_loader(data["X_train"], data["y_train"], cfg.batch_size, shuffle=True)
    val_loader = make_loader(data["X_val"], data["y_val"], cfg.batch_size, shuffle=False)

    model = CNNLSTM(
        n_features=data["X_train"].shape[-1],
        n_classes=len(CLASS_LABELS),
        dropout_input=cfg.dropout_input,
        dropout_hidden=cfg.dropout_hidden,
    ).to(dev)
    weights = class_weights(data["y_train"], dev)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    summary_rows = architecture_summary(model, tuple(data["X_train"].shape[1:]), dev)
    history: list[dict[str, float]] = []
    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
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

    train_eval_loader = make_loader(data["X_train"], data["y_train"], cfg.batch_size, shuffle=False)
    train_loss, train_acc, train_true, train_pred, _ = evaluate(model, train_eval_loader, loss_fn, dev)
    val_loss, val_acc, val_true, val_pred, val_prob = evaluate(model, val_loader, loss_fn, dev)
    val_report = classification_report(
        val_true,
        val_pred,
        labels=CLASS_LABELS,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    return {
        "model": model,
        "device": dev,
        "data": data,
        "dataset_info": dataset_info,
        "weights": weights,
        "loss_fn": loss_fn,
        "architecture_summary": summary_rows,
        "history": history,
        "best_epoch": best_epoch,
        "best_train_loss": float(train_loss),
        "best_train_accuracy": float(train_acc),
        "best_val_loss": float(val_loss),
        "best_val_accuracy": float(val_acc),
        "validation_report": val_report,
        "validation_confusion_matrix": confusion_matrix(val_true, val_pred, labels=CLASS_LABELS).tolist(),
        "validation_probabilities": val_prob,
        "train_predictions": {"y_true": train_true, "y_pred": train_pred},
        "validation_predictions": {"y_true": val_true, "y_pred": val_pred},
    }


def validation_checkpoint_metrics(fit: dict[str, object]) -> dict[str, object]:
    """Build explicit metrics for the validation-loss-selected checkpoint."""
    report = fit["validation_report"]
    fall = report["fall"]
    return {
        "selection_metric": "minimum_validation_loss",
        "best_epoch": int(fit["best_epoch"]),
        "best_val_loss": float(fit["best_val_loss"]),
        "loss": float(fit["best_val_loss"]),
        "accuracy": float(fit["best_val_accuracy"]),
        "fall_precision": float(fall["precision"]),
        "fall_recall": float(fall["recall"]),
        "fall_f1": float(fall["f1-score"]),
        "classification_report": report,
        "confusion_matrix": fit["validation_confusion_matrix"],
        "train_loss_at_best_epoch": float(fit["best_train_loss"]),
        "train_accuracy_at_best_epoch": float(fit["best_train_accuracy"]),
    }


def train(
    data_path: Path,
    output_dir: Path,
    cfg: TrainConfig,
    detailed_evaluation_outputs: bool = True,
) -> dict:
    fit = fit_validation_only(data_path, cfg)
    dev = fit["device"]
    data = fit["data"]
    dataset_info = fit["dataset_info"]
    model = fit["model"]
    weights = fit["weights"]
    loss_fn = fit["loss_fn"]
    summary_rows = fit["architecture_summary"]
    history = fit["history"]
    best_epoch = fit["best_epoch"]
    validation_metrics = validation_checkpoint_metrics(fit)
    prefix = "" if cfg.feature_mode == "raw" else f"{cfg.feature_mode}_"
    test_loader = make_loader(data["X_test"], data["y_test"], cfg.batch_size, shuffle=False)

    test_loss, test_acc, y_true, y_pred, y_prob = evaluate(model, test_loader, loss_fn, dev)
    report = classification_report(y_true, y_pred, labels=CLASS_LABELS, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    report_text = classification_report(y_true, y_pred, labels=CLASS_LABELS, target_names=CLASS_NAMES, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_LABELS)
    output_paths = (
        save_evaluation_outputs(
            y_true,
            y_pred,
            y_prob,
            report,
            report_text,
            output_dir,
            ROOT / "outputs/validation/training",
            prefix,
        )
        if detailed_evaluation_outputs
        else {}
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = (
        ROOT / f"outputs/validation/training/{prefix}cnn_lstm_training_curves.png"
        if detailed_evaluation_outputs
        else output_dir / f"{prefix}cnn_lstm_training_curves.png"
    )
    plot_history(history, plot_path)
    ckpt_path = output_dir / f"cnn_lstm_radar_{cfg.feature_mode}.pt" if cfg.feature_mode != "raw" else output_dir / "cnn_lstm_radar.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_class": "CNNLSTM",
            "feature_names": dataset_info["feature_names"],
            "class_names": CLASS_NAMES,
            "label_mapping": {name: label for name, label in zip(CLASS_NAMES, CLASS_LABELS)},
            "train_config": asdict(cfg),
            "dataset_info": dataset_info,
            "architecture_summary": summary_rows,
            "history": history,
            "best_epoch": best_epoch,
            "best_val_loss": validation_metrics["best_val_loss"],
            "validation_metrics": validation_metrics,
            "test_metrics": {
                "loss": test_loss,
                "accuracy": test_acc,
                "classification_report": report,
                "confusion_matrix": cm.tolist(),
            },
            "evaluation_outputs": output_paths,
        },
        ckpt_path,
    )
    metrics_path = output_dir / f"{prefix}cnn_lstm_metrics.json"
    metrics = {
        "dataset_info": dataset_info,
        "train_config": asdict(cfg),
        "device": str(dev),
        "architecture_summary": summary_rows,
        "history": history,
        "best_epoch": best_epoch,
        "best_val_loss": validation_metrics["best_val_loss"],
        "validation_metrics": validation_metrics,
        "class_weights": weights.detach().cpu().numpy().tolist(),
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "checkpoint": str(ckpt_path),
        "training_curve": str(plot_path),
        "evaluation_outputs": output_paths,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("Dataset structure:")
    print(json.dumps(dataset_info, indent=2))
    print("\nModel architecture:")
    print(model)
    print("\nLayer output shapes:")
    for row in summary_rows:
        print(f"{row['layer']:<24} {row['type']:<14} output={row['output_shape']} params={row['params']}")
    print(f"\nBest epoch: {best_epoch}")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    print("\nClassification report:")
    print(report_text)
    print(f"Confusion matrix rows=true, cols=pred {CLASS_NAMES}:")
    print(cm)
    print(f"\nSaved checkpoint: {ckpt_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved training curve: {plot_path}")
    return metrics


def evaluate_checkpoint(data_path: Path, checkpoint_path: Path, output_dir: Path, batch_size: int = 32) -> dict:
    dev = device()
    data, dataset_info = load_dataset(data_path)
    ckpt = torch.load(checkpoint_path, map_location=dev, weights_only=False)
    train_config = ckpt.get("train_config", {})
    feature_mode = train_config.get("feature_mode", "raw")
    data, dataset_info = apply_feature_mode(data, dataset_info, feature_mode)
    prefix = "" if feature_mode == "raw" else f"{feature_mode}_"
    model = CNNLSTM(
        n_features=data["X_test"].shape[-1],
        n_classes=len(CLASS_LABELS),
        dropout_input=float(train_config.get("dropout_input", 0.25)),
        dropout_hidden=float(train_config.get("dropout_hidden", 0.20)),
    ).to(dev)
    model.load_state_dict(ckpt["model_state_dict"])
    weights = class_weights(data["y_train"], dev)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    test_loader = make_loader(data["X_test"], data["y_test"], batch_size, shuffle=False)
    test_loss, test_acc, y_true, y_pred, y_prob = evaluate(model, test_loader, loss_fn, dev)
    report = classification_report(y_true, y_pred, labels=CLASS_LABELS, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    report_text = classification_report(y_true, y_pred, labels=CLASS_LABELS, target_names=CLASS_NAMES, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_LABELS)
    output_paths = save_evaluation_outputs(
        y_true,
        y_pred,
        y_prob,
        report,
        report_text,
        output_dir,
        ROOT / "outputs/validation/training",
        prefix,
    )
    metrics = {
        "dataset_info": dataset_info,
        "device": str(dev),
        "checkpoint": str(checkpoint_path),
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "evaluation_outputs": output_paths,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / f"{prefix}cnn_lstm_eval_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("Evaluation-only run from checkpoint")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    print("\nClassification report:")
    print(report_text)
    print(f"Confusion matrix rows=true, cols=pred {CLASS_NAMES}:")
    print(cm)
    print("\nSaved evaluation outputs:")
    for name, path in output_paths.items():
        print(f"{name}: {path}")
    print(f"metrics_json: {metrics_path}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT / "data/final_dataset/radar_dataset.npz"))
    parser.add_argument("--output-dir", default=str(ROOT / "outputs/models"))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--feature-mode", choices=("raw", "augmented"), default="raw")
    parser.add_argument("--dropout-input", type=float, default=0.25)
    parser.add_argument("--dropout-hidden", type=float, default=0.20)
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--checkpoint", default=str(ROOT / "outputs/models/cnn_lstm_radar.pt"))
    args = parser.parse_args()
    if args.evaluate_only:
        evaluate_checkpoint(Path(args.data), Path(args.checkpoint), Path(args.output_dir), args.batch_size)
        return
    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        seed=args.seed,
        feature_mode=args.feature_mode,
        dropout_input=args.dropout_input,
        dropout_hidden=args.dropout_hidden,
    )
    train(Path(args.data), Path(args.output_dir), cfg)


if __name__ == "__main__":
    main()
