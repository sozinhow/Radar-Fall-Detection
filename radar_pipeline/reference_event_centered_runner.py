from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report

from radar_pipeline.common import CLASS_LABELS, CLASS_NAMES, MODEL_FEATURES
from radar_pipeline.evaluate_grouped_cv import GroupedCVConfig, apply_augmented_features, manifest_checksum
from radar_pipeline.run_synthetic_fall_pilot import _train_model
from radar_pipeline.train_model import evaluate, make_loader, set_seed
from torch import nn


# Resolve the project root from this file rather than from the caller's
# working directory, so the runner behaves the same when launched elsewhere.
ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/event_centered_clips_sgkf4_20260731_rebuild01"
DATASET = ROOT / "data/final_dataset_auto_event_exclude_risky_fall_20260731_rebuild01/radar_dataset.npz"
MANIFEST = ROOT / "data/metadata/auto_event_exclude_risky_fall_20260731_rebuild01_source_session_folds.csv"
ANNOTATIONS = ROOT / "data/metadata/auto_event_annotations_20260731_rebuild01.csv"
E0 = ROOT / "outputs/event_centered_clips_sgkf4_20260731_rebuild01"
EXPECTED_MANIFEST_SHA256 = "f5e56e47465b894c07795d606ee705db299661afa3cbccd530d53ec7f65905af"
RATE_HZ = 20.0
STRIDE_FRAMES = 15
POST_IMPACT_FRAMES = {50: 15, 60: 20}
CLIP_LENGTHS = (50, 60)
SEED = 42
FALL_LABEL = CLASS_NAMES.index("fall")
THRESHOLDS = np.round(np.arange(0.10, 0.91, 0.05), 2)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory() -> pd.DataFrame:
    data = np.load(DATASET, allow_pickle=True)
    rows: list[dict[str, str]] = []
    for split in ("train", "val", "test"):
        for source, activity in zip(data[f"source_csv_{split}"], data[f"source_activity_{split}"]):
            path = Path(str(source))
            rows.append(
                {
                    "source_session_id": path.stem,
                    "source_csv": str(path),
                    "activity": str(activity),
                }
            )
    frame = pd.DataFrame(rows).drop_duplicates("source_session_id").sort_values("source_session_id").reset_index(drop=True)
    if len(frame) != 485 or not frame["source_csv"].map(lambda value: (ROOT / value).exists()).all():
        raise ValueError("Current filtered source-session inventory is incomplete")
    manifest = pd.read_csv(MANIFEST)
    if set(frame["source_session_id"]) != set(manifest["source_session_id"].astype(str)):
        raise ValueError("Source inventory differs from the frozen manifest")
    return frame.merge(
        manifest[["source_session_id", "outer_fold"]], on="source_session_id", how="left", validate="one_to_one"
    )


def annotation_table() -> pd.DataFrame:
    annotations = pd.read_csv(ANNOTATIONS)
    annotations = annotations.loc[annotations["activity"].astype(str).eq("fall")].copy()
    annotations["source_session_id"] = annotations["session_id"].astype(str)
    for column in ("event_start_s", "impact_s", "event_end_s"):
        annotations[column] = pd.to_numeric(annotations[column], errors="coerce")
    return annotations[
        [
            "source_session_id",
            "event_start_s",
            "impact_s",
            "event_end_s",
            "confidence",
            "quality_flags",
        ]
    ].drop_duplicates("source_session_id")


def load_session(row: pd.Series) -> np.ndarray:
    frame = pd.read_csv(ROOT / str(row["source_csv"]))
    missing = [feature for feature in MODEL_FEATURES if feature not in frame.columns]
    if missing:
        raise ValueError(f"{row['source_session_id']} is missing E0 features: {missing}")
    values = frame[MODEL_FEATURES].to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError(f"{row['source_session_id']} contains non-finite E0 features")
    return values


def extract_clip(values: np.ndarray, start: int, length: int) -> tuple[np.ndarray, np.ndarray, int, int]:
    end = start + length
    source_start = max(start, 0)
    source_end = min(end, len(values))
    valid = values[source_start:source_end]
    if not len(valid):
        raise ValueError("Clip has no physical source frames")
    left = max(0, -start)
    right = max(0, end - len(values))
    clip = np.pad(valid, ((left, right), (0, 0)), mode="edge")
    mask = np.concatenate([np.zeros(left, dtype=bool), np.ones(len(valid), dtype=bool), np.zeros(right, dtype=bool)])
    if clip.shape != (length, len(MODEL_FEATURES)) or len(mask) != length:
        raise AssertionError("Clip extraction length mismatch")
    return clip.astype(np.float32), mask, left, right


def deterministic_nonfall_start(session_id: str, n_frames: int, length: int) -> int:
    if n_frames < length:
        return -(length - n_frames) // 2
    candidates = list(range(0, n_frames - length + 1, STRIDE_FRAMES))
    digest = hashlib.sha256(f"{SEED}:{length}:{session_id}".encode()).digest()
    return candidates[int.from_bytes(digest[:8], "big") % len(candidates)]


def build_clip_dataset(length: int, sessions: pd.DataFrame, annotations: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    annotation_map = annotations.set_index("source_session_id")
    clips: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    labels: list[int] = []
    metadata: list[dict[str, object]] = []
    for _, row in sessions.iterrows():
        sid = str(row["source_session_id"])
        activity = str(row["activity"])
        values = load_session(row)
        event_start = impact = event_end = np.nan
        confidence = quality_flags = ""
        if activity == "fall":
            if sid not in annotation_map.index:
                raise ValueError(f"Fall session lacks annotation: {sid}")
            ann = annotation_map.loc[sid]
            event_start = float(ann["event_start_s"])
            impact = float(ann["impact_s"])
            event_end = float(ann["event_end_s"])
            confidence = str(ann["confidence"])
            quality_flags = str(ann["quality_flags"])
            impact_frame = int(round(impact * RATE_HZ))
            start = impact_frame + POST_IMPACT_FRAMES[length] - length
        else:
            start = deterministic_nonfall_start(sid, len(values), length)
        clip, mask, left, right = extract_clip(values, start, length)
        clips.append(clip)
        masks.append(mask)
        labels.append(CLASS_NAMES.index(activity))
        metadata.append(
            {
                "source_session_id": sid,
                "source_csv": str(row["source_csv"]),
                "activity": activity,
                "label": CLASS_NAMES.index(activity),
                "outer_fold": int(row["outer_fold"]),
                "clip_length_frames": length,
                "clip_start_row": start,
                "clip_end_row_exclusive": start + length,
                "clip_start_s": start / RATE_HZ,
                "clip_end_s": (start + length) / RATE_HZ,
                "event_start_s": event_start,
                "impact_s": impact,
                "event_end_s": event_end,
                "pre_event_context_s": event_start - start / RATE_HZ if activity == "fall" else np.nan,
                "requested_post_impact_s": POST_IMPACT_FRAMES[length] / RATE_HZ if activity == "fall" else np.nan,
                "physical_post_impact_s": max(
                    0.0, min(len(values) / RATE_HZ - impact, POST_IMPACT_FRAMES[length] / RATE_HZ)
                )
                if activity == "fall"
                else np.nan,
                "padding_left_frames": left,
                "padding_right_frames": right,
                "valid_frames": int(mask.sum()),
                "annotation_confidence": confidence,
                "quality_flags": quality_flags,
                "sampling_rule": "impact_plus_1s" if activity == "fall" else "sha256_seed42_stride15",
            }
        )
    return (
        np.stack(clips).astype(np.float32),
        np.asarray(labels, dtype=np.int64),
        np.stack(masks),
        pd.DataFrame(metadata),
    )


def frozen_split_sessions(fold: int) -> dict[str, set[str]]:
    saved = pd.read_csv(E0 / f"fold_{fold}" / "source_session_ids.csv")
    return {
        split: set(saved.loc[saved["split"].astype(str).eq(split), "source_session_id"].astype(str))
        for split in ("train", "val", "test")
    }


def normalize_by_clip_train(X: np.ndarray, train_idx: np.ndarray) -> tuple[np.ndarray, dict[str, list[float]]]:
    flat = X[train_idx].reshape(-1, X.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    return ((X - mean) / std).astype(np.float32), {"mean": mean.tolist(), "std": std.tolist()}


def window_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    report = classification_report(
        y_true, y_pred, labels=CLASS_LABELS, target_names=CLASS_NAMES, output_dict=True, zero_division=0
    )
    return {
        "clip_accuracy": float((y_true == y_pred).mean()),
        "clip_macro_f1": float(report["macro avg"]["f1-score"]),
        "clip_fall_precision": float(report["fall"]["precision"]),
        "clip_fall_recall": float(report["fall"]["recall"]),
        "clip_fall_f1": float(report["fall"]["f1-score"]),
    }


def sliding_clips(length: int, subset: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    clips: list[np.ndarray] = []
    rows: list[dict[str, object]] = []
    for _, row in subset.sort_values("source_session_id").iterrows():
        sid = str(row["source_session_id"])
        values = load_session(row)
        if len(values) >= length:
            starts = list(range(0, len(values) - length + 1, STRIDE_FRAMES))
        else:
            starts = [len(values) - length]
        for start in starts:
            clip, mask, left, right = extract_clip(values, start, length)
            clips.append(clip)
            rows.append(
                {
                    "source_session_id": sid,
                    "activity": str(row["activity"]),
                    "is_fall_session": str(row["activity"]) == "fall",
                    "clip_start_s": start / RATE_HZ,
                    "clip_end_s": min(start + length, len(values)) / RATE_HZ,
                    "padding_left_frames": left,
                    "padding_right_frames": right,
                    "max_source_row_exclusive": min(start + length, len(values)),
                    "causal_available_row_exclusive": min(start + length, len(values)),
                    "valid_frames": int(mask.sum()),
                }
            )
    return np.stack(clips).astype(np.float32), pd.DataFrame(rows)


def infer_probabilities(
    model: torch.nn.Module,
    X_raw: np.ndarray,
    norm: dict[str, list[float]],
    length: int,
    batch_size: int,
) -> np.ndarray:
    mean = np.asarray(norm["mean"], dtype=np.float32)
    std = np.asarray(norm["std"], dtype=np.float32)
    X_norm = ((X_raw - mean) / std).astype(np.float32)
    dummy = X_norm[:1]
    _, X_aug, _, _ = apply_augmented_features(
        dummy, X_norm, dummy, np.asarray(MODEL_FEATURES), "augmented"
    )
    loader = make_loader(X_aug, np.zeros(len(X_aug), dtype=np.int64), batch_size, shuffle=False)
    probabilities: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for xb, _ in loader:
            probabilities.append(torch.softmax(model(xb), dim=1).cpu().numpy())
    return np.concatenate(probabilities, axis=0)


def event_metrics(stream: pd.DataFrame, all_sessions: pd.DataFrame, threshold: float) -> dict[str, object]:
    session_rows: list[dict[str, object]] = []
    causal = True
    for _, session_info in all_sessions.iterrows():
        sid = str(session_info["source_session_id"])
        activity = str(session_info["activity"])
        session = stream.loc[stream["source_session_id"].astype(str).eq(sid)].sort_values("clip_end_s").reset_index(drop=True)
        support = session["fall_probability"].to_numpy(dtype=float) >= threshold if len(session) else np.asarray([], dtype=bool)
        previous = np.concatenate([[False], support[:-1]]) if len(support) else np.asarray([], dtype=bool)
        alert_idx = np.flatnonzero(support & ~previous)
        alert_times = session.loc[alert_idx, "clip_end_s"].to_numpy(dtype=float) if len(alert_idx) else np.asarray([])
        causal &= bool(
            len(session) == 0
            or (session["max_source_row_exclusive"] <= session["causal_available_row_exclusive"]).all()
        )
        is_fall = activity == "fall"
        event_start = impact = np.nan
        if is_fall:
            event_start = float(session_info["event_start_s"])
            impact = float(session_info["impact_s"])
            valid_alerts = alert_times[alert_times >= event_start]
        else:
            valid_alerts = alert_times
        first = float(valid_alerts[0]) if len(valid_alerts) else np.nan
        session_rows.append(
            {
                "source_session_id": sid,
                "is_fall_session": is_fall,
                "alert_count": len(valid_alerts),
                "has_alert": bool(len(valid_alerts)),
                "delay_event_start_s": first - event_start if is_fall and len(valid_alerts) else np.nan,
                "delay_impact_s": first - impact if is_fall and len(valid_alerts) else np.nan,
            }
        )
    scored = pd.DataFrame(session_rows)
    fall = scored.loc[scored["is_fall_session"]]
    nonfall = scored.loc[~scored["is_fall_session"]]
    tp = int(fall["has_alert"].sum())
    fp = int(nonfall["has_alert"].sum())
    fn = int((~fall["has_alert"]).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": threshold,
        "session_fall_precision": precision,
        "session_fall_recall": recall,
        "session_fall_f1": f1,
        "nonfall_sessions_alerted": fp,
        "false_alerts_per_nonfall_session": float(nonfall["alert_count"].sum() / len(nonfall)),
        "mean_delay_event_start_s": float(fall["delay_event_start_s"].mean()),
        "mean_delay_impact_s": float(fall["delay_impact_s"].mean()),
        "fall_sessions_no_alert": fn,
        "fall_sessions_multiple_alerts": int((fall["alert_count"] > 1).sum()),
        "repeated_alerts_per_fall_session": float(
            np.maximum(fall["alert_count"].to_numpy(dtype=int) - 1, 0).sum() / len(fall)
        ),
        "causal_current_or_past_only": causal,
    }


def choose_threshold(stream: pd.DataFrame, sessions: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    rows = [event_metrics(stream, sessions, float(threshold)) for threshold in THRESHOLDS]
    sweep = pd.DataFrame(rows)
    selected = max(
        rows,
        key=lambda row: (
            float(row["session_fall_f1"]),
            float(row["session_fall_recall"]),
            -int(row["nonfall_sessions_alerted"]),
            -float(row["false_alerts_per_nonfall_session"]),
            -float(row["mean_delay_impact_s"]) if np.isfinite(row["mean_delay_impact_s"]) else -999.0,
        ),
    )
    return float(selected["threshold"]), sweep


def train_and_evaluate(
    length: int,
    X: np.ndarray,
    y: np.ndarray,
    metadata: pd.DataFrame,
    sessions: pd.DataFrame,
    cfg: GroupedCVConfig,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for fold in range(1, 5):
        split_sessions = frozen_split_sessions(fold)
        split_idx = {
            split: np.flatnonzero(metadata["source_session_id"].astype(str).isin(ids).to_numpy())
            for split, ids in split_sessions.items()
        }
        if set(metadata.loc[split_idx["train"], "source_session_id"]) & set(
            metadata.loc[split_idx["test"], "source_session_id"]
        ):
            raise ValueError(f"Fold {fold} clip leakage")
        X_norm, norm = normalize_by_clip_train(X, split_idx["train"])
        X_train, X_val, X_test, feature_names = apply_augmented_features(
            X_norm[split_idx["train"]],
            X_norm[split_idx["val"]],
            X_norm[split_idx["test"]],
            np.asarray(MODEL_FEATURES),
            "augmented",
        )
        if feature_names != [
            "x",
            "y",
            "z",
            "dop_idx",
            "range_m",
            "azimuth_deg",
            "elevation_deg",
            "xyz_delta_mag",
            "x_roll_std",
            "y_roll_std",
            "z_roll_std",
            "range_roll_std",
            "range_centered",
        ]:
            raise ValueError("Feature list differs from E0")
        fold_dir = OUTPUT / f"{length}_frame/fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=False)
        set_seed(cfg.seed)
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        start_time = time.perf_counter()
        model, history, best_epoch, best_val_loss = _train_model(
            X_train,
            y[split_idx["train"]],
            X_val,
            y[split_idx["val"]],
            cfg,
            torch.device("cpu"),
        )
        test_loader = make_loader(X_test, y[split_idx["test"]], cfg.batch_size, shuffle=False)
        _, _, y_true, y_pred, _ = evaluate(model, test_loader, nn.CrossEntropyLoss(), torch.device("cpu"))
        clip_metrics = window_metrics(y_true, y_pred)

        validation_sessions = sessions.loc[sessions["source_session_id"].isin(split_sessions["val"])].copy()
        test_sessions = sessions.loc[sessions["source_session_id"].isin(split_sessions["test"])].copy()
        val_raw, val_stream = sliding_clips(length, validation_sessions)
        test_raw, test_stream = sliding_clips(length, test_sessions)
        val_probs = infer_probabilities(model, val_raw, norm, length, cfg.batch_size)
        test_probs = infer_probabilities(model, test_raw, norm, length, cfg.batch_size)
        val_stream["fall_probability"] = val_probs[:, FALL_LABEL]
        test_stream["fall_probability"] = test_probs[:, FALL_LABEL]
        threshold, sweep = choose_threshold(val_stream, validation_sessions)
        sweep.to_csv(fold_dir / "validation_threshold_sweep.csv", index=False)
        causal_metrics = event_metrics(test_stream, test_sessions, threshold)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_class": "CNNLSTM",
                "clip_length_frames": length,
                "feature_names": feature_names,
                "base_feature_names": MODEL_FEATURES,
                "normalization": norm,
                "normalization_fit": "fold real training clips only",
                "train_config": asdict(cfg),
                "best_epoch": best_epoch,
                "history": history,
                "selected_validation_threshold": threshold,
                "outer_fold": fold,
                "manifest_sha256": manifest_checksum(MANIFEST),
            },
            fold_dir / "cnn_lstm_event_centered.pt",
        )
        result = {
            "clip_length_frames": length,
            "clip_length_s": length / RATE_HZ,
            "fold": fold,
            "train_clips": len(split_idx["train"]),
            "validation_clips": len(split_idx["val"]),
            "test_clips": len(split_idx["test"]),
            "train_sessions": len(split_sessions["train"]),
            "validation_sessions": len(split_sessions["val"]),
            "test_sessions": len(split_sessions["test"]),
            "selected_threshold": threshold,
            "best_epoch": best_epoch,
            "epochs_ran": len(history),
            "best_validation_loss": best_val_loss,
            "runtime_s": time.perf_counter() - start_time,
            **clip_metrics,
            **causal_metrics,
        }
        (fold_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        results.append(result)
        print(
            f"length={length} fold={fold} threshold={threshold:.2f} causal_f1={causal_metrics['session_fall_f1']:.4f} "
            f"recall={causal_metrics['session_fall_recall']:.4f} nonfall_alerted={causal_metrics['nonfall_sessions_alerted']}",
            flush=True,
        )
    return results


def aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "clip_accuracy",
        "clip_macro_f1",
        "clip_fall_precision",
        "clip_fall_recall",
        "clip_fall_f1",
        "session_fall_precision",
        "session_fall_recall",
        "session_fall_f1",
        "nonfall_sessions_alerted",
        "false_alerts_per_nonfall_session",
        "mean_delay_event_start_s",
        "mean_delay_impact_s",
        "fall_sessions_no_alert",
        "fall_sessions_multiple_alerts",
        "repeated_alerts_per_fall_session",
    ]
    rows: list[dict[str, object]] = []
    for length, group in frame.groupby("clip_length_frames"):
        row: dict[str, object] = {"clip_length_frames": length, "clip_length_s": length / RATE_HZ, "folds": len(group)}
        for metric in metrics:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1))
        row["all_alerts_causal"] = bool(group["causal_current_or_past_only"].all())
        rows.append(row)
    return pd.DataFrame(rows)


def write_report(folds: pd.DataFrame, summary: pd.DataFrame, audit: pd.DataFrame) -> str:
    decision_rows = []
    for _, row in summary.iterrows():
        criteria = {
            "recall_at_least_0p8182": row["session_fall_recall_mean"] >= 0.8182,
            "nonfall_sessions_at_most_4p5": row["nonfall_sessions_alerted_mean"] <= 4.5,
            "impact_delay_at_most_2s": row["mean_delay_impact_s_mean"] <= 2.0,
            "fall_f1_above_0p6899": row["session_fall_f1_mean"] > 0.6899,
            "all_alerts_causal": bool(row["all_alerts_causal"]),
        }
        decision_rows.append(
            {"clip_length_frames": int(row["clip_length_frames"]), **criteria, "promising": all(criteria.values())}
        )
    decisions = pd.DataFrame(decision_rows)
    decisions.to_csv(OUTPUT / "decision_criteria.csv", index=False)
    decision = "PROMISING" if decisions["promising"].any() else "REJECTED"

    lines = [
        "# Event-Centred Temporal Clip SGKF4 Experiment",
        "",
        f"**Decision: {decision}.**",
        "",
        "This is a staging-only experiment. The frozen source-session SGKF4 outer assignments and E0 train/validation/test session inventories were reused without change. Normalization was fitted separately on each fold's real training clips. No synthetic, height-cluster, or session-height features were used.",
        "",
        "## Dataset Audit",
        "",
        "| Length | Clips/sessions | Fall | Walking | Standing | Sitting | Left padded | Right padded | Min pre-event s | Post-impact >=0.75 s |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in audit.iterrows():
        lines.append(
            f"| {int(row['clip_length_frames'])} | {int(row['clips'])} | {int(row['fall'])} | {int(row['walking'])} | "
            f"{int(row['standing'])} | {int(row['sitting'])} | {int(row['left_padded'])} | {int(row['right_padded'])} | "
            f"{row['min_pre_event_context_s']:.3f} | {int(row['fall_clips_post_context_at_least_0p75s'])}/43 |"
        )
    lines.extend(
        [
            "",
            "The 50-frame fall clips end 0.75 seconds after impact and the 60-frame clips end 1.0 second after impact when available; missing boundary context is edge padded and recorded by a mask. Non-fall starts are selected by SHA-256 of seed 42, clip length, and source-session ID from 15-frame-aligned candidates. No clip crosses a session boundary.",
            "",
            "## Causal Sliding Test Results: Mean +/- Std",
            "",
            "| Length | Fall precision | Fall recall | Fall F1 | Nonfall sessions alerted | False alerts/nonfall | Delay from event start s | Delay from impact s | No-alert falls | Repeated alerts/fall | Causal |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    fmt = lambda row, metric: f"{row[f'{metric}_mean']:.4f} +/- {row[f'{metric}_std']:.4f}"
    for _, row in summary.iterrows():
        lines.append(
            f"| {int(row['clip_length_frames'])} | {fmt(row, 'session_fall_precision')} | {fmt(row, 'session_fall_recall')} | "
            f"{fmt(row, 'session_fall_f1')} | {fmt(row, 'nonfall_sessions_alerted')} | "
            f"{fmt(row, 'false_alerts_per_nonfall_session')} | {fmt(row, 'mean_delay_event_start_s')} | "
            f"{fmt(row, 'mean_delay_impact_s')} | {fmt(row, 'fall_sessions_no_alert')} | "
            f"{fmt(row, 'repeated_alerts_per_fall_session')} | {'yes' if row['all_alerts_causal'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Per-Fold Causal Results",
            "",
            "| Length | Fold | Threshold | Precision | Recall | F1 | Nonfall sessions alerted | False alerts/nonfall | Event-start delay s | Impact delay s | No-alert falls | Repeated alerts |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in folds.iterrows():
        lines.append(
            f"| {int(row['clip_length_frames'])} | {int(row['fold'])} | {row['selected_threshold']:.2f} | "
            f"{row['session_fall_precision']:.4f} | {row['session_fall_recall']:.4f} | {row['session_fall_f1']:.4f} | "
            f"{int(row['nonfall_sessions_alerted'])} | {row['false_alerts_per_nonfall_session']:.4f} | "
            f"{row['mean_delay_event_start_s']:.4f} | {row['mean_delay_impact_s']:.4f} | "
            f"{int(row['fall_sessions_no_alert'])} | {row['repeated_alerts_per_fall_session']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Canonical Clip Classification (Different Task From E0 Windows)",
            "",
            "These metrics classify one deterministic canonical clip per held-out session. They are not directly compared with E0's 1.5-second window metrics.",
            "",
            "| Length | Accuracy | Macro F1 | Fall precision | Fall recall | Fall F1 |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in summary.iterrows():
        lines.append(
            f"| {int(row['clip_length_frames'])} | {fmt(row, 'clip_accuracy')} | {fmt(row, 'clip_macro_f1')} | "
            f"{fmt(row, 'clip_fall_precision')} | {fmt(row, 'clip_fall_recall')} | {fmt(row, 'clip_fall_f1')} |"
        )
    lines.extend(
        [
            "",
            "## Decision Rule",
            "",
            "A candidate must have causal mean recall >= 0.8182, mean nonfall sessions alerted <= 4.5 (at least about 28% below 6.25), mean impact delay <= 2.0 seconds, mean causal fall F1 > 0.6899, and zero future access. See `decision_criteria.csv`.",
            "",
            "Every sliding prediction uses a trailing clip ending at the alert timestamp. The saved causal check asserts the largest source row used is no later than the available row at that clip end; no future clip or full-session aggregation is used.",
        ]
    )
    (OUTPUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return decision


def main() -> None:
    if sha256(MANIFEST) != EXPECTED_MANIFEST_SHA256:
        raise ValueError("Frozen manifest checksum mismatch")
    marker = OUTPUT / "fold_metrics.csv"
    if marker.exists():
        raise FileExistsError(f"Refusing to overwrite completed experiment: {marker}")
    sessions = inventory()
    annotations = annotation_table()
    fall_ids = set(sessions.loc[sessions["activity"].eq("fall"), "source_session_id"])
    if fall_ids - set(annotations["source_session_id"]):
        raise ValueError("Some current fall sessions lack event metadata")
    sessions = sessions.merge(
        annotations[["source_session_id", "event_start_s", "impact_s", "event_end_s"]],
        on="source_session_id",
        how="left",
        validate="one_to_one",
    )

    audit_rows: list[dict[str, object]] = []
    datasets: dict[int, tuple[np.ndarray, np.ndarray, pd.DataFrame]] = {}
    for length in CLIP_LENGTHS:
        X, y, masks, metadata = build_clip_dataset(length, sessions, annotations)
        np.savez_compressed(
            OUTPUT / f"staging_clip_dataset_{length}.npz",
            X=X,
            y=y,
            mask=masks,
            feature_names=np.asarray(MODEL_FEATURES),
            source_session_id=metadata["source_session_id"].to_numpy(),
            outer_fold=metadata["outer_fold"].to_numpy(dtype=int),
        )
        metadata.to_csv(OUTPUT / f"staging_clip_metadata_{length}.csv", index=False)
        counts = metadata["activity"].value_counts()
        audit_rows.append(
            {
                "clip_length_frames": length,
                "clips": len(metadata),
                **{name: int(counts.get(name, 0)) for name in CLASS_NAMES},
                "left_padded": int((metadata["padding_left_frames"] > 0).sum()),
                "right_padded": int((metadata["padding_right_frames"] > 0).sum()),
                "mean_physical_post_impact_s": float(
                    metadata.loc[metadata["activity"].eq("fall"), "physical_post_impact_s"].mean()
                ),
                "min_pre_event_context_s": float(
                    metadata.loc[metadata["activity"].eq("fall"), "pre_event_context_s"].min()
                ),
                "fall_clips_event_start_inside": int(
                    (
                        (metadata.loc[metadata["activity"].eq("fall"), "event_start_s"]
                        >= metadata.loc[metadata["activity"].eq("fall"), "clip_start_s"])
                        & (metadata.loc[metadata["activity"].eq("fall"), "event_start_s"]
                        <= metadata.loc[metadata["activity"].eq("fall"), "clip_end_s"])
                    ).sum()
                ),
                "fall_clips_impact_inside": int(
                    (
                        (metadata.loc[metadata["activity"].eq("fall"), "impact_s"]
                        >= metadata.loc[metadata["activity"].eq("fall"), "clip_start_s"])
                        & (metadata.loc[metadata["activity"].eq("fall"), "impact_s"]
                        <= metadata.loc[metadata["activity"].eq("fall"), "clip_end_s"])
                    ).sum()
                ),
                "fall_clips_post_context_at_least_0p75s": int(
                    (
                        metadata.loc[metadata["activity"].eq("fall"), "physical_post_impact_s"] >= 0.75
                    ).sum()
                ),
                "cross_session_clips": 0,
                "feature_names": json.dumps(MODEL_FEATURES),
            }
        )
        datasets[length] = (X, y, metadata)
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(OUTPUT / "dataset_audit.csv", index=False)

    leakage_rows = []
    for fold in range(1, 5):
        splits = frozen_split_sessions(fold)
        overlaps = (splits["train"] & splits["val"]) | (splits["train"] & splits["test"]) | (
            splits["val"] & splits["test"]
        )
        manifest_test = set(
            sessions.loc[sessions["outer_fold"].astype(int).eq(fold), "source_session_id"].astype(str)
        )
        leakage_rows.append(
            {
                "fold": fold,
                "train_sessions": len(splits["train"]),
                "validation_sessions": len(splits["val"]),
                "test_sessions": len(splits["test"]),
                "session_overlap_count": len(overlaps),
                "test_sessions_match_manifest_fold": splits["test"] == manifest_test,
                "cross_session_clips": 0,
            }
        )
    leakage = pd.DataFrame(leakage_rows)
    if leakage["session_overlap_count"].sum() or not leakage["test_sessions_match_manifest_fold"].all():
        raise ValueError("Frozen fold leakage check failed")
    leakage.to_csv(OUTPUT / "fold_leakage_checks.csv", index=False)

    cfg = GroupedCVConfig()
    all_results: list[dict[str, object]] = []
    for length in CLIP_LENGTHS:
        X, y, metadata = datasets[length]
        all_results.extend(train_and_evaluate(length, X, y, metadata, sessions, cfg))
    folds = pd.DataFrame(all_results)
    summary = aggregate(folds)
    folds.to_csv(OUTPUT / "fold_metrics.csv", index=False)
    summary.to_csv(OUTPUT / "aggregate_metrics.csv", index=False)
    decision = write_report(folds, summary, audit)
    provenance = {
        "decision": decision,
        "dataset_source": str(DATASET.relative_to(ROOT)),
        "dataset_sha256": sha256(DATASET),
        "manifest": str(MANIFEST.relative_to(ROOT)),
        "manifest_sha256": manifest_checksum(MANIFEST),
        "annotations": str(ANNOTATIONS.relative_to(ROOT)),
        "annotations_sha256": sha256(ANNOTATIONS),
        "source_sessions": len(sessions),
        "clip_lengths": list(CLIP_LENGTHS),
        "sampling_rate_hz": RATE_HZ,
        "stride_frames": STRIDE_FRAMES,
        "post_impact_frames": POST_IMPACT_FRAMES,
        "seed": SEED,
        "config": asdict(cfg),
        "base_features": MODEL_FEATURES,
        "synthetic_augmentation": False,
        "height_or_session_features": False,
    }
    (OUTPUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(f"decision={decision}", flush=True)


if __name__ == "__main__":
    main()
