from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from radar_pipeline.common import CLASS_NAMES, ROOT, ensure_dirs, load_config, load_json, write_json


def _event_enabled(cfg: dict) -> bool:
    return bool((cfg.get("windowing", {}).get("event_aware", {}) or {}).get("enabled", False))


def _assert_event_staging_output(output_dir: Path, cfg: dict) -> None:
    if not _event_enabled(cfg):
        return
    if output_dir.resolve() == (ROOT / "data/final_dataset").resolve():
        raise ValueError(
            "Event-aware dataset building is staging-only and cannot write to data/final_dataset. "
            "Use data/final_dataset_auto_event_staging or data/final_dataset_manual_event_staging."
        )


def _split_indices(labels: np.ndarray, cfg: dict, sessions: np.ndarray | None = None) -> dict[str, np.ndarray]:
    split = cfg["split"]
    rng = np.random.default_rng(int(split.get("random_seed", 42)))
    train_p, val_p = float(split["train"]), float(split["val"])
    result = {"train": [], "val": [], "test": []}
    if sessions is not None:
        session_values = np.unique(sessions)
        multilabel_sessions = [session for session in session_values if len(np.unique(labels[sessions == session])) > 1]
        if multilabel_sessions:
            shuffled = np.asarray(session_values)
            rng.shuffle(shuffled)
            n_sessions = len(shuffled)
            if n_sessions == 1:
                result["train"].extend(np.arange(len(labels)))
            else:
                n_train_sessions = max(1, int(round(n_sessions * train_p)))
                n_val_sessions = max(0 if n_sessions < 3 else 1, int(round(n_sessions * val_p)))
                if n_train_sessions + n_val_sessions >= n_sessions:
                    n_val_sessions = max(0, n_sessions - n_train_sessions - 1)
                train_sessions = set(shuffled[:n_train_sessions])
                val_sessions = set(shuffled[n_train_sessions : n_train_sessions + n_val_sessions])
                test_sessions = set(shuffled[n_train_sessions + n_val_sessions :])
                result["train"].extend(np.where(np.isin(sessions, list(train_sessions)))[0])
                result["val"].extend(np.where(np.isin(sessions, list(val_sessions)))[0])
                result["test"].extend(np.where(np.isin(sessions, list(test_sessions)))[0])
            return {k: np.asarray(sorted(v), dtype=np.int64) for k, v in result.items()}
    for label in np.unique(labels):
        idx = np.where(labels == label)[0]
        if sessions is not None:
            session_values = np.unique(sessions[idx])
            rng.shuffle(session_values)
            n_sessions = len(session_values)
            if n_sessions == 1:
                result["train"].extend(idx)
                continue
            n_train_sessions = max(1, int(round(n_sessions * train_p)))
            n_val_sessions = max(0 if n_sessions < 3 else 1, int(round(n_sessions * val_p)))
            if n_train_sessions + n_val_sessions >= n_sessions:
                n_val_sessions = max(0, n_sessions - n_train_sessions - 1)
            train_sessions = set(session_values[:n_train_sessions])
            val_sessions = set(session_values[n_train_sessions : n_train_sessions + n_val_sessions])
            test_sessions = set(session_values[n_train_sessions + n_val_sessions :])
            result["train"].extend(idx[np.isin(sessions[idx], list(train_sessions))])
            result["val"].extend(idx[np.isin(sessions[idx], list(val_sessions))])
            result["test"].extend(idx[np.isin(sessions[idx], list(test_sessions))])
            continue
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(round(n * train_p))
        n_val = int(round(n * val_p))
        result["train"].extend(idx[:n_train])
        result["val"].extend(idx[n_train : n_train + n_val])
        result["test"].extend(idx[n_train + n_val :])
    return {k: np.asarray(sorted(v), dtype=np.int64) for k, v in result.items()}


def build_dataset(input_dir: Path, output_dir: Path, cfg: dict) -> dict:
    ensure_dirs()
    _assert_event_staging_output(output_dir, cfg)
    output_dir.mkdir(parents=True, exist_ok=True)
    cleaning_log = load_json(input_dir.parent / "cleaning_log.json", {})
    excluded_sessions = {
        session_id
        for activity in cleaning_log.get("activities", {}).values()
        for session_id, session in activity.get("sessions", {}).items()
        if session.get("excluded")
    }
    arrays = []
    labels = []
    activities = []
    source_activities = []
    sessions = []
    start_frames = []
    end_frames = []
    source_csvs = []
    label_sources = []
    event_phases = []
    include_flags = []
    overlap_seconds = []
    overlap_fraction = []
    window_start_s = []
    window_end_s = []
    exclude_reasons = []
    annotation_confidences = []
    quality_flags = []
    method_versions = []
    excluded_windows_filtered = 0
    feature_names = None
    for path in sorted(input_dir.glob("*_windows.npz")):
        data = np.load(path, allow_pickle=True)
        X = data["X"]
        y = data["y"]
        if X.size == 0:
            continue
        keep = np.ones(len(y), dtype=bool)
        if "session_id" in data and excluded_sessions:
            session_values = np.asarray([str(x) for x in data["session_id"]])
            keep &= ~np.isin(session_values, list(excluded_sessions))
        if "include_in_training" in data:
            keep &= np.asarray(data["include_in_training"], dtype=bool)
        keep &= np.isin(y, np.arange(len(CLASS_NAMES)))
        excluded_windows_filtered += int(len(keep) - keep.sum())
        X = X[keep]
        y = y[keep]
        if X.size == 0:
            continue
        names = [str(x) for x in data["feature_names"]]
        if feature_names is None:
            feature_names = names
        elif names != feature_names:
            raise ValueError(f"Feature mismatch in {path}: {names} != {feature_names}")
        arrays.append(X)
        labels.append(y)
        source_activity = str(data["activity"])
        source_activities.extend([source_activity] * len(y))
        activities.extend([CLASS_NAMES[int(label)] for label in y])
        if "session_id" in data:
            sessions.extend([str(x) for x in data["session_id"][keep]])
        else:
            sessions.extend([f"{source_activity}__unknown"] * len(y))
        if "start_frame" in data:
            start_frames.extend([int(x) for x in data["start_frame"][keep]])
            end_frames.extend([int(x) for x in data["end_frame"][keep]])
        else:
            start_frames.extend([-1] * len(y))
            end_frames.extend([-1] * len(y))
        if "source_csv" in data:
            source_csvs.extend([str(x) for x in data["source_csv"][keep]])
        else:
            source_csvs.extend([""] * len(y))
        if "label_source" in data:
            label_sources.extend([str(x) for x in data["label_source"][keep]])
        else:
            label_sources.extend(["legacy_folder"] * len(y))
        if "event_phase" in data:
            event_phases.extend([str(x) for x in data["event_phase"][keep]])
        else:
            event_phases.extend(["legacy"] * len(y))
        if "include_in_training" in data:
            include_flags.extend([bool(x) for x in data["include_in_training"][keep]])
        else:
            include_flags.extend([True] * len(y))
        if "overlap_seconds" in data:
            overlap_seconds.extend([float(x) for x in data["overlap_seconds"][keep]])
            overlap_fraction.extend([float(x) for x in data["overlap_fraction"][keep]])
            window_start_s.extend([float(x) for x in data["window_start_s"][keep]])
            window_end_s.extend([float(x) for x in data["window_end_s"][keep]])
        else:
            overlap_seconds.extend([float("nan")] * len(y))
            overlap_fraction.extend([float("nan")] * len(y))
            window_start_s.extend([float("nan")] * len(y))
            window_end_s.extend([float("nan")] * len(y))
        if "exclude_reason" in data:
            exclude_reasons.extend([str(x) for x in data["exclude_reason"][keep]])
        else:
            exclude_reasons.extend([""] * len(y))
        if "annotation_confidence" in data:
            annotation_confidences.extend([str(x) for x in data["annotation_confidence"][keep]])
        else:
            annotation_confidences.extend([""] * len(y))
        if "quality_flags" in data:
            quality_flags.extend([str(x) for x in data["quality_flags"][keep]])
        else:
            quality_flags.extend([""] * len(y))
        if "method_version" in data:
            method_versions.extend([str(x) for x in data["method_version"][keep]])
        else:
            method_versions.extend([""] * len(y))
    if not arrays:
        raise FileNotFoundError(f"No non-empty window files found in {input_dir}")

    X_all = np.concatenate(arrays, axis=0)
    y_all = np.concatenate(labels, axis=0)
    activities_arr = np.asarray(activities)
    source_activities_arr = np.asarray(source_activities)
    sessions_arr = np.asarray(sessions)
    start_arr = np.asarray(start_frames, dtype=np.int64)
    end_arr = np.asarray(end_frames, dtype=np.int64)
    source_arr = np.asarray(source_csvs)
    label_source_arr = np.asarray(label_sources)
    event_phase_arr = np.asarray(event_phases)
    include_arr = np.asarray(include_flags, dtype=bool)
    overlap_seconds_arr = np.asarray(overlap_seconds, dtype=np.float32)
    overlap_fraction_arr = np.asarray(overlap_fraction, dtype=np.float32)
    window_start_s_arr = np.asarray(window_start_s, dtype=np.float32)
    window_end_s_arr = np.asarray(window_end_s, dtype=np.float32)
    exclude_reason_arr = np.asarray(exclude_reasons)
    annotation_confidence_arr = np.asarray(annotation_confidences)
    quality_flags_arr = np.asarray(quality_flags)
    method_version_arr = np.asarray(method_versions)
    splits = _split_indices(y_all, cfg, sessions_arr)
    method = cfg["cleaning"].get("normalization", "zscore")
    train_x = X_all[splits["train"]]
    if method == "minmax":
        mins = train_x.reshape(-1, train_x.shape[-1]).min(axis=0)
        maxs = train_x.reshape(-1, train_x.shape[-1]).max(axis=0)
        denom = np.where((maxs - mins) == 0, 1.0, maxs - mins)
        X_norm = (X_all - mins) / denom
        normalizer = {
            feature_names[i]: {"min": float(mins[i]), "max": float(maxs[i])}
            for i in range(len(feature_names))
        }
    else:
        means = train_x.reshape(-1, train_x.shape[-1]).mean(axis=0)
        stds = train_x.reshape(-1, train_x.shape[-1]).std(axis=0)
        stds = np.where(stds == 0, 1.0, stds)
        X_norm = (X_all - means) / stds
        normalizer = {
            feature_names[i]: {"mean": float(means[i]), "std": float(stds[i])}
            for i in range(len(feature_names))
        }
    payload = {"feature_names": np.asarray(feature_names), "label_names": np.asarray(CLASS_NAMES)}
    for name, idx in splits.items():
        payload[f"X_{name}"] = X_norm[idx]
        payload[f"y_{name}"] = y_all[idx]
        payload[f"activity_{name}"] = activities_arr[idx]
        payload[f"source_activity_{name}"] = source_activities_arr[idx]
        payload[f"session_id_{name}"] = sessions_arr[idx]
        payload[f"start_frame_{name}"] = start_arr[idx]
        payload[f"end_frame_{name}"] = end_arr[idx]
        payload[f"source_csv_{name}"] = source_arr[idx]
        payload[f"label_source_{name}"] = label_source_arr[idx]
        payload[f"event_phase_{name}"] = event_phase_arr[idx]
        payload[f"include_in_training_{name}"] = include_arr[idx]
        payload[f"overlap_seconds_{name}"] = overlap_seconds_arr[idx]
        payload[f"overlap_fraction_{name}"] = overlap_fraction_arr[idx]
        payload[f"window_start_s_{name}"] = window_start_s_arr[idx]
        payload[f"window_end_s_{name}"] = window_end_s_arr[idx]
        payload[f"exclude_reason_{name}"] = exclude_reason_arr[idx]
        payload[f"annotation_confidence_{name}"] = annotation_confidence_arr[idx]
        payload[f"quality_flags_{name}"] = quality_flags_arr[idx]
        payload[f"method_version_{name}"] = method_version_arr[idx]
    np.savez_compressed(output_dir / "radar_dataset.npz", **payload)

    rows = []
    for split_name, idx in splits.items():
        for i in idx:
            rows.append(
                {
                    "index": int(i),
                    "split": split_name,
                    "label": int(y_all[i]),
                    "activity": activities_arr[i],
                    "source_activity": source_activities_arr[i],
                    "session_id": sessions_arr[i],
                    "source_csv": source_arr[i],
                    "start_frame": int(start_arr[i]),
                    "end_frame": int(end_arr[i]),
                    "label_source": label_source_arr[i],
                    "event_phase": event_phase_arr[i],
                    "include_in_training": bool(include_arr[i]),
                    "overlap_seconds": float(overlap_seconds_arr[i]),
                    "overlap_fraction": float(overlap_fraction_arr[i]),
                    "window_start_s": float(window_start_s_arr[i]),
                    "window_end_s": float(window_end_s_arr[i]),
                    "exclude_reason": exclude_reason_arr[i],
                    "annotation_confidence": annotation_confidence_arr[i],
                    "quality_flags": quality_flags_arr[i],
                    "method_version": method_version_arr[i],
                }
            )
    pd.DataFrame(rows).to_csv(output_dir / "dataset_index.csv", index=False)

    flat = X_norm.reshape(-1, X_norm.shape[-1])
    stats = {
        "total_windows": int(len(X_all)),
        "window_shape": list(X_all.shape[1:]),
        "feature_names": feature_names,
        "class_balance": pd.Series(activities_arr).value_counts().to_dict(),
        "source_activity_balance": pd.Series(source_activities_arr).value_counts().to_dict(),
        "event_phase_balance": pd.Series(event_phase_arr).value_counts().to_dict(),
        "label_source_balance": pd.Series(label_source_arr).value_counts().to_dict(),
        "annotation_confidence_balance": pd.Series(annotation_confidence_arr).value_counts().to_dict(),
        "excluded_windows_filtered": int(excluded_windows_filtered),
        "session_balance": pd.DataFrame({"activity": activities_arr, "session_id": sessions_arr})
        .drop_duplicates()
        .groupby("activity")
        .size()
        .to_dict(),
        "source_csvs": sorted(pd.unique(source_arr).tolist()),
        "splits": {name: int(len(idx)) for name, idx in splits.items()},
        "split_sessions": {name: int(len(np.unique(sessions_arr[idx]))) for name, idx in splits.items()},
        "normalization": {"method": method, "fit_split": "train", "params": normalizer},
        "feature_stats": {
            feature_names[i]: {
                "mean": float(np.mean(flat[:, i])),
                "std": float(np.std(flat[:, i])),
                "min": float(np.min(flat[:, i])),
                "max": float(np.max(flat[:, i])),
            }
            for i in range(len(feature_names))
        },
    }
    write_json(output_dir / "dataset_summary.json", stats)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=str(ROOT / "data/windowed"))
    parser.add_argument("--output-dir", default=str(ROOT / "data/final_dataset"))
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    args = parser.parse_args()
    stats = build_dataset(Path(args.input_dir), Path(args.output_dir), load_config(args.config))
    print(f"Saved dataset with {stats['total_windows']} windows.")
    print(stats["class_balance"])


if __name__ == "__main__":
    main()
