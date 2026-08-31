from __future__ import annotations

import argparse
import hashlib
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from radar_pipeline.common import CLASS_LABELS, CLASS_NAMES, ROOT
from radar_pipeline.evaluate_grouped_cv import GroupedCVConfig, _session_activity, grouped_validation_split, source_session_id


@dataclass(frozen=True)
class ManifestConfig:
    folds: int = 4
    seed: int = 42
    val_folds: int = 6

    @property
    def grouped_cv(self) -> GroupedCVConfig:
        return GroupedCVConfig(folds=self.folds, seed=self.seed, val_folds=self.val_folds)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def infer_protocol_date(dataset_path: Path, output_path: Path) -> str:
    text = f"{output_path} {dataset_path}"
    match = re.search(r"(20\d{6})", text)
    if match:
        return match.group(1)
    return datetime.now().strftime("%Y%m%d")


def split_protocol(protocol_date: str, seed: int, folds: int) -> str:
    return f"sgkf_grouped_{protocol_date}_seed{seed}_k{folds}"


def _required_npz_keys() -> list[str]:
    keys = ["feature_names", "label_names"]
    per_split = [
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
    for split in ("train", "val", "test"):
        keys.extend(f"{key}_{split}" for key in per_split)
    return keys


def load_manifest_payload(dataset_path: Path) -> dict[str, np.ndarray]:
    data = np.load(dataset_path, allow_pickle=True)
    missing = [key for key in _required_npz_keys() if key not in data.files]
    if missing:
        raise ValueError(f"Dataset is missing required grouped-manifest metadata keys: {missing[:8]}")

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

    source_csv = payload["source_csv"].astype(str)
    if any(not value or value.lower() == "nan" for value in source_csv):
        raise ValueError("Dataset contains missing source_csv values; cannot derive source_session_id")
    payload["source_session_id"] = np.asarray([source_session_id(x) for x in source_csv], dtype=str)
    if any(not value or value.lower() == "nan" for value in payload["source_session_id"].astype(str)):
        raise ValueError("Dataset contains blank source_session_id values after Path(source_csv).stem extraction")
    return payload


def _session_table(payload: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    groups = payload["source_session_id"].astype(str)
    y = payload["y"].astype(int)
    invalid = sorted(set(y.tolist()) - set(CLASS_LABELS))
    if invalid:
        raise ValueError(f"Dataset contains labels outside {CLASS_LABELS}: {invalid}")

    for sid in sorted(np.unique(groups).tolist()):
        idx = np.flatnonzero(groups == sid)
        source_activities = payload["source_activity"][idx].astype(str)
        activity_values = sorted(set(v for v in source_activities if v))
        if len(activity_values) > 1:
            raise ValueError(f"source_session_id {sid} has inconsistent source_activity values: {activity_values}")
        canonical = _session_activity(source_activities, payload["source_csv"][idx])
        if canonical not in CLASS_NAMES:
            raise ValueError(f"source_session_id {sid} has invalid canonical class: {canonical}")
        class_counts = {name: int((y[idx] == label).sum()) for label, name in zip(CLASS_LABELS, CLASS_NAMES)}
        source_csvs = sorted(set(payload["source_csv"][idx].astype(str).tolist()))
        rows.append(
            {
                "source_session_id": sid,
                "canonical_label": canonical,
                "canonical_label_id": int(CLASS_NAMES.index(canonical)),
                "source_activity": canonical,
                "source_csv": source_csvs[0],
                "source_csv_count": len(source_csvs),
                "session_window_count": int(len(idx)),
                **{f"{name}_windows": class_counts[name] for name in CLASS_NAMES},
            }
        )
    table = pd.DataFrame(rows)
    if table["source_session_id"].duplicated().any():
        dupes = table.loc[table["source_session_id"].duplicated(), "source_session_id"].tolist()
        raise ValueError(f"Duplicate source_session_id rows after aggregation: {dupes[:5]}")
    inconsistent_csv = table[table["source_csv_count"] > 1]
    if len(inconsistent_csv):
        raise ValueError(
            "Some source_session_id values map to multiple source_csv paths: "
            f"{inconsistent_csv['source_session_id'].head().tolist()}"
        )
    return table


def validate_class_support(session_table: pd.DataFrame, folds: int) -> None:
    counts = session_table.groupby("canonical_label")["source_session_id"].nunique()
    missing = sorted(set(CLASS_NAMES) - set(counts.index.astype(str)))
    if missing:
        raise ValueError(f"Dataset has no source sessions for classes: {missing}")
    weak = {name: int(counts.get(name, 0)) for name in CLASS_NAMES if int(counts.get(name, 0)) < folds}
    if weak:
        raise ValueError(f"Requested folds={folds} but class source-session support is too small: {weak}")


def create_manifest_dataframe(payload: dict[str, np.ndarray], cfg: ManifestConfig, protocol: str) -> pd.DataFrame:
    session_table = _session_table(payload)
    validate_class_support(session_table, cfg.folds)
    groups = payload["source_session_id"].astype(str)
    y = payload["y"].astype(int)
    splitter = StratifiedGroupKFold(n_splits=cfg.folds, shuffle=True, random_state=cfg.seed)
    fold_rows = []
    for fold_idx, (_, test_idx) in enumerate(splitter.split(np.arange(len(y)), y, groups), start=1):
        for sid in sorted(np.unique(groups[test_idx]).tolist()):
            fold_rows.append({"source_session_id": sid, "outer_fold": int(fold_idx)})
    folds = pd.DataFrame(fold_rows)
    manifest = session_table.merge(folds, on="source_session_id", how="left")
    if manifest["outer_fold"].isna().any():
        missing = manifest.loc[manifest["outer_fold"].isna(), "source_session_id"].tolist()
        raise ValueError(f"Fold assignment missing source sessions: {missing[:5]}")
    manifest["outer_fold"] = manifest["outer_fold"].astype(int)
    manifest["split_protocol"] = protocol
    ordered_cols = [
        "source_session_id",
        "canonical_label",
        "canonical_label_id",
        "source_activity",
        "outer_fold",
        "split_protocol",
        "source_csv",
        "session_window_count",
        *[f"{name}_windows" for name in CLASS_NAMES],
    ]
    return manifest[ordered_cols].sort_values("source_session_id").reset_index(drop=True)


def verify_zero_leakage(payload: dict[str, np.ndarray], manifest: pd.DataFrame, cfg: ManifestConfig) -> pd.DataFrame:
    groups = payload["source_session_id"].astype(str)
    y = payload["y"].astype(int)
    fold_map = dict(zip(manifest["source_session_id"].astype(str), manifest["outer_fold"].astype(int)))
    all_sessions = set(fold_map)
    rows = []
    for fold in range(1, cfg.folds + 1):
        test_sessions = {sid for sid, assigned in fold_map.items() if assigned == fold}
        train_pool_sessions = all_sessions - test_sessions
        all_idx = np.arange(len(y))
        train_pool_idx = all_idx[np.asarray([sid in train_pool_sessions for sid in groups])]
        test_idx = all_idx[np.asarray([sid in test_sessions for sid in groups])]
        train_idx, val_idx = grouped_validation_split(train_pool_idx, y, groups, cfg.grouped_cv)
        train_sessions = set(groups[train_idx].tolist())
        val_sessions = set(groups[val_idx].tolist())
        actual_test_sessions = set(groups[test_idx].tolist())
        train_val_overlap = train_sessions & val_sessions
        train_test_overlap = train_sessions & actual_test_sessions
        val_test_overlap = val_sessions & actual_test_sessions
        if train_val_overlap or train_test_overlap or val_test_overlap:
            raise ValueError(
                f"Source-session leakage in fold {fold}: "
                f"train_val={sorted(train_val_overlap)[:5]} "
                f"train_test={sorted(train_test_overlap)[:5]} "
                f"val_test={sorted(val_test_overlap)[:5]}"
            )
        present = set(y[test_idx].tolist())
        if present != set(CLASS_LABELS):
            raise ValueError(f"Fold {fold} lacks effective window classes: {sorted(set(CLASS_LABELS) - present)}")
        rows.append(
            {
                "outer_fold": fold,
                "train_sessions": len(train_sessions),
                "val_sessions": len(val_sessions),
                "test_sessions": len(actual_test_sessions),
                "train_windows": int(len(train_idx)),
                "val_windows": int(len(val_idx)),
                "test_windows": int(len(test_idx)),
                "train_val_leakage": 0,
                "train_test_leakage": 0,
                "val_test_leakage": 0,
            }
        )
    return pd.DataFrame(rows)


def class_count_table(payload: dict[str, np.ndarray], manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups = payload["source_session_id"].astype(str)
    y = payload["y"].astype(int)
    for label, name in zip(CLASS_LABELS, CLASS_NAMES):
        label_sessions = manifest.loc[manifest["canonical_label"] == name, "source_session_id"]
        rows.append(
            {
                "class": name,
                "canonical_session_count": int(label_sessions.nunique()),
                "effective_window_count": int((y == label).sum()),
            }
        )
    return pd.DataFrame(rows)


def fold_count_table(payload: dict[str, np.ndarray], manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups = payload["source_session_id"].astype(str)
    y = payload["y"].astype(int)
    fold_map = dict(zip(manifest["source_session_id"].astype(str), manifest["outer_fold"].astype(int)))
    for fold in range(1, int(manifest["outer_fold"].max()) + 1):
        fold_sessions = {sid for sid, assigned in fold_map.items() if assigned == fold}
        fold_idx = np.asarray([sid in fold_sessions for sid in groups])
        row = {
            "outer_fold": fold,
            "source_sessions": int(len(fold_sessions)),
            "windows": int(fold_idx.sum()),
        }
        for label, name in zip(CLASS_LABELS, CLASS_NAMES):
            class_sessions = manifest[
                (manifest["outer_fold"] == fold) & (manifest["canonical_label"] == name)
            ]["source_session_id"]
            row[f"{name}_sessions"] = int(class_sessions.nunique())
            row[f"{name}_windows"] = int((fold_idx & (y == label)).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def write_readme(
    readme_path: Path,
    dataset_path: Path,
    manifest_path: Path,
    payload: dict[str, np.ndarray],
    manifest: pd.DataFrame,
    leakage: pd.DataFrame,
    cfg: ManifestConfig,
    protocol: str,
) -> None:
    class_counts = class_count_table(payload, manifest)
    fold_counts = fold_count_table(payload, manifest)
    dataset_hash = sha256_file(dataset_path)
    manifest_hash = sha256_file(manifest_path)
    timestamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    lines = [
        "# Date-Versioned SGKF4 Source-Session Manifest",
        "",
        f"- Dataset path: `{dataset_path}`",
        f"- Dataset SHA-256: `{dataset_hash}`",
        f"- Manifest path: `{manifest_path}`",
        f"- Manifest SHA-256: `{manifest_hash}`",
        f"- Generation timestamp: `{timestamp}`",
        f"- Folds: {cfg.folds}",
        f"- Seed: {cfg.seed}",
        f"- Protocol name: `{protocol}`",
        "- Group key: `source_session_id / Path(source_csv).stem`",
        "- Splitter: `StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)`",
        "",
        "This manifest is frozen for only this exact dataset session inventory and must not be overwritten.",
        "",
        "## Class-Level Counts",
        "",
        "| Class | Canonical source sessions | Effective windows |",
        "|---|---:|---:|",
    ]
    for _, row in class_counts.iterrows():
        lines.append(f"| {row['class']} | {int(row['canonical_session_count'])} | {int(row['effective_window_count'])} |")
    lines.extend(
        [
            "",
            "## Per-Fold Counts",
            "",
            "| Fold | Source sessions | Windows | Walking sessions/windows | Standing sessions/windows | Sitting sessions/windows | Fall sessions/windows |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in fold_counts.iterrows():
        lines.append(
            f"| {int(row['outer_fold'])} | {int(row['source_sessions'])} | {int(row['windows'])} | "
            f"{int(row['walking_sessions'])}/{int(row['walking_windows'])} | "
            f"{int(row['standing_sessions'])}/{int(row['standing_windows'])} | "
            f"{int(row['sitting_sessions'])}/{int(row['sitting_windows'])} | "
            f"{int(row['fall_sessions'])}/{int(row['fall_windows'])} |"
        )
    lines.extend(
        [
            "",
            "## Zero-Leakage Verification",
            "",
            "| Fold | Train sessions | Val sessions | Test sessions | Train windows | Val windows | Test windows | Train/val leaks | Train/test leaks | Val/test leaks |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in leakage.iterrows():
        lines.append(
            f"| {int(row['outer_fold'])} | {int(row['train_sessions'])} | {int(row['val_sessions'])} | "
            f"{int(row['test_sessions'])} | {int(row['train_windows'])} | {int(row['val_windows'])} | "
            f"{int(row['test_windows'])} | {int(row['train_val_leakage'])} | "
            f"{int(row['train_test_leakage'])} | {int(row['val_test_leakage'])} |"
        )
    readme_path.parent.mkdir(parents=True, exist_ok=True)
    readme_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def create_grouped_manifest(
    dataset_path: Path,
    output_path: Path,
    readme_path: Path,
    cfg: ManifestConfig,
    overwrite: bool = False,
) -> pd.DataFrame:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Manifest already exists: {output_path}. Pass --overwrite to replace it.")
    if readme_path.exists() and not overwrite:
        raise FileExistsError(f"README already exists: {readme_path}. Pass --overwrite to replace it.")
    payload = load_manifest_payload(dataset_path)
    protocol = split_protocol(infer_protocol_date(dataset_path, output_path), cfg.seed, cfg.folds)
    manifest = create_manifest_dataframe(payload, cfg, protocol)
    leakage = verify_zero_leakage(payload, manifest, cfg)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_path, index=False)
    write_readme(readme_path, dataset_path, output_path, payload, manifest, leakage, cfg, protocol)
    return manifest


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Create a date-versioned frozen SGKF4 source-session manifest.")
    parser.add_argument("--data", required=True, help="Path to radar_dataset.npz")
    parser.add_argument("--output", required=True, help="Output manifest CSV path")
    parser.add_argument("--readme", required=True, help="Output README path")
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    manifest = create_grouped_manifest(
        dataset_path=Path(args.data),
        output_path=Path(args.output),
        readme_path=Path(args.readme),
        cfg=ManifestConfig(folds=args.folds, seed=args.seed),
        overwrite=args.overwrite,
    )
    print(f"saved_manifest={Path(args.output)}")
    print(f"saved_readme={Path(args.readme)}")
    print(f"source_sessions={len(manifest)}")


if __name__ == "__main__":
    main()
