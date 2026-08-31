"""Create the versioned real-record host-demo golden corpus without overwriting it."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from radar_pipeline.live_inference import BASE_FEATURES, LiveRadarRecord, load_demo_binding, sha256_file
from radar_pipeline.train_model import CNNLSTM, augment_window_features


def _records(path: Path) -> list[LiveRadarRecord]:
    required = ("x", "y", "z", "dop_idx", "range_m", "azimuth_deg", "elevation_deg")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) < 60 or any(not set(required).issubset(row) for row in rows[:60]):
        raise ValueError(f"Golden source does not contain 60 complete base records: {path}")
    return [
        LiveRadarRecord(
            timestamp_s=index / 20.0,
            frame_id=index + 1,
            x=float(row["x"]), y=float(row["y"]), z=float(row["z"]), dop_idx=int(float(row["dop_idx"])),
            range_m=float(row["range_m"]), azimuth_deg=float(row["azimuth_deg"]), elevation_deg=float(row["elevation_deg"]),
        )
        for index, row in enumerate(rows[:60])
    ]


def build(manifest: Path, source: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing golden corpus: {output}")
    binding = load_demo_binding(manifest)
    records = _records(source)
    base = np.stack([record.base_values() for record in records]).astype(np.float32)
    normalized = ((base - binding.mean) / binding.std)[np.newaxis, :, :].astype(np.float32)
    # The golden reference deliberately uses the frozen training implementation,
    # not CausalHostInference, so the parity test can catch drift in host code.
    reference_data = {
        "feature_names": np.asarray(BASE_FEATURES),
        "X_train": normalized,
        "X_val": normalized.copy(),
        "X_test": normalized.copy(),
    }
    augmented, feature_names = augment_window_features(reference_data)
    if feature_names != list(BASE_FEATURES) + [
        "xyz_delta_mag", "x_roll_std", "y_roll_std", "z_roll_std", "range_roll_std", "range_centered",
    ]:
        raise RuntimeError("Training feature contract changed while building golden corpus")
    tensor = augmented["X_train"].astype(np.float32)
    checkpoint = torch.load(binding.checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["train_config"]
    reference_model = CNNLSTM(13, 4, float(config["dropout_input"]), float(config["dropout_hidden"]))
    reference_model.load_state_dict(checkpoint["model_state_dict"])
    reference_model.eval()
    with torch.no_grad():
        probabilities = torch.softmax(reference_model(torch.from_numpy(tensor)), dim=1).numpy()[0].astype(np.float32)
    predicted_class = ("walking", "standing", "sitting", "fall")[int(np.argmax(probabilities))]
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        base_records=base,
        expected_tensor=tensor,
        expected_probabilities=probabilities,
        expected_class=np.asarray(predicted_class),
        source_csv=np.asarray(str(source)),
        source_csv_sha256=np.asarray(sha256_file(source)),
        checkpoint_sha256=np.asarray(sha256_file(binding.checkpoint_path)),
        normalization_sha256=np.asarray(sha256_file(binding.normalization_path)),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.manifest, args.source, args.output)


if __name__ == "__main__":
    main()
