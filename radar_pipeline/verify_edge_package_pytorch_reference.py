"""Verify frozen preprocessing and PyTorch outputs in an Edge AI parity package."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from radar_pipeline.train_model import CNNLSTM, augment_window_features


BASE_FEATURES = ("x", "y", "z", "dop_idx", "range_m", "azimuth_deg", "elevation_deg")
DERIVED_FEATURES = (
    "xyz_delta_mag",
    "x_roll_std",
    "y_roll_std",
    "z_roll_std",
    "range_roll_std",
    "range_centered",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_from_base(base_records: np.ndarray, mean: np.ndarray, std: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normalized = ((base_records - mean) / std)[np.newaxis, :, :].astype(np.float32)
    payload = {
        "feature_names": np.asarray(BASE_FEATURES),
        "X_train": normalized,
        "X_val": normalized.copy(),
        "X_test": normalized.copy(),
    }
    augmented, feature_names = augment_window_features(payload)
    if tuple(feature_names) != BASE_FEATURES + DERIVED_FEATURES:
        raise ValueError("Frozen feature contract differs from the training implementation")
    return normalized, augmented["X_train"].astype(np.float32)


def verify(package_dir: Path, *, atol: float = 1e-6) -> dict[str, Any]:
    package_dir = package_dir.resolve()
    manifest = json.loads((package_dir / "deployment_manifest.json").read_text(encoding="utf-8"))
    checkpoint_path = package_dir / manifest["source_checkpoint"]["path"]
    if _sha256(checkpoint_path) != manifest["source_checkpoint"]["sha256"]:
        raise ValueError("Source checkpoint checksum mismatch")
    normalization_path = package_dir / manifest["normalization"]["path"]
    if _sha256(normalization_path) != manifest["normalization"]["sha256"]:
        raise ValueError("Normalization checksum mismatch")
    normalization = json.loads(normalization_path.read_text(encoding="utf-8"))
    mean = np.asarray(normalization["mean"], dtype=np.float32)
    std = np.asarray(normalization["std"], dtype=np.float32)
    checkpoint: dict[str, Any] = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["train_config"]
    model = CNNLSTM(13, 4, float(config["dropout_input"]), float(config["dropout_hidden"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    corpus = package_dir / "golden_corpus"
    cases: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for path in sorted(corpus.glob("valid_*_60.npz")):
        data = np.load(path, allow_pickle=False)
        cases.append((path.name, data["base_records"], data["expected_tensor"], data["expected_logits"], data["expected_probabilities"]))
    cadence = np.load(corpus / "cadence_real_90_records.npz", allow_pickle=False)
    for position, end_index in enumerate(cadence["expected_invocation_indices"]):
        base = cadence["base_records"][int(end_index) - 59 : int(end_index) + 1]
        cases.append(
            (
                f"cadence_real_90_records.npz[{position}]",
                base,
                cadence["expected_tensors"][position],
                cadence["expected_logits"][position],
                cadence["expected_probabilities"][position],
            )
        )
    rows: list[dict[str, Any]] = []
    max_tensor_error = 0.0
    max_logit_error = 0.0
    max_probability_error = 0.0
    for case_id, base, expected_tensor, expected_logits, expected_probabilities in cases:
        normalized, tensor = _tensor_from_base(base, mean, std)
        with torch.inference_mode():
            logits = model(torch.from_numpy(tensor)).numpy()[0].astype(np.float32)
            probabilities = torch.softmax(torch.from_numpy(logits), dim=0).numpy().astype(np.float32)
        tensor_error = float(np.max(np.abs(tensor - expected_tensor)))
        logit_error = float(np.max(np.abs(logits - expected_logits)))
        probability_error = float(np.max(np.abs(probabilities - expected_probabilities)))
        max_tensor_error = max(max_tensor_error, tensor_error)
        max_logit_error = max(max_logit_error, logit_error)
        max_probability_error = max(max_probability_error, probability_error)
        rows.append(
            {
                "case": case_id,
                "normalized_shape": list(normalized.shape),
                "tensor_max_abs_error": tensor_error,
                "logit_max_abs_error": logit_error,
                "probability_max_abs_error": probability_error,
                "class_match": int(np.argmax(logits)) == int(np.argmax(expected_logits)),
            }
        )
    passed = (
        max_tensor_error <= atol
        and max_logit_error <= atol
        and max_probability_error <= atol
        and all(row["class_match"] for row in rows)
    )
    report = {
        "status": "pass" if passed else "fail",
        "source_checkpoint_sha256": manifest["source_checkpoint"]["sha256"],
        "normalization_sha256": manifest["normalization"]["sha256"],
        "sample_count": len(rows),
        "atol": atol,
        "max_tensor_abs_error": max_tensor_error,
        "max_logit_abs_error": max_logit_error,
        "max_probability_abs_error": max_probability_error,
        "cases": rows,
    }
    (package_dir / "pytorch_reference_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=1e-6)
    args = parser.parse_args()
    print(json.dumps(verify(args.package_dir, atol=args.atol), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
