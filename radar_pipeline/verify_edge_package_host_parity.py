"""Verify the live causal host engine against an Edge AI parity package."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from radar_pipeline.live_inference import (
    CausalHostInference,
    CNNBiLSTM,
    DemoBinding,
    LiveRadarRecord,
    RESET_REASON_MISSING_RECORD,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _record(row: np.ndarray, *, timestamp_s: float, frame_id: int) -> LiveRadarRecord:
    return LiveRadarRecord(
        timestamp_s=timestamp_s,
        frame_id=frame_id,
        x=float(row[0]),
        y=float(row[1]),
        z=float(row[2]),
        dop_idx=int(row[3]),
        range_m=float(row[4]),
        azimuth_deg=float(row[5]),
        elevation_deg=float(row[6]),
    )


def _engine(package_dir: Path) -> CausalHostInference:
    manifest = json.loads((package_dir / "deployment_manifest.json").read_text(encoding="utf-8"))
    checkpoint_path = package_dir / manifest["source_checkpoint"]["path"]
    if _sha256(checkpoint_path) != manifest["source_checkpoint"]["sha256"]:
        raise ValueError("Source checkpoint checksum mismatch")
    normalizer = json.loads((package_dir / manifest["normalization"]["path"]).read_text(encoding="utf-8"))
    checkpoint: dict[str, Any] = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["train_config"]
    model = CNNBiLSTM(float(config["dropout_input"]), float(config["dropout_hidden"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    binding = DemoBinding(
        manifest_path=package_dir / "deployment_manifest.json",
        checkpoint_path=checkpoint_path,
        normalization_path=package_dir / manifest["normalization"]["path"],
        mean=np.asarray(normalizer["mean"], dtype=np.float32),
        std=np.asarray(normalizer["std"], dtype=np.float32),
        model=model,
    )
    return CausalHostInference(binding)


def verify(package_dir: Path, *, atol: float = 1e-6) -> dict[str, Any]:
    package_dir = package_dir.resolve()
    corpus = package_dir / "golden_corpus"
    rows: list[dict[str, Any]] = []
    max_tensor_error = 0.0
    max_probability_error = 0.0
    for path in sorted(corpus.glob("valid_*_60.npz")):
        data = np.load(path, allow_pickle=False)
        engine = _engine(package_dir)
        update = None
        for index, row in enumerate(data["base_records"]):
            update = engine.ingest(_record(row, timestamp_s=index / 20.0, frame_id=index + 1))
        if update is None or update.tensor is None or update.probabilities is None:
            raise RuntimeError(f"No host inference result for {path.name}")
        tensor_error = float(np.max(np.abs(update.tensor - data["expected_tensor"])))
        probability_error = float(np.max(np.abs(update.probabilities - data["expected_probabilities"])))
        max_tensor_error = max(max_tensor_error, tensor_error)
        max_probability_error = max(max_probability_error, probability_error)
        rows.append(
            {
                "case": path.name,
                "tensor_max_abs_error": tensor_error,
                "probability_max_abs_error": probability_error,
                "class_match": update.predicted_class == str(data["expected_class"]),
            }
        )

    cadence = np.load(corpus / "cadence_real_90_records.npz", allow_pickle=False)
    engine = _engine(package_dir)
    observed = []
    for index, row in enumerate(cadence["base_records"]):
        update = engine.ingest(_record(row, timestamp_s=index / 20.0, frame_id=index + 1))
        # The host intentionally assembles a fresh ready-state tensor on every
        # complete 60-record window. A model call is signalled by a prediction
        # (and probabilities), which happens only at the 60/75/90 cadence.
        if update.predicted_class is not None:
            observed.append((index, update))
    expected_indices = cadence["expected_invocation_indices"].tolist()
    if [index for index, _ in observed] != expected_indices:
        raise AssertionError(f"Host cadence differs: {[index for index, _ in observed]} != {expected_indices}")
    for position, (index, update) in enumerate(observed):
        if update.probabilities is None or update.tensor is None:
            raise AssertionError("Causal host emitted an incomplete inference update")
        tensor_error = float(np.max(np.abs(update.tensor - cadence["expected_tensors"][position])))
        probability_error = float(np.max(np.abs(update.probabilities - cadence["expected_probabilities"][position])))
        max_tensor_error = max(max_tensor_error, tensor_error)
        max_probability_error = max(max_probability_error, probability_error)
        rows.append(
            {
                "case": f"cadence_real_90_records.npz[{position}]",
                "invocation_index_zero_based": index,
                "tensor_max_abs_error": tensor_error,
                "probability_max_abs_error": probability_error,
                "class_match": update.predicted_class == str(cadence["expected_classes"][position]),
            }
        )

    startup = np.load(corpus / "startup_0_to_59_no_output.npz", allow_pickle=False)
    engine = _engine(package_dir)
    startup_updates = [
        engine.ingest(_record(row, timestamp_s=index / 20.0, frame_id=index + 1))
        for index, row in enumerate(startup["base_records"])
    ]
    startup_pass = not any(update.tensor is not None or update.probabilities is not None for update in startup_updates)

    reset_results: list[dict[str, Any]] = []
    for path in sorted(corpus.glob("reset_*.npz")):
        data = np.load(path, allow_pickle=False)
        expected_reason = str(data["expected_reset_reason"])
        engine = _engine(package_dir)
        if expected_reason == RESET_REASON_MISSING_RECORD:
            for index, row in enumerate(data["base_records"]):
                engine.ingest(_record(row, timestamp_s=index / 20.0, frame_id=index + 1))
            update = engine.reset(RESET_REASON_MISSING_RECORD, detail="missing_scheduled_record")
        else:
            update = None
            for row, timestamp_s, frame_id in zip(data["base_records"], data["timestamps_s"], data["frame_ids"]):
                update = engine.ingest(_record(row, timestamp_s=float(timestamp_s), frame_id=int(frame_id)))
            if update is None:
                raise AssertionError(f"No update for reset case {path.name}")
        reset_results.append(
            {
                "case": path.name,
                "expected_reset_reason": expected_reason,
                "actual_reset_reason": update.reset_reason,
                "model_invocation": update.tensor is not None or update.probabilities is not None,
                "pass": update.reset_reason == expected_reason and update.tensor is None and update.probabilities is None,
            }
        )
    passed = (
        startup_pass
        and max_tensor_error <= atol
        and max_probability_error <= atol
        and all(row["class_match"] for row in rows)
        and all(row["pass"] for row in reset_results)
    )
    manifest = json.loads((package_dir / "deployment_manifest.json").read_text(encoding="utf-8"))
    report = {
        "status": "pass" if passed else "fail",
        "source_checkpoint_sha256": manifest["source_checkpoint"]["sha256"],
        "sample_count": len(rows),
        "atol": atol,
        "max_tensor_abs_error": max_tensor_error,
        "max_probability_abs_error": max_probability_error,
        "startup_no_output_pass": startup_pass,
        "inference_indices_zero_based": expected_indices,
        "valid_and_cadence_cases": rows,
        "reset_cases": reset_results,
    }
    (package_dir / "host_preprocessing_parity_report.json").write_text(
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
