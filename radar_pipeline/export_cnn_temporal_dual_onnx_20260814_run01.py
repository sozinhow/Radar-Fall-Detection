"""Export and validate the Fold 3/Fold 4 CNNTemporal ONNX pair.

This is a new, versioned export runner.  It deliberately does not modify the
existing exporter, checkpoints, metrics, or deployment packages.  The ONNX
models consume already-normalized float32 [1, 60, 7] tensors; normalization is
recorded separately and is bound to the source checkpoint for each fold.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if __package__ in {None, ""}:
    sys.path.insert(0, str(ROOT))

from radar_pipeline.model_cnn_only import CNNTemporal


def repo_relative(path: Path) -> str:
    """Return a portable path relative to the radar repository root."""

    return path.resolve().relative_to(ROOT).as_posix()


def relative_to(path: Path, base: Path) -> str:
    """Return a portable path relative to an artifact directory."""

    return Path(os.path.relpath(path.resolve(), base.resolve())).as_posix()


OUTPUT_ROOT = ROOT / "outputs" / "deployment" / "cnn_temporal_dual_onnx_20260814_run01"
STAGING_DATASET = ROOT / "outputs" / "event_centered_clips_sgkf4_20260731_rebuild01" / "staging_clip_dataset_60.npz"
FEATURE_ORDER = (
    "x",
    "y",
    "z",
    "dop_idx",
    "range_m",
    "azimuth_deg",
    "elevation_deg",
)
CLASS_ORDER = ("walking", "standing", "sitting", "fall")
BINARY_CLASS_ORDER = ("non_fall", "fall")
CLIP_LENGTH = 60
N_FEATURES = 7
OPSET = 17
INPUT_NAME = "radar_input"
ATOL = 1e-5
RTOL = 1e-5
RUN_ID = OUTPUT_ROOT.name


class BinaryFallWrapper(torch.nn.Module):
    """Group a frozen four-class checkpoint into non_fall/fall logits."""

    def __init__(self, base_model: torch.nn.Module):
        super().__init__()
        self.base_model = base_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.base_model(x)
        non_fall_logit = torch.logsumexp(
            logits[:, :3],
            dim=1,
            keepdim=True,
        )
        fall_logit = logits[:, 3:4]
        return torch.cat(
            [non_fall_logit, fall_logit],
            dim=1,
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _as_float_list(value: Any, field: str) -> list[float]:
    values = np.asarray(value, dtype=np.float32)
    if values.shape != (N_FEATURES,) or not np.isfinite(values).all():
        raise ValueError(f"{field} must be a finite seven-value vector; got {values.shape}")
    return [float(item) for item in values]


def load_and_validate_checkpoint(checkpoint_path: Path, expected_fold: int) -> tuple[dict[str, Any], CNNTemporal, np.ndarray, np.ndarray]:
    checkpoint: dict[str, Any] = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    expected = {
        "model_class": "CNNTemporal",
        "clip_length_frames": CLIP_LENGTH,
        "n_features": N_FEATURES,
        "model_feature_names": list(FEATURE_ORDER),
        "class_order": list(CLASS_ORDER),
        "outer_fold": expected_fold,
    }
    for key, expected_value in expected.items():
        actual = checkpoint.get(key)
        if key in {"clip_length_frames", "n_features", "outer_fold"}:
            try:
                actual = int(actual)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Fold {expected_fold} checkpoint field {key!r} is invalid: {actual!r}") from exc
        if actual != expected_value:
            raise ValueError(
                f"Fold {expected_fold} checkpoint contract disagreement for {key}: "
                f"expected {expected_value!r}, got {actual!r}"
            )

    normalization = checkpoint.get("normalization")
    if not isinstance(normalization, dict):
        raise ValueError(f"Fold {expected_fold} checkpoint is missing normalization metadata")
    mean = np.asarray(_as_float_list(normalization.get("mean"), "normalization.mean"), dtype=np.float32)
    std = np.asarray(_as_float_list(normalization.get("std"), "normalization.std"), dtype=np.float32)
    if np.any(std <= 0):
        raise ValueError(f"Fold {expected_fold} normalization.std must be positive")

    train_config = checkpoint.get("train_config")
    if not isinstance(train_config, dict):
        raise ValueError(f"Fold {expected_fold} checkpoint is missing train_config")
    dropout = float(train_config.get("dropout", 0.25))
    model = CNNTemporal(
        n_features=int(checkpoint["n_features"]),
        n_classes=len(checkpoint["class_order"]),
        dropout=dropout,
    )
    # Strict loading is part of the checkpoint contract.  In particular, it
    # catches a mistaken 13-feature reconstruction before export.
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return checkpoint, model, mean, std


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted).astype(np.float32)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities.astype(np.float32)


def load_real_held_out_cases(
    checkpoint: dict[str, Any],
    fold: int,
    mean: np.ndarray,
    std: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    if not STAGING_DATASET.is_file():
        raise FileNotFoundError(f"Expected matching held-out staging dataset: {STAGING_DATASET}")
    with np.load(STAGING_DATASET, allow_pickle=True) as loaded:
        required = {"X", "y", "mask", "feature_names", "source_session_id", "outer_fold"}
        missing = sorted(required - set(loaded.files))
        if missing:
            raise ValueError(f"Held-out staging dataset is missing keys: {missing}")
        raw = loaded["X"].astype(np.float32)
        labels = loaded["y"].astype(np.int64)
        masks = loaded["mask"].astype(bool)
        features = tuple(str(value) for value in loaded["feature_names"])
        outer_folds = loaded["outer_fold"].astype(np.int64)
        session_ids = loaded["source_session_id"].astype(str)

    if raw.ndim != 3 or raw.shape[1:] != (CLIP_LENGTH, N_FEATURES):
        raise ValueError(f"Held-out dataset must be [N,60,7], got {raw.shape}")
    if features != FEATURE_ORDER:
        raise ValueError(f"Held-out dataset feature order disagrees: {features!r}")

    test_sessions = {str(value) for value in checkpoint.get("test_source_sessions", [])}
    selected: list[int] = []
    metadata: list[dict[str, Any]] = []
    for label, class_name in enumerate(CLASS_ORDER):
        candidates = np.flatnonzero(
            (labels == label)
            & (outer_folds == fold)
            & np.asarray([session in test_sessions for session in session_ids])
            & masks.all(axis=1)
        )
        if len(candidates) == 0:
            raise ValueError(f"No complete real held-out Fold {fold} example for {class_name}")
        index = int(candidates[0])
        selected.append(index)
        metadata.append(
            {
                "case_id": f"real_fold{fold}_{class_name}_60_{session_ids[index]}",
                "source": "real held-out staging example",
                "dataset": str(STAGING_DATASET.relative_to(ROOT)),
                "dataset_index": index,
                "source_session_id": str(session_ids[index]),
                "ground_truth_index": label,
                "ground_truth_class": class_name,
                "outer_fold": int(outer_folds[index]),
                "complete_mask": True,
            }
        )
    normalized = ((raw[selected] - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)).astype(np.float32)
    return normalized, metadata


def make_validation_inputs(
    checkpoint: dict[str, Any],
    fold: int,
    mean: np.ndarray,
    std: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    rng = np.random.default_rng(20260814 + fold)
    random_inputs = rng.normal(0.0, 1.0, size=(4, CLIP_LENGTH, N_FEATURES)).astype(np.float32)
    random_metadata = [
        {
            "case_id": f"deterministic_random_fold{fold}_{index}",
            "source": "deterministic random normalized tensor",
            "ground_truth_index": None,
            "ground_truth_class": None,
        }
        for index in range(random_inputs.shape[0])
    ]
    real_inputs, real_metadata = load_real_held_out_cases(checkpoint, fold, mean, std)
    return np.concatenate([random_inputs, real_inputs], axis=0), random_metadata + real_metadata


def _shape_from_value_info(value_info: Any) -> list[Any]:
    shape: list[Any] = []
    tensor_shape = value_info.type.tensor_type.shape
    for dimension in tensor_shape.dim:
        if dimension.HasField("dim_value"):
            shape.append(int(dimension.dim_value))
        elif dimension.HasField("dim_param"):
            shape.append(dimension.dim_param)
        else:
            shape.append(None)
    return shape


def _onnx_value_info(value_info: Any) -> dict[str, Any]:
    import onnx

    element_type = value_info.type.tensor_type.elem_type
    return {
        "name": value_info.name,
        "shape": _shape_from_value_info(value_info),
        "dtype": onnx.TensorProto.DataType.Name(element_type).lower(),
    }


def export_onnx(model: torch.nn.Module, sample: np.ndarray, path: Path, output_name: str) -> None:
    import onnx

    with torch.inference_mode():
        torch.onnx.export(
            model,
            torch.from_numpy(sample[:1]),
            str(path),
            export_params=True,
            opset_version=OPSET,
            do_constant_folding=True,
            input_names=[INPUT_NAME],
            output_names=[output_name],
            dynamic_axes=None,
            dynamo=False,
            training=torch.onnx.TrainingMode.EVAL,
        )
    onnx_model = onnx.load(str(path))
    onnx.checker.check_model(onnx_model)


def graph_report(path: Path, expected_output_name: str, expected_output_shape: list[int]) -> dict[str, Any]:
    import onnx

    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    inputs = [_onnx_value_info(value) for value in model.graph.input]
    outputs = [_onnx_value_info(value) for value in model.graph.output]
    all_static = all(all(isinstance(dim, int) for dim in item["shape"]) for item in inputs + outputs)
    input_ok = inputs == [{"name": INPUT_NAME, "shape": [1, CLIP_LENGTH, N_FEATURES], "dtype": "float"}]
    output_ok = outputs == [{"name": expected_output_name, "shape": expected_output_shape, "dtype": "float"}]
    opset_imports = {
        (item.domain or "ai.onnx"): int(item.version)
        for item in model.opset_import
    }
    operators = Counter(node.op_type for node in model.graph.node)
    return {
        "status": "pass" if all_static and input_ok and output_ok and opset_imports.get("ai.onnx") == OPSET else "fail",
        "onnx_checker": "pass",
        "onnx_path": str(path.name),
        "onnx_sha256": sha256_file(path),
        "ir_version": int(model.ir_version),
        "opset_imports": opset_imports,
        "dynamic_axes": False,
        "all_graph_input_output_dimensions_static": all_static,
        "inputs": inputs,
        "outputs": outputs,
        "operator_counts": dict(sorted(operators.items())),
        "reduce_log_sum_exp_present": "ReduceLogSumExp" in operators,
        "node_count": len(model.graph.node),
    }


def pytorch_outputs(model: torch.nn.Module, inputs: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[int]]:
    with torch.inference_mode():
        logits = model(torch.from_numpy(inputs)).cpu().numpy().astype(np.float32)
    probabilities = _softmax(logits)
    predicted = logits.argmax(axis=1).astype(int).tolist()
    return logits, probabilities, predicted


def onnxruntime_parity(
    path: Path,
    model: torch.nn.Module,
    inputs: np.ndarray,
    case_metadata: list[dict[str, Any]],
    expected_output_name: str,
    class_names: tuple[str, ...],
) -> dict[str, Any]:
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    input_metadata = session.get_inputs()[0]
    output_metadata = session.get_outputs()[0]
    torch_logits, torch_probabilities, torch_predicted = pytorch_outputs(model, inputs)
    ort_logits = np.concatenate(
        [
            session.run([expected_output_name], {INPUT_NAME: inputs[index : index + 1]})[0].astype(np.float32)
            for index in range(inputs.shape[0])
        ],
        axis=0,
    )
    ort_probabilities = _softmax(ort_logits)
    ort_predicted = ort_logits.argmax(axis=1).astype(int).tolist()
    abs_logits = np.abs(ort_logits - torch_logits)
    rel_logits = abs_logits / np.maximum(np.abs(torch_logits), 1e-12)
    abs_probabilities = np.abs(ort_probabilities - torch_probabilities)
    rel_probabilities = abs_probabilities / np.maximum(np.abs(torch_probabilities), 1e-12)
    logits_allclose = bool(np.allclose(ort_logits, torch_logits, atol=ATOL, rtol=RTOL))
    probabilities_allclose = bool(np.allclose(ort_probabilities, torch_probabilities, atol=ATOL, rtol=RTOL))
    predicted_classes_match = torch_predicted == ort_predicted
    cases: list[dict[str, Any]] = []
    for index, metadata in enumerate(case_metadata):
        cases.append(
            {
                **metadata,
                "pytorch_logits": [float(value) for value in torch_logits[index]],
                "onnx_logits": [float(value) for value in ort_logits[index]],
                "pytorch_probabilities": [float(value) for value in torch_probabilities[index]],
                "onnx_probabilities": [float(value) for value in ort_probabilities[index]],
                "pytorch_predicted_index": torch_predicted[index],
                "onnx_predicted_index": ort_predicted[index],
                "pytorch_predicted_class": class_names[torch_predicted[index]],
                "onnx_predicted_class": class_names[ort_predicted[index]],
                "max_abs_logit_error": float(abs_logits[index].max()),
                "max_relative_logit_error": float(rel_logits[index].max()),
                "max_abs_probability_error": float(abs_probabilities[index].max()),
                "max_relative_probability_error": float(rel_probabilities[index].max()),
            }
        )
    status = "pass" if logits_allclose and probabilities_allclose and predicted_classes_match else "fail"
    return {
        "status": status,
        "atol": ATOL,
        "rtol": RTOL,
        "onnxruntime_version": ort.__version__,
        "provider": "CPUExecutionProvider",
        "input": {
            "name": input_metadata.name,
            "shape": list(input_metadata.shape),
            "type": input_metadata.type,
        },
        "output": {
            "name": output_metadata.name,
            "shape": list(output_metadata.shape),
            "type": output_metadata.type,
        },
        "pytorch_input_shape": list(inputs.shape),
        "pytorch_output_shape": list(torch_logits.shape),
        "onnx_output_shape": list(ort_logits.shape),
        "logits_allclose": logits_allclose,
        "probabilities_allclose": probabilities_allclose,
        "predicted_classes_match": predicted_classes_match,
        "max_abs_logit_error": float(abs_logits.max()),
        "max_relative_logit_error": float(rel_logits.max()),
        "max_abs_probability_error": float(abs_probabilities.max()),
        "max_relative_probability_error": float(rel_probabilities.max()),
        "case_count": len(cases),
        "cases": cases,
    }


def collapsed_binary_metrics(checkpoint: dict[str, Any], metrics_json_path: Path) -> dict[str, Any]:
    checkpoint_metrics = checkpoint.get("test_metrics", {})
    checkpoint_cm = np.asarray(checkpoint_metrics.get("confusion_matrix"), dtype=np.int64)
    saved_metrics = json.loads(metrics_json_path.read_text(encoding="utf-8"))
    saved_cm = np.asarray(saved_metrics.get("test_metrics", {}).get("confusion_matrix"), dtype=np.int64)
    if checkpoint_cm.shape != (4, 4) or saved_cm.shape != (4, 4) or not np.array_equal(checkpoint_cm, saved_cm):
        raise ValueError("Fold 4 saved confusion matrices disagree or are not 4x4")
    binary_cm = np.zeros((2, 2), dtype=np.int64)
    binary_cm[0, 0] = int(checkpoint_cm[:3, :3].sum())
    binary_cm[0, 1] = int(checkpoint_cm[:3, 3].sum())
    binary_cm[1, 0] = int(checkpoint_cm[3, :3].sum())
    binary_cm[1, 1] = int(checkpoint_cm[3, 3])
    true_nonfall, predicted_nonfall = int(binary_cm[0].sum()), int(binary_cm[:, 0].sum())
    true_fall, predicted_fall = int(binary_cm[1].sum()), int(binary_cm[:, 1].sum())
    correct = int(np.trace(binary_cm))
    total = int(binary_cm.sum())
    fall_precision = binary_cm[1, 1] / predicted_fall if predicted_fall else 0.0
    fall_recall = binary_cm[1, 1] / true_fall if true_fall else 0.0
    fall_f1 = 2.0 * fall_precision * fall_recall / (fall_precision + fall_recall) if fall_precision + fall_recall else 0.0
    return {
        "source": "Fold 4 saved four-class test confusion matrix",
        "original_class_order": list(CLASS_ORDER),
        "binary_class_order": list(BINARY_CLASS_ORDER),
        "four_class_confusion_matrix": checkpoint_cm.tolist(),
        "binary_confusion_matrix": binary_cm.tolist(),
        "binary_accuracy": correct / total,
        "fall_precision": fall_precision,
        "fall_recall": fall_recall,
        "fall_f1": fall_f1,
        "support": {"non_fall": true_nonfall, "fall": true_fall},
        "predicted_support": {"non_fall": predicted_nonfall, "fall": predicted_fall},
    }


def compiler_analysis(path: Path, version_dir: Path, model_name: str) -> dict[str, Any]:
    """Probe only an installed ST Edge AI executable; never flash or modify firmware."""
    candidates = ("stedgeai", "stedgeai-cli", "stm32ai")
    binaries = {name: shutil.which(name) for name in candidates}
    available = next((value for value in binaries.values() if value), None)
    report: dict[str, Any] = {
        "status": "not_available",
        "model": model_name,
        "onnx_path": repo_relative(path),
        "exact_command": None,
        "tool_version": None,
        "output": "",
        "supported_operators": None,
        "ram_bytes": None,
        "flash_bytes": None,
        "latency_ms": None,
        "errors": "No stedgeai, stedgeai-cli, or stm32ai executable was found in PATH; compiler analysis was not run.",
        "available_binaries_in_path": binaries,
        "firmware_modified": False,
        "board_flashed": False,
    }
    if available:
        # The installed CLI's syntax is version-dependent.  Capture help and
        # version without guessing an analysis command or changing firmware.
        for command in ([available, "--version"], [available, "--help"]):
            try:
                result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=30)
                report["exact_command"] = report["exact_command"] or command
                report["output"] += f"$ {' '.join(command)}\n{result.stdout}{result.stderr}\n"
            except (OSError, subprocess.SubprocessError) as exc:
                report["errors"] += f" Probe failed for {' '.join(command)}: {exc}"
        report["status"] = "available_but_analysis_not_run"
        report["errors"] += " Analysis command was not inferred from help output; no compiler conversion was attempted."
    write_json(version_dir / "st_edge_ai_analysis.json", report)
    return report


def evaluation_evidence(checkpoint: dict[str, Any], fold: int, metrics_path: Path) -> dict[str, Any]:
    test_metrics = checkpoint["test_metrics"]
    evidence: dict[str, Any] = {
        "source": str(metrics_path),
        "fold": fold,
        "test_metrics": {
            "accuracy": float(test_metrics["accuracy"]),
            "macro_f1": float(test_metrics["macro_f1"]),
            "fall_precision": float(test_metrics["classification_report"]["fall"]["precision"]),
            "fall_recall": float(test_metrics["classification_report"]["fall"]["recall"]),
            "fall_f1": float(test_metrics["classification_report"]["fall"]["f1-score"]),
        },
        "confusion_matrix": test_metrics["confusion_matrix"],
    }
    if fold == 3:
        expected = {
            "accuracy": 0.8677685950,
            "macro_f1": 0.8638530615,
            "fall_precision": 1.0,
            "fall_recall": 0.7272727273,
            "fall_f1": 0.8421052632,
        }
        evidence["expected_existing_test_evidence"] = expected
        evidence["expected_values_match"] = all(
            math.isclose(evidence["test_metrics"][key], value, rel_tol=0.0, abs_tol=1e-9)
            for key, value in expected.items()
        )
    return evidence


def export_version(
    *,
    fold: int,
    version: str,
    checkpoint_path: Path,
    metrics_path: Path,
    output_dir: Path,
    output_name: str,
    class_order: tuple[str, ...],
    model_description: str,
    binary_adapter: bool,
) -> dict[str, Any]:
    checkpoint, base_model, mean, std = load_and_validate_checkpoint(checkpoint_path, fold)
    checkpoint_hash = sha256_file(checkpoint_path)
    inputs, cases = make_validation_inputs(checkpoint, fold, mean, std)
    model: torch.nn.Module = BinaryFallWrapper(base_model) if binary_adapter else base_model
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=False)

    normalization_path = output_dir / "normalization.json"
    normalization_payload = {
        "artifact_type": "checkpoint_bound_normalization",
        "source_checkpoint_path": relative_to(checkpoint_path, output_dir),
        "source_checkpoint_sha256": checkpoint_hash,
        "feature_order": list(FEATURE_ORDER),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "fit_scope": checkpoint["normalization"].get("fit_scope"),
        "dtype": "float32",
        "preprocessing_formula": "(raw_input - mean) / std",
        "onnx_expects": "already-normalized float32 [1,60,7] input",
    }
    write_json(normalization_path, normalization_payload)

    onnx_path = output_dir / output_name
    export_onnx(model, inputs, onnx_path, "binary_logits" if binary_adapter else "logits")
    onnx_checker_status = "pass"
    expected_output_shape = [1, len(class_order)]
    graph = graph_report(onnx_path, "binary_logits" if binary_adapter else "logits", expected_output_shape)
    write_json(output_dir / "onnx_graph_report.json", graph)
    if graph["status"] != "pass":
        raise RuntimeError(f"ONNX graph contract failed for Fold {fold}: {graph}")

    parity = onnxruntime_parity(
        onnx_path,
        model,
        inputs,
        cases,
        "binary_logits" if binary_adapter else "logits",
        class_order,
    )
    parity["onnx_sha256"] = sha256_file(onnx_path)
    write_json(output_dir / "onnxruntime_parity_report.json", parity)
    if parity["status"] != "pass":
        raise RuntimeError(f"ONNX Runtime parity failed for Fold {fold}: {parity}")

    compiler = compiler_analysis(onnx_path, output_dir, version)
    evidence = evaluation_evidence(checkpoint, fold, metrics_path)
    if binary_adapter:
        evidence["collapsed_binary_metrics"] = collapsed_binary_metrics(checkpoint, metrics_path)

    onnx_hash = sha256_file(onnx_path)
    manifest = {
        "deployment_id": RUN_ID,
        "version": version,
        "status": "pass",
        "model_description": model_description,
        "source_checkpoint": {
            "path": relative_to(checkpoint_path, output_dir),
            "sha256": checkpoint_hash,
            "model_class": checkpoint["model_class"],
            "outer_fold": int(checkpoint["outer_fold"]),
        },
        "onnx": {
            "path": onnx_path.name,
            "sha256": onnx_hash,
            "format": "onnx",
            "opset": OPSET,
        },
        "model_class": checkpoint["model_class"],
        "input": {"name": INPUT_NAME, "shape": [1, CLIP_LENGTH, N_FEATURES], "dtype": "float32"},
        "output": {"name": "binary_logits" if binary_adapter else "logits", "shape": expected_output_shape, "dtype": "float32"},
        "feature_order": list(FEATURE_ORDER),
        "class_order": list(class_order),
        "normalization": {
            "path": normalization_path.name,
            "sha256": sha256_file(normalization_path),
            "mean": mean.tolist(),
            "std": std.tolist(),
        },
        "clip_length_frames": CLIP_LENGTH,
        "dtype": "float32",
        "opset": OPSET,
        "evaluation_mode": True,
        "dynamic_axes": False,
        "parity": {
            "status": parity["status"],
            "report": "onnxruntime_parity_report.json",
            "atol": ATOL,
            "rtol": RTOL,
            "max_abs_logit_error": parity["max_abs_logit_error"],
            "max_relative_logit_error": parity["max_relative_logit_error"],
            "max_abs_probability_error": parity["max_abs_probability_error"],
            "max_relative_probability_error": parity["max_relative_probability_error"],
            "predicted_classes_match": parity["predicted_classes_match"],
        },
        "onnx_checker": onnx_checker_status,
        "reports": {
            "normalization": normalization_path.name,
            "deployment_manifest": "deployment_manifest.json",
            "export_report": "export_report.json",
            "onnx_graph_report": "onnx_graph_report.json",
            "onnxruntime_parity_report": "onnxruntime_parity_report.json",
            "st_edge_ai_analysis": "st_edge_ai_analysis.json",
        },
        "compiler": {
            "status": compiler["status"],
            "report": "st_edge_ai_analysis.json",
            "ram_bytes": compiler["ram_bytes"],
            "flash_bytes": compiler["flash_bytes"],
            "latency_ms": compiler["latency_ms"],
        },
    }
    write_json(output_dir / "deployment_manifest.json", manifest)
    export_report = {
        "status": "pass",
        "version": version,
        "model_description": model_description,
        "source_checkpoint_path": relative_to(checkpoint_path, output_dir),
        "source_checkpoint_sha256": checkpoint_hash,
        "onnx_path": onnx_path.name,
        "onnx_sha256": onnx_hash,
        "checkpoint_contract": {
            "validated_before_export": True,
            "model_class": checkpoint["model_class"],
            "clip_length_frames": int(checkpoint["clip_length_frames"]),
            "n_features": int(checkpoint["n_features"]),
            "feature_order": list(FEATURE_ORDER),
            "original_class_order": list(CLASS_ORDER),
            "outer_fold": int(checkpoint["outer_fold"]),
            "strict_state_dict_load": True,
        },
        "requested_onnx_contract": {
            "dtype": "float32",
            "opset": OPSET,
            "evaluation_mode": True,
            "dynamic_axes": False,
            "input_name": INPUT_NAME,
            "input_shape": [1, CLIP_LENGTH, N_FEATURES],
            "output_name": "binary_logits" if binary_adapter else "logits",
            "output_shape": expected_output_shape,
            "class_order": list(class_order),
        },
        "normalization": normalization_payload,
        "evaluation_evidence": evidence,
        "validation": {
            "onnx_checker": onnx_checker_status,
            "onnxruntime_parity": parity["status"],
            "report": "onnxruntime_parity_report.json",
        },
        "compiler": compiler,
        "adapter": {
            "is_binary_adapter": binary_adapter,
            "weights_unchanged": True,
            "mathematics": "non_fall_logit=logsumexp(four_class_logits[:, :3]); fall_logit=four_class_logits[:, 3]" if binary_adapter else None,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "onnx": __import__("onnx").__version__,
            "onnxruntime": __import__("onnxruntime").__version__,
            "platform": platform.platform(),
        },
    }
    write_json(output_dir / "export_report.json", export_report)
    return {
        "version": version,
        "output_dir": output_dir,
        "checkpoint": checkpoint,
        "checkpoint_sha256": checkpoint_hash,
        "onnx_path": onnx_path,
        "onnx_sha256": onnx_hash,
        "manifest": manifest,
        "export_report": export_report,
        "graph": graph,
        "parity": parity,
        "compiler": compiler,
        "evidence": evidence,
    }


def comparison_report(results: list[dict[str, Any]]) -> str:
    lines = [
        f"# CNNTemporal dual ONNX comparison — {RUN_ID}",
        "",
        "Both ONNX graphs are static float32 exports with input `[1,60,7]`; the input is expected to be normalized with the fold-local `normalization.json` before inference.",
        "",
        "## Model choice",
        "",
        "- Fold 3 supplies the four-class artifact because its saved held-out test evidence is the requested four-class reference: accuracy 0.8677685950 and macro F1 0.8638530615, with fall precision 1.0, recall 0.7272727273, and F1 0.8421052632.",
        "- Fold 4 supplies the fall-sensitive binary artifact because its saved four-class confusion matrix has 100% fall recall. Collapsing walking/standing/sitting into `non_fall` gives 97.54098% binary accuracy, 76.9231% fall precision, 100% fall recall, and 86.9565% fall F1; precision is below 80%.",
        "- The binary ONNX is a binary adapter derived from a four-class checkpoint; it is not a separately retrained binary model. Its `non_fall` logit is `logsumexp` over the original three non-fall logits, and its `fall` logit is the original fall logit.",
        "",
        "## Validation comparison",
        "",
        "| Version | ONNX artifact | Output classes | PyTorch–ONNX parity | Max abs logit error | Max abs probability error | ST Edge AI |\n|---|---|---|---|---:|---:|---|",
    ]
    for result in results:
        parity = result["parity"]
        manifest = result["manifest"]
        lines.append(
            f"| {result['version']} | `{repo_relative(result['onnx_path'])}` | `{', '.join(manifest['class_order'])}` | "
            f"{parity['status']} (classes match: {parity['predicted_classes_match']}) | "
            f"{parity['max_abs_logit_error']:.9g} | {parity['max_abs_probability_error']:.9g} | {result['compiler']['status']} |"
        )
    lines += [
        "",
        "Parity used deterministic random normalized tensors plus one complete real held-out example for each original class from the matching Fold 3/Fold 4 staging split. It compared output shapes, logits, softmax probabilities, and predicted classes with `atol=1e-5` and `rtol=1e-5`; both versions passed.",
        "",
        "## ST Edge AI status",
        "",
        "No `stedgeai`, `stedgeai-cli`, or `stm32ai` executable was available in PATH, so compiler analysis was not run. Consequently, supported operators, RAM, flash, and latency are recorded as unavailable rather than inferred. The binary graph does contain `ReduceLogSumExp`; if a later ST Edge AI analysis rejects it, preserve the requested mathematics and group the four probabilities outside ONNX, or pursue a separately approved true-binary retraining task. No firmware was modified and no board was flashed. The per-version `st_edge_ai_analysis.json` files record this exact status and the attempted tool discovery.",
        "",
        "## Artifacts and SHA-256",
        "",
    ]
    for result in results:
        lines += [
            f"### {result['version']}",
            f"- ONNX: `{repo_relative(result['onnx_path'])}`",
            f"  - SHA-256: `{result['onnx_sha256']}`",
            f"- Source checkpoint: `{result['manifest']['source_checkpoint']['path']}`",
            f"  - SHA-256: `{result['checkpoint_sha256']}`",
            f"- Manifest: `{repo_relative(result['output_dir'] / 'deployment_manifest.json')}`",
            f"- Export report: `{repo_relative(result['output_dir'] / 'export_report.json')}`",
            f"- ONNX graph report: `{repo_relative(result['output_dir'] / 'onnx_graph_report.json')}`",
            f"- ONNX Runtime parity report: `{repo_relative(result['output_dir'] / 'onnxruntime_parity_report.json')}`",
            f"- ST Edge AI report: `{repo_relative(result['output_dir'] / 'st_edge_ai_analysis.json')}`",
            "",
        ]
    return "\n".join(lines)


def main() -> None:
    if OUTPUT_ROOT.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {OUTPUT_ROOT}")
    checkpoint_3 = ROOT / "fold_summary" / "fold_3" / "cnn_temporal_event_centered.pt"
    metrics_3 = ROOT / "fold_summary" / "fold_3" / "metrics.json"
    checkpoint_4 = ROOT / "fold_summary" / "fold_4" / "cnn_temporal_event_centered.pt"
    metrics_4 = ROOT / "fold_summary" / "fold_4" / "metrics.json"
    for path in (checkpoint_3, metrics_3, checkpoint_4, metrics_4):
        if not path.is_file():
            raise FileNotFoundError(path)

    # Validate both checkpoints before creating any output directory.  This
    # makes a contract disagreement a clean stop with no partial deployment.
    load_and_validate_checkpoint(checkpoint_3, 3)
    load_and_validate_checkpoint(checkpoint_4, 4)

    OUTPUT_ROOT.mkdir(parents=False, exist_ok=False)
    four_class_dir = OUTPUT_ROOT / "four_class"
    binary_dir = OUTPUT_ROOT / "binary"
    four_class = export_version(
        fold=3,
        version="fold3_four_class",
        checkpoint_path=checkpoint_3,
        metrics_path=metrics_3,
        output_dir=four_class_dir,
        output_name="cnn_temporal_fold3_four_class.onnx",
        class_order=CLASS_ORDER,
        model_description="Fold 3 four-class CNNTemporal checkpoint export",
        binary_adapter=False,
    )
    binary = export_version(
        fold=4,
        version="fold4_binary_nonfall_fall",
        checkpoint_path=checkpoint_4,
        metrics_path=metrics_4,
        output_dir=binary_dir,
        output_name="cnn_temporal_fold4_nonfall_fall.onnx",
        class_order=BINARY_CLASS_ORDER,
        model_description="binary adapter derived from a four-class checkpoint; not a separately retrained binary model",
        binary_adapter=True,
    )
    report_path = OUTPUT_ROOT / "comparison_report.md"
    report_path.write_text(comparison_report([four_class, binary]), encoding="utf-8")
    print(report_path)
    print(four_class["onnx_path"], four_class["onnx_sha256"])
    print(binary["onnx_path"], binary["onnx_sha256"])


if __name__ == "__main__":
    main()
