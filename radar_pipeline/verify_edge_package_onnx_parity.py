"""Compare a static ONNX Edge AI package against its frozen PyTorch references."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _max_relative(actual: np.ndarray, expected: np.ndarray) -> float:
    denominator = np.maximum(np.abs(expected), 1e-12)
    return float(np.max(np.abs(actual - expected) / denominator))


def _verify_current_binary_package(package_dir: Path) -> dict[str, Any]:
    """Verify the frozen Version 1 binary package layout.

    The older host-demo package uses model.onnx plus a golden corpus. Version 1
    stores its ONNX filename and its already-passed parity report in the
    deployment manifest, so it needs a separate read-only verification path.
    """

    manifest = json.loads((package_dir / "deployment_manifest.json").read_text(encoding="utf-8"))
    onnx_path = package_dir / Path(manifest["onnx"]["path"]).name
    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX model is missing: {onnx_path}")
    actual_sha256 = _sha256(onnx_path)
    expected_sha256 = manifest["onnx"]["sha256"]
    if actual_sha256 != expected_sha256:
        raise ValueError(f"ONNX checksum mismatch: expected {expected_sha256}, got {actual_sha256}")
    checkpoint_path = package_dir / manifest["source_checkpoint"]["path"]
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Source checkpoint is missing: {checkpoint_path}")
    checkpoint_sha256 = _sha256(checkpoint_path)
    if checkpoint_sha256 != manifest["source_checkpoint"]["sha256"]:
        raise ValueError("Version 1 source checkpoint checksum mismatch")
    normalization_path = package_dir / manifest["normalization"]["path"]
    if _sha256(normalization_path) != manifest["normalization"]["sha256"]:
        raise ValueError("Version 1 normalization checksum mismatch")

    stored_parity = json.loads((package_dir / "onnxruntime_parity_report.json").read_text(encoding="utf-8"))
    if stored_parity.get("status") != "pass":
        raise ValueError("Stored Version 1 ONNX Runtime parity report is not pass")

    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_meta = session.get_inputs()[0]
    output_meta = session.get_outputs()[0]
    expected_input = manifest["input"]
    expected_output = manifest["output"]
    if input_meta.name != expected_input["name"] or output_meta.name != expected_output["name"]:
        raise ValueError("Version 1 ONNX input/output names differ from the manifest")
    sample = np.zeros(tuple(expected_input["shape"]), dtype=np.float32)
    logits = session.run([output_meta.name], {input_meta.name: sample})[0]
    if list(logits.shape) != expected_output["shape"]:
        raise ValueError(f"Version 1 ONNX output shape differs from the manifest: {logits.shape}")
    return {
        "status": "pass",
        "version": manifest["version"],
        "onnx": str(onnx_path),
        "onnx_sha256": actual_sha256,
        "source_checkpoint_sha256": checkpoint_sha256,
        "input": {"name": input_meta.name, "shape": list(expected_input["shape"])},
        "output": {"name": output_meta.name, "shape": list(logits.shape)},
        "stored_parity_status": stored_parity["status"],
    }


def _verify_current_four_class_package(package_dir: Path) -> dict[str, Any]:
    """Verify the named-ONNX four-class package layout."""

    manifest = json.loads((package_dir / "deployment_manifest.json").read_text(encoding="utf-8"))
    onnx_path = package_dir / Path(manifest["onnx"]["path"]).name
    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX model is missing: {onnx_path}")
    actual_sha256 = _sha256(onnx_path)
    if actual_sha256 != manifest["onnx"]["sha256"]:
        raise ValueError(
            f"ONNX checksum mismatch: expected {manifest['onnx']['sha256']}, got {actual_sha256}"
        )

    checkpoint_path = package_dir / manifest["source_checkpoint"]["path"]
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Source checkpoint is missing: {checkpoint_path}")
    checkpoint_sha256 = _sha256(checkpoint_path)
    if checkpoint_sha256 != manifest["source_checkpoint"]["sha256"]:
        raise ValueError("Four-class source checkpoint checksum mismatch")

    normalization_path = package_dir / manifest["normalization"]["path"]
    if not normalization_path.is_file():
        raise FileNotFoundError(f"Normalization file is missing: {normalization_path}")
    if _sha256(normalization_path) != manifest["normalization"]["sha256"]:
        raise ValueError("Four-class normalization checksum mismatch")

    stored_parity = json.loads(
        (package_dir / "onnxruntime_parity_report.json").read_text(encoding="utf-8")
    )
    if stored_parity.get("status") != "pass":
        raise ValueError("Stored four-class ONNX Runtime parity report is not pass")

    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_meta = session.get_inputs()[0]
    output_meta = session.get_outputs()[0]
    expected_input = manifest["input"]
    expected_output = manifest["output"]
    if input_meta.name != expected_input["name"] or output_meta.name != expected_output["name"]:
        raise ValueError("Four-class ONNX input/output names differ from the manifest")
    sample = np.zeros(tuple(expected_input["shape"]), dtype=np.float32)
    logits = session.run([output_meta.name], {input_meta.name: sample})[0]
    if list(logits.shape) != expected_output["shape"]:
        raise ValueError(f"Four-class ONNX output shape differs from the manifest: {logits.shape}")
    return {
        "status": "pass",
        "version": manifest["version"],
        "onnx": str(onnx_path),
        "onnx_sha256": actual_sha256,
        "source_checkpoint_sha256": checkpoint_sha256,
        "input": {"name": input_meta.name, "shape": list(expected_input["shape"])},
        "output": {"name": output_meta.name, "shape": list(logits.shape)},
        "class_order": manifest["class_order"],
        "stored_parity_status": stored_parity["status"],
    }


def verify(package_dir: Path, *, atol: float, rtol: float) -> dict[str, Any]:
    package_dir = package_dir.resolve()
    manifest = json.loads((package_dir / "deployment_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("version") == "fold4_binary_nonfall_fall":
        return _verify_current_binary_package(package_dir)
    if manifest.get("version") == "fold3_four_class":
        return _verify_current_four_class_package(package_dir)
    onnx_path = package_dir / "model.onnx"
    report_path = package_dir / "onnx_parity_report.json"
    if importlib.util.find_spec("onnxruntime") is None:
        message = "Missing required dependency: onnxruntime. Install it in the pinned export environment before retrying."
        report = {
            "status": "blocked",
            "reason": "missing_dependency",
            "missing_dependency": "onnxruntime",
            "message": message,
            "source_checkpoint_sha256": manifest["source_checkpoint"]["sha256"],
            "onnx_sha256": _sha256(onnx_path) if onnx_path.is_file() else None,
            "sample_count": 0,
            "atol": atol,
            "rtol": rtol,
            "max_abs_error": None,
            "max_rel_error": None,
            "cases": [],
        }
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        export_report_path = package_dir / "export_report.json"
        export_report = json.loads(export_report_path.read_text(encoding="utf-8"))
        missing = [
            value
            for value in export_report.get("missing_dependencies", [])
            if value != "onnx"
        ]
        if "onnxruntime" not in missing:
            missing.insert(0, "onnxruntime")
        export_report["missing_dependencies"] = missing
        export_report["onnxruntime_parity"] = {
            "path": report_path.name,
            "status": "blocked",
            "missing_dependency": "onnxruntime",
        }
        export_report["status"] = "blocked"
        export_report["note"] = message
        export_report_path.write_text(json.dumps(export_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        manifest_missing = [
            value
            for value in manifest.get("missing_required_dependencies", [])
            if value != "onnx"
        ]
        if "onnxruntime" not in manifest_missing:
            manifest_missing.insert(0, "onnxruntime")
        manifest["missing_required_dependencies"] = manifest_missing
        manifest["status"] = "blocked"
        manifest["toolchain_status"] = "blocked"
        (package_dir / "deployment_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return report
    import onnxruntime as ort

    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX model is missing: {onnx_path}")
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    if [value.name for value in session.get_inputs()] != ["input"] or [value.name for value in session.get_outputs()] != ["logits"]:
        raise ValueError("ONNX input/output names differ from the frozen contract")
    corpus_dir = package_dir / "golden_corpus"
    samples: list[tuple[str, np.ndarray, np.ndarray]] = []
    for path in sorted(corpus_dir.glob("valid_*_60.npz")):
        data = np.load(path, allow_pickle=False)
        samples.append((path.name, data["expected_tensor"], data["expected_logits"]))
    cadence = np.load(corpus_dir / "cadence_real_90_records.npz", allow_pickle=False)
    for index, tensor in enumerate(cadence["expected_tensors"]):
        samples.append((f"cadence_real_90_records.npz[{index}]", tensor, cadence["expected_logits"][index]))
    results: list[dict[str, Any]] = []
    max_abs = 0.0
    max_rel = 0.0
    for case_id, tensor, expected_logits in samples:
        actual = session.run(["logits"], {"input": tensor.astype(np.float32)})[0][0].astype(np.float32)
        absolute = float(np.max(np.abs(actual - expected_logits)))
        relative = _max_relative(actual, expected_logits)
        max_abs = max(max_abs, absolute)
        max_rel = max(max_rel, relative)
        results.append(
            {
                "case": case_id,
                "max_abs_error": absolute,
                "max_rel_error": relative,
                "expected_class": int(np.argmax(expected_logits)),
                "actual_class": int(np.argmax(actual)),
            }
        )
    passed = all(
        np.allclose(
            session.run(["logits"], {"input": tensor.astype(np.float32)})[0][0].astype(np.float32),
            expected_logits,
            rtol=rtol,
            atol=atol,
        )
        for _, tensor, expected_logits in samples
    )
    report = {
        "status": "pass" if passed else "fail",
        "source_checkpoint_sha256": manifest["source_checkpoint"]["sha256"],
        "onnx_sha256": _sha256(onnx_path),
        "sample_count": len(samples),
        "atol": atol,
        "rtol": rtol,
        "max_abs_error": max_abs,
        "max_rel_error": max_rel,
        "cases": results,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    export_report_path = package_dir / "export_report.json"
    export_report = json.loads(export_report_path.read_text(encoding="utf-8"))
    export_report["onnxruntime_parity"] = {
        "path": report_path.name,
        "status": report["status"],
        "max_abs_error": max_abs,
        "max_rel_error": max_rel,
        "onnxruntime_version": ort.__version__,
    }
    export_report["missing_dependencies"] = [
        value
        for value in export_report.get("missing_dependencies", [])
        if value not in {"onnx", "onnxruntime"}
    ]
    st_toolchain_missing = any(
        "ST Edge AI Core / STM32Cube.AI CLI" in value
        for value in export_report["missing_dependencies"]
    )
    if passed:
        export_report["status"] = "blocked" if st_toolchain_missing else "pending_st_edge_ai_analysis"
        export_report["note"] = (
            "ONNX Runtime parity passed, but ST Edge AI Core / STM32Cube.AI CLI is missing; compiler analysis is blocked."
            if st_toolchain_missing
            else "ONNX Runtime parity passed. ST Edge AI compiler analysis remains required."
        )
    else:
        export_report["status"] = "parity_fail"
        export_report["note"] = "ONNX Runtime parity failed; do not continue to compiler or int8 evaluation."
    export_report_path.write_text(json.dumps(export_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    manifest["missing_required_dependencies"] = [
        value
        for value in manifest.get("missing_required_dependencies", [])
        if value not in {"onnx", "onnxruntime"}
    ]
    manifest["host_float_onnx_parity"] = {
        "path": report_path.name,
        "status": report["status"],
        "max_abs_error": max_abs,
        "max_rel_error": max_rel,
        "onnxruntime_version": ort.__version__,
    }
    if passed:
        manifest["status"] = "blocked" if manifest["missing_required_dependencies"] else "host_float_onnx_parity_passed"
        manifest["toolchain_status"] = "blocked" if manifest["missing_required_dependencies"] else "ready"
    else:
        manifest["status"] = "parity_fail"
        manifest["toolchain_status"] = "parity_fail"
    (package_dir / "deployment_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    args = parser.parse_args()
    report = verify(args.package_dir, atol=args.atol, rtol=args.rtol)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
