import hashlib
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "packages" / "fall_4class_nucleo_f746zg"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_four_class_package_manifest_and_model_are_consistent():
    manifest = json.loads(
        (PACKAGE_ROOT / "package_manifest.json").read_text(encoding="utf-8")
    )
    model = manifest["model"]
    onnx_path = PACKAGE_ROOT / model["onnx"]["path"]
    checkpoint_path = PACKAGE_ROOT / model["checkpoint"]["path"]

    assert manifest["package"] == "fall_4class_nucleo_f746zg"
    assert model["class_order"] == ["walking", "standing", "sitting", "fall"]
    assert model["input"]["shape"] == [1, 60, 7]
    assert model["output"]["shape"] == [1, 4]
    assert onnx_path.is_file()
    assert checkpoint_path.is_file()
    assert _sha256(onnx_path) == model["onnx"]["sha256"]
    assert _sha256(checkpoint_path) == model["checkpoint"]["sha256"]


def test_four_class_generated_ai_contract_is_present():
    firmware_root = PACKAGE_ROOT / "firmware" / "NUCLEO_F7_AI_four_class"
    network_header = (firmware_root / "AI" / "App" / "network.h").read_text(
        encoding="utf-8"
    )
    codegen_report = (
        PACKAGE_ROOT
        / "deployment"
        / "four_class"
        / "edge_ai_codegen"
        / "network_generate_report.txt"
    ).read_text(encoding="utf-8")

    assert 'STAI_NETWORK_ORIGIN_MODEL_NAME         "cnn_temporal_fold3_four_class"' in network_header
    assert "#define STAI_NETWORK_OUT_1_SHAPE       {1,4}" in network_header
    assert "#define STAI_NETWORK_ACTIVATIONS_SIZE_BYTES        (11776)" in network_header
    assert "#define STAI_NETWORK_WEIGHTS_SIZE_BYTES            (283792)" in network_header
    assert "target/series      :   stm32f4" in codegen_report
    assert (
        firmware_root
        / "Middlewares"
        / "ST"
        / "AI"
        / "Lib"
        / "NetworkRuntime1201_CM7_GCC.a"
    ).is_file()


def test_four_class_firmware_uses_four_class_normalization():
    normalization = json.loads(
        (
            PACKAGE_ROOT
            / "deployment"
            / "four_class"
            / "normalization.json"
        ).read_text(encoding="utf-8")
    )
    radar_live = (
        PACKAGE_ROOT
        / "firmware"
        / "NUCLEO_F7_AI_four_class"
        / "Core"
        / "Src"
        / "radar_live.c"
    ).read_text(encoding="utf-8")

    for value in normalization["mean"] + normalization["std"]:
        assert f"{value}f" in radar_live


def test_four_class_firmware_project_and_outputs_are_separate_from_v1():
    firmware_root = PACKAGE_ROOT / "firmware" / "NUCLEO_F7_AI_four_class"
    manifest = json.loads(
        (PACKAGE_ROOT / "package_manifest.json").read_text(encoding="utf-8")
    )
    launch = (firmware_root / "NUCLEO_F7_AI_four_class.launch").read_text(
        encoding="utf-8"
    )
    makefile = (firmware_root / "Debug" / "makefile").read_text(encoding="utf-8")

    assert (firmware_root / ".project").read_text(encoding="utf-8").find(
        "<name>NUCLEO_F7_AI_four_class</name>"
    ) >= 0
    assert (firmware_root / "NUCLEO_F7_AI_four_class.ioc").is_file()
    assert "Debug/NUCLEO_F7_AI_four_class.elf" in launch
    assert "fProjectName&quot;:&quot;NUCLEO_F7_AI_four_class" in launch
    assert "BUILD_ARTIFACT_NAME := NUCLEO_F7_AI_four_class" in makefile
    assert "arm-none-eabi-objcopy -O binary" in makefile
    assert manifest["firmware"]["elf"].endswith(
        "NUCLEO_F7_AI_four_class/Debug/NUCLEO_F7_AI_four_class.elf"
    )
    assert manifest["firmware"]["bin"].endswith(
        "NUCLEO_F7_AI_four_class/Debug/NUCLEO_F7_AI_four_class.bin"
    )
    assert not (firmware_root / "Debug" / "NUCLEO_F7_AI.elf").exists()
