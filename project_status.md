# Radar project status

Updated: 2026-08-19 HKT

Paths below are relative to the GitHub repository root: `radar/`.

## Version 1 status

Version 1 is accepted for the current test scope: Fall / Non-fall detection
on a NUCLEO-F746ZG. The owner tested genuine Falls, standing, sitting,
walking/fast walking, and leaving the detection area. Final Fall results,
non-Fall results, and edge-of-FOV guard results are behaving as intended.
Walking may still print a short CANDIDATE diagnostic state; only
RESULT=FALL is a confirmed Fall.

The frozen shareable package is:

    packages/fall_nonfall_nucleo_f746zg/

It contains the F7 CubeIDE project and ELF, the fold-4 checkpoint, and the
validated binary ONNX package.

## Model and deployment contract

- Board: NUCLEO-F746ZG / STM32F746ZGTX.
- Firmware project:
  STM32/NUCLEO_F7_AI/
- Flashable ELF:
  STM32/NUCLEO_F7_AI/Debug/NUCLEO_F7_AI.elf
- Input: 60 samples x 7 features at 20 Hz.
- Feature order: x, y, z, dop_idx, range_m, azimuth_deg, elevation_deg.
- Deployment classes: NON_FALL, FALL.
- Canonical Version 1 checkpoint:
  packages/fall_nonfall_nucleo_f746zg/training/fold4/cnn_temporal_event_centered.pt
- Canonical Version 1 ONNX package:
  packages/fall_nonfall_nucleo_f746zg/deployment/binary/
- ONNX input is already-normalized float32 [1, 60, 7].
- ONNX/PyTorch parity: pass.
- ST Edge AI compiler analysis: unavailable on this machine; no compiler
  result is claimed.

The binary model is a log-sum-exp adapter over the original four-class
checkpoint. It was not separately retrained as a binary model. The actual
on-device firmware model is the generated AI network in
STM32/NUCLEO_F7_AI/AI/App/.

## Four-class deployment candidate

The new package is isolated from Version 1 at:

    packages/fall_4class_nucleo_f746zg/

Its contract is:

- checkpoint: `training/fold3/cnn_temporal_event_centered.pt`
- ONNX: `deployment/four_class/cnn_temporal_fold3_four_class.onnx`
- classes: `walking, standing, sitting, fall`
- input: normalized float32 `[1, 60, 7]`
- output: float32 logits `[1, 4]`
- firmware normalization: `deployment/four_class/normalization.json`
- generated C code source: external `Project/F7_Model2/`
- runtime: `NetworkRuntime1201_CM7_GCC.a`
- weights: 283,792 bytes; activations: 11,776 bytes
- separate STM32CubeIDE project:
  `packages/fall_4class_nucleo_f746zg/firmware/NUCLEO_F7_AI_four_class/`
- pending outputs:
  `.../Debug/NUCLEO_F7_AI_four_class.elf` and
  `.../Debug/NUCLEO_F7_AI_four_class.bin`

The new firmware uses argmax plus softmax probabilities and does not reuse
the V1 binary threshold, candidate state, or fall latch. The package source
and generated network files are present; the new ELF build and board live
test remain pending.

## Training provenance retained

The current reproducible lineage is:

    data/raw_csv/
      -> data/staging/20260731_rebuild01/
      -> data/final_dataset_auto_event_exclude_risky_fall_20260731_rebuild01/
      -> outputs/event_centered_clips_sgkf4_20260731_rebuild01/
      -> fold_summary/fold_4/
      -> outputs/deployment/cnn_temporal_dual_onnx_20260814_run01/binary/

The dataset and split artifacts are source-session grouped. Training
normalization is fit on outer-fold training sessions only. Raw CSV source
data, current staging/final dataset, current event-centred evaluation, fold
summary, and current deployment package are retained.

## Firmware and UART

- Radar USART6: PG9 = RX (CN10 pin 16 / D0), PG14 = TX (CN10 pin 14 / D1).
- USB VCP diagnostic output: USART3.
- Serial format: 115200 8N1.
- Radar sampling: 50 ms / 20 Hz.
- Inference window: 60 samples; stride: 10 samples.
- Runtime target range: 0.5 m to 6.0 m.
- USB VCP port is machine- and connection-specific; discover it with
  `ls /dev/cu.*` on macOS or `python3 -m serial.tools.list_ports` elsewhere.

Fall confirmation thresholds are kept in the STM32 project and recorded in
STM32/checkpoint.md. Threshold changes are firmware post-processing
changes; AI Studio regeneration is not required unless model weights or model
shape change.

## Verification

The cleanup verification passed:

    43 passed in radar/tests, plus 27 passed in the STM32 host-tool tests.

Run it from the `radar/` repository root:

    python -m pytest -q
    python -m compileall -q radar_pipeline
    python -m unittest discover -s STM32/n6570dk_blink/tools -p 'test_*.py'
    python -m radar_pipeline.verify_edge_package_onnx_parity \
      --package-dir packages/fall_nonfall_nucleo_f746zg/deployment/binary
    python -m radar_pipeline.verify_edge_package_onnx_parity \
      --package-dir packages/fall_4class_nucleo_f746zg/deployment/four_class

The host collection tools remain under
STM32/n6570dk_blink/tools/. They are retained because they are needed to
collect/convert radar data, but the obsolete N6570 firmware project and old
deployment experiments are not part of Version 1.

## Version 2 boundary

Version 2 four-class source integration has started only in
`packages/fall_4class_nucleo_f746zg/`. The frozen Version 1 package remains
unchanged. Every four-class checkpoint, ONNX file, generated AI artifact,
firmware source, firmware image, and test report must remain under the new
package or its explicitly recorded output path. The four-class ELF build,
flash, and live test are still required before release.
