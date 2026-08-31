# NUCLEO-F746ZG — Four-class radar activity classifier

Package name: `fall_4class_nucleo_f746zg`
Release: four-class deployment candidate, generated 2026-08-19

This is a new package. It does not modify the frozen
`fall_nonfall_nucleo_f746zg` Version 1 package.

## Model contract

- Classes, in model-output order: `WALKING`, `STANDING`, `SITTING`, `FALL`
- Input: normalized `float32 [1,60,7]` at 20 Hz
- Output: four `float32` logits, `logits [1,4]`
- Source checkpoint: `training/fold3/cnn_temporal_event_centered.pt`
- Host ONNX: `deployment/four_class/cnn_temporal_fold3_four_class.onnx`
- ONNX SHA-256: `632ef26fc80294e20bc690714ead34bc1df57001e8cac2e4a8ce7275bed691ac`

The firmware selects the argmax class and reports a softmax probability for
each class. It does not use the Version 1 binary fall threshold, candidate
state, fall latch, or binary grouping policy.

## Firmware

Firmware source: `firmware/NUCLEO_F7_AI_four_class/`

This is a separate STM32CubeIDE project named `NUCLEO_F7_AI_four_class`; it
must not be imported or built as the V1 project named `NUCLEO_F7_AI`.

The generated ST Edge AI C package from external `Project/F7_Model2` is
integrated into `AI/App/` and `Middlewares/ST/AI/`. The generated network
uses the STM32F746ZG Cortex-M7 runtime and has these resource figures:

- weights: 283,792 bytes
- activations: 11,776 bytes
- MACC: 1,075,940
- runtime library: `NetworkRuntime1201_CM7_GCC.a`

The copied project keeps the NUCLEO-F746ZG board configuration and radar
USART6 / USB VCP wiring from the Version 1 firmware as the hardware base.
The model and live classification behavior are four-class specific.

The included `Debug/` build files are source-project build metadata. Build
outputs are intentionally four-class-specific:

- ELF: `firmware/NUCLEO_F7_AI_four_class/Debug/NUCLEO_F7_AI_four_class.elf`
- BIN: `firmware/NUCLEO_F7_AI_four_class/Debug/NUCLEO_F7_AI_four_class.bin`

The new ELF/BIN must be built with STM32CubeIDE or an STM32 ARM GCC
toolchain; the Version 1 ELF is not reused or overwritten.

## Board and serial wiring

- Board: `NUCLEO-F746ZG`, STM32F746ZGTx
- Radar RX into MCU: `PG9 / USART6_RX` (CN10 pin 16 / D0)
- Radar TX from MCU: `PG14 / USART6_TX` (CN10 pin 14 / D1)
- USB VCP output: `USART3` on the ST-LINK USB connection
- UART: `115200 8N1`

## Runtime output

After initialization, the firmware reports a four-class self-test. Each full
window reports one of `WALKING`, `STANDING`, `SITTING`, or `FALL`, together
with four class probabilities. `NO_TARGET`, range guards, and
`EDGE_OF_FOV` are diagnostic states and are not model classes.

## Verification

From the `radar/` repository root:

```bash
python -m radar_pipeline.verify_edge_package_onnx_parity \
  --package-dir packages/fall_4class_nucleo_f746zg/deployment/four_class
```

The host ONNX parity report and deployment manifest are kept beside the ONNX
file. The final package is not considered board-deployable until the new
firmware is compiled, flashed, and live-tested on NUCLEO-F746ZG.
