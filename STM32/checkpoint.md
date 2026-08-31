# Radar STM32 Checkpoint

Updated: 2026-08-19

Paths below are relative to the GitHub repository root: `radar/`.

## Current status

- Board: **NUCLEO-F746ZG** / STM32F746ZGTX.
- Active firmware project:
  `STM32/NUCLEO_F7_AI/`
- Flashable ELF:
  `STM32/NUCLEO_F7_AI/Debug/NUCLEO_F7_AI.elf`
- Latest clean build: **0 errors, 0 warnings**.
- Current live result:
  - genuine Fall → `RESULT=FALL`
  - walking/fast running → `RESULT=CANDIDATE`, not promoted to `FALL`
- No AI Studio regeneration is needed for Version 1 threshold changes. The
  current four-class candidate has its own generated network and package.

## Current AI contract

- Embedded model: `cnn_temporal_fold4_nonfall_fall`.
- Input: `60 x 7` float32 samples at **20 Hz**.
- Output classes: `[NON_FALL, FALL]`.
- Z convention: `SENSOR_NATIVE`.
- Generated AI files are in:
  `STM32/NUCLEO_F7_AI/AI/App/`
- Model wrapper:
  `STM32/NUCLEO_F7_AI/Core/Src/ai_model.c`
- Radar parsing, normalization, sampling, and geometry diagnostics:
  `STM32/NUCLEO_F7_AI/Core/Src/radar_live.c`

## Four-class deployment candidate

The new package is:

`packages/fall_4class_nucleo_f746zg/`

- Checkpoint: `training/fold3/cnn_temporal_event_centered.pt`.
- ONNX: `deployment/four_class/cnn_temporal_fold3_four_class.onnx`.
- Class order: `walking, standing, sitting, fall`.
- Input/output: normalized float32 `[1,60,7]` to float32 logits `[1,4]`.
- Firmware normalization: the fold-3 values in
  `deployment/four_class/normalization.json` (not the V1 fold-4 values).
- Generated C code provenance: external `Project/F7_Model2/`.
- Generated runtime: `NetworkRuntime1201_CM7_GCC.a`.
- Separate STM32CubeIDE project:
  `packages/fall_4class_nucleo_f746zg/firmware/NUCLEO_F7_AI_four_class/`.
- Pending four-class ELF/BIN:
  `.../Debug/NUCLEO_F7_AI_four_class.elf` and
  `.../Debug/NUCLEO_F7_AI_four_class.bin`.

The four-class firmware reports argmax class plus softmax probabilities. It
does not use the Version 1 binary fall threshold or state machine. Its ELF
build and NUCLEO-F746ZG live test are pending.

## Active thresholds

All Fall state-machine thresholds are in:

`STM32/NUCLEO_F7_AI/Core/Src/main.c`

| Setting | Value |
|---|---:|
| Raw AI Fall probability | `>= 65%` |
| Consecutive raw-Fall windows | `2` |
| Minimum height drop | `>= 30 cm` |
| Maximum post-event Z | `<= -70 cm` |
| Minimum downward fraction | `>= 52%` |
| Maximum post-event median motion | `<= 16 cm` |
| Candidate timeout | `3 s` |
| Minimum Fall latch | `5 s` |
| Clear windows after latch | `3` |
| Target stale timeout | `250 ms` |
| Alert azimuth range | `-50° to +50°` |

Radar range/sampling settings are in:

`STM32/NUCLEO_F7_AI/Core/Src/radar_live.c`

- Sampling: `50 ms` = 20 Hz.
- Inference window: 60 samples.
- Inference stride: 10 new samples.
- Supported range: `0.5 m` to `6.0 m`.

## Meaning of `CANDIDATE`

`CANDIDATE` is not a confirmed Fall alarm.

1. AI outputs `p_fall >= 65%` for two consecutive windows.
2. Firmware enters `CANDIDATE`.
3. It becomes `FALL` only when all four geometry conditions pass together:
   height drop, post-event Z, downward fraction, and post-event motion.
4. If geometry does not pass, Candidate expires after 3 seconds.

The current code may keep printing `CANDIDATE` briefly after `raw=NON_FALL` while the 3-second pending state expires. This is only diagnostic output and did not become `RESULT=FALL` in the walking/fast-running test.

## Changes already made

### `main.c`

- Added two-class Fall confirmation state machine.
- Added serial diagnostics: raw probability, streak, target count, cluster ID, drop, post-Z, downward percentage, motion, azimuth, and XYZ.
- Added fail-closed handling for stale target, too close, too far, and edge-of-FOV.
- Changed downward threshold from **53% to 52%** because a genuine Fall reached `down_pct=52`.
- Changed post-motion threshold from **10 cm to 16 cm** because printed `motion_cm=15` may be rounded down.

### `radar_live.c` / `radar_live.h`

- Parse the full HLK-LD6002B `0x0A04` target list.
- Maintain target cluster continuity.
- Reset the 60-sample window on zero targets or stale target data.
- Keep native Z and the frozen seven-feature normalization.
- Report target switches, lost targets, invalid frames, gaps, and dropped bytes.

## UART and hardware

- Radar → STM32 USART6:
  - PG9 = USART6_RX, CN10 pin 16 / D0.
  - PG14 = USART6_TX, CN10 pin 14 / D1.
- Wiring at the UART boundary: radar TX → PG9/RX; radar RX → PG14/TX; common GND.
- STM32 USB VCP diagnostic output: USART3 on PD8/PD9.
- Serial format: `115200 8N1`.
- USB port changes after reconnect. Discover it with:
  `ls /dev/cu.*`
- The USB serial port is machine- and connection-specific. Discover it with
  `ls /dev/cu.*` on macOS, or `python3 -m serial.tools.list_ports` on other
  systems.
- Generic command:
  `python3 -m serial.tools.miniterm <PORT> 115200`

## Which file to change

- Thresholds / Fall decision: `NUCLEO_F7_AI/Core/Src/main.c`.
- Radar parser / 20 Hz / normalization / geometry: `NUCLEO_F7_AI/Core/Src/radar_live.c`.
- Public radar diagnostics: `NUCLEO_F7_AI/Core/Inc/radar_live.h`.
- AI wrapper: `NUCLEO_F7_AI/Core/Src/ai_model.c` and `Core/Inc/ai_model.h`.
- AI model weights/architecture: regenerate the AI network, then replace the generated files under `NUCLEO_F7_AI/AI/App/`. Do this only when changing the model; threshold edits do not require AI Studio.
- MCU pins/UART: edit `NUCLEO_F7_AI/NUCLEO_F7_AI.ioc` in CubeMX/CubeIDE and regenerate.
- Host collection tools are separate from the flashed firmware:
  `STM32/n6570dk_blink/tools/`

## Next steps

1. Reflash the latest `NUCLEO_F7_AI.elf`.
2. Reopen the current USB VCP port at 115200 8N1.
3. Test standing/sitting, walking, fast walking/running, genuine Fall, and leaving the detection range.
4. Count only `RESULT=FALL` as a final false alarm; `CANDIDATE` is pending diagnostic state.
5. If walking creates formal `RESULT=FALL`, retune thresholds or add fast-walking negatives to model training. Do not regenerate AI Studio output for a threshold-only change.

## Frozen Version 1 hand-off

The shareable V1 package is:

`packages/fall_nonfall_nucleo_f746zg/`

It contains the F7 CubeIDE project and ELF, the fold-4 training checkpoint,
and the validated two-class ONNX package. Version 2 must be copied into a new
folder at `packages/fall_4class_nucleo_f746zg/` and must not edit this package
in place.

That four-class package now contains the fold-3 model and generated C code;
it remains a deployment candidate until a new ELF is built and live-tested.

The host GUI still uses its legacy diagnostic four-class compatibility
artifacts at
`outputs/deployment/host_demo_20260728_run01/`.
Its source checkpoint is retained because `radar_collection_gui.py` and its
host inference tests depend on that compatibility contract; it is not a
shareable package and is not the flashed NUCLEO-F746ZG model. Any future
four-class shareable package must use
`packages/fall_4class_nucleo_f746zg/`.
