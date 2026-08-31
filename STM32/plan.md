# Radar STM32 execution plan

Updated: 2026-08-19 HKT

Paths below are relative to the GitHub repository root: `radar/`.

## Completed — Version 1

- Built and flashed the NUCLEO-F746ZG F7 project.
- Confirmed the embedded contract is 60 x 7 input at 20 Hz with classes
  NON_FALL and FALL.
- Confirmed the radar wiring: USART6 PG9/RX and PG14/TX; USB VCP diagnostics
  use USART3.
- Added runtime diagnostics for raw AI probability, streak, target/cluster,
  geometry gates, and guard states.
- Tested genuine Falls, standing, sitting, walking/fast walking, and leaving
  the field of view. Only final RESULT=FALL is scored as a Fall.
- Created the frozen hand-off package:

      packages/fall_nonfall_nucleo_f746zg/

- Updated the radar runbook:

      pipeline_operation.md

## Current V1 source of truth

| Responsibility | Path |
|---|---|
| Fall thresholds/state machine | `STM32/NUCLEO_F7_AI/Core/Src/main.c` |
| Radar parser, 20 Hz sampling, normalization, geometry | `STM32/NUCLEO_F7_AI/Core/Src/radar_live.c` |
| Public radar diagnostics | `STM32/NUCLEO_F7_AI/Core/Inc/radar_live.h` |
| Generated AI network | `STM32/NUCLEO_F7_AI/AI/App/` |
| MCU pin/peripheral configuration | `STM32/NUCLEO_F7_AI/NUCLEO_F7_AI.ioc` |
| Flashable image | `STM32/NUCLEO_F7_AI/Debug/NUCLEO_F7_AI.elf` |
| Collection/conversion tools | `STM32/n6570dk_blink/tools/` |
| Current status | `STM32/checkpoint.md` |

## Locked V1 runtime policy

    fall probability >= 0.65
    two consecutive raw-Fall windows
    height drop >= 0.30 m
    post-event Z <= -0.70 m
    downward fraction >= 0.52
    post-event median motion <= 0.16 m
    Candidate timeout 3 s
    minimum confirmed-Fall latch 5 s
    clear windows 3
    target stale timeout 250 ms
    alert azimuth -50° to +50°

Radar sampling is 50 ms (20 Hz), with a 60-sample input window and inference
stride of 10 new samples. Runtime range is 0.5 m to 6.0 m. The current model
uses SENSOR_NATIVE Z.

## Next test after any firmware rebuild

1. Reflash the latest ELF.
2. Find the current VCP with ls /dev/cu.*.
3. Open it at 115200 8N1 using a Python environment with `pyserial`:
   `python3 -m serial.tools.miniterm <PORT> 115200`.
4. Confirm the startup line says input=60x7, sample_rate=20Hz, and
   z_mode=SENSOR_NATIVE.
5. Test standing, sitting, normal/fast walking, a genuine Fall, and leaving
   the detection area.
6. Count only RESULT=FALL as a confirmed Fall; keep CANDIDATE as diagnostic
   evidence.

## Version 2 — four-class deployment candidate

Version 2 source integration is now isolated at
`packages/fall_4class_nucleo_f746zg/`, outside the frozen package. The current
candidate uses the fold-3 checkpoint and class order
`walking, standing, sitting, fall`. Its input remains 60 x 7 at 20 Hz, while
the generated network output is four logits.

The external ST Edge AI C-code export is kept at `Project/F7_Model2/` and has
been integrated only into the new package. The four-class STM32CubeIDE project
is `firmware/NUCLEO_F7_AI_four_class/`, with separate outputs:

      packages/fall_4class_nucleo_f746zg/firmware/NUCLEO_F7_AI_four_class/Debug/NUCLEO_F7_AI_four_class.elf
      packages/fall_4class_nucleo_f746zg/firmware/NUCLEO_F7_AI_four_class/Debug/NUCLEO_F7_AI_four_class.bin

The next gates are to build those outputs with STM32CubeIDE/ARM GCC, flash the
four-class ELF to NUCLEO-F746ZG, and run the four-class live test. Do not edit
or regenerate the V1 package in place.
