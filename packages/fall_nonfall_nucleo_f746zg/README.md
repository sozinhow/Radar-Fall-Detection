# NUCLEO-F746ZG — Fall / Non-fall

Package name: `fall_nonfall_nucleo_f746zg`
Release: frozen binary Version 1

This folder is the frozen, shareable Version 1 package for the
NUCLEO-F746ZG deployment. It contains the accepted two-class detector:

- classes: `NON_FALL`, `FALL`
- input: `60 x 7` at `20 Hz`
- firmware: `firmware/NUCLEO_F7_AI/`
- flashable image: `firmware/NUCLEO_F7_AI/Debug/NUCLEO_F7_AI.elf`
- source checkpoint: `training/fold4/cnn_temporal_event_centered.pt`
- host/Edge AI export: `deployment/binary/cnn_temporal_fold4_nonfall_fall.onnx`

The binary model is the fold-4 four-class checkpoint grouped into
`NON_FALL = walking + standing + sitting` and `FALL = fall`. The firmware
already includes the live fall policy and debug telemetry; do not replace its
generated AI files with the ONNX file directly.

## Board and serial wiring

- Board: `NUCLEO-F746ZG`, STM32F746ZGTx
- Radar RX into MCU: `PG9 / USART6_RX` (CN10 pin 16 / D0)
- Radar TX from MCU: `PG14 / USART6_TX` (CN10 pin 14 / D1)
- USB VCP output: `USART3` on the ST-LINK USB connection
- UART: `115200 8N1`

The runtime policy is kept in the firmware source at
`firmware/NUCLEO_F7_AI/Core/Src/radar_live.c`. Important V1 gates are the
0.65 fall-probability threshold, two consecutive windows, minimum 0.30 m
height drop, post-fall Z at or below -0.70 m, and a 5-second latch. Edge-of-FOV
and too-close results are handled before fall classification.

## Use

Import `firmware/NUCLEO_F7_AI` into STM32CubeIDE, or flash the included ELF
with STM32CubeProgrammer/ST-LINK. Monitor the USB VCP at `115200 8N1`.

Keep future model families in separate packages. For the four-class model,
use `packages/fall_4class_nucleo_f746zg/`. Do not modify this binary package
when adding classes or new features.
