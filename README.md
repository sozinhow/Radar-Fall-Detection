# Radar Fall Detection Project

This folder is the single GitHub repository root for the accepted Version 1
detector and the isolated four-class deployment candidate on NUCLEO-F746ZG.
Paths in this repository are
relative to `radar/`; Project-level historical material, the
STM32F746G-DISCO / AI Studio reference, and the external `F7_Model2` code
export are intentionally outside this root.

## Layout

```text
radar/
├── STM32/                                 # active F746ZG firmware and capture tools
├── packages/
│   ├── fall_nonfall_nucleo_f746zg/        # frozen Version 1 package
│   └── fall_4class_nucleo_f746zg/         # four-class deployment candidate
├── radar_pipeline/                        # cleaning, training, export, parity tools
├── data/                                  # retained raw and processed data
├── outputs/                               # evaluation and provenance artifacts
├── fold_summary/                          # retained training checkpoint/metrics
├── tests/                                 # project regression tests
├── requirements.txt                       # canonical Python dependencies
├── README.md
├── pipeline_operation.md
├── project_status.md
├── docs/                                  # supporting contracts and notes
└── conftest.py
```

## V1 contract

- Board: NUCLEO-F746ZG / STM32F746ZGTX.
- Input: 60 × 7 samples at 20 Hz.
- Classes: `NON_FALL`, `FALL`.
- Radar UART: USART6, PG9 = RX and PG14 = TX.
- Diagnostic UART: USB VCP USART3, 115200 8N1.
- Count only final `RESULT=FALL` as a confirmed Fall. `CANDIDATE` is pending
  diagnostic output.

The frozen binary hand-off package must not be modified for a four-class
version. Any future four-class shareable package may only be created at
`packages/fall_4class_nucleo_f746zg/`, with its own data, model, firmware,
deployment artifacts, and test run IDs.

## Four-class deployment candidate

The current four-class candidate uses the fold-3 model with class order
`walking, standing, sitting, fall`:

```text
packages/fall_4class_nucleo_f746zg/
├── training/fold3/
├── deployment/four_class/
├── firmware/NUCLEO_F7_AI_four_class/
└── package_manifest.json
```

The external ST Edge AI C-code export was staged at
`Project/F7_Model2/` and integrated only into this new package. A new ELF and
board live test are still required before calling this package deployable.
The four-class project and outputs are distinct from V1:

```text
V1 ELF:       STM32/NUCLEO_F7_AI/Debug/NUCLEO_F7_AI.elf
Four-class:   packages/fall_4class_nucleo_f746zg/firmware/NUCLEO_F7_AI_four_class/Debug/NUCLEO_F7_AI_four_class.elf
Four-class:   packages/fall_4class_nucleo_f746zg/firmware/NUCLEO_F7_AI_four_class/Debug/NUCLEO_F7_AI_four_class.bin
```

## Verification from this folder

```bash
cd /path/to/radar
python -m pip install -r requirements.txt
python -m pytest -q
python -m compileall -q radar_pipeline
python -m unittest discover -s STM32/n6570dk_blink/tools -p 'test_*.py'
python -m radar_pipeline.verify_edge_package_onnx_parity \
  --package-dir packages/fall_nonfall_nucleo_f746zg/deployment/binary
python -m radar_pipeline.verify_edge_package_onnx_parity \
  --package-dir packages/fall_4class_nucleo_f746zg/deployment/four_class
```

The full operational sequence is documented in `pipeline_operation.md`.
