# Radar pipeline operation

Updated: 2026-08-19 HKT

Unless noted otherwise, paths below are relative to the GitHub repository root:
`radar/`.

This is the runnable runbook for the accepted Version 1 lineage. Version 1
is the deployed two-class detector (NON_FALL, FALL) on the NUCLEO-F746ZG.
Version 2 work must use the dedicated package folder
`packages/fall_4class_nucleo_f746zg/` and new run IDs; do not overwrite these
artifacts.

## Current contracts

- Raw source: data/raw_csv/{fall,sitting,standing,walking}/.
- Sampling: 20 Hz.
- Model window: 60 frames / 3 seconds.
- Model features, in order: x, y, z, dop_idx, range_m, azimuth_deg,
  elevation_deg.
- Model tensor: normalized float32 [1, 60, 7].
- Training class order: walking, standing, sitting, fall.
- Deployment class order: NON_FALL, FALL, where NON_FALL groups the three
  non-fall training classes.
- Splits are source-session grouped. Never split individual windows randomly.

The event-centred implementation is now stored at
radar_pipeline/reference_event_centered_runner.py; the pipeline no longer
depends on an old experiment-output directory.

## Environment and fast verification

From the `radar/` repository root:

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python -m compileall -q radar_pipeline
```

The cleanup verification on 2026-08-19 passed 43 tests. These commands do
not need a board or radar connected.

## Rebuild data and training artifacts

Use one new RUN_ID consistently. Replace the example ID before running so
that a completed run is never overwritten.

```bash
cd radar
RUN_ID=20260819_rebuild01
MPLCONFIGDIR=/tmp/radar_mpl_$RUN_ID
mkdir -p "$MPLCONFIGDIR"

# 1. Inspect the raw source without changing it.
python -m radar_pipeline.load_and_inspect \
  --input-dir data/raw_csv \
  --output-dir outputs/validation/before_cleaning_$RUN_ID

# 2. Clean into a new staging directory.
python -m radar_pipeline.clean_data \
  --input-dir data/raw_csv \
  --output-dir data/staging/$RUN_ID/cleaned_csv \
  --config radar_pipeline/config.yaml

# 3. Build fall-event annotations from that staging directory.
MPLCONFIGDIR="$MPLCONFIGDIR" python -m radar_pipeline.auto_event_annotations \
  --cleaned-fall-dir data/staging/$RUN_ID/cleaned_csv \
  --raw-fall-dir data/raw_csv/fall \
  --annotations-csv data/metadata/auto_event_annotations_$RUN_ID.csv \
  --plot-dir outputs/validation/auto_event_annotation_$RUN_ID \
  --index-csv outputs/validation/auto_event_annotation_$RUN_ID/fall/index.csv \
  --report outputs/auto_event_annotation_report_$RUN_ID.md \
  --config radar_pipeline/config.yaml \
  --cleaning-log data/staging/$RUN_ID/cleaning_log.json

# 4. Exclude only documented risky auto-fall windows and freeze SGKF4.
python -m radar_pipeline.build_risky_fall_excluded_dataset \
  --cleaned-dir data/staging/$RUN_ID/cleaned_csv \
  --annotations-csv data/metadata/auto_event_annotations_$RUN_ID.csv \
  --windowed-output-dir data/windowed_auto_event_exclude_risky_fall_$RUN_ID \
  --dataset-output-dir data/final_dataset_auto_event_exclude_risky_fall_$RUN_ID \
  --config radar_pipeline/config_event_aware_auto_20260716.yaml \
  --manifest-output data/metadata/auto_event_exclude_risky_fall_$RUN_ID_source_session_folds.csv \
  --manifest-readme data/metadata/auto_event_exclude_risky_fall_$RUN_ID_source_session_folds_README.md \
  --folds 4 \
  --seed 42

# 5. Create a new event-centred 50/60-frame experiment directory.
MPLCONFIGDIR="$MPLCONFIGDIR" python -m radar_pipeline.run_event_centered_clips \
  --dataset data/final_dataset_auto_event_exclude_risky_fall_$RUN_ID/radar_dataset.npz \
  --manifest data/metadata/auto_event_exclude_risky_fall_$RUN_ID_source_session_folds.csv \
  --annotations data/metadata/auto_event_annotations_$RUN_ID.csv \
  --output-dir outputs/event_centered_clips_sgkf4_$RUN_ID \
  --derive-splits \
  --seed 42 \
  --val-folds 6
```

The final command writes both 50- and 60-frame branches. For the V1 family,
inspect the 60-frame branch and keep the checkpoint, metrics, manifest hash,
and normalization together. Do not call a staging run deployed until host
parity and live board tests pass.

## Accepted Version 1 artifacts

The current accepted source and deployment records are:

```text
data/staging/20260731_rebuild01/
data/final_dataset_auto_event_exclude_risky_fall_20260731_rebuild01/
data/metadata/auto_event_annotations_20260731_rebuild01.csv
data/metadata/auto_event_exclude_risky_fall_20260731_rebuild01_source_session_folds.csv
outputs/event_centered_clips_sgkf4_20260731_rebuild01/
fold_summary/fold_4/cnn_temporal_event_centered.pt
fold_summary/fold_4/metrics.json
outputs/deployment/cnn_temporal_dual_onnx_20260814_run01/binary/
```

The binary ONNX package passed ONNX checker and PyTorch/ONNX Runtime parity.
It is a binary adapter over the fold-4 four-class checkpoint, not a
separately retrained binary model. The compiler report is marked unavailable
when ST Edge AI is not installed; it must not be inferred from host parity.

Verify the frozen Version 1 package without regenerating it from `radar/`:

```bash
python -m radar_pipeline.verify_edge_package_onnx_parity \
  --package-dir packages/fall_nonfall_nucleo_f746zg/deployment/binary
```

## Four-class deployment candidate

The four-class candidate is isolated from Version 1 at:

```text
packages/fall_4class_nucleo_f746zg/
```

Its model contract is:

- checkpoint: `training/fold3/cnn_temporal_event_centered.pt`
- ONNX: `deployment/four_class/cnn_temporal_fold3_four_class.onnx`
- class order: `walking, standing, sitting, fall`
- input: normalized float32 `[1,60,7]`
- output: float32 logits `[1,4]`

The ONNX was generated and host-validated before the external ST Edge AI
code-generation step. The C-code export from `Project/F7_Model2/` is kept
outside the repository and integrated into the new package's
`firmware/NUCLEO_F7_AI_four_class/AI/App/` and
`Middlewares/ST/AI/` directories.

Verify the candidate package with:

```bash
python -m radar_pipeline.verify_edge_package_onnx_parity \
  --package-dir packages/fall_4class_nucleo_f746zg/deployment/four_class
```

The candidate firmware reports the four argmax classes and softmax
probabilities. It does not reuse Version 1's binary fall threshold, fall
candidate state, or fall latch. Its separate project is
`NUCLEO_F7_AI_four_class`, with pending outputs:

```text
packages/fall_4class_nucleo_f746zg/firmware/NUCLEO_F7_AI_four_class/Debug/NUCLEO_F7_AI_four_class.elf
packages/fall_4class_nucleo_f746zg/firmware/NUCLEO_F7_AI_four_class/Debug/NUCLEO_F7_AI_four_class.bin
```

The ELF build and NUCLEO-F746ZG live test are still pending.

## STM32 deployment and live test

The current board project is:

```text
STM32/NUCLEO_F7_AI/
```

Flash this image:

```text
STM32/NUCLEO_F7_AI/Debug/NUCLEO_F7_AI.elf
```

The firmware uses radar USART6 (PG9=RX, PG14=TX), USB VCP USART3 for
diagnostics, and 115200 8N1. The radar is receive-only from the host tools;
verify TX/RX crossing, common ground, and logic levels before powering it.

Discover the current macOS VCP port and open it with:

```bash
ls /dev/cu.*
python3 -m serial.tools.miniterm \
  <PORT> 115200
```

Count only final RESULT=FALL as a confirmed Fall. CANDIDATE is a pending
diagnostic state. EDGE_OF_FOV, TOO_CLOSE, TOO_FAR, NO_TARGET, and
NO_FRESH_TARGET are guard states and should not be scored as Falls.

## Host collection and conversion

The collection tools are under
`STM32/n6570dk_blink/tools/`.
They are separate from the flashed F7 firmware and do not change the AI model.

From the `radar/` repository root:

```bash
python STM32/n6570dk_blink/tools/radar_data_pipeline.py collect \
  --serial /dev/tty.usbserial-DEVICE \
  --activity walking \
  --duration 30 \
  --output-dir STM32/n6570dk_blink/training_data
```

For an existing direct binary capture:

```bash
python STM32/n6570dk_blink/tools/radar_data_pipeline.py decode \
  --input STM32/captures/ld6002b_after_fsbl.bin \
  --output-prefix STM32/captures/ld6002b_decoded
```

The checked-in reference capture is the decoded
STM32/captures/ld6002b_after_fsbl.csv/.jsonl pair. A raw .bin is not
currently present, so the decode command is for a newly collected raw file.
The converter writes the teammate-compatible schema:
timestamp,activity,frame,cluster_id,x,y,z,dop_idx.

## Frozen shareable package

The complete V1 hand-off is:

```text
packages/fall_nonfall_nucleo_f746zg/
```

It contains the firmware project and ELF, fold-4 checkpoint, binary ONNX
package, and its README. Version 2 must be copied to a separate directory and
must not modify this package. The only permitted future four-class package
directory is `packages/fall_4class_nucleo_f746zg/`.

The current four-class candidate now exists at that permitted path. Do not
call it board-deployable until its new firmware is built and live-tested.
