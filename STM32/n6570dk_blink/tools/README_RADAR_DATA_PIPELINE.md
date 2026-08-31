# Radar data collection and conversion

`radar_data_pipeline.py` is the one-command host workflow for the
HLK-LD6002B data used by this project. It is receive-only: it never sends
configuration bytes to the radar.

Before connecting hardware, verify the radar supply voltage, UART I/O voltage,
TX/RX wiring, and common ground from the electrical documentation. The
communication-protocol PDF does not define those electrical details.

## Install

From the repository root, run the commands below from:

```sh
cd radar/STM32
```

The decoder itself uses only the Python standard library. Serial collection
needs `pyserial`:

```sh
python3 -m pip install pyserial
```

## Collect from a serial device

For a host UART/USB-UART carrying direct HLK TinyFrame bytes:

```sh
python3 n6570dk_blink/tools/radar_data_pipeline.py collect \
  --serial /dev/tty.usbserial-DEVICE \
  --activity walking \
  --duration 30 \
  --output-dir training_data
```

The default serial setting is 115200 8N1 with no flow control. Omit
`--duration` to collect until `Ctrl-C`. The default three-second countdown
clears stale bytes before writing the capture; use `--countdown 0` to disable
it.

If the STM32 raw-capture overlay is installed and the ST-LINK VCP emits RCAP
v1 records, add:

```sh
--input-format rcap
```

## Files produced

For example, a `walking` recording produces four files in `training_data/`:

```text
walking_20260817_143000_raw.bin       # exact bytes received from the host port
walking_20260817_143000_decoded.jsonl # complete decoded reports
walking_20260817_143000_decoded.csv   # one row per target/point/area report
walking_20260817_143000_20hz.csv      # training schema used by the project
```

The training CSV header is:

```text
timestamp,activity,frame,cluster_id,x,y,z,dop_idx
```

It keeps only the first target from each valid `0x0A04` target-list report,
uses a fixed 20 Hz timeline, and writes `cluster_id=1` for compatibility with
the existing dataset. The decoded JSONL/CSV and raw binary retain the full
source data for later analysis.

## Decode or convert existing data

Decode an existing direct TinyFrame binary capture:

```sh
python3 n6570dk_blink/tools/radar_data_pipeline.py decode \
  --input captures/<new_raw_capture>.bin \
  --output-prefix captures/ld6002b_decoded
```

For an RCAP v1 capture:

```sh
python3 n6570dk_blink/tools/radar_data_pipeline.py decode \
  --input captures/stm32_vcp.bin \
  --input-format rcap \
  --output-prefix captures/stm32_vcp_decoded
```

Convert a decoded CSV to the project training format:

```sh
python3 n6570dk_blink/tools/radar_data_pipeline.py convert \
  --input captures/ld6002b_decoded.csv \
  --output training_data/falling_20260817_143000_20hz.csv \
  --activity falling \
  --start-time 2026-08-17T14:30:00+08:00
```

## Tests

The tests use synthetic TinyFrame and RCAP data, so no board or serial device
is required:

```sh
python3 -m unittest discover -s n6570dk_blink/tools -p 'test_*.py'
```
