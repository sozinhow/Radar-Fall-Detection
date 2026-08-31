# HLK-LD6002B host-only capture tool

`hlk_ld6002b_capture.py` reads a saved binary capture or a host serial device.
It never sends configuration or control bytes. It decodes only the documented
radar-to-host reports `0x0A04` (targets), `0x0A08` (3D point cloud), and
`0x0A0A` (four-area presence).

## Hardware safety boundary

The HLK-LD6002B communication protocol does not define supply voltage, UART
I/O voltage, or the physical STM32 pin map. Verify the electrical datasheet,
the radar wiring, grounds, and logic levels before connecting any device.
Neither this tool nor the project establishes that STM32 PE5/PE6 are connected
to the radar.

Use a passive capture setup first. Do not attach two active transmitters to the
same radar RX line. Serial mode opens a host serial device for reading only;
the tool does not call `write()`.

## Usage

Run these commands from the repository root's STM32 project directory:

```sh
cd radar/STM32
```

Decode a saved raw byte capture (no extra packages required):

```sh
python3 n6570dk_blink/tools/hlk_ld6002b_capture.py \
  --input captures/<new_raw_capture>.bin \
  --output-prefix captures/ld6002b_decoded
```

Capture a verified host serial device for 30 seconds (requires `pyserial`):

```sh
python3 -m pip install pyserial
python3 n6570dk_blink/tools/hlk_ld6002b_capture.py \
  --serial /dev/tty.usbserial-DEVICE --duration 30 \
  --output-prefix captures/ld6002b_decoded
```

The default serial configuration is the protocol's 115200 8N1 with no flow
control. `--serial` is not a declaration of physical wiring; use it only after
the actual radar connection has been verified.

## Outputs and diagnostics

For `--output-prefix path/result`, the tool writes:

- `path/result.jsonl`: one decoded report object per line, with sorted JSON
  keys and deterministic stream byte offsets.
- `path/result.csv`: one target/point/area report record per row using a stable
  column schema.

Per-frame logs and the final compact summary are written to stderr. Counters
include received bytes, valid frames, invalid checksums, oversized frames,
partial-frame timeouts, discarded bytes, and decoded reports.

The parser finds `0x01`, checks the 8-byte TF header, rejects `LEN > 1024`,
buffers partial frames, and validates the documented complement-of-XOR header
and DATA checksums. A bad header checksum drops its leading sync byte; a
complete candidate with a valid header but bad DATA checksum is skipped at its
bounded declared length before resynchronisation.

## Tests

```sh
python3 -m unittest discover -s n6570dk_blink/tools -p 'test_*.py'
```

Tests use synthetic valid, partial/truncated, corrupt-checksum, oversized, and
concatenated TinyFrame streams. They do not require hardware or `pyserial`.
