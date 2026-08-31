# Teammate collection compatibility audit

Primary reference: `DataCollection_Tools/collect_fixed_20hz.py` as inspected on
2026-07-30. The secondary path named in the task,
`radar/STM32/n6570dk_blink/collect_fixed_20hz.py`, is not present in the working
tree or the available `n6570dk_blink.zip`, so it could not contribute behavior.
The official V1.2 protocol confirms that `0x0A04` is a radar-to-host target-list
report containing `target_num`, then repeated little-endian `x`, `y`, `z`,
`dop_idx`, and `cluster_id` fields.

## What the old program actually does

- Opens `serial.Serial(port, baudrate, timeout=0.01)`, defaults to `COM4` and
  115200 baud, waits 0.5 seconds, and never writes to the port. Each loop reads
  exactly `in_waiting` bytes when the count is nonzero.
- Scans each read chunk independently for byte `0x01`; reads `LEN` and `TYPE`
  big-endian; accepts only `TYPE=0x0A04`; requires `LEN >= 24` and the declared
  payload to be present. It does not retain split frames, validate either
  checksum, inspect frame ID, or require an exact payload length.
- Reads `target_num` as little-endian unsigned 32-bit and accepts 1-10. It reads
  only target index 0: little-endian float `x/y/z` and signed 32-bit `dop_idx`.
  It does not read the reported `cluster_id`.
- Accepts the first candidate whose Euclidean distance is 0.3-8.0 m inclusive
  and whose Z is -3.0-2.0 m inclusive. The stricter ranges in
  `validate_data_quality.py` are post-collection diagnostics, not collection
  gates. There is no continuity, nearest-target, or best-target selection.
- After a three-second countdown, samples the latest accepted target on a host
  50 ms schedule. It always saves duplicates, holds the last target indefinitely
  through empty/invalid reads, saves nothing until a target has been accepted,
  and emits at most one row per loop when late (missed ticks are not backfilled).
- Builds rows in memory, then writes
  `training_data/<activity>_<YYYYmmdd_HHMMSS>_20hz.csv`. Columns are exactly
  `timestamp,activity,frame,cluster_id,x,y,z,dop_idx`; `timestamp` is
  `datetime.now().isoformat()` at the save tick; `frame` is consecutive over
  saved rows; and `cluster_id` is always the literal `1`. With zero rows it
  creates no CSV.

## Differences from the pre-compatibility current tools

### `hlk_ld6002b_capture.py` CLI

- Uses an incremental bounded parser, validates header/payload checksums and
  exact payload lengths, recovers across chunks, and decodes documented
  `0x0A04`, `0x0A08`, and `0x0A0A` reports.
- Writes event-rate JSONL plus an 18-column diagnostic CSV, flattening every
  target/point. It has no old distance/Z gate, first-target reduction, held
  sample, activity filename, or fixed-rate training schema.

### `radar_collection_gui.py` before this mode

- Used the robust incremental parser with nonblocking 512-byte reads.
- Defaulted to strict 0.5-6.0 m and -2.5-1.0 m gates, cluster continuity, then
  nearest-target ranking. It could therefore save a later target instead of
  target index 0.
- Invalid/empty target reports stopped current selection, and a 0.5-second
  freshness timeout stopped writes. Its scheduler backfilled multiple overdue
  rows and generated idealized grid timestamps from collection start.
- It opened a header-only CSV immediately and had no three-second countdown.
  Its filename, column order, hard-coded output `cluster_id=1`, activity values,
  and nominal 20 Hz rate already resembled the old output.

## Implemented compatibility mode

The GUI now defaults to **Teammate-compatible** and restores the old serial
timeout/read shape, stateless chunk parser, first-target-only rule, legacy
gates, indefinite latest-value hold, three-second countdown, one-tick/no-catchup
scheduler, wall-clock row timestamps, lazy no-empty-file behavior, filename,
and CSV schema. The existing smart selection modes remain separate for
diagnostics. All collection paths remain receive-only; no radar command is sent.

## Unavoidable or intentional host differences

- Tk polls about every 10 ms instead of the old script sleeping 1 ms, and OS/UI
  scheduling still determines timestamp jitter and the exact row count.
- The GUI writes each accepted row to disk immediately after the first sample
  instead of buffering every row until the session ends. Final CSV content and
  ordering are unchanged, while an interrupted session is more recoverable.
- The GUI writes to this project's fixed `n6570dk_blink/training_data` directory
  rather than resolving `training_data` from the process working directory.
- It uses exclusive file creation rather than silently overwriting a same-second
  filename. This is the only deliberate safety divergence in file handling.
