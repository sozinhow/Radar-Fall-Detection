# Decoded CSV to Edge Impulse dataset CSV

`decoded_csv_to_edge_impulse.py` converts the decoded CSV produced by
`hlk_ld6002b_capture.py` into the teammate-compatible schema:

```text
timestamp,activity,frame,cluster_id,x,y,z,dop_idx
```

It uses only `report == target_list` rows. Point-cloud and area-presence rows
are intentionally excluded. To match the teammate's historical collection
format, it keeps the first target only from each target-list report, writes one
row per output frame, and always writes `cluster_id` as `1`.

## Convert one capture

Run from `radar/STM32/` relative to the repository root:

```sh
python3 n6570dk_blink/tools/decoded_csv_to_edge_impulse.py \
  --input captures/ld6002b_after_fsbl.csv \
  --activity falling \
  --start-time 2026-07-28T13:45:00
```

The default output is:

```text
    captures/ld6002b_after_fsbl_edge_impulse.csv
```

Use `--output path/file.csv` to choose another location.

## Labels and timestamps

- `--activity` is copied unchanged to every output row.  Use the exact label
  vocabulary agreed with the teammate dataset, such as `falling`, `walking`,
  `standing`, or `empty`.
- `--start-time` is the time assigned to output frame `0`.  It accepts an
  ISO-8601 timestamp, for example `2026-07-28T13:45:00` or
  `2026-07-28T13:45:00+08:00`.
- The export rate is fixed at 20 Hz to match `collect_fixed_20hz.py` and
  `resample_to_20hz`. Frame `n` receives `start-time + n / 20`; values are
  rounded deterministically to microseconds.
- A target-list report with zero targets has no coordinate row and is skipped.
  It does not consume an output frame number.

The converter validates the Doppler index as an integer and x/y/z as finite
numbers. It preserves the first target's x/y/z/dop_idx values and intentionally
replaces the raw radar cluster ID with the teammate-compatible constant `1`.

## Edge Impulse / teammate hand-off

Compare the generated header and labels with the teammate CSV before upload.
The output has exactly one target row per exported frame. Retain the original
decoded CSV alongside it: it preserves the full multi-target radar data for
future analyses even though this training export is intentionally single-target.
