# Canonical Naming Convention

This is the permanent naming reference for all future radar data.

## Activity folders

Use:

```text
{activity}_{distance_band}/
```

Rules:

- Lowercase only.
- Words separated with underscores.
- No dates, version numbers, or collection-round numbers in folder names.
- `distance_band` is a controlled value such as `near`, `middle`, or `far`.
- New classes use the same pattern. Examples: `walking_far`, `fall_forward_near`, `fall_forward_standing_middle`.
- The folder name is the canonical source of activity and distance metadata.

## CSV filenames

Use:

```text
{activity}_{YYYYMMDD_HHMMSS}_{samplerate}.csv
```

Examples:

```text
sitting_20260714_135735_20hz.csv
walking_20260714_142000_20hz.csv
fall_forward_20260714_143000_20hz.csv
```

Rules:

- Keep the activity prefix lowercase and underscore-separated.
- Use the recording start timestamp in local collection time.
- Use the measured/declared sampling rate, such as `20hz`.
- Use the same filename in corresponding raw and cleaned folders.
- Preserve `session_id`, `recording_id`, and `source_file` columns inside every CSV.
- Do not encode distance bands, versions, or collection rounds in filenames; those belong in the folder path and status manifest.

## Canonical path example

```text
data/raw_csv/sitting_near/sitting_20260714_135735_20hz.csv
data/cleaned_csv/sitting_near/sitting_20260714_135735_20hz.csv
```
