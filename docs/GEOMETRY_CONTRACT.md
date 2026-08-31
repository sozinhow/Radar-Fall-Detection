# Version 1 geometry and model-input contract

Updated: 2026-08-19 HKT

This is the observed project contract for the accepted Version 1 detector. It
is not a vendor-certified radar packet specification.

## Input

The model consumes these seven channels in exactly this order:

    x, y, z, dop_idx, range_m, azimuth_deg, elevation_deg

The live tensor is 60 samples at 20 Hz. The seven values are normalized with
the fold-specific mean/std saved beside the checkpoint and deployment
package. The STM32 firmware uses the generated, frozen normalization and
SENSOR_NATIVE Z convention.

The source CSVs are tracked-target records, not raw radar IQ, range-Doppler
cubes, or unprocessed packet captures. The teammate-compatible schema is:

    timestamp,activity,frame,cluster_id,x,y,z,dop_idx

## Geometry

When range or angle columns are absent, the current host canonicalization uses:

    range_m       = sqrt(x^2 + y^2 + z^2)
    azimuth_deg   = degrees(atan2(x, y))
    elevation_deg = degrees(atan2(z, sqrt(x^2 + y^2)))

These are project conventions. They do not prove the radar vendor's axis,
origin, handedness, tilt correction, or internal tracker formula. The final
seven-channel model tensor treats the fields as one correlated, post-processed
feature vector; do not regenerate range/angles after smoothing or transform
only selected channels.

The coordinate frame is intentionally recorded as unknown/vendor-undocumented.
The project treats x/y/z numerically as metres because the training and live
geometry gates use metre-valued quantities. That treatment is not a hardware
calibration claim.

## Doppler

V1 keeps the integer doppler index as dop_idx. The owner-supplied conversion
doppler_velocity_mps = dop_idx * 0.92 is reserved for a separately versioned
experiment. Do not replace dop_idx in the frozen V1 model input.

## V1 safety rules

- Keep the source-session grouped split.
- Keep fold-local normalization.
- Keep 20 Hz sampling and the 60-frame window.
- Do not add a physical velocity threshold without packet/version and sign
  documentation.
- Do not change range, angle, or Z conventions in the live firmware without a
  new model/preprocessing version and a new test record.

## Evidence and implementation

- Canonicalization: radar_pipeline/common.py
- Cleaning/smoothing: radar_pipeline/cleaning_ops.py
- Model contract: fold_summary/fold_4/metrics.json
- Deployment contract:
  packages/fall_nonfall_nucleo_f746zg/deployment/binary/deployment_manifest.json
- Four-class candidate contract:
  packages/fall_4class_nucleo_f746zg/deployment/four_class/deployment_manifest.json
- Four-class firmware parser and live sampling:
  packages/fall_4class_nucleo_f746zg/firmware/NUCLEO_F7_AI_four_class/Core/Src/radar_live.c
- STM32 parser and live sampling:
  STM32/NUCLEO_F7_AI/Core/Src/radar_live.c
