# Automatic Event Annotation Report

These annotations are `auto_event_annotations`: algorithmically inferred from radar time series.
They are not manually verified ground truth and are not commercially validated fall-event labels.

## Method

- Method version: `auto_event_heuristic_v1`
- Summary: Heuristic radar-only event inference using robust normalized xyz displacement, height/range/elevation derivatives, peak prominence, post-impact low-motion stability, estimated-height change, and cleaning-drop quality gates. Not manually verified ground truth.
- Impact point: maximum smoothed combined score from xyz displacement, absolute height/range/elevation derivatives.
- Event start: first sustained pre-impact score rise above 35% of peak, bounded to a short event window.
- Event end: first post-impact low-motion run, bounded to the target event duration.
- Confidence: downgraded for weak motion peak, weak height evidence, broad/multiple peaks, boundary events, missing post-event stability, geometry-edge warnings, and high cleaning drop.
- Doppler index is not treated as calibrated velocity.

## Thresholds

```yaml
exclude_ambiguous_fall_windows: false
high_drop_threshold: 0.4
low_motion_z: 0.8
max_event_duration_s: 2.2
middle_max_m: 3.2
min_event_duration_s: 0.8
near_max_m: 2.0
peak_z_threshold: 2.5
```

## Counts

- Sessions: 100
- Confidence counts: `{'medium': 78, 'high': 12, 'low': 10}`
- Include counts: `{True: 90, False: 10}`
- Event duration seconds: min=0.750, median=1.800, max=2.200

## Excluded sessions

- `falling_20260714_161059_20hz`: confidence=low; flags=event_near_recording_boundary;geometry_edge_warning;high_cleaning_drop=57.0%;weak_height_evidence
- `falling_20260714_161608_20hz`: confidence=low; flags=event_near_recording_boundary;high_cleaning_drop=40.0%;missing_post_event_stability;weak_height_evidence
- `falling_20260714_161708_20hz`: confidence=low; flags=event_near_recording_boundary;geometry_edge_warning;weak_height_evidence;weak_motion_peak
- `falling_20260714_161723_20hz`: confidence=low; flags=event_near_recording_boundary;geometry_edge_warning;weak_height_evidence;weak_motion_peak
- `falling_20260714_161735_20hz`: confidence=low; flags=event_near_recording_boundary;geometry_edge_warning;weak_height_evidence;weak_motion_peak
- `falling_20260715_140520_20hz`: confidence=low; flags=event_duration_outside_target;event_near_recording_boundary
- `falling_20260715_143444_20hz`: confidence=low; flags=event_near_recording_boundary;geometry_edge_warning;high_cleaning_drop=70.0%;missing_post_event_stability;weak_height_evidence
- `falling_20260717_140918_20hz`: confidence=low; flags=event_near_recording_boundary;geometry_edge_warning;high_cleaning_drop=43.0%
- `falling_20260717_143914_20hz`: confidence=low; flags=event_near_recording_boundary;high_cleaning_drop=56.0%;missing_post_event_stability;weak_height_evidence
- `falling_20260717_143932_20hz`: confidence=low; flags=event_near_recording_boundary;geometry_edge_warning;weak_motion_peak

## Cleaning drop summary

- `falling_20260714_160429_20hz`: non-frozen drop 13.0%
- `falling_20260714_160514_20hz`: non-frozen drop 4.0%
- `falling_20260714_160558_20hz`: non-frozen drop 16.0%
- `falling_20260714_160643_20hz`: non-frozen drop 3.0%
- `falling_20260714_160712_20hz`: non-frozen drop 19.0%
- `falling_20260714_160746_20hz`: non-frozen drop 5.0%
- `falling_20260714_160910_20hz`: non-frozen drop 19.0%
- `falling_20260714_161012_20hz`: non-frozen drop 6.0%
- `falling_20260714_161059_20hz`: non-frozen drop 57.0%
- `falling_20260714_161118_20hz`: non-frozen drop 24.0%
- `falling_20260714_161242_20hz`: non-frozen drop 5.0%
- `falling_20260714_161320_20hz`: non-frozen drop 4.0%
- `falling_20260714_161412_20hz`: non-frozen drop 0.0%
- `falling_20260714_161434_20hz`: non-frozen drop 7.0%
- `falling_20260714_161551_20hz`: non-frozen drop 20.0%
- `falling_20260714_161608_20hz`: non-frozen drop 40.0%
- `falling_20260714_161632_20hz`: non-frozen drop 15.0%
- `falling_20260714_161708_20hz`: non-frozen drop 1.0%
- `falling_20260714_161723_20hz`: non-frozen drop 0.0%
- `falling_20260714_161735_20hz`: non-frozen drop 1.0%
- `falling_20260715_135643_20hz`: non-frozen drop 18.0%
- `falling_20260715_135656_20hz`: non-frozen drop 19.0%
- `falling_20260715_135739_20hz`: non-frozen drop 9.0%
- `falling_20260715_135800_20hz`: non-frozen drop 3.0%
- `falling_20260715_135815_20hz`: non-frozen drop 12.0%
- `falling_20260715_135901_20hz`: non-frozen drop 10.0%
- `falling_20260715_135915_20hz`: non-frozen drop 1.0%
- `falling_20260715_140019_20hz`: non-frozen drop 3.0%
- `falling_20260715_140033_20hz`: non-frozen drop 20.0%
- `falling_20260715_140205_20hz`: non-frozen drop 3.0%
- `falling_20260715_140220_20hz`: non-frozen drop 2.0%
- `falling_20260715_140305_20hz`: non-frozen drop 1.0%
- `falling_20260715_140329_20hz`: non-frozen drop 30.0%
- `falling_20260715_140350_20hz`: non-frozen drop 32.0%
- `falling_20260715_140506_20hz`: non-frozen drop 4.0%
- `falling_20260715_140520_20hz`: non-frozen drop 17.0%
- `falling_20260715_140533_20hz`: non-frozen drop 23.0%
- `falling_20260715_140548_20hz`: non-frozen drop 16.0%
- `falling_20260715_140602_20hz`: non-frozen drop 7.0%
- `falling_20260715_141456_20hz`: non-frozen drop 2.0%
- `falling_20260715_141607_20hz`: non-frozen drop 1.0%
- `falling_20260715_141624_20hz`: non-frozen drop 12.0%
- `falling_20260715_141756_20hz`: non-frozen drop 1.0%
- `falling_20260715_141826_20hz`: non-frozen drop 1.0%
- `falling_20260715_141852_20hz`: non-frozen drop 0.0%
- `falling_20260715_141918_20hz`: non-frozen drop 1.0%
- `falling_20260715_141936_20hz`: non-frozen drop 1.0%
- `falling_20260715_142008_20hz`: non-frozen drop 2.0%
- `falling_20260715_142031_20hz`: non-frozen drop 21.0%
- `falling_20260715_142055_20hz`: non-frozen drop 0.0%
- `falling_20260715_142152_20hz`: non-frozen drop 5.0%
- `falling_20260715_142600_20hz`: non-frozen drop 5.0%
- `falling_20260715_142652_20hz`: non-frozen drop 3.0%
- `falling_20260715_142712_20hz`: non-frozen drop 31.0%
- `falling_20260715_142728_20hz`: non-frozen drop 6.0%
- `falling_20260715_142807_20hz`: non-frozen drop 3.0%
- `falling_20260715_142831_20hz`: non-frozen drop 34.0%
- `falling_20260715_142944_20hz`: non-frozen drop 10.0%
- `falling_20260715_143008_20hz`: non-frozen drop 19.0%
- `falling_20260715_143227_20hz`: non-frozen drop 1.0%
- `falling_20260715_143326_20hz`: non-frozen drop 16.0%
- `falling_20260715_143341_20hz`: non-frozen drop 34.0%
- `falling_20260715_143405_20hz`: non-frozen drop 27.0%
- `falling_20260715_143422_20hz`: non-frozen drop 26.0%
- `falling_20260715_143444_20hz`: non-frozen drop 70.0%
- `falling_20260717_140851_20hz`: non-frozen drop 4.0%
- `falling_20260717_140918_20hz`: non-frozen drop 43.0%
- `falling_20260717_140959_20hz`: non-frozen drop 6.0%
- `falling_20260717_141015_20hz`: non-frozen drop 4.0%
- `falling_20260717_141137_20hz`: non-frozen drop 4.0%
- `falling_20260717_141220_20hz`: non-frozen drop 1.0%
- `falling_20260717_141339_20hz`: non-frozen drop 31.0%
- `falling_20260717_141506_20hz`: non-frozen drop 18.0%
- `falling_20260717_141553_20hz`: non-frozen drop 40.0%
- `falling_20260717_141621_20hz`: non-frozen drop 0.0%
- `falling_20260717_141634_20hz`: non-frozen drop 24.0%
- `falling_20260717_141837_20hz`: non-frozen drop 6.0%
- `falling_20260717_142021_20hz`: non-frozen drop 3.0%
- `falling_20260717_142033_20hz`: non-frozen drop 1.0%
- `falling_20260717_142102_20hz`: non-frozen drop 5.0%
- `falling_20260717_142207_20hz`: non-frozen drop 11.0%
- `falling_20260717_142243_20hz`: non-frozen drop 7.0%
- `falling_20260717_142318_20hz`: non-frozen drop 0.0%
- `falling_20260717_142341_20hz`: non-frozen drop 2.0%
- `falling_20260717_142423_20hz`: non-frozen drop 10.0%
- `falling_20260717_142524_20hz`: non-frozen drop 5.0%
- `falling_20260717_142618_20hz`: non-frozen drop 7.0%
- `falling_20260717_142754_20hz`: non-frozen drop 1.0%
- `falling_20260717_142808_20hz`: non-frozen drop 0.0%
- `falling_20260717_142825_20hz`: non-frozen drop 4.0%
- `falling_20260717_143032_20hz`: non-frozen drop 1.0%
- `falling_20260717_143153_20hz`: non-frozen drop 2.0%
- `falling_20260717_143219_20hz`: non-frozen drop 5.0%
- `falling_20260717_143235_20hz`: non-frozen drop 1.0%
- `falling_20260717_143854_20hz`: non-frozen drop 8.0%
- `falling_20260717_143914_20hz`: non-frozen drop 56.0%
- `falling_20260717_143932_20hz`: non-frozen drop 4.0%
- `falling_20260717_144138_20hz`: non-frozen drop 5.0%
- `falling_20260717_144308_20hz`: non-frozen drop 2.0%
- `falling_20260717_144337_20hz`: non-frozen drop 2.0%

## Limitations

- The heuristic may select the largest radar motion artifact rather than the clinical fall event.
- Low-height postures can overlap with sitting or lying non-fall activities.
- Cleaning may remove or distort parts of the transition.
- These labels are suitable only for staging experiments and audit, not for commercial validation claims.
