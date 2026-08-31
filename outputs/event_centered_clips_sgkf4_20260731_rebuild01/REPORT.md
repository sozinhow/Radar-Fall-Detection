# Event-Centred Temporal Clip SGKF4 Experiment

**Decision: PROMISING.**

This is a staging-only experiment. The supplied frozen source-session SGKF4 outer assignments were used; grouped train/validation roles were derived from the supplied manifest. Normalization was fitted separately on each fold's real training clips. No synthetic, height-cluster, or session-height features were used.

## Dataset Audit

| Length | Clips/sessions | Fall | Walking | Standing | Sitting | Left padded | Right padded | Min pre-event s | Post-impact >=0.75 s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | 515 | 73 | 136 | 171 | 135 | 20 | 9 | -0.150 | 64/43 |
| 60 | 515 | 73 | 136 | 171 | 135 | 28 | 18 | 0.100 | 64/43 |

The requested branches are 50-frame, 60-frame. Fall clips request one second of post-impact context when available; missing boundary context is edge padded and recorded by a mask. Non-fall starts use the historical SHA-256 seed-42, clip-length, source-session selection from 15-frame-aligned candidates. No clip crosses a session boundary.

## Causal Sliding Test Results: Mean +/- Std

| Length | Fall precision | Fall recall | Fall F1 | Nonfall sessions alerted | False alerts/nonfall | Delay from event start s | Delay from impact s | No-alert falls | Repeated alerts/fall | Causal |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 50 | 0.8151 +/- 0.0970 | 0.8756 +/- 0.0840 | 0.8404 +/- 0.0666 | 3.7500 +/- 2.3629 | 0.0360 +/- 0.0217 | 1.6998 +/- 0.2727 | 0.7481 +/- 0.3758 | 2.2500 +/- 1.5000 | 0.0125 +/- 0.0250 | yes |
| 60 | 0.8619 +/- 0.0635 | 0.8381 +/- 0.0724 | 0.8470 +/- 0.0362 | 2.5000 +/- 1.2910 | 0.0225 +/- 0.0115 | 1.9107 +/- 0.3129 | 0.9543 +/- 0.4426 | 3.0000 +/- 1.4142 | 0.0000 +/- 0.0000 | yes |

## Per-Fold Causal Results

| Length | Fold | Threshold | Precision | Recall | F1 | Nonfall sessions alerted | False alerts/nonfall | Event-start delay s | Impact delay s | No-alert falls | Repeated alerts |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | 1 | 0.20 | 0.6957 | 0.9412 | 0.8000 | 7 | 0.0625 | 1.2969 | 0.2313 | 1 | 0.0000 |
| 50 | 2 | 0.70 | 0.7778 | 0.7778 | 0.7778 | 4 | 0.0450 | 1.8214 | 0.7821 | 4 | 0.0000 |
| 50 | 3 | 0.85 | 0.9048 | 0.9500 | 0.9268 | 2 | 0.0185 | 1.8974 | 1.1289 | 1 | 0.0500 |
| 50 | 4 | 0.75 | 0.8824 | 0.8333 | 0.8571 | 2 | 0.0180 | 1.7833 | 0.8500 | 3 | 0.0000 |
| 60 | 1 | 0.90 | 0.8421 | 0.9412 | 0.8889 | 3 | 0.0268 | 1.4500 | 0.3438 | 1 | 0.0000 |
| 60 | 2 | 0.90 | 0.8750 | 0.7778 | 0.8235 | 2 | 0.0180 | 2.0393 | 1.0000 | 4 | 0.0000 |
| 60 | 3 | 0.70 | 0.9412 | 0.8000 | 0.8649 | 1 | 0.0093 | 2.1469 | 1.4000 | 4 | 0.0000 |
| 60 | 4 | 0.65 | 0.7895 | 0.8333 | 0.8108 | 4 | 0.0360 | 2.0067 | 1.0733 | 3 | 0.0000 |

## Canonical Clip Classification (Different Task From E0 Windows)

These metrics classify one deterministic canonical clip per held-out session. They are not directly compared with E0's 1.5-second window metrics.

| Length | Accuracy | Macro F1 | Fall precision | Fall recall | Fall F1 |
|---:|---:|---:|---:|---:|---:|
| 50 | 0.7611 +/- 0.0381 | 0.7673 +/- 0.0527 | 0.8140 +/- 0.1101 | 0.8931 +/- 0.0750 | 0.8507 +/- 0.0921 |
| 60 | 0.7941 +/- 0.0266 | 0.7938 +/- 0.0326 | 0.8337 +/- 0.0935 | 0.8292 +/- 0.1430 | 0.8215 +/- 0.0648 |

## Decision Rule

A candidate must have causal mean recall >= 0.8182, mean nonfall sessions alerted <= 4.5 (at least about 28% below 6.25), mean impact delay <= 2.0 seconds, mean causal fall F1 > 0.6899, and zero future access. See `decision_criteria.csv`.

Every sliding prediction uses a trailing clip ending at the alert timestamp. The saved causal check asserts the largest source row used is no later than the available row at that clip end; no future clip or full-session aggregation is used.
