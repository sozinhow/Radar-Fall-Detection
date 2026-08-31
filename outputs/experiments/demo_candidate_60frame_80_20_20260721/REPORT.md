# 60-Frame Event-Centred CNN-BiLSTM Demo Candidate

**Single deterministic 80/20 source-session split; prototype demonstration candidate.**

The split is session-grouped, not random window-level splitting. Validation was derived only from the 80% training pool. Test signal values and predictions were untouched until training, early stopping, validation-only threshold selection, checkpoint selection, and checkpoint saving were complete.

This run is not directly comparable to the SGKF4 mean +/- standard deviation. It does not supersede the frozen SGKF4 staging evidence. This is suitable for a single-model demo candidate, not a deployment, clinical-reliability, or final-product validation claim.

## Split Audit

| Split | Sessions/clips | Walking | Standing | Sitting | Fall |
|---|---:|---:|---:|---:|---:|
| train | 327 | 92 | 115 | 91 | 29 |
| validation | 61 | 17 | 22 | 17 | 5 |
| test | 97 | 27 | 34 | 27 | 9 |

Every class is present in every split. Held-out test contains nine fall sessions, above the predeclared minimum of five. Source-session overlap is zero.

## Selected Training State

- Selected epoch: 13 of 25 epochs run.
- Best validation loss: 0.441733.
- Validation-selected fall-alert threshold: 0.50.
- Architecture: accepted CNN-BiLSTM; seven normalized base inputs plus the accepted deterministic six-feature derivation (13 model channels).
- Padding/mask behavior: saved edge-padded 60-frame clips are consumed exactly as in the accepted CNN-BiLSTM, whose architecture has no padding-mask input; masks remain recorded for audit.

## Held-Out Canonical Four-Class Argmax

| Accuracy | Macro precision | Macro recall | Macro F1 | Weighted precision | Weighted recall | Weighted F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.7423 | 0.7279 | 0.7372 | 0.7299 | 0.7379 | 0.7423 | 0.7372 |

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| walking | 0.7083 | 0.6296 | 0.6667 | 27 |
| standing | 0.7949 | 0.9118 | 0.8493 | 34 |
| sitting | 0.7083 | 0.6296 | 0.6667 | 27 |
| fall | 0.7000 | 0.7778 | 0.7368 | 9 |

Rows are true classes and columns are predicted classes.

| True \ Predicted | walking | standing | sitting | fall |
|---|---:|---:|---:|---:|
| walking | 17 | 3 | 6 | 1 |
| standing | 3 | 31 | 0 | 0 |
| sitting | 4 | 4 | 17 | 2 |
| fall | 0 | 1 | 1 | 7 |

## Held-Out Causal Streaming Fall Alerts

Only trailing 60-frame clips ending at the current alert timestamp were used; no future frame or full-session aggregation was available to an earlier decision.

| Fall precision | Fall recall | Fall F1 | Non-fall sessions alerted | False alerts/non-fall | No-alert falls | Fall sessions with repeated alerts | Event-start delay s | Impact delay s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.5385 | 0.7778 | 0.6364 | 6 | 0.0682 | 2 | 0 | 1.8929 | 0.7571 |

Causal session confusion matrix: rows are true `non_fall, fall`; columns are predicted `no_alert, alert`.

| True \ Predicted | no_alert | alert |
|---|---:|---:|
| non_fall | 82 | 6 |
| fall | 2 | 7 |

## Artifact Locations

- Checkpoint: `outputs/experiments/demo_candidate_60frame_80_20_20260721/model/cnn_bilstm_60frame_demo_candidate.pt`
- Split manifest: `outputs/experiments/demo_candidate_60frame_80_20_20260721/split_manifest.csv`
- Training configuration: `outputs/experiments/demo_candidate_60frame_80_20_20260721/training_config.json`
- Normalization: `outputs/experiments/demo_candidate_60frame_80_20_20260721/normalization_stats.json`
- Four-class metrics: `outputs/experiments/demo_candidate_60frame_80_20_20260721/metrics/test_four_class_metrics.json`
- Causal metrics: `outputs/experiments/demo_candidate_60frame_80_20_20260721/metrics/test_causal_fall_alert_metrics.json`
- Session diagnostics: `outputs/experiments/demo_candidate_60frame_80_20_20260721/metrics/test_causal_session_diagnostics.csv`
- Training curve: `outputs/experiments/demo_candidate_60frame_80_20_20260721/plots/training_curve.png`

## Integrity Checks

- Exactly one new checkpoint was saved; no ensemble and no existing fold model was selected.
- Normalization was fitted on train sessions only and then frozen.
- Early stopping and fall-alert threshold selection used validation only.
- The test split was scored once after model and threshold freeze.
- Every causal current/past-only and alert-time-equals-clip-end assertion passed.
- Protected paths were verified unchanged after the run.
