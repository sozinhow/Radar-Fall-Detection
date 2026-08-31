# 50-Frame CNN-BiLSTM: Canonical Four-Class Evaluation Artifacts

Canonical clip classification only. Each held-out source session contributes one deterministic 50-frame clip. The model prediction is four-class argmax in the fixed order `walking, standing, sitting, fall`. No causal threshold, alert rule, temporal smoothing, binary conversion, future clip, or full-session aggregation is used.

## Mean +/- Std Across Outer Folds

| Accuracy | Macro precision | Macro recall | Macro F1 | Weighted precision | Weighted recall | Weighted F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.7611 +/- 0.0381 | 0.7738 +/- 0.0496 | 0.7744 +/- 0.0502 | 0.7673 +/- 0.0527 | 0.7702 +/- 0.0444 | 0.7611 +/- 0.0381 | 0.7584 +/- 0.0440 |

## Aggregate Per-Class Metrics

| Class | Precision | Recall | F1 | Pooled support |
|---|---:|---:|---:|---:|
| walking | 0.7716 +/- 0.0700 | 0.7307 +/- 0.1519 | 0.7441 +/- 0.0920 | 136 |
| standing | 0.8064 +/- 0.0815 | 0.7655 +/- 0.0970 | 0.7791 +/- 0.0327 | 171 |
| sitting | 0.7032 +/- 0.0608 | 0.7085 +/- 0.1408 | 0.6956 +/- 0.0505 | 135 |
| fall | 0.8140 +/- 0.1101 | 0.8931 +/- 0.0750 | 0.8507 +/- 0.0921 | 73 |

## Pooled Confusion Matrix

Rows are true class; columns are predicted class.

| True \ Predicted | walking | standing | sitting | fall |
|---|---:|---:|---:|---:|
| walking | 100 | 9 | 19 | 8 |
| standing | 14 | 131 | 24 | 2 |
| sitting | 13 | 20 | 96 | 6 |
| fall | 3 | 5 | 0 | 65 |

## Protocol Checks

- Every source session is evaluated once as an outer-fold held-out canonical clip.
- Train/validation/test source-session overlap is zero in every fold.
- Each checkpoint's fold-local training normalization is used unchanged for held-out inference.
- Edge-padded clips are consumed exactly as by the CNN training/inference implementation; padding counts are recorded per fold.
- These matrices are supplementary to, and must not be confused with, causal trailing-clip fall-alert metrics in `fold_metrics.csv` and `aggregate_metrics.csv`.
