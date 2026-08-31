# 60-Frame CNN-BiLSTM: Canonical Four-Class Evaluation Artifacts

Canonical clip classification only. Each held-out source session contributes one deterministic 60-frame clip. The model prediction is four-class argmax in the fixed order `walking, standing, sitting, fall`. No causal threshold, alert rule, temporal smoothing, binary conversion, future clip, or full-session aggregation is used.

## Mean +/- Std Across Outer Folds

| Accuracy | Macro precision | Macro recall | Macro F1 | Weighted precision | Weighted recall | Weighted F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.7941 +/- 0.0266 | 0.8055 +/- 0.0124 | 0.7980 +/- 0.0457 | 0.7938 +/- 0.0326 | 0.8040 +/- 0.0223 | 0.7941 +/- 0.0266 | 0.7916 +/- 0.0267 |

## Aggregate Per-Class Metrics

| Class | Precision | Recall | F1 | Pooled support |
|---|---:|---:|---:|---:|
| walking | 0.7569 +/- 0.0604 | 0.8884 +/- 0.0645 | 0.8166 +/- 0.0552 | 136 |
| standing | 0.8311 +/- 0.0577 | 0.7948 +/- 0.0689 | 0.8090 +/- 0.0182 | 171 |
| sitting | 0.8003 +/- 0.0802 | 0.6795 +/- 0.0931 | 0.7280 +/- 0.0283 | 135 |
| fall | 0.8337 +/- 0.0935 | 0.8292 +/- 0.1430 | 0.8215 +/- 0.0648 | 73 |

## Pooled Confusion Matrix

Rows are true class; columns are predicted class.

| True \ Predicted | walking | standing | sitting | fall |
|---|---:|---:|---:|---:|
| walking | 121 | 4 | 10 | 1 |
| standing | 16 | 136 | 15 | 4 |
| sitting | 15 | 20 | 92 | 8 |
| fall | 8 | 5 | 0 | 60 |

## Protocol Checks

- Every source session is evaluated once as an outer-fold held-out canonical clip.
- Train/validation/test source-session overlap is zero in every fold.
- Each checkpoint's fold-local training normalization is used unchanged for held-out inference.
- Edge-padded clips are consumed exactly as by the CNN training/inference implementation; padding counts are recorded per fold.
- These matrices are supplementary to, and must not be confused with, causal trailing-clip fall-alert metrics in `fold_metrics.csv` and `aggregate_metrics.csv`.
