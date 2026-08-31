# CNNTemporal dual ONNX comparison — cnn_temporal_dual_onnx_20260814_run01

Both ONNX graphs are static float32 exports with input `[1,60,7]`; the input is expected to be normalized with the fold-local `normalization.json` before inference.

## Model choice

- Fold 3 supplies the four-class artifact because its saved held-out test evidence is the requested four-class reference: accuracy 0.8677685950 and macro F1 0.8638530615, with fall precision 1.0, recall 0.7272727273, and F1 0.8421052632.
- Fold 4 supplies the fall-sensitive binary artifact because its saved four-class confusion matrix has 100% fall recall. Collapsing walking/standing/sitting into `non_fall` gives 97.54098% binary accuracy, 76.9231% fall precision, 100% fall recall, and 86.9565% fall F1; precision is below 80%.
- The binary ONNX is a binary adapter derived from a four-class checkpoint; it is not a separately retrained binary model. Its `non_fall` logit is `logsumexp` over the original three non-fall logits, and its `fall` logit is the original fall logit.

## Validation comparison

| Version | ONNX artifact | Output classes | PyTorch–ONNX parity | Max abs logit error | Max abs probability error | ST Edge AI |
|---|---|---|---|---:|---:|---|
| fold3_four_class | `outputs/deployment/cnn_temporal_dual_onnx_20260814_run01/four_class/cnn_temporal_fold3_four_class.onnx` | `walking, standing, sitting, fall` | pass (classes match: True) | 2.38418579e-06 | 5.36441803e-07 | not_available |
| fold4_binary_nonfall_fall | `outputs/deployment/cnn_temporal_dual_onnx_20260814_run01/binary/cnn_temporal_fold4_nonfall_fall.onnx` | `non_fall, fall` | pass (classes match: True) | 2.38418579e-06 | 2.98023224e-07 | not_available |

Parity used deterministic random normalized tensors plus one complete real held-out example for each original class from the matching Fold 3/Fold 4 staging split. It compared output shapes, logits, softmax probabilities, and predicted classes with `atol=1e-5` and `rtol=1e-5`; both versions passed.

## ST Edge AI status

No `stedgeai`, `stedgeai-cli`, or `stm32ai` executable was available in PATH, so compiler analysis was not run. Consequently, supported operators, RAM, flash, and latency are recorded as unavailable rather than inferred. The binary graph does contain `ReduceLogSumExp`; if a later ST Edge AI analysis rejects it, preserve the requested mathematics and group the four probabilities outside ONNX, or pursue a separately approved true-binary retraining task. No firmware was modified and no board was flashed. The per-version `st_edge_ai_analysis.json` files record this exact status and the attempted tool discovery.

## Artifacts and SHA-256

### fold3_four_class
- ONNX: `outputs/deployment/cnn_temporal_dual_onnx_20260814_run01/four_class/cnn_temporal_fold3_four_class.onnx`
  - SHA-256: `632ef26fc80294e20bc690714ead34bc1df57001e8cac2e4a8ce7275bed691ac`
- Source checkpoint: `fold_summary/fold_3/cnn_temporal_event_centered.pt`
  - SHA-256: `c5afe7b9556a4bf5bc55bdbb14c2b718b71a613b2620ac0601cc245ffdbf462f`
- Manifest: `outputs/deployment/cnn_temporal_dual_onnx_20260814_run01/four_class/deployment_manifest.json`
- Export report: `outputs/deployment/cnn_temporal_dual_onnx_20260814_run01/four_class/export_report.json`
- ONNX graph report: `outputs/deployment/cnn_temporal_dual_onnx_20260814_run01/four_class/onnx_graph_report.json`
- ONNX Runtime parity report: `outputs/deployment/cnn_temporal_dual_onnx_20260814_run01/four_class/onnxruntime_parity_report.json`
- ST Edge AI report: `outputs/deployment/cnn_temporal_dual_onnx_20260814_run01/four_class/st_edge_ai_analysis.json`

### fold4_binary_nonfall_fall
- ONNX: `outputs/deployment/cnn_temporal_dual_onnx_20260814_run01/binary/cnn_temporal_fold4_nonfall_fall.onnx`
  - SHA-256: `10f8ed0d368aa95373be470a3ac17f7010569bb886689d977cb6ddcdc035b438`
- Source checkpoint: `fold_summary/fold_4/cnn_temporal_event_centered.pt`
  - SHA-256: `7789f13877133a6c00c52e58f2cde4c1b00713d895dfe2b6f278ce11aba5e8e8`
- Manifest: `outputs/deployment/cnn_temporal_dual_onnx_20260814_run01/binary/deployment_manifest.json`
- Export report: `outputs/deployment/cnn_temporal_dual_onnx_20260814_run01/binary/export_report.json`
- ONNX graph report: `outputs/deployment/cnn_temporal_dual_onnx_20260814_run01/binary/onnx_graph_report.json`
- ONNX Runtime parity report: `outputs/deployment/cnn_temporal_dual_onnx_20260814_run01/binary/onnxruntime_parity_report.json`
- ST Edge AI report: `outputs/deployment/cnn_temporal_dual_onnx_20260814_run01/binary/st_edge_ai_analysis.json`
