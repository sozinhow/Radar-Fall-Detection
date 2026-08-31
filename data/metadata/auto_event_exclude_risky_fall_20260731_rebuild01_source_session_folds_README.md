# Date-Versioned SGKF4 Source-Session Manifest

- Dataset path: `data/final_dataset_auto_event_exclude_risky_fall_20260731_rebuild01/radar_dataset.npz`
- Dataset SHA-256: `f7956b4beec90fee0f8e7c04173e01d98ea8ee862a62b6e4115ebb3e72368ce2`
- Manifest path: `data/metadata/auto_event_exclude_risky_fall_20260731_rebuild01_source_session_folds.csv`
- Manifest SHA-256: `812e55172fc18275ee813681987e95d50a21fa6681eb805c499dfe342f616c92`
- Generation timestamp: `2026-07-31T12:36:16+08:00`
- Folds: 4
- Seed: 42
- Protocol name: `sgkf_grouped_20260731_seed42_k4`
- Group key: `source_session_id / Path(source_csv).stem`
- Splitter: `StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)`

This manifest is frozen for only this exact dataset session inventory and must not be overwritten.

## Class-Level Counts

| Class | Canonical source sessions | Effective windows |
|---|---:|---:|
| walking | 136 | 656 |
| standing | 171 | 903 |
| sitting | 135 | 669 |
| fall | 73 | 144 |

## Per-Fold Counts

| Fold | Source sessions | Windows | Walking sessions/windows | Standing sessions/windows | Sitting sessions/windows | Fall sessions/windows |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 129 | 594 | 34/163 | 44/233 | 34/168 | 17/30 |
| 2 | 129 | 607 | 35/173 | 43/225 | 33/164 | 18/45 |
| 3 | 128 | 575 | 32/150 | 43/229 | 33/163 | 20/33 |
| 4 | 129 | 596 | 35/170 | 41/216 | 35/174 | 18/36 |

## Zero-Leakage Verification

| Fold | Train sessions | Val sessions | Test sessions | Train windows | Val windows | Test windows | Train/val leaks | Train/test leaks | Val/test leaks |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 321 | 65 | 129 | 1489 | 289 | 594 | 0 | 0 | 0 |
| 2 | 322 | 64 | 129 | 1478 | 287 | 607 | 0 | 0 | 0 |
| 3 | 322 | 65 | 128 | 1496 | 301 | 575 | 0 | 0 | 0 |
| 4 | 321 | 65 | 129 | 1483 | 293 | 596 | 0 | 0 | 0 |
