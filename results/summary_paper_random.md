# Results: params=paper, protocol=random, trials=20, seed=0


## Iris

| Method | Mapping | Test acc. (ours) | Test acc. (paper) | Diff | #SV (ours) | #SV (paper) | Train acc. (ours) |
|---|---|---|---|---|---|---|---|
| SVM | gaussian | 96.20 ± 1.45 | 93.33 ± 0.00 | +2.87 | 21.2 | 29 | 97.93 |
| SVM | poly | 96.73 ± 1.70 | 96.00 ± 0.00 | +0.73 | 17.1 | 23 | 98.53 |
| Unified ELM | gaussian | 96.67 ± 2.10 | 97.33 ± 0.00 | -0.66 | all (dense) | - | 99.00 |
| Unified ELM | poly | 95.87 ± 1.98 | 97.33 ± 0.00 | -1.46 | all (dense) | - | 99.00 |
| Unified ELM | sigmoid | 95.80 ± 1.45 | 97.20 ± 0.00 | -1.40 | all (dense) | - | 99.20 |
| Unified ELM | sinusoid | 93.67 ± 2.20 | 96.20 ± 0.01 | -2.53 | all (dense) | - | 99.53 |
| Sparse ELM | gaussian | 95.80 ± 1.58 | 97.27 ± 1.60 | -1.47 | 53.5 | 54.45 | 98.27 |
| Sparse ELM | poly | 96.13 ± 1.72 | 97.47 ± 1.66 | -1.34 | 22.9 | 38.4 | 98.27 |
| Sparse ELM | sigmoid | 95.20 ± 2.31 | 97.40 ± 0.66 | -2.20 | 12.4 | 46.8 | 99.67 |
| Sparse ELM | sinusoid | 94.67 ± 2.63 | 97.33 ± 0.42 | -2.66 | 13.9 | 47.4 | 100.00 |

## Wine

| Method | Mapping | Test acc. (ours) | Test acc. (paper) | Diff | #SV (ours) | #SV (paper) | Train acc. (ours) |
|---|---|---|---|---|---|---|---|
| SVM | gaussian | 97.92 ± 1.64 | 97.75 ± 0.00 | +0.17 | 49.0 | 72 | 100.00 |
| SVM | poly | 96.46 ± 1.94 | 97.75 ± 0.00 | -1.29 | 33.5 | 46 | 100.00 |
| Unified ELM | gaussian | 97.98 ± 1.69 | 98.89 ± 0.00 | -0.91 | all (dense) | - | 99.61 |
| Unified ELM | poly | 98.48 ± 1.22 | 98.88 ± 0.00 | -0.40 | all (dense) | - | 99.61 |
| Unified ELM | sigmoid | 97.70 ± 1.39 | 99.16 ± 0.01 | -1.46 | all (dense) | - | 100.00 |
| Unified ELM | sinusoid | 95.62 ± 1.99 | 99.10 ± 0.01 | -3.48 | all (dense) | - | 100.00 |
| Sparse ELM | gaussian | 96.40 ± 1.39 | 97.92 ± 0.82 | -1.52 | 168.1 | 149.05 | 100.00 |
| Sparse ELM | poly | 96.40 ± 2.18 | 97.46 ± 1.89 | -1.06 | 40.5 | 49.9 | 100.00 |
| Sparse ELM | sigmoid | 96.57 ± 1.65 | 99.16 ± 0.70 | -2.59 | 26.9 | 54.55 | 100.00 |
| Sparse ELM | sinusoid | 97.02 ± 1.60 | 99.10 ± 0.84 | -2.08 | 37.4 | 53.25 | 100.00 |
