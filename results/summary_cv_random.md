# Results: params=cv, protocol=random, trials=20, seed=0


## Iris

| Method | Mapping | Test acc. (ours) | Test acc. (paper) | Diff | #SV (ours) | #SV (paper) | Train acc. (ours) |
|---|---|---|---|---|---|---|---|
| SVM | gaussian | 96.53 ± 1.58 | 93.33 ± 0.00 | +3.20 | 20.1 | 29 | 98.27 |
| SVM | poly | 96.33 ± 1.49 | 96.00 ± 0.00 | +0.33 | 21.6 | 23 | 98.20 |
| Unified ELM | gaussian | 95.93 ± 2.39 | 97.33 ± 0.00 | -1.40 | all (dense) | - | 96.20 |
| Unified ELM | poly | 95.93 ± 1.91 | 97.33 ± 0.00 | -1.40 | all (dense) | - | 97.13 |
| Unified ELM | sigmoid | 96.20 ± 2.13 | 97.20 ± 0.00 | -1.00 | all (dense) | - | 97.53 |
| Unified ELM | sinusoid | 95.87 ± 2.07 | 96.20 ± 0.01 | -0.33 | all (dense) | - | 96.53 |
| Sparse ELM | gaussian | 96.53 ± 1.58 | 97.27 ± 1.60 | -0.74 | 23.1 | 54.45 | 98.27 |
| Sparse ELM | poly | 95.73 ± 2.06 | 97.47 ± 1.66 | -1.74 | 43.4 | 38.4 | 96.67 |
| Sparse ELM | sigmoid | 95.93 ± 1.47 | 97.40 ± 0.66 | -1.47 | 26.8 | 46.8 | 97.80 |
| Sparse ELM | sinusoid | 96.00 ± 1.84 | 97.33 ± 0.42 | -1.33 | 26.6 | 47.4 | 98.07 |

## Wine

| Method | Mapping | Test acc. (ours) | Test acc. (paper) | Diff | #SV (ours) | #SV (paper) | Train acc. (ours) |
|---|---|---|---|---|---|---|---|
| SVM | gaussian | 97.87 ± 1.20 | 97.75 ± 0.00 | +0.12 | 78.4 | 72 | 98.15 |
| SVM | poly | 97.36 ± 1.60 | 97.75 ± 0.00 | -0.39 | 52.0 | 46 | 98.26 |
| Unified ELM | gaussian | 98.09 ± 0.90 | 98.89 ± 0.00 | -0.80 | all (dense) | - | 100.00 |
| Unified ELM | poly | 98.43 ± 0.85 | 98.88 ± 0.00 | -0.45 | all (dense) | - | 99.83 |
| Unified ELM | sigmoid | 97.64 ± 1.50 | 99.16 ± 0.01 | -1.52 | all (dense) | - | 100.00 |
| Unified ELM | sinusoid | 97.53 ± 1.53 | 99.10 ± 0.01 | -1.57 | all (dense) | - | 100.00 |
| Sparse ELM | gaussian | 97.30 ± 1.61 | 97.92 ± 0.82 | -0.62 | 87.8 | 149.05 | 99.78 |
| Sparse ELM | poly | 97.47 ± 1.58 | 97.46 ± 1.89 | +0.01 | 71.0 | 49.9 | 98.20 |
| Sparse ELM | sigmoid | 97.36 ± 1.90 | 99.16 ± 0.70 | -1.80 | 62.6 | 54.55 | 98.31 |
| Sparse ELM | sinusoid | 97.13 ± 1.77 | 99.10 ± 0.84 | -1.97 | 39.0 | 53.25 | 100.00 |
