# Results: params=paper, protocol=fixed, trials=20, seed=0


## Iris

| Method | Mapping | Test acc. (ours) | Test acc. (paper) | Diff | #SV (ours) | #SV (paper) | Train acc. (ours) |
|---|---|---|---|---|---|---|---|
| SVM | gaussian | 93.33 ± 0.00 | 93.33 ± 0.00 | +0.00 | 20.2 | 29 | 98.67 |
| SVM | poly | 94.67 ± 0.00 | 96.00 ± 0.00 | -1.33 | 17.0 | 23 | 97.33 |
| Unified ELM | gaussian | 94.67 ± 0.00 | 97.33 ± 0.00 | -2.66 | all (dense) | - | 98.67 |
| Unified ELM | poly | 93.33 ± 0.00 | 97.33 ± 0.00 | -4.00 | all (dense) | - | 100.00 |
| Unified ELM | sigmoid | 94.47 ± 0.49 | 97.20 ± 0.00 | -2.73 | all (dense) | - | 98.67 |
| Unified ELM | sinusoid | 88.93 ± 0.76 | 96.20 ± 0.01 | -7.27 | all (dense) | - | 100.00 |
| Sparse ELM | gaussian | 94.67 ± 0.00 | 97.27 ± 1.60 | -2.60 | 51.6 | 54.45 | 98.67 |
| Sparse ELM | poly | 93.33 ± 0.00 | 97.47 ± 1.66 | -4.14 | 22.0 | 38.4 | 98.67 |
| Sparse ELM | sigmoid | 94.67 ± 0.00 | 97.40 ± 0.66 | -2.73 | 11.5 | 46.8 | 100.00 |
| Sparse ELM | sinusoid | 93.33 ± 0.00 | 97.33 ± 0.42 | -4.00 | 12.4 | 47.4 | 100.00 |

## Wine

| Method | Mapping | Test acc. (ours) | Test acc. (paper) | Diff | #SV (ours) | #SV (paper) | Train acc. (ours) |
|---|---|---|---|---|---|---|---|
| SVM | gaussian | 98.88 ± 0.00 | 97.75 ± 0.00 | +1.13 | 51.1 | 72 | 100.00 |
| SVM | poly | 94.38 ± 0.00 | 97.75 ± 0.00 | -3.37 | 36.0 | 46 | 100.00 |
| Unified ELM | gaussian | 97.75 ± 0.00 | 98.89 ± 0.00 | -1.14 | all (dense) | - | 98.88 |
| Unified ELM | poly | 97.75 ± 0.00 | 98.88 ± 0.00 | -1.13 | all (dense) | - | 97.75 |
| Unified ELM | sigmoid | 96.63 ± 0.89 | 99.16 ± 0.01 | -2.53 | all (dense) | - | 100.00 |
| Unified ELM | sinusoid | 93.99 ± 1.22 | 99.10 ± 0.01 | -5.11 | all (dense) | - | 100.00 |
| Sparse ELM | gaussian | 96.63 ± 0.00 | 97.92 ± 0.82 | -1.29 | 173.0 | 149.05 | 100.00 |
| Sparse ELM | poly | 93.26 ± 0.00 | 97.46 ± 1.89 | -4.20 | 40.0 | 49.9 | 100.00 |
| Sparse ELM | sigmoid | 94.66 ± 0.50 | 99.16 ± 0.70 | -4.50 | 26.4 | 54.55 | 100.00 |
| Sparse ELM | sinusoid | 95.22 ± 0.62 | 99.10 ± 0.84 | -3.88 | 37.8 | 53.25 | 100.00 |
