# Downstream NER Utility — span-level F1 on held-out original docs

Train set: 400 docs (substituted per mode), test set: 100 docs (always original).
Seeds: [0, 1, 2, 3, 4] (n=5). Reported as **mean ± SD** across seeds.

| Mode | Train spans | Precision | Recall | F1 |
|------|-------------|-----------|--------|----|
| original | 2592 | 0.947 ± 0.006 | 0.974 ± 0.012 | **0.960 ± 0.004** |
| redact | 2592 | 0.000 ± 0.000 | 0.000 ± 0.000 | **0.000 ± 0.000** |
| faker | 2592 | 0.984 ± 0.008 | 0.493 ± 0.038 | **0.656 ± 0.033** |

## Per-seed F1 (raw)

| Mode | seed=0 | seed=1 | seed=2 | seed=3 | seed=4 |
|------|----|----|----|----|----|
| original | 0.962 | 0.965 | 0.961 | 0.954 | 0.959 |
| redact | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| faker | 0.603 | 0.669 | 0.671 | 0.637 | 0.701 |
