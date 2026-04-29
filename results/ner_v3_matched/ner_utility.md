# Downstream NER Utility — span-level F1 on held-out original docs

Train set: 160 docs (substituted per mode), test set: 40 docs (always original).
Seeds: [0, 1, 2, 3, 4] (n=5). Reported as **mean ± SD** across seeds.

| Mode | Train spans | Precision | Recall | F1 |
|------|-------------|-----------|--------|----|
| original | 1046 | 0.895 ± 0.006 | 0.923 ± 0.004 | **0.908 ± 0.003** |
| redact | 1046 | 0.000 ± 0.000 | 0.000 ± 0.000 | **0.000 ± 0.000** |
| faker | 1046 | 0.955 ± 0.017 | 0.347 ± 0.052 | **0.506 ± 0.056** |

## Per-seed F1 (raw)

| Mode | seed=0 | seed=1 | seed=2 | seed=3 | seed=4 |
|------|----|----|----|----|----|
| original | 0.911 | 0.908 | 0.912 | 0.907 | 0.905 |
| redact | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| faker | 0.456 | 0.589 | 0.437 | 0.544 | 0.506 |
