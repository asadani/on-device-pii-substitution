# Downstream NER Utility — span-level F1 on held-out original docs

Train set: 47 docs (substituted per mode), test set: 11 docs (always original).
Seeds: [0, 1, 2, 3, 4] (n=5). Reported as **mean ± SD** across seeds.

| Mode | Train spans | Precision | Recall | F1 |
|------|-------------|-----------|--------|----|
| original | 283 | 0.830 ± 0.033 | 0.653 ± 0.090 | **0.726 ± 0.046** |
| redact | 283 | 0.000 ± 0.000 | 0.000 ± 0.000 | **0.000 ± 0.000** |
| faker | 283 | 0.843 ± 0.042 | 0.310 ± 0.071 | **0.448 ± 0.082** |
| hybrid | 283 | 0.835 ± 0.061 | 0.240 ± 0.039 | **0.371 ± 0.049** |

## Per-seed F1 (raw)

| Mode | seed=0 | seed=1 | seed=2 | seed=3 | seed=4 |
|------|----|----|----|----|----|
| original | 0.661 | 0.794 | 0.750 | 0.696 | 0.727 |
| redact | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| faker | 0.442 | 0.520 | 0.293 | 0.489 | 0.495 |
| hybrid | 0.353 | 0.440 | 0.376 | 0.293 | 0.395 |
