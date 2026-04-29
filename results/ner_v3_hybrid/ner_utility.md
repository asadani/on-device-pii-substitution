# Downstream NER Utility — span-level F1 on held-out original docs

Train set: 160 docs (substituted per mode), test set: 40 docs (always original).
Seeds: [0, 1, 2, 3, 4] (n=5). Reported as **mean ± SD** across seeds.

| Mode | Train spans | Precision | Recall | F1 |
|------|-------------|-----------|--------|----|
| hybrid | 1046 | 0.909 ± 0.015 | 0.215 ± 0.033 | **0.346 ± 0.044** |

## Per-seed F1 (raw)

| Mode | seed=0 | seed=1 | seed=2 | seed=3 | seed=4 |
|------|----|----|----|----|----|
| hybrid | 0.319 | 0.391 | 0.364 | 0.383 | 0.274 |
