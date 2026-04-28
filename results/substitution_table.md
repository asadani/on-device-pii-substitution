# PII Substitution Evaluation

Generated: 2026-04-28 05:52 UTC

N docs per mode: 100

## Primary Metrics

| Mode | Leak rate ↓ | Naturalness PPL ↓ | Consistency ↑ | Length pres ↑ | Avg latency (s) |
|------|-------------|-------------------|---------------|---------------|----------------|
| redact | 0.226 | 27.8 | 1.000 | 0.973 | 1.2 |
| faker | 0.226 | 47.3 | 1.000 | 0.978 | 1.1 |
| hybrid | 0.226 | 42.8 | 1.000 | 0.982 | 38.1 |

## Latency Breakdown

| Mode | Detect (s) | Surrogate (s) | Total (s) |
|------|-----------|--------------|----------|
| redact | 1.24 | 0.00 | 1.2 |
| faker | 1.15 | 0.00 | 1.1 |
| hybrid | 1.26 | 36.80 | 38.1 |
