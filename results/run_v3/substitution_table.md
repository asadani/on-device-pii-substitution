# PII Substitution Evaluation

Generated: 2026-04-29 09:54 UTC

N docs per mode: 100

## Primary Metrics

| Mode | Leak rate ↓ | Naturalness PPL ↓ | Consistency ↑ | Length pres ↑ | Avg latency (s) |
|------|-------------|-------------------|---------------|---------------|----------------|
| redact | 0.249 | 39.4 | 1.000 | 0.979 | 1.7 |
| faker | 0.249 | 84.0 | 1.000 | 0.976 | 1.6 |
| hybrid | 0.249 | 69.9 | 1.000 | 0.982 | 41.2 |

## Latency Breakdown

| Mode | Detect (s) | Surrogate (s) | Total (s) |
|------|-----------|--------------|----------|
| redact | 1.68 | 0.00 | 1.7 |
| faker | 1.62 | 0.00 | 1.6 |
| hybrid | 1.45 | 39.78 | 41.2 |
