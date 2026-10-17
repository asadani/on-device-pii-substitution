# PII Substitution Evaluation

Generated: 2026-04-29 03:10 UTC

N docs per mode: 5

## Primary Metrics

| Mode | Leak rate ↓ | Naturalness PPL ↓ | Consistency ↑ | Length pres ↑ | Avg latency (s) |
|------|-------------|-------------------|---------------|---------------|----------------|
| redact | 0.289 | 42.9 | 1.000 | 0.987 | 4.8 |
| faker | 0.289 | 80.0 | 1.000 | 0.980 | 1.2 |

## Latency Breakdown

| Mode | Detect (s) | Surrogate (s) | Total (s) |
|------|-----------|--------------|----------|
| redact | 4.79 | 0.00 | 4.8 |
| faker | 1.23 | 0.00 | 1.2 |
