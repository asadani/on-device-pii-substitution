# Post-Eval Analysis

Generated from `results/substitution_results.json`. N docs = 100.

## Bonsai SLM telemetry (hybrid mode)

- Bonsai calls attempted: **509**
- Succeeded (response not echoed): **509** (100.0% of attempts)
- Echoed input or empty: **0**
- Errored / timeout: **0**

## Per-locale breakdown

| Locale | n | Mode | Leak ↓ | PPL ↓ | Length pres ↑ |
|--------|---|------|--------|-------|---------------|
| de_DE | 12 | redact | 0.276 | 30.7 | 0.980 |
| de_DE | 12 | faker | 0.276 | 51.3 | 0.975 |
| de_DE | 12 | hybrid | 0.276 | 47.6 | 0.984 |
| en_IN | 16 | redact | 0.186 | 22.8 | 0.966 |
| en_IN | 16 | faker | 0.186 | 44.5 | 0.972 |
| en_IN | 16 | hybrid | 0.186 | 38.5 | 0.982 |
| en_US | 42 | redact | 0.208 | 28.7 | 0.972 |
| en_US | 42 | faker | 0.208 | 46.4 | 0.983 |
| en_US | 42 | hybrid | 0.208 | 42.2 | 0.979 |
| es_MX | 10 | redact | 0.333 | 26.1 | 0.986 |
| es_MX | 10 | faker | 0.333 | 35.6 | 0.980 |
| es_MX | 10 | hybrid | 0.333 | 32.6 | 0.986 |
| ja_JP | 10 | redact | 0.218 | 24.7 | 0.975 |
| ja_JP | 10 | faker | 0.218 | 49.7 | 0.976 |
| ja_JP | 10 | hybrid | 0.218 | 44.1 | 0.974 |
| zh_CN | 10 | redact | 0.203 | 33.1 | 0.963 |
| zh_CN | 10 | faker | 0.203 | 59.9 | 0.971 |
| zh_CN | 10 | hybrid | 0.203 | 55.3 | 0.995 |

## Per-template breakdown

| Template | n | Mode | Leak ↓ | PPL ↓ |
|----------|---|------|--------|-------|
| 1099 | 14 | redact | 0.274 | 60.9 |
| 1099 | 14 | faker | 0.274 | 87.7 |
| 1099 | 14 | hybrid | 0.274 | 72.8 |
| auto_insurance | 15 | redact | 0.078 | 27.9 |
| auto_insurance | 15 | faker | 0.078 | 71.8 |
| auto_insurance | 15 | hybrid | 0.078 | 67.3 |
| bank_statement | 15 | redact | 0.254 | 9.1 |
| bank_statement | 15 | faker | 0.254 | 13.1 |
| bank_statement | 15 | hybrid | 0.254 | 12.8 |
| invoice | 14 | redact | 0.293 | 19.6 |
| invoice | 14 | faker | 0.293 | 29.4 |
| invoice | 14 | hybrid | 0.293 | 28.1 |
| mortgage_insurance | 14 | redact | 0.143 | 34.7 |
| mortgage_insurance | 14 | faker | 0.143 | 71.7 |
| mortgage_insurance | 14 | hybrid | 0.143 | 65.3 |
| paystub | 14 | redact | 0.236 | 26.2 |
| paystub | 14 | faker | 0.236 | 37.4 |
| paystub | 14 | hybrid | 0.236 | 34.5 |
| w2 | 14 | redact | 0.312 | 17.3 |
| w2 | 14 | faker | 0.312 | 20.6 |
| w2 | 14 | hybrid | 0.312 | 19.3 |

## Surrogate distinctness

(Not available — re-run eval with `--keep-text` to populate output_text fields.)
