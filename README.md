# On-Device PII Substitution with a 1-bit Small Language Model

Reproducible pipeline and evaluation harness for the paper:

> **Consistency-Preserving On-Device PII Substitution with a 1-bit Small
> Language Model.**
> Anuj Sadani, Infrrd.ai, 2026.
> [`paper/main.pdf`](paper/main.pdf)

A fully on-device PII substitution stack combining

| Component | Role | Paper cite |
|-----------|------|------------|
| [`openai/privacy-filter`](https://huggingface.co/openai/privacy-filter) (1.5 B MoE, 50 M active) | PII span detection (BIOES) | §3.1 |
| [Bonsai-1.7B Q1\_0](https://github.com/PrismML-Eng/Bonsai-demo) (1-bit) | Contextual surrogate proposer for `PERSON`, `ADDRESS`, `DATE` | §3.2 |
| [`faker`](https://github.com/joke2k/faker) | Rule-based generator for `EMAIL`, `PHONE`, `ACCT`, `URL`, `SECRET` | §3.2 |

with a **locale-conditioned rotating few-shot demonstrations** prompting
strategy that fixes the few-shot regurgitation failure mode of small SLMs
(§3.2, §4.3 in the paper).

## Headline results (n=100 documents, 7 templates × 6 locales)

### Table 1 — primary metrics

| Mode | Leak↓ | PPL↓ | Consistency↑ | Length pres.↑ | Latency |
|------|-------|------|--------------|---------------|---------|
| redact | 0.226 | 27.8 | 1.000 | 0.973 | 1.2 s |
| faker | 0.226 | 47.3 | 1.000 | 0.978 | 1.1 s |
| **hybrid** | 0.226 | **42.8** | 1.000 | **0.982** | 38.1 s |

*Hybrid wins on PPL in all 6 locales and on length preservation in 5/6.*

### Table 2 — downstream NER F1 (mean ± SD over 5 spaCy training seeds)

| Mode | F1 | Δ vs original |
|------|-----|---------------|
| original | 0.726 ± 0.046 | (baseline) |
| redact | 0.000 ± 0.000 | −0.726 |
| faker | 0.448 ± 0.082 | −0.278 |
| hybrid | 0.371 ± 0.049 | −0.355 |

*Faker vs hybrid difference: Welch's t = 1.80, p ≈ 0.11 (n=5, not significant).*
*Redact destroys downstream training utility entirely.*

## What's in this repo

```
.
├── pii_substitute.py        # detect → resolve → propose → splice pipeline
├── eval_substitution.py     # 4-metric harness (Table 1)
├── eval_ner_utility.py      # multi-seed downstream NER F1 (Table 2)
├── analyze_results.py       # per-locale / per-template breakdowns
├── paper/
│   ├── main.tex             # arXiv-ready LaTeX (CJKutf8 + booktabs + microtype)
│   ├── main.pdf             # compiled paper (11 pages)
│   ├── refs.bib             # 14 references (URL + urldate)
│   ├── build.sh
│   └── figures/
├── data/
│   └── samples_2000.json    # 2000-doc synthetic eval set (7 templates × 6 locales, ~4.7 MB)
├── results/                 # JSON + Markdown outputs from the runs cited in the paper
├── docs/
│   └── short_paper.md       # markdown source (paper precursor)
├── SETUP.md                 # one-time install + Bonsai-demo prerequisite
├── requirements.txt
├── LICENSE                  # MIT
└── README.md
```

## Quickstart

After [SETUP.md](SETUP.md) is done:

```bash
# Substitute PII in a single text file
echo "Hi, my name is John Smith and my email is john@example.com." > /tmp/sample.txt
python pii_substitute.py --input /tmp/sample.txt --mode hybrid

# Reproduce Table 1 (~60 min on CPU)
python eval_substitution.py --n 100 --modes redact faker hybrid --output results/

# Reproduce Table 2 (~25 min on CPU, 5 seeds)
python eval_ner_utility.py --modes original redact faker hybrid \
       --seeds 0 1 2 3 4 --output results/
```

## Key research findings

1. **Few-shot regurgitation is a prompting failure, not a quantization
   failure.** A naive fixed three-shot prompt makes both 1-bit Bonsai and
   1.58-bit Ternary-Bonsai produce *byte-for-byte identical* wrong
   surrogates (杨娟 → "Alice Johnson"). Quantization is not the cause.
   See §4.3.
2. **Locale-conditioned rotating demonstrations fix it.** A character-range
   locale heuristic + per-input MD5-seeded sampling of 3 demos from a
   locale-pure pool restores diverse, locale-correct surrogates
   (杨娟 → 李伟). Adds <50 ms per call. See §3.2.
3. **Redact destroys downstream NER training utility entirely** (F1 = 0.000)
   while substitution recovers ~50 – 60 % of the original-data baseline.
   See §4.5.

## Citation

```bibtex
@misc{sadani2026pii,
  author       = {Sadani, Anuj},
  title        = {Consistency-Preserving On-Device {PII} Substitution
                  with a 1-bit Small Language Model},
  year         = {2026},
  howpublished = {Manuscript},
  note         = {Affiliation: Infrrd.ai. Contact: anujsadani@infrrd.ai}
}
```

## License

[MIT](LICENSE) — © 2026 Anuj Sadani.
