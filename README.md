# On-Device PII Substitution with Small Language Models

Reproducible pipeline and evaluation harness for the paper:

> **Locale-Conditioned Few-Shot Prompting Prevents Demonstration
> Regurgitation in On-Device PII Substitution with Small Language Models.**
> Anuj Sadani, Infrrd.ai, 2026.
> [`paper/main.pdf`](paper/main.pdf)

A fully on-device PII substitution stack combining:

| Component | Role | Paper §  |
|-----------|------|----------|
| [`openai/privacy-filter`](https://huggingface.co/openai/privacy-filter) (1.5 B MoE, 50 M active) | PII span detection (BIOES) | §3.1 |
| [Bonsai-1.7B Q1\_0](https://github.com/PrismML-Eng/Bonsai-demo) (1-bit) | Contextual surrogate proposer for `PERSON`, `ADDRESS`, `DATE` | §3.2 |
| [`faker`](https://github.com/joke2k/faker) | Rule-based generator for `EMAIL`, `PHONE`, `ACCT`, `URL`, `SECRET` | §3.2 |

Plus two pieces of measurement instrumentation that are **part of the eval
harness, not the substitution pipeline**:

| Component | Role | Paper §  |
|-----------|------|----------|
| [`facebook/xglm-564M`](https://huggingface.co/facebook/xglm-564M) (564 M, multilingual causal LM) | Naturalness PPL evaluator (Table 1) | §3.2 |
| [spaCy](https://spacy.io/) blank English NER | Downstream-utility experiment trains a fresh NER model on each substituted training corpus and evaluates against held-out original docs (Table 2) | §3.5 |

The central methodological contribution is a **locale-conditioned rotating
few-shot demonstrations** prompting strategy that fixes a "few-shot
regurgitation" failure mode of small SLMs at extreme quantization (§3.2,
§4.3 in the paper).

## Headline results

### Table 1 — primary metrics (N = 100 documents, 7 templates × 6 locales)

PPL is computed under multilingual XGLM-564M.

| Mode | Leak↓ | PPL↓ | Consistency↑ | Length pres.↑ | Latency |
|------|-------|------|--------------|---------------|---------|
| redact | 0.249 | 39.4 | 1.000 | 0.979 | 1.7 s |
| faker | 0.249 | 84.0 | 1.000 | 0.976 | 1.6 s |
| **hybrid** | 0.249 | **69.9** | 1.000 | **0.982** | 41.2 s |

*Hybrid beats faker on PPL in all six locales (e.g. zh\_CN 68.3 vs. 78.9,
ja\_JP 55.8 vs. 82.4) and on length preservation in 4 of 6 locales.
Bonsai produced 482 / 482 successful surrogate proposals (no echoes,
no errors) under the locale-conditioned prompting strategy.*

### Table 2 — downstream NER F1 (5 spaCy training seeds, mean ± SD)

Two scales — large for the cheap modes, matched 4-mode subset for `hybrid`
(which requires Bonsai surrogate generation per training doc):

**Large-scale (400 train / 100 test):**

| Mode | F1 | Δ vs original |
|------|------|--------------|
| original | 0.960 ± 0.004 | (baseline) |
| redact | 0.000 ± 0.000 | −0.960 |
| faker | 0.656 ± 0.033 | −0.304 |

**Matched subset (160 train / 40 test):**

| Mode | F1 | Δ vs original |
|------|------|--------------|
| original | 0.908 ± 0.003 | (baseline) |
| redact | 0.000 ± 0.000 | −0.908 |
| faker | **0.506 ± 0.056** | −0.402 |
| hybrid | 0.346 ± 0.044 | −0.562 |

*At the matched scale, faker beats hybrid by 0.160 F1 (Welch's t = 5.0,
p < 0.001). This is reported as an **honest negative finding** (§4.5):
contextually-realistic SLM surrogates produce more natural-looking text
but a less varied training distribution, and downstream NER training
benefits more from variety than from naturalness.*

## What's in this repo

```
.
├── pii_substitute.py          # detect → resolve → propose → splice pipeline
├── eval_substitution.py       # primary metrics harness (Table 1)
├── eval_ner_utility.py        # multi-seed downstream NER F1 (Table 2)
├── analyze_results.py         # per-locale / per-template breakdowns
├── paper/
│   ├── main.tex               # arXiv-ready LaTeX (CJKutf8 + booktabs + microtype)
│   ├── main.pdf               # compiled paper (12 pages)
│   ├── refs.bib               # references
│   └── build.sh
├── data/
│   └── samples_2000.json      # 2000-doc synthetic eval set (7 templates × 6 locales, ~4.7 MB)
├── results/                   # JSON + Markdown outputs from the runs cited in the paper
│   ├── run_v3/                # primary metrics, n=100, all 3 modes (Table 1)
│   ├── ner_v3/                # NER large-scale, n=400/100, 3 modes (Table 2 top)
│   ├── ner_v3_hybrid/         # NER hybrid only, n=160/40
│   └── ner_v3_matched/        # NER matched, n=160/40, all 4 modes (Table 2 bottom)
├── SETUP.md                   # one-time install + Bonsai-demo prerequisite
├── requirements.txt
├── LICENSE                    # MIT
└── README.md
```

## Quickstart

After [SETUP.md](SETUP.md) is done (one-time install of `transformers >=
5.6`, `spacy`, `sentencepiece`, plus the Bonsai-demo sibling clone for
`llama-cli`):

```bash
# Substitute PII in a single text file
echo "Hi, my name is John Smith and my email is john@example.com." > /tmp/sample.txt
python pii_substitute.py --input /tmp/sample.txt --mode hybrid

# Reproduce Table 1: primary metrics, n=100, all 3 modes (~3 hr on CPU,
# dominated by Bonsai hybrid + XGLM-564M PPL)
python eval_substitution.py --n 100 --modes redact faker hybrid \
       --output results/run_v3

# Reproduce Table 2 (top): NER utility, large scale, 3 modes (~2.5 hr)
python eval_ner_utility.py --modes original redact faker --n-max 500 \
       --seeds 0 1 2 3 4 --output results/ner_v3

# Reproduce Table 2 (bottom): NER utility, matched 4-mode subset (~1.5 hr)
python eval_ner_utility.py --modes original redact faker hybrid --n-max 200 \
       --seeds 0 1 2 3 4 --output results/ner_v3_matched
```

The eval scripts load `data/samples_2000.json` by default; pass
`--samples-path <other.json>` to override.

## Key research findings

1. **Few-shot regurgitation is a prompting failure, not a quantization
   failure.** A naive fixed three-shot prompt makes both 1-bit Bonsai and
   1.58-bit Ternary-Bonsai produce *byte-for-byte identical* wrong
   surrogates (杨娟 → "Alice Johnson"). Quantization is not the cause.
   See §4.3.
2. **Locale-conditioned rotating demonstrations fix it.** A character-range
   locale heuristic + per-input MD5-seeded sampling of 3 demos from a
   locale-pure pool restores diverse, locale-correct surrogates
   (杨娟 → 李伟). 482 / 482 unique Bonsai calls succeed under this
   strategy. Adds < 50 ms per call. See §3.2.
3. **Redact destroys downstream NER training utility entirely**
   (F1 = 0.000), while substitution preserves a substantial fraction
   (faker recovers 68 % of the original-text baseline at the large
   scale). See §4.5.
4. **Hybrid is significantly worse than faker on downstream NER training
   utility** at the matched 160 / 40 scale (F1 0.346 vs. 0.506,
   p < 0.001), even though hybrid wins on PPL and length preservation.
   Reported as an honest negative: *natural-looking* substitution and
   *useful-for-training* substitution are not the same objective. See §4.5.

## Citation

```bibtex
@misc{sadani2026pii,
  author       = {Sadani, Anuj},
  title        = {Locale-Conditioned Few-Shot Prompting Prevents
                  Demonstration Regurgitation in On-Device {PII}
                  Substitution with Small Language Models},
  year         = {2026},
  howpublished = {Manuscript},
  note         = {Affiliation: Infrrd.ai. Contact: anujsadani@infrrd.ai}
}
```

## License

[MIT](LICENSE) — © 2026 Anuj Sadani.
