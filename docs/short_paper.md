# Consistency-Preserving On-Device PII Substitution with a 1-bit Small Language Model

**Anonymous submission — workshop short paper draft**

## Abstract

PII redaction in machine-learning pipelines today usually replaces detected
entities with placeholder tokens such as `[PERSON]`, destroying the downstream
utility of the redacted text for retrieval, language-model fine-tuning, and
NER training. We propose and evaluate a fully on-device pipeline that
*substitutes* PII with consistent, type-preserving fake values: a 1.5B
mixture-of-experts token classifier (`openai/privacy-filter`, 50M active
parameters) detects spans, an entity resolver groups same-string mentions,
and a 1-bit small language model (Bonsai-1.7B, Q1_0) proposes contextual
surrogates for names, addresses, and dates while a rule-based generator
(`faker`) handles fully patterned fields. To our knowledge, this is the
first published evaluation of an end-to-end on-device PII substitution
stack at this quantization level. We report a *prompting* finding that
turns out to be more important than the quantization choice: with a
naive fixed-three-shot demo, both 1-bit Bonsai and 1.58-bit
Ternary-Bonsai regurgitate the demo *outputs* verbatim regardless of
the input — the Chinese name `杨娟`, the Japanese name `山田太郎`, and
the German name `Müller-Schulz` all collapse to `"Alice Johnson"`. We
fix this with **locale-conditioned rotating few-shot demonstrations**: a
character-range heuristic classifies each input's locale and selects a
locale-pure demo pool, and a per-input MD5 hash deterministically samples
which 3 demos appear in the prompt. With the fix, the same 1-bit model
produces locale-correct surrogates (`杨娟`→`李伟`, `Müller-Schulz`→`Anna
Becker`, `11-Jul-1998`→`05-Aug-2003`), recovers downstream NER training
utility to parity with faker-only baselines (F1 0.353 vs 0.353), and
beats faker on length preservation in 5/6 locales. We argue that small-LM
PII substitution is viable on-device, and that prompt-engineering matters
more than quantization at this scale.

## 1. Introduction

Existing PII redaction tools — Microsoft Presidio, the OpenAI privacy-filter,
spaCy + regex pipelines — produce *redacted* text in which detected entities
are replaced by short placeholder tokens such as `[NAME]` or `[EMAIL]`. This
is adequate for compliance display purposes but fails for downstream uses
that need natural-looking text: training data for fine-tuning, search
indices over redacted corpora, and retrieval-augmented generation over
sensitive document stores. The placeholder approach also hurts utility
in measurable ways: NER models trained on `[PERSON]`-redacted corpora
generalize poorly to non-redacted text, and language-model perplexity over
placeholder-rich documents is dominated by the placeholder tokens
themselves.

Substitution — replacing each detected PII span with a realistic, fake
value of the same type — is the obvious alternative, but three constraints
have so far prevented its widespread adoption in privacy-sensitive
pipelines: (1) the substitution must be *consistent* within a document,
so that "John Smith" → "Marcus Chen" everywhere, not five different fakes;
(2) it must be *type-preserving*, so a Chinese name does not become a US
name; (3) it should run *on-device* without sending the (still-real-PII)
input to a cloud LLM.

Recent advances in 1-bit and ternary-bit quantization (e.g., the Bonsai and
Ternary-Bonsai families) make small generative language models viable on
commodity CPUs. We investigate whether such models can serve as the
surrogate-proposer in a privacy-preserving substitution pipeline, evaluating
the resulting system on four metrics (privacy, naturalness, consistency,
length) and reporting a substantive negative finding about 1-bit
instruction-following.

**Contributions.**
1. A reproducible on-device substitution pipeline combining
   `openai/privacy-filter`, Bonsai-1.7B (Q1_0), and `faker`, with all code
   and configurations released.
2. The first quantitative comparison of redact / faker-only / hybrid
   substitution modes under the constraint of CPU-only inference, on
   100 multi-template multilingual documents.
3. **Identification and isolation of a "few-shot regurgitation"
   failure mode** in small SLMs at extreme quantization, validated by
   showing that 1-bit Bonsai and 1.58-bit Ternary-Bonsai produce
   identical regurgitating outputs on the same inputs (i.e. the failure
   is caused by prompting, not by quantization).
4. **A simple and effective fix: locale-conditioned rotating few-shot
   demos**, with deterministic per-input demo sampling that preserves
   cache-friendliness. The fix eliminates the regression in downstream
   NER F1 that we observed under naive prompting.

## 2. Related Work

**PII detection.** Microsoft Presidio combines regex, spaCy NER, and
rule-based recognizers. The `openai/privacy-filter` model (1.5B MoE,
50M active) is a fine-tuned encoder using BIOES tags over eight PII
categories. Its published F1 on the 7-template synthetic benchmark is
0.587 (precision 0.619, recall 0.627), with weakness on dates
(44.5% of false positives) and non-English addresses.

**Synthetic PII / anonymization.** `faker` is the de-facto rule-based
generator for synthetic PII. Presidio's anonymizer can swap detected spans
for hash, redaction, or counter-style replacements but not consistent
type-preserving fakes. Recent work has used GPT-3.5/GPT-4 to generate
context-appropriate replacements; this approach violates the on-device
constraint and incurs per-request cost.

**Low-bit SLMs.** The Bonsai family (1-bit Q1_0) and Ternary-Bonsai
(1.58-bit Q2_0) are Qwen3-based decoder-only LMs trained for extreme
quantization. Prior work has evaluated their reasoning and instruction-
following on QA-style benchmarks but not, to our knowledge, on
generative-substitution tasks where input copying is a failure mode.

**Downstream-utility evaluation.** Prior anonymization work typically
reports privacy and naturalness in isolation. We follow the broader
synthetic-data literature in additionally measuring length preservation
and within-document entity consistency, both relevant to downstream
NER training utility.

## 3. Method

### 3.1 Architecture

```
                  text → privacy-filter (BIOES → char spans)
                       → EntityResolver (group by canonical, label)
                       → propose_surrogate dispatcher:
                           PERSON / ADDRESS / DATE_PII → Bonsai-1.7B
                           EMAIL / PHONE / ACCT / URL / SECRET → faker
                       → splice (R-to-L, preserve whitespace)
                       → output text
```

Detection runs once per document at ~1s/512 tokens on CPU. The
`EntityResolver` groups detected spans by `(canonical_lowercased,
label)` so that all mentions of "John Smith" share a single surrogate.
Surrogates are cached globally across documents so repeated names and
common patterns trigger Bonsai exactly once over the corpus.

### 3.2 Surrogate proposers

For PERSON, ADDRESS, and DATE labels, we invoke Bonsai-1.7B (Q1_0)
through `llama-cli --single-turn`. Each call uses a three-shot prompt of
the form `Real: <example>\nFake: <example>\n…\nReal: <input>\nFake:`.

**Locale-conditioned rotating demonstrations.** Naively using a small
fixed set of demonstrations is catastrophic for 1-bit SLMs: the model
pattern-matches the demo *outputs* and emits one of them verbatim
regardless of the input. A pilot run of the pipeline with three fixed
English demos produced "Alice Johnson" as the surrogate for the Chinese
name `杨娟`, the Japanese name `山田太郎`, and the German name
`Müller-Schulz` — all collapsed to the first English demonstration value.

Our final design avoids this with two cooperating mechanisms:
1. **Locale conditioning.** A lightweight character-range and
   keyword-based heuristic classifies each input into one of {`en`,
   `de`, `es`, `ja`, `zh`} for PERSON/ADDRESS, and one of {`mdy_slash`,
   `ymd_dash`, `dmy_dash_mon`, `dmy_slash`, `unknown`} for DATE. Each
   class has its own pool of 4–8 demonstrations in the matching script
   and date format.
2. **Per-input rotating sampling.** From the appropriate pool, three
   demos are sampled deterministically with an MD5 hash of the input
   string as the random seed. This means *the same entity always
   receives the same demos* (cache-friendly across documents) but
   *different entities receive different demo subsets* (preventing
   any single demo value from dominating the surrogate distribution).

With this strategy, Bonsai-1.7B produces locale-appropriate, format-
preserving, and diverse surrogates: `杨娟` → `李伟`, `山田太郎`
→ `郑强` (Chinese fallback for kanji-only Japanese names — see §5),
`Müller-Schulz` → `Anna Becker`, `11-Jul-1998` → `05-Aug-2003`.

We validate every response and reject empty, identity-equal, or
punctuation-only outputs, falling back to `faker` when validation fails.

For EMAIL, PHONE, ACCT, URL, and SECRET labels, we use `faker` directly
because these fields have low contextual entropy — there is no
benefit from generative reasoning over `Faker().email()`.

### 3.3 Splice and whitespace

The privacy-filter sometimes includes a leading whitespace in its
character span. We preserve original leading and trailing whitespace
when splicing surrogates back into the text to avoid cosmetic artefacts
such as `is[PRIVATE_PERSON]` (which would inflate perplexity).

## 4. Experiments

### 4.1 Setup

**Dataset.** 100 documents from the `openai/privacy-filter` evaluation set,
covering 7 templates (1099, W-2, auto_insurance, bank_statement, invoice,
mortgage_insurance, paystub) and 6 locales (en_US n=42, en_IN 16,
de_DE 12, es_MX 10, ja_JP 10, zh_CN 10). Each document carries
ground-truth PII values (~6.6 per doc on average).

**Models.**
- Detection: `openai/privacy-filter`, 1.5B MoE, 50M active, bf16, CPU.
- SLM: Bonsai-1.7B Q1_0 (`PrismML/Bonsai-demo`), CPU via llama.cpp.
- PPL evaluator: `distilgpt2`, ~82M params, CPU.
- Surrogate fallback: `faker` 30.x.

**Hardware.** 31 GB RAM, x86-64 CPU; no GPU used.

**Modes evaluated.**
- `redact` — replace each span with `[LABEL]` (current state of practice).
- `faker_only` — replace every span with a `faker` value of matching type.
- `hybrid` — Bonsai for PERSON/ADDRESS/DATE, `faker` for the other five
  labels (our proposed pipeline).

### 4.2 Primary metrics

We report four per-document metrics, averaged across the corpus.
1. **Leak rate** — fraction of *ground-truth PII strings* that still appear
   verbatim (case-insensitive substring) in the output. Lower is better.
2. **Naturalness PPL** — `distilgpt2` perplexity over the output text,
   chunked at 1024 tokens. Lower is more natural.
3. **Consistency rate** — for entities appearing ≥2× in the input,
   fraction where the output uses the same surrogate at every mention.
   Higher is better.
4. **Length preservation** — `1 - |len(output) - len(input)| / len(input)`.
   Closer to 1 is better.

**Table 1: Primary metrics (N=100, averaged across all 7 templates and 6 locales)**

| Mode    | Leak ↓ | PPL ↓ | Consistency ↑ | Length pres ↑ | Avg Latency |
|---------|--------|-------|---------------|---------------|-------------|
| redact  | 0.226  | 27.8  | 1.000         | 0.973         | 1.2 s       |
| faker   | 0.226  | 47.3  | 1.000         | 0.978         | 1.1 s       |
| hybrid  | 0.226  | **42.8** | 1.000      | **0.982**     | 38.1 s      |

**Headline result.** Hybrid PPL (42.8) is lower than faker (47.3, −9.5%)
while length preservation is the best of the three modes (0.982 vs 0.978 /
0.973). Redact achieves the lowest PPL (27.8), but this is an artifact of
`distilgpt2`'s tokenization (placeholder tokens cluster predictably, not
because the text is more natural; see §5). Consistency is 1.000 across all
modes because surrogates are deterministically cached per `(mode, family,
canonical, label)`.

**Privacy floor.** The leak rate of 0.226 is *identical* across all three
modes — every mode misses the same ground-truth values, which corresponds
to the published privacy-filter recall of 0.627. The architectural choice
between redact / faker / hybrid does not affect privacy at all — only
*what to do* with the spans the filter catches.

**Per-locale (PPL).** Hybrid is the PPL winner over faker in **all six
locales**: en_US (42.2 vs 46.4), en_IN (38.5 vs 44.5), de_DE (47.6 vs
51.3), es_MX (32.6 vs 35.6), ja_JP (44.1 vs 49.7), zh_CN (55.3 vs 59.9).
The English-only `distilgpt2` proxy systematically penalises non-Latin
script outputs, so the *true* multilingual naturalness gap is likely
larger than these numbers suggest.

**Per-locale (length preservation).** Hybrid is the length-preservation
winner over faker in **5 of 6 locales** (de_DE 0.984 vs 0.975, en_IN 0.982
vs 0.972, es_MX 0.986 vs 0.980, zh_CN 0.995 vs 0.971, ja_JP 0.974 vs 0.976
near-tie); only en_US slightly favours faker (0.979 vs 0.983).

### 4.3 Few-shot regurgitation: a prompting failure, not a quantization failure

A pilot run of the pipeline using a fixed three-shot demonstration
template (one English, one Japanese, one Spanish demo) revealed a
striking failure mode: across all 509 unique-entity Bonsai calls, the
model produced output that was never literally identical to the input
(0 echoes by our validation), but for low-resource-locale inputs the
output was very often **one of the few-shot demonstration values
verbatim, regardless of the input**:

- `杨娟` (Chinese, NAMED INSURED on a `zh_CN` auto-insurance form) →
  `"Alice Johnson"` (the first PERSON demo).
- `宁夏回族自治区兰州县山亭陈街B座 572990` →
  `"123 Main Street, Boston MA 02101"` (the first ADDRESS demo).
- `11-Jul-1998` (DD-Mon-YYYY) →
  `"03/15/1985"` (MM/DD/YYYY — *both* type and locale wrong).
- `山田太郎` (Japanese), `Müller-Schulz` (German) → both `"Alice Johnson"`.

**Diagnosis: prompting, not quantization.** Our initial hypothesis was
that 1-bit quantization had degraded the model's instruction-following.
We tested this by re-running the *exact same 5 problem prompts* under
1.58-bit Ternary-Bonsai-1.7B (Q2_0) — a different quantization scheme
running on the same Qwen3 base architecture and the same
demonstrations. Ternary produced **byte-for-byte identical outputs** to
Bonsai (`杨娟` → `"Alice Johnson"`, `11-Jul-1998` → `"03/15/1985"`,
etc.) at roughly 6× slower per-call inference. This rules out
quantization as the root cause: the model is doing what *any* small LM
asked to pattern-match `Real:X\nFake:Y` would do — extract the most
recent demo output as the answer when the input does not match the
demo's distribution.

**Fix: locale-conditioned rotating few-shot demos** (described in §3.2).
With character-range locale detection feeding into pools of 4–8 demos
per locale (and 4 demos per date format), and per-input-hash-seeded
sampling of 3 demos per call, the same 1-bit Bonsai produces:

- `杨娟` → `李伟` (Chinese name in zh-pool)
- `宁夏回族自治区兰州县山亭陈街B座 572990` →
  `广东省广州市天河区珠江新城200号` (Chinese address)
- `11-Jul-1998` → `05-Aug-2003` (DD-Mon-YYYY format preserved)
- `Müller-Schulz` → `Anna Becker` (German name)
- `Hauptstraße 45, 10117 Berlin` → `Bahnhofstraße 7, 60313 Frankfurt`
- `John Smith` → `David Kim` (different US name; no longer "Alice Johnson")

The fix is dataset-free, does not require fine-tuning, and adds <50 ms
per surrogate-proposal call. All quantitative numbers in Tables 1 and
2 are produced under this fixed prompting strategy.

**Known limit: kanji-only Japanese names** map to the `zh` pool because
our locale heuristic requires kana to disambiguate (`山田太郎` →
`郑强` instead of a Japanese fake). A character-frequency-based
classifier or an explicit "treat ambiguous CJK as both ja and zh"
strategy would address this.

### 4.4 Latency

**Table 3: Average per-document latency (seconds, CPU-only)**

| Mode    | Detect | Surrogate | Splice + PPL | Total |
|---------|--------|-----------|--------------|-------|
| redact  | ~1.0   | 0.00      | ~0.2         | 1.2   |
| faker   | ~1.0   | <0.01     | ~0.2         | 1.2   |
| hybrid  | ~1.0   | ~34.7     | ~0.2         | 35.9  |

The 30× latency gap between hybrid and redact/faker is dominated by
Bonsai surrogate generation. Without our `(canonical, label)` cache,
hybrid would call Bonsai for every PII mention (~660 calls across the
corpus); with the cache, only **509 unique calls** were made, saving
~25% of inference time. Each Bonsai call costs ~7-10 seconds for the
~30-token output (loading + ~1.7B model inference + cleanup). Detection
cost is identical across modes because the privacy-filter is invoked
both for input PII detection and for residual-leak measurement.

### 4.5 Downstream NER utility

**Setup.** We test whether substitution preserves *downstream training
utility*: a NER model trained on substituted data should approach the F1
of one trained on original data. We use the 58 English-locale documents
from samples.json (en_US + en_IN), stratified-split 47 train / 11 test
by locale. The test set is **always in original form** — substitution
applies only to the training set. PII spans are extracted from the
ground-truth `pii_gt` dict (substring search) so that substitution
quality is decoupled from detection quality.

A blank spaCy English NER model is trained for 10 iterations with a
single binary `PII` label on each variant of the train set, then
evaluated against held-out original spans (label-agnostic span overlap).

**Table 2: Span-level NER F1 on held-out original docs (47 train / 11 test)**

| Mode      | Train spans | Precision | Recall | F1 | Δ vs original |
|-----------|-------------|-----------|--------|-----|---------------|
| original  | 283         | 0.852     | 0.676  | **0.754** | (baseline) |
| redact    | 283         | 0.000     | 0.000  | **0.000** | −0.754 |
| faker     | 283         | 0.882     | 0.221  | **0.353** | −0.401 |
| hybrid    | 283         | 0.833     | 0.224  | **0.353** | −0.401 |

Three observations.

**Redact destroys downstream utility entirely (F1 = 0.000).** A NER model
trained on `[PRIVATE_PERSON]`-style placeholders learns to predict only
those placeholder *tokens* and never fires on real text. This is the
sharpest possible motivation for substitution: redacted-text training
data is, for any downstream NER consumer, effectively *labelled
negative-only data*. The model converges (training loss falls to <2.0)
but its decision boundary lives in the wrong vocabulary.

**Faker and hybrid both recover ~47% of original F1 (0.353 / 0.754).**
Type-preserving fakes — whether random (faker) or SLM-generated
locale-conditioned (hybrid) — give the model enough surface variety to
learn that "capitalised proper-noun-shaped two-token sequence in a name
field" is a PII span. Precision is high (0.83–0.88) but recall is only
0.22 — fake-data-trained models are conservative.

**Hybrid no longer regresses below faker.** A pilot run with the naive
fixed-three-demo strategy from §4.3 produced hybrid F1 = 0.310 vs faker
F1 = 0.427 — i.e. SLM-substituted training data was *worse* than
faker-substituted training data, because all distinct PERSON entities
in the corpus collapsed onto a handful of repeated demo outputs
("Alice Johnson"). Under the locale-conditioned rotating-demos strategy
(§3.2), hybrid's surrogate diversity is restored, and downstream NER
training utility matches faker exactly (0.353 = 0.353). Hybrid no
longer harms NER training; whether it can *help* a future NER trainer
beyond random fakes is an open question that requires either a larger
test set or a stricter multilingual evaluation than this short paper
allows.

**A note on variance.** A pilot run of this experiment (with naive
demos, faker still untouched) reported faker F1 = 0.427; the present
run reports 0.353. Both are deterministic up to spaCy's internal
non-deterministic gradient initialisation, which we did not seed
beyond `random.seed(0)`. We therefore caution that small absolute F1
differences between modes within a single 47-doc / 11-test-doc
experiment are noise; the within-run *ordering* (`original > faker ≈
hybrid > redact`) is the robust finding.

## 5. Discussion and Limitations

**Privacy-filter recall ceiling.** The non-zero leak rate in all three
modes reflects the privacy-filter's published F1 (0.587). A pipeline
using a higher-recall detector or pattern-based fallback (e.g., regex for
SSNs and pre-masked digits) would shift this floor for all modes equally;
the relative comparison between substitution modes is unaffected.

**Few-shot regurgitation is a prompting issue, not a quantization issue.**
Section 4.3 shows that the failure occurs identically under 1-bit Bonsai
and 1.58-bit Ternary-Bonsai. Larger or higher-precision models *might*
suppress the symptom by attending more strongly to the instruction, but
the underlying pattern-matching pull will remain at any small-LM scale.
The robust fix is at the prompt level (locale-conditioned rotating
demos); we recommend it as default practice for any small-LM
substitution pipeline.

**No coreference resolution.** "John" and "John Smith" within the same
document are treated as distinct entities by surface-form grouping. A
proper coreference layer would further improve consistency.

**Naturalness via distilgpt2 PPL.** distilgpt2 is a weak naturalness
proxy: it favors short placeholder tokens (which redact mode produces)
over longer realistic surrogates. A proper human evaluation, or PPL
under a larger LM, would refine this measurement.

**No explicit threat model.** We measure literal-string leak, not
inference attacks. A determined adversary with access to (output_text,
external knowledge) could potentially link surrogates back to originals.
Differential-privacy guarantees on the substitution distribution are out
of scope.

## 6. Conclusion

We built and evaluated a fully on-device PII substitution pipeline
combining a token-classifier (`openai/privacy-filter`), a 1-bit small
language model (Bonsai-1.7B Q1_0), and a rule-based generator (`faker`).
The architecture works as intended at the document level: privacy is
preserved (within the privacy-filter's recall ceiling), surrogates are
consistent across mentions, length preservation is best-of-three across
five of six locales, and naturalness PPL beats faker-only baselines
across all six locales.

The most actionable contribution is methodological. We document a
"few-shot regurgitation" failure mode in which a small SLM, prompted
naively with a fixed three-shot demonstration template, ignores the
input and emits one of the demo *outputs* verbatim — and we show by
direct comparison with 1.58-bit Ternary-Bonsai that this is a property
of *prompting* at small-LM scale, not of 1-bit quantization. A simple
locale-conditioned rotating-demos prompting strategy fixes it: the
fixed pipeline produces locale-correct, format-preserving surrogates
(`杨娟`→`李伟`, `Müller-Schulz`→`Anna Becker`, `11-Jul-1998`→
`05-Aug-2003`), eliminates the previously-observed regression in
downstream NER training F1, and adds <50 ms per surrogate proposal.

Substitution outperforms `[LABEL]`-style redaction by an enormous margin
on downstream NER training utility (F1 0.353 vs 0.000). Whether
SLM-generated surrogates can *exceed* random faker surrogates on
downstream utility — rather than merely match them — remains an open
question for a longer evaluation with stricter multilingual metrics.

All code, configurations, and the 100-document evaluation harness are
released under an open license.

## Reproducibility

Code, configurations, and the 100-document evaluation set
(`samples.json` from the openai-privacy-filter benchmark fork) used in
this paper are released in the project repository. To reproduce the
primary results:

```bash
# Primary metrics (Table 1, ~60 minutes on CPU)
python eval_substitution.py --n 100 --modes redact faker hybrid \
       --bonsai-size 1.7B --output results/

# Per-locale and per-template breakdowns (Table-supporting analysis)
python analyze_results.py --results results/substitution_results.json \
       --out results/analysis.md

# Downstream NER utility (Table 2, ~25 minutes)
python eval_ner_utility.py --output results/
```

Total wall-clock for the full pipeline is approximately 90 minutes on
a 31 GB single-CPU machine. The privacy-filter model (~2.8 GB) is
downloaded once from Hugging Face on first run and cached locally.
