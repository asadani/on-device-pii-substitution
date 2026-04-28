# Setup

This repo is the *paper artefact*: pipeline code, evaluation harnesses, and
results. The Bonsai-1.7B small language model and its `llama.cpp` runtime
live in a separate upstream repository (`PrismML/Bonsai-demo`) that we do
not vendor here. Install it once, then point this code at it.

Steps below assume Linux + Python 3.12 + a CPU-only machine (no GPU
required). The full setup takes ~10 minutes plus model download time.

## 1. Clone Bonsai-demo (sibling directory)

```bash
cd ..                                        # parent of this repo
git clone https://github.com/PrismML-Eng/Bonsai-demo.git bonsai-demo
cd bonsai-demo

# Download the 1.7B 1-bit (Q1_0) model — ~237 MB.
BONSAI_FAMILY=bonsai BONSAI_MODEL=1.7B ./scripts/download_models.sh

# Optional: download the Ternary-Bonsai-1.7B model used in the §4.3
# regurgitation cross-check experiment — ~442 MB.
BONSAI_FAMILY=ternary BONSAI_MODEL=1.7B ./scripts/download_models.sh
```

Then either symlink it next to this repo or set the env var:

```bash
# Option A: symlink (this repo's pii_substitute.py looks at ./bonsai-demo by default)
cd <this-repo>
ln -s ../bonsai-demo .

# Option B: env var (works regardless of where you cloned bonsai-demo)
export BONSAI_DEMO_DIR=/absolute/path/to/bonsai-demo
```

## 2. Python environment

```bash
cd <this-repo>
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The first run will download `openai/privacy-filter` (~2.8 GB) and
`distilgpt2` (~80 MB) from Hugging Face into `~/.cache/huggingface/`.

## 3. Smoke test

```bash
echo "Hi, my name is John Smith and my email is john@example.com." > /tmp/sample.txt
python pii_substitute.py --input /tmp/sample.txt --mode hybrid
```

Expected: a single Bonsai call (~10 s on CPU) replaces "John Smith" with
a different name; the email is replaced by `faker`. The output will look
like:

```
Hi, my name is David Kim and my email is some.address@example.org.
```

## 4. Reproduce the paper

The full evaluation takes ~90 minutes wall-clock on a single 31 GB CPU
machine (model download cached after the first run).

```bash
# Table 1 — primary metrics (~60 min, hybrid mode dominates)
python eval_substitution.py --n 100 --modes redact faker hybrid \
       --output results/

# Per-locale and per-template breakdowns
python analyze_results.py --results results/substitution_results.json \
       --out results/analysis.md

# Table 2 — downstream NER F1 with multi-seed mean ± SD (~25 min)
python eval_ner_utility.py --modes original redact faker hybrid \
       --seeds 0 1 2 3 4 --output results/
```

## 5. Build the paper

```bash
cd paper

# Full polish (requires latex-cjk-all + texlive-latex-extra)
./build.sh main      # → main.pdf

# If you do not have CJKutf8.sty installed, the local-build variant
# uses ASCII romanizations for CJK examples:
# ./build.sh main_local
```

On Debian/Ubuntu the LaTeX dependencies are:

```bash
sudo apt-get install -y --no-install-recommends \
    texlive-latex-recommended \
    texlive-latex-extra \
    texlive-fonts-recommended \
    latex-cjk-all
```

## Troubleshooting

* **`llama-cli not found`** — Bonsai-demo's `setup.sh` was not run, so the
  pre-built `bin/cpu/llama-cli` binary is missing. Re-run from inside the
  bonsai-demo directory.
* **`accelerate` import error** — `pip install accelerate>=1.0.0` (it is
  required by `transformers>=5.6` for `device_map="cpu"`).
* **HF gated dataset error on `pii-masking-200k`** — we do *not* use that
  dataset; the harness loads the local `samples.json` shipped with
  openai-privacy-filter. If you see this error it means the loader path
  is wrong; check `--samples` if you overrode the default.
