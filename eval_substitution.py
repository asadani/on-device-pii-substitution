"""Evaluation harness for the PII substitution pipeline.

Loads N documents from ai4privacy/pii-masking-200k, runs each through three
modes (redact, faker, hybrid), and reports:

    1. Residual leak rate    — re-run privacy-filter on output; count flagged spans.
    2. Naturalness PPL       — perplexity under distilgpt2 on output text.
    3. Consistency rate      — for entities mentioned ≥2× in input, fraction where
                                output uses one consistent surrogate.
    4. Length preservation   — 1 - |len(out) - len(in)| / len(in).

Outputs:
    results/substitution_results.json  — raw per-doc + aggregated metrics
    results/substitution_table.md      — markdown comparison table
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from pii_substitute import (
    LABEL_ACCOUNT, LABEL_ADDRESS, LABEL_DATE, LABEL_EMAIL, LABEL_PERSON,
    LABEL_PHONE, LABEL_SECRET, LABEL_URL, Span, clear_surrogate_cache,
    detect_pii, get_bonsai_stats, reset_bonsai_stats, substitute,
)


@dataclass
class DocResult:
    doc_id: int
    mode: str
    input_text: str
    output_text: str
    n_input_pii_spans: int
    n_gt_values: int
    n_gt_leaked: int
    leak_rate: float           # fraction of GT PII *strings* still verbatim in output
    naturalness_ppl: float
    consistency_rate: float    # nan if no repeated entities
    n_repeated_entities: int
    length_preservation: float
    detect_s: float
    surrogate_s: float
    splice_s: float
    total_s: float


# ─── Naturalness (perplexity) ─────────────────────────────────────────────────

_PPL_MODEL_CACHE: dict = {}


def _get_ppl_model():
    if "model" in _PPL_MODEL_CACHE:
        return _PPL_MODEL_CACHE["tokenizer"], _PPL_MODEL_CACHE["model"]
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained("distilgpt2")
    mdl = AutoModelForCausalLM.from_pretrained("distilgpt2")
    mdl.eval()
    _PPL_MODEL_CACHE["tokenizer"] = tok
    _PPL_MODEL_CACHE["model"] = mdl
    return tok, mdl


def perplexity(text: str, max_chunk_tokens: int = 1024) -> float:
    """Compute perplexity of text under distilgpt2.

    Chunks long texts and averages NLL across chunks (token-weighted).
    """
    import torch
    tok, mdl = _get_ppl_model()
    ids = tok(text, return_tensors="pt", truncation=False)["input_ids"][0]
    if ids.numel() < 2:
        return float("nan")
    nll_total = 0.0
    n_total = 0
    for start in range(0, ids.numel(), max_chunk_tokens):
        chunk = ids[start:start + max_chunk_tokens]
        if chunk.numel() < 2:
            continue
        with torch.no_grad():
            out = mdl(input_ids=chunk.unsqueeze(0), labels=chunk.unsqueeze(0))
        nll_total += out.loss.item() * (chunk.numel() - 1)
        n_total += (chunk.numel() - 1)
    if n_total == 0:
        return float("nan")
    return math.exp(nll_total / n_total)


# ─── Consistency ──────────────────────────────────────────────────────────────

def _find_all_substr(haystack: str, needle: str) -> list[int]:
    """Return all start offsets of needle in haystack (case-insensitive)."""
    if not needle:
        return []
    starts = []
    h_lo = haystack.lower()
    n_lo = needle.lower()
    pos = 0
    while True:
        i = h_lo.find(n_lo, pos)
        if i < 0:
            break
        starts.append(i)
        pos = i + 1
    return starts


def consistency_rate(input_text: str, output_text: str, spans: list[Span],
                     surrogates: dict[tuple[str, str], str]) -> tuple[float, int]:
    """For entities appearing >=2 times in input, check that output uses
    the same surrogate at every mention.

    Returns (rate, n_repeated_entities). rate is nan if no repeated entities.
    """
    # Group input spans by canonical, label
    by_entity: dict[tuple[str, str], list[Span]] = defaultdict(list)
    for s in spans:
        if s.label in {LABEL_PERSON, LABEL_ADDRESS, LABEL_DATE, LABEL_EMAIL,
                       LABEL_PHONE, LABEL_ACCOUNT, LABEL_URL, LABEL_SECRET}:
            by_entity[(s.text.strip().lower(), s.label)].append(s)
    repeated = {k: v for k, v in by_entity.items() if len(v) >= 2}
    if not repeated:
        return float("nan"), 0
    consistent = 0
    for key, mentions in repeated.items():
        surrogate = surrogates.get(key)
        if not surrogate:
            continue
        # Count occurrences of the surrogate in output text
        n_occurrences = len(_find_all_substr(output_text, surrogate))
        if n_occurrences >= len(mentions):
            consistent += 1
    return consistent / len(repeated), len(repeated)


# ─── Per-doc evaluation ───────────────────────────────────────────────────────

def _flatten_gt_values(pii_gt: dict) -> list[str]:
    """Extract all string PII values from pii_gt, including inside lists."""
    values = []
    for v in pii_gt.values():
        if isinstance(v, list):
            values.extend(str(x) for x in v)
        elif v is not None:
            values.append(str(v))
    out = [v.strip() for v in values if v.strip()]
    # Keep only non-trivial strings (≥3 chars) to avoid spurious matches like "1"
    return [v for v in out if len(v) >= 3]


def evaluate_doc(doc_id: int, text: str, mode: str, pii_gt: dict,
                 bonsai_size: str = "1.7B", bonsai_family: str = "bonsai") -> DocResult:
    """Run substitute() on text and compute leak/naturalness/consistency/length metrics.

    leak_rate uses ground-truth PII *strings*: fraction of GT values that still
    appear verbatim (case-insensitive substring) in the output text.
    """
    output, entities, timings = substitute(text, mode=mode, bonsai_size=bonsai_size,
                                            bonsai_family=bonsai_family)

    surrogate_map = {(ent.canonical, ent.label): ent.surrogate for ent in entities.values()}
    n_input_spans = sum(len(ent.spans) for ent in entities.values())

    # Ground-truth leak: count GT PII values still present in output
    gt_values = _flatten_gt_values(pii_gt)
    out_lower = output.lower()
    leaked = [v for v in gt_values if v.lower() in out_lower]
    leak = len(leaked) / max(1, len(gt_values)) if gt_values else 0.0

    # Naturalness
    ppl = perplexity(output) if output.strip() else float("nan")

    # Consistency
    input_spans = [s for ent in entities.values() for s in ent.spans]
    cons, n_rep = consistency_rate(text, output, input_spans, surrogate_map)

    # Length preservation
    in_len = max(1, len(text))
    length_pres = 1.0 - abs(len(output) - len(text)) / in_len

    return DocResult(
        doc_id=doc_id,
        mode=mode,
        input_text=text,
        output_text=output,
        n_input_pii_spans=n_input_spans,
        n_gt_values=len(gt_values),
        n_gt_leaked=len(leaked),
        leak_rate=leak,
        naturalness_ppl=ppl,
        consistency_rate=cons,
        n_repeated_entities=n_rep,
        length_preservation=length_pres,
        detect_s=timings["detect_s"],
        surrogate_s=timings["surrogate_s"],
        splice_s=timings["splice_s"],
        total_s=timings["total_s"],
    )


# ─── Aggregation ──────────────────────────────────────────────────────────────

def _safe_mean(xs: list[float]) -> float:
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return statistics.mean(xs) if xs else float("nan")


def aggregate(results: list[DocResult]) -> dict[str, dict[str, float]]:
    """Group by mode, average metrics."""
    by_mode: dict[str, list[DocResult]] = defaultdict(list)
    for r in results:
        by_mode[r.mode].append(r)
    summary = {}
    for mode, rs in by_mode.items():
        summary[mode] = {
            "n_docs": len(rs),
            "leak_rate": _safe_mean([r.leak_rate for r in rs]),
            "naturalness_ppl": _safe_mean([r.naturalness_ppl for r in rs]),
            "consistency_rate": _safe_mean([r.consistency_rate for r in rs]),
            "length_preservation": _safe_mean([r.length_preservation for r in rs]),
            "avg_total_s": _safe_mean([r.total_s for r in rs]),
            "avg_detect_s": _safe_mean([r.detect_s for r in rs]),
            "avg_surrogate_s": _safe_mean([r.surrogate_s for r in rs]),
        }
    return summary


# ─── Output ───────────────────────────────────────────────────────────────────

def save_results(results: list[DocResult], summary: dict, out_dir: Path,
                 keep_text: bool = False) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Strip large text fields to keep JSON small unless keep_text is requested
    serialized = []
    for r in results:
        d = asdict(r)
        if not keep_text:
            d.pop("input_text", None)
            d.pop("output_text", None)
        serialized.append(d)

    json_path = out_dir / "substitution_results.json"
    json_path.write_text(json.dumps({
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "summary": summary,
        "per_doc": serialized,
    }, indent=2))

    md_path = out_dir / "substitution_table.md"
    with open(md_path, "w") as f:
        f.write("# PII Substitution Evaluation\n\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}\n\n")
        f.write(f"N docs per mode: {summary[next(iter(summary))]['n_docs']}\n\n")
        f.write("## Primary Metrics\n\n")
        f.write("| Mode | Leak rate ↓ | Naturalness PPL ↓ | Consistency ↑ | Length pres ↑ | Avg latency (s) |\n")
        f.write("|------|-------------|-------------------|---------------|---------------|----------------|\n")
        for mode in ["redact", "faker", "hybrid"]:
            if mode not in summary:
                continue
            s = summary[mode]
            f.write(
                f"| {mode} | {s['leak_rate']:.3f} | {s['naturalness_ppl']:.1f} | "
                f"{s['consistency_rate']:.3f} | {s['length_preservation']:.3f} | "
                f"{s['avg_total_s']:.1f} |\n"
            )
        f.write("\n## Latency Breakdown\n\n")
        f.write("| Mode | Detect (s) | Surrogate (s) | Total (s) |\n")
        f.write("|------|-----------|--------------|----------|\n")
        for mode in ["redact", "faker", "hybrid"]:
            if mode not in summary:
                continue
            s = summary[mode]
            f.write(f"| {mode} | {s['avg_detect_s']:.2f} | {s['avg_surrogate_s']:.2f} | {s['avg_total_s']:.1f} |\n")

    print(f"\nFull results → {json_path}")
    print(f"Comparison table → {md_path}")


# ─── Dataset loader ───────────────────────────────────────────────────────────

DEFAULT_SAMPLES_PATH = Path(
    "/home/anujsadani/gitrepo/asadani/openai-privacy-filter/samples.json"
)


def load_samples(n: int = 100, samples_path: Path = DEFAULT_SAMPLES_PATH):
    """Load N samples from a local JSON file with ground-truth PII labels.

    The samples.json shipped with openai-privacy-filter has 100 docs across
    7 templates (W2, 1099, invoice, paystub, bank_statement, auto_insurance,
    mortgage_insurance) and 6 locales (en_US, en_IN, de_DE, es_MX, ja_JP, zh_CN).
    Each doc has 'text', 'pii_gt' (ground-truth PII values), 'non_pii_data', and
    'metadata' fields.
    """
    with open(samples_path, encoding="utf-8") as f:
        records = json.load(f)
    docs = []
    for r in records[:n]:
        docs.append({
            "id": r.get("id", len(docs)),
            "text": r["text"],
            "pii_gt": r.get("pii_gt", {}),
            "metadata": r.get("metadata", {}),
        })
    return docs


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate PII substitution modes")
    parser.add_argument("--n", type=int, default=100, help="Number of documents")
    parser.add_argument("--modes", nargs="+", default=["redact", "faker", "hybrid"],
                        choices=["redact", "faker", "hybrid"])
    parser.add_argument("--bonsai-size", choices=["1.7B", "4B", "8B"], default="1.7B")
    parser.add_argument("--bonsai-family", choices=["bonsai", "ternary"], default="bonsai")
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--keep-text", action="store_true",
                        help="Include full input/output text in JSON (default: omit to keep file small)")
    args = parser.parse_args()

    print(f"Loading {args.n} docs from local samples.json...")
    docs = load_samples(n=args.n)
    types = sorted(set(d["metadata"].get("type", "?") for d in docs))
    locales = sorted(set(d["metadata"].get("locale", "?") for d in docs))
    print(f"  Got {len(docs)} docs (types: {types}, locales: {locales}).\n")

    # Suppress noisy transformers warning that interleaves with progress prints
    import logging
    logging.getLogger("transformers").setLevel(logging.ERROR)

    results: list[DocResult] = []
    bonsai_stats_per_mode: dict[str, dict] = {}
    for mode in args.modes:
        clear_surrogate_cache()
        reset_bonsai_stats()
        print(f"\n{'='*60}\n  Mode: {mode}\n{'='*60}")
        for idx, doc in enumerate(docs, 1):
            print(f"  [{idx:3d}/{len(docs)}] doc_id={doc['id']} ", end="", flush=True)
            t0 = time.perf_counter()
            try:
                r = evaluate_doc(doc["id"], doc["text"], mode, doc.get("pii_gt", {}),
                                 bonsai_size=args.bonsai_size,
                                 bonsai_family=args.bonsai_family)
                results.append(r)
                print(
                    f"leak={r.leak_rate:.2f}({r.n_gt_leaked}/{r.n_gt_values}) "
                    f"ppl={r.naturalness_ppl:.0f} "
                    f"cons={r.consistency_rate:.2f} len={r.length_preservation:.2f} "
                    f"t={r.total_s:.1f}s"
                )
            except Exception as e:
                print(f"ERROR: {type(e).__name__}: {e}")
        bonsai_stats_per_mode[mode] = get_bonsai_stats()
        bs = bonsai_stats_per_mode[mode]
        print(f"  [bonsai] attempted={bs['attempted']} succeeded={bs['succeeded']} "
              f"echoed={bs['echoed_or_empty']} errored={bs['errored']}")

    summary = aggregate(results)
    # Attach bonsai stats to summary for the paper
    for mode, stats in bonsai_stats_per_mode.items():
        if mode in summary:
            summary[mode]["bonsai_attempted"] = stats["attempted"]
            summary[mode]["bonsai_succeeded"] = stats["succeeded"]
            summary[mode]["bonsai_echoed"] = stats["echoed_or_empty"]
            summary[mode]["bonsai_errored"] = stats["errored"]
    save_results(results, summary, args.output, keep_text=args.keep_text)

    print("\n" + "="*60 + "\nSummary:\n" + "="*60)
    for mode, s in summary.items():
        print(f"  {mode:6s}: leak={s['leak_rate']:.3f} ppl={s['naturalness_ppl']:.1f} "
              f"cons={s['consistency_rate']:.3f} lenpres={s['length_preservation']:.3f} "
              f"avg_t={s['avg_total_s']:.1f}s")


if __name__ == "__main__":
    main()
