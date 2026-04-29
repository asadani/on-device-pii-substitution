"""Post-eval analysis: extract paper-relevant secondary metrics from results JSON.

Run after eval_substitution.py completes. Reads results/substitution_results.json
and emits:
  - Per-locale leak / PPL breakdown (for the multilingual discussion)
  - Bonsai call statistics: attempted, succeeded, echoed, errored, unique outputs
  - Surrogate distinctness: unique surrogates / unique entities (for the
    1-bit copying discussion)
"""

from __future__ import annotations
import argparse
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def load_samples(path: Path):
    return json.load(open(path))


def per_locale(results, samples):
    """Group per-doc metrics by locale for stratified reporting."""
    samples_by_id = {s["id"]: s for s in samples}
    by_locale_mode: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for r in results:
        sample = samples_by_id.get(r["doc_id"])
        if not sample:
            continue
        loc = sample.get("metadata", {}).get("locale", "?")
        mode = r["mode"]
        by_locale_mode[loc][mode].append(r)
    return by_locale_mode


def per_template(results, samples):
    samples_by_id = {s["id"]: s for s in samples}
    by_tpl_mode: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for r in results:
        sample = samples_by_id.get(r["doc_id"])
        if not sample:
            continue
        tpl = sample.get("metadata", {}).get("type", "?")
        mode = r["mode"]
        by_tpl_mode[tpl][mode].append(r)
    return by_tpl_mode


def extract_hybrid_surrogates(results_with_text):
    """Best-effort: from hybrid output_text, look at how unique the substitutions are.

    Heuristic: tokenize words, count distinct alpha-only person-name-like tokens
    appearing in output but not in input. This is approximate but indicative.
    """
    surrogates_seen = []
    for r in results_with_text:
        if r["mode"] != "hybrid":
            continue
        out = r.get("output_text", "")
        inp = r.get("input_text", "")
        if not out or not inp:
            continue
        out_words = set(re.findall(r"[A-Z][a-z]+\s+[A-Z][a-z]+", out))
        inp_words = set(re.findall(r"[A-Z][a-z]+\s+[A-Z][a-z]+", inp))
        novel = out_words - inp_words
        for w in novel:
            surrogates_seen.append(w)
    return Counter(surrogates_seen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("results/run_v3/substitution_results.json"))
    ap.add_argument("--samples", type=Path,
                    default=Path(__file__).parent / "data" / "samples_2000.json")
    ap.add_argument("--out", type=Path, default=Path("results/run_v3/analysis.md"))
    args = ap.parse_args()

    payload = json.load(open(args.results))
    results = payload["per_doc"]
    summary = payload.get("summary", {})
    samples = load_samples(args.samples)

    lines: list[str] = []
    lines.append("# Post-Eval Analysis\n")
    lines.append(f"Generated from `{args.results}`. N docs = {len(set(r['doc_id'] for r in results))}.\n")

    # ── Bonsai stats from summary ──
    lines.append("## Bonsai SLM telemetry (hybrid mode)\n")
    if "hybrid" in summary:
        h = summary["hybrid"]
        att = h.get("bonsai_attempted", 0)
        succ = h.get("bonsai_succeeded", 0)
        echo = h.get("bonsai_echoed", 0)
        err = h.get("bonsai_errored", 0)
        rate = (succ / att * 100) if att else 0.0
        lines.append(f"- Bonsai calls attempted: **{att}**")
        lines.append(f"- Succeeded (response not echoed): **{succ}** ({rate:.1f}% of attempts)")
        lines.append(f"- Echoed input or empty: **{echo}**")
        lines.append(f"- Errored / timeout: **{err}**")
        lines.append("")

    # ── Per-locale breakdown ──
    lines.append("## Per-locale breakdown\n")
    lines.append("| Locale | n | Mode | Leak ↓ | PPL ↓ | Length pres ↑ |")
    lines.append("|--------|---|------|--------|-------|---------------|")
    by_loc = per_locale(results, samples)
    for loc in sorted(by_loc.keys()):
        for mode in ["redact", "faker", "hybrid"]:
            rs = by_loc[loc].get(mode, [])
            if not rs:
                continue
            leak = statistics.mean(r["leak_rate"] for r in rs)
            ppl = statistics.mean(r["naturalness_ppl"] for r in rs if not _isnan(r["naturalness_ppl"]))
            lp = statistics.mean(r["length_preservation"] for r in rs)
            lines.append(f"| {loc} | {len(rs)} | {mode} | {leak:.3f} | {ppl:.1f} | {lp:.3f} |")
    lines.append("")

    # ── Per-template breakdown ──
    lines.append("## Per-template breakdown\n")
    lines.append("| Template | n | Mode | Leak ↓ | PPL ↓ |")
    lines.append("|----------|---|------|--------|-------|")
    by_tpl = per_template(results, samples)
    for tpl in sorted(by_tpl.keys()):
        for mode in ["redact", "faker", "hybrid"]:
            rs = by_tpl[tpl].get(mode, [])
            if not rs:
                continue
            leak = statistics.mean(r["leak_rate"] for r in rs)
            ppl = statistics.mean(r["naturalness_ppl"] for r in rs if not _isnan(r["naturalness_ppl"]))
            lines.append(f"| {tpl} | {len(rs)} | {mode} | {leak:.3f} | {ppl:.1f} |")
    lines.append("")

    # ── Surrogate distinctness (only if --keep-text was used) ──
    has_text = any("output_text" in r for r in results)
    if has_text:
        lines.append("## Hybrid surrogate distinctness (PERSON-like)\n")
        cnt = extract_hybrid_surrogates(results)
        lines.append(f"- Unique novel two-token capitalized substrings in output: **{len(cnt)}**")
        lines.append(f"- Total occurrences: **{sum(cnt.values())}**")
        if cnt:
            lines.append(f"- Top 10 most-repeated:")
            for s, n in cnt.most_common(10):
                lines.append(f"  - `{s}` × {n}")
        lines.append("")
    else:
        lines.append("## Surrogate distinctness\n")
        lines.append("(Not available — re-run eval with `--keep-text` to populate output_text fields.)\n")

    args.out.write_text("\n".join(lines))
    print(f"Wrote {args.out}")
    print("\n--- Summary ---")
    print("\n".join(lines))


def _isnan(x):
    import math
    return isinstance(x, float) and math.isnan(x)


if __name__ == "__main__":
    main()
