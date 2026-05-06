"""Surrogate-distinctness analysis for the hybrid-vs-faker NER root cause.

Replicates the en-locale 160-doc training set used in the matched 160/40
NER experiment (results/ner_v3_matched/) and counts distinct surrogate
values produced by each substitution mode, broken down by PII label.

The doc loading and stratified split mirror eval_ner_utility.py exactly,
so the surrogates collected here are the same ones the NER trainer saw.

Output: results/surrogate_distinctness.json with per-(mode, label):
    n_total_mentions, n_unique_surrogates, type_token_ratio,
    top-5 most-repeated surrogates.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from collections import Counter

from faker import Faker

from pii_substitute import (
    apply_substitution, clear_surrogate_cache, propose_surrogate,
    resolve_entities,
)
from eval_ner_utility import (
    DEFAULT_SAMPLES_PATH, load_english_docs, stratified_split,
)


def collect_surrogates(train_docs, mode, bonsai_size="1.7B",
                       bonsai_family="bonsai"):
    """Run substitution per-doc and gather surrogate -> mention-count by label.

    Replicates substitute_with_spans's per-doc Faker re-seeding, so the
    distribution we measure here is the exact distribution the NER trainer
    consumed for the matching mode.
    """
    by_label: dict[str, list[str]] = {}
    for i, d in enumerate(train_docs):
        faker_inst = Faker()
        Faker.seed(0)
        entities = resolve_entities(d["gt_spans"], d["text"])
        for ent in entities.values():
            ent.surrogate = propose_surrogate(
                ent, mode, faker_inst,
                bonsai_size=bonsai_size, bonsai_family=bonsai_family,
            )
            if ent.surrogate is None:
                continue
            n_mentions = len(ent.spans)
            by_label.setdefault(ent.label, []).extend(
                [ent.surrogate] * n_mentions
            )
        if (i + 1) % 20 == 0:
            print(f"  ... processed {i + 1}/{len(train_docs)} docs")
    return by_label


def stats(values: list[str]) -> dict:
    n_total = len(values)
    cnt = Counter(values)
    n_unique = len(cnt)
    ttr = n_unique / n_total if n_total else 0.0
    return {
        "n_total_mentions": n_total,
        "n_unique_surrogates": n_unique,
        "type_token_ratio": round(ttr, 4),
        "top5_most_repeated": cnt.most_common(5),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-max", type=int, default=200,
                    help="En-locale doc cap before split. 200 matches the "
                         "matched-160/40 NER experiment.")
    ap.add_argument("--samples-path", type=Path, default=DEFAULT_SAMPLES_PATH)
    ap.add_argument("--modes", nargs="+", default=["faker", "hybrid"],
                    choices=["faker", "hybrid"])
    ap.add_argument("--bonsai-size", choices=["1.7B", "4B", "8B"], default="1.7B")
    ap.add_argument("--bonsai-family", choices=["bonsai", "ternary"], default="bonsai")
    ap.add_argument("--output", type=Path,
                    default=Path("results/surrogate_distinctness.json"))
    args = ap.parse_args()

    print(f"Loading English docs from {args.samples_path} (n_max={args.n_max})...")
    docs = load_english_docs(path=args.samples_path, n_max=args.n_max)
    print(f"  Got {len(docs)} en docs")
    train_docs, _ = stratified_split(docs)
    print(f"  Train split: {len(train_docs)} docs "
          f"(matches NER training set when n_max=200)\n")

    out: dict = {
        "n_train_docs": len(train_docs),
        "n_max_loaded": args.n_max,
        "modes": {},
    }
    for mode in args.modes:
        print(f"=== Mode: {mode} ===")
        clear_surrogate_cache()
        t0 = time.time()
        by_label = collect_surrogates(
            train_docs, mode,
            bonsai_size=args.bonsai_size, bonsai_family=args.bonsai_family,
        )
        elapsed = time.time() - t0
        per_label = {label: stats(vs) for label, vs in by_label.items()}
        out["modes"][mode] = {
            "elapsed_s": round(elapsed, 1),
            "per_label": per_label,
        }
        print(f"  ({elapsed:.1f}s)")
        for label in sorted(per_label.keys()):
            s = per_label[label]
            top1 = s["top5_most_repeated"][0] if s["top5_most_repeated"] else None
            print(f"    {label:8s}  n_total={s['n_total_mentions']:4d}  "
                  f"n_unique={s['n_unique_surrogates']:4d}  "
                  f"TTR={s['type_token_ratio']:.3f}  top1={top1}")
        print()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
