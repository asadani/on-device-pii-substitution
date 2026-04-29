"""Downstream NER utility experiment for the substitution paper.

Question: does training a NER model on PII-substituted documents preserve the
F1 it would achieve if trained on the original documents?

Setup:
  - English subset of the bundled samples JSON (default:
    data/samples_2000.json; ~1159 English docs out of 2000 total).
  - Stratified 80/20 split by locale (test docs are ALWAYS in their
    original form; the substitution affects only training data).
  - Ground-truth PII spans are derived from each doc's `pii_gt` dict by
    locating each value as a substring in the source text.
  - Single binary "PII" label (multiclass would need more data).
  - For each mode (original / redact / faker / hybrid), we substitute the
    train docs (using ground-truth spans for clean experimental separation
    from detection error), train a blank spaCy English NER pipeline, and
    evaluate span-level precision/recall/F1 on the held-out original docs.

Outputs:
  - results/ner_utility.json — raw per-mode F1 / precision / recall.
  - results/ner_utility.md — markdown table for the paper.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from pii_substitute import (
    LABEL_ACCOUNT, LABEL_ADDRESS, LABEL_DATE, LABEL_EMAIL, LABEL_PERSON,
    LABEL_PHONE, LABEL_SECRET, LABEL_URL, EntityRef, Span,
    apply_substitution, clear_surrogate_cache, propose_surrogate,
    resolve_entities, get_bonsai_stats, reset_bonsai_stats,
)


PII_LABEL = "PII"  # Single binary label across all 8 PII types


# ─── Ground-truth span extraction ─────────────────────────────────────────────

def _gt_value_to_label(key: str) -> str:
    """Map a pii_gt dict key to one of the 8 privacy-filter labels."""
    k = key.lower()
    if any(t in k for t in ("name", "agent_name", "officer_name")):
        return LABEL_PERSON
    if "address" in k or "street" in k:
        return LABEL_ADDRESS
    if any(t in k for t in ("date_of_birth", "dob", "birth_date")):
        return LABEL_DATE
    if "email" in k:
        return LABEL_EMAIL
    if "phone" in k or "mobile" in k or "fax" in k:
        return LABEL_PHONE
    if "url" in k or "website" in k:
        return LABEL_URL
    if "secret" in k or "api_key" in k or "token" in k:
        return LABEL_SECRET
    # Numeric IDs — SSN, account, license, policy, plate, vin
    return LABEL_ACCOUNT


def gt_spans(text: str, pii_gt: dict) -> list[Span]:
    """Convert a pii_gt dict to char-level Spans by substring search.

    Some GT values are lists (e.g., multiple VINs) — handle both shapes.
    """
    spans: list[Span] = []
    text_lower = text.lower()
    for key, val in pii_gt.items():
        if val is None:
            continue
        values = val if isinstance(val, list) else [val]
        label = _gt_value_to_label(key)
        for v in values:
            v = str(v).strip()
            if len(v) < 3:
                continue
            v_lower = v.lower()
            pos = 0
            while True:
                i = text_lower.find(v_lower, pos)
                if i < 0:
                    break
                spans.append(Span(label=label, start=i, end=i + len(v), text=text[i:i + len(v)]))
                pos = i + 1
    # Sort by start, drop overlaps (keep longest)
    spans.sort(key=lambda s: (s.start, -(s.end - s.start)))
    deduped: list[Span] = []
    for s in spans:
        if deduped and s.start < deduped[-1].end:
            continue
        deduped.append(s)
    return deduped


# ─── Substitution with span tracking ──────────────────────────────────────────

def substitute_with_spans(text: str, spans: list[Span], mode: str,
                          bonsai_size: str = "1.7B",
                          bonsai_family: str = "bonsai") -> tuple[str, list[Span]]:
    """Substitute PII spans in text and return new text + new spans.

    For training a NER model on substituted data, we need to know where the
    PII labels live in the OUTPUT text — not the input.
    """
    from faker import Faker
    faker_inst = Faker()
    Faker.seed(0)

    entities = resolve_entities(spans, text)
    for ent in entities.values():
        ent.surrogate = propose_surrogate(ent, mode, faker_inst,
                                          bonsai_size=bonsai_size,
                                          bonsai_family=bonsai_family)
    output = apply_substitution(text, entities)

    # Recompute spans in the output text by locating each entity's surrogate.
    new_spans: list[Span] = []
    for ent in entities.values():
        if ent.surrogate is None:
            continue
        # Find all occurrences of the surrogate in the output text. We look for
        # exactly as many occurrences as there were original mentions, in order.
        n_expected = len(ent.spans)
        found = _find_all_occurrences(output, ent.surrogate)
        for pos in found[:n_expected]:
            new_spans.append(Span(
                label=ent.label,
                start=pos,
                end=pos + len(ent.surrogate),
                text=ent.surrogate,
            ))
    new_spans.sort(key=lambda s: s.start)
    return output, new_spans


def _find_all_occurrences(text: str, needle: str) -> list[int]:
    if not needle:
        return []
    out = []
    pos = 0
    while True:
        i = text.find(needle, pos)
        if i < 0:
            break
        out.append(i)
        pos = i + 1
    return out


# ─── spaCy training ───────────────────────────────────────────────────────────

def make_doc_with_ents(nlp, text: str, spans: list[Span]):
    """Create a spaCy Doc with PII span annotations."""
    from spacy.tokens import Span as SpacySpan
    doc = nlp.make_doc(text)
    ents = []
    for s in spans:
        # spaCy aligns char spans to tokens; missing alignment yields None
        span_obj = doc.char_span(s.start, s.end, label=PII_LABEL, alignment_mode="expand")
        if span_obj is not None:
            ents.append(span_obj)
    # Drop overlapping entities (spaCy requires non-overlapping)
    ents.sort(key=lambda x: (x.start, -(x.end - x.start)))
    deduped = []
    for e in ents:
        if deduped and e.start < deduped[-1].end:
            continue
        deduped.append(e)
    try:
        doc.ents = deduped
    except ValueError:
        # Final fallback: drop all if still overlapping
        doc.ents = []
    return doc


def train_ner_pipeline(train_pairs: list[tuple[str, list[Span]]], n_iter: int = 30,
                       seed: int = 0, batch_size: int = 8, verbose: bool = False):
    """Train a blank English NER pipeline on (text, spans) pairs.

    Uses spaCy 3.x's required pattern: get the optimizer from initialize() and
    pass sgd=optimizer to nlp.update() for actual gradient updates.

    `spacy.util.fix_random_seed(seed)` controls Python random + numpy +
    torch (if installed) so the seed is actually honoured by the thinc
    backend, not just by our Python-level shuffling.
    """
    import spacy
    from spacy.training import Example
    from spacy.util import minibatch
    spacy.util.fix_random_seed(seed)
    random.seed(seed)

    nlp = spacy.blank("en")
    ner = nlp.add_pipe("ner")
    ner.add_label(PII_LABEL)

    examples: list = []
    for text, spans in train_pairs:
        ref_doc = make_doc_with_ents(nlp, text, spans)
        if not ref_doc.ents:
            continue
        examples.append(Example(nlp.make_doc(text), ref_doc))

    if not examples:
        raise ValueError("No training examples with valid annotations.")

    optimizer = nlp.initialize(lambda: examples)
    for itn in range(n_iter):
        random.shuffle(examples)
        losses: dict = {}
        for batch in minibatch(examples, size=batch_size):
            nlp.update(batch, sgd=optimizer, drop=0.2, losses=losses)
        if verbose and (itn % 5 == 0 or itn == n_iter - 1):
            print(f"    iter {itn}: loss={losses.get('ner', 0):.2f}")
    return nlp


def evaluate_ner(nlp, eval_pairs: list[tuple[str, list[Span]]]) -> dict:
    """Span-level precision/recall/F1 against ground-truth spans.

    A predicted span counts as a TP if it overlaps any GT span (label-agnostic).
    """
    tp = fp = fn = 0
    for text, gt in eval_pairs:
        doc = nlp(text)
        gt_set = [(g.start, g.end) for g in gt]
        pred_set = [(e.start_char, e.end_char) for e in doc.ents]
        # TP: each pred that overlaps any GT
        for ps, pe in pred_set:
            if any(ps < ge and pe > gs for gs, ge in gt_set):
                tp += 1
            else:
                fp += 1
        # FN: each GT not overlapped by any pred
        for gs, ge in gt_set:
            if not any(ps < ge and pe > gs for ps, pe in pred_set):
                fn += 1
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f1}


# ─── Data loading ─────────────────────────────────────────────────────────────

DEFAULT_SAMPLES_PATH = Path(__file__).parent / "data" / "samples_2000.json"


def load_english_docs(path: Path = DEFAULT_SAMPLES_PATH, n_max: Optional[int] = None):
    """Load English-locale docs from the samples JSON with their pii_gt and gt_spans."""
    records = json.load(open(path))
    docs = []
    for r in records:
        loc = r.get("metadata", {}).get("locale", "")
        if not loc.lower().startswith("en"):
            continue
        spans = gt_spans(r["text"], r.get("pii_gt", {}))
        if not spans:
            continue
        docs.append({
            "id": r.get("id", len(docs)),
            "text": r["text"],
            "pii_gt": r.get("pii_gt", {}),
            "gt_spans": spans,
            "locale": loc,
        })
        if n_max and len(docs) >= n_max:
            break
    return docs


def stratified_split(docs, test_frac=0.2, seed=0):
    """Stratified split by locale. Returns (train_docs, test_docs)."""
    by_loc: dict[str, list] = defaultdict(list)
    for d in docs:
        by_loc[d["locale"]].append(d)
    random.seed(seed)
    train, test = [], []
    for loc, items in by_loc.items():
        items = list(items)
        random.shuffle(items)
        n_test = max(1, int(len(items) * test_frac))
        test.extend(items[:n_test])
        train.extend(items[n_test:])
    return train, test


# ─── Main ─────────────────────────────────────────────────────────────────────

@dataclass
class SeedResult:
    mode: str
    seed: int
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int
    train_seconds: float


@dataclass
class NerResult:
    """Aggregated across seeds for a single mode."""
    mode: str
    n_train_docs: int
    n_test_docs: int
    n_train_spans: int
    n_seeds: int
    f1_mean: float
    f1_std: float
    precision_mean: float
    precision_std: float
    recall_mean: float
    recall_std: float
    per_seed: list[SeedResult] = field(default_factory=list)
    bonsai_attempted: int = 0
    bonsai_succeeded: int = 0
    bonsai_echoed: int = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-max", type=int, default=None,
                    help="Max docs to load (default: all English docs in samples_2000.json)")
    ap.add_argument("--samples-path", type=Path, default=DEFAULT_SAMPLES_PATH,
                    help="Path to samples JSON (default: data/samples_2000.json)")
    ap.add_argument("--n-iter", type=int, default=30, help="spaCy training iterations")
    ap.add_argument("--verbose-train", action="store_true", help="Print training loss per epoch")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4],
                    help="Random seeds for spaCy training (each yields one F1 value; "
                         "report mean +/- SD across seeds)")
    ap.add_argument("--modes", nargs="+", default=["original", "redact", "faker", "hybrid"],
                    choices=["original", "redact", "faker", "hybrid"])
    ap.add_argument("--bonsai-size", choices=["1.7B", "4B", "8B"], default="1.7B")
    ap.add_argument("--bonsai-family", choices=["bonsai", "ternary"], default="bonsai")
    ap.add_argument("--output", type=Path, default=Path("results"))
    args = ap.parse_args()

    print(f"Loading English docs from {args.samples_path}...")
    docs = load_english_docs(path=args.samples_path, n_max=args.n_max)
    print(f"  Got {len(docs)} English docs (locales: {Counter(d['locale'] for d in docs)})")
    print(f"  GT spans per doc: avg={sum(len(d['gt_spans']) for d in docs)/len(docs):.1f}")

    train_docs, test_docs = stratified_split(docs)
    print(f"Split: {len(train_docs)} train / {len(test_docs)} test\n")

    # Test set is ALWAYS original docs with GT spans — measures generalization
    eval_pairs = [(d["text"], d["gt_spans"]) for d in test_docs]

    results: list[NerResult] = []
    for mode in args.modes:
        print(f"\n{'='*60}\n  Mode: {mode}  (seeds: {args.seeds})\n{'='*60}")
        clear_surrogate_cache()
        reset_bonsai_stats()

        # Build the training set ONCE per mode (deterministic given mode +
        # bonsai_family). Spacy training stochasticity will then be the only
        # source of variance, isolated to a single comparison axis.
        train_pairs: list[tuple[str, list[Span]]] = []
        n_total_spans = 0
        for d in train_docs:
            if mode == "original":
                text, spans = d["text"], d["gt_spans"]
            else:
                text, spans = substitute_with_spans(
                    d["text"], d["gt_spans"], mode,
                    bonsai_size=args.bonsai_size,
                    bonsai_family=args.bonsai_family,
                )
            train_pairs.append((text, spans))
            n_total_spans += len(spans)

        print(f"  Built {len(train_pairs)} training examples with {n_total_spans} total spans.")

        # Train once per seed
        per_seed: list[SeedResult] = []
        for seed in args.seeds:
            t0 = time.perf_counter()
            try:
                nlp = train_ner_pipeline(train_pairs, n_iter=args.n_iter,
                                          seed=seed, verbose=args.verbose_train)
            except Exception as e:
                print(f"  [seed={seed}] TRAIN ERROR: {type(e).__name__}: {e}")
                continue
            train_s = time.perf_counter() - t0
            m = evaluate_ner(nlp, eval_pairs)
            sr = SeedResult(
                mode=mode, seed=seed,
                precision=m["precision"], recall=m["recall"], f1=m["f1"],
                tp=m["tp"], fp=m["fp"], fn=m["fn"],
                train_seconds=train_s,
            )
            per_seed.append(sr)
            print(f"  [seed={seed}] P={m['precision']:.3f} R={m['recall']:.3f} "
                  f"F1={m['f1']:.3f}  (tp={m['tp']} fp={m['fp']} fn={m['fn']}, "
                  f"train={train_s:.0f}s)")

        if not per_seed:
            print(f"  no successful runs for mode={mode}, skipping.")
            continue

        f1s = [s.f1 for s in per_seed]
        ps  = [s.precision for s in per_seed]
        rs  = [s.recall for s in per_seed]

        def _std(xs):
            return statistics.pstdev(xs) if len(xs) >= 2 else 0.0

        bs = get_bonsai_stats()
        r = NerResult(
            mode=mode,
            n_train_docs=len(train_pairs),
            n_test_docs=len(eval_pairs),
            n_train_spans=n_total_spans,
            n_seeds=len(per_seed),
            f1_mean=statistics.mean(f1s), f1_std=_std(f1s),
            precision_mean=statistics.mean(ps), precision_std=_std(ps),
            recall_mean=statistics.mean(rs), recall_std=_std(rs),
            per_seed=per_seed,
            bonsai_attempted=bs["attempted"], bonsai_succeeded=bs["succeeded"],
            bonsai_echoed=bs["echoed_or_empty"],
        )
        results.append(r)
        print(f"  >> mean: P={r.precision_mean:.3f}+/-{r.precision_std:.3f}  "
              f"R={r.recall_mean:.3f}+/-{r.recall_std:.3f}  "
              f"F1={r.f1_mean:.3f}+/-{r.f1_std:.3f}")

    # Save
    args.output.mkdir(parents=True, exist_ok=True)
    json_path = args.output / "ner_utility.json"
    json_path.write_text(json.dumps({
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_train": len(train_docs),
        "n_test": len(test_docs),
        "seeds": args.seeds,
        "results": [asdict(r) for r in results],
    }, indent=2))
    print(f"\nWrote {json_path}")

    md_path = args.output / "ner_utility.md"
    with open(md_path, "w") as f:
        f.write("# Downstream NER Utility — span-level F1 on held-out original docs\n\n")
        f.write(f"Train set: {len(train_docs)} docs (substituted per mode), "
                f"test set: {len(test_docs)} docs (always original).\n")
        f.write(f"Seeds: {args.seeds} (n={len(args.seeds)}). "
                f"Reported as **mean ± SD** across seeds.\n\n")
        f.write("| Mode | Train spans | Precision | Recall | F1 |\n")
        f.write("|------|-------------|-----------|--------|----|\n")
        for r in results:
            f.write(
                f"| {r.mode} | {r.n_train_spans} | "
                f"{r.precision_mean:.3f} ± {r.precision_std:.3f} | "
                f"{r.recall_mean:.3f} ± {r.recall_std:.3f} | "
                f"**{r.f1_mean:.3f} ± {r.f1_std:.3f}** |\n"
            )
        f.write("\n## Per-seed F1 (raw)\n\n")
        f.write("| Mode | " + " | ".join(f"seed={s}" for s in args.seeds) + " |\n")
        f.write("|------|" + "|".join(["----"] * len(args.seeds)) + "|\n")
        for r in results:
            seed_to_f1 = {s.seed: s.f1 for s in r.per_seed}
            row = " | ".join(f"{seed_to_f1.get(s, float('nan')):.3f}" for s in args.seeds)
            f.write(f"| {r.mode} | {row} |\n")
    print(f"Wrote {md_path}")

    print("\nSummary (mean ± SD over n seeds):")
    for r in results:
        print(f"  {r.mode:10s}  F1={r.f1_mean:.3f}±{r.f1_std:.3f}  "
              f"P={r.precision_mean:.3f}±{r.precision_std:.3f}  "
              f"R={r.recall_mean:.3f}±{r.recall_std:.3f}  "
              f"(n={r.n_seeds})")


if __name__ == "__main__":
    main()
