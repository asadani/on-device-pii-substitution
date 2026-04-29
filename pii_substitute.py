"""On-device consistency-preserving PII substitution pipeline.

Pipeline:
    text → openai/privacy-filter (detect PII spans)
         → EntityResolver (group same-string mentions, assign stable surrogate IDs)
         → propose_surrogate dispatcher:
             - private_person / private_address / private_date → Bonsai-1.7B SLM
             - private_email / private_phone / account_number / private_url / secret → faker
         → apply_substitution (right-to-left splice)

CLI:
    python pii_substitute.py --input file.txt --mode {redact,faker,hybrid}
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Privacy-filter model is loaded lazily via _get_model()
_MODEL_CACHE: dict = {}

# Where is your local clone of PrismML/Bonsai-demo? Override with the
# BONSAI_DEMO_DIR environment variable. Default: ./bonsai-demo next to
# this file (matches the SETUP.md sibling-clone instruction).
BONSAI_DEMO_DIR = Path(
    os.environ.get("BONSAI_DEMO_DIR", str(Path(__file__).parent / "bonsai-demo"))
).expanduser().resolve()
BONSAI_RUN_SCRIPT = BONSAI_DEMO_DIR / "scripts" / "run_llama.sh"

# Privacy-filter labels (from openai/privacy-filter model card)
LABEL_PERSON = "private_person"
LABEL_ADDRESS = "private_address"
LABEL_DATE = "private_date"
LABEL_EMAIL = "private_email"
LABEL_PHONE = "private_phone"
LABEL_ACCOUNT = "account_number"
LABEL_URL = "private_url"
LABEL_SECRET = "secret"

# Labels routed to Bonsai (need contextual generation)
SLM_LABELS = {LABEL_PERSON, LABEL_ADDRESS, LABEL_DATE}
# Labels routed to faker (deterministic patterns)
FAKER_LABELS = {LABEL_EMAIL, LABEL_PHONE, LABEL_ACCOUNT, LABEL_URL, LABEL_SECRET}


@dataclass
class Span:
    label: str
    start: int
    end: int
    text: str = ""


@dataclass
class EntityRef:
    """A unique entity within a document, with all its mention offsets."""
    canonical: str           # the surface form, lowercased for grouping
    label: str
    surrogate: Optional[str] = None
    spans: list[Span] = field(default_factory=list)


# ─── Detection (privacy-filter) ───────────────────────────────────────────────

def _get_model():
    """Lazy-load the privacy-filter tokenizer and model. Requires
    transformers>=5.6.0 (see requirements.txt) for the custom
    OpenAIPrivacyFilter architecture and TokenizersBackend tokenizer class.
    """
    if "model" in _MODEL_CACHE:
        return _MODEL_CACHE["tokenizer"], _MODEL_CACHE["model"]
    import torch
    from transformers import AutoTokenizer, AutoModelForTokenClassification

    tokenizer = AutoTokenizer.from_pretrained("openai/privacy-filter")
    model = AutoModelForTokenClassification.from_pretrained(
        "openai/privacy-filter",
        dtype=torch.bfloat16,
        device_map="cpu",
    )
    model.eval()
    _MODEL_CACHE["tokenizer"] = tokenizer
    _MODEL_CACHE["model"] = model
    return tokenizer, model


def _bioes_to_spans(labels: list[str], offset_mapping) -> list[dict]:
    """Convert per-token BIOES labels to character-level spans.

    Adapted from openai-privacy-filter/infer.py:bioes_to_spans.
    """
    spans = []
    current = None
    for i, label in enumerate(labels):
        if label == "O":
            current = None
            continue
        if "-" not in label:
            current = None
            continue
        prefix, entity_type = label.split("-", 1)
        if prefix == "S":
            current = None
            cs = offset_mapping[i][0].item()
            ce = offset_mapping[i][1].item()
            if cs != ce:
                spans.append({"label": entity_type, "start": cs, "end": ce})
        elif prefix == "B":
            current = {"label": entity_type, "start_tok": i}
        elif prefix == "I":
            if current is None or current["label"] != entity_type:
                current = {"label": entity_type, "start_tok": i}
        elif prefix == "E":
            if current and current["label"] == entity_type:
                cs = offset_mapping[current["start_tok"]][0].item()
                ce = offset_mapping[i][1].item()
                if cs != ce:
                    spans.append({"label": entity_type, "start": cs, "end": ce})
            current = None
    return spans


def detect_pii(text: str, max_length: int = 8000) -> list[Span]:
    """Run privacy-filter on text, return list of detected PII spans."""
    import torch
    tokenizer, model = _get_model()
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
    )
    offset_mapping = inputs.pop("offset_mapping")[0]
    with torch.no_grad():
        outputs = model(**inputs)
    label_ids = outputs.logits[0].argmax(dim=-1).tolist()
    labels = [model.config.id2label[lid] for lid in label_ids]
    raw = _bioes_to_spans(labels, offset_mapping)
    return [Span(label=s["label"], start=s["start"], end=s["end"], text=text[s["start"]:s["end"]]) for s in raw]


# ─── Entity resolution ────────────────────────────────────────────────────────

def resolve_entities(spans: list[Span], text: str) -> dict[tuple[str, str], EntityRef]:
    """Group spans into unique entities by (lowercased surface form, label).

    Returns a dict keyed on (canonical, label) → EntityRef with all mention spans.
    """
    entities: dict[tuple[str, str], EntityRef] = {}
    for s in spans:
        s.text = text[s.start:s.end]
        key = (s.text.strip().lower(), s.label)
        if not key[0]:
            continue
        if key not in entities:
            entities[key] = EntityRef(canonical=key[0], label=s.label, spans=[])
        entities[key].spans.append(s)
    return entities


# ─── Surrogate proposers ──────────────────────────────────────────────────────

# Module-level surrogate cache; keyed on (mode, canonical_lower, label) → surrogate.
# Mode is part of the key so faker / hybrid runs do not contaminate each other
# (we run all modes back-to-back in eval_substitution.py).
_SURROGATE_CACHE: dict[tuple[str, str, str], str] = {}


def clear_surrogate_cache() -> None:
    """Reset the cross-document surrogate cache. Called between mode runs."""
    _SURROGATE_CACHE.clear()


# Telemetry for the paper: how often does Bonsai get called, succeed, or echo?
_BONSAI_STATS = {"attempted": 0, "succeeded": 0, "echoed_or_empty": 0, "errored": 0}


def get_bonsai_stats() -> dict:
    return dict(_BONSAI_STATS)


def reset_bonsai_stats() -> None:
    for k in _BONSAI_STATS:
        _BONSAI_STATS[k] = 0


def _faker_surrogate(entity: EntityRef, faker_inst) -> str:
    """Generate a deterministic surrogate for low-entropy types via faker."""
    label = entity.label
    if label == LABEL_EMAIL:
        return faker_inst.email()
    if label == LABEL_PHONE:
        return faker_inst.phone_number()
    if label == LABEL_ACCOUNT:
        # Match digit-length of original where possible
        digits = re.sub(r"\D", "", entity.canonical)
        n = max(8, min(20, len(digits) or 12))
        return "".join(str(faker_inst.random_int(0, 9)) for _ in range(n))
    if label == LABEL_URL:
        return faker_inst.url()
    if label == LABEL_SECRET:
        return "sk-" + faker_inst.lexify("?" * 24, letters="abcdefghijklmnopqrstuvwxyz0123456789")
    if label == LABEL_PERSON:
        return faker_inst.name()
    if label == LABEL_ADDRESS:
        return faker_inst.address().replace("\n", ", ")
    if label == LABEL_DATE:
        return faker_inst.date()
    return faker_inst.bothify("???###")


def _detect_locale(text: str) -> str:
    """Heuristic locale detection from character ranges and keywords."""
    if re.search(r"[一-鿿]", text):
        # CJK ideographs — could be zh or ja; check for kana to disambiguate
        if re.search(r"[぀-ヿ]", text):
            return "ja"
        return "zh"
    if re.search(r"[぀-ヿ]", text):
        return "ja"
    if re.search(r"[äöüÄÖÜß]", text) or re.search(r"\bstr(?:aße|\.)\b", text, re.IGNORECASE):
        return "de"
    if re.search(r"[ñáéíóúÁÉÍÓÚÑ]", text) or re.search(r"\b(calle|avenida|colonia)\b", text, re.IGNORECASE):
        return "es"
    return "en"


def _detect_date_format(text: str) -> str:
    """Return one of: 'mdy_slash', 'ymd_dash', 'dmy_dash_mon', 'dmy_slash', 'unknown'."""
    if re.match(r"^\d{1,2}/\d{1,2}/\d{4}$", text):
        return "mdy_slash"
    if re.match(r"^\d{4}-\d{1,2}-\d{1,2}$", text):
        return "ymd_dash"
    if re.match(r"^\d{1,2}-[A-Za-z]{3}-\d{4}$", text):
        return "dmy_dash_mon"
    if re.match(r"^\d{1,2}/\d{1,2}/\d{2,4}$", text):
        return "dmy_slash"
    return "unknown"


# Locale-conditioned few-shot demonstration pools.
# Each entry is (real_input, fake_output). For PERSON / ADDRESS, demos are
# locale-pure (Chinese demos for zh inputs, etc.). For DATE, demos are
# format-pure (MM/DD/YYYY demos for MM/DD/YYYY inputs, etc.).
PERSON_POOL: dict[str, list[tuple[str, str]]] = {
    "en": [
        ("John Carter", "Marcus Chen"),
        ("Linda Vasquez", "Olivia Brennan"),
        ("David Kim", "Theo Pemberton"),
        ("Sarah Patel", "Maya Iyer"),
        ("Robert Williams", "Daniel Foster"),
        ("Priya Krishnamurthy", "Nadia Subramanian"),
        ("Michael O'Brien", "Patrick Donovan"),
        ("Jennifer Wong", "Cynthia Park"),
    ],
    "de": [
        ("Hans Müller", "Karl Schmidt"),
        ("Anna Becker", "Lena Hoffmann"),
        ("Klaus Wagner", "Erik Krüger"),
        ("Ingrid Weber", "Petra Neumann"),
        ("Stefan Fischer", "Dietrich Bauer"),
        ("Helga Zimmermann", "Brigitte Klein"),
    ],
    "es": [
        ("Juan García", "Carlos Hernández"),
        ("María Rodríguez", "Ana Fernández"),
        ("Diego Sánchez", "Luis Castillo"),
        ("Carmen Ortiz", "Lucía Vázquez"),
        ("Roberto Jiménez", "Pablo Morales"),
        ("Sofía Ramírez", "Elena Aguilar"),
    ],
    "ja": [
        ("山田太郎", "鈴木一郎"),
        ("佐藤花子", "田中美咲"),
        ("渡辺健", "高橋翔"),
        ("中村裕子", "小林由香"),
        ("加藤博之", "斎藤大輔"),
        ("井上恵美", "松本香織"),
    ],
    "zh": [
        ("李伟", "王芳"),
        ("张敏", "刘洋"),
        ("陈杰", "黄燕"),
        ("周磊", "吴娟"),
        ("徐明", "孙丽"),
        ("郑强", "马晶"),
    ],
}

ADDRESS_POOL: dict[str, list[tuple[str, str]]] = {
    "en": [
        ("123 Main Street, Boston MA 02101", "47 Oakwood Drive, Albany NY 12203"),
        ("550 Pine Avenue, Austin TX 78701", "812 Cedar Lane, Denver CO 80202"),
        ("88 Riverside Court, Seattle WA 98109", "201 Hilltop Road, Madison WI 53703"),
        ("1500 Broadway, New York NY 10036", "742 Mission Street, San Francisco CA 94103"),
        ("36/74 MG Road, Bangalore 560001", "108 Park Street, Kolkata 700016"),
        ("21 Nehru Place, New Delhi 110019", "55 Anna Salai, Chennai 600002"),
    ],
    "de": [
        ("Hauptstraße 45, 10117 Berlin", "Lindenallee 12, 80331 München"),
        ("Bahnhofstraße 7, 60313 Frankfurt", "Marktplatz 22, 50667 Köln"),
        ("Goethestraße 18, 70173 Stuttgart", "Schillerweg 33, 04109 Leipzig"),
        ("Kaiserplatz 5, 20095 Hamburg", "Mozartgasse 14, 01067 Dresden"),
    ],
    "es": [
        ("Calle Reforma 123, 06600 CDMX", "Avenida Insurgentes 456, 03100 CDMX"),
        ("Paseo Montejo 89, 97000 Mérida", "Calle Hidalgo 12, 44100 Guadalajara"),
        ("Avenida Juárez 78, 64000 Monterrey", "Calle Morelos 56, 91000 Xalapa"),
        ("Boulevard Kukulcán 15, 77500 Cancún", "Avenida 5 de Mayo 234, 72000 Puebla"),
    ],
    "ja": [
        ("東京都新宿区西新宿1-1-1", "大阪府大阪市北区梅田2-2-2"),
        ("神奈川県横浜市中区本町3-3-3", "京都府京都市左京区岡崎4-4-4"),
        ("愛知県名古屋市中区栄5-5-5", "福岡県福岡市博多区博多駅前6-6-6"),
        ("北海道札幌市中央区大通7-7-7", "兵庫県神戸市中央区三宮8-8-8"),
    ],
    "zh": [
        ("北京市朝阳区建国路1号", "上海市浦东新区世纪大道100号"),
        ("广东省广州市天河区珠江新城200号", "四川省成都市锦江区春熙路300号"),
        ("浙江省杭州市西湖区文三路50号", "江苏省南京市玄武区中山路80号"),
        ("湖北省武汉市江汉区解放大道150号", "陕西省西安市雁塔区高新路250号"),
    ],
}

DATE_POOL: dict[str, list[tuple[str, str]]] = {
    "mdy_slash": [
        ("03/15/1985", "11/22/1987"),
        ("07/04/1992", "09/18/1994"),
        ("12/31/2001", "02/14/2003"),
        ("06/30/1978", "08/12/1980"),
        ("01/20/2010", "04/05/2012"),
    ],
    "ymd_dash": [
        ("1992-07-04", "1990-03-19"),
        ("1985-11-22", "1987-09-15"),
        ("2003-02-14", "2001-12-31"),
        ("1978-08-12", "1976-04-30"),
    ],
    "dmy_dash_mon": [
        ("11-Jul-1998", "14-Sep-2001"),
        ("23-Mar-1985", "07-Nov-1988"),
        ("18-Dec-1992", "02-Jun-1995"),
        ("05-Aug-2003", "29-Jan-2006"),
    ],
    "dmy_slash": [
        ("15/03/1985", "22/11/1987"),
        ("04/07/1992", "18/09/1994"),
        ("31/12/2001", "14/02/2003"),
    ],
    "unknown": [
        ("1985", "1988"),
        ("March 1992", "July 1995"),
    ],
}


def _select_demos(pool: list[tuple[str, str]], original: str, k: int = 3) -> list[tuple[str, str]]:
    """Deterministically pick k demos from the pool, seeded by the original
    input so the same entity always gets the same demos (cache-friendly) but
    different entities sample different subsets (diversity-preserving)."""
    import hashlib, random as _random
    seed = int(hashlib.md5(original.encode("utf-8")).hexdigest()[:8], 16)
    rng = _random.Random(seed)
    if len(pool) <= k:
        return list(pool)
    return rng.sample(pool, k)


def _bonsai_surrogate(entity: EntityRef, bonsai_size: str = "1.7B",
                      bonsai_family: str = "bonsai", timeout: int = 180) -> Optional[str]:
    """Ask Bonsai SLM to generate a context-appropriate surrogate.

    Uses **locale-conditioned rotating few-shot demos**: the input's locale
    (or date format) selects the demo pool; a per-input deterministic hash
    selects which 3 demos from that pool appear in the prompt. This breaks
    the "always copy the first demo" failure mode of small SLMs at extreme
    quantization.
    """
    label = entity.label
    original = entity.canonical

    if label == LABEL_PERSON:
        loc = _detect_locale(original)
        pool = PERSON_POOL.get(loc, PERSON_POOL["en"])
        demos = _select_demos(pool, original, k=3)
        demo_block = "\n\n".join(f"Real: {r}\nFake: {f}" for r, f in demos)
        prompt = (
            "You replace real names with completely different fake names. "
            "Keep the same gender and cultural style. Output only the new name.\n\n"
            f"{demo_block}\n\n"
            f"Real: {original}\nFake:"
        )
    elif label == LABEL_ADDRESS:
        loc = _detect_locale(original)
        pool = ADDRESS_POOL.get(loc, ADDRESS_POOL["en"])
        demos = _select_demos(pool, original, k=3)
        demo_block = "\n\n".join(f"Real: {r}\nFake: {f}" for r, f in demos)
        prompt = (
            "You replace real addresses with completely different fake addresses "
            "in the same country. Output only the new address on one line.\n\n"
            f"{demo_block}\n\n"
            f"Real: {original}\nFake:"
        )
    elif label == LABEL_DATE:
        fmt = _detect_date_format(original)
        pool = DATE_POOL.get(fmt, DATE_POOL["unknown"])
        demos = _select_demos(pool, original, k=3)
        demo_block = "\n\n".join(f"Real: {r}\nFake: {f}" for r, f in demos)
        prompt = (
            "You replace real dates with different fake dates within 5 years, "
            "using the exact same format. Output only the new date.\n\n"
            f"{demo_block}\n\n"
            f"Real: {original}\nFake:"
        )
    else:
        return None

    env = os.environ.copy()
    env["BONSAI_FAMILY"] = bonsai_family
    env["BONSAI_MODEL"] = bonsai_size

    cmd = [str(BONSAI_RUN_SCRIPT), "-p", prompt, "-n", "30", "--single-turn"]
    _BONSAI_STATS["attempted"] += 1
    try:
        result = subprocess.run(
            cmd, env=env, capture_output=True, stdin=subprocess.DEVNULL,
            text=True, timeout=timeout, cwd=str(BONSAI_DEMO_DIR),
        )
        if result.returncode != 0:
            _BONSAI_STATS["errored"] += 1
            return None
        response = _extract_bonsai_response(result.stdout)
        if not response:
            _BONSAI_STATS["echoed_or_empty"] += 1
            return None
        response = re.sub(r"^(Fake|Real)\s*:\s*", "", response, flags=re.IGNORECASE).strip()
        if response.lower().strip() == original.lower().strip():
            _BONSAI_STATS["echoed_or_empty"] += 1
            return None
        if not re.search(r"\w", response):
            _BONSAI_STATS["echoed_or_empty"] += 1
            return None
        _BONSAI_STATS["succeeded"] += 1
        return response
    except (subprocess.TimeoutExpired, Exception):
        _BONSAI_STATS["errored"] += 1
        return None


def _extract_bonsai_response(raw: str) -> Optional[str]:
    """Pull just the model's generated answer out of llama-cli stdout.

    Adapted from eval.py:_clean_response. Returns the first non-empty line of
    actual model output, stripping spinner chars and trailing junk.
    """
    lines = raw.splitlines()
    end_idx = len(lines)
    for i, ln in enumerate(lines):
        if re.match(r"^\s*\[ Prompt:", ln) or ln.strip() == "Exiting...":
            end_idx = i
            break
    start_idx = 0
    for i, ln in enumerate(lines[:end_idx]):
        if re.match(r"^> ", ln):
            start_idx = i + 1
            while start_idx < end_idx and lines[start_idx].strip() == "":
                start_idx += 1
            break
    response_lines = lines[start_idx:end_idx]
    response_lines = [
        re.sub(r"llama_memory_breakdown_print:.*$", "", ln) for ln in response_lines
    ]
    # Strip llama-cli's animated spinner sequences (|, -, \, /) wherever they
    # appear — they get streamed between generated tokens, not just at the
    # start of the response.
    cleaned_lines = []
    for ln in response_lines:
        # Remove runs of 2+ spinner-only chars (with optional spaces between them).
        ln = re.sub(r"(?:[|\-\\/]\s*){2,}", " ", ln)
        # Collapse multiple spaces.
        ln = re.sub(r"\s{2,}", " ", ln).strip()
        cleaned_lines.append(ln)
    cleaned = "\n".join(cleaned_lines).strip()
    for ln in cleaned.splitlines():
        ln = ln.strip().strip('"').strip("'")
        if ln:
            return ln
    return None


def propose_surrogate(entity: EntityRef, mode: str, faker_inst,
                      bonsai_size: str = "1.7B",
                      bonsai_family: str = "bonsai") -> str:
    """Dispatch to the right proposer based on mode and label.

    mode = 'redact'  → "[LABEL]"
    mode = 'faker'   → faker for everything
    mode = 'hybrid'  → Bonsai/Ternary SLM for SLM_LABELS, faker for FAKER_LABELS
    """
    if mode == "redact":
        return f"[{entity.label.upper()}]"

    # Cache key includes family so bonsai vs ternary runs don't collide.
    key = (mode, bonsai_family, entity.canonical, entity.label)
    if key in _SURROGATE_CACHE:
        return _SURROGATE_CACHE[key]

    surrogate: Optional[str] = None
    if mode == "hybrid" and entity.label in SLM_LABELS:
        surrogate = _bonsai_surrogate(entity, bonsai_size=bonsai_size,
                                      bonsai_family=bonsai_family)
    if surrogate is None:
        surrogate = _faker_surrogate(entity, faker_inst)

    _SURROGATE_CACHE[key] = surrogate
    return surrogate


# ─── Splicing ─────────────────────────────────────────────────────────────────

def apply_substitution(text: str, entities: dict[tuple[str, str], EntityRef]) -> str:
    """Replace each span with its entity's surrogate, right-to-left.

    Preserves leading/trailing whitespace from the original span: the
    privacy-filter sometimes includes a leading space in its character span,
    which would otherwise produce ugly output like "is[PRIVATE_PERSON]".
    """
    all_spans: list[tuple[Span, str]] = []
    for ent in entities.values():
        if ent.surrogate is None:
            continue
        for s in ent.spans:
            all_spans.append((s, ent.surrogate))
    for span, surrogate in sorted(all_spans, key=lambda x: x[0].start, reverse=True):
        original = text[span.start:span.end]
        leading_ws = original[:len(original) - len(original.lstrip())]
        trailing_ws = original[len(original.rstrip()):]
        replacement = leading_ws + surrogate + trailing_ws
        text = text[:span.start] + replacement + text[span.end:]
    return text


# ─── End-to-end ───────────────────────────────────────────────────────────────

def substitute(text: str, mode: str = "hybrid", bonsai_size: str = "1.7B",
               bonsai_family: str = "bonsai",
               spans: Optional[list[Span]] = None) -> tuple[str, dict, dict]:
    """Run the full pipeline on a single text.

    Returns (output_text, entities_dict, timing_dict).
    If `spans` is provided, skip detection (used to substitute against ground-truth
    spans for downstream-utility experiments).
    """
    from faker import Faker
    faker_inst = Faker()
    Faker.seed(0)

    timings = {}
    t0 = time.perf_counter()
    if spans is None:
        spans = detect_pii(text)
    timings["detect_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    entities = resolve_entities(spans, text)
    for ent in entities.values():
        ent.surrogate = propose_surrogate(ent, mode, faker_inst,
                                          bonsai_size=bonsai_size,
                                          bonsai_family=bonsai_family)
    timings["surrogate_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    output = apply_substitution(text, entities)
    timings["splice_s"] = time.perf_counter() - t0
    timings["total_s"] = sum(timings.values())

    return output, entities, timings


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="On-device PII substitution pipeline")
    parser.add_argument("--input", type=Path, required=True, help="Input text file")
    parser.add_argument("--output", type=Path, default=None, help="Output file (stdout if omitted)")
    parser.add_argument("--mode", choices=["redact", "faker", "hybrid"], default="hybrid")
    parser.add_argument("--bonsai-size", choices=["1.7B", "4B", "8B"], default="1.7B")
    parser.add_argument("--bonsai-family", choices=["bonsai", "ternary"], default="bonsai")
    args = parser.parse_args()

    text = args.input.read_text(encoding="utf-8")
    output, entities, timings = substitute(text, mode=args.mode,
                                            bonsai_size=args.bonsai_size,
                                            bonsai_family=args.bonsai_family)

    if args.output:
        args.output.write_text(output, encoding="utf-8")
        print(f"Wrote {len(output)} chars to {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(output)

    print(f"\n[stats] mode={args.mode} entities={len(entities)} "
          f"detect={timings['detect_s']:.1f}s surrogate={timings['surrogate_s']:.1f}s "
          f"splice={timings['splice_s']:.2f}s total={timings['total_s']:.1f}s",
          file=sys.stderr)


if __name__ == "__main__":
    main()
