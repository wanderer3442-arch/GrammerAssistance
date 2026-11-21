"""Prepare a CoEdit GEC training subset (minimal-edit, single-sentence, ~5k).

CoEdit `src` includes instruction prefixes ("Remove all grammatical errors from
this text: ...") that must be stripped. We keep only single-sentence rows where
the correction is a MINIMAL edit (small word-level change), so the model learns
to correct -- not rewrite. Dedupes against the held-out test split + train pool
so our benchmark stays honest. Caps at N.
"""
import csv
import re
from pathlib import Path
from datasets import load_dataset

ROOT = Path(__file__).resolve().parent.parent
N = 5000
MAX_WORDS = 40
MAX_EDIT_RATIO = 0.4   # keep pairs where <=40% of words changed

PREFIX_RE = re.compile(
    r"^(Remove all grammatical errors from this text:|"
    r"Improve the grammaticality( of this sentence)?:|"
    r"Fix grammaticality in this sentence:|"
    r"Improve the grammar of this text:|"
    r"Fix the grammaticality in this text:|"
    r"Improve grammaticality in this sentence:|"
    r"Fix the grammaticality in this sentence:|"
    r"Remove the grammatical errors from this text:)\s*",
    re.I,
)


def strip_prefix(s: str) -> str:
    prev = None
    while prev != s:
        prev = s
        s = PREFIX_RE.sub("", s).strip()
    return s


def words(s):
    return s.split()


def is_minimal_edit(src, tgt):
    """Keep only if word-level edit distance is a small fraction of length."""
    sw, tw = words(src), words(tgt)
    if not sw or not tw:
        return False
    # cheap alignment-independent estimate: len(lcs) based
    a, b = set(sw), set(tw)
    inter = len(a & b)
    union = len(a | b)
    if union == 0:
        return False
    jaccard_sim = inter / union
    # minimal edit => high word overlap (>= 60%)
    return jaccard_sim >= (1 - MAX_EDIT_RATIO)


def main():
    print("loading CoEdit ...")
    ds = load_dataset("grammarly/coedit", split="train")
    gec = ds.filter(lambda r: r["task"] == "gec")
    print(f"CoEdit GEC rows: {len(gec)}")

    # existing held-out + train data to avoid overlap
    existing = set()
    for rel in ("eval/test_split.csv", "data/train_pool.csv"):
        p = ROOT / rel
        if p.exists():
            for r in csv.DictReader(open(p, encoding="utf-8")):
                existing.add((r["input"].strip(), r["target"].strip()))

    seen = set()
    kept = []
    stats = {"stripped": 0, "too_long": 0, "identical": 0, "big_edit": 0, "overlap": 0}

    for r in gec:
        src = strip_prefix(r["src"]).strip()
        tgt = (r["tgt"] or "").strip()
        if src != r["src"].strip():
            stats["stripped"] += 1
        if not src or not tgt:
            continue
        if len(words(src)) > MAX_WORDS or len(words(tgt)) > MAX_WORDS:
            stats["too_long"] += 1
            continue
        if src == tgt:
            stats["identical"] += 1
            continue
        if not is_minimal_edit(src, tgt):
            stats["big_edit"] += 1
            continue
        key = (src, tgt)
        skey = (src.lower(), tgt.lower())
        if key in seen or skey in seen or key in existing or skey in existing:
            stats["overlap"] += 1
            continue
        seen.add(key)
        kept.append({"input": src, "target": tgt})
        if len(kept) >= N:
            break

    out = ROOT / "data" / "coedit_train.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["input", "target"])
        w.writeheader()
        w.writerows(kept)

    print(f"\nkept {len(kept)} minimal-edit single-sentence pairs -> {out}")
    print("filter stats:", stats)
    print("\n--- 5 sample pairs ---")
    for p in kept[:5]:
        print(f"  IN : {p['input']}")
        print(f"  OUT: {p['target']}")


if __name__ == "__main__":
    main()
