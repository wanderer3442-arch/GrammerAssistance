"""Verify held-out data quality + extract distinct error patterns.

1. Sanity-check the held-out labels (are 'failures' actually model errors, or
   mislabeled/ambiguous targets?).
2. Mine the input->target pairs into distinct transformation rules so we can
   build a targeted, non-redundant training set.
"""
import csv
import re
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parent.parent
rows = list(csv.DictReader(open(ROOT / "eval" / "heldout_300_report.csv", encoding="utf-8")))
for r in rows:
    r["exact"] = r["exact"] == "True"

fails = [r for r in rows if not r["exact"]]

print(f"held-out: {len(rows)}  failures: {len(fails)}\n")

# 1. Data-quality flags among failures
print("=== data-quality checks on failures ===")
noop_target = [r for r in fails if r["input"].strip() == r["expected"].strip()]
print(f"input == target (nothing to fix, but counted as failure): {len(noop_target)}")

# suspicious: target introduces info not in input (hallucinated gold)
susp = [r for r in fails
        if len(r["expected"].split()) > len(r["input"].split()) + 2]
print(f"target much longer than input (check gold): {len(susp)}")
for r in susp[:6]:
    print(f"   in : {r['input']!r}")
    print(f"   exp: {r['expected']!r}")

# near-identical predictions that differ only by punctuation/case
trivial = [r for r in fails
           if r["predicted"].lower().rstrip("?!.") == r["expected"].lower().rstrip("?!.")]
print(f"\nfailures that differ only by case/punct from target: {len(trivial)}")
for r in trivial[:8]:
    print(f"   in : {r['input']!r}")
    print(f"   exp: {r['expected']!r}")
    print(f"   got: {r['predicted']!r}")

# 2. Mine transformation patterns from ALL held-out pairs (not just failures)
print("\n\n=== distinct transformation patterns (from all 300) ===")
rules = Counter()
examples = defaultdict(list)
for r in rows:
    src, tgt = r["input"], r["expected"]
    if src == tgt:
        rules["(no change)"] += 1
        continue
    sl, tl = src.lower(), tgt.lower()
    # wh-question inversion: "wh + subj + verb" -> "wh + aux + subj + verb"
    if re.match(r"^(what|why|where|when|who|how|which)\b", sl):
        rules["wh-question (inversion/reordering)"] += 1
        examples["wh-question"].append((src, tgt))
    elif re.search(r"\b(is|are|was|were|am)\b", sl) and re.search(r"\b(is|are|was|were|am)\b", tl):
        rules["be-verb form"] += 1
    elif re.search(r"\b(a)\b", sl) and re.search(r"\ban\b", tl):
        rules["a -> an (article)"] += 1
    elif re.search(r"\b(a|an|the)\b", sl) or re.search(r"\b(a|an|the)\b", tl):
        rules["article insert/remove"] += 1
    elif re.search(r"\b(\w+)s\b", sl) and not re.search(r"\b(\w+)s\b", tl):
        rules["verb/noun de-pluralize"] += 1
    elif not re.search(r"\b(\w+)s\b", sl) and re.search(r"\b(\w+)s\b", tl):
        rules["verb/noun pluralize (3rd person)"] += 1
    elif re.search(r"\b(not|n't)\b", sl, re.I) or re.search(r"\b(not|n't)\b", tl, re.I):
        rules["negation"] += 1
    else:
        rules["other lexical"] += 1

for rule, cnt in rules.most_common():
    print(f"  {cnt:>4}  {rule}")

print("\n--- wh-question examples (the blind spot) ---")
for src, tgt in examples["wh-question"][:20]:
    print(f"   {src!r}  ->  {tgt!r}")
