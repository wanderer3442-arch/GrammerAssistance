"""Categorize held-out failures to guide targeted training.

Groups the 300 held-out examples into error categories and reports exact-match
per category, isolating which categories are the model's blind spots.
"""
import csv
import re
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
rep = ROOT / "eval" / "heldout_300_report.csv"

rows = list(csv.DictReader(open(rep, encoding="utf-8")))
for r in rows:
    r["exact"] = r["exact"] == "True"

def is_question(s):
    return bool(re.match(r"^\s*(what|why|where|when|who|how|which)\b", s, re.I))

def has_inversion_gap(s):
    # "Why he work" style: wh-word + pronoun/noun + verb  (missing aux inversion)
    return bool(re.match(
        r"^\s*(what|why|where|when|who|how|which)\s+\w+\s+(work|live|go|look|put|"
        r"play|cry|drive|explain|decide|solve|do|is|are|was|were|can|did|have|has)\b",
        s, re.I))

cats = defaultdict(list)
for r in rows:
    s = r["input"]
    if is_question(s):
        cats["wh-question"].append(r)
        if has_inversion_gap(s):
            cats["wh-question (inversion gap)"].append(r)
    elif re.search(r"\b(is|are|was|were|am)\b", s, re.I):
        cats["be-verb"].append(r)
    elif re.search(r"\b(a|an|the)\b", s, re.I):
        cats["articles"].append(r)
    else:
        cats["other-agreement"].append(r)

print(f"total held-out: {len(rows)}   overall exact: "
      f"{100*sum(r['exact'] for r in rows)/len(rows):.1f}%\n")
print(f"{'CATEGORY':<34}{'N':>4}{'EXACT%':>9}{'FAILED':>8}")
print("-" * 56)
for cat in ["wh-question", "wh-question (inversion gap)", "be-verb",
            "articles", "other-agreement"]:
    sub = cats.get(cat, [])
    if not sub:
        continue
    acc = 100 * sum(r["exact"] for r in sub) / len(sub)
    print(f"{cat:<34}{len(sub):>4}{acc:>9.1f}{len(sub)-sum(r['exact'] for r in sub):>8}")

# breakdown of question subtypes
print("\n--- wh-question failures: what's the model outputting? ---")
qf = [r for r in cats['wh-question'] if not r['exact']]
halluc = sum(1 for r in qf if len(r['predicted'].split()) >
             len(r['expected'].split()) + 2)
print(f"question failures: {len(qf)}   of which look like hallucinated answers: {halluc}")
for r in qf[:8]:
    print(f"  {r['input']!r}")
    print(f"    exp {r['expected']!r}")
    print(f"    got {r['predicted']!r}")
