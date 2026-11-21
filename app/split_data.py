"""Split held-out 300 into a permanent TEST set (60) + TRAIN pool (240).

The 60 test pairs are saved to eval/test_split.csv and MUST be excluded from
all training so we get an honest before/after generalization number.
Stratified by error category so the test set mirrors the real distribution
(especially the wh-question blind spot).
"""
import csv
import random
import re
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
EVAL = ROOT / "eval"
EVAL.mkdir(exist_ok=True)

random.seed(42)

rows = list(csv.DictReader(open(DATA / "grammar_supplementary_300_reviewed_v2.csv",
                                 encoding="utf-8")))

def category(s):
    if re.match(r"^(what|why|where|when|who|how|which)\b", s.strip(), re.I):
        return "wh-question"
    if re.search(r"\b(is|are|was|were|am)\b", s, re.I):
        return "be-verb"
    if re.search(r"\b(a|an|the)\b", s, re.I):
        return "articles"
    return "other"

# group by category, then sample ~20% of each into the test set (target 60)
by_cat = defaultdict(list)
for r in rows:
    by_cat[category(r["input"])].append(r)

total = len(rows)
target_test = 60
test, train = [], []
for cat, items in by_cat.items():
    n_test = round(target_test * len(items) / total)
    random.shuffle(items)
    test.extend(items[:n_test])
    train.extend(items[n_test:])

# trim/pad to exactly 60 by moving from largest train category
while len(test) > target_test:
    biggest = max(by_cat, key=lambda c: len([r for r in train if category(r["input"]) == c]))
    cand = [r for r in train if category(r["input"]) == biggest][0]
    train.remove(cand); test.append(cand)
while len(test) < target_test:
    cand = train[-1]; train.pop(); test.append(cand)

random.shuffle(test); random.shuffle(train)

def write(pairs, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["input", "target"])
        w.writeheader()
        w.writerows({"input": r["input"], "target": r["target"]} for r in pairs)

write(test, EVAL / "test_split.csv")
write(train, DATA / "train_pool.csv")

print(f"TEST (never train on): {len(test)}  -> eval/test_split.csv")
print(f"TRAIN pool           : {len(train)}  -> data/train_pool.csv")
print("\ntest-split category distribution:")
dist = defaultdict(int)
for r in test:
    dist[category(r["input"])] += 1
for c, n in sorted(dist.items()):
    print(f"  {c:<14} {n}")
