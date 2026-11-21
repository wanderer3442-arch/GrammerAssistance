"""Apples-to-apples before/after on the FULL 300 held-out set.

Compares the backed-up pre-training model against the newly trained one on the
identical 300 pairs, broken down by category, to detect regressions (especially
be-verb, which looked like it dropped) and confirm the wh-question gain.
"""
import csv
import re
from pathlib import Path
from collections import defaultdict
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "model"
BACKUP = ROOT / "_archive" / "pre_targeted_backup"
DATA = ROOT / "data"
MAX_LEN = 128

def category(s):
    if re.match(r"^(what|why|where|when|who|how|which)\b", s.strip(), re.I):
        return "wh-question"
    if re.search(r"\b(is|are|was|were|am)\b", s, re.I):
        return "be-verb"
    if re.search(r"\b(a|an|the)\b", s, re.I):
        return "articles"
    return "other"

rows = list(csv.DictReader(open(DATA / "grammar_supplementary_300_reviewed_v2.csv",
                                 encoding="utf-8")))
pairs = [(r["input"].strip(), r["target"].strip()) for r in rows]

def evaluate(model_dir, label):
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    model.eval()
    per = defaultdict(lambda: [0, 0])
    total = 0
    changed_back = []   # correct before, wrong after (regressions)
    with torch.no_grad():
        for src, tgt in pairs:
            enc = tok(src, return_tensors="pt", truncation=True, max_length=MAX_LEN)
            out = model.generate(**enc, do_sample=False, max_length=MAX_LEN)
            hyp = tok.decode(out[0], skip_special_tokens=True).strip()
            ok = (hyp == tgt)
            total += ok
            c = category(src)
            per[c][1] += 1
            per[c][0] += ok
    del model
    return total, per

print("evaluating PRE-training backup model ...")
pre_total, pre_per = evaluate(BACKUP, "before")
print("evaluating POST-training model ...")
post_total, post_per = evaluate(MODEL_DIR, "after")

n = len(pairs)
print(f"\nFULL 300 held-out set:")
print(f"  BEFORE: {100*pre_total/n:.1f}%   AFTER: {100*post_total/n:.1f}%   "
      f"({post_total-pre_total:+d} correct)")
print(f"\n{'CATEGORY':<14}{'BEFORE':>8}{'AFTER':>8}{'DELTA':>8}   (n)")
print("-" * 46)
for c in ["wh-question", "be-verb", "articles", "other"]:
    if c in pre_per:
        b, tot = pre_per[c]
        a, _ = post_per[c]
        print(f"{c:<14}{100*b/tot:>7.1f}%{100*a/tot:>7.1f}%{a-b:>+7}   ({tot})")
