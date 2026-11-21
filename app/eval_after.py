"""Re-evaluate the (now targeted-trained) model on the 60-pair test split,
broken down by error category, so we can confirm the wh-question fix didn't
come at the cost of the other categories (be-verb, articles, agreement).
"""
import csv
import re
from pathlib import Path
from collections import defaultdict
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "model"
EVAL = ROOT / "eval"
MAX_LEN = 128

tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
model.eval()

rows = list(csv.DictReader(open(EVAL / "test_split.csv", encoding="utf-8")))

def category(s):
    if re.match(r"^(what|why|where|when|who|how|which)\b", s.strip(), re.I):
        return "wh-question"
    if re.search(r"\b(is|are|was|were|am)\b", s, re.I):
        return "be-verb"
    if re.search(r"\b(a|an|the)\b", s, re.I):
        return "articles"
    return "other"

exact = 0
per_cat = defaultdict(lambda: [0, 0])  # cat -> [correct, total]
fails = []

with torch.no_grad():
    for r in rows:
        src, tgt = r["input"].strip(), r["target"].strip()
        enc = tok(src, return_tensors="pt", truncation=True, max_length=MAX_LEN)
        out = model.generate(**enc, do_sample=False, max_length=MAX_LEN)
        hyp = tok.decode(out[0], skip_special_tokens=True).strip()
        ok = (hyp == tgt)
        exact += ok
        c = category(src)
        per_cat[c][1] += 1
        per_cat[c][0] += ok
        if not ok:
            fails.append((src, tgt, hyp))

print(f"TEST split overall exact-match: {100*exact/len(rows):.1f}%  ({exact}/{len(rows)})\n")
print(f"{'CATEGORY':<14}{'CORRECT':>9}{'TOTAL':>7}{'EXACT%':>9}")
print("-" * 40)
for c in ["wh-question", "be-verb", "articles", "other"]:
    if c in per_cat:
        corr, tot = per_cat[c]
        print(f"{c:<14}{corr:>9}{tot:>7}{100*corr/tot:>9.1f}")

print(f"\n--- {len(fails)} failures ---")
for src, tgt, hyp in fails:
    print(f"  in : {src}")
    print(f"  exp: {tgt}")
    print(f"  got: {hyp}")
