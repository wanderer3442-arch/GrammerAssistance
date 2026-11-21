"""Diagnose remaining failures: are they a data-coverage gap or model limits?

Clusters the current model's errors on the full-300 held-out set into linguistic
categories and reports how many held-out items exist per category, so we know
whether more targeted data would help (coverage gap) or the model is maxed out.
"""
import csv
import re
from pathlib import Path
from collections import defaultdict
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "model"
DATA = ROOT / "data"
MAX_LEN = 128

IRREG = {
    "eat": "ate/eaten", "teach": "taught", "begin": "began", "forget": "forgot/forgotten",
    "go": "went/gone", "see": "saw/seen", "write": "wrote/written", "take": "took/taken",
    "come": "came/come", "know": "knew/known", "buy": "bought", "catch": "caught",
    "bring": "brought", "think": "thought", "leave": "left", "keep": "kept",
    "feel": "felt", "meet": "met", "pay": "paid", "say": "said", "tell": "told",
    "hold": "held", "lose": "lost", "send": "sent", "build": "built", "spend": "spent",
}
COLLOC = {
    "waiting for": "for", "congratulated on": "on", "succeeded in": "in",
    "different from": "from", "afraid of": "of", "depend on": "on",
    "listen to": "to", "wait for": "for", "belong to": "to", "consist of": "of",
}

def bucket(src, tgt):
    sl, tl = src.lower(), tgt.lower()
    if re.match(r"^(what|why|where|when|who|how|which)\b", sl):
        return "wh-question"
    for v in IRREG:
        if re.search(rf"\b{v}\b", sl) and IRREG[v].split("/")[0] not in sl:
            return "irregular-verb"
    for c in COLLOC:
        if c in sl and tl.count(c.split()[1]) > sl.count(c.split()[1]):
            return "preposition-collocation"
    if re.search(r"\b(a)\b", sl) and re.search(r"\ban\b", tl):
        return "article-a_to_an"
    if re.search(r"\ban\b", sl) and re.search(r"\ba\b", tl):
        return "article-an_to_a"
    if re.search(r"\b(is|are|was|were|am)\b", sl):
        return "be-verb"
    if re.search(r"\b(a|an|the)\b", sl) or re.search(r"\b(a|an|the)\b", tl):
        return "article-other"
    return "agreement-other"

rows = list(csv.DictReader(open(DATA / "grammar_supplementary_300_reviewed_v2.csv",
                                 encoding="utf-8")))
pairs = [(r["input"].strip(), r["target"].strip()) for r in rows]

tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
model.eval()

total_n = defaultdict(int)
total_fail = defaultdict(int)
examples = defaultdict(list)
overall_correct = 0

with torch.no_grad():
    for src, tgt in pairs:
        enc = tok(src, return_tensors="pt", truncation=True, max_length=MAX_LEN)
        out = model.generate(**enc, do_sample=False, max_length=MAX_LEN)
        hyp = tok.decode(out[0], skip_special_tokens=True).strip()
        ok = hyp == tgt
        overall_correct += ok
        b = bucket(src, tgt)
        total_n[b] += 1
        total_fail[b] += (not ok)
        if not ok and len(examples[b]) < 3:
            examples[b].append((src, tgt, hyp))

print(f"overall: {100*overall_correct/len(pairs):.1f}%  "
      f"({overall_correct}/{len(pairs)})\n")
print(f"{'BUCKET':<26}{'N':>5}{'FAIL':>6}{'EXACT%':>8}{'MISS%':>8}")
print("-" * 54)
for b in sorted(total_n, key=lambda x: -total_fail[x]):
    n, f = total_n[b], total_fail[b]
    print(f"{b:<26}{n:>5}{f:>6}{100*(n-f)/n:>8.1f}{100*f/n:>8.1f}")

print("\n--- examples per failing bucket (in / exp / got) ---")
for b in sorted(total_n, key=lambda x: -total_fail[x]):
    if total_fail[b] == 0:
        continue
    print(f"\n[{b}]  ({total_fail[b]} failures of {total_n[b]})")
    for src, tgt, hyp in examples[b]:
        print(f"  {src}")
        print(f"    exp {tgt}")
        print(f"    got {hyp}")
