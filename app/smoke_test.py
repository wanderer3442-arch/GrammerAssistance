"""Smoke test: confirm the fine-tuned model loads and corrects sensibly."""
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "model"

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL)
model.eval()

probes = [
    # wh-question inversion (was the blind spot)
    ("Why he work here?", "Why does he work here?"),
    ("What you want?", "What do you want?"),
    ("Where they are going?", "Where are they going?"),
    # articles
    ("He bought a apple.", "He bought an apple."),
    ("This is an useful tool.", "This is a useful tool."),
    # agreement / 3rd person
    ("My sister work at a hospital.", "My sister works at a hospital."),
    ("The baby cry every night.", "The baby cries every night."),
    # be-verb
    ("They was happy yesterday.", "They were happy yesterday."),
    # already-correct -> must NOT change
    ("She speaks English very well.", "She speaks English very well."),
    ("The weather is pleasant today.", "The weather is pleasant today."),
]

print(f"{'INPUT':<38}{'EXPECTED':<38}{'GOT':<38}OK")
print("-" * 150)
right = 0
with torch.no_grad():
    for src, exp in probes:
        enc = tok(src, return_tensors="pt", truncation=True, max_length=128)
        out = model.generate(**enc, do_sample=False, max_length=128)
        hyp = tok.decode(out[0], skip_special_tokens=True).strip()
        ok = hyp == exp
        right += ok
        print(f"{src:<38}{exp:<38}{hyp:<38}{'Y' if ok else 'n'}")
print("-" * 150)
print(f"{right}/{len(probes)} exact")
