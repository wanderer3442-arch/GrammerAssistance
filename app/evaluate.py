"""Generalization eval for the fine-tuned grammar model.

Evaluates on the 300-pair held-out set (never trained on) and on the
15-example set (trained on, for memorization-vs-generalization comparison).

Metrics:
  - exact match rate (strict)
  - chrF  (character n-gram F-score; robust for near-miss corrections)
  - overcorrection rate: already-correct inputs that the model broke
  - no-op rate: unchanged outputs (correct or missed)

Writes a detailed per-row report + a summary JSON to ../eval/.
"""
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

import csv as _csv

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "model"
EVAL_DIR = ROOT / "eval"
EVAL_DIR.mkdir(exist_ok=True)

MAX_LENGTH = 128


def chrf(ref: str, hyp: str, n: int = 6, beta: float = 2.0) -> float:
    """Character n-gram F-score (chrF-ish). Higher = closer."""
    ref, hyp = ref.lower().strip(), hyp.lower().strip()
    if ref == hyp:
        return 100.0

    def ngrams(s, k):
        return [s[i:i + k] for i in range(len(s) - k + 1)] if len(s) >= k else [s]

    ref_counts, hyp_counts = defaultdict(int), defaultdict(int)
    for k in range(1, n + 1):
        for g in ngrams(ref, k):
            ref_counts[g] += 1
        for g in ngrams(hyp, k):
            hyp_counts[g] += 1

    overlap = sum(min(c, hyp_counts[g]) for g, c in ref_counts.items())
    ref_total = sum(ref_counts.values())
    hyp_total = sum(hyp_counts.values())
    if ref_total == 0 or hyp_total == 0:
        return 0.0
    prec = overlap / hyp_total
    rec = overlap / ref_total
    if prec + rec == 0:
        return 0.0
    beta2 = beta * beta
    f = (1 + beta2) * prec * rec / (beta2 * rec + prec)
    return 100.0 * f


def load_csv_pairs(path):
    pairs = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in _csv.DictReader(f):
            if row.get("input") and row.get("target"):
                pairs.append((row["input"].strip(), row["target"].strip()))
    return pairs


def load_json_pairs(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [(d["input"].strip(), d["target"].strip()) for d in data]


def evaluate(model, tok, pairs, label):
    rows = []
    exact = 0
    chrf_sum = 0.0
    noop = 0          # model left input unchanged
    overcor = 0       # input == target (already correct) but model changed it
    missed = 0        # input != target (needs fix) but model left it unchanged
    already_correct_total = 0

    with torch.no_grad():
        for src, tgt in pairs:
            already_correct = (src == tgt)
            if already_correct:
                already_correct_total += 1
            enc = tok(src, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
            out = model.generate(**enc, do_sample=False, max_length=MAX_LENGTH)
            hyp = tok.decode(out[0], skip_special_tokens=True).strip()

            is_exact = (hyp == tgt)
            c = chrf(tgt, hyp)
            exact += is_exact
            chrf_sum += c
            if hyp == src:
                noop += 1
                if not already_correct:
                    missed += 1
            elif already_correct:
                overcor += 1

            rows.append({
                "input": src, "expected": tgt, "predicted": hyp,
                "exact": is_exact, "chrf": round(c, 2),
                "already_correct": already_correct,
            })

    n = len(pairs)
    summary = {
        "set": label,
        "n": n,
        "exact_match_pct": round(100.0 * exact / n, 2),
        "mean_chrf": round(chrf_sum / n, 2),
        "noop_pct": round(100.0 * noop / n, 2),
        "missed_fix_pct": round(100.0 * missed / n, 2) if n else 0.0,
        "already_correct_n": already_correct_total,
        "overcorrection_pct": round(100.0 * overcor / already_correct_total, 2)
                              if already_correct_total else None,
    }
    return summary, rows


def main():
    print("loading model ...")
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
    model.eval()
    print("model loaded\n")

    heldout = load_csv_pairs(ROOT / "data" / "grammar_supplementary_300_reviewed_v2.csv")
    trained = load_json_pairs(ROOT / "data" / "supplementary_pairs_v1.json")

    # de-dup trained set (it has repeated pairs) for a clean memorization read
    trained = list(dict.fromkeys(trained))

    s_held, r_held = evaluate(model, tok, heldout, "heldout_300_reviewed_v2")
    s_trained, r_trained = evaluate(model, tok, trained, "trained_15_unique")

    print("=" * 60)
    print(f"{'SET':<28}{'N':>4}{'EXACT%':>9}{'chrF':>8}{'NOOP%':>8}")
    print("-" * 60)
    for s in (s_held, s_trained):
        print(f"{s['set']:<28}{s['n']:>4}{s['exact_match_pct']:>9}"
              f"{s['mean_chrf']:>8}{s['noop_pct']:>8}")
    print("=" * 60)

    print("\nheld-out detail:")
    print(f"  needs-a-fix missed (left unchanged): {s_held['missed_fix_pct']}%")
    print(f"  overcorrection of correct inputs   : {s_held['overcorrection_pct']}%")

    # error breakdown by keyword pattern on held-out
    patterns = {
        "subject-verb agreement": [" work ", " works ", " live ", " lives ",
                                   " speak ", " speaks ", " go ", " goes "],
        "be-verb / is-are": [" is ", " are ", " was ", " were ", " am "],
        "articles (a/the)": [" a ", " an ", " the "],
        "negation (not/n't)": [" not ", "n't"],
        "modal (can/to)": [" can to ", " to drive", " must to "],
    }
    print("\nheld-out exact-match by rough error category:")
    for cat, toks in patterns.items():
        subset = [r for r in r_held if any(t in r["input"].lower() for t in toks)]
        if subset:
            acc = 100.0 * sum(r["exact"] for r in subset) / len(subset)
            print(f"  {cat:<26} n={len(subset):<4} exact={acc:.1f}%")

    # write reports
    for name, rows in (("heldout_300", r_held), ("trained_15", r_trained)):
        p = EVAL_DIR / f"{name}_report.csv"
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {p}")

    (EVAL_DIR / "summary.json").write_text(
        json.dumps({"heldout": s_held, "trained": s_trained}, indent=2))
    print(f"wrote {EVAL_DIR / 'summary.json'}")

    # show worst held-out failures for inspection
    fails = [r for r in r_held if not r["exact"]]
    fails.sort(key=lambda r: r["chrf"])
    print(f"\n--- {len(fails)} held-out failures; 12 worst by chrF ---")
    for r in fails[:12]:
        print(f"  in : {r['input']}")
        print(f"  exp: {r['expected']}")
        print(f"  got: {r['predicted']}   (chrF {r['chrf']})")
        print()


if __name__ == "__main__":
    main()
