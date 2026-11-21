"""Compare decoding strategies for the project's own grammar model.

Strategies
----------
greedy          num_beams=1
beam4           num_beams=4
beam4_norep     num_beams=4 + no_repeat_ngram_size=3
beam4_rerank    num_beams=4, 4 candidates, pick the one maximising
                log P(candidate | input) - alpha * token_edit_distance(input, candidate)
                (favours fluent candidates that stay close to the source)

Evaluation is exact-match plus no-op / wrong-edit counts on
  * eval/test_split.csv          (60)
  * data/synthetic_test.csv      (700)
  * eval/heldout_300_report.csv  (300)

Usage
-----
    python app/eval_decoding.py --model model_round5 --limit 300
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "eval"


def load_pairs(path: Path, target_col: str = "target"):
    with open(path, encoding="utf-8", newline="") as fh:
        return [(r["input"].strip(), r[target_col].strip())
                for r in csv.DictReader(fh)
                if r.get("input") and r.get(target_col)]


def token_edit_distance(a: str, b: str) -> int:
    """Levenshtein distance over whitespace tokens."""
    ta, tb = a.split(), b.split()
    prev = list(range(len(tb) + 1))
    for i, wa in enumerate(ta, 1):
        cur = [i]
        for j, wb in enumerate(tb, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (wa != wb)))
        prev = cur
    return prev[-1]


@torch.no_grad()
def generate(model, tok, src, max_len, strategy, alpha=0.05):
    enc = tok(src, return_tensors="pt", truncation=True, max_length=max_len)
    if strategy == "greedy":
        out = model.generate(**enc, do_sample=False, num_beams=1,
                             max_length=max_len)
        return tok.decode(out[0], skip_special_tokens=True).strip()

    if strategy in ("beam4", "beam4_norep"):
        kwargs = {"do_sample": False, "num_beams": 4, "max_length": max_len}
        if strategy == "beam4_norep":
            kwargs["no_repeat_ngram_size"] = 3
        out = model.generate(**enc, **kwargs)
        return tok.decode(out[0], skip_special_tokens=True).strip()

    # beam4_rerank
    out = model.generate(
        **enc,
        do_sample=False,
        num_beams=4,
        num_return_sequences=4,
        max_length=max_len,
        no_repeat_ngram_size=3,
        output_scores=True,
        return_dict_in_generate=True,
    )
    cands = [tok.decode(seq, skip_special_tokens=True).strip()
             for seq in out.sequences]
    scores = out.sequences_scores.tolist()  # length-normalised log-prob
    best, best_value = cands[0], float("-inf")
    for cand, score in zip(cands, scores):
        value = score - alpha * token_edit_distance(src, cand)
        if value > best_value:
            best, best_value = cand, value
    return best


@torch.no_grad()
def run_strategy(model, tok, pairs, max_len, strategy, alpha=0.05):
    exact = noop = wrong = 0
    rows = []
    t0 = time.time()
    for src, tgt in pairs:
        hyp = generate(model, tok, src, max_len, strategy, alpha)
        ok = hyp == tgt
        exact += ok
        if not ok and hyp == src:
            noop += 1
        elif not ok:
            wrong += 1
        rows.append({"input": src, "expected": tgt, "predicted": hyp,
                     "exact": ok})
    n = len(pairs)
    print(f"    {strategy:14s} alpha={alpha:<5} exact {exact:4d}/{n} = "
          f"{100*exact/n:5.2f}%   no-op {noop:3d}   wrong {wrong:3d}   "
          f"({time.time()-t0:.0f}s)")
    return {"strategy": strategy, "alpha": alpha, "n": n, "exact": exact,
            "exact_pct": round(100 * exact / n, 2), "noop": noop,
            "wrong": wrong, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="model_round5")
    parser.add_argument("--max-len", type=int, default=192)
    parser.add_argument("--limit", type=int, default=0,
                        help="cap each evaluation set (0 = all)")
    args = parser.parse_args()

    model_dir = ROOT / args.model
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    model.eval()

    sets = {
        "test60": load_pairs(EVAL_DIR / "test_split.csv"),
        "synth700": load_pairs(ROOT / "data" / "synthetic_test.csv"),
        "held300": load_pairs(EVAL_DIR / "heldout_300_report.csv", "expected"),
    }
    if args.limit:
        sets = {k: v[: args.limit] for k, v in sets.items()}

    strategies = [
        ("greedy", 0.0),
        ("beam4", 0.0),
        ("beam4_norep", 0.0),
        ("beam4_rerank", 0.02),
        ("beam4_rerank", 0.05),
        ("beam4_rerank", 0.10),
    ]

    results = {}
    for name, pairs in sets.items():
        print(f"\n{name}  (n={len(pairs)})")
        results[name] = []
        for strategy, alpha in strategies:
            label = strategy if strategy != "beam4_rerank" else f"{strategy}"
            results[name].append(
                run_strategy(model, tok, pairs, args.max_len, strategy, alpha))

    out_path = EVAL_DIR / f"decoding_{args.model}.json"
    out_path.write_text(json.dumps(
        {name: [{k: v for k, v in r.items() if k != "rows"} for r in rows]
         for name, rows in results.items()},
        indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}")

    # dump the best reranked predictions for error analysis
    best = {}
    for name, rows in results.items():
        reranked = [r for r in rows if r["strategy"] == "beam4_rerank"]
        best[name] = max(reranked, key=lambda r: r["exact_pct"])
    for name, r in best.items():
        path = EVAL_DIR / f"decoding_{args.model}_{name}.csv"
        with open(path, "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(
                fh, fieldnames=["input", "expected", "predicted", "exact"])
            writer.writeheader()
            writer.writerows(r["rows"])
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
