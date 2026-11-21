"""Compare checkpoints on the three held-out sets (greedy decoding).

Usage
-----
    python app/compare_models.py --models model model_round5
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


@torch.no_grad()
def evaluate(model, tok, pairs, max_len):
    exact = noop = wrong = 0
    for src, tgt in pairs:
        enc = tok(src, return_tensors="pt", truncation=True, max_length=max_len)
        out = model.generate(**enc, do_sample=False, max_length=max_len)
        hyp = tok.decode(out[0], skip_special_tokens=True).strip()
        ok = hyp == tgt
        exact += ok
        if not ok and hyp == src:
            noop += 1
        elif not ok:
            wrong += 1
    n = len(pairs)
    return {"n": n, "exact": exact,
            "exact_pct": round(100 * exact / n, 2),
            "noop": noop, "wrong": wrong}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["model", "model_round5"])
    parser.add_argument("--max-len", type=int, default=192)
    args = parser.parse_args()

    sets = {
        "test60": load_pairs(EVAL_DIR / "test_split.csv"),
        "synth700": load_pairs(ROOT / "data" / "synthetic_test.csv"),
        "held300": load_pairs(EVAL_DIR / "heldout_300_report.csv", "expected"),
    }

    results = {}
    for name in args.models:
        path = ROOT / name
        if not path.exists():
            print(f"skip {name}: not found")
            continue
        print(f"\n{name}")
        tok = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSeq2SeqLM.from_pretrained(path)
        model.eval()
        results[name] = {}
        for set_name, pairs in sets.items():
            t0 = time.time()
            r = evaluate(model, tok, pairs, args.max_len)
            results[name][set_name] = r
            print(f"  {set_name:9s} exact {r['exact']:3d}/{r['n']:3d} = "
                  f"{r['exact_pct']:5.2f}%   no-op {r['noop']:3d}   "
                  f"wrong {r['wrong']:3d}   ({time.time()-t0:.0f}s)")
        del model

    print("\nsummary (exact-match %)")
    header = f"{'model':18s}" + "".join(f"{k:>11s}" for k in sets)
    print(header)
    for name, by_set in results.items():
        print(f"{name:18s}" + "".join(
            f"{by_set[k]['exact_pct']:>10.2f}%" for k in sets))

    out = EVAL_DIR / "model_comparison.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
