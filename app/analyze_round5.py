"""Break down round-5 predictions by error family.

Reads the per-sentence prediction files written by train_round5.py and reports
exact-match plus failure counts for each error family.  For the synthetic test
set the family is taken from `data/synthetic_test.csv`; for the 300-pair
benchmark it is inferred from the input/expected diff.

Usage
-----
    python app/analyze_round5.py
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "eval"


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z']+", text)


def _is_s_form(a: str, b: str) -> bool:
    """True when b is the 3rd-person-singular form of a (or vice versa)."""
    if a == b:
        return False
    if b == a + "s" or b == a + "es":
        return True
    if a.endswith("y") and b == a[:-1] + "ies":
        return True
    if a.endswith("e") and b == a + "s":
        return True
    return False


def _strip_s(word: str) -> str:
    if word.endswith("ies"):
        return word[:-3] + "y"
    if word.endswith("es"):
        return word[:-2]
    if word.endswith("s"):
        return word[:-1]
    return word


def classify(input_text: str, expected: str) -> str:
    """Infer the error family from the gold edit."""
    a, b = _words(input_text), _words(expected)
    if len(a) == len(b):
        diffs = [(x, y) for x, y in zip(a, b) if x != y]
        if not diffs:
            if input_text != expected:
                return "capitalization"
            return "identity"
        if len(diffs) == 1:
            x, y = diffs[0]
            if _is_s_form(x.lower(), y.lower()) or _is_s_form(y.lower(), x.lower()):
                return "agreement"
            if x.lower() in ("a", "an") or y.lower() in ("a", "an"):
                return "article"
            return "other"
        if all(_is_s_form(x.lower(), y.lower()) for x, y in diffs):
            return "agreement"
    return "other"


def summarize(rows: list[dict], type_map: dict[str, str] | None,
              label: str) -> dict:
    by_type: dict[str, Counter] = defaultdict(Counter)
    samples: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        kind = (type_map or {}).get(r["input"]) or classify(
            r["input"], r["expected"])
        ok = r["exact"] == "True"
        by_type[kind]["n"] += 1
        by_type[kind]["exact"] += ok
        if not ok:
            by_type[kind]["fail"] += 1
            if len(samples[kind]) < 4:
                samples[kind].append(r)

    print(f"\n=== {label} ===")
    print(f"{'family':16s} {'n':>5s} {'exact':>6s} {'acc':>7s} {'fail':>5s}")
    out = {}
    for kind, c in sorted(by_type.items(), key=lambda kv: -kv[1]["n"]):
        acc = 100 * c["exact"] / c["n"] if c["n"] else 0
        print(f"{kind:16s} {c['n']:5d} {c['exact']:6d} {acc:6.1f}% {c['fail']:5d}")
        out[kind] = {"n": c["n"], "exact": c["exact"],
                     "accuracy_pct": round(acc, 1), "fail": c["fail"]}
        for r in samples[kind][:2]:
            print(f"    IN : {r['input']}")
            print(f"    EXP: {r['expected']}")
            print(f"    GOT: {r['predicted']}")
    return out


def main() -> None:
    report: dict[str, dict] = {}

    synth_rows = read_csv(EVAL_DIR / "round5_synth700.csv")
    type_map = {r["input"]: r["type"]
                for r in read_csv(ROOT / "data" / "synthetic_test.csv")}
    if synth_rows:
        report["synthetic"] = summarize(synth_rows, type_map, "synthetic (700)")

    held_rows = read_csv(EVAL_DIR / "round5_held300.csv")
    if held_rows:
        report["held300"] = summarize(held_rows, None, "held-out (300)")

    test_rows = read_csv(EVAL_DIR / "round5_test60.csv")
    if test_rows:
        report["test60"] = summarize(test_rows, None, "test_split (60)")

    path = EVAL_DIR / "round5_analysis.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
