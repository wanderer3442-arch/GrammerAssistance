"""Round 5: continued training with programmatically generated error pairs.

This is the modelling stage of the project.  It continues training the
project's own T5 checkpoint (`model/`) on a mixture of

  * synthetic pairs produced by app/gen_synthetic.py  (morphology + agreement
    + irregular verbs + a/an + plurals + capitalization + no-op examples),
  * the real GEC subset from Grammarly CoEdit (generalization / fluency),
  * the earlier hand-written targeted pairs (replay, avoids forgetting).

The checkpoint is written to a NEW directory (`model_round5/`) so the current
production model is never overwritten until the new one has been evaluated.

Evaluation is exact-match on three untouched sets:
  * eval/test_split.csv               (60 pairs, the original benchmark)
  * data/synthetic_test.csv           (700 pairs, held out from generation)
  * eval/heldout_300_report.csv       (300 pairs, the project benchmark)

Usage
-----
    python app/train_round5.py --epochs 1 --lr 1e-5
    python app/train_round5.py --smoke          # 120 pairs, quick pipeline test
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import random
import shutil
import time
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "eval"


def load_pairs(path: Path) -> list[tuple[str, str]]:
    with open(path, encoding="utf-8", newline="") as fh:
        return [(r["input"].strip(), r["target"].strip())
                for r in csv.DictReader(fh)
                if r.get("input") and r.get("target")]


def load_report_pairs(path: Path) -> list[tuple[str, str]]:
    """The 300-pair report uses `expected` instead of `target`."""
    with open(path, encoding="utf-8", newline="") as fh:
        return [(r["input"].strip(), r["expected"].strip())
                for r in csv.DictReader(fh)
                if r.get("input") and r.get("expected")]


class PairDataset(Dataset):
    def __init__(self, pairs, tok, max_len):
        self.pairs = pairs
        self.tok = tok
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        src, tgt = self.pairs[i]
        model_in = self.tok(src, truncation=True, max_length=self.max_len)
        labels = self.tok(text_target=tgt, truncation=True, max_length=self.max_len)
        model_in["labels"] = labels["input_ids"]
        return model_in


def build_train_set(args) -> list[tuple[str, str]]:
    rng = random.Random(args.seed)
    pairs: list[tuple[str, str]] = []

    synthetic_path = ROOT / "data" / "synthetic_round5.csv"
    synthetic = load_pairs(synthetic_path)
    rng.shuffle(synthetic)
    pairs.extend(synthetic[: args.synthetic_limit])

    coedit_path = ROOT / "data" / "coedit_train.csv"
    coedit = load_pairs(coedit_path)
    rng.shuffle(coedit)
    pairs.extend(coedit[: args.coedit_limit])

    replay_files = [
        ROOT / "data" / "train_pool.csv",
        ROOT / "data" / "targeted_pairs_round2.csv",
        ROOT / "data" / "targeted_pairs_round3.csv",
    ]
    for path in replay_files:
        if path.exists():
            pairs.extend(load_pairs(path))

    pairs = list(dict.fromkeys(pairs))
    rng.shuffle(pairs)
    return pairs


@torch.no_grad()
def evaluate(model, tok, pairs, max_len, label, collect=False):
    model.eval()
    exact = 0
    noop = 0
    wrong = 0
    rows = []
    t0 = time.time()
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
        if collect:
            rows.append({"input": src, "expected": tgt, "predicted": hyp,
                         "exact": ok})
    n = len(pairs)
    print(f"  {label:22s} exact {exact}/{n} = {100*exact/n:5.2f}%   "
          f"no-op misses {noop:3d}   wrong edits {wrong:3d}   "
          f"({time.time()-t0:.0f}s)")
    return {"n": n, "exact": exact, "exact_pct": round(100 * exact / n, 2),
            "noop": noop, "wrong": wrong, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="model")
    parser.add_argument("--out", default="model_round5")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--max-len", type=int, default=192)
    parser.add_argument("--synthetic-limit", type=int, default=6500)
    parser.add_argument("--coedit-limit", type=int, default=3500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true",
                        help="tiny run to validate the pipeline")
    args = parser.parse_args()

    if args.smoke:
        args.epochs = 1.0
        args.synthetic_limit = 80
        args.coedit_limit = 20
        args.out = "model_smoke"

    base_path = Path(args.base)
    # A local directory (the project's own checkpoint) or a Hugging Face id
    # such as "google-t5/t5-small" when training from the pretrained base.
    base_dir = base_path if base_path.exists() else args.base
    out_dir = ROOT / args.out

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}   torch: {torch.__version__}")

    train_pairs = build_train_set(args)
    print(f"train pairs: {len(train_pairs)}")

    test60 = load_pairs(EVAL_DIR / "test_split.csv")
    synth700 = load_pairs(ROOT / "data" / "synthetic_test.csv")
    held300 = load_report_pairs(EVAL_DIR / "heldout_300_report.csv")

    if args.smoke:
        test60 = test60[:10]
        synth700 = synth700[:20]
        held300 = held300[:10]

    tok = AutoTokenizer.from_pretrained(base_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(base_dir)

    print("\nBEFORE training:")
    before = {
        "test60": evaluate(model, tok, test60, args.max_len, "test_split (60)"),
        "synth700": evaluate(model, tok, synth700, args.max_len, "synthetic (700)"),
        "held300": evaluate(model, tok, held300, args.max_len, "held-out (300)"),
    }

    ds = PairDataset(train_pairs, tok, args.max_len)
    collator = DataCollatorForSeq2Seq(tok, model=model)

    tmp_dir = ROOT / "app" / "_trainer_tmp"
    steps_per_epoch = max(1, len(train_pairs) // (args.batch * args.grad_accum))
    warmup = max(1, int(0.06 * steps_per_epoch * args.epochs))
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(tmp_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="linear",
        warmup_steps=warmup,
        logging_steps=25,
        save_strategy="no",
        report_to=[],
        seed=args.seed,
        remove_unused_columns=False,
        predict_with_generate=False,
        dataloader_num_workers=0,
        fp16=torch.cuda.is_available(),
    )

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=collator,
    )
    if "processing_class" in inspect.signature(Seq2SeqTrainer).parameters:
        trainer_kwargs["processing_class"] = tok
    else:  # transformers < 4.46
        trainer_kwargs["tokenizer"] = tok
    trainer = Seq2SeqTrainer(**trainer_kwargs)

    print(f"\ntraining {args.epochs} epoch(s) on {len(train_pairs)} pairs (CPU) ...")
    t0 = time.time()
    result = trainer.train()
    minutes = (time.time() - t0) / 60
    print(f"done in {minutes:.1f} min   final loss: {result.training_loss:.4f}")

    model.eval()
    print("\nAFTER training:")
    after = {
        "test60": evaluate(model, tok, test60, args.max_len, "test_split (60)"),
        "synth700": evaluate(model, tok, synth700, args.max_len, "synthetic (700)"),
        "held300": evaluate(model, tok, held300, args.max_len, "held-out (300)"),
    }

    # save to a fresh directory
    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))
    cfg_path = out_dir / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    # Preserve the base checkpoint's tying convention: the project's custom
    # model/ is untied (false), while google-t5/t5-small is tied (true).
    if isinstance(base_dir, Path) and (base_dir / "config.json").exists():
        base_cfg = json.loads((base_dir / "config.json").read_text(encoding="utf-8"))
        cfg["tie_word_embeddings"] = base_cfg.get(
            "tie_word_embeddings", cfg.get("tie_word_embeddings", False))
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    metrics = {
        "base": args.base,
        "out": args.out,
        "train_pairs": len(train_pairs),
        "epochs": args.epochs,
        "lr": args.lr,
        "minutes": round(minutes, 1),
        "final_train_loss": round(result.training_loss, 4),
        "before": {k: {kk: vv for kk, vv in v.items() if kk != "rows"}
                   for k, v in before.items()},
        "after": {k: {kk: vv for kk, vv in v.items() if kk != "rows"}
                  for k, v in after.items()},
    }
    metrics_path = EVAL_DIR / "round5_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nwrote {metrics_path}")

    if not args.smoke:
        # keep the per-sentence predictions for error analysis
        for key, data in after.items():
            path = EVAL_DIR / f"round5_{key}.csv"
            with open(path, "w", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(
                    fh, fieldnames=["input", "expected", "predicted", "exact"])
                writer.writeheader()
                writer.writerows(data["rows"])
            print(f"wrote {path}")

    shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"saved model -> {out_dir}")


if __name__ == "__main__":
    main()
