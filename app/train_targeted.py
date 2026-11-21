"""Targeted continued-pretraining for the grammar model.

Train set  = original 15 pairs + 174 targeted pairs (wh-question heavy)
Test set   = the 60-pair split in eval/test_split.csv (NEVER trained on)

This fixes the measured wh-question blind spot (5.9% exact) while using a
low epoch count to avoid the overfitting that 5 epochs on 15 pairs caused.

Saves the result to model/ (backs up the original first) and re-evaluates on
the untouched test split for an honest before/after.
"""
import json
import shutil
import time
from pathlib import Path

import torch
import csv as _csv
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "model"
EVAL_DIR = ROOT / "eval"
BACKUP_DIR = ROOT / "_archive" / "pre_targeted_backup"

EPOCHS = 2
LR = 2e-5
BATCH = 8
MAX_LEN = 256


def load_pairs(path):
    return [(r["input"].strip(), r["target"].strip())
            for r in _csv.DictReader(open(path, encoding="utf-8"))]


class PairDataset(Dataset):
    def __init__(self, pairs, tok, max_len=MAX_LEN):
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


def build_train_set():
    pairs = []
    # original 15
    orig = ROOT / "data" / "supplementary_pairs_v1.json"
    for d in json.loads(orig.read_text(encoding="utf-8")):
        pairs.append((d["input"].strip(), d["target"].strip()))
    # targeted round 1 (wh-question heavy)
    pairs.extend(load_pairs(ROOT / "data" / "targeted_pairs.csv"))
    # targeted round 2 (article rebalance, fixes forgetting)
    r2 = ROOT / "data" / "targeted_pairs_round2.csv"
    if r2.exists():
        pairs.extend(load_pairs(r2))
    # targeted round 3 (an->a reversal, -ies spelling, irregulars, collocations)
    r3 = ROOT / "data" / "targeted_pairs_round3.csv"
    if r3.exists():
        pairs.extend(load_pairs(r3))
    # real GEC data: CoEdit minimal-edit subset (the generalization lever)
    coedit = ROOT / "data" / "coedit_train.csv"
    if coedit.exists():
        pairs.extend(load_pairs(coedit))
    # de-dup
    return list(dict.fromkeys(pairs))


def main():
    train_pairs = build_train_set()
    test_pairs = load_pairs(EVAL_DIR / "test_split.csv")
    print(f"train pairs: {len(train_pairs)}   test pairs (held out): {len(test_pairs)}")

    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
    model.eval()

    # ---- BEFORE evaluation on test split ----
    before = eval_exact(model, tok, test_pairs)
    print(f"\nTEST exact-match BEFORE training: {before:.1f}%")

    # back up the original model once
    if not BACKUP_DIR.exists():
        BACKUP_DIR.mkdir(parents=True)
        for f in MODEL_DIR.glob("*"):
            if f.is_file() and "tie_false.variant" not in f.name:
                shutil.copy2(f, BACKUP_DIR / f.name)
        print(f"backed up original model -> {BACKUP_DIR}")

    ds = PairDataset(train_pairs, tok)
    collator = DataCollatorForSeq2Seq(tok, model=model)

    out_dir = ROOT / "app" / "_trainer_tmp"
    steps_per_epoch = max(1, len(train_pairs) // BATCH)
    warmup = max(1, int(0.1 * steps_per_epoch * EPOCHS))
    args = Seq2SeqTrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH,
        learning_rate=LR,
        lr_scheduler_type="linear",
        warmup_steps=warmup,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        seed=42,
        remove_unused_columns=False,
        predict_with_generate=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        data_collator=collator,
        processing_class=tok,
    )

    print(f"\ntraining {EPOCHS} epochs on {len(train_pairs)} pairs (CPU) ...")
    t0 = time.time()
    result = trainer.train()
    print(f"done in {(time.time()-t0)/60:.1f} min   final loss: {result.training_loss:.4f}")

    # ---- AFTER evaluation on the SAME untouched test split ----
    model.eval()
    after = eval_exact(model, tok, test_pairs)
    print(f"\nTEST exact-match AFTER  training: {after:.1f}%   "
          f"(was {before:.1f}%, {after-before:+.1f} pts)")

    # save the fine-tuned model into model/ (keep the tie=false config)
    trainer.save_model(str(MODEL_DIR))
    tok.save_pretrained(str(MODEL_DIR))
    # re-assert tie=false config
    cfg_path = MODEL_DIR / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg["tie_word_embeddings"] = False
    cfg_path.write_text(json.dumps(cfg, indent=2))

    # persist metrics
    metrics = {"test_exact_before": round(before, 2),
               "test_exact_after": round(after, 2),
               "train_pairs": len(train_pairs),
               "epochs": EPOCHS, "lr": LR,
               "final_train_loss": round(result.training_loss, 4)}
    (EVAL_DIR / "training_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nwrote {EVAL_DIR / 'training_metrics.json'}")
    print(f"saved fine-tuned model -> {MODEL_DIR}")

    # cleanup tmp trainer dir
    shutil.rmtree(out_dir, ignore_errors=True)


@torch.no_grad()
def eval_exact(model, tok, pairs):
    exact = 0
    for src, tgt in pairs:
        enc = tok(src, return_tensors="pt", truncation=True, max_length=MAX_LEN)
        out = model.generate(**enc, do_sample=False, max_length=MAX_LEN)
        hyp = tok.decode(out[0], skip_special_tokens=True).strip()
        exact += (hyp == tgt)
    return 100.0 * exact / len(pairs)


if __name__ == "__main__":
    main()
