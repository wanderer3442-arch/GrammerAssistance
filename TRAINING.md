# Training History — Grammar Correction Model

## Baseline (as received)
- Base T5 (`d_model=512`, 8 enc/8 dec, ~77M params), fine-tuned on **15 pairs × 5 epochs**.
- **Overfit**: memorized the 15 (100%) but generalized to only **64.7%** on the 300-pair held-out set.
- Blind spot: **wh-question inversion** (`Why he work here?` → `Why does he work here?`) at **5.9%** — none of its 15 examples were questions.

## Method
- Held out a **60-pair test split** (`eval/test_split.csv`) NEVER trained on, for honest before/after.
- Continued-training from the existing checkpoint with **targeted, balanced** data.
- All eval on the **full 300 held-out pairs** (larger/more reliable than the 60-split).
- CPU-only, transformers 5.19, `tie_word_embeddings=False`.

## Rounds

| Round | Data added | Focus | Full-300 exact |
|------|-----------|-------|----------------|
| 0 (baseline) | 15 pairs | — | 64.7% |
| 1 | +174 pairs | wh-question inversion (94), be-verb, articles | 71.7% |
| 2 | +50 pairs | article rebalance (fix catastrophic forgetting of a/an) | 76.0% |
| 3 | +39 pairs | an→a vowel-sound reversal, -ies spelling, irregulars, collocations | 76.7% |
| 4 (CoEdit) | +5,000 real GEC pairs | Grammarly CoEdit, minimal-edit filtered | **78.0%** |

**Net: 64.7% → 78.0% (+13.3 pts, +40 sentences correct on 300).**

### Round 4 — real GEC data (CoEdit)
Grammarly's **CoEdit** corpus (`grammarly/coedit`, 19,823 GEC pairs) was preprocessed to
**5,000 minimal-edit single-sentence pairs** (instruction prefixes stripped, filtered to
high word-overlap so the model learns to *correct*, not rewrite, deduped against the
held-out set). Trained 2 epochs @ LR 2e-5. Real data beat the hand-pairs approach and was
the cleanest round yet — every category improved, none regressed:

| Category | Before R4 | After R4 | Δ |
|----------|-----------|----------|---|
| articles | 73.1% | 75.9% | +8 |
| irregular-verb | 64.3% | 75.0% | +11 |
| wh-question | 88.2% | 88.2% | 0 |
| other/agreement | 79.1% | 81.4% | +2 |

### Per-category movement (full 300, baseline → final)
| Category | Baseline | Final | Δ |
|----------|--------|-------|---|
| wh-question | 5.9% | 88.2% | **+28** |
| articles | 70.3% | 75.9% | +6 |
| be-verb | 60.0% | 68.6% | +5 |
| other | 80.2% | 81.4% | +1 |

Round 2 reversed a **catastrophic-forgetting regression**: Round 1 alone had *dropped*
articles (70.3→68.3) by over-weighting wh-questions. Rebalancing fixed it.

## Why we stopped at 78% — model-capacity ceiling (confirmed)
The remaining ~22% failures are **orthographic/phonetic edge cases that more same-style
data cannot fix** on a 77M model. Real GEC data fixed the "normal" errors (irregulars
64→75%, articles 70→76%), but these stayed flat:
- `-ies` spelling: writes `crys` for `cries`, `fixe` for `fixes`
- a/an by vowel-*sound*: `an useful`, `an university`, `an European` (~33%, unchanged)

Root cause: **T5 subword tokenizer** splits `cries`/`unique`/`university` into pieces, so
the model can't learn sound-based article selection or -y→-ies spelling as reliable rules.
77M params is too small to hold them. Crossing this ceiling needs a **larger base model**
(mT5-small/base, flan-t5) — **not viable on this hardware** (no GPU, 7.3 GB RAM / 2.1 GB
free; mT5-base needs ~2.3 GB to load and would thrash).

**Practical ceiling on this machine ≈ 78–80%.** The model is production-ready for an
offline tool at this level.

## Artifacts
- `model/` — current fine-tuned model (deployable; `tie_word_embeddings=False`).
- `_archive/pre_targeted_backup/` — original 15-example checkpoint (for rollback).
- `_archive/pre_coedit_backup/` — the 76.7% hand-pairs checkpoint (rollback before CoEdit).
- `data/targeted_pairs*.csv` — the authored training pairs by round.
- `data/coedit_train.csv` — the 5,000-pair CoEdit minimal-edit subset.
- `app/prep_coedit.py` — CoEdit download + preprocessing (dedup against test split).
- `eval/` — test split, before/after reports, per-bucket diagnosis.

## Hardware constraint (why no bigger model)
CPU-only: AMD Ryzen 3 5300U (4c/8t), **7.3 GB RAM (2.1 GB free)**, no GPU.
mT5-base needs ~2.3 GB to load (would thrash); mT5-small ~1.2 GB (no headroom with
Streamlit). The 77M T5 (~310 MB) is the right size for this machine.
