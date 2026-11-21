# PLAN (as executed) — Offline Grammar Tool

Goal achieved: a fully-offline Streamlit app that corrects sentences on a
fine-tuned T5 model. No internet needed after the one-time setup.

**Status: built and verified.** See `README.md` to run it and `TRAINING.md`
for the model's training/eval history.

## What was actually built

```
app/
  engine.py           # loads model+tokenizer (cached), correct() / correct_batch()
  app.py              # Streamlit UI: single + batch tabs, diff highlight, CSV export
  run.bat             # one-click launcher (activates venv, starts streamlit)
  requirements.txt    # pinned deps: torch CPU 2.14.1, transformers 5.19, streamlit 1.65
```

- **engine.py** — `correct(text, max_length, num_beams)` and `correct_batch(...)`.
  Loads the model once via `lru_cache`. Greedy decoding by default (deterministic,
  fast); `num_beams=4` offered for slightly better quality at ~4x latency.
- **app.py** — two tabs (single / batch). Word-level diff highlighting via difflib
  (green = inserted, red strike-through = removed). Sidebar: beams + max length.
  Batch mode offers CSV download. Verified to boot headless and serve.

## Environment
`uv` venv pinned to Python 3.12 (system 3.14 too new for torch). CPU-only torch
(AMD Ryzen 3 5300U, no GPU, 7.3 GB RAM). Install:
```
uv venv --python 3.12 .venv
uv pip install --python .venv torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv -r app/requirements.txt
```

## Resolved during build
- **transformers 5.19 API changes** — `AutoModelForSeq2SeqGeneration` →
  `AutoModelForSeq2SeqLM`; Trainer `tokenizer=` → `processing_class=`;
  `warmup_ratio` → `warmup_steps`; `as_target_tokenizer()` removed (T5 needs no
  target-side special handling).
- **`tie_word_embeddings`** — checkpoint stores *separate* `shared`/`lm_head`
  weights, so the correct setting is `false`. Set in `model/config.json`. The
  remaining transformers warning is cosmetic (comes from safetensors metadata, not
  the config) and does not affect behavior.
- **Task prefix** — none needed; the model was trained on raw sentences.

## Model accuracy (honest number)
**78.0%** exact-match on a 300-pair held-out set (from 64.7% baseline).
Capacity ceiling is ~78–80% on this 77M model; the residual failures are
subword-tokenizer orthographic edge cases (`crys`→`cries`, `an useful`→`a useful`)
that need a larger base model — not viable on this hardware. Details in `TRAINING.md`.
