# GrammerAssistance — Offline Writing Assistant (T5 grammar correction + rule-based feedback)

NLP course project: a local writing assistant that finds spelling and style problems in a
draft and can rewrite it with a fine-tuned sequence-to-sequence model. Everything runs on
the machine — no accounts, no API keys, no network calls at run time.

- **Report (graded document):** [`NLP_PROJECT_REPORT.md`](NLP_PROJECT_REPORT.md)
- **Training history:** [`TRAINING.md`](TRAINING.md)
- **Model:** fine-tuned **T5-small** (~77M params) in `model/`, stored through **Git LFS**
- **Result:** **78.0%** exact match on a 300-pair held-out set (`eval/summary.json`)

---

## 1. How it works

**Layer 1 — instant rule-based analysis** (`app/analysis.py`, milliseconds, no model):
spelling (`pyspellchecker`, ranked suggestions, personal dictionary), clarity (passive
voice, hedging, filler), engagement (clichés, adverbs, weak words, repetition), delivery
(long sentences, contractions in formal mode) and readability (Flesch Reading Ease,
Flesch–Kincaid grade, reading time, tone). Every finding carries character offsets, so
`Accept` rewrites the exact span.

**Layer 2 — neural rewrite** (`app/engine.py`, ~1s per sentence on CPU): the draft is split
into paragraphs, then sentences; each sentence is passed through the fine-tuned T5-small
with greedy decoding and rejoined, preserving paragraph structure. The UI shows
Original | Rewritten with an inline diff and `Accept rewrite`. The model is loaded once and
cached; rewriting is button-gated so typing stays responsive.

**UI** (`app/app_simple.py` + `app/style_simple.css`, Streamlit): a draft card with the
textarea and the ✦ Rewrite button in its footer, beside a score/suggestions panel. Below
560 px wide the panel moves under the card. Shortcuts: `Ctrl+Enter` rewrite, `Ctrl+I`
suggestions, `Escape` close.

---

## 2. How to run

Windows, Python 3.12, ~500 MB disk, 4 GB RAM, no GPU needed.

Double-click **`app/run.bat`**, then open <http://localhost:8501>. If Windows blocks the
console interpreter, the launcher starts detached with `pythonw.exe` and opens the browser
for you; `app/stop.bat` stops it.

Manually:

```
.venv\Scripts\python.exe -m streamlit run app\app_simple.py --server.port 8501
```

Fresh environment:

```
python -m venv .venv
.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python.exe -m pip install -r app\requirements.txt
```

Clone (the checkpoint is in Git LFS):

```
git lfs install
git clone https://github.com/wanderer3442-arch/GrammerAssistance.git
cd GrammerAssistance
git lfs pull
```

If `model/model.safetensors` is a ~130-byte text file, `git lfs pull` was skipped.

---

## 3. How to check it

| Check | Command | What you get |
|-------|---------|--------------|
| Unit tests | `python -m unittest discover -s tests -p "test_*.py" -v` | pass/fail on analysis and engine behaviour |
| Smoke probes | `python app\smoke_test.py` | model output for known edge cases |
| Held-out evaluation | `python app\evaluate.py` | rewrites `eval/heldout_300_report.csv`, `eval/trained_15_report.csv`, `eval/summary.json` (~2–5 min) |

`eval/summary.json` reports exact match, chrF, no-op rate, missed-fix rate and
overcorrection rate for the 300-pair held-out set and the 15-pair trained set.

**Reading the report:** §1–2 what is built and why offline matters · §3–4 architecture and
NLP components · §5 interface · §7 evaluation and reproduction (§7.4) · §8–10 limitations,
future work, ethics · Appendix A/B run commands and dependencies.

---

## 4. Layout

```
model/     fine-tuned checkpoint loaded by the app (Git LFS)
data/      datasets, training curve
eval/      held-out reports and summary.json
app/       app_simple.py (UI), analysis.py, engine.py, evaluate.py, training scripts
tests/     unittest suite
colab/     GPU training notebook
NLP_PROJECT_REPORT.md, TRAINING.md
```

Note: `model/config.json` must keep `tie_word_embeddings=false` and the model must use its
own `tokenizer.json` — the fine-tune rebuilt its vocabulary.
