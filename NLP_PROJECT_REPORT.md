# Offline Writing Assistant — NLP Project Report

**Course:** Natural Language Processing (College Project)
**Student Name:** K. Chetan Reddy
**Roll No:** 124AD0030
**Date:** 2026
**Repository:** <https://github.com/wanderer3442-arch/GrammerAssistance>

---

## 1. Abstract

This project is an **offline, privacy-first writing assistant** built on two complementary
NLP methods: (1) **instant rule-based analysis** — spelling, style, readability and tone in
under 50 ms with no model; and (2) **neural sequence-to-sequence rewriting** — a fine-tuned
**T5 Transformer** (77M parameters, 8 encoder + 8 decoder layers) that rewrites a draft
sentence by sentence on CPU at about one second per sentence.

The neural half is a text-to-text Transformer: the encoder reads the sentence
bidirectionally, the decoder generates the correction token by token using causal
self-attention and cross-attention over the encoder. Five rounds of targeted fine-tuning
(15 hand-written pairs → +174 → +5,000 real GEC pairs from Grammarly CoEdit) lifted exact
match from 64.7% to **78.0%** on a 300-pair held-out set. Everything runs locally with zero
network calls.

---

## 2. Problem Statement

Grammarly, LanguageTool and Word Editor all require connectivity, an account or
subscription, and upload of the user's text to external servers. For academic, legal or
personal writing that is unacceptable. This project delivers comparable assistance with all
computation on the user's own machine.

---

## 3. System Architecture

```
Streamlit UI  ──  draft card · score + suggestion panel · revisions view
      │
      ▼
analysis.py  (rules,  5–30 ms)      engine.py  (T5 Transformer, ~1 s/sentence)
spelling · style · readability      tokenize → generate → detokenize
tone · diff explanation             sentence-at-a-time, layout preserved
```

**Data flow.** The draft lives in `st.session_state["draft"]`. On each rerun the rule layer
produces spelling, style and tone findings against the live text. On **Rewrite**, the draft
is split into sentences, each is passed through T5, and the results are rejoined with the
original paragraph structure; the diff is then shown for the user to accept or reject.

---

## 4. NLP Components

### 4.1 Rule-Based Layer (`analysis.py`)

| Component | Method |
|---|---|
| Spell checking | `pyspellchecker` (Norvig + frequency corpus); skips ALL-CAPS, mid-sentence title case and tokens with digits; personal dictionary in `dictionary.json` |
| Style detection | ten regex/span detectors for passive voice, hedging, fillers, wordy phrases, clichés, `-ly` adverbs, weak words, repetition, long sentences, contractions — all returning character spans so the UI can underline precisely |
| Readability | Flesch Reading Ease, Flesch–Kincaid grade, word/sentence/paragraph counts, average and longest sentence (`textstat`) |
| Tone | formality and confidence scores from passive ratio, contraction rate, word/sentence length and hedging → Formal / Semi-formal / Casual |
| Diff explanation | `explain_diff()` labels each model edit as spelling, agreement/article, conciseness or word choice |

### 4.2 Neural Rewrite Layer — the T5 Transformer (`engine.py`)

This is the modelling core of the project: a compact **encoder–decoder Transformer**
(`T5ForConditionalGeneration`) fine-tuned on grammatical-error-correction pairs.

#### 4.2.1 Why sequence-to-sequence

Grammar correction is *transduction*: the corrected sentence usually differs in length
(`He buy a apple` → `He buys an apple`). Token tagging (BERT-style, GECToR) cannot
confidently insert or delete and needs edit-alignment labels this project does not have; a
prompted LLM is neither offline nor reproducible; T5-base (220M) would not load beside
Streamlit on the 7.3 GB CPU-only laptop. T5 also frames every task as text-to-text, so no
task-specific output head is needed — the target *is* the corrected sentence.

#### 4.2.2 Architecture

Read from `model/config.json`:

| Component | Value |
|---|---|
| Family | T5, encoder–decoder |
| Encoder / decoder layers | **8 / 8** |
| Model width `d_model` | 512 |
| Attention heads | **6** (`d_kv` = 64) |
| Feed-forward | 1024, gated GELU |
| Normalisation | RMSNorm, pre-norm, no bias |
| Positions | **relative** attention bias, 32 log-spaced buckets, max distance 128 |
| Vocabulary | 32,128 SentencePiece subwords, shared by both stacks |
| Parameters / weights | ≈ 77 M / ~308 MB float32 safetensors |

T5 has **no absolute position embeddings** (order enters through a learned relative bias
added to the attention logits) and **RMSNorm without bias**, which keeps parameters low for
a given width.

#### 4.2.3 Tokenization

`AutoTokenizer` loads the checkpoint's SentencePiece model; one 32k vocabulary serves source
and target, so input and label ids share an embedding space. Subwords mean no
out-of-vocabulary tokens (`university` → `un` + `iversity`). Sentences are encoded with
`truncation=True, max_length=256`; decoding uses `skip_special_tokens=True`.

#### 4.2.4 What the attention layers compute

**Encoder (8 blocks).** Multi-head self-attention then a gated feed-forward network, each in
a residual branch with RMSNorm. Attention is **bidirectional**, so the representation of
*apple* already knows that *a* precedes it. Per head:

```
scores  = (Q · Kᵀ) / √d_kv  +  b(i, j)     # b = learned relative-position bias
weights = softmax(scores)                  # masked where needed
output  = weights · V
```

`b(i, j)` maps the offset `i − j` into one of 32 log-spaced buckets (fine nearby, saturating
at 128 tokens), each with a learned scalar per head — that is why T5 has no position table
yet still knows word order.

**Decoder (8 blocks).** **Masked self-attention** over tokens generated so far (causal mask,
so position *t* cannot see *t+1*), **cross-attention** whose queries come from the decoder
but whose keys and values come from the encoder's final states, then the feed-forward
network. Cross-attention is where each new token looks back at the source and where "this
verb needs *-s*" is decided. The decoder state is projected onto the vocabulary for the next
subword, and generation repeats until `</s>`. Cost is `O(n²·d)`, acceptable at sentence
length.

#### 4.2.5 Fine-tuning

Teacher forcing: the decoder input is the target shifted right, so one forward pass gives a
next-token prediction everywhere, and the loss is cross-entropy averaged over non-padding
positions (`DataCollatorForSeq2Seq` masks padding with `-100`). AdamW, learning rate 1e-5
(round 5) / 2e-5 (CoEdit round), ~6% linear warm-up, batch 8, fp32 on CPU, 1–2 epochs per
round to avoid overfitting small data.

| Round | Data added | Focus | Full-300 exact |
|---|---|---|---|
| 0 (received) | 15 pairs × 5 epochs | — | 64.7% |
| 1 | +174 pairs | wh-questions, be-verb, articles | 71.7% |
| 2 | +50 pairs | article rebalance | 76.0% |
| 3 | +39 pairs | `a/an` by vowel sound, `-ies`, irregulars | 76.7% |
| 4 | +5,000 real GEC pairs | CoEdit, minimal-edit filtered | **78.0%** |

Round 4 was decisive: real human corrections taught *minimal* edits instead of rewriting,
and every category improved at once (articles 73.1 → 75.9%, irregular verbs 64.3 → 75.0%).

#### 4.2.6 Inference and decoding

Paragraphs are split on blank lines, each into sentences, each sentence tokenized, passed to
`model.generate(do_sample=False, num_beams=1)` under `torch.no_grad()` with `model.eval()`,
then detokenized and rejoined.

- **Greedy decoding** is deterministic (same draft → same rewrite) and ~4× cheaper than
  4-beam search; beams and minimal-edit reranking are compared in `app/eval_decoding.py`.
- **Loaded once** behind `lru_cache(maxsize=1)`: ~3 s cold start, then ~1 s per sentence.
- **Sentence-level**, because the model was fine-tuned on sentence pairs — this bounds
  hallucination on long paragraphs.
- **Layout preserved**: paragraph breaks and indentation are restored verbatim.

#### 4.2.7 Where it falls short

78.0% exact match, chrF 96.22, 37.67% no-op, 17.67% missed fixes, **0.00%
overcorrection** on 60 already-correct inputs, 100% on the 15 training pairs (correcting,
not memorising). By family: subject–verb agreement 100% (n=17), negation 100% (n=3),
modals 100% (n=2), wh-inversion 88.2% (n=17), articles 75.5% (n=139), `be`-verb 71.1%
(n=45). Residual error is article usage and **phonological/orthographic** cases the
tokenizer does not expose — `an useful`, `an university`, `cries → crys`. That is a capacity
ceiling at 77M parameters on this hardware, documented rather than hidden.

#### 4.2.8 Why both layers

| | Rule layer | Transformer |
|---|---|---|
| Latency | 5–30 ms | ~1 s / sentence |
| Strength | explicit, explainable, exact spans | contextual grammar, agreement, fluency |
| Weakness | no cross-clause inference | opaque: cannot guarantee what changed |
| Editing | accept a precise span | accept or reject a whole rewrite |

The rules give instant reversible feedback; the Transformer supplies corrections that
hand-written patterns cannot express.

---

## 5. User Interface

A single dark editorial surface (petrol `#1b353e`, verdigris `#5fc7ad`, Sitka Text / Segoe UI
Variable / Cascadia Mono) with **zero chrome**: one draft card and one score-and-suggestions
panel side by side, the panel at `clamp(240px, 32%, 320px)` and moving beneath the draft
below 560 px. States progress Empty → Writing → Score + suggestions → Revisions (original vs
rewritten plus an inline diff). Severity-coloured wavy underlines mark findings; `Ctrl+Enter`
rewrites, `Ctrl+I` jumps to suggestions, `Escape` dismisses.

---

## 6. Implementation

| Component | Version |
|---|---|
| Python | 3.12 (tested 3.12.14) |
| Streamlit | 1.65.0 |
| PyTorch | 2.14.1+cpu |
| Transformers | 5.19.0 |
| pyspellchecker | 0.9.1 |
| textstat | 0.7.13 |

```
app/    app_simple.py (UI) · analysis.py · engine.py · evaluate.py · run.bat · requirements.txt
tests/  unittest suite for analysis.py and engine.py
eval/   heldout_300_report.csv · trained_15_report.csv · summary.json
model/  T5 checkpoint (config.json, model.safetensors, tokenizer.json) — Git LFS
```

| Operation | Typical time |
|---|---|
| Spell check / style / tone | 5–15 / 10–30 / <5 ms |
| T5 rewrite | ~1 s per sentence |
| UI rerun | ~50 ms |

---

## 7. Evaluation

**Rule layer:** precise spans with a proper-noun filter keeping false positives low; recall
limited by regex coverage; Flesch values match standard implementations; tone is heuristic.

**Neural layer (300 held-out pairs):** exact match **78.0%** (234/300), mean chrF 96.22,
37.67% no-op, 17.67% missed fixes, 0.00% overcorrection (0/60), 100% on the 15 trained
pairs. GLEU/BLEU and human preference are future work.

**Reproducing:**

```
.venv\Scripts\python.exe app\evaluate.py
```

writes `eval/heldout_300_report.csv`, `eval/trained_15_report.csv` and `eval/summary.json`
in ~2–5 minutes on CPU. Rules are checked with
`python -m unittest discover -s tests -p "test_*.py"`; the UI/model path with
`python app\smoke_test.py`.

**Usability:** ~3 s cold start, 1–4 s warm rewrite, ~1.2 GB RAM, ~320 MB disk.

---

## 8. Limitations & Future Work

1. **English only** — spellchecker and checkpoint are English-specific.
2. **No cross-clause grammar in the rule layer** — that is the Transformer's job.
3. **Occasional hallucination** on unusual input; bounded by sentence-level processing.
4. **No persistence** — the draft is lost on browser close, by design.
5. **Single-threaded UI** — Streamlit reruns block during inference.
6. **77M capacity ceiling** — article and vowel-sound errors stay near 75% (§4.2.7).

Future work: multilingual mT5, background async inference, explicit agreement rules,
export to `.txt`/`.md`/`.docx`, opt-in encrypted sync, a plugin API for custom rules.

---

## 9. Ethics

Privacy by design: no telemetry, no network requests, no accounts, no data collection —
training on user text is not even possible. The tool works in air-gapped environments, the
rule logic is open source and the weights are inspectable.

---

## 10. Conclusion

High-quality writing assistance does not require cloud infrastructure. Pairing classical NLP
(fast, interpretable, precise) with a compact fine-tuned T5 Transformer (holistic rewriting)
gives a professional editing experience on commodity hardware, offline, with no subscription.

The modelling result is the part that matters here: from a checkpoint that memorised 15
examples and scored 64.7%, five rounds of diagnosed fine-tuning reached 78.0% on held-out
data with zero overcorrection, inside 308 MB of weights and about one second per sentence on
a CPU. The remaining gap is a capacity ceiling rooted in subword tokenization and model size,
not in missing data.

---

## Appendix A: Running the Project

```bash
git lfs install
git clone https://github.com/wanderer3442-arch/GrammerAssistance.git
cd GrammerAssistance
git lfs pull

app\run.bat            # or:
.venv\Scripts\python.exe -m streamlit run app\app_simple.py --server.port 8501
```

Open <http://localhost:8501>. `run.bat` falls back to `pythonw.exe` if the console
interpreter is blocked by policy, starting the server detached; `app\stop.bat` stops it.

Requirements: Python 3.12, 4 GB RAM minimum, no GPU, ~500 MB disk plus ~310 MB for the
checkpoint. Fresh install:

```
python -m venv .venv
.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python.exe -m pip install -r app\requirements.txt
```

**Entry points:** `analysis.spell_check / style_report / tone_report / explain_diff` and
`engine.polish(text, max_length=256, num_beams=1)`.

---

**End of Report**
