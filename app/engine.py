"""Model engine for the offline writing assistant.

Loads the fine-tuned T5 once and exposes:
  - polish(text)          : single-paragraph polish (returns corrected string)
  - polish_paragraphs(list): batch polish for multi-paragraph input
The model is slow (~1s/sentence on CPU), so the UI pairs it with the instant
rule-based layer in analysis.py to give immediate feedback while this runs.
"""
from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
# Canonical checkpoint: the project's own 77M T5 fine-tune (78.0% on the
# 300-pair held-out set). Set GRAMMAR_MODEL_DIR to point at another checkpoint
# (for example 124AD0030/grammar_corrector_finetuned) without editing code.
_env_model = os.environ.get("GRAMMAR_MODEL_DIR")
MODEL_DIR = (Path(_env_model).expanduser().resolve() if _env_model
             else ROOT / "model")

# Split into paragraphs on blank lines, keep non-empty ones.
# Capture the delimiter so polish() can put the writer's paragraph spacing back.
_PARA_SPLIT = re.compile(r"(\n\s*\n)")


@lru_cache(maxsize=1)
def _load():
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
    model.eval()
    return tok, model


def load_model() -> None:
    """Warm the cache so the first request isn't slow. Returns None."""
    _load()


def _split_sentences(text: str) -> list[str]:
    """Crude but effective sentence splitter that preserves the original text
    for offset math. Keeps terminal punctuation attached to each sentence."""
    out: list[str] = []
    start = 0
    for i, ch in enumerate(text):
        if ch in ".!?":
            # consume trailing quotes/brackets
            j = i + 1
            while j < len(text) and text[j] in "\"'”’)]":
                j += 1
            seg = text[start:j].strip()
            if seg:
                out.append(seg)
            start = j
    tail = text[start:].strip()
    if tail:
        out.append(tail)
    return out


def _polish_block(text: str, max_length: int, num_beams: int) -> str:
    """Polish one paragraph while retaining its surrounding whitespace."""
    if not text.strip():
        return text

    leading = text[: len(text) - len(text.lstrip())]
    trailing = text[len(text.rstrip()):]
    body = text.strip()
    sentences = _split_sentences(body)
    if not sentences:
        return text

    tok, model = _load()
    corrected: list[str] = []
    with torch.no_grad():
        for sent in sentences:
            enc = tok(sent, return_tensors="pt", truncation=True,
                      max_length=max_length)
            out = model.generate(
                **enc,
                do_sample=False,
                num_beams=num_beams,
                max_length=max_length,
            )
            corrected.append(tok.decode(out[0], skip_special_tokens=True).strip())
    return leading + " ".join(corrected) + trailing


def polish(text: str, max_length: int = 256, num_beams: int = 1) -> str:
    """Polish a single paragraph: split into sentences, correct each, rejoin.

    The T5 model was trained on single sentences, so we segment the paragraph,
    correct sentence-by-sentence, and glue back together with the original
    inter-sentence spacing. Already-correct sentences come back unchanged.
    """
    if not text or not text.strip():
        return ""
    # Keep blank-line paragraph boundaries. The model is trained sentence by
    # sentence, but flattening a multi-paragraph draft into one block makes a
    # rewrite surprisingly destructive to the writer's layout.
    chunks = _PARA_SPLIT.split(text)
    return "".join(
        chunk if _PARA_SPLIT.fullmatch(chunk or "")
        else _polish_block(chunk, max_length, num_beams)
        for chunk in chunks
    )


def polish_paragraphs(paragraphs: list[str], max_length: int = 256,
                      num_beams: int = 1) -> list[str]:
    """Polish each paragraph independently. Blank-line separated input is split
    by the caller; this handles a list."""
    return [polish(p, max_length=max_length, num_beams=num_beams) for p in paragraphs]


if __name__ == "__main__":
    demo = ("Why he work here? The report was written by me and it is very good. "
            "He buy a apple every day. Their going to the store.")
    print("IN :", demo)
    print("OUT:", polish(demo))
