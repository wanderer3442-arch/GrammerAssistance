"""Offline Writing Assistant — Minimal, Focused, Free.

Two layers working together:
  * analysis.py -> INSTANT rule-based feedback (spelling, style, readability)
  * engine.py   -> fine-tuned T5 rewrite (~1s/sentence on CPU), gated behind a button.

Design: single writing surface. No sidebar, no tabs, no chrome. Just writing.
"""
from __future__ import annotations

import difflib
import html
import json
import math
import re
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path

import streamlit as st

import analysis
import engine

DICT_PATH = Path(__file__).with_name("dictionary.json")

st.set_page_config(
    page_title="Writing Assistant",
    page_icon="✎",
    layout="centered",
    initial_sidebar_state="collapsed",
)


def _load_personal_dict() -> set[str]:
    try:
        data = json.loads(DICT_PATH.read_text(encoding="utf-8"))
        return {str(w).lower() for w in data}
    except Exception:
        return set()


if "personal_dict" not in st.session_state:
    st.session_state["personal_dict"] = _load_personal_dict()

# Minimal shell state
for k, v in {
    "show_suggestions": False,
    "dismissed": [],
    "polished": None,
    "pol_source": None,
    "pol_time": 0.0,
}.items():
    st.session_state.setdefault(k, v)

st.session_state.setdefault("draft", st.session_state.get("pm_text", ""))


def _set_text(value: str) -> None:
    st.session_state["draft"] = value
    st.session_state["pm_text"] = value


def _sync_draft() -> None:
    st.session_state["draft"] = st.session_state.get("pm_text", "")


def _use_example(txt: str) -> None:
    _set_text(txt)
    st.session_state.pop("polished", None)
    st.session_state.pop("pol_source", None)
    st.session_state["dismissed"] = []
    st.session_state["show_suggestions"] = False


def _clear() -> None:
    _set_text("")
    st.session_state.pop("polished", None)
    st.session_state.pop("pol_source", None)
    st.session_state["dismissed"] = []
    st.session_state["show_suggestions"] = False


def _apply_span(start: int, end: int, replacement: str) -> None:
    t = st.session_state.get("pm_text", "")
    if 0 <= start < end <= len(t):
        _set_text(t[:start] + replacement + t[end:])
    else:
        word = t[start:end] if end <= len(t) else ""
        _set_text(t.replace(word, replacement, 1))
    st.session_state.pop("polished", None)
    st.session_state.pop("pol_source", None)


def _fix_all() -> None:
    t = st.session_state.get("pm_text", "")
    hits = analysis.spell_check(t, ignore=st.session_state.get("personal_dict", set()))
    if hits:
        _set_text(analysis.apply_spelling_fixes(t, hits))
    st.session_state.pop("polished", None)
    st.session_state.pop("pol_source", None)


def _add_to_dict(word: str) -> None:
    words = set(st.session_state.get("personal_dict", set()))
    words.add(word.lower())
    st.session_state["personal_dict"] = words
    try:
        DICT_PATH.write_text(json.dumps(sorted(words)), encoding="utf-8")
    except Exception:
        pass


def _dismiss(sid: str) -> None:
    cur = list(st.session_state.get("dismissed", []))
    if sid not in cur:
        cur.append(sid)
    st.session_state["dismissed"] = cur[-200:]


def _accept_rewrite() -> None:
    polished = st.session_state.get("polished")
    if polished:
        _set_text(polished)
    st.session_state.pop("polished", None)
    st.session_state.pop("pol_source", None)
    st.session_state["show_suggestions"] = False


def _copy_rewrite() -> None:
    polished = st.session_state.get("polished")
    if not polished:
        return
    try:
        tmp = Path(tempfile.gettempdir()) / "gt_clipboard.txt"
        tmp.write_text(polished, encoding="utf-8")
        subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             f"Set-Clipboard -Value (Get-Content -Raw -Encoding UTF8 '{tmp}')"],
            timeout=10, check=False,
        )
        st.toast("Copied!", icon="✓")
    except Exception:
        st.toast("Copy failed", icon="✗")


# ────────────────────────────────────────────── Helpers
_BUCKET_OF_CAT = {
    "Spelling": "corr", "Grammar": "corr", "Agreement": "corr",
    "Article": "corr", "Conciseness": "corr",
    "Clarity": "clar", "Wordiness": "clar", "Filler": "clar",
    "Hedging": "clar",
    "Cliché": "eng", "Adverbs": "eng", "Weak word": "eng",
    "Repetition": "eng", "Word choice": "eng",
    "Readability": "del", "Tone": "del",
}
_LEVEL = {"Light": 1, "Balanced": 2, "Strict": 3}
_BUCKET_LEVEL = {"corr": 1, "del": 1, "clar": 2, "eng": 3}

_CONTRACTION_MAP = {
    "can't": "cannot", "won't": "will not", "don't": "do not",
    "doesn't": "does not", "didn't": "did not", "isn't": "is not",
    "aren't": "are not", "wasn't": "was not", "weren't": "were not",
    "hasn't": "has not", "haven't": "have not", "hadn't": "had not",
    "wouldn't": "would not", "shouldn't": "should not",
    "couldn't": "could not", "mustn't": "must not", "it's": "it is",
    "that's": "that is", "there's": "there is", "here's": "here is",
    "what's": "what is", "who's": "who is", "let's": "let us",
    "i'm": "I am", "i've": "I have", "i'll": "I will", "i'd": "I would",
    "you're": "you are", "you've": "you have", "you'll": "you will",
    "you'd": "you would", "we're": "we are", "we've": "we have",
    "we'll": "we will", "we'd": "we would", "they're": "they are",
    "they've": "they have", "they'll": "they will", "they'd": "they would",
    "he's": "he is", "she's": "she is", "he'll": "he will",
    "she'll": "she will", "he'd": "he would", "she'd": "she would",
}


def _sev(cat: str) -> str:
    return _BUCKET_OF_CAT.get(cat, "clar")


def _match_case(source: str, replacement: str) -> str:
    if source[:1].isupper() and replacement[:1].islower():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def score_ring(score: int | None, size: int = 64) -> str:
    stroke = 5
    r = (size - stroke) / 2
    c = 2 * math.pi * r
    cx = cy = size / 2
    if score is None:
        pct, color, label = 0.0, "var(--line)", "—"
    else:
        pct = max(0.0, min(100.0, score)) / 100
        color = ("var(--accent)" if score >= 85 else "var(--clar)" if score >= 65
                 else "var(--corr)")
        label = str(score)
    dash = c * pct
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}">'
        f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="var(--line)" '
        f'stroke-width="{stroke}"/>'
        f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{color}" '
        f'stroke-width="{stroke}" stroke-linecap="round" '
        f'stroke-dasharray="{dash:.1f} {c - dash:.1f}" '
        f'transform="rotate(-90 {cx} {cy})"/>'
        f'<text x="{cx}" y="{cy + 6}" text-anchor="middle" '
        f'font-family="Georgia,serif" font-size="16" font-weight="600" '
        f'fill="var(--ink)">{label}</text>'
        f'</svg>'
    )


def markup_html(text: str, spans: list[tuple[int, int, str, str]]) -> str:
    prio = {"corr": 0, "clar": 1, "eng": 2, "del": 3}
    spans = sorted(spans, key=lambda s: (s[0], prio.get(s[2], 3)))
    out: list[str] = []
    pos = 0
    for start, end, sev, tip in spans:
        if start < pos or end <= start or start >= len(text):
            continue
        out.append(html.escape(text[pos:start]))
        out.append(
            f'<span class="w-mk {sev}" title="{html.escape(tip)}">'
            f'{html.escape(text[start:end])}</span>'
        )
        pos = end
    out.append(html.escape(text[pos:]))
    return "".join(out)


def diff_html(original: str, polished: str) -> str:
    o, p = original.split(), polished.split()
    sm = difflib.SequenceMatcher(None, o, p)
    parts: list[str] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            parts.append(" ".join(html.escape(w) for w in o[i1:i2]))
        elif tag == "delete":
            parts += [f'<span class="w-del">{html.escape(w)}</span>' for w in o[i1:i2]]
        elif tag == "insert":
            parts += [f'<span class="w-ins">{html.escape(w)}</span>' for w in p[j1:j2]]
        else:
            parts += [f'<span class="w-del">{html.escape(w)}</span>' for w in o[i1:i2]]
            parts += [f'<span class="w-ins">{html.escape(w)}</span>' for w in p[j1:j2]]
    return " ".join(parts)


# ────────────────────────────────────────────── Instant analysis
sensitivity = "Balanced"
formal_tone = False
level = _LEVEL.get(sensitivity, 2)

text = st.session_state.get("draft", "") or ""
spelling: list = []
style = None
tone = None
issues: list[dict] = []
score: int | None = None
polished = None
t_instant = 0.0
sub_scores: dict[str, int] = {}

if text.strip():
    t0 = time.time()
    spelling = analysis.spell_check(
        text, ignore=st.session_state.get("personal_dict", set()))
    style = analysis.style_report(text, formal=bool(formal_tone))
    tone = analysis.tone_report(text)
    t_instant = time.time() - t0

    if st.session_state.get("pol_source") == text:
        polished = st.session_state.get("polished")
    model_changes = (analysis.explain_diff(text.strip(), polished)
                     if polished else [])

    dismissed = set(st.session_state.get("dismissed", []))
    seen: set[int] = set()

    def push(sid, sev, cat, before, after, expl, accept=None, src="Rule", span=None):
        if sid in dismissed:
            return
        if _BUCKET_LEVEL[sev] > level:
            return
        if span is not None:
            if span[0] in seen:
                return
            seen.add(span[0])
        issues.append(dict(sid=sid, sev=sev, cat=cat, src=src, before=before,
                           after=after, expl=expl, accept=accept))

    for h in spelling:
        push(f"sp_{h.start}_{h.word}", "corr", "Spelling", h.word,
             (h.suggestions[0] if h.suggestions else None),
             "Not in the dictionary — likely a typo.",
             accept=(h.start, h.end, h.suggestions[0]) if h.suggestions else None,
             span=(h.start, h.end))
    for s, e in style.passive_spans:
        push(f"pv_{s}", "clar", "Clarity", text[s:e], None,
             "Passive voice distances the reader from the action — prefer active voice.", span=(s, e))
    for s, e in style.hedging_spans:
        push(f"hg_{s}", "clar", "Hedging", text[s:e], None,
             "Softeners dilute your point — cutting them reads with more confidence.", span=(s, e))
    for s, e in style.filler_spans:
        push(f"fl_{s}", "clar", "Filler", text[s:e], None,
             "Intensifier — the sentence is stronger without it.", span=(s, e))
    for s, e, repl in style.wordy_spans:
        before = text[s:e]
        push(f"wd_{s}", "clar", "Wordiness", before, repl,
             f"Wordy phrase — “{repl}” says the same thing in fewer words.",
             accept=(s, e, _match_case(before, repl)), span=(s, e))
    for s, e in style.cliche_spans:
        push(f"cl_{s}", "eng", "Cliché", text[s:e], None,
             "Cliché — a fresher phrase lands better.", span=(s, e))
    for s, e in style.adverb_spans:
        push(f"av_{s}", "eng", "Adverbs", text[s:e], None,
             "Adverb — a stronger verb usually does the job on its own.", span=(s, e))
    for s, e, repl in style.weak_spans:
        before = text[s:e]
        push(f"wk_{s}", "eng", "Weak word", before, repl,
             "Vague word — a more specific choice adds precision.",
             accept=(s, e, _match_case(before, repl)), span=(s, e))
    for s, e, word in style.repetition_spans:
        push(f"rp_{s}", "eng", "Repetition", text[s:e], None,
             f"“{word}” is repeated — vary the wording or cut the repeat.", span=(s, e))
    for s, e in style.sentence_spans:
        snippet = text[s:e].strip()
        push(f"ls_{s}", "del", "Readability",
             (snippet if len(snippet) <= 90 else snippet[:90] + "…"), None,
             f"{len(snippet.split())} words — split it up to keep the reader with you.", span=(s, e))
    for s, e in style.contraction_spans:
        raw = text[s:e]
        expanded = _CONTRACTION_MAP.get(raw.lower())
        push(f"ct_{s}", "del", "Tone", raw, expanded,
             "Contraction — spell it out for a formal tone.",
             accept=(s, e, expanded) if expanded else None, span=(s, e))
    for i, c in enumerate(model_changes):
        push(f"mc_{i}", _sev(c.category), c.category, c.original,
             (c.polished or None), c.explanation, src="Model")

    n_corr = sum(1 for i in issues if i["sev"] == "corr")
    n_clar = sum(1 for i in issues if i["sev"] == "clar")
    n_eng = sum(1 for i in issues if i["sev"] == "eng")
    n_del = sum(1 for i in issues if i["sev"] == "del")
    sub_scores = {
        "Correctness": max(0, 100 - 7 * n_corr),
        "Clarity": max(0, 100 - 4 * n_clar),
        "Engagement": max(0, 100 - 2 * n_eng),
        "Delivery": max(0, 100 - 4 * n_del),
    }
    score = max(0, min(100, 100 - 6 * n_corr - 4 * n_clar - 2 * n_eng - 3 * n_del))

# ────────────────────────────────────────────── Rewrite
# The button itself is rendered inside the draft card below; the click is
# handled after the layout so no button can render above the page header.
rewrite_clicked = False

# ────────────────────────────────────────────── CSS
CSS = Path(__file__).with_name("style_simple.css").read_text(encoding="utf-8")
st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)

# CSS variables - dynamic page width based on layout
page_w = "1100px" if text.strip() else "720px"
st.markdown(
    f"<style>:root{{--accent:#5fc7ad;--accent-2:#95ecd5;--ink:#e9f2ef;"
    f"--line:#31525b;--corr:#f08a80;--clar:#74b9e8;--eng:#b09be8;--del:#7f9fcc;"
    f"--page:#1b353e;--paper:#1b353e;--font-doc:'Sitka Text',Georgia,serif;"
    f"--font-size-base:16px;--leading:1.75;--page-w:{page_w};}}</style>",
    unsafe_allow_html=True,
)

# ────────────────────────────────────────────── UI
st.markdown('<div class="w-wrap">', unsafe_allow_html=True)

# Header — minimal
st.markdown(
    '<div class="w-header">'
    '<span class="w-title">✎ Writing Assistant</span>'
    '<span class="w-sub">Private • Offline • Free</span>'
    '</div>', unsafe_allow_html=True)

# Writing surface
_has_text = bool(text.strip())

if not _has_text:
    # Empty state — clean and inviting
    st.markdown(
        '<div class="w-empty">'
        '<h1>Write with confidence.</h1>'
        '<p>Start typing or pick a sample below. Suggestions appear instantly; '
        'Rewrite gives you a polished second pass.</p>'
        '</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3, gap="small")
    with c1:
        st.button("Rough draft", key="hero_rough",
                  on_click=_use_example,
                  args=("i recieved your message yesterday and wanted to reply sooner "
                        "but i was definately busy . their going to seperate the "
                        "reports tommorow .",), width="stretch")
    with c2:
        st.button("Formal note", key="hero_formal",
                  on_click=_use_example,
                  args=("The report was written by me and it is very good. "
                        "The data was analyzed by us and the results were very surprising.",),
                      width="stretch")
    with c3:
        st.button("Clean copy", key="hero_clean",
                  on_click=_use_example,
                  args=("She walks to the park every morning. The book was interesting.",),
                      width="stretch", type="primary")

else:
    # Side-by-side layout when there are suggestions
    # Draft and the score/suggestions panel always sit side by side
    left_col, right_col = st.columns([2.5, 1], gap="small")

    with left_col:
        with st.container(key="draft_card"):
            # Textarea — the main writing surface
            st.text_area(
                "Document",
                key="pm_text",
                height=520,
                label_visibility="collapsed",
                placeholder="Begin with a sentence.",
                on_change=_sync_draft,
            )
            # Rewrite sits in the draft card, under the text
            with st.container(key="draft_actions"):
                rewrite_clicked = st.button(
                    "✦ Rewrite", type="primary", key="rewrite_btn",
                    width="stretch",
                    help="Rewrite with local model (Ctrl+Enter)",
                )

    # Right column — Suggestions Panel
    if right_col is not None:
        with right_col:
            with st.container(key="side_panel"):

                # Score ring + summary
                if score is not None:
                    verdict_text = ("Reads clean." if score >= 95 else
                                    "A few things to tighten." if score >= 80 else
                                    "Worth a pass.")
                    st.markdown(
                        f'<div class="w-side-score">'
                        f'{score_ring(score, 72)}'
                        f'<div class="w-side-text">'
                        f'<span class="w-side-label">Writing score</span>'
                        f'<span class="w-side-verdict">{verdict_text}</span>'
                        f'</div></div>', unsafe_allow_html=True)

                # Quick actions
                n_spell = sum(1 for i in issues if i["cat"] == "Spelling")
                if n_spell:
                    st.button(f"✓ Fix all spelling ({n_spell})", key="fixall",
                              on_click=_fix_all, width="stretch")

                # Suggestions list
                st.markdown('<div class="w-side-sugs-title">Suggestions</div>', unsafe_allow_html=True)

                if not issues:
                    st.markdown(
                        '<div class="w-side-empty">Nothing to fix — this draft reads clean.</div>',
                        unsafe_allow_html=True,
                    )

                for it in issues:
                    sev = it["sev"]
                    sid = it["sid"]
                    phrase = f'<span class="w-sug-old">{html.escape(it["before"])}</span>'
                    if it["after"]:
                        phrase += (f'<span class="w-sug-arrow">→</span>'
                                   f'<span class="w-sug-new">{html.escape(it["after"])}</span>')

                    st.markdown(
                        f'<div class="w-side-item {sev}">'
                        f'<span class="w-sug-badge">{it["cat"]}</span>'
                        f'<div class="w-sug-phrase">{phrase}</div>'
                        f'<div class="w-sug-expl">{html.escape(it["expl"])}</div>',
                        unsafe_allow_html=True)

                    if it["accept"]:
                        s, e, rep = it["accept"]
                        cols = st.columns(3 if it["cat"] == "Spelling" else 2, gap="small")
                        with cols[0]:
                            st.button("Accept", key=f"acc_{sid}", type="primary",
                                      on_click=_apply_span, args=(s, e, rep),
                                      width="stretch", use_container_width=True)
                        if it["cat"] == "Spelling":
                            with cols[1]:
                                st.button("Add word", key=f"add_{sid}",
                                          on_click=_add_to_dict, args=(it["before"],),
                                          width="stretch", use_container_width=True)
                            with cols[2]:
                                st.button("Dismiss", key=f"dis_{sid}",
                                          on_click=_dismiss, args=(sid,),
                                          width="stretch", use_container_width=True)
                        else:
                            with cols[1]:
                                st.button("Dismiss", key=f"dis_{sid}",
                                          on_click=_dismiss, args=(sid,),
                                          width="stretch", use_container_width=True)
                    else:
                        st.button("Dismiss", key=f"dis_{sid}",
                                  on_click=_dismiss, args=(sid,),
                                  width="stretch", use_container_width=True)

                    st.markdown('</div>', unsafe_allow_html=True)


# ────────────────────────────────────────────── Rewrite run
if rewrite_clicked and text.strip():
    with st.spinner(f"Rewriting — {style.sentences} sentence(s), ~{max(1, style.sentences)}s..."):
        try:
            t1 = time.time()
            new_text = engine.polish(text, max_length=256, num_beams=1)
            t_model = time.time() - t1
        except Exception as exc:
            st.error(f"Model error: {exc}")
        else:
            st.session_state["polished"] = new_text
            st.session_state["pol_source"] = text
            st.session_state["pol_time"] = t_model
            st.session_state["show_suggestions"] = True
            st.rerun()

polished = (st.session_state.get("polished")
            if st.session_state.get("pol_source") == text else None)
t_model = st.session_state.get("pol_time", 0.0) if polished else 0.0

# ────────────────────────────────────────────── Revisions (after rewrite)
if polished and text.strip():
    st.markdown('<div class="w-revisions">', unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1], gap="small")
    with c1:
        st.button("✓ Accept rewrite", key="acc_rw", type="primary",
                  on_click=_accept_rewrite, width="stretch")
    with c2:
        st.button("Copy", key="cpy_rw", on_click=_copy_rewrite, width="stretch")

    st.markdown('<div class="w-diff-title">Track Changes</div>', unsafe_allow_html=True)

    changed = polished.strip() != text.strip()
    col1, col2 = st.columns(2, gap="small")
    with col1:
        st.markdown(
            '<div class="w-rev-pane"><div class="w-rev-label">Original</div>'
            f'<div class="w-rev-text">{html.escape(text.strip())}</div></div>',
            unsafe_allow_html=True)
    with col2:
        st.markdown(
            '<div class="w-rev-pane"><div class="w-rev-label">Rewritten</div>'
            f'<div class="w-rev-text">{html.escape(polished.strip())}</div></div>',
            unsafe_allow_html=True)

    if changed:
        st.markdown(
            f'<div class="w-diff-inline">{diff_html(text.strip(), polished.strip())}</div>',
            unsafe_allow_html=True)

    st.caption(f"⏱ Rules {t_instant * 1000:.0f}ms • Model {t_model:.2f}s")
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown(
    '<div class="w-footer">'
    '<span>Offline — nothing leaves this device</span>'
    '<span>Ctrl+Enter rewrite • Ctrl+I suggestions</span>'
    '</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ────────────────────────────────────────────── Keyboard shortcuts
st.html(
    """<script>(function(){
    const p = window.parent.document;
    function hit(sel){
      const b = p.querySelector(sel);
      if (b && !b.disabled) b.click();
    }
    function onKey(e){
      const mod = e.metaKey || e.ctrlKey;
      if (mod && e.key === 'Enter') { e.preventDefault();
        hit('.st-key-rewrite_btn button'); }
      if (mod && e.key.toLowerCase() === 'i') { e.preventDefault();
        // Scroll to suggestions panel if visible
        const sugPanel = p.querySelector('.st-key-side_panel');
        if (sugPanel) { sugPanel.scrollIntoView({behavior: 'smooth', block: 'center'}); } }
      if (e.key === 'Escape') {
        const details = p.querySelector('details.w-sug-list[open]');
        if (details) details.removeAttribute('open'); }
    }
    function attach(){
      try {
        if (p.__gtKeyHandler) {
          p.removeEventListener('keydown', p.__gtKeyHandler, true);
        }
        p.__gtKeyHandler = onKey;
        p.addEventListener('keydown', onKey, true);
      } catch (err) {}
    }
    attach();
    setInterval(attach, 1200);
    })();</script>""")

# Print handling
if st.session_state.pop("do_print", False):
    st.html("<script>window.parent.print();</script>")