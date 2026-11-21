"""Offline Writing Assistant — Refined, Focused, Elegant.

Two layers working together:
  * analysis.py -> INSTANT rule-based feedback (spelling, style, readability)
  * engine.py   -> the fine-tuned neural "rewrite" (slower, ~1s/sentence on
                    CPU), gated behind a button so typing never feels sluggish.

Design: a single focused writing surface with a collapsible insights drawer.
No ribbon, no tabs, no chrome. Just writing and insight.
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
    layout="wide",
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

# Shell state
_SHELL_DEFAULTS = {
    "show_insights": False,
    "editor_mode": "Write",
    "dismissed": [],
    "polished": None,
    "pol_source": None,
    "pol_time": 0.0,
    "doc_font": "Sitka Text",
    "doc_size": 12,
    "accent": "Verdigris",
    "page_bg": "Petrol",
    "page_w": 720,
    "line": 1.75,
    "zoom": 100,
    "copied": False,
    "do_print": False,
}
for _k, _v in _SHELL_DEFAULTS.items():
    st.session_state.setdefault(_k, _v)

# The draft is canonical. The text area is only one view of it, so the text
# survives switching to the read-only Proof page and back.
st.session_state.setdefault("draft", st.session_state.get("pm_text", ""))


def _index(options, value, fallback: int = 0) -> int:
    try:
        return list(options).index(value)
    except (TypeError, ValueError):
        return fallback


def _sync_widget(source: str, target: str) -> None:
    st.session_state[target] = st.session_state[source]


def _set_text(value: str) -> None:
    """Write the canonical draft and refresh the editor widget."""
    st.session_state["draft"] = value
    st.session_state["pm_text"] = value


def _sync_draft() -> None:
    st.session_state["draft"] = st.session_state.get("pm_text", "")

# One-time refinement pass for sessions created before the studio redesign.
if not st.session_state.get("ui_v3"):
    st.session_state["accent"] = "Verdigris"
    st.session_state["page_bg"] = "Petrol"
    st.session_state["page_w"] = 720
    st.session_state["line"] = 1.75
    st.session_state["editor_mode"] = "Write"
    st.session_state["ui_v3"] = True

_ACCENTS = {
    "Verdigris": ("#5fc7ad", "#95ecd5"),
    "Coral": ("#f08a80", "#ffb4aa"),
    "Cobalt": ("#7aa2ff", "#abc4ff"),
    "Lilac": ("#b09be8", "#d1c4f6"),
}
_PAGE_BGS = {
    "Petrol": "#1b353e",
    "Slate": "#1b2b36",
    "Forest": "#12271f",
    "Plum": "#241d2e",
}

# Bring older sessions out of the legacy dark palette without touching the
# writing workflow or the frozen model.
if not st.session_state.get("ui_v4"):
    if st.session_state.get("accent") not in _ACCENTS:
        st.session_state["accent"] = "Verdigris"
    if st.session_state.get("page_bg") not in _PAGE_BGS:
        st.session_state["page_bg"] = "Petrol"
    st.session_state.pop("accent_widget", None)
    st.session_state.pop("page_bg_widget", None)
    st.session_state["ui_v4"] = True

_FONT_STACKS = {
    "Sitka Text": '"Sitka Text", "Sitka Display", "Constantia", Georgia, serif',
    "Constantia": '"Constantia", "Sitka Text", Georgia, serif',
    "Aptos": '"Aptos", "Aptos Display", "Calibri", "Segoe UI", sans-serif',
    "Calibri": '"Calibri", "Segoe UI", sans-serif',
    "Georgia": 'Georgia, "Times New Roman", serif',
    "Segoe UI": '"Segoe UI", system-ui, sans-serif',
    "Cascadia Mono": '"Cascadia Mono", "Consolas", monospace',
}

# Load stylesheet
CSS = Path(__file__).with_name("style.css").read_text(encoding="utf-8")
st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)

# CSS variables driven by session state
_doc_font = st.session_state.get("doc_font", "Sitka Text")
_doc_size = st.session_state.get("doc_size", 12)
_zoom = max(0.8, min(1.4, st.session_state.get("zoom", 100) / 100))
_page_w = st.session_state.get("page_w", 720)
_line = st.session_state.get("line", 1.75)
_accent, _accent_dk = _ACCENTS.get(
    st.session_state.get("accent", "Verdigris"), _ACCENTS["Verdigris"])
_page_bg = _PAGE_BGS.get(
    st.session_state.get("page_bg", "Petrol"), _PAGE_BGS["Petrol"])
_doc_stack = _FONT_STACKS.get(_doc_font, _FONT_STACKS["Sitka Text"])
_doc_px = round(float(_doc_size) * 1.3333, 1)
st.markdown(
    f"<style>:root{{--doc:{_doc_stack};--doc-size:{_doc_px}px;"
    f"--font-doc:{_doc_stack};--font-size-base:{_doc_px}px;"
    f"--zoom:{_zoom};--page-w:{_page_w}px;--leading:{_line};"
    f"--leading-normal:{_line};--paper:{_page_bg};--page:{_page_bg};"
    f"--color-paper:{_page_bg};--accent:{_accent};--accent-2:{_accent_dk};"
    f"--color-accent:{_accent};--color-accent-2:{_accent_dk};}}"
    "</style>",
    unsafe_allow_html=True,
)

# Examples
EX_GRAMMAR = ("i recieved your message yesterday and wanted to reply sooner "
              "but i was definately busy . their going to seperate the "
              "reports tommorow .")
EX_STYLE = ("The report was written by me and it is very good. "
            "The data was analyzed by us and the results were very surprising.")
EX_CLEAN = "She walks to the park every morning. The book was interesting."


def _use_example(txt: str) -> None:
    _set_text(txt)
    st.session_state.pop("polished", None)
    st.session_state.pop("pol_source", None)
    st.session_state["dismissed"] = []
    st.session_state["show_insights"] = False


def _clear() -> None:
    _set_text("")
    st.session_state.pop("polished", None)
    st.session_state.pop("pol_source", None)
    st.session_state["dismissed"] = []
    st.session_state["show_insights"] = False
    st.session_state["editor_mode"] = "Write"


def _toggle_insights() -> None:
    st.session_state["show_insights"] = not st.session_state.get(
        "show_insights", False)


def _close_insights() -> None:
    st.session_state["show_insights"] = False


def _request_print() -> None:
    st.session_state["do_print"] = True


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
    """Accept every spelling suggestion at once (Grammarly's 'Fix all')."""
    t = st.session_state.get("pm_text", "")
    hits = analysis.spell_check(
        t, ignore=st.session_state.get("personal_dict", set()))
    if hits:
        _set_text(analysis.apply_spelling_fixes(t, hits))
    st.session_state.pop("polished", None)
    st.session_state.pop("pol_source", None)


def _add_to_dict(word: str) -> None:
    words = set(st.session_state.get("personal_dict", set()))
    words.add(word.lower())
    st.session_state["personal_dict"] = words
    try:
        DICT_PATH.write_text(json.dumps(sorted(words), indent=0), encoding="utf-8")
    except Exception:
        pass


def _clear_dict() -> None:
    st.session_state["personal_dict"] = set()
    try:
        DICT_PATH.unlink()
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
    st.session_state["show_insights"] = False


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
        st.session_state["copied"] = True
    except Exception:
        st.session_state["copied"] = False


def _reset_polish() -> None:
    st.session_state.pop("polished", None)
    st.session_state.pop("pol_source", None)


# Helpers
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


def score_ring(score: int | None, size: int = 80) -> str:
    stroke = 6
    r = (size - stroke) / 2
    c = 2 * math.pi * r
    cx = cy = size / 2
    if score is None:
        pct, color, label = 0.0, "#2a2e34", "—"
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
        f'<text x="{cx}" y="{cy + 7}" text-anchor="middle" '
        f'font-family="Sitka Display,Georgia,serif" font-size="20" '
        f'font-weight="600" fill="var(--ink)">{label}</text>'
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
        out.append(f'<span class="w-mk {sev}" title="{html.escape(tip)}">'
                   f'{html.escape(text[start:end])}</span>')
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


# Warm model
@st.cache_resource(show_spinner="Loading model (first run only)...")
def _warm():
    engine.load_model()


_warm()

# --------------------------------------------------------------- instant layer
sensitivity = st.session_state.get("sensitivity", "Balanced")
formal_tone = st.session_state.get("formal_tone", False)
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
n_corr = n_clar = n_eng = n_del = 0

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

    def push(sid, sev, cat, before, after, expl, accept=None, src="Rule",
             span=None):
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
             "Passive voice distances the reader from the action — prefer "
             "active voice.", span=(s, e))
    for s, e in style.hedging_spans:
        push(f"hg_{s}", "clar", "Hedging", text[s:e], None,
             "Softeners dilute your point — cutting them reads with more "
             "confidence.", span=(s, e))
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
             f"“{word}” is repeated — vary the wording or cut the repeat.",
             span=(s, e))
    for s, e in style.sentence_spans:
        snippet = text[s:e].strip()
        push(f"ls_{s}", "del", "Readability",
             (snippet if len(snippet) <= 90 else snippet[:90] + "…"), None,
             f"{len(snippet.split())} words — split it up to keep the reader "
             f"with you.", span=(s, e))
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
    score = max(0, min(100, 100 - 6 * n_corr - 4 * n_clar
                       - 2 * n_eng - 3 * n_del))

# ================================================================ HEADER
show_insights = bool(st.session_state.get("show_insights", False))
editor_mode = st.session_state.get("editor_mode", "Write")
rewrite_clicked = False
num_beams = 1
max_len = 256

if show_insights:
    st.markdown('<div class="w-insights-on"></div>', unsafe_allow_html=True)

with st.container(key="header"):
    h_brand, h_meta, h_actions = st.columns([1.15, 1.35, 1.75],
                                            gap="small")
    with h_brand:
        st.markdown(
            '<div class="w-brand">'
            '<span class="w-brand-mark">✎</span>'
            '<span class="w-brand-copy">'
            '<span class="w-brand-name">Writing Assistant</span>'
            '<span class="w-brand-sub">Private writing studio</span>'
            '</span></div>',
            unsafe_allow_html=True)
    with h_meta:
        st.markdown(
            '<div class="w-docmeta">'
            '<span class="w-docmeta-k">Draft</span>'
            '<span class="w-docmeta-v">Untitled</span>'
            '<span class="w-docmeta-dot"></span>'
            f'<span class="w-docmeta-m">{datetime.now().strftime("%b %d, %Y")}</span>'
            '</div>',
            unsafe_allow_html=True)
    with h_actions:
        a1, a2, a3 = st.columns([1.15, 1.1, .48], gap="small")
        with a1:
            st.button(
                f"Insights{('  ·  ' + str(len(issues))) if issues else ''}",
                key="toggle_insights", on_click=_toggle_insights,
                width="stretch",
                help="Open or close the insights panel (Ctrl+I)")
        with a2:
            rewrite_clicked = st.button(
                "✦ Rewrite", type="primary", key="rewrite_btn",
                disabled=not bool(text.strip()), width="stretch",
                help="Rewrite the draft with the local model (Ctrl+Enter)")
        with a3:
            with st.popover("⚙", help="Settings (Ctrl+,)"):
                st.markdown('<div class="w-pop-head">Settings</div>',
                            unsafe_allow_html=True)
                st.markdown('<div class="w-pop-label">Proofing</div>',
                            unsafe_allow_html=True)
                st.segmented_control(
                    "Sensitivity", ["Light", "Balanced", "Strict"],
                    default="Balanced", key="sensitivity", width="stretch",
                    help="Light: correctness only. Balanced: + clarity. "
                         "Strict: + engagement.")
                st.toggle("Formal tone", key="formal_tone",
                          help="Also flags contractions like “don't” and “it's”.")
                st.markdown('<div class="w-pop-label">Rewrite</div>',
                            unsafe_allow_html=True)
                q1, q2 = st.columns(2, gap="small")
                with q1:
                    num_beams = st.slider(
                        "Quality", 1, 4, 1,
                        help="1 = fastest (greedy). 4 = slightly better, "
                             "about 4x slower.")
                with q2:
                    max_len = st.slider(
                        "Max length", 32, 512, 256,
                        help="Longest sentence, in tokens, the model may emit.")
                st.markdown('<div class="w-pop-label">Appearance</div>',
                            unsafe_allow_html=True)
                _accent_opts = list(_ACCENTS)
                _bg_opts = list(_PAGE_BGS)
                _font_opts = list(_FONT_STACKS)
                _size_opts = [10, 11, 12, 14, 16, 18, 20]
                _width_opts = [560, 660, 720, 820]
                _line_opts = [1.4, 1.6, 1.75, 1.9]
                ca, cb = st.columns(2, gap="small")
                with ca:
                    st.selectbox(
                        "Accent", _accent_opts,
                        index=_index(_accent_opts,
                                     st.session_state.get("accent", "Verdigris")),
                        key="accent_widget", on_change=_sync_widget,
                        args=("accent_widget", "accent"))
                with cb:
                    st.selectbox(
                        "Page", _bg_opts,
                        index=_index(_bg_opts,
                                     st.session_state.get("page_bg", "Petrol")),
                        key="page_bg_widget", on_change=_sync_widget,
                        args=("page_bg_widget", "page_bg"))
                ct, cs = st.columns(2, gap="small")
                with ct:
                    st.selectbox(
                        "Typeface", _font_opts,
                        index=_index(_font_opts,
                                     st.session_state.get("doc_font",
                                                          "Sitka Text")),
                        key="doc_font_widget", on_change=_sync_widget,
                        args=("doc_font_widget", "doc_font"))
                with cs:
                    st.selectbox(
                        "Text size", _size_opts,
                        index=_index(_size_opts,
                                     st.session_state.get("doc_size", 12)),
                        key="doc_size_widget", on_change=_sync_widget,
                        args=("doc_size_widget", "doc_size"))
                cw, cl = st.columns(2, gap="small")
                with cw:
                    st.selectbox(
                        "Page width", _width_opts,
                        index=_index(_width_opts,
                                     st.session_state.get("page_w", 720)),
                        format_func=lambda v: {560: "Narrow", 660: "Medium",
                                               720: "Wide",
                                               820: "Extra wide"}[v],
                        key="page_w_widget", on_change=_sync_widget,
                        args=("page_w_widget", "page_w"))
                with cl:
                    st.selectbox(
                        "Line spacing", _line_opts,
                        index=_index(_line_opts,
                                     st.session_state.get("line", 1.75)),
                        format_func=lambda v: {1.4: "Tight", 1.6: "Normal",
                                               1.75: "Relaxed",
                                               1.9: "Airy"}[v],
                        key="line_widget", on_change=_sync_widget,
                        args=("line_widget", "line"))
                st.slider(
                    "Zoom", 80, 140,
                    value=int(st.session_state.get("zoom", 100)),
                    key="zoom_widget", on_change=_sync_widget,
                    args=("zoom_widget", "zoom"), step=5, format="%d%%")
                st.markdown('<div class="w-pop-label">Document</div>',
                            unsafe_allow_html=True)
                b1, b2, b3 = st.columns(3, gap="small")
                with b1:
                    st.button("New draft", key="new_draft", on_click=_clear,
                              width="stretch")
                with b2:
                    st.button("Print", key="print_doc",
                              on_click=_request_print, width="stretch",
                              help="Print or save the draft as a PDF.")
                with b3:
                    st.button("Clear words", key="clr_dict",
                              on_click=_clear_dict, width="stretch",
                              help="Remove words added to your personal dictionary.")
                st.caption("Offline. Nothing leaves this device.")

# ---------------------------------------------------------------- rewrite
polished = (st.session_state.get("polished")
            if st.session_state.get("pol_source") == text else None)
t_model = st.session_state.get("pol_time", 0.0) if polished else 0.0

if rewrite_clicked and text.strip():
    with st.spinner(
            f"Rewriting — {style.sentences} sentence(s), "
            f"~{max(1, style.sentences)}s..."):
        try:
            t1 = time.time()
            new_text = engine.polish(text, max_length=max_len,
                                     num_beams=num_beams)
            t_model = time.time() - t1
        except Exception as exc:
            st.error(f"The model could not finish: {exc}")
        else:
            st.session_state["polished"] = new_text
            st.session_state["pol_source"] = text
            st.session_state["pol_time"] = t_model
            st.session_state["show_insights"] = True
            st.rerun()

polished = (st.session_state.get("polished")
            if st.session_state.get("pol_source") == text else None)
t_model = st.session_state.get("pol_time", 0.0) if polished else 0.0

# ---------------------------------------------------------------- workspace
if show_insights:
    page_col, panel_col = st.columns([1, .56], gap="large")
else:
    page_col, panel_col = st.container(), None

with page_col:
    with st.container(key="page"):
        _has_text = bool(text.strip())
        m1, m2 = st.columns([2.4, 1.15], gap="small")
        with m1:
            st.markdown(
                '<div class="w-page-meta">'
                '<span class="w-page-meta-k">Draft 01</span>'
                '<span class="w-page-meta-sep"></span>'
                f'<span class="w-page-meta-date">'
                f'{datetime.now().strftime("%B %d, %Y")}</span></div>',
                unsafe_allow_html=True)
        with m2:
            if _has_text:
                st.segmented_control(
                    "View", ["Write", "Proof"], key="editor_mode",
                    label_visibility="collapsed", width="stretch")

        if not _has_text:
            st.markdown('<div class="w-empty-mode"></div>',
                        unsafe_allow_html=True)
            st.markdown(
                '<div class="w-empty">'
                '<div class="w-empty-kicker">A quiet place to begin</div>'
                '<div class="w-empty-title">Write with confidence.</div>'
                '<div class="w-empty-sub">Paste a draft or start typing. '
                'Instant suggestions appear as you write, and the local '
                'model is here when you want a second pass.</div>'
                '</div>',
                unsafe_allow_html=True)
            with st.container(key="empty_actions"):
                e1, e2, e3 = st.columns(3, gap="small")
                with e1:
                    st.button("Rough draft", key="hero_rough",
                              on_click=_use_example, args=(EX_GRAMMAR,),
                              width="stretch")
                with e2:
                    st.button("Formal note", key="hero_formal",
                              on_click=_use_example, args=(EX_STYLE,),
                              width="stretch")
                with e3:
                    st.button("Clean copy", key="hero_clean",
                              on_click=_use_example, args=(EX_CLEAN,),
                              width="stretch")
            st.markdown(
                '<div class="w-empty-hint">or begin with a sentence below'
                '</div>', unsafe_allow_html=True)

        if _has_text and editor_mode == "Proof":
            spans: list[tuple[int, int, str, str]] = []

            def mark(s, e, sev, tip):
                if _BUCKET_LEVEL[sev] <= level:
                    spans.append((s, e, sev, tip))

            for h in spelling:
                tip = (f"Spelling: {h.word}"
                       + (f" → {h.suggestions[0]}" if h.suggestions else ""))
                mark(h.start, h.end, "corr", tip)
            for s, e in style.passive_spans:
                mark(s, e, "clar", "Passive voice — consider active voice")
            for s, e in style.hedging_spans:
                mark(s, e, "clar", "Hedging word — cutting it adds confidence")
            for s, e in style.filler_spans:
                mark(s, e, "clar", "Intensifier — the sentence is stronger without it")
            for s, e, repl in style.wordy_spans:
                mark(s, e, "clar", f"Wordy phrase — try “{repl}”")
            for s, e in style.cliche_spans:
                mark(s, e, "eng", "Cliché — try a fresher phrase")
            for s, e in style.adverb_spans:
                mark(s, e, "eng", "Adverb — a stronger verb may do the job")
            for s, e, repl in style.weak_spans:
                mark(s, e, "eng", f"Vague word — try “{repl}”")
            for s, e, word in style.repetition_spans:
                mark(s, e, "eng", f"“{word}” is repeated")
            for s, e in style.sentence_spans:
                mark(s, e, "del", "Long sentence — consider splitting it")
            for s, e in style.contraction_spans:
                mark(s, e, "del", "Contraction — spell it out for a formal tone")

            legend = (
                '<div class="w-legend">'
                '<span class="w-legend-item"><i class="corr"></i>Correctness</span>'
                '<span class="w-legend-item"><i class="clar"></i>Clarity</span>'
                '<span class="w-legend-item"><i class="eng"></i>Engagement</span>'
                '<span class="w-legend-item"><i class="del"></i>Delivery</span>'
                '</div>')
            st.markdown(legend, unsafe_allow_html=True)
            st.markdown(
                f'<div class="w-proof-page">{markup_html(text, spans)}</div>',
                unsafe_allow_html=True)
            st.markdown(
                '<div class="w-proof-note">Read-only proof · hover a marked '
                'word for the reason · switch to <b>Write</b> to edit</div>',
                unsafe_allow_html=True)
        else:
            _text_kwargs = ({} if "pm_text" in st.session_state
                            else {"value": text})
            st.text_area(
                "Document",
                key="pm_text",
                on_change=_sync_draft,
                height=480 if _has_text else 210,
                label_visibility="collapsed",
                placeholder="Begin with a sentence." if not _has_text else "",
                **_text_kwargs,
            )
            if _has_text:
                st.markdown(
                    '<div class="w-page-hint"><span class="w-page-hint-dot">'
                    '</span>Proofing marks are available in <b>Proof</b> view.'
                    '</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------- inspector
if panel_col is not None:
    with panel_col:
        with st.container(key="inspector"):
            p1, p2 = st.columns([3.6, .55], gap="small")
            with p1:
                st.markdown(
                    '<div class="w-panel-kicker">Editorial pass</div>'
                    '<div class="w-panel-title">Insights</div>'
                    f'<div class="w-panel-sub">{len(issues)} suggestion'
                    f'{"s" if len(issues) != 1 else ""} · {sensitivity} '
                    f'sensitivity</div>',
                    unsafe_allow_html=True)
            with p2:
                st.button("✕", key="close_insights", on_click=_close_insights,
                          help="Close insights (Esc)")

            if _has_text:
                if score is not None:
                    if score >= 95:
                        vtxt, vcol = "Reads clean.", "var(--accent)"
                    elif score >= 80:
                        vtxt, vcol = "A few things to tighten.", "var(--clar)"
                    else:
                        vtxt, vcol = "Worth a pass.", "var(--corr)"
                else:
                    vtxt, vcol = "Ready when you are.", "var(--accent)"
                st.markdown(
                    f'<div class="w-score-card">'
                    f'<div class="w-ring">{score_ring(score, 104)}</div>'
                    f'<div class="w-score-copy">'
                    f'<div class="w-score-label">Writing score</div>'
                    f'<div class="w-verdict">'
                    f'<span class="w-verdict-dot" style="background:{vcol}">'
                    f'</span>{vtxt}</div>'
                    f'<div class="w-score-sub">out of 100</div>'
                    f'</div></div>',
                    unsafe_allow_html=True)

                if tone:
                    st.markdown(
                        f'<div class="w-tone-row">'
                        f'<span class="w-tone">{html.escape(tone.tone)}</span>'
                        f'<span class="w-tone-meta">Formality '
                        f'{tone.formality}/100</span></div>',
                        unsafe_allow_html=True)

                if sub_scores:
                    colors = {"Correctness": "var(--corr)",
                              "Clarity": "var(--clar)",
                              "Engagement": "var(--eng)",
                              "Delivery": "var(--del)"}
                    rows = "".join(
                        f'<div class="w-bar"><span>{k}</span>'
                        f'<div class="w-bar-track"><div class="w-bar-fill" '
                        f'style="width:{v}%;background:{colors[k]}"></div></div>'
                        f'<span class="w-bar-value">{v}</span></div>'
                        for k, v in sub_scores.items())
                    st.markdown(
                        f'<div class="w-section-label">Breakdown</div>'
                        f'<div class="w-bars">{rows}</div>',
                        unsafe_allow_html=True)

                n_spell = sum(1 for i in issues if i["cat"] == "Spelling")
                if n_spell:
                    st.button(f"Fix all spelling ({n_spell})", key="fixall",
                              on_click=_fix_all, width="stretch")
                    st.markdown('<div class="w-gap-2"></div>',
                                unsafe_allow_html=True)

                if issues:
                    choice = st.selectbox(
                        "Filter", ["All", "Correctness", "Clarity",
                                   "Engagement", "Delivery"],
                        key="cat_filter", label_visibility="collapsed")
                    want = {"All": None, "Correctness": "corr",
                            "Clarity": "clar", "Engagement": "eng",
                            "Delivery": "del"}[choice]
                    shown = [i for i in issues
                             if want is None or i["sev"] == want]
                    st.markdown('<div class="w-section-label">Suggestions</div>',
                                unsafe_allow_html=True)
                    if not shown:
                        st.markdown(
                            '<div class="w-clean">'
                            '<span class="w-clean-check">✓</span>'
                            '<div><div class="w-clean-title">Nothing here.'
                            '</div><div class="w-clean-sub">No suggestions in '
                            'this category at this sensitivity.</div></div>'
                            '</div>', unsafe_allow_html=True)
                    else:
                        for it in shown:
                            sev = it["sev"]
                            sid = it["sid"]
                            phrase = (f'<span class="w-sug-old">'
                                      f'{html.escape(it["before"])}</span>')
                            if it["after"]:
                                phrase += ('<span class="w-sug-arrow">→</span>'
                                           f'<span class="w-sug-new">'
                                           f'{html.escape(it["after"])}</span>')
                            with st.container(key=f"sug_{sid}"):
                                st.markdown(
                                    f'<div class="w-sug-marker {sev}"></div>'
                                    f'<div class="w-sug-top">'
                                    f'<span class="w-sug-cat {sev}">'
                                    f'{html.escape(it["cat"])}</span>'
                                    f'<span class="w-sug-src">{it["src"]}'
                                    f'</span></div>'
                                    f'<div class="w-sug-phrase">{phrase}</div>'
                                    f'<div class="w-sug-expl">'
                                    f'{html.escape(it["expl"])}</div>',
                                    unsafe_allow_html=True)
                                if it["accept"]:
                                    s, e, rep = it["accept"]
                                    cols = st.columns(
                                        3 if it["cat"] == "Spelling" else 2,
                                        gap="small")
                                    with cols[0]:
                                        st.button(
                                            "Accept", key=f"acc_{sid}",
                                            type="primary", on_click=_apply_span,
                                            args=(s, e, rep), width="stretch")
                                    if it["cat"] == "Spelling":
                                        with cols[1]:
                                            st.button(
                                                "Add word", key=f"add_{sid}",
                                                on_click=_add_to_dict,
                                                args=(it["before"],),
                                                width="stretch")
                                        with cols[2]:
                                            st.button(
                                                "Ignore", key=f"dis_{sid}",
                                                on_click=_dismiss,
                                                args=(sid,), width="stretch")
                                    else:
                                        with cols[1]:
                                            st.button(
                                                "Ignore", key=f"dis_{sid}",
                                                on_click=_dismiss,
                                                args=(sid,), width="stretch")
                                else:
                                    st.button("Ignore", key=f"dis_{sid}",
                                              on_click=_dismiss, args=(sid,),
                                              width="stretch")
                else:
                    st.markdown(
                        '<div class="w-clean w-clean-lg">'
                        '<span class="w-clean-check">✓</span>'
                        '<div><div class="w-clean-title">Reads clean.</div>'
                        '<div class="w-clean-sub">No suggestions at this '
                        'sensitivity. Try Rewrite for a second pass.</div>'
                        '</div></div>', unsafe_allow_html=True)
            else:
                st.markdown(
                    '<div class="w-guide">'
                    '<div class="w-guide-title">Ready when you are.</div>'
                    '<div class="w-guide-sub">Start typing or try a sample. '
                    'Your draft stays private on this device.</div>'
                    '<ol class="w-guide-steps">'
                    '<li class="w-guide-step"><span class="w-guide-number">01'
                    '</span><div><div class="w-guide-step-title">Checks run '
                    'as you type</div><p class="w-guide-step-desc">Spelling, '
                    'clarity, engagement and delivery, in milliseconds.</p>'
                    '</div></li>'
                    '<li class="w-guide-step"><span class="w-guide-number">02'
                    '</span><div><div class="w-guide-step-title">Rewrite when '
                    'you want a second pass</div><p class="w-guide-step-desc">'
                    'The local model reworks the whole draft on request.</p>'
                    '</div></li>'
                    '<li class="w-guide-step"><span class="w-guide-number">03'
                    '</span><div><div class="w-guide-step-title">Compare and '
                    'keep what works</div><p class="w-guide-step-desc">Track '
                    'changes shows every edit; accept the rewrite when ready.'
                    '</p></div></li></ol>'
                    '<div class="w-guide-note"><span class="w-guide-note-dot">'
                    '</span>Local only — nothing leaves this machine.</div>'
                    '</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------- revisions
if polished and text.strip():
    # Re-enter the page column so revisions sit under the draft, beside the
    # inspector instead of being pushed below the full-height panel.
    with page_col, st.container(key="revisions"):
        r1, r2 = st.columns([2.5, 1.4], gap="small")
        with r1:
            st.markdown(
                '<div class="w-revisions-kicker">Second pass</div>'
                '<div class="w-revisions-title">Revisions</div>'
                '<div class="w-revisions-sub">Track changes from the local '
                'model. Keep the whole rewrite, or go back to your draft.'
                '</div>', unsafe_allow_html=True)
        with r2:
            b1, b2, b3 = st.columns(3, gap="small")
            with b1:
                st.button("Accept", type="primary", key="acc_rw",
                          on_click=_accept_rewrite, width="stretch",
                          help="Replace the draft with the rewritten text.")
            with b2:
                st.button("Copy", key="cpy", on_click=_copy_rewrite,
                          width="stretch")
            with b3:
                st.button("Discard", key="discard_rw", on_click=_reset_polish,
                          width="stretch")

        changed = polished.strip() != text.strip()
        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown('<div class="w-revision-label">Original</div>'
                        f'<div class="w-revision-pane">'
                        f'<div class="w-revision-text">'
                        f'{html.escape(text.strip())}</div></div>',
                        unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="w-revision-label">Rewritten</div>'
                        f'<div class="w-revision-pane">'
                        f'<div class="w-revision-text">'
                        f'{html.escape(polished.strip())}</div></div>',
                        unsafe_allow_html=True)
        if changed:
            st.markdown(
                '<div class="w-revision-label w-revision-label-inline">'
                'Inline changes · strike = removed · underline = added</div>'
                f'<div class="w-diff">'
                f'{diff_html(text.strip(), polished.strip())}</div>',
                unsafe_allow_html=True)

        if st.session_state.get("copied") is True:
            st.success("Rewritten text copied to your clipboard.")

        st.markdown(
            f'<div class="w-timing">Rules {t_instant * 1000:.0f} ms · model '
            f'{t_model:.2f} s · {style.sentences} sentence(s) · '
            f'{num_beams} beam(s)</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------- footer
_words = style.words if style else 0
_rt = style.reading_time_min if style else 0.0
_rt_lbl = "<1 min read" if _rt < 1 else f"{_rt:.0f} min read"
_n = len(issues)
with st.container(key="footer"):
    st.markdown(
        '<div class="w-footer-inner">'
        f'<span class="w-footer-stat"><b>{_words}</b> words</span>'
        '<span class="w-footer-sep"></span>'
        f'<span class="w-footer-stat">{_rt_lbl}</span>'
        '<span class="w-footer-sep"></span>'
        f'<span class="w-footer-stat">{_n} suggestion'
        f'{"s" if _n != 1 else ""}</span>'
        '<span class="w-footer-sep"></span>'
        '<span class="w-footer-stat w-footer-local"><i></i>Local only</span>'
        '<span class="w-footer-spacer"></span>'
        '<span class="w-footer-hint">Ctrl+I insights · Ctrl+Enter rewrite</span>'
        '</div>', unsafe_allow_html=True)

# Print-only copy: a textarea clips long drafts when printed.
st.markdown(f'<div class="w-printonly">{html.escape(text)}</div>',
            unsafe_allow_html=True)

# ---------------------------------------------------------------- shortcuts
st.html(
    """<script>(function(){
    const p = window.parent.document;
    function hit(sel){
      const b = p.querySelector(sel);
      if (b && !b.disabled) b.click();
    }
    function onKey(e){
      const mod = e.metaKey || e.ctrlKey;
      if (mod && e.key === ',') { e.preventDefault();
        hit('.st-key-header [data-testid="stPopover"] button'); }
      if (mod && e.key === 'Enter') { e.preventDefault();
        hit('.st-key-rewrite_btn button'); }
      if (mod && e.key.toLowerCase() === 'i') { e.preventDefault();
        hit('.st-key-toggle_insights button'); }
      if (mod && e.shiftKey && e.key.toLowerCase() === 'x') { e.preventDefault();
        hit('.st-key-new_draft button'); }
      if (e.key === 'Escape') {
        hit('.st-key-close_insights button'); }
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
    // Streamlit may recreate the component iframe on a rerun. Replacing the
    // handler keeps exactly one listener alive in the parent document.
    setInterval(attach, 1200);
    })();</script>""")

if st.session_state.pop("do_print", False):
    st.html("<script>window.parent.print();</script>")
