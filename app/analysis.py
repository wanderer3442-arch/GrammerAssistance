"""Rule-based writing analysis layer -- instant, no model needed.

Complements the (slow) T5 model with feedback that appears immediately:
  - spell_check()   : fast typo detection + suggestions (pyspellchecker)
  - explain_diff()  : categorize what changed between original and polished text
  - style_report()  : readability + heuristic style flags (passive, long sentences,
                      filler, wordiness, cliches, adverbs, repetition, weak words)
  - tone_report()   : rough tone + formality read (confident / tentative / formal...)
These run in milliseconds, so the UI can show them while the T5 polish is pending.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from spellchecker import SpellChecker

_spell = SpellChecker()


# --------------------------------------------------------------------------- spell
@dataclass
class SpellHit:
    word: str
    suggestions: list[str]
    start: int          # char offset in the original string
    end: int


def spell_check(text: str, max_suggestions: int = 3,
                ignore: set[str] | None = None) -> list[SpellHit]:
    """Detect likely misspellings. Skips proper nouns / capitalized mid-sentence
    words and anything with digits, to cut false positives. `ignore` is a personal
    dictionary of lowercase words the writer has accepted."""
    ignore = ignore or set()
    hits: list[SpellHit] = []
    for m in re.finditer(r"[A-Za-z][A-Za-z']*", text):
        w = m.group(0)
        if any(ch.isdigit() for ch in w):
            continue
        # Skip ALL-CAPS and title-case words that are not sentence-initial
        # (likely names). Newlines count as sentence boundaries too; otherwise
        # a heading such as "Project Notes" after a blank line would be missed.
        if w.isupper():
            continue
        prefix = text[:m.start()].rstrip()
        sentence_initial = not prefix or prefix[-1] in ".!?\n"
        if w[0].isupper() and not sentence_initial:
            continue
        lower = w.lower()
        if lower in ignore or lower in _spell:
            continue
        # SpellChecker returns a set, so iterating it directly makes the first
        # suggestion nondeterministic. Rank by corpus frequency, then use the
        # spelling itself as a stable tie-breaker. This prevents rare words
        # such as "tommyrot" from outranking the common "tomorrow".
        suggestions = sorted(
            _spell.candidates(lower),
            key=lambda candidate: (-_spell.word_frequency[candidate], candidate),
        )
        # keep only suggestions that preserve the original capitalization shape
        sugg = []
        for s in list(suggestions)[: max_suggestions * 2]:
            if w[0].isupper():
                s = s[0].upper() + s[1:]
            sugg.append(s)
            if len(sugg) >= max_suggestions:
                break
        hits.append(SpellHit(word=w, suggestions=sugg, start=m.start(), end=m.end()))
    return hits


def apply_spelling_fixes(text: str, hits: list[SpellHit]) -> str:
    """Apply every spelling suggestion at once, right-to-left so offsets hold."""
    out = text
    for h in sorted(hits, key=lambda x: x.start, reverse=True):
        if not h.suggestions:
            continue
        out = out[:h.start] + h.suggestions[0] + out[h.end:]
    return out


# ------------------------------------------------------------------- explain diff
# Heuristics to guess *what kind* of fix a diff represents, so we can label it
# ("grammar", "agreement", "article", "spelling", "word choice") like Grammarly.
_BE_WORDS = {"is", "are", "was", "were", "be", "been", "being", "am"}
_ARTICLES = {"a", "an", "the"}
_AUX = {"do", "does", "did", "have", "has", "had", "can", "could", "will",
        "would", "shall", "should", "may", "might", "must"}


@dataclass
class Change:
    original: str
    polished: str
    category: str
    explanation: str


def _classify(orig_tok: list[str], pol_tok: list[str]) -> tuple[str, str]:
    """Best-effort label for a replaced span given the changed token lists."""
    o = [t.lower().strip(".,!?;:'\"()") for t in orig_tok]
    p = [t.lower().strip(".,!?;:'\"()") for t in pol_tok]
    oset, pset = set(o), set(p)

    # obvious misspelling -> spelling (use the spell-checker as source of truth)
    if (len(o) == 1 and len(p) == 1
            and o[0] not in _spell and p[0] in _spell
            and _close(o[0], p[0])):
        return ("Spelling", "Corrected a spelling mistake.")

    # article swap / insert / delete
    if (oset & _ARTICLES) != (pset & _ARTICLES) or (
        bool(pset & _ARTICLES) and not (oset & _ARTICLES)
    ):
        if "an" in pset and "a" in oset:
            return ("Article", "Use 'an' before a vowel sound (e.g. 'an apple').")
        if "a" in pset and "an" in oset:
            return ("Article", "Use 'a' before a consonant sound (e.g. 'a user').")
        if bool(pset & _ARTICLES) and not (oset & _ARTICLES):
            return ("Article", "Added a missing article (a/an/the).")
        return ("Article", "Article usage corrected (a/an/the).")

    # be-verb / auxiliary change
    if (oset & _BE_WORDS) != (pset & _BE_WORDS):
        return ("Grammar", "Corrected a linking/auxiliary verb (is/are/was/were).")
    if (oset & _AUX) != (pset & _AUX):
        return ("Grammar", "Corrected an auxiliary/modal verb.")

    # subject-verb agreement: same lemma, different -s/-es ending
    def _lemma(t):
        return re.sub(r"(s|es|ed|d)$", "", t)
    if len(o) == 1 and len(p) == 1 and _lemma(o[0]) == _lemma(p[0]) and o[0] != p[0]:
        return ("Agreement", "Fixed subject-verb agreement (singular/plural form).")

    # pure spelling-ish (small edit distance, similar)
    if len(o) == 1 and len(p) == 1 and _close(o[0], p[0]):
        return ("Spelling", "Corrected a spelling mistake.")

    if len(p) > len(o) + 1:
        return ("Word choice", "Added a word for clarity or grammatical completeness.")
    if len(o) > len(p) + 1:
        return ("Conciseness", "Removed an unnecessary word.")
    return ("Word choice", "Improved word choice / phrasing.")


def _close(a: str, b: str) -> bool:
    """Cheap edit-distance<=2 test for short words."""
    if abs(len(a) - len(b)) > 2:
        return False
    if a == b:
        return True
    # trim common affixes then compare
    for suf in ("ing", "ed", "es", "s", "ly", "er", "est"):
        if a.endswith(suf) and b.endswith(suf):
            a2, b2 = a[: -len(suf)], b[: -len(suf)]
            return a2 == b2 or abs(len(a2) - len(b2)) <= 1
    return False


def explain_diff(original: str, polished: str) -> list[Change]:
    """Word-level diff -> categorized, human-readable change explanations."""
    import difflib

    o = original.split()
    p = polished.split()
    sm = difflib.SequenceMatcher(None, o, p)
    changes: list[Change] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        orig_span = o[i1:i2]
        pol_span = p[j1:j2]
        if not orig_span and not pol_span:
            continue
        cat, expl = _classify(orig_span, pol_span)
        changes.append(Change(
            original=" ".join(orig_span),
            polished=" ".join(pol_span),
            category=cat,
            explanation=expl,
        ))
    return changes


# ------------------------------------------------------------------ style report
@dataclass
class StyleReport:
    words: int
    sentences: int
    paragraphs: int
    avg_sentence_len: float
    reading_ease: float            # Flesch reading ease (higher = easier)
    grade_level: float             # Flesch-Kincaid grade
    reading_time_min: float
    flesch_label: str
    longest_sentence_words: int
    passive_hits: list[str] = field(default_factory=list)
    long_sentences: list[str] = field(default_factory=list)
    hedging_hits: list[str] = field(default_factory=list)
    filler_hits: list[str] = field(default_factory=list)
    wordy_hits: list[str] = field(default_factory=list)
    cliche_hits: list[str] = field(default_factory=list)
    adverb_hits: list[str] = field(default_factory=list)
    contraction_hits: list[str] = field(default_factory=list)
    weak_hits: list[tuple[str, list[str]]] = field(default_factory=list)
    repeated_words: list[tuple[str, int]] = field(default_factory=list)
    # char-offset spans so the UI can underline flagged text
    passive_spans: list[tuple[int, int]] = field(default_factory=list)
    hedging_spans: list[tuple[int, int]] = field(default_factory=list)
    filler_spans: list[tuple[int, int]] = field(default_factory=list)
    wordy_spans: list[tuple[int, int, str]] = field(default_factory=list)
    cliche_spans: list[tuple[int, int]] = field(default_factory=list)
    adverb_spans: list[tuple[int, int]] = field(default_factory=list)
    contraction_spans: list[tuple[int, int]] = field(default_factory=list)
    weak_spans: list[tuple[int, int, str]] = field(default_factory=list)
    repetition_spans: list[tuple[int, int, str]] = field(default_factory=list)
    sentence_spans: list[tuple[int, int]] = field(default_factory=list)


_PASSIVE_RE = re.compile(
    r"\b(?:\w+\s+)?(?:is|are|was|were|be|been|being)\s+"
    r"(?:\w+ly\s+)?(?:\w+ed|taken|given|made|seen|known|found|shown|written|"
    r"built|held|kept|lost|sent|done|gone|broken|chosen|spoken|eaten)\b",
    re.I)

# uncertainty softeners (Grammarly "hedging")
_HEDGE_WORDS = {"maybe", "probably", "possibly", "perhaps", "seemingly",
                "apparently", "somewhat", "rather", "arguably", "presumably"}
# intensifiers / verbal filler (Hemingway "adverbs & qualifiers")
_FILLER_WORDS = {"very", "really", "quite", "just", "actually", "basically",
                 "literally", "simply", "definitely", "totally", "absolutely",
                 "extremely", "incredibly", "truly", "honestly", "seriously",
                 "certainly", "clearly", "obviously", "super"}
# wordy phrases -> tighter replacement (ProWritingAid "sticky sentences")
_WORDY: list[tuple[str, str]] = sorted([
    ("in order to", "to"),
    ("due to the fact that", "because"),
    ("at this point in time", "now"),
    ("at the present time", "now"),
    ("in the event that", "if"),
    ("in spite of the fact that", "although"),
    ("for the purpose of", "to"),
    ("with regard to", "about"),
    ("in the near future", "soon"),
    ("in the process of", "currently"),
    ("it is important to note that", "note that"),
    ("the fact of the matter is", "in fact"),
    ("the fact that", "that"),
    ("has the ability to", "can"),
    ("have the ability to", "can"),
    ("is able to", "can"),
    ("are able to", "can"),
    ("make a decision", "decide"),
    ("on a daily basis", "daily"),
    ("each and every", "every"),
    ("a large number of", "many"),
    ("a majority of", "most"),
    ("a number of", "several"),
    ("a lot of", "many"),
    ("in my personal opinion", "I think"),
], key=lambda p: len(p[0]), reverse=True)
# cliches (ProWritingAid "cliches & redundancies")
_CLICHES: list[str] = sorted([
    "at the end of the day", "think outside the box", "low-hanging fruit",
    "in this day and age", "needless to say", "last but not least",
    "when all is said and done", "in the nick of time",
    "better late than never", "back to the drawing board", "bite the bullet",
    "piece of cake", "tip of the iceberg", "avoid like the plague",
    "every cloud has a silver lining", "a blessing in disguise",
    "the calm before the storm", "it goes without saying",
], key=len, reverse=True)
# weak / vague words -> stronger alternatives (Word "vocabulary enhancement")
_WEAK: list[tuple[str, list[str]]] = sorted([
    ("a lot", ["much", "considerably"]),
    ("good", ["strong", "effective", "solid"]),
    ("bad", ["poor", "harmful", "flawed"]),
    ("big", ["large", "substantial", "major"]),
    ("small", ["minor", "slight", "compact"]),
    ("things", ["factors", "aspects", "elements"]),
    ("thing", ["factor", "aspect", "element"]),
    ("stuff", ["material", "content", "items"]),
    ("nice", ["pleasant", "thoughtful", "polished"]),
    ("interesting", ["compelling", "noteworthy", "engaging"]),
    ("important", ["critical", "essential", "significant"]),
    ("help", ["assist", "support", "enable"]),
    ("show", ["demonstrate", "illustrate", "reveal"]),
    ("start", ["begin", "launch", "initiate"]),
    ("buy", ["purchase", "acquire"]),
    ("ask", ["request", "inquire"]),
    ("tell", ["inform", "explain"]),
    ("need", ["require", "depend on"]),
    ("want", ["aim", "seek"]),
    ("get", ["obtain", "receive", "gain"]),
    ("make", ["create", "produce", "build"]),
    ("use", ["apply", "employ", "leverage"]),
], key=lambda p: len(p[0]), reverse=True)

_ADVERB_EXCLUDE = {
    "only", "family", "early", "reply", "apply", "supply", "rely", "holy",
    "ugly", "silly", "friendly", "likely", "daily", "weekly", "monthly",
    "yearly", "ally", "italy", "assembly", "anomaly", "monopoly", "multiply",
    "imply", "comply", "fly", "ply", "belly", "jelly", "rally", "tally",
    "bully", "folly", "chilly", "lily", "july", "jolly", "melancholy",
    "lonely", "lovely", "costly", "deadly", "elderly", "orderly", "timely",
    "unlikely", "namely", "barely", "hardly", "scarcely", "rarely", "fully",
    "apply", "supply", "reply", "multiply", "assembly", "italy", "july",
}

_CONTRACTION_RE = re.compile(
    r"\b(?:[A-Za-z]+n't"
    r"|(?:it|that|there|here|what|who|he|she|let)'(?:s|re|ve|ll|d|m)"
    r"|(?:I|you|we|they)'(?:re|ve|ll|d|m))\b", re.I)

_STOPWORDS = {
    "the", "and", "that", "this", "with", "from", "have", "has", "had",
    "were", "was", "are", "but", "not", "you", "your", "they", "their",
    "them", "there", "then", "than", "when", "what", "which", "will",
    "would", "could", "should", "about", "into", "over", "also", "just",
    "very", "more", "most", "some", "such", "only", "other", "these",
    "those", "been", "being", "because", "while", "where", "after",
    "before", "each", "every", "both", "does", "doing", "done", "make",
    "made", "many", "much", "here", "how", "its", "it's", "for", "our",
}


def _flesch_label(ease: float) -> str:
    if ease >= 80:
        return "Very easy"
    if ease >= 70:
        return "Easy"
    if ease >= 60:
        return "Plain English"
    if ease >= 50:
        return "Fairly hard"
    if ease >= 30:
        return "Difficult"
    return "Very difficult"


def _phrase_spans(text: str, phrase: str) -> list[tuple[int, int]]:
    pat = re.compile(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", re.I)
    return [(m.start(), m.end()) for m in pat.finditer(text)]


def style_report(text: str, formal: bool = False) -> StyleReport:
    try:
        import textstat
        ease = float(textstat.flesch_reading_ease(text))
        grade = float(textstat.flesch_kincaid_grade(text))
        n_sent = int(textstat.sentence_count(text))
        n_words = int(textstat.lexicon_count(text))
    except Exception:
        words = text.split()
        n_words = len(words)
        n_sent = len(re.findall(r"[.!?]+", text))
        if n_words and not n_sent:
            n_sent = 1
        ease, grade = 0.0, 0.0
    if n_words == 0:
        n_sent = 0

    paragraphs = len([p for p in re.split(r"\n\s*\n", text) if p.strip()]) or 1

    passive_matches = list(_PASSIVE_RE.finditer(text))
    passive = [m.group(0) for m in passive_matches][:5]
    passive_spans = [(m.start(), m.end()) for m in passive_matches]

    # Sentences, with offsets so the UI can underline the whole sentence.
    sentence_matches = list(re.finditer(r"[^.!?]+[.!?]+|[^.!?]+$", text))
    long_matches = [m for m in sentence_matches if len(m.group(0).split()) > 28]
    long_sents = [m.group(0).strip() for m in long_matches][:3]
    sentence_spans = [(m.start(), m.end()) for m in long_matches]
    longest = max((len(m.group(0).split()) for m in sentence_matches), default=0)

    # Uncertainty softeners.
    hedge_matches = [m for m in re.finditer(r"\b[\w']+\b", text)
                     if m.group(0).lower() in _HEDGE_WORDS]
    hedge_matches += list(re.finditer(r"\b(?:kind|sort)\s+of\b", text, re.I))
    hedging = sorted({m.group(0).lower() for m in hedge_matches})[:6]
    hedging_spans = [(m.start(), m.end()) for m in hedge_matches]

    # Intensifiers / filler.
    filler_matches = [m for m in re.finditer(r"\b[\w']+\b", text)
                      if m.group(0).lower() in _FILLER_WORDS]
    filler = sorted({m.group(0).lower() for m in filler_matches})[:8]
    filler_spans = [(m.start(), m.end()) for m in filler_matches][:20]

    # Wordy phrases -> replacement.
    wordy_spans: list[tuple[int, int, str]] = []
    wordy_hits: list[str] = []
    for phrase, repl in _WORDY:
        for s, e in _phrase_spans(text, phrase):
            wordy_spans.append((s, e, repl))
            wordy_hits.append(phrase)
    wordy_spans.sort(key=lambda t: t[0])
    wordy_spans = wordy_spans[:12]
    wordy_hits = wordy_hits[:8]

    # Cliches.
    cliche_spans: list[tuple[int, int]] = []
    cliche_hits: list[str] = []
    for phrase in _CLICHES:
        for s, e in _phrase_spans(text, phrase):
            cliche_spans.append((s, e))
            cliche_hits.append(phrase)
    cliche_spans.sort()
    cliche_spans = cliche_spans[:8]
    cliche_hits = cliche_hits[:5]

    # Adverbs (-ly manner words, Hemingway style).
    adverb_spans: list[tuple[int, int]] = []
    adverb_hits: list[str] = []
    for m in re.finditer(r"\b[A-Za-z]{4,}ly\b", text):
        w = m.group(0).lower()
        if w in _ADVERB_EXCLUDE:
            continue
        adverb_spans.append((m.start(), m.end()))
        adverb_hits.append(w)
    adverb_spans = adverb_spans[:20]
    adverb_hits = sorted(set(adverb_hits))[:8]

    # Contractions (only when a formal tone is requested).
    contraction_spans: list[tuple[int, int]] = []
    contraction_hits: list[str] = []
    if formal:
        for m in _CONTRACTION_RE.finditer(text):
            contraction_spans.append((m.start(), m.end()))
            contraction_hits.append(m.group(0))
        contraction_spans = contraction_spans[:20]
        contraction_hits = sorted(set(contraction_hits))[:8]

    # Weak / vague words -> stronger alternatives.
    weak_spans: list[tuple[int, int, str]] = []
    weak_hits: list[tuple[str, list[str]]] = []
    for word, alts in _WEAK:
        for s, e in _phrase_spans(text, word):
            weak_spans.append((s, e, alts[0]))
            weak_hits.append((word, alts))
    weak_spans.sort(key=lambda t: t[0])
    weak_spans = weak_spans[:12]
    weak_hits = weak_hits[:8]

    # Repeated content words (used 3+ times) and immediate "the the" doubles.
    tokens = [(m.group(0).lower(), m.start(), m.end())
              for m in re.finditer(r"[A-Za-z][A-Za-z'-]*", text)]
    counts: dict[str, int] = {}
    for w, _, _ in tokens:
        if len(w) >= 4 and w not in _STOPWORDS:
            counts[w] = counts.get(w, 0) + 1
    repeated = sorted(((w, c) for w, c in counts.items() if c >= 3),
                      key=lambda t: -t[1])[:6]
    repeated_set = {w for w, _ in repeated}
    repetition_spans: list[tuple[int, int, str]] = []
    for w, s, e in tokens:
        if w in repeated_set:
            repetition_spans.append((s, e, w))
    repetition_spans = repetition_spans[:24]
    for prev, cur in zip(tokens, tokens[1:]):
        if prev[0] == cur[0] and len(cur[0]) >= 3:
            repetition_spans.append((cur[1], cur[2], cur[0]))
    repetition_spans.sort()

    return StyleReport(
        words=n_words,
        sentences=n_sent,
        paragraphs=paragraphs,
        avg_sentence_len=round(n_words / n_sent, 1) if n_sent else 0.0,
        reading_ease=round(ease, 1),
        grade_level=round(grade, 1),
        reading_time_min=round(n_words / 200, 1),
        flesch_label=_flesch_label(ease) if n_words else "—",
        longest_sentence_words=longest,
        passive_hits=passive,
        long_sentences=long_sents,
        hedging_hits=hedging,
        filler_hits=filler,
        wordy_hits=wordy_hits,
        cliche_hits=cliche_hits,
        adverb_hits=adverb_hits,
        contraction_hits=contraction_hits,
        weak_hits=weak_hits,
        repeated_words=repeated,
        passive_spans=passive_spans,
        hedging_spans=hedging_spans,
        filler_spans=filler_spans,
        wordy_spans=wordy_spans,
        cliche_spans=cliche_spans,
        adverb_spans=adverb_spans,
        contraction_spans=contraction_spans,
        weak_spans=weak_spans,
        repetition_spans=repetition_spans,
        sentence_spans=sentence_spans,
    )


# -------------------------------------------------------------------- tone report
@dataclass
class ToneReport:
    tone: str              # Confident / Tentative / Positive / Critical / Neutral
    formality: int         # 0 (very casual) .. 100 (very formal)
    note: str


_POSITIVE = {"great", "excellent", "good", "happy", "success", "strong",
             "improve", "benefit", "effective", "clear", "confident", "love",
             "enjoy", "win", "achieve", "best", "better", "helpful", "positive",
             "exciting", "valuable", "delighted", "proud", "opportunity"}
_NEGATIVE = {"bad", "poor", "fail", "problem", "issue", "difficult", "hard",
             "worry", "risk", "angry", "sad", "hate", "wrong", "error",
             "damage", "loss", "weak", "negative", "confusing", "broken",
             "unfortunately", "concern", "delay", "bug"}
_FORMAL = {"therefore", "furthermore", "moreover", "thus", "hence",
           "consequently", "regarding", "pursuant", "aforementioned", "shall",
           "endeavor", "utilize", "commence", "hereby", "thereby"}
_INFORMAL = {"gonna", "wanna", "kinda", "sorta", "yeah", "yep", "okay",
             "guys", "stuff", "cool", "awesome", "huge", "btw", "lol",
             "hey", "super", "totally", "freaking"}


def tone_report(text: str) -> ToneReport:
    low = text.lower()
    words = re.findall(r"[a-z']+", low)
    wset = set(words)
    pos = len(wset & _POSITIVE)
    neg = len(wset & _NEGATIVE)
    formal = len(wset & _FORMAL)
    informal = len(wset & _INFORMAL)
    contractions = len(_CONTRACTION_RE.findall(text))
    exclaims = text.count("!")
    hedges = len([w for w in words if w in _HEDGE_WORDS])
    fillers = len([w for w in words if w in _FILLER_WORDS])

    if hedges >= 2 or fillers >= 3:
        tone, note = "Tentative", "Softeners make the writing sound unsure."
    elif neg > pos:
        tone, note = "Critical", "The wording leans negative."
    elif pos > neg:
        tone, note = "Positive", "The wording leans optimistic."
    else:
        tone, note = "Neutral", "Balanced, matter-of-fact wording."

    formality = 50 + 9 * formal - 9 * informal - 4 * contractions - 3 * exclaims
    formality = max(0, min(100, formality))
    return ToneReport(tone=tone, formality=formality, note=note)


if __name__ == "__main__":
    demo = ("At the end of the day, the report was written by me and it is very good. "
            "Their going to the store. This are a test of the sistem. "
            "In order to make a decision, we need to think outside the box.")
    print("== spell_check ==")
    for h in spell_check(demo):
        print(f"  {h.word!r} -> {h.suggestions}")
    print("\n== style_report ==")
    sr = style_report(demo, formal=True)
    print(f"  words={sr.words} sentences={sr.sentences} paras={sr.paragraphs}")
    print(f"  ease={sr.reading_ease} ({sr.flesch_label}) grade={sr.grade_level}")
    print(f"  passive={sr.passive_hits}")
    print(f"  hedging={sr.hedging_hits} filler={sr.filler_hits}")
    print(f"  wordy={sr.wordy_hits} cliche={sr.cliche_hits}")
    print(f"  adverbs={sr.adverb_hits} contractions={sr.contraction_hits}")
    print(f"  weak={sr.weak_hits} repeated={sr.repeated_words}")
    print("\n== tone_report ==")
    tr = tone_report(demo)
    print(f"  {tr.tone} / formality {tr.formality} -- {tr.note}")
    print("\n== explain_diff ==")
    polished = "The report was written by me and it is good. They are going to the store. This is a test of the system."
    for c in explain_diff(demo, polished):
        print(f"  [{c.category}] {c.original!r} -> {c.polished!r}  ({c.explanation})")
