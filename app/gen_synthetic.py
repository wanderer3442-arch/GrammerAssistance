"""Generate synthetic grammar-error pairs for training the project's own model.

This is the data-generation (NLP) stage of the project: instead of hand-writing
pairs, we programmatically inject known error types into fluent sentences to
produce (error, correction) pairs at scale.

Error families
--------------
1. subject-verb agreement  (3rd-person singular -s / -es / -ies)
2. irregular past tense     ("Yesterday, they see" -> "saw")
3. past participle          ("has broke" -> "has broken")
4. a / an by vowel *sound*  ("an useful tool" -> "a useful tool")
5. plural noun marking      ("two book" -> "two books", "three child" -> "children")
6. capitalization           ("my sister ..." -> "My sister ...", "i" -> "I")
7. identity / no-op pairs   (already-correct sentences -> teaches when NOT to edit)

Every verb is paired with an object or adverbial that fits it, so the sentences
stay grammatical.  The generator deliberately uses different subjects, verbs and
complements from the 300-pair held-out set, and a separate synthetic test split
is held out as well.

Usage
-----
    python app/gen_synthetic.py --out data/synthetic_round5.csv \
                                --test data/synthetic_test.csv --seed 42
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------
# Lexicon
# --------------------------------------------------------------------------
# (base, 3rd-person singular, simple past, past participle)
VERBS = [
    ("study", "studies", "studied", "studied"),
    ("cry", "cries", "cried", "cried"),
    ("try", "tries", "tried", "tried"),
    ("carry", "carries", "carried", "carried"),
    ("hurry", "hurries", "hurried", "hurried"),
    ("worry", "worries", "worried", "worried"),
    ("fly", "flies", "flew", "flown"),
    ("fix", "fixes", "fixed", "fixed"),
    ("wash", "washes", "washed", "washed"),
    ("watch", "watches", "watched", "watched"),
    ("teach", "teaches", "taught", "taught"),
    ("reach", "reaches", "reached", "reached"),
    ("push", "pushes", "pushed", "pushed"),
    ("miss", "misses", "missed", "missed"),
    ("pass", "passes", "passed", "passed"),
    ("finish", "finishes", "finished", "finished"),
    ("go", "goes", "went", "gone"),
    ("do", "does", "did", "done"),
    ("have", "has", "had", "had"),
    ("see", "sees", "saw", "seen"),
    ("eat", "eats", "ate", "eaten"),
    ("drink", "drinks", "drank", "drunk"),
    ("begin", "begins", "began", "begun"),
    ("leave", "leaves", "left", "left"),
    ("know", "knows", "knew", "known"),
    ("wear", "wears", "wore", "worn"),
    ("feel", "feels", "felt", "felt"),
    ("break", "breaks", "broke", "broken"),
    ("forget", "forgets", "forgot", "forgotten"),
    ("run", "runs", "ran", "run"),
    ("throw", "throws", "threw", "thrown"),
    ("ride", "rides", "rode", "ridden"),
    ("sing", "sings", "sang", "sung"),
    ("speak", "speaks", "spoke", "spoken"),
    ("take", "takes", "took", "taken"),
    ("write", "writes", "wrote", "written"),
    ("give", "gives", "gave", "given"),
    ("make", "makes", "made", "made"),
    ("open", "opens", "opened", "opened"),
    ("ring", "rings", "rang", "rung"),
    ("stop", "stops", "stopped", "stopped"),
    ("plan", "plans", "planned", "planned"),
    ("walk", "walks", "walked", "walked"),
    ("work", "works", "worked", "worked"),
    ("play", "plays", "played", "played"),
    ("live", "lives", "lived", "lived"),
    ("close", "closes", "closed", "closed"),
    ("start", "starts", "started", "started"),
    ("enjoy", "enjoys", "enjoyed", "enjoyed"),
    ("buy", "buys", "bought", "bought"),
    ("bring", "brings", "brought", "brought"),
    ("think", "thinks", "thought", "thought"),
    ("catch", "catches", "caught", "caught"),
    ("build", "builds", "built", "built"),
    ("send", "sends", "sent", "sent"),
    ("find", "finds", "found", "found"),
    ("hold", "holds", "held", "held"),
    ("keep", "keeps", "kept", "kept"),
    ("lose", "loses", "lost", "lost"),
    ("meet", "meets", "met", "met"),
    ("pay", "pays", "paid", "paid"),
    ("say", "says", "said", "said"),
    ("sell", "sells", "sold", "sold"),
    ("sit", "sits", "sat", "sat"),
    ("sleep", "sleeps", "slept", "slept"),
    ("spend", "spends", "spent", "spent"),
    ("stand", "stands", "stood", "stood"),
    ("swim", "swims", "swam", "swum"),
    ("win", "wins", "won", "won"),
    ("choose", "chooses", "chose", "chosen"),
    ("drive", "drives", "drove", "driven"),
    ("grow", "grows", "grew", "grown"),
    ("draw", "draws", "drew", "drawn"),
    ("fall", "falls", "fell", "fallen"),
    ("hide", "hides", "hid", "hidden"),
    ("lead", "leads", "led", "led"),
    ("shake", "shakes", "shook", "shaken"),
    ("steal", "steals", "stole", "stolen"),
    ("wake", "wakes", "woke", "woken"),
    ("answer", "answers", "answered", "answered"),
    ("arrive", "arrives", "arrived", "arrived"),
    ("clean", "cleans", "cleaned", "cleaned"),
    ("collect", "collects", "collected", "collected"),
    ("deliver", "delivers", "delivered", "delivered"),
    ("design", "designs", "designed", "designed"),
    ("discover", "discovers", "discovered", "discovered"),
    ("explain", "explains", "explained", "explained"),
    ("follow", "follows", "followed", "followed"),
    ("help", "helps", "helped", "helped"),
    ("invite", "invites", "invited", "invited"),
    ("join", "joins", "joined", "joined"),
    ("learn", "learns", "learned", "learned"),
    ("listen", "listens", "listened", "listened"),
    ("move", "moves", "moved", "moved"),
    ("need", "needs", "needed", "needed"),
    ("paint", "paints", "painted", "painted"),
    ("prepare", "prepares", "prepared", "prepared"),
    ("remember", "remembers", "remembered", "remembered"),
    ("repair", "repairs", "repaired", "repaired"),
    ("solve", "solves", "solved", "solved"),
    ("travel", "travels", "travelled", "travelled"),
    ("visit", "visits", "visited", "visited"),
    ("wait", "waits", "waited", "waited"),
]

# Object (or prepositional phrase) that makes each verb's sentence natural.
OBJECTS = {
    "study": "the report", "try": "the new method", "carry": "the heavy box",
    "worry": "about the exam", "fly": "to the island", "fix": "the engine",
    "wash": "the car", "watch": "the match", "teach": "mathematics",
    "reach": "the station", "push": "the heavy cart", "miss": "the morning train",
    "pass": "the final exam", "finish": "the report", "do": "the homework",
    "have": "a new idea", "see": "the old bridge", "eat": "a fresh meal",
    "drink": "a glass of water", "begin": "the lesson", "leave": "the office",
    "know": "the answer", "wear": "a blue jacket", "feel": "the cold wind",
    "break": "the old chair", "forget": "the password", "throw": "the ball",
    "ride": "a bicycle", "sing": "a familiar song", "speak": "French",
    "take": "the early train", "write": "a short letter",
    "give": "a helpful answer", "make": "a wooden table",
    "open": "the front door", "ring": "the bell", "stop": "the car",
    "plan": "the summer trip", "play": "the guitar", "close": "the window",
    "start": "the meeting", "enjoy": "the concert", "buy": "a new phone",
    "bring": "a warm coat", "think": "about the problem",
    "catch": "the last bus", "build": "a small bridge", "send": "the invoice",
    "find": "the missing key", "hold": "the heavy box",
    "keep": "the old letters", "lose": "the house key",
    "meet": "the new client", "pay": "the electricity bill", "say": "the truth",
    "sell": "fresh bread", "spend": "the whole afternoon",
    "win": "the first prize", "choose": "the blue option",
    "drive": "the old van", "grow": "tomatoes", "draw": "a quick sketch",
    "hide": "the spare key", "lead": "the morning meeting",
    "shake": "the bottle", "steal": "the silver watch",
    "wake": "the sleeping child", "answer": "the phone",
    "clean": "the kitchen", "collect": "the empty bottles",
    "deliver": "the parcel", "design": "a small garden",
    "discover": "a hidden path", "explain": "the new rule",
    "follow": "the narrow path", "help": "the new student",
    "invite": "the whole class", "join": "the reading club",
    "learn": "a new language", "listen": "to the radio",
    "move": "the heavy table", "need": "a longer break",
    "paint": "the garden fence", "prepare": "the evening meal",
    "remember": "the appointment", "repair": "the broken chair",
    "solve": "the difficult problem", "visit": "the city museum",
    "wait": "for the next bus",
}

# Intransitive verbs: a short list of adverbials that fit each one.
INTRANS = {
    "arrive": ["at the station", "before noon", "on time"],
    "cry": ["every night", "during the film", "without a sound"],
    "hurry": ["to the station", "after work", "before the rain"],
    "go": ["to the market", "to the office", "home"],
    "run": ["along the path", "in the park", "every morning"],
    "walk": ["to the station", "along the river", "after lunch"],
    "work": ["at the community centre", "in the office", "from home"],
    "live": ["near the station", "in the old town", "on the top floor"],
    "sit": ["near the window", "on the bench", "by the fire"],
    "sleep": ["in the small room", "after lunch", "until noon"],
    "stand": ["near the door", "by the window", "in the queue"],
    "swim": ["in the lake", "every morning", "with the club"],
    "fall": ["on the wet floor", "during the storm", "behind the shed"],
    "travel": ["by train", "every summer", "with the team"],
}

ADVERBIALS = [
    "every morning", "after lunch", "before the meeting", "on Saturday",
    "with great care", "again", "in the evening", "at the weekend",
    "during the break", "without delay",
]
# Safe with an explicit past-time marker ("Last night, ...").
NEUTRAL_ADVERBIALS = [
    "with great care", "again", "without delay", "for a moment", "in the end",
]
# Intransitive phrases that clash with an explicit past-time marker.
TIME_BOUND = {
    "every morning", "after lunch", "before noon", "every summer",
    "after work", "before the rain", "until noon", "during the film",
    "during the storm", "on time",
}

# Subjects chosen to differ from the held-out set's exact wording.
SUBJ_SG = [
    "The mechanic", "My neighbour", "The librarian", "The pilot",
    "The chef", "My brother", "The nurse", "The farmer", "The architect",
    "The journalist", "My grandmother", "The electrician", "The coach",
    "The waiter", "The dentist", "The translator", "The photographer",
    "The receptionist", "My roommate", "The plumber", "The scientist",
    "The cashier", "The driver", "The singer", "The painter",
    "The programmer", "The volunteer", "The tourist", "The lawyer",
    "The shopkeeper", "The cyclist", "The musician", "The carpenter",
    "The editor", "The sailor", "The guard", "The baker",
]

SUBJ_PL = [
    "The mechanics", "My neighbours", "The librarians", "The pilots",
    "The chefs", "My brothers", "The nurses", "The farmers",
    "The architects", "The journalists", "My grandparents", "The electricians",
    "The coaches", "The waiters", "The dentists", "The translators",
    "The photographers", "The receptionists", "My roommates", "The plumbers",
    "The scientists", "The cashiers", "The drivers", "The singers",
    "The painters", "The programmers", "The volunteers", "The tourists",
    "The lawyers", "The shopkeepers", "The cyclists", "The musicians",
    "The carpenters", "The editors", "The sailors", "The guards",
    "The bakers",
]

TIME_MARKERS = [
    "Yesterday", "Last night", "Last week", "Last month", "Last year",
    "Earlier today", "Two days ago", "On Monday", "Last summer",
    "A few minutes ago", "Last Friday", "The day before",
]

# a / an depends on the *sound* of the next word, not the letter.
OBJ_A = [
    "book", "table", "phone", "car", "house", "horse", "uniform",
    "university", "useful tool", "European country", "one-way street",
    "user account", "unique idea", "yellow jacket", "red apple",
    "small office", "wooden chair", "metal box", "camera", "laptop",
    "garden", "village", "museum", "station", "window", "door", "key",
    "map", "bottle", "pencil", "notebook", "ticket", "wallet",
    "European city", "useful website", "one-hour lesson", "user manual",
    "unique painting", "wooden spoon", "plastic bag", "green scarf",
    "heavy suitcase", "quiet room", "busy street",
]
OBJ_AN = [
    "apple", "orange", "egg", "idea", "umbrella", "island", "elephant",
    "exam", "event", "accident", "opinion", "opportunity", "empty bottle",
    "interesting article", "easy question", "old friend", "address",
    "email", "invitation", "interview", "answer", "example", "exercise",
    "album", "envelope", "onion", "honest answer", "unusual request",
    "old photograph", "empty cup", "interesting story", "easy recipe",
]
ROLE_A = [
    "teacher", "doctor", "student", "artist", "European", "user",
    "musician", "dentist", "pilot", "lawyer", "Canadian", "German",
    "nurse", "scientist",
]
ROLE_AN = [
    "engineer", "American", "Indian", "MBA", "uncle", "honest man",
    "actor", "editor", "electrician", "architect", "Australian",
    "Italian", "optimist", "expert",
]

OBJ_FRAMES = [
    ("She bought", "yesterday."),
    ("He found", "in the drawer."),
    ("I need", "for the trip."),
    ("We saw", "near the station."),
    ("The shop sells", "at a fair price."),
    ("My sister wants", "for her birthday."),
    ("He borrowed", "from the library."),
    ("The guide showed us", "near the entrance."),
]
ROLE_FRAMES = ["He is", "She is", "My friend is"]

# (singular, plural)
PLURAL_NOUNS = [
    ("book", "books"), ("car", "cars"), ("table", "tables"),
    ("phone", "phones"), ("bird", "birds"), ("idea", "ideas"),
    ("chair", "chairs"), ("box", "boxes"), ("watch", "watches"),
    ("dish", "dishes"), ("child", "children"), ("man", "men"),
    ("woman", "women"), ("person", "people"), ("mouse", "mice"),
    ("foot", "feet"), ("tooth", "teeth"), ("leaf", "leaves"),
    ("knife", "knives"), ("shelf", "shelves"), ("city", "cities"),
    ("country", "countries"), ("story", "stories"), ("baby", "babies"),
    ("family", "families"), ("party", "parties"), ("bus", "buses"),
    ("class", "classes"), ("glass", "glasses"), ("branch", "branches"),
]
PLURAL_NUMBERS = ["two", "three", "four", "five", "six", "several", "many"]
PLURAL_FRAMES = [
    ("There are", "here."),
    ("I noticed", "today."),
    ("We need", "for the project."),
]

# Already-correct sentences for the identity (no-op) task.
IDENTITY_SENTENCES = [
    "The mechanic repairs the engine carefully.",
    "My neighbour waters the plants every morning.",
    "The librarian organises the shelves on Fridays.",
    "The chef prepares a fresh meal for the guests.",
    "The nurse checks the patient before the round.",
    "The farmer collects the eggs at dawn.",
    "The architect designs a small community centre.",
    "The journalist interviews the mayor after lunch.",
    "The dentist explains the procedure very clearly.",
    "The translator works on the report until midnight.",
    "The photographer captures the light across the river.",
    "The receptionist answers every call politely.",
    "The plumber repairs the pipe under the sink.",
    "The scientist records the results in a notebook.",
    "The cashier counts the money at the end of the shift.",
    "The driver checks the tyres before the journey.",
    "The singer practises the song in the evening.",
    "The painter mixes the colours on a wooden board.",
    "The programmer tests the code before the release.",
    "The volunteer helps the children with their homework.",
    "The tourists visit the museum on Sunday.",
    "The lawyer reviews the contract with the client.",
    "The shopkeeper opens the store at nine.",
    "The cyclist rides along the river every weekend.",
    "The musician tunes the guitar before the show.",
    "The carpenter builds a small table for the kitchen.",
    "The editor reads the article twice.",
    "The sailor checks the ropes before the storm.",
    "The guard locks the gate after sunset.",
    "The baker sells fresh bread every morning.",
    "I have finished the assignment already.",
    "She has written three letters this week.",
    "They have eaten dinner at the new restaurant.",
    "He has broken the old chair.",
    "We have known each other for years.",
    "The children have finished their project.",
    "She bought an apple on the way home.",
    "He found a useful tool in the drawer.",
    "I need an umbrella for the trip.",
    "They ordered a European dish from the menu.",
    "There are three children in the room.",
    "I saw several cities near the border.",
    "My brother works at the community centre.",
    "The nurses start their shift at seven.",
    "Yesterday, the pilot flew to the island.",
    "Last week, the students wrote a long report.",
    "Two days ago, the mechanic repaired the engine.",
]

CAPITALIZATION_PAIRS = [
    ("i think I can finish it.", "I think I can finish it."),
    ("She said i should wait.", "She said I should wait."),
    ("My friend and i went home.", "My friend and I went home."),
    ("i forgot the password again.", "I forgot the password again."),
    ("He asked whether i was ready.", "He asked whether I was ready."),
    ("my brother and i fixed the bike.", "My brother and I fixed the bike."),
]


def _lower_first(text: str) -> str:
    """Lowercase the first letter unless the word is the pronoun 'I'."""
    if text.startswith("I ") or text == "I":
        return text
    return text[0].lower() + text[1:] if text else text


def _article(noun: str) -> str:
    return "an" if noun in OBJ_AN or noun in ROLE_AN else "a"


def _complement(rng: random.Random, base: str, allow_time: bool = True) -> str:
    """Return a natural object/adverbial for a verb."""
    if base in INTRANS:
        options = INTRANS[base]
        if not allow_time:
            options = [o for o in options if o not in TIME_BOUND] or [""]
        return rng.choice(options)
    obj = OBJECTS[base]
    pool = ADVERBIALS if allow_time else NEUTRAL_ADVERBIALS
    if rng.random() < 0.55:
        return f"{obj} {rng.choice(pool)}"
    return obj


def generate(seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    pairs: list[dict] = []
    seen: set[str] = set()

    def add(src: str, tgt: str, kind: str) -> None:
        src, tgt = src.strip(), tgt.strip()
        if src == tgt or src in seen:
            return
        seen.add(src)
        pairs.append({"input": src, "target": tgt, "type": kind})

    def add_identity(sentence: str) -> None:
        sentence = sentence.strip()
        if sentence in seen:
            return
        seen.add(sentence)
        pairs.append({"input": sentence, "target": sentence, "type": "identity"})

    # ---- 1. subject-verb agreement -------------------------------------
    for _ in range(3400):
        base, third, _past, _part = rng.choice(VERBS)
        comp = _complement(rng, base)
        if rng.random() < 0.5:
            subj = rng.choice(SUBJ_SG)
            add(f"{subj} {base} {comp}.", f"{subj} {third} {comp}.", "agreement_sg")
        else:
            subj = rng.choice(SUBJ_PL)
            add(f"{subj} {third} {comp}.", f"{subj} {base} {comp}.", "agreement_pl")

    # ---- 2. irregular / regular simple past ----------------------------
    for _ in range(2400):
        base, _third, past, _part = rng.choice(VERBS)
        if past == base:  # no visible change -> no usable error
            continue
        subj = rng.choice(SUBJ_SG + SUBJ_PL)
        comp = _complement(rng, base, allow_time=False)
        marker = rng.choice(TIME_MARKERS)
        add(f"{marker}, {_lower_first(subj)} {base} {comp}.",
            f"{marker}, {_lower_first(subj)} {past} {comp}.",
            "past_tense")

    # ---- 3. past participle after have / has --------------------------
    for _ in range(2200):
        base, _third, past, part = rng.choice(VERBS)
        if past == part:  # regular verbs -> past == participle, no error
            continue
        if rng.random() < 0.5:
            subj, have = rng.choice(SUBJ_SG), "has"
        else:
            subj, have = rng.choice(SUBJ_PL), "have"
        comp = _complement(rng, base)
        add(f"{subj} {have} {past} {comp}.",
            f"{subj} {have} {part} {comp}.",
            "participle")

    # ---- 4. a / an by vowel sound --------------------------------------
    for _ in range(1800):
        if rng.random() < 0.75:
            noun = rng.choice(OBJ_A + OBJ_AN)
            start, end = rng.choice(OBJ_FRAMES)
            right = f"{_article(noun)} {noun}"
            wrong = f"{'a' if _article(noun) == 'an' else 'an'} {noun}"
            add(f"{start} {wrong} {end}", f"{start} {right} {end}", "article")
        else:
            role = rng.choice(ROLE_A + ROLE_AN)
            start = rng.choice(ROLE_FRAMES)
            right = f"{_article(role)} {role}"
            wrong = f"{'a' if _article(role) == 'an' else 'an'} {role}"
            add(f"{start} {wrong}.", f"{start} {right}.", "article")

    # ---- 5. plural noun marking ----------------------------------------
    for _ in range(1600):
        singular, plural = rng.choice(PLURAL_NOUNS)
        number = rng.choice(PLURAL_NUMBERS)
        start, end = rng.choice(PLURAL_FRAMES)
        add(f"{start} {number} {singular} {end}",
            f"{start} {number} {plural} {end}",
            "plural_noun")

    # ---- 6. capitalization ---------------------------------------------
    for sentence in rng.sample(IDENTITY_SENTENCES, 24):
        add(_lower_first(sentence), sentence, "capitalization")
    for _ in range(700):
        base, _third, past, _part = rng.choice(VERBS)
        subj = rng.choice(SUBJ_SG + SUBJ_PL)
        comp = _complement(rng, base)
        marker = rng.choice(TIME_MARKERS)
        sentence = f"{marker}, {_lower_first(subj)} {past} {comp}."
        add(_lower_first(sentence), sentence, "capitalization")
    for _ in range(250):
        wrong, right = rng.choice(CAPITALIZATION_PAIRS)
        add(wrong, right, "capitalization")

    # ---- 7. identity / no-op -------------------------------------------
    for sentence in IDENTITY_SENTENCES:
        add_identity(sentence)
    # Reuse correct targets as no-op examples (teaches the model to copy).
    targets = [p["target"] for p in pairs if p["type"] != "identity"]
    rng.shuffle(targets)
    for sentence in targets[:2200]:
        add_identity(sentence)

    rng.shuffle(pairs)
    return pairs


def _load_inputs(*paths: Path) -> set[str]:
    out: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        with open(path, encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                src = (row.get("input") or "").strip()
                if src:
                    out.add(src)
    return out


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["input", "target", "type"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/synthetic_round5.csv")
    parser.add_argument("--test", default="data/synthetic_test.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=int, default=700)
    parser.add_argument("--show", type=int, default=12)
    args = parser.parse_args()

    # Never let a generated input collide with the evaluation sets.
    protected = _load_inputs(
        ROOT / "eval" / "test_split.csv",
        ROOT / "data" / "grammar_supplementary_300_reviewed_v2.csv",
        ROOT / "data" / "grammar_supplementary_300_capitalization_fixed.csv",
    )

    rows = [r for r in generate(args.seed) if r["input"] not in protected]

    rng = random.Random(args.seed + 1)
    rng.shuffle(rows)
    test_rows = rows[: args.test_size]
    train_rows = rows[args.test_size :]

    _write(ROOT / args.out, train_rows)
    _write(ROOT / args.test, test_rows)

    print(f"train: {len(train_rows)}  -> {args.out}")
    print(f"test : {len(test_rows)}  -> {args.test}")
    print(f"protected inputs checked: {len(protected)}")

    print("\ntrain by type:")
    for kind, n in Counter(r["type"] for r in train_rows).most_common():
        print(f"  {kind:16s} {n}")
    print("\ntest by type:")
    for kind, n in Counter(r["type"] for r in test_rows).most_common():
        print(f"  {kind:16s} {n}")

    print(f"\nsamples ({args.show}):")
    for row in train_rows[: args.show]:
        print(f"  [{row['type']}]")
        print(f"    IN : {row['input']}")
        print(f"    OUT: {row['target']}")


if __name__ == "__main__":
    main()
