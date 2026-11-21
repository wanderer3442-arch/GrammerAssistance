"""Round-2 targeted pairs: fix the article regression (catastrophic forgetting)
without starving wh-questions.

From the full-300 re-eval, articles dropped 70.3% -> 68.3%. The specific a/an
failures seen:
  a honest -> an honest,  an user -> a user,  a unique -> a unique,
  a expensive -> an expensive,  a honest,  a hour -> an hour
Rule: 'a' before consonant SOUND, 'an' before vowel SOUND (not letter!).
Also cover 'the'/'a' swaps and article insertion/removal.
"""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

pairs = [
    # ---- 'an' before vowel SOUND (letter may be consonant) ----
    ("I met a honest person.", "I met an honest person."),
    ("He is a honest man.", "He is an honest man."),
    ("She gave me a honest answer.", "She gave me an honest answer."),
    ("It was a honest mistake.", "It was an honest mistake."),
    ("They bought a expensive car.", "They bought an expensive car."),
    ("She wore a elegant dress.", "She wore an elegant dress."),
    ("He ate a apple yesterday.", "He ate an apple yesterday."),
    ("I need a orange juice.", "I need an orange juice."),
    ("She has a hour left.", "She has an hour left."),
    ("It took a hour to finish.", "It took an hour to finish."),
    ("He is a artist by profession.", "He is an artist by profession."),
    ("She is a engineer.", "She is an engineer."),
    ("We saw a eagle flying.", "We saw an eagle flying."),
    ("He waited for a hour.", "He waited for an hour."),
    ("That was a amazing story.", "That was an amazing story."),
    ("She gave a odd response.", "She gave an odd response."),
    ("He showed a ugly side.", "He showed an ugly side."),
    ("It is a honor to meet you.", "It is an honor to meet you."),
    ("She has a umbrella.", "She has an umbrella."),
    ("He ate a ice cream.", "He ate an ice cream."),

    # ---- 'a' before CONSONANT sound (letter may be vowel) ----
    ("I created an user account.", "I created a user account."),
    ("That is an unique idea.", "That is a unique idea."),
    ("She bought an university bag.", "She bought a university bag."),
    ("He is an European player.", "He is a European player."),
    ("It was an one-time offer.", "It was a one-time offer."),
    ("She is an useful helper.", "She is a useful helper."),
    ("He gave a Ouija board.", "He gave a Ouija board."),
    ("That is an UFO sighting.", "That is a UFO sighting."),
    ("She joined an MBA program.", "She joined a MBA program."),
    ("He played a Xbox game.", "He played a Xbox game."),
    ("It was an huge mistake.", "It was a huge mistake."),
    ("She made a wonderful painting.", "She made a wonderful painting."),
    ("He is a honest but unique case.", "He is a unique case."),

    # ---- article swaps / insertion / removal ----
    ("She is best player here.", "She is the best player here."),
    ("He is tallest in class.", "He is the tallest in class."),
    ("Moon looks bright tonight.", "The moon looks bright tonight."),
    ("Sun rises in east.", "The sun rises in the east."),
    ("Please lock door.", "Please lock the door."),
    ("I left keys on table.", "I left the keys on the table."),
    ("She plays guitar well.", "She plays the guitar well."),
    ("He is owner of shop.", "He is the owner of the shop."),
    ("They are best team.", "They are the best team."),
    ("She is most kind person.", "She is the most kind person."),
    ("I need information about course.", "I need information about the course."),
    ("He is only child.", "He is an only child."),
    ("She go to school daily.", "She goes to school daily."),
    ("They arrived at airport early.", "They arrived at the airport early."),
    ("He is waiting at bus stop.", "He is waiting at the bus stop."),
    ("She works in hospital.", "She works in the hospital."),
    ("We watched movie last night.", "We watched the movie last night."),
    ("He put plate on table.", "He put the plate on the table."),
    ("She is afraid of dogs.", "She is afraid of the dogs."),
    ("They live near park.", "They live near the park."),
]

# keep pairs disjoint from test split so eval stays honest
testp = ROOT / "eval" / "test_split.csv"
existing = set()
if testp.exists():
    for r in csv.DictReader(open(testp, encoding="utf-8")):
        existing.add((r["input"].strip(), r["target"].strip()))
pool = ROOT / "data" / "train_pool.csv"
if pool.exists():
    for r in csv.DictReader(open(pool, encoding="utf-8")):
        existing.add((r["input"].strip(), r["target"].strip()))

seen = set()
clean = []
for s, t in pairs:
    k = (s.strip(), t.strip())
    if k in seen or k in existing:
        continue
    seen.add(k)
    clean.append({"input": s, "target": t})

out = ROOT / "data" / "targeted_pairs_round2.csv"
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["input", "target"])
    w.writeheader()
    w.writerows(clean)

print(f"round-2 pairs: {len(pairs)} -> {len(clean)} after de-dup/overlap")
print(f"written to {out}")
