"""Round-3 targeted pairs: close the final measurable gaps.

Priority (from diagnose.py):
  1. article an_to_a  (86.7% miss) — 'an' + consonant-SOUND word:
     an useful/university/European/user/unique/one-time -> a ...
  2. 3rd-person spelling — model writes 'crys'/'fixs' instead of 'cries'/'fixes'
  3. irregular past participles — ate->eaten, saw->seen, came->came, began, taught
  4. adjective+preposition collocations — good at, keen on, waiting for,
     congratulated on, succeeded in, different from
"""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

pairs = [
    # ---- 1. an -> a before consonant SOUND (highest priority) ----
    ("This is an useful tool.", "This is a useful tool."),
    ("She is an university student.", "She is a university student."),
    ("They visited an European country.", "They visited a European country."),
    ("I created an user account.", "I created a user account."),
    ("That is an unique idea.", "That is a unique idea."),
    ("It was an one-time offer.", "It was a one-time offer."),
    ("He is an European player.", "He is a European player."),
    ("She bought an university jacket.", "She bought a university jacket."),
    ("He gave an Ouija board to her.", "He gave a Ouija board to her."),
    ("That is an UFO that we saw.", "That is a UFO that we saw."),
    ("She joined an MBA course last year.", "She joined a MBA course last year."),
    ("He played an Xbox game all night.", "He played a Xbox game all night."),
    ("It was an huge error in the code.", "It was a huge error in the code."),
    ("She is an undergraduate here.", "She is a undergraduate here."),
    ("He made an honest but useful point.", "He made a useful point."),
    ("They bought an old but useful tool.", "They bought a useful tool."),
    ("She is an union member.", "She is a union member."),
    ("He is an worthy opponent.", "He is a worthy opponent."),
    ("It is an one-way street.", "It is a one-way street."),
    ("She wore an uniform to class.", "She wore a uniform to class."),

    # ---- 2. 3rd-person singular spelling (-ies, not -ys) ----
    ("The baby cry every night.", "The baby cries every night."),
    ("My uncle fix old bicycles.", "My uncle fixes old bicycles."),
    ("The phone ring during class.", "The phone rings during class."),
    ("Her friend teach mathematics.", "Her friend teaches mathematics."),
    ("The train stop at this station.", "The train stops at this station."),
    ("My cousins studies computer science.", "My cousins study computer science."),
    ("My uncles fixes old bicycles.", "My uncles fix old bicycles."),
    ("The boys carries the boxes.", "The boys carry the boxes."),
    ("She study English every day.", "She studies English every day."),
    ("He go to school by bus.", "He goes to school by bus."),
    ("The child play in the park.", "The child plays in the park."),
    ("My sister work at a hospital.", "My sister works at a hospital."),
    ("The teacher give us homework.", "The teacher gives us homework."),
    ("It rain a lot in spring.", "It rains a lot in spring."),
    ("She dance very gracefully.", "She dances very gracefully."),

    # ---- 3. irregular verbs / participles ----
    ("They have ate dinner.", "They have eaten dinner."),
    ("Yesterday, They see a bird.", "Yesterday, they saw a bird."),
    ("Yesterday, They come home.", "Yesterday, they came home."),
    ("Yesterday, They begin the meeting.", "Yesterday, they began the meeting."),
    ("He has forgot the name.", "He has forgotten the name."),
    ("Yesterday, He teach the class.", "Yesterday, he taught the class."),
    ("She has wrote three letters.", "She has written three letters."),
    ("They have went to the store.", "They have gone to the store."),
    ("He has took the money.", "He has taken the money."),
    ("I have saw that movie before.", "I have seen that movie before."),
    ("She has began the project.", "She has begun the project."),
    ("We have came a long way.", "We have come a long way."),
    ("He has drank all the water.", "He has drunk all the water."),
    ("They have drove to the coast.", "They have driven to the coast."),
    ("She has wrote a novel.", "She has written a novel."),

    # ---- 4. adjective + preposition collocations ----
    ("I am waiting you outside.", "I am waiting for you outside."),
    ("He is good in mathematics.", "He is good at mathematics."),
    ("She is keen in learning English.", "She is keen on learning English."),
    ("They congratulated her for her success.", "They congratulated her on her success."),
    ("He succeeded to pass the exam.", "He succeeded in passing the exam."),
    ("He is different than his brother.", "He is different from his brother."),
    ("She is afraid from spiders.", "She is afraid of spiders."),
    ("It depend on the weather.", "It depends on the weather."),
    ("I listen to music every day.", "I listen to music every day."),
    ("This belong to me.", "This belongs to me."),
    ("She is married with a doctor.", "She is married to a doctor."),
    ("He is interested on physics.", "He is interested in physics."),
    ("The book consist of ten chapters.", "The book consists of ten chapters."),
    ("She is tired from walking.", "She is tired of walking."),
    ("I am waiting for the bus.", "I am waiting for the bus."),
]

# keep disjoint from test split + train pool so eval stays honest
existing = set()
for rel in ("eval/test_split.csv", "data/train_pool.csv"):
    p = ROOT / rel
    if p.exists():
        for r in csv.DictReader(open(p, encoding="utf-8")):
            existing.add((r["input"].strip(), r["target"].strip()))

seen, clean = set(), []
for s, t in pairs:
    k = (s.strip(), t.strip())
    if k in seen or k in existing:
        continue
    seen.add(k)
    clean.append({"input": s, "target": t})

out = ROOT / "data" / "targeted_pairs_round3.csv"
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["input", "target"])
    w.writeheader()
    w.writerows(clean)

print(f"round-3 pairs: {len(pairs)} -> {len(clean)} after de-dup/overlap")
print(f"written to {out}")
