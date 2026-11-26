import random
import sys
from pathlib import Path

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "data" / "demo" / "demo_cohort.csv"
N = 100
HEADER = "identifier,favourite_colour,ice_cream,city,hobbies,pet,coffees_per_day"

COLOURS = {"Red": 24, "Blue": 22, "Green": 20, "Yellow": 18, "Purple": 16}
ICE = {"Vanilla": 26, "Chocolate": 24, "Strawberry": 20, "Pistachio": 16, "Mango": 14}
CITIES = {"Lisbon": 22, "Oslo": 18, "Nairobi": 17, "Kyoto": 16, "Lima": 14, "Toronto": 13}
PETS = {"Cat": 28, "Dog": 26, "Fish": 18, "Parrot": 12, "None": 16}
COFFEE = {0: 8, 1: 20, 2: 30, 3: 24, 4: 12, 5: 6}
HOBBIES = ["Chess", "Surfing", "Baking", "Hiking", "Gaming", "Painting", "Karaoke"]


def expand(counts):
    out = []
    for value, n in counts.items():
        out.extend([value] * n)
    assert len(out) == N, (counts, len(out))
    return out


def hobby_column(rng):
    values = []
    for i in range(N):
        if i < 33:
            a, b = rng.sample(HOBBIES, 2)
            values.append(f"{a}|{b}")
        else:
            values.append(rng.choice(HOBBIES))
    rng.shuffle(values)
    return values


def build(seed):
    rng = random.Random(seed)
    cols = {
        "favourite_colour": expand(COLOURS),
        "ice_cream": expand(ICE),
        "city": expand(CITIES),
        "hobbies": hobby_column(rng),
        "pet": expand(PETS),
        "coffees_per_day": [str(v) for v in expand(COFFEE)],
    }
    for name in cols:
        if name != "hobbies":
            rng.shuffle(cols[name])
    return cols


def rows_of(cols):
    order = ["favourite_colour", "ice_cream", "city", "hobbies", "pet", "coffees_per_day"]
    return [tuple(cols[c][i] for c in order) for i in range(N)]


rng_seed = 2026
while True:
    cols = build(rng_seed)
    rows = rows_of(cols)
    if len(set(rows)) == N:
        break
    rng_seed += 1

lines = [HEADER]
for i, row in enumerate(rows, start=1):
    lines.append(",".join([str(i)] + list(row)))

with open(OUT, "w", encoding="utf-8", newline="\n") as f:
    f.write("\n".join(lines) + "\n")

print("written", OUT, "seed used", rng_seed)
