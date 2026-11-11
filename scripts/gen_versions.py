"""Create successive passage versions by editing numeric facts (years, counts, quantities).

Version 0 is the original text; versions 1..V each change one more numeric fact, which models
the typical Wikipedia fact update (dates, populations, scores) while keeping the passage id.
"""
import os
import random
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from unison.common import DATA, read_jsonl, write_jsonl

NUM = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)(?![\w.])")
V = 3


def edit_once(text, rng):
    spans = [m for m in NUM.finditer(text)]
    if not spans:
        return None
    m = rng.choice(spans)
    tok = m.group(1)
    raw = tok.replace(",", "")
    if "." in raw:
        val = float(raw) * rng.choice([0.8, 1.25, 1.5])
        new = f"{val:.{len(raw.split('.')[1])}f}"
    else:
        x = int(raw)
        if 1500 <= x <= 2030:  # year
            new = str(x + rng.choice([-7, -4, -2, 2, 3, 5]))
        elif x < 10:
            new = str(x + rng.choice([1, 2, 3]))
        else:
            new = str(int(x * rng.choice([0.7, 0.85, 1.2, 1.4])) + 1)
            if "," in tok:
                new = f"{int(new):,}"
    if new == tok:
        return None
    return text[:m.start(1)] + new + text[m.end(1):]


for ds in ["hotpotqa", "musique"]:
    for split in os.environ.get("SPLITS", "eval,train").split(","):
        d = os.path.join(DATA, ds, split)
        rng = random.Random(13)
        rows = []
        for c in read_jsonl(os.path.join(d, "corpus.jsonl")):
            texts, t = [c["text"]], c["text"]
            for _ in range(V):
                t2 = edit_once(t, rng)
                if t2 is None:
                    break
                texts.append(t2)
                t = t2
            if len(texts) > 1:
                rows.append(dict(pid=c["pid"], versions=texts))
        write_jsonl(os.path.join(d, "versions.jsonl"), rows)
        print(ds, split, len(rows), "editable passages")
