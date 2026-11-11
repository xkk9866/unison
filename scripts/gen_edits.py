"""Targeted, decision-changing passage versions for the version-safety stress test.

For every passage that contains a gold answer of a question whose context includes the passage, version 1
edits the answer-bearing sentence in one of three ways (chosen deterministically):
  entity    the answer span is replaced by another pool answer of the same shape (numeric / capitalized)
  number    every number inside the answer span is shifted (years by a few years, counts by 20-40%)
  negation  'not' is inserted after the first auxiliary verb of the sentence
Unlike gen_versions.py (random numeric edits anywhere), every edit here hits the sentence that a correct
note cites, so a reuse across versions is wrong unless the consumer's own computation re-derives it.
Output: <pool>/versions_edit.jsonl  {pid, versions: [v0, v1], edit: {type, sent, old, new}}
usage: python gen_edits.py [hotpotqa musique]"""
import os
import random
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from unison.common import DATA, read_jsonl, split_sentences, write_jsonl

AUX = re.compile(r"\b(is|was|are|were|has|had|did|does|can|could|will|would)\b")
NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")


def shape(a):
    return "num" if re.search(r"\d", a) else ("cap" if a[:1].isupper() else "low")


def shift_numbers(a, rng):
    def f(m):
        raw = m.group(0).replace(",", "")
        if "." in raw:
            return f"{float(raw) * rng.choice([0.8, 1.25]):.{len(raw.split('.')[1])}f}"
        x = int(raw)
        if 1500 <= x <= 2030:
            return str(x + rng.choice([-6, -3, 2, 4, 7]))
        return str(max(1, int(x * rng.choice([0.7, 1.3, 1.4])) + 1))
    out = NUM.sub(f, a)
    return out if out != a else None


def find(sent, a):
    m = re.search(r"(?i)(?<!\w)" + re.escape(a) + r"(?!\w)", sent)
    return m


def main(dss):
    for ds in dss:
        d = os.path.join(DATA, ds, "eval")
        corpus = {c["pid"]: c for c in read_jsonl(os.path.join(d, "corpus.jsonl"))}
        qs = read_jsonl(os.path.join(d, "questions.jsonl"))
        rng = random.Random(29)
        pool = {}
        for q in qs:
            for a in q["answers"]:
                if len(a) >= 2 and a.lower() not in ("yes", "no"):
                    pool.setdefault(shape(a), []).append(a)
        rows, done, kinds = [], set(), {}
        for q in qs:
            for pid in q.get("context_pids", []):
                if pid in done or pid not in corpus:
                    continue
                sents = split_sentences(corpus[pid]["text"])
                for si, sent in enumerate(sents):
                    hit = None
                    for a in q["answers"]:
                        if len(a) >= 2 and a.lower() not in ("yes", "no"):
                            m = find(sent, a)
                            if m:
                                hit = (a, m)
                                break
                    if hit is None:
                        continue
                    a, m = hit
                    order = ["entity", "number", "negation"]
                    rng.shuffle(order)
                    new_sent, kind = None, None
                    for kind in order:
                        if kind == "number" and shape(a) == "num":
                            b = shift_numbers(m.group(0), rng)
                        elif kind == "entity":
                            alts = [x for x in pool.get(shape(a), []) if x.lower() != a.lower()]
                            b = rng.choice(alts) if alts else None
                        elif kind == "negation":
                            mm = AUX.search(sent)
                            if mm:
                                new_sent = sent[:mm.end()] + " not" + sent[mm.end():]
                                break
                            b = None
                        else:
                            b = None
                        if b:
                            new_sent = sent[:m.start()] + b + sent[m.end():]
                            break
                    if not new_sent or new_sent == sent:
                        continue
                    text = corpus[pid]["text"]
                    v1 = text.replace(sent, new_sent, 1)
                    if v1 == text:
                        continue
                    rows.append(dict(pid=pid, versions=[text, v1],
                                     edit=dict(type=kind, sent=si + 1, old=sent, new=new_sent, qid=q["qid"])))
                    kinds[kind] = kinds.get(kind, 0) + 1
                    done.add(pid)
                    break
        write_jsonl(os.path.join(d, "versions_edit.jsonl"), rows)
        print(ds, len(rows), "passages with a decisive edit", kinds)


if __name__ == "__main__":
    main(sys.argv[1:] or ["hotpotqa", "musique"])
