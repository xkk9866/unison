"""Reuse coverage x error by relation class, from the per-call records of a run and its re-execution reference.

For every verification call served by reuse (not computed by the engine for this call), the relation between
the consumer and the computation it reused is classified:
  identical     byte-identical prompt (identity sharing; exact by determinism)
  teammate      another agent of the same team (same question, different decomposition strategy) -- a different
                sub-question wording on the same passage version
  paraphrase    an earlier team with the same underlying question (paraphrased), same passage version
  other         a different question, same passage version
  cross-version the reused computation read another version of the passage
The error of a reuse is a decision (relevance, normalized answer) that differs from an independent re-execution
of the consumer's own prompt. Semantic caches do not record the source call; their reuses are classified by
version only ('estimated' class), which is what their key (the sub-question) can see.
usage: python reuse_classes.py <results_dir> <tag> <ds> <tokenizer path> <paper_dir>"""
import gzip
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from unison.common import decision_equiv, parse_note

R, TAG, DS, TOK, P = sys.argv[1:6]
U = os.environ.get("UNISON_U", "u17")  # the Unison method name of the campaign
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(TOK)
dec = lambda ids: tok.decode(ids, skip_special_tokens=True)


def load(m):
    with gzip.open(os.path.join(R, TAG, DS, f"{m}.calls.jsonl.gz"), "rt") as f:
        return [json.loads(l) for l in f]


ref = json.load(gzip.open(os.path.join(R, TAG, DS, "replay.json.gz"), "rt"))
qids = {q["qi"]: q["qid"] for q in json.load(open(os.path.join(R, TAG, DS, f"{U}.queries.json")))}
rows = {}
for m in ("gptcache", "vcache-official", U):
    p = os.path.join(R, TAG, DS, f"{m}.calls.jsonl.gz")
    if not os.path.exists(p):
        continue
    recs = load(m)
    by_rid = {r["rid"]: r for r in recs if r.get("rid")}
    ver = [r for r in recs if r["kind"] == "verify"]
    n_total = len(ver)
    cls = {}
    for r in ver:
        if r["source"] == "engine":
            continue
        ok = decision_equiv(parse_note(dec(r["out"])), parse_note(dec(ref[r["phash"]])))
        if r["source"] == "exact":
            c = "identical"
        elif r["source"] == "dcert":
            s = by_rid.get(r.get("src"))
            if r.get("src_version", r["version"]) != r["version"]:
                c = "cross-version"
            elif s is None:
                c = "other"
            elif s["team"] == r["team"]:
                c = "teammate"
            elif qids.get(s["team"]) == qids.get(r["team"]):
                c = "paraphrase"
            else:
                c = "other"
        else:  # sem: the cache cannot see the relation; it sees a similar sub-question on the same passage
            c = "cross-version" if r.get("src_version", r["version"]) != r["version"] else "estimated"
        d = cls.setdefault(c, [0, 0])
        d[0] += 1
        d[1] += not ok
    rows[m] = (n_total, cls)

order = ["identical", "teammate", "paraphrase", "other", "cross-version", "estimated"]
names = {U: r"\textbf{Unison}", "gptcache": "GPTCache", "vcache-official": "vCache"}
with open(os.path.join(P, f"tab_reuse_{DS}.tex"), "w") as fo:
    for m, (n, cls) in rows.items():
        cells = []
        for c in order:
            k, e = cls.get(c, (0, 0))
            cells.append(f"{k} / {e}" if k else "0 / 0")
        tot = sum(v[0] for v in cls.values())
        err = sum(v[1] for v in cls.values())
        fo.write(f"{names[m]} & {n} & " + " & ".join(cells) + f" & {100 * tot / n:.0f}\\% & {err} \\\\\n")
macros = []
k = "Ho" if DS == "hotpotqa" else "Mu"
for m, (n, cls) in rows.items():
    key = {U: "Unison", "gptcache": "Gptcache", "vcache-official": "Vcache"}[m]
    for c in order:
        kk, e = cls.get(c, (0, 0))
        cn = c.replace("-", "").capitalize()
        macros.append((f"{k}Reuse{key}{cn}", str(kk)))
        macros.append((f"{k}ReuseErr{key}{cn}", str(e)))
    tot = sum(v[0] for v in cls.values())
    macros.append((f"{k}ReuseCov{key}", f"{100 * tot / n:.0f}"))
    non = sum(v[0] for c, v in cls.items() if c != "identical")
    nerr = sum(v[1] for c, v in cls.items() if c != "identical")
    macros.append((f"{k}ReuseNonId{key}", str(non)))
    macros.append((f"{k}ReuseNonIdErr{key}", str(nerr)))
    macros.append((f"{k}ReuseNonIdCov{key}", f"{100 * non / n:.0f}"))
with open(os.path.join(P, f"numbers_reuse_{DS}.tex"), "w") as fo:
    for a, b in macros:
        fo.write(f"\\providecommand{{\\{a}}}{{{b}}}\n")
for m, (n, cls) in rows.items():
    print(m, n, {c: tuple(v) for c, v in cls.items()})
