"""Audit quality of delivered provenance: claim-keyed (question-free) vs per-call (question-conditioned).

Pairs are calls with the *same prompt* in two runs of the same stream (e.g. u14 and u14-sbd of v14_abl), whose
delivered decisions are equal and claim evidence, so the only difference is the provenance program. A judge
from a different model generation (default Qwen2.5-7B) answers three audit questions per note:
  grounded    the justification states only what the passage says
  responsive  it explains why the answer answers *this question*
  accept      an auditor would accept the answer for this question from the passage and justification
plus the mechanical check that the cited sentence of the consumer's own passage version contains the answer.
Output: <results>/<tag>/<ds>/audit.json (rates, paired differences with bootstrap 95% CIs, n).
usage: python audit_eval.py --tag v14_abl --ds musique --a u14 --b u14-sbd [--n 400] [--judge qwen2.5-7b]"""
import argparse
import gzip
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

from unison.agents import World
from unison.common import (DATA, MODELS, RESULTS, normalize_answer, number_sentences, parse_note, parse_support,
                           split_sentences)

Q = ["grounded", "responsive", "accept"]


def load(tag, ds, m):
    with gzip.open(os.path.join(RESULTS, tag, ds, f"{m}.calls.jsonl.gz"), "rt") as f:
        return [json.loads(l) for l in f]


def prompt(title, text, q, ans, k, sent, prov):
    user = (f"Passage (title: {title}):\n{number_sentences(text)}\n\nQuestion: {q}\n\n"
            f"A verification note for this question claims:\nAnswer: {ans}\n"
            f"Supporting sentence [{k}]: {sent}\nJustification: {prov}\n\n"
            "Audit the note with three questions.\n"
            "Grounded: does the justification state only what the passage says?\n"
            "Responsive: does the justification explain why this answer answers the question?\n"
            "Accept: would you accept this answer for this question from the passage and the justification?\n"
            "Reply with exactly these three lines and nothing else, each completed with yes or no:\n"
            "Grounded:\nResponsive:\nAccept:")
    return [{"role": "system", "content": "You are a strict auditor of evidence citations."},
            {"role": "user", "content": user}]


def fields(t):
    """The three yes/no answers of a reply, or None if the reply does not contain them. A field is read by its
    label, which may be wrapped in markdown or brackets ("**Grounded:** <yes>"). A judge that echoes a generic
    label instead ("Label: yes" on each line) has still answered the questions in the order they were asked:
    a reply with exactly three yes/no values is read in that order. (The first version of this script read
    labels only and counted such replies as three times no, which is a parsing error, not a judgment.)"""
    lab = [re.search(rf"(?i){q}\W{{0,4}}\s*[<\[\(]?\s*\**\s*(yes|no)\b", t) for q in Q]
    if all(lab):
        return [m.group(1).lower() == "yes" for m in lab]
    vals = re.findall(r"(?i)\b(yes|no)\b", t)
    if len(vals) == 3:
        return [v.lower() == "yes" for v in vals]
    return None


def parse(t):
    """A reply without the three answers counts as three times no and is reported separately (unparsed rate)."""
    return fields(t) or [False, False, False]


def parsed(t):
    return fields(t) is not None


def prov_text(r):
    """The delivered payload as the auditor sees it: field labels removed, all lines kept (wide payloads span
    an Evidence line and a multi-sentence Reasoning line)."""
    lines = [l.strip() for l in r["prov"].strip().split("\n") if l.strip()]
    out = []
    for l in lines:
        l = re.sub(r"(?i)^(evidence|reasoning)\s*:\s*", "", l)
        if l:
            out.append(l)
    return " ".join(out)


def boot(d, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    d = np.asarray(d, dtype=float)
    s = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(B)]
    return float(d.mean()), float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def main(a):
    summ = json.load(open(os.path.join(RESULTS, a.tag, a.ds, "summary.json")))
    cfg = summ[0]["cfg"]
    world = World(os.path.join(DATA, a.ds, cfg.get("split", "eval")), cfg.get("versions", "versions.jsonl"))
    A = {r["phash"]: r for r in load(a.tag, a.ds, a.a) if r["kind"] == "verify" and r.get("prov")}
    B = {r["phash"]: r for r in load(a.tag, a.ds, a.b) if r["kind"] == "verify" and r.get("prov")}
    from transformers import AutoTokenizer
    gen_tok = AutoTokenizer.from_pretrained(MODELS[cfg.get('model', 'qwen3-8b')])  # tokenizer of the generator
    pairs = []
    for h in sorted(set(A) & set(B)):
        ra, rb = A[h], B[h]
        if ra.get("source") == "exact" or rb.get("source") == "exact":
            continue
        ta = gen_tok.decode(ra["out"], skip_special_tokens=True)
        tb = gen_tok.decode(rb["out"], skip_special_tokens=True)
        rela, ansa = parse_note(ta)
        relb, ansb = parse_note(tb)
        if not (rela and relb and normalize_answer(ansa) == normalize_answer(ansb)
                and parse_support(ta) == parse_support(tb)):
            continue
        pairs.append((ra, rb))
    random.Random(0).shuffle(pairs)
    pairs = pairs[:a.n]
    if not pairs:
        raise RuntimeError("No matched evidence claims with provenance in both runs")
    from vllm import LLM, SamplingParams
    llm = LLM(MODELS[a.judge], gpu_memory_utilization=a.gpu_util, max_model_len=4096, seed=0)
    msgs, meta = [], []
    for ra, rb in pairs:
        for side, r in (("a", ra), ("b", rb)):
            head = gen_tok.decode(r["out"], skip_special_tokens=True)
            rel, ans = parse_note(head)
            k = parse_support(head)
            text = world.text(r["pid"], r["version"])
            sents = split_sentences(text)
            sent = sents[k - 1] if 0 < k <= len(sents) else "none"
            prov = prov_text(r)
            msgs.append(prompt(world.corpus[r["pid"]]["title"], text, r["cond"], ans, k, sent, prov))
            meta.append((side, 0 < k <= len(sents) and normalize_answer(ans) in normalize_answer(sent), len(prov)))
    outs = llm.chat(msgs, SamplingParams(temperature=0.0, max_tokens=48))
    res = {"a": [], "b": []}
    unparsed = {"a": 0, "b": 0}
    examples = []
    raw = os.path.join(RESULTS, a.tag, a.ds, f"audit_{a.a}_vs_{a.b}.raw.jsonl")
    with open(raw, "w") as fr:  # every reply of the judge is kept, so the scores can be recomputed from the replies
        for (side, mech, plen), o in zip(meta, outs):
            txt = o.outputs[0].text
            res[side].append(parse(txt) + [mech, plen])
            unparsed[side] += not parsed(txt)
            fr.write(json.dumps(dict(side=side, reply=txt, cited_contains_answer=bool(mech), prov_len=plen)) + "\n")
            if len(examples) < 6:
                examples.append(txt)
    ra, rb = np.array(res["a"], dtype=float), np.array(res["b"], dtype=float)
    rep = dict(tag=a.tag, ds=a.ds, a=a.a, b=a.b, judge=a.judge, n_pairs=len(pairs),
               unparsed_a=unparsed["a"] / max(1, len(pairs)), unparsed_b=unparsed["b"] / max(1, len(pairs)),
               judge_examples=examples)
    for j, q in enumerate(Q + ["cited_contains_answer"]):
        rep[f"{q}_a"], rep[f"{q}_b"] = float(ra[:, j].mean()), float(rb[:, j].mean())
        rep[f"{q}_diff"] = boot(ra[:, j] - rb[:, j])
    rep["prov_len_a"], rep["prov_len_b"] = float(ra[:, -1].mean()), float(rb[:, -1].mean())
    out = os.path.join(RESULTS, a.tag, a.ds, f"audit_{a.a}_vs_{a.b}.json")
    json.dump(rep, open(out, "w"), indent=1)
    print(json.dumps(rep, indent=1), flush=True)
    # the result is on disk; vLLM's engine teardown can hang and would block the queue, so end the engine's
    # worker processes explicitly and exit without the interpreter's teardown
    try:
        import psutil
        for c in psutil.Process().children(recursive=True):
            c.kill()
    except Exception:
        pass
    os._exit(0)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="v14_abl")
    p.add_argument("--ds", default="musique")
    p.add_argument("--a", default="u14")
    p.add_argument("--b", default="u14-sbd")
    p.add_argument("--n", type=int, default=400)
    p.add_argument("--judge", default="qwen2.5-7b")
    p.add_argument("--gpu_util", type=float, default=0.85)
    main(p.parse_args())
