"""Replay-trace dataset from independent-execution logs (deployment-style candidate sets).

Calls are visited in arrival order. For a verification call w whose exact prompt has not been seen
(identical prompts take the certified single-flight path), the candidate set is one representative
output per *decision class* among the earlier calls on the same passage (any version). Each candidate
y_u is replayed under w's own prompt (condition + evidence version) in one teacher-forced pass; the
surprisal trace and the label agree = [decision(y_w) == decision(y_u)] are stored.
Optionally the text-only agreement predictors (AHN, condition cosine) are scored on the same pairs.
"""
import argparse
import asyncio
import gzip
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

from unison.agents import World
from unison.common import AHN_BACKBONE, DATA, MODELS, RESULTS, normalize_answer, parse_note, verify_messages
from unison.engine import Engine
from unison.trace import decision_mask


def decision(tok, ids):
    rel, ans = parse_note(tok.decode(ids, skip_special_tokens=True))
    return rel, normalize_answer(ans)


def load_verify(tag, ds):
    with gzip.open(os.path.join(RESULTS, tag, ds, "noshare.calls.jsonl.gz"), "rt") as f:
        recs = sorted((json.loads(l) for l in f), key=lambda r: r["t_arrive"])
    return [r for r in recs if r["kind"] == "verify"]


async def main(a):
    E = Engine(MODELS[a.model], gpu_util=a.gpu_util)
    await E.init()
    tok = E.tok
    rows = []
    for ds in a.ds.split(","):
        world = World(os.path.join(DATA, ds, a.split))
        recs = load_verify(a.tag, ds)
        hist, seen, jobs = {}, set(), []
        for i, r in enumerate(recs):
            r["dec"] = decision(tok, r["out"])
            r["i"] = i
            if r["phash"] not in seen:
                reps = {}
                for u in hist.get(r["pid"], []):
                    if u["phash"] != r["phash"]:
                        reps[u["dec"]] = u  # latest output of each decision class
                if reps:
                    jobs.append((r, list(reps.values())))
            seen.add(r["phash"])
            hist.setdefault(r["pid"], []).append(r)
        print(ds, "calls", len(recs), "consumers", len(jobs), "pairs", sum(len(c) for _, c in jobs), flush=True)
        sem = asyncio.Semaphore(96)

        async def one(r, cands):
            c = world.corpus[r["pid"]]
            ids = E.chat_ids(verify_messages(c["title"], world.text(r["pid"], r["version"]), r["cond"]))
            assert len(ids) == r["n_prompt"], (len(ids), r["n_prompt"])
            out = []
            async with sem:
                res = await asyncio.gather(*[E.replay(ids, u["out"]) for u in cands])
            for u, (lp, lp1, top1, nxt, _) in zip(cands, res):
                y = np.asarray(u["out"])
                ok = top1 == y
                acc = int(ok.cumprod().sum())
                lcp = 0
                while lcp < min(len(y), len(r["out"])) and y[lcp] == r["out"][lcp]:
                    lcp += 1
                out.append(dict(ds=ds, w=r["i"], u=u["i"], pid=r["pid"], v_w=r["version"], v_u=u["version"],
                                cond_w=r["cond"], cond_u=u["cond"], lp=lp.astype(np.float16),
                                lp1=lp1.astype(np.float16), dmask=decision_mask(tok, u["out"]), n=len(y),
                                acc=acc, lcp=lcp, certified=bool(ok.all()), exact=u["out"] == r["out"],
                                agree=int(u["dec"] == r["dec"]), t_w=r["t_arrive"], t_end_u=u["t_end"]))
            return out

        res = await asyncio.gather(*[one(r, c) for r, c in jobs])
        part = [x for g in res for x in g]
        acc_ok = np.mean([x["acc"] == x["lcp"] for x in part])
        print(ds, "replayed", len(part), "acc==lcp", round(float(acc_ok), 4),
              "agree", round(float(np.mean([x["agree"] for x in part])), 4),
              "certified", round(float(np.mean([x["certified"] for x in part])), 4), flush=True)
        rows.extend(part)
    if a.text_baselines:
        from unison.ahn import AHNScorer, token_pieces, version_delta
        from unison.retriever import Encoder
        ahn = AHNScorer(os.path.join(RESULTS, "ahn", "ahn.pt"), AHN_BACKBONE, tok)
        enc = Encoder()
        worlds = {ds: World(os.path.join(DATA, ds, a.split)) for ds in a.ds.split(",")}
        outs = {ds: [r["out"] for r in load_verify(a.tag, ds)] for ds in a.ds.split(",")}
        for k in range(0, len(rows), 512):
            b = rows[k:k + 512]
            items = [(x["cond_w"], (x["cond_u"], version_delta(worlds[x["ds"]].text(x["pid"], x["v_u"]),
                                                               worlds[x["ds"]].text(x["pid"], x["v_w"])),
                                    token_pieces(tok, outs[x["ds"]][x["u"]]))) for x in b]
            sc = ahn._score_pairs(items)
            ew = enc.encode([x["cond_w"] for x in b])
            eu = enc.encode([x["cond_u"] for x in b])
            for x, (_, q), s in zip(b, sc, (ew * eu).sum(1)):
                x["q_ahn"], x["cos"] = float(q), float(s)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with gzip.open(a.out, "wb") as f:
        pickle.dump(rows, f)
    print("saved", len(rows), a.out, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="ahndata")
    p.add_argument("--split", default="train")
    p.add_argument("--ds", default="hotpotqa,musique")
    p.add_argument("--model", default="qwen3-8b")
    p.add_argument("--out", required=True)
    p.add_argument("--gpu_util", type=float, default=0.6)
    p.add_argument("--text_baselines", type=int, default=0)
    asyncio.run(main(p.parse_args()))
