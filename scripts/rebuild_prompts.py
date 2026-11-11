"""Recover the prompt of every call of a collection run, without the engine.

The agent program is a deterministic function of the outputs it receives. This script runs the program of the
collection run again (same stream, same retriever) and answers every call with the output the run recorded for it
(looked up by agent, hop and passage). A recovered prompt is exact when its hash equals the recorded prompt hash;
the few calls that read an edited passage version or whose retrieval differs are marked inexact.

  python scripts/rebuild_prompts.py v17_collect/hotpotqa      # writes <run>/collect.prompts.npz
"""
import argparse
import asyncio
import gzip
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

from unison.agents import AgentRuntime, World, make_stream
from unison.common import DATA, MODELS, RESULTS, read_jsonl
from unison.executors import phash
from unison.retriever import Encoder, Retriever


class Tok:
    """Tokenizer-only stand-in for the engine (the program needs chat templates and decoding, nothing else)."""

    def __init__(self, path):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(path)

    def chat_ids(self, messages):
        return self.tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)

    def decode(self, toks):
        return self.tok.decode(toks, skip_special_tokens=True)


class Diverged(Exception):
    pass


class Recorded:
    """Executor that returns the recorded output of each call and keeps the prompt the program built for it."""
    dataflow_boundaries = False

    def __init__(self, calls, fmt):
        self.note_format = fmt
        self.by = {(r["agent"], r["hop"], r["kind"], r["group"] if r["kind"] == "verify" else ""): r for r in calls}
        self.prompts, self.exact, self.miss = {}, {}, 0

    def _answer(self, key, ids):
        r = self.by.get(key)
        if r is None:
            self.miss += 1
            raise Diverged(key)
        self.prompts[r["rid"]] = np.asarray(ids, dtype=np.int32)
        self.exact[r["rid"]] = phash(ids) == r["phash"]
        return r["out"]

    async def plain(self, info, ids, max_tokens):
        return self._answer((info["agent"], info["hop"], "plan", ""), ids)

    async def verify(self, info, ids, max_tokens):
        return self._answer((info["agent"], info["hop"], "verify", f"p{info['pid']}"), ids)


async def main(a):
    run = os.path.join(RESULTS, a.run)
    with gzip.open(os.path.join(run, "collect.calls.jsonl.gz"), "rt") as f:
        calls = [json.loads(l) for l in f]
    cfg = json.load(open(os.path.join(run, "summary.json")))[0]["cfg"]
    pool = os.path.join(DATA, cfg["ds"], cfg["split"])
    E = Tok(MODELS[cfg["model"]])
    enc = Encoder(device=a.device)
    retr = Retriever(pool, enc)
    world = World(pool, cfg["versions"])
    qs = read_jsonl(os.path.join(pool, "questions.jsonl"))
    paras = read_jsonl(os.path.join(pool, "paraphrases.jsonl"))
    stream = make_stream(qs, paras, cfg["n"], cfg["rate"], cfg["zipf"], cfg["seed"], cfg["arrival"], None, cfg["qsplit"])
    X = Recorded(calls, cfg["note"])
    rt = AgentRuntime(E, X, world, retr, k=cfg["k"], max_hops=3 if cfg["ds"] == "hotpotqa" else 4, team=cfg["team"])
    diverged = 0

    async def one(qi, ag, item):
        nonlocal diverged
        try:
            await rt.agent(qi, ag, item["question"], item.get("qid"))
        except Diverged:
            diverged += 1

    await asyncio.gather(*[one(qi, ag, item) for qi, item in enumerate(stream) for ag in range(cfg["team"])])
    rids = [r["rid"] for r in calls if r["rid"] in X.prompts]
    off = np.zeros(len(rids) + 1, dtype=np.int64)
    off[1:] = np.cumsum([len(X.prompts[r]) for r in rids])
    np.savez_compressed(os.path.join(run, "collect.prompts.npz"), rids=np.array(rids), off=off,
                        ids=np.concatenate([X.prompts[r] for r in rids]) if rids else np.zeros(0, np.int32),
                        exact=np.array([X.exact[r] for r in rids]))
    kinds = {r["rid"]: r["kind"] for r in calls}
    stats = dict(calls=len(calls), recovered=len(rids), exact=int(sum(X.exact[r] for r in rids)),
                 agents_diverged=diverged, lookups_missed=X.miss)
    for k in ("plan", "verify"):
        rr = [r for r in rids if kinds[r] == k]
        stats[k] = dict(calls=sum(1 for r in calls if r["kind"] == k), recovered=len(rr),
                        exact=int(sum(X.exact[r] for r in rr)),
                        prompt_tokens_mean=float(np.mean([len(X.prompts[r]) for r in rr])) if rr else 0.0,
                        prompt_tokens_p95=float(np.percentile([len(X.prompts[r]) for r in rr], 95)) if rr else 0.0)
    print("PROMPTS", json.dumps(stats), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("--device", default="cuda")
    asyncio.run(main(p.parse_args()))
