"""Micro-benchmark: cost of a replay request vs. an equivalent plain 1-token request under decode load."""
import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

from unison.common import MODELS, verify_messages
from unison.engine import Engine, StepLog
from unison.agents import World
from unison.common import DATA


async def main():
    E = Engine(MODELS["qwen3-8b"])
    await E.init()
    w = World(os.path.join(DATA, "hotpotqa", "eval"))
    rng = np.random.default_rng(0)
    pids = rng.choice(len(w.corpus), 64, replace=False)
    base = [E.chat_ids(verify_messages(w.corpus[p]["title"], w.corpus[p]["text"], "Who founded it?")) for p in pids]
    outs = {}
    for i, ids in enumerate(base):  # warm the evidence prefixes and produce candidate outputs
        _, outs[i], _ = await E.generate(ids, 64, {"mode": "none"})
    for kind in ["none", "plain", "replay", "replay_nocache"]:
        StepLog.records.clear()
        bg = [E.generate(base[i % 64] + [100 + i], 256, {"mode": "none"}) for i in range(48)]
        t0 = time.time()

        async def side():
            await asyncio.sleep(0.5)
            lat = []
            for j in range(40):
                i = j % 64
                ids = E.chat_ids(verify_messages(w.corpus[pids[i]]["title"], w.corpus[pids[i]]["text"],
                                                 f"When was it established? ({j})"))
                t = time.time()
                if kind == "plain":
                    await E.generate(ids + outs[i], 1, {"mode": "none"})
                elif kind == "replay":
                    await E.replay(ids, outs[i])
                elif kind == "replay_nocache":
                    await E.replay([1] * 8 + ids, outs[i])
                lat.append(time.time() - t)
                await asyncio.sleep(0.05)
            return lat

        res = await asyncio.gather(side(), *bg)
        rec = StepLog.records
        dts = np.diff([r[0] for r in rec])
        print(kind, "side_lat_ms", round(1000 * float(np.mean(res[0])), 1) if res[0] else None,
              "step_ms mean", round(1000 * float(np.mean(dts)), 2), "p90", round(1000 * float(np.percentile(dts, 90)), 2),
              "iters", len(rec), "wall", round(time.time() - t0, 2), flush=True)
        await E.reset()
        for i, ids in enumerate(base):
            await E.generate(ids, 1, {"mode": "none"})


asyncio.run(main())
