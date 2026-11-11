"""Sanity check of consumer-conditioned replay: self-replay certifies, cross replay diverges, prefix is cached."""
import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from unison.common import MODELS, verify_messages
from unison.engine import Engine

PASS = ("Marie Curie was a Polish and naturalised-French physicist and chemist who conducted pioneering "
        "research on radioactivity. She was the first woman to win a Nobel Prize, in 1903, and the only "
        "person to win Nobel Prizes in two scientific fields.")


async def main():
    E = Engine(MODELS["qwen3-8b"], gpu_util=0.5)
    await E.init()
    pa = E.chat_ids(verify_messages("Marie Curie", PASS, "When did Marie Curie win her first Nobel Prize?"))
    pb = E.chat_ids(verify_messages("Marie Curie", PASS, "In which year was Curie first awarded a Nobel Prize?"))
    pc = E.chat_ids(verify_messages("Marie Curie", PASS, "Where was Marie Curie born?"))
    _, ya, _ = await E.generate(pa, 128, {"mode": "none"})
    print("ya", repr(E.decode(ya)))
    for name, p in [("self", pa), ("para", pb), ("other", pc)]:
        t = time.time()
        lp, lp1, top1, nxt, tm = await E.replay(p, ya)
        acc = int((top1 == ya).cumprod().sum())
        sig = lp1 - lp
        print(name, "n", len(ya), "accepted", acc, "all", bool((top1 == ya).all()), "sum_sigma", float(sig.sum()),
              "dt", round(time.time() - t, 3))
        _, yy, _ = await E.generate(p, 128, {"mode": "none"})
        lcp = 0
        while lcp < min(len(yy), len(ya)) and yy[lcp] == ya[lcp]:
            lcp += 1
        print("   independent LCP", lcp, "top1@div", int(top1[acc]) if acc < len(ya) else None,
              "indep@div", yy[lcp] if lcp < len(yy) else None, repr(E.decode(yy))[:160])


asyncio.run(main())
