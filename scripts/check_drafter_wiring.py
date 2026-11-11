"""Engine-side wiring of the consensus drafter against the offline pipeline.

The engine plugin's draft routines are given the state a request has inside the engine (the hidden-state row of its
last token, the live outputs of its group, its prompt in the engine's token buffer) for windows of a collection
run, and must return the drafts that train_drafter.py's evaluation path produces for the same windows: _nd_batch
for requests that have made a pass, _nd_drafts without hidden states for requests that are being admitted (the
pre-draft). The engine path runs the captured program, the offline path the same loop without a graph. Also times
the routine at several batch sizes (the drafter's cost per engine iteration).

  python scripts/check_drafter_wiring.py v17_collect/hotpotqa [--drafter results/drafter/cd.pt]
"""
import argparse
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch

import train_drafter as td
import unison_plugin.unison_vllm_patch as pl
from unison.common import MODELS, RESULTS


class Obj:
    pass


LEX = {}


def ea_of(r, group, eos, price, run):
    # a + delta B = B and ctok = price: the verification price is `price` at any load
    return dict(mode="nd", group=group, cost=dict(a=0.0, delta=1.0, ctok=price), k=td.K, eos=eos, run=run,
                n_nl=3 if r["kind"] == "verify" else 0)


def engine_state(W, calls, H, eos, price, run):
    """A fake model runner holding the windows `W` as one engine batch, the plugin's draft items for the windows
    of requests that have made a pass and its entries for the windows of requests being admitted."""
    store = pl._store(run)
    runner, ib = Obj(), Obj()
    runner.model = Obj()
    runner.model.lm_head = Obj()
    runner.model.lm_head.weight = LEX["table"]  # the engine exposes its output head; here: the checkpoint's
    ib.req_ids = [f"{run}-r{b}" for b in range(len(W))]
    width = max(len(calls[rid]["prompt"]) for rid, _, _ in W) + td.K + 64
    ib.token_ids_cpu = np.zeros((len(W), width), dtype=np.int32)
    runner.input_batch, runner._unison_nd = ib, {}
    runner._unison_h = torch.stack([H[rid][max(t, 0)] for rid, t, _ in W]).cuda()
    items, ents = [], []
    for b, (rid, t, peers) in enumerate(W):
        r, req = calls[rid], ib.req_ids[b]
        p = r["prompt"]
        ib.token_ids_cpu[b, :len(p)] = p
        st = Obj()
        st.num_prompt_tokens = len(p)
        g = store.groups.setdefault(f"g{b}", [])
        for j, pp in enumerate(reversed(peers)):  # the group list is in arrival order, the window's peers newest first
            store.live[f"{run}-p{b}_{j}"] = np.asarray(pp, dtype=np.int64)
            g.append(f"{run}-p{b}_{j}")
        ea = ea_of(r, f"g{b}", eos, price, run)
        if t == td.PRE:  # admitted in this step: no output, no state, not yet a member of its group
            ents.append((req, ea, store, int(p[-1]), -1, pl._tail(p, td.C, True)))
        else:
            store.live[req] = np.asarray(r["key"][:t + 1], dtype=np.int64)
            g.append(req)
            items.append((b, req, st, ea, 1))
    return runner, items, ents


def main(a):
    from transformers import AutoTokenizer
    os.environ["UNISON_DRAFTER"] = a.drafter
    tok = AutoTokenizer.from_pretrained(MODELS[a.model])
    eos = sorted({tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")} - {None})
    calls, H, groups = td.load_run(os.path.join(RESULTS, a.run), tok, a.note, eos)
    rng = random.Random(0)
    rids = rng.sample(sorted(H), min(len(H), 400))
    W = td.windows(calls, H, groups, rids)
    W = rng.sample(W, min(len(W), a.n)) + rng.sample([w for w in W if w[1] == td.PRE], min(len(rids), a.n // 4))
    from unison.drafter import load_lexicon
    LEX["table"] = load_lexicon(MODELS[a.model], "cuda")
    net = pl._drafter(engine_state(W[:1], calls, H, eos, a.price, "init")[0])
    pk = td.Pack(W, calls, H, "cuda")
    h, y, ctx, pe, _, has = pk.batch(torch.arange(len(W)))
    off = net.propose(h, y, ctx, pe, td.K, a.price, eos, has_h=has)
    runner, items, ents = engine_state(W, calls, H, eos, a.price, "wiring")
    pl._ND["t"] = [0.0, 0]  # no step price: the offline path has none
    on = pl._nd_batch(runner, list(items), td.K)
    pl._ND["t"] = [0.0, 0]
    on.update(pl._nd_drafts(net, None, ents, len(W), td.K))
    pl._ND["t"] = [0.0, 0]
    same = {False: 0, True: 0}
    for b, (rid, t, _) in enumerate(W):
        d = off[b]
        if calls[rid]["kind"] == "verify" and d and d[-1] in eos:
            d = d[:-1]
        same[t == td.PRE] += d == on.get(runner.input_batch.req_ids[b], [])
    n_pre = sum(t == td.PRE for _, t, _ in W)
    print(f"WIRING {same[False]} of {len(W) - n_pre} drafts after a pass and {same[True]} of {n_pre} pre-drafts "
          f"identical; drafted tokens offline {sum(map(len, off))}, engine path {sum(map(len, on.values()))}", flush=True)
    # cost per engine iteration: a batch of B requests whose prompt memories are cached (every iteration but a
    # request's first) and a batch of B new requests
    Wp = [w for w in W if w[1] != td.PRE]
    for B in (4, 8, 16, 32, 64):
        ws = [rng.sample(Wp, B) for _ in range(12)]
        t_new, t_hot = [], []
        pl._ND["t"] = [0.0, 0]
        for i, w in enumerate(ws):
            runner, items, _ = engine_state(w, calls, H, eos, a.price, f"cost{B}_{i}")
            for tl in (t_new, t_hot):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                pl._nd_batch(runner, list(items), td.K)
                tl.append(time.perf_counter() - t0)
        print(f"COST B={B}: first iteration of a request {1e3 * np.median(t_new[2:]):.2f} ms, later iterations "
              f"{1e3 * np.median(t_hot[2:]):.2f} ms, positions decoded per call {pl._ND['t'][1] / 24:.1f}, "
              f"time per decoded position {1e3 * pl._ND['t'][0] / max(1, pl._ND['t'][1]):.3f} ms", flush=True)
    pl._ND["t"] = [0.0, 0]
    ok = sum(same.values()) >= 0.99 * len(W)  # float16 kernels are not bit-identical across batch layouts
    print("WIRING_OK" if ok else "WIRING_BAD", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("--drafter", default=os.path.join(RESULTS, "drafter", "cd.pt"))
    p.add_argument("--model", default="qwen3-8b")
    p.add_argument("--note", default="dfp")
    p.add_argument("--price", type=float, default=0.05)
    p.add_argument("--n", type=int, default=384)
    sys.exit(main(p.parse_args()))
