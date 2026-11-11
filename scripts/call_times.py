"""Where the time of a call goes, per method of a run: time to the first token and time after it, by call kind and by
decision; what a verification round and an agent's critical path are made of. Reads <run>/<method>.calls.jsonl.gz and
<run>/<method>.esteps.npy and writes <run>/calltimes.json (fetched with the summaries, read by analyze_v17.py).

usage: python call_times.py <run dir> [<run dir> ...]
All times are means in milliseconds over the calls the engine served (identity-served and cache-served calls take no
pass). A pass is one engine step in which the call takes part; the first pass processes the prompt.
"""
import glob
import gzip
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from unison.common import MODELS, parse_note  # noqa: E402

_TOK = {}


def tokenizer():
    if "t" not in _TOK:
        from transformers import AutoTokenizer
        _TOK["t"] = AutoTokenizer.from_pretrained(MODELS[os.environ.get("UNISON_MODEL", "qwen3-8b")])
    return _TOK["t"]


def ms(x):
    return round(1e3 * float(np.mean(x)), 2) if len(x) else None


def call_stats(rs, E):
    """rs: engine-served calls of one kind. E: the engine's step log or None."""
    if not rs:
        return dict(n=0)
    sub = np.array([r.get("t_submit", r["t_arrive"]) for r in rs])
    arr = np.array([r["t_arrive"] for r in rs])
    first = np.array([r["t_first"] for r in rs])
    end = np.array([r["t_end"] for r in rs])
    st = np.array([r.get("steps", 0) for r in rs], dtype=float)
    out = dict(n=len(rs), queue_ms=ms(sub - arr), ttft_ms=ms(first - sub), after_ms=ms(end - first),
               total_ms=ms(end - arr), passes=round(float(st.mean()), 3), one_pass=round(float((st == 1).mean()), 4),
               tokens=round(float(np.mean([r.get("n_out", len(r.get("out", ()))) for r in rs])), 2))
    if st.max() > 1:  # what one pass after the first costs a call
        later = st > 1
        out["later_pass_ms"] = round(1e3 * float((end - first)[later].sum() / (st[later] - 1).sum()), 2)
    if E is not None:
        s_end, run, pre = E[:, 0], E[:, 1], E[:, 6]
        s_start, dur = s_end - run - pre, run + pre
        i = np.clip(np.searchsorted(s_end, first, side="right") - 1, 0, len(s_end) - 1)  # the step of the first token
        out.update(wait_ms=ms(np.maximum(0.0, s_start[i] - sub)), first_step_ms=ms(dur[i]),
                   first_step_positions=round(float(E[i, 3].mean()), 1), first_step_drafts=round(float(E[i, 4].mean()), 1),
                   first_step_admitted=round(float(E[i, 5].mean()), 2))
    return out


def one(run, m):
    f = os.path.join(run, f"{m}.calls.jsonl.gz")
    R = [json.loads(l) for l in gzip.open(f, "rt")]
    e = os.path.join(run, f"{m}.esteps.npy")
    E = np.load(e) if os.path.exists(e) else None
    if E is not None and (E.ndim != 2 or E.shape[1] < 8 or not len(E)):
        E = None
    served = [r for r in R if r.get("t_first") and r.get("source") not in ("exact", "sem")]
    plans = [r for r in served if r["kind"] == "plan"]
    ver = [r for r in served if r["kind"] == "verify"]
    out = dict(plan=call_stats(plans, E), verify=call_stats(ver, E))
    # verification calls by decision
    allv = [r for r in R if r["kind"] == "verify" and r.get("out")]
    txt = tokenizer().batch_decode([r["out"] for r in allv]) if allv else []
    pos = {id(r): bool(parse_note(t)[0]) for r, t in zip(allv, txt)}
    vp = [r for r in ver if pos.get(id(r))]
    vn = [r for r in ver if id(r) in pos and not pos[id(r)]]
    out["verify_pos"], out["verify_neg"] = call_stats(vp, E), call_stats(vn, E)
    out["pos_share"] = round(len(vp) / max(1, len(vp) + len(vn)), 4)
    # an agent's critical path: plan calls, verification rounds, the rest (retrieval, scheduling)
    ag = defaultdict(list)
    for r in R:
        if r["kind"] in ("plan", "verify"):
            ag[(r.get("team"), r.get("agent"))].append(r)
    span, plan_t, round_t, n_plan, n_round = [], [], [], [], []
    rd = {True: [], False: []}
    for rs in ag.values():
        p = sorted((r for r in rs if r["kind"] == "plan"), key=lambda r: r["t_arrive"])
        if not p:
            continue
        v = defaultdict(list)
        for r in rs:
            if r["kind"] == "verify":
                v[r.get("hop")].append(r)
        rt = 0.0
        for vs in v.values():
            a = min(r["t_arrive"] for r in vs)
            nxt = [q["t_arrive"] for q in p if q["t_arrive"] >= a]
            b = min(nxt) if nxt else max(r["t_end"] for r in vs)  # the round ends when the agent plans again
            rt += b - a
            if nxt:
                rd[any(pos.get(id(r), False) for r in vs)].append(b - a)
        span.append(max(r["t_end"] for r in p) - min(r["t_arrive"] for r in rs))
        plan_t.append(sum(r["t_end"] - r["t_arrive"] for r in p))
        round_t.append(rt)
        n_plan.append(len(p))
        n_round.append(len(v))
    out["agent"] = dict(n=len(span), span_ms=ms(span), plan_ms=ms(plan_t), round_ms=ms(round_t),
                        other_ms=ms(np.array(span) - np.array(plan_t) - np.array(round_t)) if span else None,
                        plans=round(float(np.mean(n_plan)), 2) if n_plan else 0,
                        rounds=round(float(np.mean(n_round)), 2) if n_round else 0)
    out["round"] = dict(pos_n=len(rd[True]), pos_ms=ms(rd[True]), neg_n=len(rd[False]), neg_ms=ms(rd[False]))
    # the two cache services: what a lookup costs a call before the engine is asked (or before the hit returns)
    cl = [r for r in R if r["kind"] == "verify" and "c_look" in r]
    if cl:
        hit = [r for r in cl if r.get("source") == "sem"]
        miss = [r for r in cl if r.get("source") != "sem"]
        look = 1e3 * np.array([r["c_look"] for r in cl])
        out["cache"] = dict(n=len(cl), hits=len(hit), emb_ms=ms([r["c_emb"] for r in cl]), look_ms=ms(look / 1e3),
                            look_p50_ms=round(float(np.percentile(look, 50)), 2),
                            look_p99_ms=round(float(np.percentile(look, 99)), 2),
                            hit_ms=ms([r["t_end"] - r["t_arrive"] for r in hit]),
                            hit_look_ms=ms([r["c_look"] for r in hit]), miss_look_ms=ms([r["c_look"] for r in miss]),
                            post_ms=ms([r["c_post"] for r in miss if "c_post" in r]),
                            svc_ms=ms([r["c_svc"] for r in cl if "c_svc" in r]))
    if E is not None:
        run_s, adm, pre = E[:, 1], E[:, 5], E[:, 6]
        a = adm > 0
        dur = run_s + pre
        out["step"] = dict(n=int(len(E)), admit_share=round(float(a.mean()), 4), admit_ms=ms(dur[a]),
                           other_ms=ms(dur[~a]) if (~a).any() else None,
                           busy_share=round(float(dur.sum() / max(1e-9, E[-1, 0] - (E[0, 0] - dur[0]))), 4),
                           weighted_ms=round(1e3 * float((dur ** 2).sum() / dur.sum()), 2))
    return out


def main():
    for run in sys.argv[1:]:
        out = {}
        for f in sorted(glob.glob(os.path.join(run, "*.calls.jsonl.gz"))):
            m = os.path.basename(f)[:-len(".calls.jsonl.gz")]
            try:
                out[m] = one(run, m)
            except Exception as e:  # a method without a usable log is reported, never guessed
                print(f"[call_times] {run}/{m}: {e!r}", flush=True)
        if out:
            with open(os.path.join(run, "calltimes.json"), "w") as fo:
                json.dump(out, fo)
            print(f"[call_times] {run}: {len(out)} methods", flush=True)


if __name__ == "__main__":
    main()
