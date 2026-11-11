"""Concurrent multi-hop research agents, versioned evidence store and request streams."""
import asyncio
import os
import random
import time
from collections import Counter, deque

import numpy as np

from .common import (em_f1, normalize_answer, parse_note, parse_plan, plan_messages, read_jsonl,
                     verify_messages)


class World:
    """Versioned corpus with snapshot reads. Version v of passage p becomes visible at t_from."""

    def __init__(self, pool_dir, versions="versions.jsonl"):
        self.corpus = read_jsonl(os.path.join(pool_dir, "corpus.jsonl"))
        self.vtexts = {r["pid"]: r["versions"] for r in read_jsonl(os.path.join(pool_dir, versions))}
        self.reset()

    def reset(self):
        self.timeline = {}  # pid -> [(t_from, v)]
        self.recent = deque(maxlen=4000)
        self.n_updates = 0

    def version_at(self, pid, t):
        tl = self.timeline.get(pid)
        if not tl:
            return 0
        v = 0
        for t_from, ver in tl:
            if t_from <= t:
                v = ver
        return v

    def text(self, pid, v):
        return self.vtexts[pid][v] if v else self.corpus[pid]["text"]

    def publish_update(self, rng):
        """Edit a recently accessed (hot) passage: hot pages are the ones that get edited."""
        cands = [p for p in set(self.recent) if p in self.vtexts]
        rng.shuffle(cands)
        for p in cands:
            cur = self.version_at(p, time.time())
            if cur + 1 < len(self.vtexts[p]):
                self.timeline.setdefault(p, []).append((time.time(), cur + 1))
                self.n_updates += 1
                return p
        return None


def make_stream(questions, paraphrases, n, rate, zipf_s, seed, arrival="poisson", trace=None, qsplit="all"):
    if qsplit != "all":  # disjoint halves of one workload: shadow calibration vs. test
        import zlib
        want = 0 if qsplit == "calib" else 1
        questions = [q for q in questions if zlib.crc32(q["qid"].encode()) % 2 == want]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(questions))
    w = (np.arange(1, len(questions) + 1, dtype=np.float64)) ** (-zipf_s)
    draws = rng.choice(len(questions), size=n, p=w / w.sum())
    para = {r["qid"]: r["paraphrases"] for r in paraphrases}
    seen = Counter()
    if arrival == "poisson":
        gaps = rng.exponential(1.0 / rate, size=n)
    else:  # replay inter-arrival gaps of a public trace, rescaled to the target rate
        g = np.diff(np.asarray(trace, dtype=np.float64))
        g = g[g > 0]
        start = int(rng.integers(0, max(1, len(g) - n)))
        g = g[start:start + n]
        gaps = g / g.mean() / rate
    t = np.cumsum(gaps) - gaps[0]
    stream = []
    for i, d in enumerate(draws):
        q = questions[perm[d]]
        variants = [q["question"]] + para.get(q["qid"], [])
        text = variants[seen[q["qid"]] % len(variants)]
        seen[q["qid"]] += 1
        stream.append(dict(t=float(t[i]), qid=q["qid"], question=text, answers=q["answers"]))
    return stream


def team_answer(answers):
    """The program's aggregation: the most frequent normalized answer, ties broken by agent order."""
    cnt = Counter(normalize_answer(x) for x in answers)
    return max(answers, key=lambda x: (cnt[normalize_answer(x)], -answers.index(x)))


def majority_determined(done, team):
    """Whether the team answer is already fixed by the agents that have finished.

    `done` maps agent index -> raw answer. The unfinished agents may still return any answer: one of the answers
    seen so far, a fresh answer shared among them, or fresh answers distinct from each other. The aggregation is
    evaluated under every such completion; the team answer is determined iff all completions agree. Returns the
    determined raw answer or None. (The enumeration is over at most (|seen|+2)^|unfinished| cases; teams are small.)"""
    import itertools
    unfinished = [a for a in range(team) if a not in done]
    if not unfinished:
        return team_answer([done[a] for a in range(team)])
    seen = sorted({normalize_answer(x) for x in done.values()})
    # fresh answers: shared by a subset of the unfinished agents (two independent groups suffice for the
    # team sizes used here) or unique to one agent
    options = seen + ["\x00shared", "\x00shared2", "\x00own"]
    winner = None
    for combo in itertools.product(options, repeat=len(unfinished)):
        full = [None] * team
        for a, x in done.items():
            full[a] = normalize_answer(x)
        for a, v in zip(unfinished, combo):
            full[a] = f"\x00own{a}" if v == "\x00own" else v
        cnt = Counter(full)
        w = full[max(range(team), key=lambda i: (cnt[full[i]], -i))]
        if winner is None:
            winner = w
        elif w != winner:
            return None
    if winner is None or winner.startswith("\x00"):
        return None
    for a in sorted(done):  # the raw string of the determined answer (same normalization, hence same EM/F1)
        if normalize_answer(done[a]) == winner:
            return done[a]
    return None


class AgentRuntime:
    """Executes the agent program. With `dataflow_boundaries` (an attribute of the executor), every operator
    returns at its consumption boundary: a verification round returns as soon as the first relevant note in
    rank order is known, and a team returns as soon as its majority answer is determined. Unconsumed work is
    not cancelled: it completes off the critical path (it feeds peers through the executor's sharing
    structures) and is drained before the run ends. Delivered answers are identical to the plain program's."""

    def __init__(self, engine, executor, world, retriever, k=4, max_hops=3, team=3):
        self.E, self.X, self.W, self.R = engine, executor, world, retriever
        self.k, self.max_hops, self.team = k, max_hops, team
        self.df = bool(getattr(executor, "dataflow_boundaries", False))
        self.pending = []      # unconsumed verification calls still running
        self.n_round_early = 0  # rounds that returned before all notes finished
        self.n_team_early = 0   # teams that returned before all agents finished
        self.n_df_mismatch = 0  # safety check: determined team answer vs. full aggregation (must stay 0)

    async def agent(self, qi, a, question, qid=None):
        t_snap = time.time()
        steps = []
        base = dict(team=qi, agent=f"{qi}.{a}", qid=qid)
        for hop in range(self.max_hops + 1):
            force = hop == self.max_hops
            ids = self.E.chat_ids(plan_messages(a, question, steps, force_final=force))
            toks = await self.X.plain(dict(base, kind="plan", hop=hop), ids, 48)
            kind, val = parse_plan(self.E.decode(toks))
            if kind == "final" or force:
                return val, steps
            sub_q = val
            pids = await self.R.asearch(sub_q, self.k)
            self.W.recent.extend(pids)
            jobs = []
            for rank, pid in enumerate(pids):
                v = self.W.version_at(pid, t_snap)
                c = self.W.corpus[pid]
                ids = self.E.chat_ids(verify_messages(c["title"], self.W.text(pid, v), sub_q, getattr(self.X, "note_format", "cot")))
                info = dict(base, kind="verify", hop=hop, pid=pid, version=v, cond=sub_q, rank=rank)
                jobs.append(asyncio.ensure_future(self.X.verify(info, ids, getattr(self.X, "note_max_tokens", 128))))
            ans = "none"
            if self.df:  # round boundary: the agent consumes the first relevant note in rank order
                for j in jobs:
                    rel, a_ = parse_note(self.E.decode(await j))
                    if rel:
                        ans = a_
                        break
                rest = [j for j in jobs if not j.done()]
                if rest:
                    self.n_round_early += 1
                    self.pending.extend(rest)
            else:
                notes = await asyncio.gather(*jobs)
                for n in notes:
                    rel, a_ = parse_note(self.E.decode(n))
                    if rel:
                        ans = a_
                        break
            steps.append((sub_q, ans))
        return "unknown", steps

    async def query(self, qi, item):
        t0 = time.time()
        tasks = [asyncio.ensure_future(self.agent(qi, a, item["question"], item.get("qid"))) for a in range(self.team)]
        t_det, det = None, None
        if self.df:  # team boundary: return once the majority answer is determined
            done = {}
            pending = set(tasks)
            while pending:
                fin, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                for d in fin:
                    done[tasks.index(d)] = d.result()[0]
                det = majority_determined(done, self.team)
                if det is not None:
                    t_det = time.time()
                    if pending:
                        self.n_team_early += 1
                    break
        outs = await asyncio.gather(*tasks)  # unconsumed agents complete off the critical path
        t_all = time.time()
        if t_det is None:
            t_det = t_all
        answers = [o[0] for o in outs]
        best = team_answer(answers)
        if det is not None and normalize_answer(det) != normalize_answer(best):
            self.n_df_mismatch += 1
        em, f1 = em_f1(best, item["answers"])
        return dict(qi=qi, qid=item["qid"], question=item["question"], answer=best, answers=answers,
                    em=em, f1=f1, t_arrive=t0, t_end=t_det, latency=t_det - t0, latency_all=t_all - t0,
                    hops=[len(o[1]) for o in outs])

    async def run(self, stream, update_rate=0.0, seed=0):
        rng = random.Random(seed)
        t0 = time.time()
        stop = asyncio.Event()

        async def updater():
            while not stop.is_set():
                await asyncio.sleep(rng.expovariate(update_rate))
                self.W.publish_update(rng)

        upd = asyncio.create_task(updater()) if update_rate > 0 else None
        tasks = []
        for qi, item in enumerate(stream):
            delay = t0 + item["t"] - time.time()
            if delay > 0:
                await asyncio.sleep(delay)
            tasks.append(asyncio.create_task(self.query(qi, item)))
        res = await asyncio.gather(*tasks)
        if self.pending:  # unconsumed verification calls: finish (their results feed peers), count their work
            await asyncio.gather(*self.pending, return_exceptions=True)
        stop.set()
        if upd:
            upd.cancel()
        return res, time.time() - t0
