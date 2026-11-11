"""Execution policies for evidence-verification calls (planner calls are always run plainly).

All policies run on the same vLLM engine with the same prompts; they differ only in how a
verification call is served:
  NoShare        independent execution (APC already shares the evidence KV across conditions)
  ExactShare     single-flight coalescing + memo on the exact prompt (evidence version included)
  PLD / Suffix   engine-level speculative decoding (prompt lookup / SuffixDecoding)
  ExactSuffix    ExactShare + SuffixDecoding (strongest lossless combination of prior work)
  SemCache       GPTCache-style: same evidence and condition cosine >= tau -> reuse the result
  SemCRC         SemCache with the cosine threshold calibrated to a decision-risk budget eps
  Unison         optimistic sharing control over the condition/version-annotated evidence DAG
"""
import asyncio
import hashlib
import os
import time
from collections import defaultdict

import numpy as np

from .ahn import token_pieces, version_delta


def phash(ids):
    return hashlib.blake2b(np.asarray(ids, dtype=np.int32).tobytes(), digest_size=12).hexdigest()


def _now_tm(t):
    n = time.time()
    return dict(t_submit=t, t_first=n, t_end=n, steps=0)


class Executor:
    name = "noshare"
    verify_mode = {"mode": "none"}
    plan_mode = {"mode": "none"}

    def __init__(self, engine, run_id, world=None, **kw):
        self.E = engine
        self.run = run_id
        self.world = world
        self.recs = []

    def _ea(self, base):
        d = dict(base)
        d["run"] = self.run
        return d

    def _rec(self, info, ids, toks, tm, t_arrive, **extra):
        r = dict(info)
        r.update(method=self.name, phash=phash(ids), n_prompt=len(ids), out=list(toks), n_out=len(toks),
                 t_arrive=t_arrive, **tm)
        r.update(extra)
        self.recs.append(r)
        return r

    async def _engine(self, ids, max_tokens, ea, rid=None, on_tokens=None):
        return await self.E.generate(ids, max_tokens, self._ea(ea), rid=rid, on_tokens=on_tokens)

    async def plain(self, info, ids, max_tokens):
        t = time.time()
        _, toks, tm = await self._engine(ids, max_tokens, self.plan_mode)
        self._rec(info, ids, toks, tm, t, source="engine")
        return toks

    async def verify(self, info, ids, max_tokens):
        t = time.time()
        _, toks, tm = await self._engine(ids, max_tokens, self.verify_mode)
        self._rec(info, ids, toks, tm, t, source="engine")
        return toks


class PLD(Executor):
    name = "pld"
    verify_mode = plan_mode = {"mode": "pld", "k": 10}


class Suffix(Executor):
    name = "suffix"
    verify_mode = plan_mode = {"mode": "suffix", "spec_factor": 1.0, "min_prob": 0.1, "k": 32}


class ExactShare(Executor):
    name = "exact"

    def __init__(self, *a, suffix=False, **kw):
        super().__init__(*a, **kw)
        self.table = {}
        if suffix:
            self.name = "exact+suffix"
            self.verify_mode = self.plan_mode = Suffix.verify_mode

    async def verify(self, info, ids, max_tokens):
        t = time.time()
        key = phash(ids)
        fut = self.table.get(key)
        if fut is not None:
            toks = await asyncio.shield(fut)
            self._rec(info, ids, toks, _now_tm(t), t, source="exact")
            return toks
        fut = self.table[key] = asyncio.get_running_loop().create_future()
        _, toks, tm = await self._engine(ids, max_tokens, self.verify_mode)
        fut.set_result(toks)
        self._rec(info, ids, toks, tm, t, source="engine")
        return toks


class SemCache(Executor):
    """Semantic result cache on (evidence, condition embedding); tau fixed or risk-calibrated."""
    name = "semcache"

    def __init__(self, *a, encoder=None, tau=0.9, versioned=False, label=None, suffix=False, **kw):
        super().__init__(*a, **kw)
        if suffix:  # misses are served with the same history speculator as Exact+Suffix / Unison
            self.verify_mode = self.plan_mode = Suffix.verify_mode
        self.enc = encoder
        self.tau = tau
        self.versioned = versioned
        self.name = label or f"semcache{'V' if versioned else ''}-{tau:.2f}"
        self.entries = defaultdict(list)  # pid -> [(emb, version, phash, fut)]

    async def verify(self, info, ids, max_tokens):
        t = time.time()
        ph = phash(ids)
        e = (await self.enc.aencode([info["cond"]]))[0]
        best, best_s = None, -1.0
        for emb, ver, h, fut in self.entries[info["pid"]]:
            if h == ph:  # identical prompt: certified reuse
                best, best_s = (ver, fut), 2.0
                break
            if self.versioned and ver != info["version"]:
                continue
            s = float(e @ emb)
            if s > best_s:
                best, best_s = (ver, fut), s
        if best is not None and best_s >= self.tau:
            toks = await asyncio.shield(best[1])
            self._rec(info, ids, toks, _now_tm(t), t, source="sem", sim=best_s, src_version=best[0])
            return toks
        fut = asyncio.get_running_loop().create_future()
        self.entries[info["pid"]].append((e, info["version"], ph, fut))
        _, toks, tm = await self._engine(ids, max_tokens, self.verify_mode)
        fut.set_result(toks)
        self._rec(info, ids, toks, tm, t, source="engine")
        return toks


class Node:
    __slots__ = ("rid", "pid", "version", "cond", "ph", "t0", "done", "toks", "pieces", "ev")

    def __init__(self, rid, pid, version, cond, ph):
        self.rid, self.pid, self.version, self.cond, self.ph = rid, pid, version, cond, ph
        self.t0 = time.time()
        self.done = False
        self.toks = []
        self.pieces = []
        self.ev = asyncio.Event()


class OSC(Executor):
    """Text-law Optimistic Sharing Control (previous design; kept as a baseline: 'osc-*').

    One survival law from the AHN drives every decision for a verification call w:
      commit   take a producer's decision without computing (certified if the prompt is
               identical; otherwise only if the calibrated agreement q >= theta_eps)
      wait     join an in-flight producer when committing to it beats computing alone
      speculate otherwise compute, drafting from the producers' (live) outputs up to the
               break-even horizon S(t) >= kappa; verification keeps the output exact.
    """
    MAX_CANDS = 12   # AHN scoring budget per call (most recent producers of the same evidence)
    MAX_NODES = 32   # evidence-DAG producer nodes retained per passage
    plan_mode = Suffix.verify_mode  # non-evidence calls use the engine's default speculator

    def __init__(self, *a, ahn=None, encoder=None, eps=0.0, variant="full", **kw):
        super().__init__(*a, **kw)
        self.ahn = ahn
        self.enc = encoder
        self.eps = eps
        self.variant = variant
        self.name = "osc" + (f"-e{eps:g}" if eps > 0 else "") + ("" if variant == "full" else f"-{variant}")
        if eps > 0:
            key = "cos_thresholds" if variant == "cosine" else "thresholds"
            self.theta = ahn.meta[key].get(f"{eps:.3f}", 1.01)
        else:
            self.theta = None
        self.dag = defaultdict(list)
        self.len_mean, self.len_n = 60.0, 0
        self.n_seq = 0
        self.delta_cache = {}
        self.by_ph = {}  # certified-equivalence index over all producers (identical prompt)

    # ---------------------------------------------------------------- helpers
    def _pieces(self, node):
        if len(node.pieces) < len(node.toks):
            node.pieces.extend(token_pieces(self.E.tok, node.toks[len(node.pieces):]))
        return node.pieces[:len(node.toks)]

    def _delta(self, pid, v_from, v_to):
        key = (pid, v_from, v_to)
        if key not in self.delta_cache:
            w = self.world
            self.delta_cache[key] = version_delta(w.text(pid, v_from), w.text(pid, v_to))
        return self.delta_cache[key]

    async def _score(self, node, cands):
        if not cands:
            return []
        if self.variant == "cosine":  # similarity in place of the learned survival law
            e = await self.enc.aencode([node.cond] + [u.cond for u in cands])
            sims = np.clip(e[1:] @ e[0], 0.0, 1.0)
            return [(np.full(len(u.toks), float(s)), float(s)) for u, s in zip(cands, sims)]
        items = [(u.cond, "none" if self.variant == "nodelta" else self._delta(node.pid, u.version, node.version),
                  self._pieces(u)) for u in cands]
        return await self.ahn.ascore(node.cond, items)

    def _L_go(self):
        return max(self.E.step_ewma, 1e-3) * (self.len_mean + 1.0)

    def _horizon(self, S):
        """Break-even draft horizon: keep token t while S(t) * benefit >= marginal verification cost."""
        tau = max(self.E.step_ewma, 1e-3)
        dlt, ctok = self.E.cost["delta"], self.E.cost["ctok"]
        B = max(self.E.n_running, 1)
        kappa = ctok * B / (tau + dlt * B)
        keep = np.nonzero(S >= kappa)[0]
        return int(keep[-1]) + 1 if len(keep) else 0

    async def _join(self, u, certified=False):
        """Wait for an in-flight producer, never longer than computing alone (ski-rental cap)."""
        t = time.time()
        if not u.done and self.variant != "nojoin":
            try:
                await asyncio.wait_for(u.ev.wait(), timeout=None if certified else self._L_go())
            except asyncio.TimeoutError:
                pass
        return time.time() - t

    def _commit(self, info, ids, u, t_arr, waited, kind, q=1.0):
        self._rec(info, ids, u.toks, _now_tm(t_arr), t_arr, source=kind, src=u.rid, waited=waited, q=q,
                  src_version=u.version)
        return list(u.toks)

    # ---------------------------------------------------------------- policy
    async def verify(self, info, ids, max_tokens):
        t_arr = time.time()
        pid, ph = info["pid"], phash(ids)
        self.n_seq += 1
        node = Node(f"{self.run}-v{self.n_seq}", pid, info["version"], info["cond"], ph)
        pool = self.dag[pid]
        prior = pool[-self.MAX_CANDS:]
        if self.variant == "noinflight":
            prior = [u for u in prior if u.done]
        waited = 0.0
        # (1) certified equivalence: identical prompt => identical greedy computation
        u = self.by_ph.get(ph)
        if u is not None:
            waited += await self._join(u, certified=True)
            if u.done:
                return self._commit(info, ids, u, t_arr, waited, "certified")
        # (2) risk-bounded commit under the calibrated agreement law
        scores = await self._score(node, prior)
        if self.theta is not None and scores:
            j = int(np.argmax([q for _, q in scores]))
            u, q = prior[j], scores[j][1]
            if q >= self.theta:
                if not u.done:
                    waited += await self._join(u)
                    if u.done:
                        q = (await self._score(node, [u]))[0][1]
                if u.done and q >= self.theta:
                    return self._commit(info, ids, u, t_arr, waited, "risk", q)
                prior = pool[-self.MAX_CANDS:]
                scores = await self._score(node, prior)
        # (3) verified speculation from the evidence DAG (exact output)
        peers, drafts, dlen = [], {}, {}
        if scores and self.variant != "nospec":
            order = sorted(range(len(prior)), key=lambda i: (-float(scores[i][0].sum()), -scores[i][1]))
            for i in order[:4]:
                u, (S, _) = prior[i], scores[i]
                peers.append(u.rid)
                if u.done:
                    drafts[u.rid] = u.toks
                    dlen[u.rid] = len(u.toks) if self.variant == "fulldraft" else self._horizon(S)
        pool.append(node)
        self.by_ph.setdefault(ph, node)
        if len(pool) > self.MAX_NODES:
            del pool[:len(pool) - self.MAX_NODES]
        ea = {"mode": "unison", "peers": peers, "drafts": drafts, "dlen": dlen, "fallback": "suffix", "k": 32}

        def on_tokens(toks):
            node.toks = toks

        try:
            _, toks, tm = await self._engine(ids, max_tokens, ea, rid=node.rid, on_tokens=on_tokens)
        finally:
            node.done = True
            node.ev.set()
        node.toks = toks
        self.len_n += 1
        self.len_mean += (len(toks) - self.len_mean) / min(self.len_n, 200)
        self._rec(info, ids, toks, tm, t_arr, source="engine", waited=waited, n_peers=len(peers))
        return toks


class VCache(SemCache):
    """vCache-style semantic cache: every cached entry learns its own similarity threshold online.

    A lookup whose nearest entry e (same passage) has similarity s is answered from e only when s exceeds
    e's learned threshold t_e and the exploration draw does not fire; otherwise the call is computed and the
    outcome (did e's decision equal the fresh decision?) is recorded on e. t_e is the largest similarity at
    which e was observed to disagree (initial t_e = tau0, the global calibrated cosine threshold); the
    exploration probability at s is 1/(1 + number of agreeing observations of e at similarity <= s).
    """

    def __init__(self, *a, tau0=0.9, floor=0.7, seed=0, **kw):
        super().__init__(*a, tau=tau0, **kw)
        self.name = "vcache+S" if kw.get("suffix") else "vcache"
        self.floor = floor
        self.thr = {}   # entry phash -> learned threshold
        self.obs = defaultdict(list)  # entry phash -> [(sim, agree)]
        self.rng = np.random.default_rng(seed)

    def _dec(self, toks):
        from .common import parse_note, normalize_answer
        rel, ans = parse_note(self.E.decode(toks))
        return rel, normalize_answer(ans)

    async def verify(self, info, ids, max_tokens):
        t = time.time()
        ph = phash(ids)
        e = (await self.enc.aencode([info["cond"]]))[0]
        best, best_s = None, -1.0
        for emb, ver, h, fut in self.entries[info["pid"]]:
            if h == ph:
                best, best_s = (ver, fut, h), 2.0
                break
            s = float(e @ emb)
            if s > best_s:
                best, best_s = (ver, fut, h), s
        if best is not None and best_s >= 2.0:
            toks = await asyncio.shield(best[1])
            self._rec(info, ids, toks, _now_tm(t), t, source="exact", src_version=best[0])
            return toks
        explore = None
        if best is not None and best_s >= self.floor:
            h = best[2]
            thr = self.thr.get(h, self.tau)
            n_ok = sum(1 for s, ok in self.obs[h] if ok and s <= best_s)
            if best_s > thr and best[1].done() and self.rng.random() >= 1.0 / (1.0 + n_ok):
                toks = best[1].result()
                self._rec(info, ids, toks, _now_tm(t), t, source="sem", sim=best_s, src_version=best[0])
                return toks
            explore = best
        fut = asyncio.get_running_loop().create_future()
        self.entries[info["pid"]].append((e, info["version"], ph, fut))
        _, toks, tm = await self._engine(ids, max_tokens, self.verify_mode)
        fut.set_result(toks)
        if explore is not None and explore[1].done():
            ok = self._dec(explore[1].result()) == self._dec(toks)
            h = explore[2]
            self.obs[h].append((best_s, ok))
            if not ok:
                self.thr[h] = max(self.thr.get(h, self.tau), best_s)
        self._rec(info, ids, toks, tm, t, source="engine")
        return toks


async def _key_vec(enc, key):
    """Embedding of a cache key (the sub-question) for an external cache library: the vector the retriever's
    pass left for it (retriever.Encoder.keys), or an encoder pass of its own if the key was never searched."""
    v = enc.keys.get(key) if enc.keys is not None else None
    if v is None:
        v = (await enc.aencode([key]))[0]
    return np.asarray(v, dtype=np.float32)


class GPTCacheExec(Executor):
    """GPTCache (official library, gptcache==0.1.44): its data manager (sqlite + FAISS), its
    SearchDistanceEvaluation and its default similarity threshold (Config().similarity_threshold).
    One cache per passage, keyed by the sub-question; misses are decoded by the engine and put.
    The library runs as a service in processes of its own (cache_service.py); the sub-question is embedded here
    with the retriever's encoder. Every record carries the time of the embedding (c_emb), of the lookup (c_look,
    of which c_svc inside the service) and, for a miss, of the put that follows the engine (c_post)."""

    def __init__(self, *a, encoder=None, suffix=False, **kw):
        super().__init__(*a, **kw)
        from .cache_service import GPTCacheService
        if suffix:
            self.verify_mode = self.plan_mode = Suffix.verify_mode
        self.name = "gptcache+S" if suffix else "gptcache"
        self.enc = encoder
        self.enc.keys = {}
        self.svc = GPTCacheService()

    async def verify(self, info, ids, max_tokens):
        t = time.time()
        key = info["cond"]
        vec = await _key_vec(self.enc, key)
        t1 = time.time()
        hit, svc = await self.svc.get(info["pid"], key, vec)
        t2 = time.time()
        if hit:
            ver, body = _untag(hit, info["version"])
            toks = [int(x) for x in body.split()]
            self._rec(info, ids, toks, _now_tm(t), t, source="sem", src_version=ver, c_emb=t1 - t, c_look=t2 - t1,
                      c_svc=svc)
            return toks
        _, toks, tm = await self._engine(ids, max_tokens, self.verify_mode)
        t3 = time.time()
        # the stored value carries the passage version it was computed on (accounting only: the cache never
        # reads it), so that a hit can be classified as a reuse across versions afterwards
        await self.svc.put(info["pid"], key, vec, f"{info['version']}|" + " ".join(map(str, toks)))
        self._rec(info, ids, toks, tm, t, source="engine", c_emb=t1 - t, c_look=t2 - t1, c_svc=svc,
                  c_post=time.time() - t3)
        return toks


def _untag(value, default):
    """(version, token string) of a cached value written as '<version>|<tokens>'."""
    ver, sep, body = value.rpartition("|")
    return (int(ver) if sep else default), body


class VCacheOfficial(Executor):
    """vCache (official implementation, github.com/vcache-project/vCache): VerifiedDecisionPolicy with its
    default delta, HNSW vector store, in-memory metadata. Correctness labels for its online threshold
    learning compare the decisions (relevance, normalized answer) of two notes, the analogue of the
    string comparison it uses for short answers. One cache per passage, keyed by the sub-question.
    The implementation runs as a service in a process of its own (cache_service.py) and calls the model through
    this executor. Records carry c_emb, c_look (until the engine submission or the hit) and c_post as GPTCache's."""

    def __init__(self, *a, encoder=None, delta=0.02, suffix=False, **kw):
        super().__init__(*a, **kw)
        from .cache_service import VCacheService
        if suffix:
            self.verify_mode = self.plan_mode = Suffix.verify_mode
        self.name = "vcache-official+S" if suffix else "vcache-official"
        self.enc = encoder
        self.enc.keys = {}
        self.delta = delta
        self.svc = VCacheService(delta=delta)

    def _dec(self, s):
        from .common import parse_note, normalize_answer
        rel, ans = parse_note(self.E.decode([int(x) for x in _untag(s, 0)[1].split()]))
        return rel, normalize_answer(ans)

    async def verify(self, info, ids, max_tokens):
        t = time.time()
        key = info["cond"]
        vec = await _key_vec(self.enc, key)
        t1 = time.time()
        out = []

        async def run_engine():
            _, toks, tm = await self._engine(ids, max_tokens, self.verify_mode)
            out.append((toks, tm, time.time()))
            # tagged with the passage version (accounting only; the label compares the decisions)
            resp = f"{info['version']}|" + " ".join(map(str, toks))
            return resp, self._dec(resp)

        hit, resp, _, _ = await self.svc.infer(info["pid"], key, vec, run_engine)
        now = time.time()
        ver, body = _untag(resp, info["version"])
        toks = [int(x) for x in body.split()]
        if hit or not out:
            self._rec(info, ids, toks, _now_tm(t), t, source="sem", src_version=ver, c_emb=t1 - t, c_look=now - t1)
        else:
            self._rec(info, ids, toks, out[0][1], t, source="engine", c_emb=t1 - t,
                      c_look=out[0][1]["t_submit"] - t1, c_post=now - out[0][2])
        return toks
