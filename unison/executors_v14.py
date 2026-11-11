"""Unison v14: boundary-factored execution of evidence-verification calls.

This module is the *only* implementation of the method described in paper/unison14. Every variant used in
the paper's ablation is a configuration of one class; no variant copies a peer's rationale.

A verification note is consumed only up to its *consumption boundary* (the decision: Relevant, Answer,
Support). Unison factors every call at that boundary into
  key      the decision prefix, computed on the agent's critical path by the consumer's own greedy
           speculative decoding. Drafts come from a *cooperative key trie*: the decision prefixes of all
           calls on the same passage, finished ones and in-flight ones. Every emitted token is the consumer's
           own argmax, so the delivered decision is exactly the consumer's greedy decision (Thm. 1, 2).
           The request ends at the boundary.
  payload  the provenance of an evidence claim, late-materialized off the critical path by a canonical
           program over (passage version, supporting sentence, answer) that shares the evidence prefix of
           the key program. Equal claims issue byte-identical requests and are computed once (Thm. 3).
           Payload is executed in compute-minimal form (no speculation) at background priority inside the
           engine scheduler (plugin, UNISON_PRIO), so it never takes a batch slot or KV memory that a
           waiting foreground request needs (Prop. 5).

Ablation axes (paper Table "ablation"):
  key      self   the consumer decodes its decision with SuffixDecoding and stops at its own boundary
           trie   + drafts from finished peers' decision prefixes
           coop   + drafts from in-flight peers (cooperative key trie)          [Unison]
  payload  none   no provenance (decision-only lower bound)
           call   per-call provenance: continuation of the consumer's own request
           keyed  claim-keyed canonical provenance                               [Unison]
  exec     min    background priority, no speculation, evidence-prefix-shared   [Unison]
           fg     foreground priority, speculation, separate prompt (prior design)
Every variant has the same audit obligation: every evidence claim (Relevant: yes) gets provenance.
"""
import asyncio
import time
from collections import Counter, defaultdict

from .common import (BOUNDARY_FIELD, normalize_answer, parse_note, parse_support, provenance_ids,
                     provenance_messages)
from .executors import Executor, Suffix, _now_tm, phash

VARIANTS = {
    "": ("coop", "keyed", "min"),        # Unison
    "sbd": ("self", "call", "min"),      # A: Self-Boundary + Deferred provenance (strongest composite)
    "coop": ("coop", "call", "min"),     # B: + cooperative key
    "keyed": ("self", "keyed", "min"),   # C: + claim-keyed provenance
    "nolive": ("trie", "keyed", "min"),  # Unison, finished peers only
    "noprio": ("coop", "keyed", "fg"),   # Unison, payload executed like foreground work
    "lazy": ("self", "none", None),      # decision only, no provenance (lower bound)
}
PROV_TOKENS = 64
NOTE_LINES = {"dfirst": 2}  # lines of the decision prefix per format (default 3: Relevant, Answer, Support)


def decision_end(tok, ids, field="support:"):
    """Number of leading tokens through the newline that closes the boundary line."""
    text, pos = "", []
    for t in ids:
        text += tok.decode([t], skip_special_tokens=True)
        pos.append(len(text))
    j = text.lower().find(field)
    if j < 0:
        return len(ids)
    nl = text.find("\n", j)
    if nl < 0:
        return len(ids)
    for t, e in enumerate(pos):
        if e > nl:
            return t + 1
    return len(ids)


class KNode:
    __slots__ = ("rid", "pid", "version", "ph", "done", "toks", "dec", "dend", "ev")

    def __init__(self, rid, pid, version, ph):
        self.rid, self.pid, self.version, self.ph = rid, pid, version, ph
        self.done, self.toks, self.dec, self.dend = False, [], None, 0
        self.ev = asyncio.Event()


class Unison14(Executor):
    plan_mode = Suffix.verify_mode

    def __init__(self, *a, variant="", max_cands=8, **kw):
        super().__init__(*a, **kw)
        if variant not in VARIANTS:
            raise ValueError(f"unknown Unison variant {variant!r}")
        self.variant = variant
        self.key, self.payload, self.exec = VARIANTS[variant]
        self.max_cands = max_cands
        self.name = "u14" + (f"-{variant}" if variant else "")
        self._nl = None
        self.bg = []                       # payload tasks (awaited by run_exp before the run is closed)
        self.dag = defaultdict(list)       # pid -> finished decision prefixes
        self.by_ph = {}                    # identical-prompt single flight
        self.prov = {}                     # claim-keyed payload: prompt hash -> future of its tokens
        self.prov_out = {}                 # payload key -> tokens (for audit evaluation)
        self.prov_recs = []
        self.n_seq = 0

    # ------------------------------------------------------------------ helpers
    def _field(self):
        return BOUNDARY_FIELD.get(getattr(self, "note_format", ""), "answer:")

    def _nnl(self):
        return NOTE_LINES.get(getattr(self, "note_format", ""), 3)

    def _nl_ids(self):
        if self._nl is None:
            tok = self.E.tok
            self._nl = [i for i in range(len(tok)) if "\n" in tok.decode([i])]
        return self._nl

    def _decision(self, toks):
        text = self.E.decode(toks)
        rel, ans = parse_note(text)
        return parse_support(text), rel, normalize_answer(ans)

    def _trie(self, pid, ph):
        """Distinct finished decision prefixes on the passage: most frequent decision first, then latest."""
        fin = [u for u in self.dag[pid] if u.ph != ph and u.toks and u.dend]
        if not fin:
            return []
        cnt = Counter(u.dec for u in fin)
        seen, out = set(), []
        for u in sorted(reversed(fin), key=lambda u: -cnt[u.dec]):
            key = tuple(u.toks[:u.dend])
            if key not in seen:
                seen.add(key)
                out.append(u)
                if len(out) >= self.max_cands:
                    break
        return out

    def _payload_mode(self):
        if self.exec == "min":  # compute-minimal: no speculation, background class in the engine scheduler
            return {"mode": "none", "bg": True}
        return dict(Suffix.verify_mode)

    # ------------------------------------------------------------------ payload
    async def _payload_call(self, node, ids, head, rec, t_ret):
        """Per-call provenance: continue the consumer's own request after its decision (prefix-cache hit)."""
        ea = self._payload_mode()
        ea["resume"] = len(head)
        _, rest, _ = await self._engine(list(ids) + list(head), PROV_TOKENS, ea, rid=node.rid + "-p")
        self.prov_out[node.rid] = list(rest)
        rec["prov_key"] = node.rid
        self.prov_recs.append(dict(kind="call", t_ret=t_ret, t_end=time.time(), n_out=len(rest), engine=True))

    async def _payload_keyed(self, info, head, rec, t_ret):
        """Claim-keyed provenance: one computation per (passage version, supporting sentence, answer)."""
        text = self.E.decode(head)
        rel, ans = parse_note(text)
        k = parse_support(text)
        w = self.world
        title, body = w.corpus[info["pid"]]["title"], w.text(info["pid"], info["version"])
        if self.exec == "min":  # shares the evidence prefix of the key program (system, instructions, passage)
            ids = provenance_ids(self.E, title, body, ans, k, getattr(self, "note_format", "dfp"))
        else:
            ids = self.E.chat_ids(provenance_messages(title, body, ans, k))
        key = phash(ids)
        rec["prov_key"] = key
        fut = self.prov.get(key)
        if fut is not None:  # the identical payload computation exists: share it
            toks = await asyncio.shield(fut)
            self.prov_recs.append(dict(kind="keyed", t_ret=t_ret, t_end=time.time(), n_out=len(toks), engine=False))
            return
        fut = self.prov[key] = asyncio.get_running_loop().create_future()
        self.n_seq += 1
        try:
            _, toks, _ = await self._engine(ids, PROV_TOKENS, self._payload_mode(), rid=f"{self.run}-p{self.n_seq}")
        except Exception as e:
            fut.set_exception(e)
            raise
        fut.set_result(list(toks))
        self.prov_out[key] = list(toks)
        self.prov_recs.append(dict(kind="keyed", t_ret=t_ret, t_end=time.time(), n_out=len(toks), engine=True))

    # ------------------------------------------------------------------ key
    async def verify(self, info, ids, max_tokens):
        t_arr = time.time()
        pid, ph = info["pid"], phash(ids)
        u = self.by_ph.get(ph)
        if u is not None:  # identical prompt: single flight with the identical computation
            await u.ev.wait()
            self._rec(info, ids, u.toks, _now_tm(t_arr), t_arr, source="exact", src=u.rid, src_version=u.version)
            return list(u.toks)
        self.n_seq += 1
        node = KNode(f"{self.run}-k{self.n_seq}", pid, info["version"], ph)
        self.by_ph[ph] = node
        ea = dict(Suffix.verify_mode)
        ea.update(nl=self._nl_ids(), n_nl=self._nnl())  # end at the consumer's own boundary
        trie = []
        if self.key == "self":
            ea["mode"] = "lazy"
        else:
            trie = self._trie(pid, ph)
            ea.update(mode="dbc", cands=[list(c.toks[:c.dend]) for c in trie], dends=[c.dend for c in trie],
                      group=f"p{pid}", live=self.key == "coop")
        src = None
        try:
            _, toks, tm = await self._engine(ids, max_tokens, ea, rid=node.rid)
            toks = list(toks)
            if toks and toks[-1] in self.E.eos:
                toks = toks[:-1]
            for c in trie:  # ended exactly at a finished peer's boundary (same tokens by Thm. 1)
                if len(toks) == c.dend and toks == c.toks[:c.dend]:
                    src = c
                    break
            node.toks = toks
            node.dec = self._decision(toks)
            node.dend = decision_end(self.E.tok, toks, self._field())
        finally:
            node.done = True
            node.ev.set()
        t_ret = time.time()
        if src is None and node.dend:
            self.dag[pid].append(node)
        rec = self._rec(info, ids, node.toks, tm, t_arr, source="dcert" if src is not None else "engine",
                        rid=node.rid, n_trie=len(trie), src=None if src is None else src.rid, dend=node.dend,
                        src_version=src.version if src is not None else info["version"])
        rel = node.dec[1]
        if self.payload != "none" and rel:  # audit obligation: every evidence claim gets provenance
            if self.payload == "keyed":
                self.bg.append(asyncio.ensure_future(self._payload_keyed(info, node.toks, rec, t_ret)))
            else:
                self.bg.append(asyncio.ensure_future(self._payload_call(node, ids, node.toks, rec, t_ret)))
        return node.toks


class LaCacheStyle(Executor):
    """LaCache-style semantic cache (Liang et al., arXiv 2608.01718; our re-implementation): the nearest cached
    note on the same passage (condition cosine >= tau) is accepted only if the consumer's own speculative
    verification accepts its first `look` tokens; then the whole cached note is returned. Otherwise the call
    is decoded normally (SuffixDecoding) and inserted."""

    def __init__(self, *a, encoder=None, tau=0.8, look=16, **kw):
        super().__init__(*a, **kw)
        self.enc, self.tau, self.look = encoder, tau, look
        self.name = "lacache"
        self.entries = defaultdict(list)  # pid -> [(emb, version, phash, fut)]
        self.verify_mode = self.plan_mode = Suffix.verify_mode

    async def verify(self, info, ids, max_tokens):
        t = time.time()
        ph = phash(ids)
        e = (await self.enc.aencode([info["cond"]]))[0]
        best, best_s = None, -1.0
        for emb, ver, h, fut in self.entries[info["pid"]]:
            if h == ph:
                best, best_s = (ver, fut), 2.0
                break
            if fut.done():
                s = float(e @ emb)
                if s > best_s:
                    best, best_s = (ver, fut), s
        if best is not None and best_s >= 2.0:
            toks = await asyncio.shield(best[1])
            self._rec(info, ids, toks, _now_tm(t), t, source="exact", src_version=best[0])
            return toks
        fut = asyncio.get_running_loop().create_future()
        self.entries[info["pid"]].append((e, info["version"], ph, fut))
        ea = dict(self.verify_mode)
        cand = None
        if best is not None and best_s >= self.tau:
            cand = list(best[1].result())
            L = min(self.look, len(cand))
            if L > 0:
                ea.update(mode="dbc", cands=[cand[:L]], dends=[L])
        try:
            _, toks, tm = await self._engine(ids, max_tokens, ea)
        except Exception as ex:
            fut.set_exception(ex)
            raise
        toks = list(toks)
        if cand is not None and toks and toks[-1] in self.E.eos and len(toks) - 1 == min(self.look, len(cand)) \
                and toks[:-1] == cand[:len(toks) - 1] and len(cand) > len(toks) - 1:
            fut.set_result(cand)  # lookahead check passed: the cached note is returned
            self._rec(info, ids, cand, tm, t, source="sem", sim=best_s, src_version=best[0])
            return cand
        fut.set_result(toks)
        self._rec(info, ids, toks, tm, t, source="engine")
        return toks
