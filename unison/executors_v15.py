"""Unison v15: execution of concurrent agent dataflows on the sharing frontier.

A call of a neural operator can be shared with another call in exactly three ways:
  identity   byte-identical input: the output is the same by determinism; zero work (coalescing / memo)
  verified   a peer's output is drafted and verified under the consumer's own input (speculative decoding):
             exact, but the verification pass costs at least the forward work of decoding the output, so it
             saves sequential steps only, never GPU work (Proposition 1, measured in the paper)
  estimated  the peer's output is reused without verification (semantic caches): saves work, inexact
Unison therefore (i) shrinks what is materialized on the critical path to the consumed key (boundary
factorization, at the call, the round and the team level: executors_v15 + agents.AgentRuntime),
(ii) shares identity everywhere (every key computation, plans included, is coalesced), (iii) makes the
payload identity-shareable by canonicalizing its program (claim-keyed provenance), and (iv) uses verified
sharing for what it can do: fewer key steps and certified cross-condition / cross-version reuse.

Variants (ablation): see VARIANTS15. Every variant shares the program, the engine and the speculator; they
differ only in the sharing mode of keys and payloads and in which boundaries the runtime observes.
"""
import asyncio
import time
from collections import Counter

from .common import normalize_answer, parse_note, parse_support, payload_tokens, provenance_ids, provenance_messages
from .executors import Suffix, _now_tm, phash
from .executors_v14 import KNode, Unison14, decision_end

#          key     payload  exec    identity  dataflow boundaries
VARIANTS15 = {
    "":        ("coop", "keyed", "min", True,  True),   # Unison
    "sbd":     ("self", "call",  "min", True,  True),   # Boundary+Defer: identity sharing + factorization only
    "coop":    ("coop", "call",  "min", True,  True),   # + cooperative (verified) key sharing
    "keyed":   ("self", "keyed", "min", True,  True),   # + canonical (identity-shared) payload
    "noid":    ("coop", "keyed", "min", False, True),   # Unison without identity sharing (verified only)
    "nodf":    ("coop", "keyed", "min", True,  False),  # Unison with call-level boundaries only
    "nolive":  ("trie", "keyed", "min", True,  True),   # finished peers only
    "noprio":  ("coop", "keyed", "fg",  True,  True),   # payload executed as foreground work, with speculation
    "lazy":    ("self", "none",  None,  True,  True),   # decision only, no payload (lower bound)
}


class Unison15(Unison14):
    plan_mode = Suffix.verify_mode

    def __init__(self, engine, run_id, world=None, variant="", max_cands=8, **kw):
        if variant not in VARIANTS15:
            raise ValueError(f"unknown Unison v15 variant {variant!r}")
        key, payload, ex, identity, df = VARIANTS15[variant]
        super().__init__(engine, run_id, world, variant="", max_cands=max_cands, **kw)
        self.variant = variant
        self.key, self.payload, self.exec = key, payload, ex
        self.identity = identity
        self.dataflow_boundaries = df
        self.name = "u15" + (f"-{variant}" if variant else "")
        self.plan_table = {}   # identity sharing of plan (key) computations: prompt hash -> future

    # ------------------------------------------------------------------ payload width follows the program
    def _prov_tokens(self):
        return payload_tokens(getattr(self, "note_format", "dfp"))

    async def _payload_call(self, node, ids, head, rec, t_ret):
        ea = self._payload_mode()
        ea["resume"] = len(head)
        _, rest, _ = await self._engine(list(ids) + list(head), self._prov_tokens(), ea, rid=node.rid + "-p")
        self.prov_out[node.rid] = list(rest)
        rec["prov_key"] = node.rid
        self.prov_recs.append(dict(kind="call", t_ret=t_ret, t_end=time.time(), n_out=len(rest), engine=True))

    async def _payload_keyed(self, info, head, rec, t_ret):
        text = self.E.decode(head)
        rel, ans = parse_note(text)
        k = parse_support(text)
        w = self.world
        title, body = w.corpus[info["pid"]]["title"], w.text(info["pid"], info["version"])
        if self.exec == "min":
            ids = provenance_ids(self.E, title, body, ans, k, getattr(self, "note_format", "dfp"))
        else:
            ids = self.E.chat_ids(provenance_messages(title, body, ans, k))
        key = phash(ids)
        rec["prov_key"] = key
        fut = self.prov.get(key)
        if fut is not None:
            toks = await asyncio.shield(fut)
            self.prov_recs.append(dict(kind="keyed", t_ret=t_ret, t_end=time.time(), n_out=len(toks), engine=False))
            return
        fut = self.prov[key] = asyncio.get_running_loop().create_future()
        self.n_seq += 1
        try:
            _, toks, _ = await self._engine(ids, self._prov_tokens(), self._payload_mode(), rid=f"{self.run}-p{self.n_seq}")
        except Exception as e:
            fut.set_exception(e)
            raise
        fut.set_result(list(toks))
        self.prov_out[key] = list(toks)
        self.prov_recs.append(dict(kind="keyed", t_ret=t_ret, t_end=time.time(), n_out=len(toks), engine=True))

    # ------------------------------------------------------------------ plan calls: identity sharing
    async def plain(self, info, ids, max_tokens):
        t = time.time()
        if not self.identity:
            return await super().plain(info, ids, max_tokens)
        ph = phash(ids)
        fut = self.plan_table.get(ph)
        if fut is not None:  # identical program state (same strategy, question wording and findings)
            toks = await asyncio.shield(fut)
            self._rec(info, ids, toks, _now_tm(t), t, source="exact")
            return list(toks)
        fut = self.plan_table[ph] = asyncio.get_running_loop().create_future()
        try:
            _, toks, tm = await self._engine(ids, max_tokens, self.plan_mode)
        except Exception as e:
            fut.set_exception(e)
            raise
        toks = list(toks)
        fut.set_result(toks)
        self._rec(info, ids, toks, tm, t, source="engine")
        return toks

    # ------------------------------------------------------------------ verification calls: key
    async def verify(self, info, ids, max_tokens):
        t_arr = time.time()
        pid, ph = info["pid"], phash(ids)
        u = self.by_ph.get(ph) if self.identity else None
        if u is not None:  # identity: single flight with the identical computation
            await u.ev.wait()
            self._rec(info, ids, u.toks, _now_tm(t_arr), t_arr, source="exact", src=u.rid, src_version=u.version)
            return list(u.toks)
        self.n_seq += 1
        node = KNode(f"{self.run}-k{self.n_seq}", pid, info["version"], ph)
        if self.identity:
            self.by_ph[ph] = node
        ea = dict(Suffix.verify_mode)
        ea.update(nl=self._nl_ids(), n_nl=self._nnl())
        trie = []
        if self.key == "self":
            ea["mode"] = "lazy"
        else:  # without identity sharing, identical finished peers are legitimate (verified) draft sources too
            trie = self._trie(pid, ph if self.identity else None)
            ea.update(mode="dbc", cands=[list(c.toks[:c.dend]) for c in trie], dends=[c.dend for c in trie],
                      group=f"p{pid}", live=self.key == "coop")
        src = None
        try:
            _, toks, tm = await self._engine(ids, max_tokens, ea, rid=node.rid)
            toks = list(toks)
            if toks and toks[-1] in self.E.eos:
                toks = toks[:-1]
            for c in trie:  # ended exactly at a finished peer's boundary: verified equality of the keys
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
        if self.payload != "none" and rel:  # audit obligation: every evidence claim gets its payload
            if self.payload == "keyed":
                self.bg.append(asyncio.ensure_future(self._payload_keyed(info, node.toks, rec, t_ret)))
            else:
                self.bg.append(asyncio.ensure_future(self._payload_call(node, ids, node.toks, rec, t_ret)))
        return node.toks
