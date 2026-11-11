"""Unison v17: consensus drafting of key computations.

Every key computation of the agent program (a plan, a verification decision) is decoded by the consumer's own model
under speculative verification. The drafts come from one learned module, the consensus drafter (unison.drafter),
a pointer into the record of the call: the tail of the consumer's own prompt and the outputs of the other calls on
the same evidence (verification) or the same question (plan), finished or in flight. The first draft of a call is
written from the record alone and verified by the pass that processes the call's prompt, so a call whose draft
holds takes one pass; later drafts are driven by the consumer's hidden state at its last token. The draft horizon
follows the drafter's own confidence and the engine's measured verification price: a token is drafted while the
estimated probability that the draft so far is accepted exceeds the share of an iteration that verifying it adds
for the running requests (plugin, _verification_price). Verification keeps every delivered token equal to the
consumer's own greedy token, so learning changes the cost of a key, never its value.

Identity sharing (byte-identical programs computed once), the claim-keyed canonical payload at background
priority, and the consumption boundaries of the dataflow are inherited from v15.

Variants (ablation): see VARIANTS17.
"""
import asyncio
import time

from .executors import Executor, Suffix, _now_tm, phash
from .executors_v14 import NOTE_LINES, KNode, decision_end
from .executors_v15 import Unison15

K_DRAFT = 32  # per-request draft cap, the same as the SuffixDecoding baseline's

#                    key    plans  identity df     h      ctx    peers  pre    rationale
VARIANTS17 = {
    "":        ("nd",   "nd",   True,  True,  True,  True,  True,  True,  "keyed"),  # Unison
    "nopre":   ("nd",   "nd",   True,  True,  True,  True,  True,  False, "keyed"),  # no draft before a call's first pass
    "nopeer":  ("nd",   "nd",   True,  True,  True,  True,  False, True,  "keyed"),  # record without the peers' outputs
    "noctx":   ("nd",   "nd",   True,  True,  True,  False, True,  True,  "keyed"),  # record without the consumer's own context
    "noh":     ("nd",   "nd",   True,  True,  False, True,  True,  True,  "keyed"),  # drafter never given the consumer's state
    "noplan":  ("nd",   "self", True,  True,  True,  True,  True,  True,  "keyed"),  # plans decoded with SuffixDecoding
    "noid":    ("nd",   "nd",   False, True,  True,  True,  True,  True,  "keyed"),  # no identity sharing
    "nodf":    ("nd",   "nd",   True,  False, True,  True,  True,  True,  "keyed"),  # call-level boundaries only
    "rule":    ("coop", "self", True,  True,  True,  True,  True,  False, "keyed"),  # v15: rule-based cooperative drafting (= u15)
    # the same layer with SuffixDecoding as the proposer of every key: differs from Unison in the drafts only
    "suf":     ("self", "self", True,  True,  True,  True,  True,  False, "keyed"),  # (= u15-keyed)
    "sbd":     ("self", "self", True,  True,  True,  True,  True,  False, "call"),   # ... one rationale per call (= u15-sbd)
}
# the v15 variant that provides the key mode and the rationale program of a v17 variant
_V15 = {("nd", "keyed"): "", ("coop", "keyed"): "", ("self", "keyed"): "keyed", ("self", "call"): "sbd"}


class Unison17(Unison15):
    def __init__(self, engine, run_id, world=None, variant="", **kw):
        if variant not in VARIANTS17:
            raise ValueError(f"unknown Unison v17 variant {variant!r}")
        key, plans, identity, df, h, ctx, peers, pre, rationale = VARIANTS17[variant]
        self.use_pre = pre
        v15 = _V15[(key, rationale)]
        super().__init__(engine, run_id, world, variant=v15, **kw)
        if not identity:
            self.identity = False
        self.dataflow_boundaries = df
        self.key, self.plans = key, plans
        self.use_h, self.use_ctx, self.use_peers = h, ctx, peers
        self.variant = variant
        self.name = "u17" + (f"-{variant}" if variant else "")

    def _nd(self, group):
        c = self.E.cost  # the engine's own step-cost profile a + delta * B + ctok * D: sets the verification price
        return dict(mode="nd", group=group, cost=dict(a=float(c["a"]), delta=float(c["delta"]), ctok=float(c["ctok"])),
                    k=K_DRAFT, nd_h=self.use_h, nd_ctx=self.use_ctx, nd_peers=self.use_peers, pre=int(self.use_pre))

    # ------------------------------------------------------------------ plan calls
    async def plain(self, info, ids, max_tokens):
        if self.plans != "nd":
            return await super().plain(info, ids, max_tokens)
        t = time.time()
        ph = phash(ids)
        fut = self.plan_table.get(ph) if self.identity else None
        if fut is not None:  # identical program state: identity sharing
            toks = await asyncio.shield(fut)
            self._rec(info, ids, toks, _now_tm(t), t, source="exact")
            return list(toks)
        if self.identity:
            fut = self.plan_table[ph] = asyncio.get_running_loop().create_future()
        self.n_seq += 1
        rid = f"{self.run}-q{self.n_seq}"
        try:
            _, toks, tm = await self._engine(ids, max_tokens, self._nd(f"q{info.get('qid', info.get('team'))}"), rid=rid)
        except Exception as e:
            if fut is not None:
                fut.set_exception(e)
            raise
        toks = list(toks)
        if fut is not None:
            fut.set_result(toks)
        self._rec(info, ids, toks, tm, t, source="engine", rid=rid)
        return toks

    # ------------------------------------------------------------------ verification calls: key
    async def verify(self, info, ids, max_tokens):
        if self.key != "nd":
            return await super().verify(info, ids, max_tokens)
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
        # finished peers' decision prefixes are kept for the boundary check only (a key that ends exactly at a
        # finished peer's boundary with equal tokens is the same decision: certified cross-condition reuse)
        trie = self._trie(pid, ph if self.identity else None)
        ea = self._nd(f"p{pid}")
        ea.update(nl=self._nl_ids(), n_nl=self._nnl(), cands=[list(c.toks[:c.dend]) for c in trie],
                  dends=[c.dend for c in trie])
        src = None
        try:
            _, toks, tm = await self._engine(ids, max_tokens, ea, rid=node.rid)
            toks = list(toks)
            if toks and toks[-1] in self.E.eos:
                toks = toks[:-1]
            for c in trie:
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


class Collect(Executor):
    """Training-data run: every call executed plainly (no sharing, no drafts) with the hidden state that emitted each
    output token recorded by the engine plugin (UNISON_COLLECT), together with the call's group and timing."""
    name = "collect"
    verify_mode = plan_mode = {"mode": "none", "collect": True}

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.n_seq = 0
        self._nl = None

    def _boundary(self):
        """The decision boundary of a note as the engine sees it (newline tokens, lines of the decision): the
        engine keeps the hidden states of a verification call up to its boundary, the only positions a key has."""
        if self._nl is None:
            tok = self.E.tok
            self._nl = [i for i in range(len(tok)) if "\n" in tok.decode([i])]
        return dict(nl=self._nl, n_nl=NOTE_LINES.get(getattr(self, "note_format", ""), 3))

    async def plain(self, info, ids, max_tokens):
        t = time.time()
        self.n_seq += 1
        rid = f"{self.run}-q{self.n_seq}"
        group = f"q{info.get('qid', info.get('team'))}"
        _, toks, tm = await self._engine(ids, max_tokens, dict(self.plan_mode, group=group), rid=rid)
        self._rec(info, ids, toks, tm, t, source="engine", rid=rid, group=group)
        return toks

    async def verify(self, info, ids, max_tokens):
        t = time.time()
        self.n_seq += 1
        rid = f"{self.run}-k{self.n_seq}"
        group = f"p{info['pid']}"
        _, toks, tm = await self._engine(ids, max_tokens, dict(self.verify_mode, group=group, **self._boundary()), rid=rid)
        self._rec(info, ids, toks, tm, t, source="engine", rid=rid, group=group)
        return toks
