"""LEGACY (v9-v13). Not used by the v14 paper (paper/unison14); the v14 method is unison/executors_v14.py.
Kept only so that historical results can be reproduced. The variants "full", "single", "note", "bstop" and
"defer" copy a certified peer's full note; no v14 configuration does.

Decision-boundary certification (DBC) with decision-keyed provenance: the Unison execution policy for
evidence-verification calls.

A verification note is (decision, provenance). Downstream agents consume only the decision; the
provenance (one sentence of reasoning that justifies the evidence claim) is kept for audit. With the
decision serialized first, the decision of a greedy execution is a function of a short output *prefix*:
the tokens up to the decision boundary b (the end of the 'Support:' line).

Head: certificate. Let y_u be a peer's note on the same passage and w the consumer. If the consumer's
speculative-verification pass accepts y_u[0:b_u] (every token is w's own argmax under w's prompt, i.e.
w's condition and w's evidence version), then w's own execution would emit exactly y_u[0:b_u] and hence
reach exactly y_u's decision. The engine ends the request at the boundary.

Tail: decision-keyed provenance. The reasoning that justifies an evidence claim (answer a, stated by
sentence k of passage version v) is computed by a canonical program whose prompt contains only (v, k, a).
Every call that reaches the same claim on the same version therefore issues the *identical* provenance
prompt, and identical-prompt coalescing shares that computation exactly. Certification turns semantically
different calls into syntactically identical tails: the head is certified, the tail is coalesced, and no
part of a delivered note is estimated. A note that claims no evidence (Relevant: no) has no provenance.

Provenance is decoded off the agent's critical path, and strictly after foreground work: a provenance
request is admitted only when the engine has no foreground request waiting for a batch slot.

Peer-trie drafting. All distinct decision prefixes computed on the passage form a trie. At every step
the draft is the continuation (up to its boundary) of the peer prefix that matches what the consumer
has emitted so far. If none matches, the request continues with the same history speculator as the
strongest lossless baseline (SuffixDecoding), at no loss.
"""
import asyncio
import time
from collections import Counter, defaultdict

from .common import BOUNDARY_FIELD, normalize_answer, parse_note, parse_support, provenance_messages
from .engine import StepLog
from .executors import Executor, Suffix, _now_tm, phash

ANSWER_FIELD = "answer:"
# variants that return at the decision boundary and decode the provenance afterwards
DEFER = ("defer", "sbd", "sbdcanon", "canon", "cont", "canonnofg")
CANON = ("sbdcanon", "canon", "canonnofg")   # decision-keyed provenance (else: per-call continuation)
NOTRIE = ("nodraft", "lazy", "sbd", "sbdcanon")  # no peer drafts (the consumer decodes its own decision)
PROV_TOKENS = 64


def decision_end(tok, ids, field=ANSWER_FIELD):
    """Number of leading tokens that fix the decision of a decision-first note (through the newline that
    closes the boundary line). A note that ends inside its decision returns its full length."""
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


class DNode:
    __slots__ = ("rid", "pid", "version", "ph", "done", "toks", "dec", "dend", "ev")

    def __init__(self, rid, pid, version, ph):
        self.rid, self.pid, self.version, self.ph = rid, pid, version, ph
        self.done, self.toks, self.dec, self.dend = False, [], None, None
        self.ev = asyncio.Event()


class UnisonDBC(Executor):
    """Unison: identical prompts are coalesced; every other call is served by a verification pass that
    drafts the peer decision trie and ends at a certified decision boundary.

    variants:
      full      certify only: trie drafting, the request ends at a certified boundary, else full note
      single    one candidate, no trie;  note: certify the whole note;  nodraft: coalescing + suffix only
      bstop     Decide-then-Lookup: own decoding with the history speculator, stop at the boundary when the
                prefix equals a peer's (no peer drafts, full note otherwise)
      lazy      every request stops at its own decision boundary; no provenance is produced (lower bound)
      sbd       Self-Boundary + Deferred provenance (the strongest decision-first baseline): no peer drafts,
                every call returns at its own boundary, its reasoning is decoded afterwards as a continuation
                of the same request (prefix-cache hit), foreground-first
      sbdcanon  sbd with decision-keyed provenance (Unison without certification)
      cont      certification + per-call continuation provenance (Unison without decision-keyed provenance)
      canon     Unison: certification + decision-keyed provenance, foreground-first
      canonnofg canon without foreground-first admission of provenance
      defer     earlier design (certified notes inherit the peer's reasoning); kept for reference
    """
    plan_mode = Suffix.verify_mode

    def __init__(self, *a, variant="full", max_cands=8, **kw):
        super().__init__(*a, **kw)
        self.variant = variant
        self.max_cands = max_cands
        self.name = "unison" + ("" if variant == "full" else f"-{variant}")
        self._nl = None
        self.bg = []
        self.dag = defaultdict(list)
        self.by_ph = {}
        self.prov = {}       # decision-keyed provenance: prompt hash -> future of its tokens
        self.prov_recs = []  # one record per provenance request (engine or coalesced)
        self.n_seq = 0

    def _boundary(self, toks):
        return len(toks) if self.variant == "note" else decision_end(self.E.tok, toks, self._field())

    def _decision(self, toks):
        text = self.E.decode(toks)
        rel, ans = parse_note(text)
        return parse_support(text), rel, normalize_answer(ans)

    def _trie(self, pid, ph):
        """Distinct decision prefixes of the passage, most frequent decision class first, latest first."""
        fin = [u for u in self.dag[pid] if u.done and u.ph != ph and u.toks]
        if not fin:
            return []
        cnt = Counter(u.dec for u in fin)
        seen, out = set(), []
        for u in sorted(reversed(fin), key=lambda u: -cnt[u.dec]):
            key = tuple(u.toks[:u.dend])
            if key in seen:
                continue
            seen.add(key)
            out.append(u)
            if self.variant == "single" or len(out) >= self.max_cands:
                break
        return out

    async def _admit(self):
        """Foreground-first: a provenance request enters the engine only when no foreground request is
        waiting for a batch slot (the engine's waiting queue at its latest iteration is empty)."""
        if self.variant in ("defer", "canonnofg"):
            return
        while True:
            rec = StepLog.records
            if not rec or rec[-1][2] == 0 or time.time() - rec[-1][0] > 0.5:
                return
            await asyncio.sleep(0.01)

    async def _reason(self, node, ids, prefix, max_tokens, t_ret):
        """Per-call provenance: the continuation of the consumer's own decision prefix."""
        await self._admit()
        ea = dict(Suffix.verify_mode)
        ea["resume"] = len(prefix)
        _, rest, tm = await self._engine(list(ids) + prefix, max(1, max_tokens - len(prefix)), ea,
                                         rid=node.rid + "-r")
        node.toks = prefix + list(rest)
        self.prov_recs.append(dict(kind="cont", t_ret=t_ret, t_end=time.time(), n_out=len(rest), engine=True))

    async def _canon(self, info, head, t_ret):
        """Decision-keyed provenance: one canonical computation per (passage version, support, answer)."""
        text = self.E.decode(head)
        rel, ans = parse_note(text)
        k = parse_support(text)
        if not rel or k <= 0:
            return  # no evidence claim, nothing to justify
        w = self.world
        ids = self.E.chat_ids(provenance_messages(w.corpus[info["pid"]]["title"], w.text(info["pid"], info["version"]),
                                                  ans, k))
        key = phash(ids)
        fut = self.prov.get(key)
        if fut is not None:  # the identical provenance computation exists: share it
            toks = await asyncio.shield(fut)
            self.prov_recs.append(dict(kind="canon", t_ret=t_ret, t_end=time.time(), n_out=len(toks), engine=False))
            return
        fut = self.prov[key] = asyncio.get_running_loop().create_future()
        await self._admit()
        self.n_seq += 1
        _, toks, _ = await self._engine(ids, PROV_TOKENS, Suffix.verify_mode, rid=f"{self.run}-p{self.n_seq}")
        fut.set_result(list(toks))
        self.prov_recs.append(dict(kind="canon", t_ret=t_ret, t_end=time.time(), n_out=len(toks), engine=True))

    def _nnl(self):  # lines of the certified prefix: Relevant, Answer (+ Support)
        return 2 if getattr(self, "note_format", "") == "dfirst" else 3

    def _field(self):
        return BOUNDARY_FIELD.get(getattr(self, "note_format", ""), ANSWER_FIELD)

    def _nl_ids(self):
        if self._nl is None:
            tok = self.E.tok
            self._nl = [i for i in range(len(tok)) if "\n" in tok.decode([i])]
        return self._nl

    async def verify(self, info, ids, max_tokens):
        t_arr = time.time()
        pid, ph = info["pid"], phash(ids)
        u = self.by_ph.get(ph)
        if u is not None:  # identical prompt: coalesce with the earlier, identical computation
            await u.ev.wait()
            self._rec(info, ids, u.toks, _now_tm(t_arr), t_arr, source="certified", src=u.rid,
                      src_version=u.version)
            return list(u.toks)
        self.n_seq += 1
        node = DNode(f"{self.run}-d{self.n_seq}", pid, info["version"], ph)
        self.by_ph[ph] = node
        trie = [] if self.variant in NOTRIE else self._trie(pid, ph)
        self.dag[pid].append(node)
        defer = self.variant in DEFER
        ea = dict(Suffix.verify_mode)
        if trie:
            ea.update(mode="bstop" if self.variant == "bstop" else "dbc",
                      cands=[list(c.toks[:c.dend]) for c in trie], dends=[c.dend for c in trie])
        if self.variant == "lazy" or (defer and not trie):
            ea.update(mode="lazy", nl=self._nl_ids(), n_nl=self._nnl())
        elif defer:
            ea.update(nl=self._nl_ids(), n_nl=self._nnl())  # certified boundary, else the consumer's own boundary
        try:
            _, toks, tm = await self._engine(ids, max_tokens, ea, rid=node.rid)
            toks = list(toks)
            src = None
            if trie and toks and toks[-1] in self.E.eos:
                for c in trie:  # the engine ended the request at a certified decision boundary
                    if len(toks) - 1 == c.dend and toks[:-1] == c.toks[:c.dend] and len(c.toks) >= c.dend:
                        src = c
                        break
            t_ret = time.time()
            if src is not None and self.variant in ("full", "single", "note", "bstop", "defer"):
                toks = list(src.toks)  # earlier designs: the certified note carries the peer's tokens
            elif src is not None:
                toks = list(src.toks[:src.dend])  # certified head; the tail is the consumer's own (below)
            elif defer and toks and toks[-1] in self.E.eos:
                toks = toks[:-1]  # own decision head
            if defer and not (src is not None and self.variant == "defer"):
                head = list(toks)
                if self.variant in CANON:
                    self.bg.append(asyncio.ensure_future(self._canon(info, head, t_ret)))
                else:
                    self.bg.append(asyncio.ensure_future(self._reason(node, ids, head, max_tokens, t_ret)))
            node.toks = toks
            node.dec = self._decision(toks)
            node.dend = decision_end(self.E.tok, toks, self._field()) if defer else self._boundary(toks)
        finally:
            node.done = True
            node.ev.set()
        if src is not None:  # a certified note must not seed the trie twice
            self.dag[pid].remove(node)
        self._rec(info, ids, node.toks, tm, t_arr, source="dcert" if src is not None else "engine", rid=node.rid,
                  n_trie=len(trie), src=None if src is None else src.rid, dend=node.dend,
                  src_version=src.version if src is not None else info["version"])
        return node.toks
