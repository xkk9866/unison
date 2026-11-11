"""Consumer-conditioned replay (CCR): the Unison execution policy for evidence-verification calls."""
import asyncio
import time
from collections import Counter, defaultdict

from .common import normalize_answer, parse_note
from .executors import Executor, Suffix, _now_tm, phash
from .trace import decision_mask


class RNode:
    """A verification computation in the evidence DAG."""
    __slots__ = ("rid", "pid", "version", "ph", "done", "toks", "dec", "ev", "dmask")

    def __init__(self, rid, pid, version, ph):
        self.rid, self.pid, self.version, self.ph = rid, pid, version, ph
        self.done, self.toks, self.dec, self.dmask = False, [], None, None
        self.ev = asyncio.Event()


class Unison(Executor):
    """Consumer-conditioned replay (CCR), executed inside the serving engine.

    A consumer w that reads a passage on which peers already computed draws ONE candidate output y_u:
    a representative of the passage's most frequent decision class (any evidence version). y_u becomes
    w's first speculative draft. The engine's verification pass over that draft is exactly the
    teacher-forced replay of y_u under w's own prompt (w's condition, w's evidence version), so the
    same pass yields, at zero extra compute,
      certify  all drafted tokens accepted: y_u *is* w's greedy output (exact)
      commit   otherwise the Surprisal Survival law reads the trace sigma_t of that very pass; if
               P(agree) >= theta_eps (calibrated) the engine accepts the whole draft
      continue otherwise ordinary decoding resumes from the first divergence (which speculative
               decoding has already accepted), drafting from the history speculator
    The consumer never waits for anything but identical computations that started earlier.
    """
    plan_mode = Suffix.verify_mode

    def __init__(self, *a, ssn=None, eps=0.0, variant="full", workload=None, **kw):
        super().__init__(*a, **kw)
        self.ssn = ssn
        self.eps = eps
        self.variant = variant
        self.name = "unison" + (f"-e{eps:g}" if eps > 0 else "") + ("" if variant == "full" else f"-{variant}")
        self.theta = None
        if eps > 0:
            key = {"nolaw": "survival", "offline": "offline"}.get(variant, "deploy")
            tb = ssn.meta["thresholds_by"]
            if key == "deploy" and f"deploy_{workload}" in tb:  # self-calibrated on this workload's shadow traffic
                key = f"deploy_{workload}"
            self.theta = tb[key].get(f"{eps:.3f}", 1.01)
        self.horizon = None if variant == "fullreplay" else ssn.meta.get("horizon") if ssn is not None else None
        self.dag = defaultdict(list)
        self.by_ph = {}
        self.n_seq = 0

    def _decision(self, toks):
        rel, ans = parse_note(self.E.decode(toks))
        return rel, normalize_answer(ans)

    def _candidate(self, pid, ph):
        fin = [u for u in self.dag[pid] if u.done and u.ph != ph and u.toks]
        if not fin:
            return None
        if self.variant == "latest":  # ablation: the most recent computation, no decision-class prior
            return fin[-1]
        cnt = Counter(u.dec for u in fin)
        best = max(cnt.values())
        return [u for u in fin if cnt[u.dec] == best][-1]  # latest member of the most frequent class

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
        node = RNode(f"{self.run}-v{self.n_seq}", pid, info["version"], ph)
        self.by_ph[ph] = node
        # a candidate is drafted only when the pass can end in a commit (epsilon > 0) or in shadow
        # calibration; at epsilon = 0 a full-length draft could only certify, which history speculation
        # already does at lower verification cost
        drafting = (self.theta is not None or self.variant == "shadow") and self.variant != "noreplay"
        cand = self._candidate(pid, ph) if drafting else None
        self.dag[pid].append(node)
        if self.variant == "nospec":
            ea = {"mode": "none"}
        elif cand is None:
            ea = dict(Suffix.verify_mode)
        else:
            if cand.dmask is None:
                cand.dmask = decision_mask(self.E.tok, cand.toks).tolist()
            ea = dict(Suffix.verify_mode, mode="ccr", cand=list(cand.toks), dmask=cand.dmask, theta=self.theta,
                      log_q=True, k_spec=32, horizon=self.horizon)

        def on_tokens(toks):
            node.toks = toks

        try:
            _, toks, tm = await self._engine(ids, max_tokens, ea, rid=node.rid, on_tokens=on_tokens)
            toks = list(toks)
            if cand is not None and self.theta is not None and len(toks) < len(cand.toks) and toks[-1:]                     and toks[-1] in self.E.eos and toks[:-1] == list(cand.toks[:len(toks) - 1]):
                toks = list(cand.toks)  # in-pass commit: adopt the whole candidate
            node.toks = toks
            node.dec = self._decision(node.toks)
        finally:
            node.done = True
            node.ev.set()
        shared = cand is not None and node.toks == list(cand.toks)
        if shared:  # an adopted result is not independent evidence: it must not vote in the class prior
            self.dag[pid].remove(node)
        self._rec(info, ids, node.toks, tm, t_arr, source="ccr" if shared else "engine", rid=node.rid,
                  cand=None if cand is None else cand.rid, cand_dec=None if cand is None else list(cand.dec),
                  own_dec=list(node.dec), src_version=cand.version if shared else info["version"],
                  cand_version=None if cand is None else cand.version)
        return node.toks
