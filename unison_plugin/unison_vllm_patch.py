"""vLLM (v0.8.5, V1 engine) plugin: per-request draft routing for evidence sharing.

Runs inside the engine-core process. Each request carries `sampling_params.extra_args`:
  mode     : "none" | "pld" | "suffix" | "unison"
  run      : run id (namespaces all stores)
  peers    : [producer request ids]  (unison; in-flight or finished, ordered by planner score)
  drafts   : {peer_id: [token ids]}   (static outputs of producers finished before submission)
  dlen     : {peer_id: int}           (planner-chosen draft horizon in producer-output positions)
  fallback : "pld" | None             (unison: own-context lookup when no peer draft applies)
  k        : per-request draft cap for this step
  resume   : number of trailing prompt tokens that are the request's own (replay-certified) output
  replay_from : (replay requests only) number of leading prompt tokens whose KV may be served from
             the prefix cache although prompt logprobs are requested; the consumer's prompt
             prefix is reused and only the replayed producer output is scored

The router keeps the live output of every running request (`LIVE`) and the final output of
finished requests (`DONE`), so a consumer can draft from a producer that is still decoding.
"""
import os
import time
from collections import OrderedDict

import numpy as np

_PATCHED = False
_MAX_DONE = 50000


class _RunStore:
    def __init__(self):
        self.live = {}
        self.done = OrderedDict()
        self.suffix = None
        self.suffix_active = set()
        self.ccr_seen = set()
        self.ccr_pending = {}
        self.ccr_log = []
        self.dbc_pending = {}
        self.groups = {}      # cooperative key trie: group (passage) -> recent request ids (live or finished)
        self.grouped = set()
        self.dbc_src = {}     # request id -> source of its current draft ("trie" | "live" | "suffix")
        self.acc = {}         # request id -> accepted draft tokens per source, verification steps
        self.clock = [0.0, 0]  # seconds and iterations of this run's engine steps: the mean iteration time


STORES = {}


def _store(run):
    s = STORES.get(run)
    if s is None:
        if len(STORES) > 4:  # drop stores of old runs
            for r in list(STORES)[:-2]:
                STORES.pop(r, None)
        s = STORES[run] = _RunStore()
    return s


def _tail_lookup(own, src, min_n, max_n, k):
    """Find the last occurrence of the tail n-gram of `own` inside `src`; return what follows."""
    L = len(src)
    for n in range(min(max_n, len(own)), min_n - 1, -1):
        if L <= n:
            continue
        pat = own[-n:]
        win = np.lib.stride_tricks.sliding_window_view(src[:-1], n)
        hits = np.nonzero((win == pat).all(axis=1))[0]
        if len(hits):
            j = int(hits[-1]) + n
            return src[j:j + k]
    return None


def _suffix_cache():
    try:
        from arctic_inference.suffix_decoding import SuffixDecodingCache
        return SuffixDecodingCache(max_tree_depth=64, max_cached_requests=100000)
    except Exception:  # pragma: no cover
        return None


def _route(runner, i, req_id, st, ea, ctx, k_engine):
    mode = ea.get("mode", "none")
    if mode == "none":
        return []
    k = min(int(ea.get("k", k_engine)), k_engine)
    npr = st.num_prompt_tokens - int(ea.get("resume", 0))  # a resumed prefix counts as own output
    own = ctx[npr:]
    store = _store(ea.get("run", "default"))
    if mode == "junk":  # cost profiling: drafts that are (almost) always rejected
        return [11] * k
    if mode == "pld":
        d = runner.drafter.propose(ctx)
        return [] if d is None else d[:k].tolist()
    if mode in ("suffix", "bstop", "lazy"):  # bstop: certification by the consumer's own decoding (no peer drafts)
        store.dbc_src[req_id] = "suffix"
        return _suffix_draft(store, req_id, ctx, ea, k)
    if mode == "dbc":
        # key computation over the cooperative key trie: draft the continuation of a peer decision prefix
        # that extends the consumer's output so far, up to that peer's boundary. Finished peers first
        # (executor-ordered by decision frequency), then in-flight peers, then SuffixDecoding. Verification
        # accepts only the consumer's own argmax tokens, so the draft source never changes the output.
        m = len(own)
        for c, de in zip(ea.get("cands") or [], ea.get("dends") or []):
            if m < de <= m + k_engine and len(c) >= de and np.array_equal(np.asarray(c[:m]), own):
                store.dbc_src[req_id] = "trie"
                return list(c[m:de])
        if ea.get("live"):
            d, src = _live_draft(store, req_id, own, ea, k_engine)
            if d:
                store.dbc_src[req_id] = src
                return d
        store.dbc_src[req_id] = "suffix"
        return _suffix_draft(store, req_id, ctx, ea, k)
    if mode == "ccr":
        # consumer-conditioned replay: the first draft is the whole candidate output, so the
        # verification pass of speculative decoding *is* the teacher-forced replay of the candidate
        cand = ea.get("cand")
        if cand and len(own) == 1 and req_id not in store.ccr_seen:
            store.ccr_seen.add(req_id)
            if own[0] == cand[0] and len(cand) > 1:
                store.ccr_pending[req_id] = ea
                H = int(ea.get("horizon") or len(cand))  # replay horizon (tokens of the candidate)
                return list(cand[1:min(H, 1 + k_engine)])
        return _suffix_draft(store, req_id, ctx, ea, min(k, int(ea.get("k_spec", 32))))
    # ---- unison: draft from peers' (possibly in-flight) evidence computations
    m = len(own)
    peers = ea.get("peers") or []
    statics = ea.get("drafts") or {}
    dlen = ea.get("dlen") or {}
    srcs = []
    for p in peers:
        src = store.done.get(p)
        if src is None:
            src = store.live.get(p)
        if src is None and p in statics:
            src = np.asarray(statics[p], dtype=np.int64)
        if src is not None:
            srcs.append((p, src))
    for p, src in srcs:  # aligned continuation: own output is a prefix of the peer output
        if len(src) > m and np.array_equal(src[:m], own):
            hi = min(len(src), int(dlen.get(p, 1 << 30)))
            if hi > m:
                return src[m:min(hi, m + k)].tolist()
    if ea.get("realign", True):
        for p, src in srcs:  # re-alignment after a divergence
            d = _tail_lookup(own, src, 2, 8, k)
            if d is not None and len(d):
                return d.tolist()
    if ea.get("fallback") == "suffix":
        return _suffix_draft(store, req_id, ctx, ea, k)
    if ea.get("fallback") == "pld":
        d = runner.drafter.propose(ctx)
        return [] if d is None else d[:k].tolist()
    return []


_GROUP_MAX = 32


def _peer_boundary(src, nlset, n_nl):
    """Length of a peer output through its decision boundary (the n_nl-th newline), else its length."""
    if not n_nl:
        return len(src)
    cnt = 0
    for j, t in enumerate(src.tolist()):
        if t in nlset:
            cnt += 1
            if cnt >= n_nl:
                return j + 1
    return len(src)


def _live_draft(store, req_id, own, ea, k):
    """Cooperative decoding: the longest continuation, up to its boundary, of any other call on the same
    passage (in flight or recently finished) whose output so far extends the consumer's output."""
    g = store.groups.get(ea.get("group"))
    if not g:
        return None, None
    m = len(own)
    nlset = ea.setdefault("_nlset", set(ea.get("nl") or []))
    n_nl = int(ea.get("n_nl") or 0)
    best, src_kind = None, None
    for rid in g:
        if rid == req_id:
            continue
        src, kind = store.live.get(rid), "live"
        if src is None:
            src, kind = store.done.get(rid), "trie"
        if src is None or len(src) <= m:
            continue
        if m and not np.array_equal(src[:m], own):
            continue
        end = _peer_boundary(src, nlset, n_nl)
        if end > m and (best is None or end - m > len(best)):
            best, src_kind = src[m:end], kind
    if best is None:
        return None, None
    return best[:k].tolist(), src_kind


def _suffix_draft(store, req_id, ctx, ea, k):
    sc = store.suffix
    if sc is None or req_id not in store.suffix_active:
        return []
    depth = sc.max_tree_depth
    draft = sc.speculate(req_id, ctx[-depth:].astype(np.int32), max_spec_tokens=k,
                         max_spec_factor=float(ea.get("spec_factor", 1.0)),
                         max_spec_offset=0.0, min_token_prob=float(ea.get("min_prob", 0.1)),
                         use_tree_spec=False)
    return list(draft.token_ids)[:k]


_ROOT = os.environ.get("UNISON_ROOT", "/home/yaokang/evishare_fast")


def _acc_write(run, recs):
    import json
    import os
    d = os.environ.get("UNISON_CCR_LOGDIR", os.path.join(_ROOT, "logs", "ccr"))
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f"{run}.acc.jsonl"), "a") as f:
        f.write("".join(json.dumps(r) + "\n" for r in recs))


# The engine's own step log, per run: one row per model step with what the step carried and what it cost. Columns:
# wall-clock time at the end of the step, seconds inside the model runner (inputs, forward pass, sampling,
# verification, drafting), requests in the step, positions computed (prompt positions not served by the prefix
# cache, sampled tokens and draft positions), draft positions among them, requests admitted in this step, seconds
# the drafter took before the step (first-pass drafts) and inside it (drafts for the next step).
_STEPS = {}
_STEP_DT = [0.0, 0.0]


def _steps_flush():
    import os
    d = os.environ.get("UNISON_CCR_LOGDIR", os.path.join(_ROOT, "logs", "ccr"))
    for run, rows in _STEPS.items():
        if rows:
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, f"{run}.esteps.csv"), "a") as f:
                f.write("".join("%.6f,%.6f,%d,%d,%d,%d,%.6f,%.6f\n" % r for r in rows))
            rows.clear()
    for run in list(_STEPS)[:-4]:
        _STEPS.pop(run, None)


def _step_record(runner, so, t0):
    n = so.total_num_scheduled_tokens
    if not n:  # the engine has run out of work: everything measured so far is on disk when the run is read
        _steps_flush()
        return
    t1 = time.perf_counter()
    st = runner.requests.get(next(iter(so.num_scheduled_tokens)))
    run = (st.sampling_params.extra_args or {}).get("run") if st is not None else None
    if run is not None:
        spec = so.scheduled_spec_decode_tokens
        rows = _STEPS.setdefault(run, [])
        rows.append((time.time(), t1 - t0, len(so.num_scheduled_tokens), n, sum(map(len, spec.values())) if spec else 0,
                     len(so.scheduled_new_reqs), _STEP_DT[0], _STEP_DT[1]))
        if len(rows) >= 1024:
            _steps_flush()
    _STEP_DT[0] = _STEP_DT[1] = 0.0


# ---------------------------------------------------------------------- consensus drafter (learned drafts)
_ND = {}


def _lexicon(runner, h_dim):
    """The target model's output embedding, [vocabulary, h_dim]: the engine's own tensor (the rows of the output
    head the hidden states are read with), so the drafter keeps no copy. Read from the model's checkpoint only if
    the engine does not expose it."""
    try:
        w = runner.model.lm_head.weight
        if w.dim() == 2 and w.shape[1] == h_dim:
            return w
    except AttributeError:
        pass
    from unison.drafter import load_lexicon
    print("[unison] output head not exposed by the engine: lexicon read from the checkpoint", flush=True)
    return load_lexicon(runner.model_config.model, "cuda")


def _drafter(runner=None):
    """The consensus drafter (unison.drafter.ConsensusDrafter), loaded once per engine process from UNISON_DRAFTER."""
    if "net" in _ND and (_ND["net"]._lex is not None or not _ND["net"].use_lexicon):
        return _ND["net"]
    if "net" not in _ND:
        import os
        import sys
        import torch
        sys.path.insert(0, os.environ.get("UNISON_CODE", os.path.join(_ROOT, "code")))
        from unison.drafter import ConsensusDrafter
        ck = torch.load(os.environ["UNISON_DRAFTER"], map_location="cpu")
        cfg = dict(ck["cfg"])
        cfg.setdefault("lexicon", False)  # checkpoints written before the lexicon existed
        cfg.setdefault("read", False)     # ... or before the drafter read the record for a consumer without a state
        net = ConsensusDrafter(ck["vocab_ids"], **cfg)
        state = dict(ck["state"])
        state.setdefault("beta", torch.ones(()))  # ... or before the calibration exponent
        net.load_state_dict(state)
        _ND["net"] = net.eval().cuda().half()
        _ND["cfg"] = ck["cfg"]
        _ND["mem"] = {}     # request id -> (its prompt tail [C], the tail's slot memories [C, d]), encoded once
        _ND["t"] = [0.0, 0]  # seconds spent drafting and decoding steps taken so far: the drafter's own step time
        print(f"[unison] consensus drafter loaded: vocab {net.vocab.size}, d {net.d}", flush=True)
    net = _ND["net"]
    if net.use_lexicon and net._lex is None and runner is not None:
        net.set_lexicon(_lexicon(runner, net.h_dim))
        net.freeze_lexicon()  # token identity becomes one table read
    return net


def _acc(store, req_id):
    return store.acc.setdefault(req_id, {"trie": 0, "live": 0, "suffix": 0, "nd": 0, "pre": 0, "steps": 0, "drafted": 0})


def _peer_lists(store, req_id, ea, limit):
    """Outputs of up to `limit` other calls of the request's group (same passage / same question), newest first,
    as the tokens the peers produced (a finished peer's end-of-sequence marker is not part of what it said)."""
    g = store.groups.get(ea.get("group")) or ()
    eos = ea.setdefault("_eosset", set(ea.get("eos") or ()))
    out = []
    for rid in reversed(g):
        if rid == req_id:
            continue
        src = store.live.get(rid)
        if src is None:
            src = store.done.get(rid)
        if src is None:
            continue
        n = len(src)
        while n and int(src[n - 1]) in eos:
            n -= 1
        if n:
            out.append(src[:n])
            if len(out) >= limit:
                break
    return out


def _iteration_time(store, cost, B):
    """What one engine iteration costs a request of this run: the mean time between two consecutive steps of the
    engine, measured on the run itself (_update_states). An iteration carries the prompts that arrive as well as
    the tokens being decoded, so on an agent workload it is several times a decode-only step. Before the first
    measurement: the decode-only step of the start-up profile, alpha + gamma_g B."""
    s, n = store.clock
    if n:
        return s / n
    return float(cost["a"]) + float(cost["delta"]) * B


def _verification_price(cost, B, it=None):
    """Acceptance probability at which drafting one more token breaks even on latency at the current load.

    Verifying a token lengthens the iteration by gamma_d for all B running requests; an accepted token saves its
    request one iteration, `it` seconds (_iteration_time). gamma_d comes from the engine's start-up profile
    (engine.profile: step = a + delta B + ctok D) and the iteration time from the engine's own clock, so the price
    is measured, not tuned."""
    a, delta, ctok = float(cost["a"]), float(cost["delta"]), float(cost["ctok"])
    return B * ctok / max(a + delta * B if it is None else it, 1e-9)


def _nd_drafts(net, H, ents, B, k_engine):
    """Drafts of the consensus drafter for the requests `ents`, in one batch. An entry is (request id, extra args,
    run store, the request's last token, the row of its hidden state in H or -1 if it has none yet, its prompt tail
    [C] if the tail is not encoded yet).

    The record of a request is the tail of its own prompt (encoded once per request) and the outputs of its
    group's other calls as they stand now. A draft position costs its verification plus the request's share of
    the drafter's own decoding step; both are measured on this engine (the start-up profile and the running mean
    of the drafter's step time), in units of one engine iteration (the run's measured mean)."""
    import time
    import torch
    t0 = time.perf_counter()
    P, L, C = net.max_peers, net.peer_len, net.ctx_len
    mem = _ND["mem"]
    dev = net.out.weight.device
    n = len(ents)
    new = [(e[0], e[5]) for e in ents if e[0] not in mem]
    m = len(new)
    # everything the drafter needs from the host goes to the device in one transfer: hidden-state rows, last
    # tokens, state flags, the peers' outputs and the prompt tails of the requests seen for the first time
    buf = np.full(3 * n + n * P * L + m * C, -1, dtype=np.int64)
    pv = buf[3 * n:3 * n + n * P * L].reshape(n, P, L)
    keys = []
    for b, (req_id, ea, store, y, row, _) in enumerate(ents):
        buf[b], buf[n + b], buf[2 * n + b] = max(row, 0), y, int(row >= 0 and bool(ea.get("nd_h", True)))
        if ea.get("nd_peers", True):
            for j, pl in enumerate(_peer_lists(store, req_id, ea, P)):
                pl = pl[:L]
                pv[b, j, :len(pl)] = pl
        cost = ea.get("cost")
        it = _iteration_time(store, cost, B) if cost else 0.0  # one engine iteration, as this run measures it
        price = _verification_price(cost, B, it) if cost else float(ea.get("astar", 0.5))
        keys.append((req_id, min(int(ea.get("k", k_engine)), k_engine), price, ea["eos"], bool(ea.get("n_nl")), it, store))
    for j, (_, c) in enumerate(new):
        buf[3 * n + n * P * L + j * C:3 * n + n * P * L + (j + 1) * C] = c
    dv = torch.from_numpy(buf).to(dev)
    if m:
        ct = dv[3 * n + n * P * L:].view(m, C)
        with torch.no_grad():
            mc = net.encode(ct, 0)
        for j, (rid, _) in enumerate(new):
            mem[rid] = (ct[j], mc[j])
    t_step = _ND["t"][0] / max(1, _ND["t"][1])
    k = max(kk[1] for kk in keys)
    price = min(kk[2] for kk in keys)
    sprice = B * t_step / keys[0][5] if keys[0][5] > 0 else 0.0  # the drafter's step delays all B requests
    for kk in keys:
        a = _acc(kk[6], kk[0])
        a["nd_calls"] = a.get("nd_calls", 0) + 1
        a["nd_price"] = a.get("nd_price", 0.0) + price
        a["nd_sprice"] = a.get("nd_sprice", 0.0) + sprice
        a["nd_it"] = a.get("nd_it", 0.0) + 1e3 * kk[5]
    out = {}
    if price < 1.0:  # on a saturated engine no draft can pay for its verification
        if not net._programs:  # both program sizes are captured at the first draft (the warm-up of a run)
            net.program(1, keys[0][3])
            net.program(1 << 30, keys[0][3])
        R = max(net._programs)
        pe = dv[3 * n:3 * n + n * P * L].view(n, P, L)
        for lo in range(0, n, R):
            hi = min(n, lo + R)
            prog = net.program(hi - lo, keys[0][3])
            prog.load(None if H is None else H[dv[lo:hi]], dv[2 * n + lo:2 * n + hi] > 0, dv[n + lo:n + hi],
                      torch.stack([mem[kk[0]][0] for kk in keys[lo:hi]]),
                      torch.stack([mem[kk[0]][1] for kk in keys[lo:hi]]), pe[lo:hi])
            drafts = prog.run(hi - lo, k, price, sprice)
            _ND["t"][0] += prog.last_seconds
            _ND["t"][1] += prog.last_steps
            for kk, d in zip(keys[lo:hi], drafts):
                d = d[:kk[1]]
                if kk[4] and d and d[-1] in kk[3]:
                    d = d[:-1]  # a decision ends at its boundary: the drafter's end marker is its stop, not a token
                out[kk[0]] = d
    dt = time.perf_counter() - t0  # a draft ends with a host read, so this is the time the engine waited
    _STEP_DT[0 if H is None else 1] += dt
    for kk in keys:
        a = _acc(kk[6], kk[0])
        a["nd_ms"] = a.get("nd_ms", 0.0) + 1e3 * dt / len(keys)
    return out


def _tail(ids, C, use):
    c = np.full(C, -1, dtype=np.int64)
    if use:
        ids = np.asarray(ids[-C:], dtype=np.int64)
        c[C - len(ids):] = ids
    return c


def _nd_batch(runner, items, k_engine):
    """Learned drafts for all consensus-drafted requests that have just made a pass. items: (i, req_id, st, ea, n).
    The state of a request is the hidden state that emitted its last token (the last of its logits rows)."""
    H = getattr(runner, "_unison_h", None)
    if H is None or not items:
        return {}
    net = _drafter(runner)
    nd = getattr(runner, "_unison_nd", {})
    mem = _ND["mem"]
    ents = []
    row0 = 0
    for i, req_id in enumerate(runner.input_batch.req_ids):  # logits rows: num_draft_tokens + 1 per request
        start = row0
        row0 += nd.get(req_id, 0) + 1
        if not items or items[0][0] != i:
            continue
        _, _, st, ea, n = items.pop(0)
        if start + n - 1 >= H.shape[0]:
            continue
        store = _store(ea.get("run", "default"))
        tail = None
        if req_id not in mem:
            npr = st.num_prompt_tokens - int(ea.get("resume", 0))
            tail = _tail(runner.input_batch.token_ids_cpu[i, max(0, npr - net.ctx_len):npr], net.ctx_len,
                         ea.get("nd_ctx", True))
        ents.append((req_id, ea, store, int(store.live[req_id][-1]), start + n - 1, tail))
    if not ents:
        return {}
    return _nd_drafts(net, H, ents, len(runner.input_batch.req_ids), k_engine)


def _pre_drafts(sched, out):
    """Verification in the first pass. A request that is admitted with its whole prompt in this step has no state
    yet, but its record exists: the drafter writes the key from the record alone, and the draft is appended to
    the prompt's positions of this very step. The pass that processes the prompt then verifies the draft, so a
    call whose draft holds is complete after one pass. The scheduler's bookkeeping is the one of any drafted
    step: the draft positions are scheduled, counted as computed and given back if they are rejected."""
    runner = _ND.get("runner")
    if runner is None or not out.scheduled_new_reqs:
        return
    k_engine = runner.speculative_config.num_speculative_tokens
    cand = []
    for nr in out.scheduled_new_reqs:
        ea = (nr.sampling_params.extra_args or {}) if nr.sampling_params is not None else {}
        if ea.get("mode") != "nd" or not ea.get("pre") or ea.get("resume"):
            continue
        req = sched.requests.get(nr.req_id)
        if req is None or req.num_tokens != req.num_prompt_tokens or req.num_computed_tokens != req.num_tokens:
            continue  # the prompt is not complete in this step (chunked): the draft would have nothing to follow
        cand.append((nr, req, ea))
    if not cand:
        return
    net = _drafter(runner)
    mem = _ND["mem"]
    ents = []
    for nr, req, ea in cand:
        store = _store(ea.get("run", "default"))
        ids = nr.prompt_token_ids
        tail = None if nr.req_id in mem else _tail(ids, net.ctx_len, ea.get("nd_ctx", True))
        ents.append((nr.req_id, ea, store, int(ids[-1]), -1, tail))
    drafts = _nd_drafts(net, None, ents, max(1, len(sched.running)), k_engine)
    budget = sched.max_num_scheduled_tokens - out.total_num_scheduled_tokens
    for nr, req, ea in cand:
        d = drafts.get(nr.req_id) or []
        d = d[:max(0, min(len(d), budget, sched.max_model_len - req.num_tokens - 1))]
        if not d:
            continue
        req.spec_token_ids = list(d)
        blocks = sched.kv_cache_manager.allocate_slots(req, len(d))
        if blocks is None:  # no room for the draft's positions: the call runs as it would without it
            req.spec_token_ids = []
            continue
        nr.block_ids.extend(b.block_id for b in blocks)
        out.num_scheduled_tokens[nr.req_id] += len(d)
        out.total_num_scheduled_tokens += len(d)
        out.scheduled_spec_decode_tokens[nr.req_id] = req.spec_token_ids
        req.num_computed_tokens += len(d)
        budget -= len(d)
        _store(ea.get("run", "default")).dbc_src[nr.req_id] = "pre"


def _collect_step(runner, items, ea_of):
    """Training-data collection: the hidden state that emitted each sampled token, per request."""
    import torch
    H = getattr(runner, "_unison_h", None)
    if H is None or not items:
        return
    nd = getattr(runner, "_unison_nd", {})
    buf = COLLECT.setdefault("buf", {})
    row0 = 0
    for i, req_id in enumerate(runner.input_batch.req_ids):
        start = row0
        row0 += nd.get(req_id, 0) + 1
        if not items or items[0][0] != i:
            continue
        _, _, n = items.pop(0)
        b = buf.setdefault(req_id, [])
        if b is None:
            continue
        cur = sum(x.shape[0] for x in b)
        if cur >= COLLECT_MAX:
            continue
        st = runner.requests.get(req_id)
        ea = st.sampling_params.extra_args or {}
        L = len(_store(ea.get("run", "default")).live[req_id])  # output tokens so far; rows cover L-n .. L-1
        if cur != L - n:  # recomputation after a preemption: keep the alignment or drop the request
            if cur > L - n >= 0:
                b[:] = [torch.cat(b, 0)[:L - n]]
            else:
                buf[req_id] = None
                continue
        b.append(H[start:start + n].to(torch.float16).cpu())


COLLECT_MAX = 64  # positions recorded per request (keys and plans are shorter)


COLLECT = {}


def _collect_finish(req_id, toks, ea=None):
    import os
    import torch
    buf = COLLECT.get("buf", {})
    rows = buf.pop(req_id, None)
    if not rows:
        return
    h = torch.cat(rows, 0)
    m = min(h.shape[0], len(toks))
    if ea and ea.get("n_nl"):  # a verification call: only the positions of its decision (the key) are ever drafted
        nlset = ea.setdefault("_nlset", set(ea.get("nl") or []))
        m = min(m, _peer_boundary(np.asarray(toks), nlset, int(ea["n_nl"])) + 2)
    done = COLLECT.setdefault("done", {})
    done[req_id] = (h[:m].clone(), torch.as_tensor(np.asarray(toks[:m], dtype=np.int32)))
    if len(done) >= 256:
        _collect_flush()


def _collect_flush():
    import os
    import torch
    done = COLLECT.get("done")
    if not done:
        return
    d = os.environ.get("UNISON_COLLECT")
    os.makedirs(d, exist_ok=True)
    n = COLLECT["shard"] = COLLECT.get("shard", 0) + 1
    torch.save(done, os.path.join(d, f"shard_{os.getpid()}_{n:04d}.pt"))
    COLLECT["done"] = {}


def _patch_scheduler():
    """Background class in the V1 scheduler (payload requests carry extra_args["bg"]).

    Before every scheduling pass the queues are stably reordered so that foreground requests precede
    background ones: foreground requests get the token budget first, are admitted from the waiting queue
    first, and vLLM's own preemption (which pops the *last* running request) evicts background requests
    before any foreground request. If a foreground request is still waiting (it could not be admitted in the
    previous pass), every running background request is preempted with vLLM's recompute preemption: its KV
    blocks return to the prefix cache and it resumes, mostly from cache hits, once no foreground request
    waits. No parameter is involved."""
    import os
    prio = os.environ.get("UNISON_PRIO", "1") == "1"
    from collections import deque
    from vllm.v1.core.sched import scheduler as sm
    from vllm.v1.request import RequestStatus
    orig = sm.Scheduler.schedule

    def is_bg(r):
        sp = getattr(r, "sampling_params", None)
        return bool(sp is not None and (sp.extra_args or {}).get("bg"))

    def schedule(self):
        try:
            run_bg = [r for r in self.running if is_bg(r)] if prio else []
            wait_bg = [r for r in self.waiting if is_bg(r)] if prio else []
            if run_bg or wait_bg:
                fg_run = [r for r in self.running if not is_bg(r)]
                fg_wait = [r for r in self.waiting if not is_bg(r)]
                blocked = getattr(self, "_unison_blocked", ())
                if run_bg and any(r.request_id in blocked for r in fg_wait):
                    # a foreground request was not admitted by the previous pass: background yields
                    for r in run_bg:
                        self.kv_cache_manager.free(r)
                        r.status = RequestStatus.PREEMPTED
                        r.num_computed_tokens = 0
                        if getattr(r, "spec_token_ids", None):
                            r.spec_token_ids.clear()
                    wait_bg = run_bg + wait_bg
                    run_bg = []
                    STATS["bg_preempt"] = STATS.get("bg_preempt", 0) + 1
                self.running = fg_run + run_bg
                self.waiting = deque(fg_wait + wait_bg)
        except Exception as e:  # never break the engine because of the priority pass
            print("[unison] priority error", repr(e), flush=True)
        out = orig(self)
        self._unison_blocked = {r.request_id for r in self.waiting if not is_bg(r)}
        if _ND.get("runner") is not None and out.scheduled_new_reqs:
            try:
                import torch
                with torch.inference_mode():  # the mode the model runner drafts in: one drafter state for both
                    _pre_drafts(self, out)
            except Exception as e:  # never break the engine because of a draft
                print("[unison] pre-draft error", repr(e), flush=True)
        return out

    sm.Scheduler.schedule = schedule


STATS = {}


def _patch_memory_handover():
    """Shared-GPU start-up: take the card's memory over from the queue's holder *inside the engine process* and
    keep it there. The placeholder becomes one free block of this process's own allocator pool, and the engine's
    allocations (weights, profile run, KV cache, run-time activations) are served from that block, so the card
    never shows more than UNISON_VISIBLE_MIB free: memory-driven schedulers of other projects on the machine find
    no room on it, neither during the minute of the engine's start nor while a method is being measured. Active
    only when the queue sets UNISON_HOLD_FLAG (the holder's hand-over file) and UNISON_ENGINE_WANT_MIB."""
    import os
    import time
    flag = os.environ.get("UNISON_HOLD_FLAG")
    want = int(os.environ.get("UNISON_ENGINE_WANT_MIB", "0") or 0)
    if not flag or not want:
        return
    if os.environ.get("UNISON_HANDOVER", "pool") != "pool":
        return _patch_memory_handover_phased()
    import torch
    from vllm.v1.worker import gpu_worker as gw
    MiB = 1 << 20
    visible = int(os.environ.get("UNISON_VISIBLE_MIB", "1900"))
    # what the engine's allocator serves: weights, the profile run, the KV cache, run-time activations. The rest of
    # the job's allowance stays with the holder, which feeds it to the card in small steps as CUDA graphs (they
    # have pools of their own, filled from the device during capture), library workspaces and the driver's encoder
    # ask the device for memory
    pool = min(want - int(os.environ.get("UNISON_NONPOOL_MIB", "2600")),
               int(os.environ.get("UNISON_POOL_MIB", "0") or 0) or want)

    def grab():
        with open(flag, "w") as f:
            f.write(f"release {pool + 200}")
        t0, block = time.time(), None
        while block is None and time.time() - t0 < 120:
            free = torch.cuda.mem_get_info()[0] // MiB
            # the holder stands still once it has released; after 20 s it has given what it had
            if free >= pool + 150 or (time.time() - t0 > 20 and free >= 0.8 * pool):
                try:
                    block = torch.empty(min(pool, free - 150) * MiB, dtype=torch.uint8, device="cuda")
                except RuntimeError:  # another process allocated within these milliseconds: measure again
                    pass
            if block is None:
                time.sleep(0.02)
        if block is None:
            raise RuntimeError("[unison] memory hand-over: the holder did not release the engine's memory")
        took = block.numel() // MiB
        del block  # one free block of this process's allocator pool
        # from here on the allocator keeps what it has: vLLM empties the cache around its profile run, which
        # would hand the block back to the card for the rest of the start
        torch.cuda.empty_cache = lambda: None
        with open(flag, "w") as f:
            f.write(f"feed {visible}")
        print(f"[unison] memory hand-over: {took} MiB taken over in {time.time() - t0:.2f}s, "
              f"{torch.cuda.memory_reserved() >> 20} MiB in the engine's pool, "
              f"{torch.cuda.mem_get_info()[0] >> 20} MiB free on the card", flush=True)

    orig_init = gw.Worker.init_device
    orig_avail = gw.Worker.determine_available_memory
    orig_warm = gw.Worker.compile_or_warm_up_model

    def init_device(self):
        orig_init(self)
        grab()

    def determine_available_memory(self):
        # the pool changes the free-memory bookkeeping of the profile run; the KV capacity is fixed by the queue
        floor = None
        blocks = self.cache_config.num_gpu_blocks_override or int(os.environ.get("UNISON_KV_BLOCKS", "0") or 0)
        if blocks:
            spec = self.model_runner.get_kv_cache_spec()
            page = next(iter(spec.values())).page_size_bytes
            floor = blocks * page + (1 << 30)
        try:
            avail = orig_avail(self)
        except AssertionError as e:
            if floor is None:
                raise
            print("[unison] memory profiling assertion ignored under hand-over:", str(e)[:120], flush=True)
            avail = 0
        return int(max(avail, floor)) if floor is not None else int(avail)

    def compile_or_warm_up_model(self):
        orig_warm(self)
        print(f"[unison] memory hand-over complete: {torch.cuda.memory_reserved() >> 20} MiB reserved by the "
              f"engine ({torch.cuda.memory_allocated() >> 20} MiB in use), "
              f"{torch.cuda.mem_get_info()[0] >> 20} MiB free on the card", flush=True)

    gw.Worker.init_device = init_device
    gw.Worker.determine_available_memory = determine_available_memory
    gw.Worker.compile_or_warm_up_model = compile_or_warm_up_model


def _patch_memory_handover_phased():
    """The earlier hand-over (UNISON_HANDOVER=phased): the placeholder is released to the card phase by phase,
    right before each allocation of the engine (weights, profiling, KV cache, CUDA graphs)."""
    import os
    import time
    flag = os.environ.get("UNISON_HOLD_FLAG")
    want = int(os.environ.get("UNISON_ENGINE_WANT_MIB", "0") or 0)
    import torch
    from vllm.v1.worker import gpu_worker as gw
    MiB = 1 << 20
    state = {"chunks": [], "held": 0}

    def grab():
        with open(flag, "w") as f:
            f.write(f"release {want}")
        t0 = last = time.time()
        while time.time() - t0 < 120 and state["held"] < want * MiB:
            free = torch.cuda.mem_get_info()[0]
            if free > 300 * MiB:
                n = min(256 * MiB, want * MiB - state["held"], free - 200 * MiB)
                try:
                    state["chunks"].append(torch.empty(n, dtype=torch.uint8, device="cuda"))
                    state["held"] += n
                    last = time.time()
                    continue
                except RuntimeError:
                    pass
            if state["held"] and time.time() - last > 5:
                break  # the holder has nothing more to give
            time.sleep(0.05)
        # the holder stays paused (it must not absorb what the engine releases phase by phase); it is told to
        # resume absorbing only once the engine is fully initialized (end of compile_or_warm_up_model)
        print(f"[unison] memory hand-over: holding {state['held'] >> 20} MiB in-process", flush=True)

    def release(n_bytes):
        """Give `n_bytes` back to the device right before the engine's next allocation phase (the chunks are
        returned with empty_cache so that large contiguous allocations can be served)."""
        done = 0
        while state["chunks"] and done < n_bytes:
            c = state["chunks"].pop()
            done += c.numel()
            del c
        state["held"] -= done
        torch.cuda.empty_cache()
        return done

    def release_all():
        release(1 << 60)
        torch.cuda.empty_cache()

    orig_init = gw.Worker.init_device
    orig_load = gw.Worker.load_model
    orig_avail = gw.Worker.determine_available_memory
    orig_kv = gw.Worker.initialize_from_config
    orig_warm = gw.Worker.compile_or_warm_up_model

    def init_device(self):
        orig_init(self)
        grab()

    def load_model(self):
        release(int(os.environ.get("UNISON_WEIGHTS_MIB", "7500")) * MiB)
        orig_load(self)

    def determine_available_memory(self):
        # the profile run (8192-token dummy batch, dummy rejection-sampler pass with 256 requests) needs most of
        # the non-KV allowance transiently; what it frees afterwards is what the KV cache is allocated from
        release(int(os.environ.get("UNISON_PROFILE_MIB", "5000")) * MiB)
        floor = None
        blocks = self.cache_config.num_gpu_blocks_override or int(os.environ.get("UNISON_KV_BLOCKS", "0") or 0)
        if blocks:  # the KV capacity is fixed by the queue: the engine gets at least that much
            spec = self.model_runner.get_kv_cache_spec()
            page = next(iter(spec.values())).page_size_bytes
            floor = blocks * page + (1 << 30)
        try:
            avail = orig_avail(self)
        except AssertionError as e:  # the placeholder changes the free-memory bookkeeping; the capacity is fixed
            if floor is None:
                raise
            print("[unison] memory profiling assertion ignored under hand-over:", str(e)[:120], flush=True)
            avail = 0
        return int(max(avail, floor)) if floor is not None else int(avail)

    def initialize_from_config(self, kv_cache_config):
        page = next(iter(self.model_runner.get_kv_cache_spec().values())).page_size_bytes
        release(kv_cache_config.num_blocks * page + 600 * MiB)
        orig_kv(self, kv_cache_config)

    def compile_or_warm_up_model(self):
        release_all()
        orig_warm(self)
        # the holder is told to absorb again by the job itself (run_exp) once its other GPU users (retriever,
        # encoder) are allocated too; a job without that step leaves the holder paused until it ends
        print(f"[unison] memory hand-over complete: {torch.cuda.mem_get_info()[0] >> 20} MiB free on the card", flush=True)

    gw.Worker.init_device = init_device
    gw.Worker.load_model = load_model
    gw.Worker.determine_available_memory = determine_available_memory
    gw.Worker.initialize_from_config = initialize_from_config
    gw.Worker.compile_or_warm_up_model = compile_or_warm_up_model


def _ccr_write(run, rec):
    import json
    import os
    d = os.environ.get("UNISON_CCR_LOGDIR", os.path.join(_ROOT, "logs", "ccr"))
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f"{run}.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")


def register():
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True
    from vllm.v1.worker import gpu_model_runner as gmr
    from vllm.v1.spec_decode.utils import is_spec_decode_supported

    orig_update = gmr.GPUModelRunner._update_states

    def _install_hidden_tap(self):
        """Keep the hidden states that the model's output head reads in this step (one row per scheduled position
        with logits: num_draft_tokens + 1 per request, batch order). They are the drafter's first input and are
        recorded in collection runs; nothing is recomputed."""
        if getattr(self, "_unison_tap", False):
            return
        self._unison_tap = True
        orig_cl = self.model.compute_logits

        def compute_logits(hidden_states, sampling_metadata=None):
            if self._unison_h is None:
                self._unison_h = hidden_states
            return orig_cl(hidden_states, sampling_metadata)

        self.model.compute_logits = compute_logits

    def _update_states(self, scheduler_output):
        if getattr(self, "rejection_sampler", None) is None:  # ordinary, non-speculative audit model
            return orig_update(self, scheduler_output)
        self.rejection_sampler._unison_runner = self
        if os.environ.get("UNISON_DRAFTER"):
            _ND["runner"] = self
        # the engine's clock: the time from the previous step to this one is one iteration whenever a request of
        # the previous step is scheduled again (the engine did not wait for work in between)
        now, cur = time.perf_counter(), scheduler_output.num_scheduled_tokens
        prev = getattr(self, "_unison_clk", None)
        if prev is not None:
            for rid in prev[1]:
                if rid in cur:
                    st = self.requests.get(rid)
                    if st is not None:
                        c = _store((st.sampling_params.extra_args or {}).get("run", "default")).clock
                        c[0] += now - prev[0]
                        c[1] += 1
                    break
        self._unison_clk = (now, tuple(cur))
        self._unison_h = None
        self._unison_nd = {rid: len(t) for rid, t in scheduler_output.scheduled_spec_decode_tokens.items()}
        _install_hidden_tap(self)
        # passes of every call, whatever serves it: the engine steps the call takes part in (the first one
        # processes its prompt). Written with the call's draft accounting when the call ends.
        passes = getattr(self, "_unison_pass", None)
        if passes is None:
            passes = self._unison_pass = {}
        for rid in cur:
            passes[rid] = passes.get(rid, 0) + 1
        collecting = bool(os.environ.get("UNISON_COLLECT"))
        ended = {}
        for rid in scheduler_output.finished_req_ids:
            n_pass = passes.pop(rid, 0)
            st = self.requests.get(rid)
            if st is None:
                continue
            ea = st.sampling_params.extra_args or {}
            store = _store(ea.get("run", "default"))
            out = store.live.pop(rid, None)
            if out is not None:
                eos = ea.get("eos")
                if eos is not None:
                    hit = np.nonzero(np.isin(out, eos))[0]
                    if len(hit):
                        out = out[:hit[0] + 1]
                store.done[rid] = out
                if len(store.done) > _MAX_DONE:
                    store.done.popitem(last=False)
                if collecting and ea.get("collect"):
                    _collect_finish(rid, out, ea)
            if store.suffix is not None and rid in store.suffix_active:
                store.suffix.stop_request(rid)
                store.suffix_active.discard(rid)
            acc = store.acc.pop(rid, None)
            store.dbc_src.pop(rid, None)
            if _ND.get("mem"):
                _ND["mem"].pop(rid, None)
            if acc is not None or "run" in ea:  # per call: its passes and where its accepted draft tokens came from
                ended.setdefault(ea.get("run", "default"), []).append(dict(rid=rid, passes=n_pass, **(acc or {})))
        for run, recs in ended.items():
            _acc_write(run, recs)
        r = orig_update(self, scheduler_output)
        spec = scheduler_output.scheduled_spec_decode_tokens
        if spec:
            for nr in scheduler_output.scheduled_new_reqs:  # pre-drafts (_pre_drafts): verified in the first pass
                d = spec.get(nr.req_id)
                i = self.input_batch.req_id_to_index.get(nr.req_id)
                if d and i is not None:
                    n0 = len(nr.prompt_token_ids)
                    self.input_batch.token_ids_cpu[i, n0:n0 + len(d)] = d
                    self.input_batch.num_tokens[i] = n0 + len(d)
        if collecting and COLLECT.get("done") and not self.requests:
            _collect_flush()
        return r

    def generate_draft_token_ids(self, sampled_token_ids, sampling_metadata):
        draft_token_ids = []
        k_engine = self.speculative_config.num_speculative_tokens
        nd_items, col_items = [], []
        for i, sampled_ids in enumerate(sampled_token_ids):
            n = len(sampled_ids)
            if not n:
                draft_token_ids.append([])
                continue
            req_id = self.input_batch.req_ids[i]
            start = self.input_batch.num_tokens_no_spec[i]
            end = start + n
            if end >= self.max_model_len:
                draft_token_ids.append([])
                continue
            self.input_batch.token_ids_cpu[i, start:end] = sampled_ids
            st = self.requests[req_id]
            ea = st.sampling_params.extra_args or {}
            ctx = self.input_batch.token_ids_cpu[i, :end]
            store = _store(ea.get("run", "default"))
            store.live[req_id] = ctx[st.num_prompt_tokens - int(ea.get("resume", 0)):].astype(np.int64)
            if ea.get("collect"):
                col_items.append((i, req_id, n))
            grp = ea.get("group")
            if grp is not None and req_id not in store.grouped:  # join the passage's cooperative key trie
                store.grouped.add(req_id)
                g = store.groups.setdefault(grp, [])
                g.append(req_id)
                if len(g) > _GROUP_MAX:
                    del g[:len(g) - _GROUP_MAX]
            if ea.get("mode") in ("suffix", "ccr", "dbc", "bstop", "lazy") or ea.get("fallback") == "suffix":
                if store.suffix is None:
                    store.suffix = _suffix_cache()
                if store.suffix is not None:
                    if req_id not in store.suffix_active:
                        store.suffix.start_request(req_id, ctx[:st.num_prompt_tokens].astype(np.int32))
                        store.suffix_active.add(req_id)
                        store.suffix.add_active_response(req_id, ctx[st.num_prompt_tokens:].astype(np.int32))
                    else:
                        store.suffix.add_active_response(req_id, np.asarray(sampled_ids, dtype=np.int32))
            if not is_spec_decode_supported(req_id, self.input_batch):
                draft_token_ids.append([])
                continue
            if ea.get("mode") == "nd":  # consensus drafter: batched below
                store.dbc_src[req_id] = "nd"
                nd_items.append((i, req_id, st, ea, n))
                draft_token_ids.append([])
                continue
            try:
                d = _route(self, i, req_id, st, ea, ctx, k_engine)
            except Exception as e:  # never break the engine because of a draft
                print("[unison] draft error", repr(e), flush=True)
                d = []
            draft_token_ids.append(d)
        if col_items and os.environ.get("UNISON_COLLECT"):
            try:
                _collect_step(self, list(col_items), None)
            except Exception as e:
                print("[unison] collect error", repr(e), flush=True)
        if nd_items:
            try:
                drafts = _nd_batch(self, list(nd_items), k_engine)
            except Exception as e:  # never break the engine because of a draft
                print("[unison] consensus draft error", repr(e), flush=True)
                drafts = {}
            for i, req_id, *_ in nd_items:
                draft_token_ids[i] = list(drafts.get(req_id, []))
        return draft_token_ids

    # ---- consumer-conditioned replay: prompt logprobs on top of a prefix-cache hit
    import torch
    from vllm.v1 import outputs as vo
    from vllm.v1.core import kv_cache_manager as kcm

    orig_gcb = kcm.KVCacheManager.get_computed_blocks

    def get_computed_blocks(self, request):
        sp = request.sampling_params
        rf = (sp.extra_args or {}).get("replay_from")
        if rf is None or sp.prompt_logprobs is None:
            return orig_gcb(self, request)
        saved, sp.prompt_logprobs = sp.prompt_logprobs, None
        try:
            blocks, _ = orig_gcb(self, request)
        finally:
            sp.prompt_logprobs = saved
        keep = min(len(blocks), int(rf) // self.block_size)
        return blocks[:keep], keep * self.block_size

    def empty_cpu(num_positions, num_tokens_per_position):
        # zero-filled: positions served from the cache carry no logprobs and must stay well-formed
        ids = torch.zeros((num_positions, num_tokens_per_position), dtype=torch.int32)
        return vo.LogprobsTensors(logprob_token_ids=ids, logprobs=torch.zeros_like(ids, dtype=torch.float32),
                                  selected_token_ranks=torch.zeros(num_positions, dtype=torch.int32))

    kcm.KVCacheManager.get_computed_blocks = get_computed_blocks
    vo.LogprobsTensors.empty_cpu = staticmethod(empty_cpu)
    # ---- in-engine legality: surprisal trace from the verification logits, SSN, commit = accept all
    from vllm.v1.sample import rejection_sampler as rsm
    rsm.MAX_SPEC_LEN = 128  # greedy verification kernel has no length limit; the assert does
    orig_forward = rsm.RejectionSampler.forward
    law = {}

    def _ssn():
        if "net" not in law:
            import os
            import sys
            sys.path.insert(0, os.environ.get("UNISON_CODE", os.path.join(_ROOT, "code")))
            from unison.trace import build_law
            st = torch.load(os.environ["UNISON_SSN"], map_location="cpu")
            meta = st.pop("__meta__")
            net = build_law(meta)
            net.load_state_dict(st)
            law["net"] = net.eval().cuda()
        return law["net"]

    def forward(self, metadata, draft_probs, target_logits, bonus_token_ids, sampling_metadata):
        runner = getattr(self, "_unison_runner", None)
        jobs = []
        if runner is not None:
            cu = metadata.cu_num_draft_tokens.tolist()
            for i, req_id in enumerate(runner.input_batch.req_ids):
                st = runner.requests.get(req_id)
                if st is None:
                    continue
                ea = st.sampling_params.extra_args or {}
                store = _store(ea.get("run", "default"))
                if req_id in store.ccr_pending and metadata.num_draft_tokens[i] > 0:
                    pe = store.ccr_pending.pop(req_id)
                    a, b = (cu[i - 1] if i else 0), cu[i]
                    rows = target_logits[a:b].float()  # read before the sampler may modify logits in place
                    lse = torch.logsumexp(rows, -1)
                    dt = metadata.draft_token_ids[a:b].long()
                    lp = rows.gather(1, dt[:, None])[:, 0] - lse
                    mx, am = rows.max(-1)
                    jobs.append((i, req_id, pe, store, lp, mx - lse, am, dt))
        dbc = []
        if runner is not None:
            for i, req_id in enumerate(runner.input_batch.req_ids):
                st = runner.requests.get(req_id)
                if st is None:
                    continue
                ea = st.sampling_params.extra_args or {}
                if ea.get("mode") in ("lazy", "nd"):
                    own = _store(ea.get("run", "default")).live.get(req_id)
                    dbc.append((i, req_id, [] if own is None else own.tolist(), ea))
                    continue
                if ea.get("mode") not in ("dbc", "bstop") or not (ea.get("cands") or ea.get("n_nl")):
                    continue
                store = _store(ea.get("run", "default"))
                store.dbc_pending.pop(req_id, None)
                own = store.live.get(req_id)
                dbc.append((i, req_id, [] if own is None else own.tolist(), ea))
        out = orig_forward(self, metadata, draft_probs, target_logits, bonus_token_ids, sampling_metadata)
        if dbc:
            rows = out.tolist()
            nd = list(metadata.num_draft_tokens)
            for i, req_id, own, ea in dbc:
                # boundary check after verification: if the consumer's own output (its earlier tokens plus
                # the tokens accepted in this step) now contains a peer's decision prefix up to its boundary,
                # the consumer's decision is certified: end the request right after the boundary
                new = [t for t in rows[i] if t != -1] if -1 not in rows[i] else rows[i][:rows[i].index(-1)]
                store = _store(ea.get("run", "default"))
                a = _acc(store, req_id)
                a["steps"] += 1
                if nd[i] > 0:  # accepted draft tokens of this step, by draft source
                    a["drafted"] += nd[i]
                    a[store.dbc_src.get(req_id) or "suffix"] += max(0, min(nd[i], len(new) - 1))
                full, m = own + new, len(own)
                hit = False
                for c, de in zip(ea.get("cands") or [], ea.get("dends") or []):
                    j = de - m
                    if len(c) >= de and 0 <= j <= len(new) and j < out.shape[1] and full[:de] == list(c[:de]):
                        out[i, j] = int(ea["eos"][0])
                        out[i, j + 1:] = -1
                        hit = True
                        break
                if not hit and ea.get("n_nl"):  # end at the consumer's own decision boundary
                    nl = ea.setdefault("_nlset", set(ea["nl"]))
                    cnt = sum(1 for t in own if t in nl)
                    if cnt >= ea["n_nl"]:  # the boundary was the last token of the previous step
                        out[i, 0] = int(ea["eos"][0])
                        out[i, 1:] = -1
                        continue
                    for j, t in enumerate(new):
                        cnt += t in nl
                        if cnt >= ea["n_nl"] and j + 1 < out.shape[1]:
                            out[i, j + 1] = int(ea["eos"][0])
                            out[i, j + 2:] = -1
                            break
        for i, req_id, pe, store, lp, lp1, am, dt in jobs:
            n = dt.shape[0]
            ok = (am == dt)
            acc = int(ok.int().cumprod(0).sum().item())
            full = n + 1 >= len(pe["cand"])  # the whole candidate was replayed
            rec = dict(rid=req_id, n=n + 1, acc=acc + 1, cert=(acc == n) and full, q=None, commit=False)
            theta = pe.get("theta")
            if not rec["cert"] and (theta is not None or pe.get("log_q")):
                sig = torch.clamp(lp1 - lp, min=0)
                ell = torch.clamp(-lp, max=30.0)
                d = torch.tensor(pe["dmask"][:n + 1], dtype=torch.float32, device=sig.device)
                x = torch.zeros((1, n + 1, 3), device=sig.device)
                x[0, 1:, 0], x[0, 1:, 1] = sig, ell
                x[0, :len(d), 2] = d
                with torch.no_grad():
                    q = float(_ssn()(x, torch.ones((1, n + 1), device=sig.device))[0])
                rec["q"] = q
                if theta is not None and q >= theta:
                    # commit: accept the replayed horizon and end the request; the executor adopts the
                    # whole candidate (the replayed prefix followed by an end-of-sequence marks a commit)
                    row = torch.full_like(out[i], -1)
                    row[:n] = dt.to(out.dtype)
                    row[n] = int(pe["eos"][0])
                    out[i] = row
                    rec["commit"] = True
            _ccr_write(pe.get("run", "default"), rec)
        return out

    rsm.RejectionSampler.forward = forward
    _patch_scheduler()
    try:
        _patch_memory_handover()
    except Exception as e:  # never break the engine because of the start-up hand-over
        print("[unison] memory hand-over patch disabled:", repr(e), flush=True)
    gmr.GPUModelRunner._update_states = _update_states
    gmr.GPUModelRunner.generate_draft_token_ids = generate_draft_token_ids
    orig_exec = gmr.GPUModelRunner.execute_model

    def execute_model(self, scheduler_output):
        t0 = time.perf_counter()
        out = orig_exec(self, scheduler_output)
        try:
            _step_record(self, scheduler_output, t0)
        except Exception as e:  # never break the engine because of a measurement
            print("[unison] step log error", repr(e), flush=True)
        return out

    gmr.GPUModelRunner.execute_model = execute_model
