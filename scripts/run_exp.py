"""Run a set of execution policies on the same concurrent-agent stream with one engine instance."""
import argparse
import asyncio
import gzip
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

from unison.agents import AgentRuntime, World, make_stream
from unison.common import (AHN_BACKBONE, DATA, MODELS, RESULTS, decision_equiv, normalize_answer, parse_note,
                           parse_support, payload_tokens, read_jsonl, split_sentences)
from unison.engine import Engine, StepLog
from unison.executors import GPTCacheExec, VCacheOfficial, OSC, PLD, ExactShare, Executor, SemCache, Suffix, VCache
from unison.executors_ccr import Unison
from unison.executors_dbc import UnisonDBC
from unison.executors_v14 import LaCacheStyle, Unison14
from unison.executors_v15 import Unison15
from unison.executors_v17 import Collect, Unison17
from unison.retriever import Encoder, Retriever


CERT_SRC = ("certified", "exact", "replay-cert", "dcert")
RISK_SRC = ("risk", "sem")
ALL_SRC = ("certified", "exact", "replay-cert", "dcert", "risk", "sem", "engine")
CCR_LOGDIR = os.environ.setdefault("UNISON_CCR_LOGDIR", os.path.join(RESULTS, "..", "logs", "ccr"))


def merge_ccr(recs, run_id):
    """Attach the engine's in-pass legality records (q, acceptance, commit) to the calls."""
    p = os.path.join(CCR_LOGDIR, f"{run_id}.jsonl")
    if not os.path.exists(p):
        return 0
    with open(p) as f:
        log = {r["rid"]: r for r in map(json.loads, f)}
    for r in recs:
        g = log.get(r.get("rid"))
        if g is None:
            continue
        r.update(q=g["q"], ccr_acc=g["acc"], ccr_n=g["n"], ccr_commit=g["commit"], ccr_cert=g["cert"])
        if r["source"] == "ccr":
            r["source"] = "risk" if g["commit"] else "replay-cert"
    for r in recs:  # a full-length match that the engine did not record is an exact (certified) replay
        if r["source"] == "ccr":
            r["source"] = "replay-cert"
    return len(log)


def merge_acc(recs, run_id):
    """Attach the engine's per-request draft accounting (accepted key tokens by draft source)."""
    p = os.path.join(CCR_LOGDIR, f"{run_id}.acc.jsonl")
    if not os.path.exists(p):
        return 0
    with open(p) as f:
        log = {r["rid"]: r for r in map(json.loads, f)}
    for r in recs:
        g = log.get(r.get("rid")) if r.get("source") in ("engine", "dcert") else None  # the calls the engine served
        if g is not None:
            r["acc_passes"] = g.get("passes", 0)  # engine steps the call took part in (every method)
            if "steps" in g:
                r.update(acc_trie=g["trie"], acc_live=g["live"], acc_suffix=g["suffix"], acc_nd=g.get("nd", 0),
                         acc_pre=g.get("pre", 0), acc_drafted=g["drafted"], acc_steps=g["steps"])
            if "nd_calls" in g:  # the consensus drafter's own cost and the prices it drafted at
                r.update(nd_calls=g["nd_calls"], nd_ms=g.get("nd_ms", 0.0), nd_price=g.get("nd_price", 0.0),
                         nd_sprice=g.get("nd_sprice", 0.0), nd_it=g.get("nd_it", 0.0))
    return len(log)


def engine_steps(run_id):
    """The engine's own step log of a run (plugin, _step_record): an array [steps, 8] and its totals. Columns: time
    at the end of the step, seconds in the model runner, requests, positions computed, draft positions, requests
    admitted, seconds of first-pass drafting before the step, seconds of drafting inside it."""
    p = os.path.join(CCR_LOGDIR, f"{run_id}.esteps.csv")
    if not os.path.exists(p):
        return None, {}
    E = np.loadtxt(p, delimiter=",", ndmin=2)
    if not len(E):
        return None, {}
    # between two steps the engine schedules, hands results to the frontend and, if no call is waiting, idles:
    # a gap is engine overhead when a request of the next step was already running in the previous one
    gap = np.diff(E[:, 0]) - E[1:, 1] - E[1:, 6]
    back = E[1:, 5] < E[1:, 2]  # the step carried a request that was admitted earlier
    return E, dict(x_steps=int(len(E)), x_runner_s=float(E[:, 1].sum()), x_positions=int(E[:, 3].sum()),
                   x_draft_positions=int(E[:, 4].sum()), x_requests=float(E[:, 2].mean()),
                   x_pre_s=float(E[:, 6].sum()), x_nd_s=float(E[:, 7].sum()),
                   x_between_s=float(np.clip(gap[back], 0, None).sum()),
                   x_between_ms=float(1e3 * np.median(gap[back])) if back.any() else 0.0)


def is_prefix(out, ref, eos):
    """Delivered tokens are a prefix of the independent re-execution (end-of-sequence markers aside). A call that
    ends at its consumption boundary delivers a prefix of its full output; a call that runs to its end delivers
    all of it."""
    strip = lambda t: t[:-1] if t and t[-1] in eos else t
    out, ref = strip(list(out)), strip(list(ref))
    return out == ref[:len(out)]


def code_sha():
    """Hash of the code that produced a run (method, plugin, runner), stored in every summary."""
    import hashlib
    h = hashlib.sha1()
    base = os.path.join(os.path.dirname(__file__), "..")
    for d in ("unison", "unison_plugin", "scripts"):
        for fn in sorted(os.listdir(os.path.join(base, d))):
            if fn.endswith(".py"):
                with open(os.path.join(base, d, fn), "rb") as f:
                    h.update(fn.encode() + f.read())
    return h.hexdigest()[:12]


def gpu_others():
    """Memory held on this GPU by other processes (co-tenant check, MiB)."""
    import subprocess
    try:
        g = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
        used = int(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", g],
                                  capture_output=True, text=True).stdout.strip())
        mine = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits",
                               "-i", g], capture_output=True, text=True).stdout
        import psutil
        me = {os.getpid()} | {c.pid for c in psutil.Process().children(recursive=True)}
        if os.environ.get("UNISON_HOLDER_PID"):  # the queue's memory holder on an otherwise empty card is ours
            me.add(int(os.environ["UNISON_HOLDER_PID"]))
        own = sum(int(l.split(",")[1]) for l in mine.strip().splitlines()
                  if l.strip() and int(l.split(",")[0]) in me)
        return used - own
    except Exception:
        return -1


class CoTenantMonitor:
    """Samples the SM utilization of *other* processes on our GPU (nvidia-smi pmon) every few seconds while a
    method runs; the summary records mean/max so that a co-tenant's compute activity (not only its memory) is
    visible when judging whether a latency measurement is clean."""

    def __init__(self, period=5.0):
        import threading
        self.period, self.samples, self._stop = period, [], threading.Event()
        self.t = threading.Thread(target=self._run, daemon=True)

    def _mine(self):
        import psutil
        me = {os.getpid()} | {c.pid for c in psutil.Process().children(recursive=True)}
        if os.environ.get("UNISON_HOLDER_PID"):
            me.add(int(os.environ["UNISON_HOLDER_PID"]))
        return me

    def _sample(self):
        import subprocess
        g = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
        out = subprocess.run(["nvidia-smi", "pmon", "-c", "1", "-s", "u", "-i", g], capture_output=True, text=True,
                             timeout=20).stdout
        mine, others = self._mine(), 0.0
        for line in out.splitlines():
            f = line.split()
            if len(f) < 4 or not f[1].isdigit():
                continue
            try:
                sm = float(f[3])
            except ValueError:
                continue
            if int(f[1]) not in mine:
                others += sm
        return others

    def _run(self):
        while not self._stop.is_set():
            try:
                self.samples.append(self._sample())
            except Exception:
                pass
            self._stop.wait(self.period)

    def start(self):
        self.t.start()
        return self

    def stop(self):
        self._stop.set()
        s = self.samples
        thr = float(os.environ.get("UNISON_COTENANT_SM", "40"))  # co-tenant SM share that counts as "busy"
        return dict(gpu_others_sm_mean=float(np.mean(s)) if s else -1.0, gpu_others_sm_max=float(max(s)) if s else -1.0,
                    gpu_others_sm_busy=float(np.mean([x >= thr for x in s])) if s else -1.0,
                    gpu_others_sm_active=float(np.mean([x >= 10 for x in s])) if s else -1.0, gpu_others_samples=len(s))


def hard_exit(code):
    """Leave at once from inside the event loop; the cache services and the engine process go with us."""
    try:
        from unison.cache_service import close_all
        close_all()
    except Exception:
        pass
    try:
        import psutil
        for c in psutil.Process().children(recursive=True):
            c.kill()
    except Exception:
        pass
    os._exit(code)


def pct(x, q):
    return float(np.percentile(x, q)) if len(x) else float("nan")


def busy_time(rec):
    t = [r[0] for r in rec]
    return float(sum(b - a for a, b in zip(t, t[1:]) if b - a < 1.0))


def build(name, engine, run_id, world, enc, ahn, ssn=None, ds=None, note="dfirst"):
    fmt = note
    if "@" in name:  # method@cot / method@dfirst: the note format this policy runs with
        name, fmt = name.split("@")
    if enc is not None:
        enc.keys = None  # the two cache executors ask the retriever's pass for their key embeddings
    X = _build(name, engine, run_id, world, enc, ahn, ssn, ds)
    X.note_format = fmt
    X.note_max_tokens = 128 + max(0, payload_tokens(fmt) - 64)  # a full note must fit for every system
    if fmt != note:
        X.name += f"@{fmt}"
    return X


def _build(name, engine, run_id, world, enc, ahn, ssn=None, ds=None):
    if name == "u17" or name.startswith("u17-"):  # Unison (consensus drafting) and its ablation variants
        return Unison17(engine, run_id, world, variant=name[4:])
    if name == "collect":  # training-data run for the consensus drafter
        return Collect(engine, run_id, world)
    if name == "u15" or name.startswith("u15-"):  # Unison v15 (paper/unison15) and its ablation variants
        return Unison15(engine, run_id, world, variant=name[4:])
    if name == "u14" or name.startswith("u14-"):  # Unison v14 (paper/unison14) and its ablation variants
        return Unison14(engine, run_id, world, variant=name[4:])
    if name == "lacache":  # LaCache-style lookahead-checked semantic cache (re-implementation)
        return LaCacheStyle(engine, run_id, world, encoder=enc)
    if name == "dbc" or name.startswith("dbc-"):  # Unison (decision-boundary certification) and variants
        return UnisonDBC(engine, run_id, world, variant=name[4:] or "full")
    if name == "noshare":
        return Executor(engine, run_id, world)
    if name == "exact":
        return ExactShare(engine, run_id, world)
    if name == "exact+suffix":
        return ExactShare(engine, run_id, world, suffix=True)
    if name == "pld":
        return PLD(engine, run_id, world)
    if name == "suffix":
        return Suffix(engine, run_id, world)
    if name in ("gptcache", "gptcacheS"):  # official GPTCache library, default threshold
        return GPTCacheExec(engine, run_id, world, encoder=enc, suffix=name.endswith("S"))
    if name in ("vcacheoff", "vcacheoffS"):  # official vCache implementation, default delta
        return VCacheOfficial(engine, run_id, world, encoder=enc, suffix=name.endswith("S"))
    if name.startswith("vcache"):  # per-entry online thresholds, initialized at the eps=0.05 global threshold
        tau0 = ahn.meta["cos_thresholds"].get("0.050", 0.9) if ahn is not None else 0.9
        return VCache(engine, run_id, world, encoder=enc, tau0=tau0, suffix=name.startswith("vcacheS"))
    if name.startswith("semcrc"):  # cosine threshold calibrated to a decision-risk budget; semcrcS = + Suffix
        eps = float(name.split("-")[1])
        tau = ahn.meta["cos_thresholds"].get(f"{eps:.3f}", 1.01)
        return SemCache(engine, run_id, world, encoder=enc, tau=tau, label=name, suffix=name.startswith("semcrcS"))
    if name.startswith("sem"):  # sem0.90 / semV0.90 / semS0.90 / semVS0.90 (S = misses use SuffixDecoding)
        flags = name[3:].rstrip("0123456789.")
        tau = float(name[3 + len(flags):])
        lab = f"semcache{'V' if 'V' in flags else ''}{'+S' if 'S' in flags else ''}-{tau:.2f}"
        return SemCache(engine, run_id, world, encoder=enc, tau=tau, versioned="V" in flags, suffix="S" in flags,
                        label=lab)
    if name.startswith("unison") or name.startswith("osc") or name.startswith("law"):
        name = name.replace("law", "unison", 1)
        eps, variant = 0.0, "full"
        for part in name.split("-")[1:]:
            if part[:1] == "e" and part[1:2].isdigit():
                eps = float(part[1:])
            else:
                variant = part
        if name.startswith("osc"):
            return OSC(engine, run_id, world, ahn=ahn, encoder=enc, eps=eps, variant=variant)
        X = Unison(engine, run_id, world, ssn=ssn, eps=eps, variant=variant, workload=ds)
        X.name = X.name.replace("unison", "law", 1)  # the earlier estimated-legality design (prefix law)
        return X
    raise ValueError(name)


def summarize(name, res, makespan, recs, steps):
    lat = [r["latency"] for r in res]
    ver = [r for r in recs if r["kind"] == "verify"]
    pln = [r for r in recs if r["kind"] == "plan"]
    vl = [r["t_end"] - r["t_arrive"] for r in ver]
    eng = [r for r in ver if r["source"] == "engine"]
    gen = sum(r[3] for r in steps)
    by_pid, ph_cnt = {}, {}
    for r in ver:  # workload redundancy (motivation): same passage within +-60 s, identical prompt anywhere
        by_pid.setdefault(r["pid"], []).append(r["t_arrive"])
        ph_cnt[r["phash"]] = ph_cnt.get(r["phash"], 0) + 1
    near = 0
    for ts in by_pid.values():
        ts = sorted(ts)
        for i, t in enumerate(ts):
            near += (i > 0 and t - ts[i - 1] <= 60) or (i + 1 < len(ts) and ts[i + 1] - t <= 60)
    redund = dict(redund_passage60=near / max(1, len(ver)),
                  redund_identical=sum(c for c in ph_cnt.values() if c > 1) / max(1, len(ver)))
    lat_all = [r.get("latency_all", r["latency"]) for r in res]
    return dict(**redund,
        method=name, n_queries=len(res), makespan=makespan, throughput=len(res) / makespan,
        lat_mean=float(np.mean(lat)), lat_p50=pct(lat, 50), lat_p95=pct(lat, 95), lat_p99=pct(lat, 99),
        lat_all_mean=float(np.mean(lat_all)), lat_all_p99=pct(lat_all, 99),
        vlat_mean=float(np.mean(vl)), vlat_p50=pct(vl, 50), vlat_p99=pct(vl, 99),
        em=float(np.mean([r["em"] for r in res])), f1=float(np.mean([r["f1"] for r in res])),
        n_verify=len(ver), n_plan=len(pln), verify_out_tokens=sum(r["n_out"] for r in ver),
        verify_engine_calls=len(eng), verify_steps=sum(r["steps"] for r in eng),
        plan_steps=sum(r["steps"] for r in pln), n_plan_exact=sum(r["source"] == "exact" for r in pln),
        plan_engine_calls=sum(r["source"] == "engine" for r in pln),
        prefix_queries=int(sum(r[7] for r in steps if len(r) > 8)),
        prefix_hits=int(sum(r[8] for r in steps if len(r) > 8)),
        reuse_calls=sum(r["source"] != "engine" for r in ver),
        certified=sum(r["source"] in CERT_SRC for r in ver),
        risk_commits=sum(r["source"] in RISK_SRC for r in ver),
        **{f"src_{k}": sum(r["source"] == k for r in ver) for k in ALL_SRC},
        resumed_calls=sum(1 for r in ver if r.get("resumed")), resumed_tokens=sum(r.get("resumed", 0) for r in ver),
        waited_mean=float(np.mean([r.get("waited", 0.0) for r in ver])) if ver else 0.0,
        engine_iters=len(steps), engine_busy=busy_time(steps), gen_tokens=int(gen),
        prompt_tokens=int(sum(r[4] for r in steps)), draft_tokens=int(sum(r[5] for r in steps)),
        accepted_tokens=int(sum(r[6] for r in steps)))


async def replay(engine, prompts, run_id, conc=192):
    """Independent re-execution of every distinct prompt: the reference outputs for fidelity."""
    ref, sem = {}, asyncio.Semaphore(conc)

    async def one(h, ids, mt):
        async with sem:
            _, toks, _ = await engine.generate(ids, mt, {"mode": "none", "run": run_id})
            ref[h] = toks

    await asyncio.gather(*[one(h, ids, mt) for h, (ids, mt) in prompts.items()])
    return ref


def reserve_gpu(allowance_mib, headroom_mib):
    """Hold the free memory of an (otherwise empty) card beyond what the engine needs, so that memory-driven job
    schedulers of other projects on the machine do not place work next to a timing-sensitive run. The
    placeholder lives in this process for the whole run and is released at exit; it is never used on a card
    that already hosts someone else's process (the queue decides)."""
    import torch
    free, total = torch.cuda.mem_get_info()
    reserve = int(free) - (allowance_mib + headroom_mib) * (1 << 20)
    if reserve < (512 << 20):
        print(f"reserve: skipped (free {free >> 20} MiB, allowance {allowance_mib} MiB)", flush=True)
        return None
    hold = torch.empty(reserve, dtype=torch.uint8, device="cuda")
    print(f"reserve: holding {reserve >> 20} MiB of {total >> 20} MiB (free was {free >> 20})", flush=True)
    return hold


async def main(a):
    os.environ["UNISON_SSN"] = a.ssn  # the engine-side law (engine core inherits the environment)
    os.environ["UNISON_DRAFTER"] = a.drafter  # the consensus drafter (loaded lazily by the engine plugin)
    if a.collect:
        os.environ["UNISON_COLLECT"] = a.collect
    pool = os.path.join(DATA, a.ds, a.split)
    out_dir = os.path.join(RESULTS, a.tag, a.ds)
    os.makedirs(out_dir, exist_ok=True)
    hold = None
    if os.environ.get("UNISON_RESERVE") == "1":
        hold = reserve_gpu(int(os.environ.get("UNISON_ALLOWANCE_MIB", "16050")), int(os.environ.get("UNISON_HEADROOM_MIB", "1200")))
    abort_busy = float(os.environ.get("UNISON_ABORT_COTENANT_BUSY", "0") or 0)  # abort a timed run under co-tenant compute
    stall_s = float(os.environ.get("UNISON_ABORT_STALL_S", "0") or 0)  # abort a timed run when the host stalls us
    flag = os.environ.get("UNISON_HOLD_FLAG")  # hand-over with the queue's memory holder (see hold_gpu.py)
    engine = Engine(MODELS[a.model], gpu_util=a.gpu_util)
    await engine.init()
    cost, rows = await engine.profile()
    print("profile", cost, rows, flush=True)
    enc = Encoder()
    retr = Retriever(pool, enc)
    if flag:  # engine, encoder and retriever hold their memory now. The holder keeps the card's free memory at the
        # run-time headroom: from both sides under the pool hand-over (what the engine still asks the device for
        # comes out of the holder's reserve), by absorbing only under the phased one
        cmd = "feed" if os.environ.get("UNISON_HANDOVER", "pool") == "pool" else "resume"
        with open(flag, "w") as f:
            f.write(f"{cmd} {os.environ.get('UNISON_RUN_HEADROOM_MIB', '3500')}")
    world = World(pool, a.versions)
    ssn = None
    if any(m.startswith("unison") or m.startswith("law") for m in a.methods.split(",")):
        from unison.trace import SSNScorer
        ssn = SSNScorer(a.ssn, device="cpu")
    ahn = None
    # vcacheoff is the official vCache client and does not use the AHN scorer. Loading that second
    # encoder after vLLM has taken the card exhausts the GPU.
    def _needs_ahn(name):
        name = name.split("@", 1)[0]
        if name.startswith(("osc", "semcrc")):
            return True
        return name.startswith("vcache") and not name.startswith("vcacheoff")
    if any(_needs_ahn(m) for m in a.methods.split(",")):
        from unison.ahn import AHNScorer
        ahn = AHNScorer(a.ahn, AHN_BACKBONE, engine.tok)
    qs = read_jsonl(os.path.join(pool, "questions.jsonl"))
    paras = read_jsonl(os.path.join(pool, "paraphrases.jsonl"))
    trace = None
    if a.arrival == "azure":
        import csv
        with open(os.path.join(DATA, "azure_conv.csv")) as f:
            rd = csv.DictReader(f)
            from datetime import datetime
            trace = [datetime.fromisoformat(r["TIMESTAMP"][:26]).timestamp() for r in rd]
    stream = make_stream(qs, paras, a.n, a.rate, a.zipf, a.seed, a.arrival, trace, a.qsplit)
    hops = 3 if a.ds == "hotpotqa" else 4
    prompts_all, summaries = {}, []
    order = a.methods.split(",")
    if a.shuffle:  # randomized system order per seed: co-tenant load cannot favor a fixed position
        np.random.default_rng(1000 + a.seed).shuffle(order)
    from unison.cache_service import close_all as close_cache_services
    for m in order:
        close_cache_services()  # the previous method's cache processes do not stay on the host next to this one
        if a.warmup > 0:
            # warm-up: the method's own code paths (draft lengths, kernels, graphs, allocator) on the first queries
            # of the stream with compressed arrivals; executor state and caches are then discarded, so the measured
            # run starts from the same cold caches as every other method but on a warm engine
            world.reset()
            await engine.reset()
            Xw = build(m, engine, f"{a.tag}-{a.ds}-{m}-warm-{int(time.time())}", world, enc, ahn, ssn, a.ds, a.note)
            rtw = AgentRuntime(engine, Xw, world, retr, k=a.k, max_hops=hops, team=a.team)
            ws = [dict(s, t=0.3 * i) for i, s in enumerate(stream[:a.warmup])]
            t_w = time.time()
            await rtw.run(ws, update_rate=0.0, seed=a.seed)
            if getattr(Xw, "bg", None):
                await asyncio.gather(*Xw.bg, return_exceptions=True)
            print(f"warmup {m}: {len(ws)} queries in {time.time() - t_w:.1f}s", flush=True)
        world.reset()
        await engine.reset()
        run_id = f"{a.tag}-{a.ds}-{m}-{int(time.time())}"
        X = build(m, engine, run_id, world, enc, ahn, ssn, a.ds, a.note)
        orig_rec = X._rec
        col_prompts = {}  # collection run: the prompt of every call (the drafter's record holds its tail)

        def rec_hook(info, ids, toks, tm, t_arrive, _o=orig_rec, **extra):
            r = _o(info, ids, toks, tm, t_arrive, **extra)
            prompts_all.setdefault(r["phash"], (list(ids), getattr(X, "note_max_tokens", 128) if info["kind"] == "verify" else 48))
            if a.collect and r.get("rid") is not None:
                col_prompts[r["rid"]] = np.asarray(ids, dtype=np.int32)
            return r

        X._rec = rec_hook
        rt = AgentRuntime(engine, X, world, retr, k=a.k, max_hops=hops, team=a.team)
        StepLog.records.clear()
        lag = []

        async def lagmon():  # event-loop responsiveness (token streaming of all requests runs on this loop)
            while True:
                t = time.perf_counter()
                await asyncio.sleep(0.01)
                lag.append(time.perf_counter() - t - 0.01)
                if stall_s and lag[-1] > stall_s:
                    # the host stopped this process for seconds (memory pressure on the machine): the arrivals
                    # of the stream were not delivered on time and nothing measured here belongs to the method
                    print(f"HOST_STALL_ABORT method={X.name} stalled {lag[-1]:.1f}s "
                          f"after {time.time() - t_run0:.0f}s", flush=True)
                    hard_exit(3)

        others0 = gpu_others()
        cot = CoTenantMonitor().start()
        t_run0 = time.time()
        mon = asyncio.create_task(lagmon())
        res, makespan = await rt.run(stream, update_rate=a.mu, seed=a.seed)
        t_fg_end = time.time()
        n_bg_open = sum(1 for f in getattr(X, "bg", []) if not f.done())
        if getattr(X, "bg", None):  # deferred provenance still decoding: its GPU work is charged to the run
            await asyncio.gather(*X.bg)
        t_all_end = time.time()
        mon.cancel()
        cot_stats = cot.stop()
        if abort_busy and cot_stats["gpu_others_sm_busy"] > abort_busy:
            print(f"COTENANT_ABORT method={X.name} others_sm_busy={cot_stats['gpu_others_sm_busy']:.2f} "
                  f"others_sm_mean={cot_stats['gpu_others_sm_mean']:.0f}", flush=True)
            sys.exit(3)
        await asyncio.sleep(0.5)
        n_ccr = merge_ccr(X.recs, run_id)
        n_acc = merge_acc(X.recs, run_id)
        esteps, xs = engine_steps(run_id)
        po = getattr(X, "prov_out", {})
        for r in X.recs:  # delivered provenance text (audit evaluation)
            if r.get("prov_key") in po:
                r["prov"] = engine.decode(po[r["prov_key"]])
        steps = list(StepLog.records)
        s = summarize(X.name, res, makespan, X.recs, steps)
        pr = getattr(X, "prov_recs", [])
        pl = [r["t_end"] - r["t_ret"] for r in pr]
        s.update(prov_requests=len(pr), prov_engine=sum(r["engine"] for r in pr),
                 prov_tokens=int(sum(r["n_out"] for r in pr if r["engine"])),
                 prov_lag_mean=float(np.mean(pl)) if pl else 0.0, prov_lag_p50=pct(pl, 50) if pl else 0.0,
                 prov_lag_p99=pct(pl, 99) if pl else 0.0, bg_open_at_end=n_bg_open,
                 drain_s=t_all_end - t_fg_end, makespan_all=t_all_end - t_run0,
                 busy_fg=busy_time([r for r in steps if r[0] <= t_fg_end]),
                 gpu_others_mib=max(others0, gpu_others()), code_sha=code_sha(), n_acc=n_acc,
                 prov_coalesced=sum(not r["engine"] for r in pr), kv_blocks=engine.kv_blocks, **cot_stats)
        vv = [r for r in X.recs if r["kind"] == "verify"]
        pp = [r for r in X.recs if r["kind"] == "plan"]
        for src in ("trie", "live", "suffix", "nd", "pre"):
            s[f"acc_{src}"] = int(sum(r.get(f"acc_{src}", 0) for r in vv))
            s[f"pacc_{src}"] = int(sum(r.get(f"acc_{src}", 0) for r in pp))
        # passes of the engine per call it served (the first one processes the prompt) and the calls that took
        # one pass; counted by the engine for every method
        for tag, rr in (("key", vv), ("plan", pp)):
            st = [r["acc_passes"] for r in rr if r.get("acc_passes")]
            s[f"{tag}_passes"] = float(np.mean(st)) if st else 0.0
            s[f"{tag}_one_pass"] = float(np.mean([x == 1 for x in st])) if st else 0.0
            s[f"{tag}_engine_calls"] = len(st)
            s[f"{tag}_pass_hist"] = np.bincount(np.clip(st, 0, 64), minlength=65).tolist() if st else []
            s[f"{tag}_out_tokens"] = int(sum(r["n_out"] for r in rr if r.get("acc_passes")))
        s["acc_drafted"] = int(sum(r.get("acc_drafted", 0) for r in vv))
        s["pacc_drafted"] = int(sum(r.get("acc_drafted", 0) for r in pp))
        s["pacc_steps"] = int(sum(r.get("acc_steps", 0) for r in pp))
        s["key_tokens"] = int(sum(r["n_out"] for r in vv if r["source"] == "engine" or r["source"] == "dcert"))
        s["key_steps"] = int(sum(r["steps"] for r in vv if r["source"] in ("engine", "dcert")))
        ndc = sum(r.get("nd_calls", 0) for r in X.recs)
        if ndc:  # consensus drafter: its share of the engine's time and the mean prices of a draft position
            s.update(nd_calls=int(ndc), nd_seconds=float(sum(r.get("nd_ms", 0.0) for r in X.recs)) / 1e3,
                     nd_price=float(sum(r.get("nd_price", 0.0) for r in X.recs)) / ndc,
                     nd_sprice=float(sum(r.get("nd_sprice", 0.0) for r in X.recs)) / ndc,
                     nd_iter_ms=float(sum(r.get("nd_it", 0.0) for r in X.recs)) / ndc)
        s.update(xs)
        if esteps is not None:
            np.save(os.path.join(out_dir, f"{X.name}.esteps.npy"), esteps)
        s.update(df_boundaries=int(rt.df), n_round_early=rt.n_round_early, n_team_early=rt.n_team_early,
                 n_df_mismatch=rt.n_df_mismatch, n_pending_drained=len(rt.pending))
        s.update(cfg=vars(a), cost=cost, n_updates=world.n_updates, n_ccr=n_ccr,
                 ccr_drafts=sum(1 for r in X.recs if r.get("cand") is not None),
                 loop_lag_ms=1000 * float(np.mean(lag)) if lag else 0.0,
                 loop_lag_p99_ms=1000 * pct(lag, 99) if lag else 0.0,
                 loop_lag_max_ms=1000 * float(max(lag)) if lag else 0.0)
        if getattr(X, "shadow", None):
            with gzip.open(os.path.join(out_dir, f"{X.name}.shadow.jsonl.gz"), "wt") as f:
                for r in X.shadow:
                    f.write(json.dumps(r) + "\n")
        summaries.append(s)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items() if k != "cfg"}),
              flush=True)
        with gzip.open(os.path.join(out_dir, f"{X.name}.calls.jsonl.gz"), "wt") as f:
            for r in X.recs:
                f.write(json.dumps(r) + "\n")
        with open(os.path.join(out_dir, f"{X.name}.queries.json"), "w") as f:
            json.dump(res, f)
        np.save(os.path.join(out_dir, f"{X.name}.steps.npy"), np.array(steps, dtype=np.float64))
        if col_prompts:  # same layout as scripts/rebuild_prompts.py (which recovers them for runs made without)
            rids = list(col_prompts)
            off = np.zeros(len(rids) + 1, dtype=np.int64)
            off[1:] = np.cumsum([len(col_prompts[r]) for r in rids])
            np.savez_compressed(os.path.join(out_dir, f"{X.name}.prompts.npz"), rids=np.array(rids), off=off,
                                ids=np.concatenate([col_prompts[r] for r in rids]),
                                exact=np.ones(len(rids), dtype=bool))
    close_cache_services()
    if a.replay:
        await engine.reset()
        t = time.time()
        ref = await replay(engine, prompts_all, f"{a.tag}-replay")
        print(f"replayed {len(ref)} prompts in {time.time() - t:.1f}s", flush=True)
        with gzip.open(os.path.join(out_dir, "replay.json.gz"), "wt") as f:
            json.dump(ref, f)
        if a.replay2:
            await engine.reset()
            items = list(prompts_all.items())
            np.random.default_rng(7).shuffle(items)
            ref2 = await replay(engine, dict(items), f"{a.tag}-replay2", conc=48)
            vh = [h for h, (ids, mt) in prompts_all.items() if mt != 48]  # verification prompts
            nd = [not decision_equiv(parse_note(engine.decode(ref[h])), parse_note(engine.decode(ref2[h]))) for h in vh]
            floor = dict(nd_prompts=len(vh), nd_decision_diff=int(sum(nd)),
                         nd_token_diff=int(sum(ref[h] != ref2[h] for h in vh)))
            print("FLOOR", json.dumps(floor), flush=True)
        else:
            floor = {}
        base_ans = None
        for s in summaries:
            with gzip.open(os.path.join(out_dir, f"{s['method']}.calls.jsonl.gz"), "rt") as f:
                recs = [json.loads(l) for l in f]
            for kind in ("verify", "plan"):
                rr = [r for r in recs if r["kind"] == kind]
                s[f"fidelity_{kind}"] = float(np.mean([r["out"] == ref[r["phash"]] for r in rr])) if rr else 1.0
                s[f"fidelity_prefix_{kind}"] = float(np.mean([is_prefix(r["out"], ref[r["phash"]], engine.eos)
                                                               for r in rr])) if rr else 1.0
            vv = [r for r in recs if r["kind"] == "verify"]
            dec = [decision_equiv(parse_note(engine.decode(r["out"])), parse_note(engine.decode(ref[r["phash"]])))
                   for r in vv]
            s["fidelity_decision"] = float(np.mean(dec)) if dec else 1.0
            for src in ALL_SRC:
                d = [x for x, r in zip(dec, vv) if r["source"] == src]
                if d:
                    s[f"dec_err_{src}"] = float(1 - np.mean(d))
            rk = [(x, r) for x, r in zip(dec, vv) if r["source"] in RISK_SRC]
            s["dec_err_risk"] = float(1 - np.mean([x for x, _ in rk])) if rk else float("nan")
            xv = [(x, r) for x, r in zip(dec, vv) if r["source"] != "engine" and r.get("src_version", r["version"]) != r["version"]]
            s["stale_reuse"] = len(xv)  # results reused across evidence versions
            s["stale_err"] = int(sum(1 - x for x, _ in xv))  # ... whose decision differs from a fresh execution
            s.update(floor)
            # decisive edits: the edit changed the supporting sentence of the delivered note, or the
            # delivered answer occurs in exactly one of the two versions
            dz = []
            for x, r in xv:
                t = engine.decode(r["out"])
                rel, an = parse_note(t)
                A = split_sentences(world.text(r["pid"], r["src_version"]))
                B = split_sentences(world.text(r["pid"], r["version"]))
                k = parse_support(t)
                sup = 0 < k <= min(len(A), len(B)) and A[k - 1] != B[k - 1]
                na = normalize_answer(an)
                occ = rel and (na in normalize_answer(" ".join(A))) != (na in normalize_answer(" ".join(B)))
                dz.append((x, bool(sup or occ)))
            s["stale_decisive"] = int(sum(d for _, d in dz))
            s["stale_decisive_err"] = int(sum(d and not x for x, d in dz))
            for src in ("dcert", "sem", "certified", "exact"):
                xs = [d for (x, r), (_, d) in zip(xv, dz) if r["source"] == src]
                s[f"stale_{src}"], s[f"stale_decisive_{src}"] = len(xs), int(sum(xs))
            s["dec_err_total"] = float(1 - np.mean(dec)) if dec else 0.0
            # counts per reuse source (exact coalescing vs certified vs estimated), and provenance: the support
            # pointer of the delivered note vs a fresh execution, and whether the pointed sentence of the
            # consumer's own passage version contains the delivered answer
            for src in ALL_SRC:
                d = [x for x, r in zip(dec, vv) if r["source"] == src]
                s[f"n_{src}"], s[f"nerr_{src}"] = len(d), int(len(d) - sum(d))
            texts = [engine.decode(r["out"]) for r in vv]
            sup = [parse_support(t) == parse_support(engine.decode(ref[r["phash"]])) for t, r in zip(texts, vv)]
            reused = [r["source"] != "engine" for r in vv]
            s["n_reused"] = int(sum(reused))
            s["nerr_reused"] = int(sum(1 - x for x, u in zip(dec, reused) if u))
            s["nerr_reused_prov"] = int(sum(1 - (x and y) for x, y, u in zip(dec, sup, reused) if u))
            val = []
            for t, r in zip(texts, vv):
                rel, ans = parse_note(t)
                if rel:
                    sents = split_sentences(world.text(r["pid"], r["version"]))
                    k = parse_support(t)
                    val.append(0 < k <= len(sents) and normalize_answer(ans) in normalize_answer(sents[k - 1]))
            s["support_valid"] = float(np.mean(val)) if val else 1.0
            with open(os.path.join(out_dir, f"{s['method']}.queries.json")) as f:
                q = json.load(f)
            ans = [x["answer"] for x in sorted(q, key=lambda x: x["qi"])]
            if base_ans is None:
                base_ans = ans
            s["answer_agree"] = float(np.mean([x == y for x, y in zip(ans, base_ans)]))
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summaries, f, indent=1)
    for s in summaries:
        print("FINAL", json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()
                                   if k not in ("cfg", "cost")}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ds", default="hotpotqa")
    p.add_argument("--split", default="eval")
    p.add_argument("--methods", default="noshare")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--rate", type=float, default=0.5)
    p.add_argument("--zipf", type=float, default=0.8)
    p.add_argument("--team", type=int, default=3)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--mu", type=float, default=0.0)
    p.add_argument("--arrival", default="poisson")
    p.add_argument("--model", default="qwen3-8b")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="dev")
    p.add_argument("--replay", type=int, default=1)
    p.add_argument("--ahn", default=os.path.join(RESULTS, "ahn", "ahn.pt"))
    p.add_argument("--ssn", default=os.path.join(RESULTS, "ssn", "law.pt"))
    p.add_argument("--gpu_util", type=float, default=0.78)
    p.add_argument("--qsplit", default="all")
    p.add_argument("--note", default="dfirst")
    p.add_argument("--shuffle", type=int, default=1)
    p.add_argument("--replay2", type=int, default=0)
    p.add_argument("--versions", default="versions.jsonl")
    p.add_argument("--warmup", type=int, default=12, help="queries of a discarded warm-up pass before each method (0: none)")
    p.add_argument("--drafter", default=os.path.join(RESULTS, "drafter", "cd.pt"))
    p.add_argument("--collect", default="", help="directory for the hidden-state shards of a 'collect' run")
    asyncio.run(main(p.parse_args()))
