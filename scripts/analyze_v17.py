"""Tables, figures and number macros for paper/unison17 from <results>/v17_*/<ds>/{summary.json,*.queries.json,
*.esteps.npy} and <results>/drafter*/eval*.json.

usage: python analyze_v17.py <results_dir> <paper_dir>
Every number in the paper's evaluation comes from here. Missing runs print '?' (never silently '-').
Statistics: paired bootstrap over the queries of a stream for latency ratios and EM differences, exact McNemar test
for EM, Clopper-Pearson upper bounds for error rates, bootstrap over engine steps for the step-time model, and the
spread over three independent streams for the main comparison.
"""
import glob
import json
import os
import re
import sys
from decimal import ROUND_HALF_UP, Decimal

import numpy as np
from scipy.stats import beta, binomtest

R, P = sys.argv[1], sys.argv[2]
os.makedirs(P, exist_ok=True)
os.makedirs(os.path.join(P, "figs"), exist_ok=True)
DS = {"hotpotqa": "HotpotQA", "musique": "MuSiQue"}
DSK = {"hotpotqa": "Ho", "musique": "Mu"}
V = "v17"
U, SBD, RULE, NOPRE, SUF = "u17", "u17-sbd", "u17-rule", "u17-nopre", "u17-suf"
# Comparison systems are published systems, run from their own code or from the engine's implementation of that
# method. Component variants of Unison appear only in the attribution tables.
PUB = ["noshare", "pld", "suffix", "gptcache", "vcache-official"]
MAIN = [("noshare", r"vLLM + APC~\cite{kwon2023vllm}"), ("pld", r"PLD~\cite{saxena2023pld}"),
        ("suffix", r"SuffixDecoding~\cite{oliaro2025suffix}"), ("gptcache", r"GPTCache~\cite{bang2023gptcache}"),
        ("vcache-official", r"vCache~\cite{schroeder2025vcache}"), (U, r"\textbf{Unison}")]
SHORT = [("noshare", "vLLM + APC"), ("pld", "PLD"), ("suffix", "SuffixDecoding"), ("gptcache", "GPTCache"),
         ("vcache-official", "vCache"), (U, "\\textbf{Unison}")]
ABL = [(SUF, "(A) layer with suffix lookup"),
       (RULE, "(B) A, decisions drafted by the peer rule"),
       (NOPRE, "(C) record drafter after the first pass"),
       (U, r"\textbf{(D) Unison}: C + first-pass verification"),
       ("u17-nopeer", r"\quad D, record without peer outputs"),
       ("u17-noctx", r"\quad D, record without the prompt tail"),
       ("u17-noh", r"\quad D, drafter without the consumer state"),
       ("u17-noplan", r"\quad D, plans drafted by suffix lookup"),
       ("u17-noid", r"\quad D without identity sharing"),
       ("u17-nodf", r"\quad D with call-level boundaries only"),
       (SBD, r"\quad A with one rationale per call")]
WIDE = SHORT
LOADM = SHORT
RATES = [0.6, 1.2, 1.8, 2.4, 3.0]
WRATES = [1.2, 1.8, 2.4]
RATEK = {0.6: "ZeroSix", 1.2: "Main", 1.8: "OneEight", 2.4: "TwoFour", 3.0: "Three"}  # macro names are alphabetic
ZIPFK = {"0": "Zero", "0.8": "Main", "1.2": "Heavy"}
SLO = 10.0
EDITM = [("gptcache", "GPTCache"), ("vcache-official", "vCache"), (U, "\\textbf{Unison}")]
ZIPFM = [("suffix", "SuffixDecoding"), ("gptcache", "GPTCache"), (U, "\\textbf{Unison}")]
KEYN = {"noshare": "Vllm", "pld": "Pld", "suffix": "Suffix", "gptcache": "Gptcache", "vcache-official": "Vcache",
        SBD: "Sbd", SUF: "Suf", RULE: "Rule", NOPRE: "Nopre", U: "Unison", "u17-nopeer": "Nopeer", "u17-noctx": "Noctx",
        "u17-noh": "Noh", "u17-noplan": "Noplan", "u17-noid": "Noid", "u17-nodf": "Nodf"}
M = {}
WARN = []
ISO = {}


def mac(name, val):
    M["".join(ch for ch in name if ch.isalpha())] = val


def f2(x):
    return f"{x:.2f}"


def f1(x):
    return f"{x:.1f}"


def pct(a, b):
    """Reduction of a against b in percent."""
    return 100 * (1 - a / b) if b else float("nan")


def load(tag, ds):
    p = os.path.join(R, tag, ds, "summary.json")
    if not os.path.exists(p):
        WARN.append(f"missing {tag}/{ds}")
        return {}
    out = {r["method"]: r for r in json.load(open(p))}
    timed = not (tag.endswith("_edit") or tag.endswith("_smoke"))
    for m, r in out.items():
        if timed and r.get("gpu_others_sm_busy", 0) > 0.6:
            WARN.append(f"CONTAMINATED? {tag}/{ds}/{m}: co-tenant SM >= 40% in {100 * r['gpu_others_sm_busy']:.0f}% of samples -> rerun")
        elif timed and r.get("gpu_others_sm_mean", 0) > 5:
            WARN.append(f"NOISY {tag}/{ds}/{m}: mean co-tenant SM share {r['gpu_others_sm_mean']:.0f}%")
        if r.get("n_df_mismatch", 0):
            WARN.append(f"DATAFLOW MISMATCH {tag}/{ds}/{m}: {r['n_df_mismatch']}")
        # the host stopped the process (memory pressure): the stream's arrivals were late, the run is void
        lm, lp, lx = r.get("loop_lag_ms", 0.0), r.get("loop_lag_p99_ms", 0.0), r.get("loop_lag_max_ms")
        if timed and (lx > 3000 if lx is not None else (lm > 5 and lm > 0.25 * lp)):
            WARN.append(f"HOST STALL {tag}/{ds}/{m}: event-loop lag mean {lm:.1f} ms, p99 {lp:.0f} ms"
                        + (f", max {lx / 1e3:.1f} s" if lx is not None else "") + " -> rerun")
    if timed and out:  # isolation as measured, per comparison group (macros <Ds><Group>CoSm / CoMib / BaseMs)
        sm = [r.get("gpu_others_sm_mean", -1) for r in out.values() if r.get("gpu_others_sm_mean", -1) >= 0]
        grp = "".join(ch for ch in tag.replace(f"{V}_", "") if ch.isalpha()) or "x"
        k = DSK.get(ds, "X") + grp.capitalize()
        mac(f"{k}CoSm", f"{np.mean(sm):.0f}" if sm else "0")
        base = [r["cost"]["a"] * 1000 for r in out.values() if isinstance(r.get("cost"), dict)]
        mac(f"{k}BaseMs", f"{np.mean(base):.1f}" if base else "?")
        mib = [r.get("gpu_others_mib", 0) for r in out.values()]
        mac(f"{k}CoMib", f"{max(mib) / 1024:.1f}" if mib else "0")
        # isolation over the whole campaign: every timed run that enters the paper, its largest per-method mean
        # compute share of other processes and the memory they held
        ISO[(tag, ds)] = (max(sm) if sm else 0.0, max(mib) if mib else 0)
        mac("CoRuns", str(len(ISO)))
        mac("CoMibMax", f"{max(v[1] for v in ISO.values()):.0f}")
        mac("CoSmMax", f"{max(v[0] for v in ISO.values()):.1f}")
        mac("CoNoisyRuns", str(sum(v[0] > 5 for v in ISO.values())))
        mac("CoSharedRuns", str(sum(v[1] > 500 for v in ISO.values())))
        kv = {r.get("kv_blocks") for r in out.values()}
        if len(kv) > 1:
            WARN.append(f"KV capacity differs within {tag}/{ds}: {kv}")
    return out


def qs(tag, ds, m):
    p = os.path.join(R, tag, ds, f"{m}.queries.json")
    return {q["qi"]: q for q in json.load(open(p))} if os.path.exists(p) else None


def gpu_s(r):
    return r["engine_busy"] / max(1, r["n_queries"])


def nonexact(r):
    n = sum(r.get(f"n_{s}", 0) for s in ("dcert", "sem", "risk"))
    e = sum(r.get(f"nerr_{s}", 0) for s in ("dcert", "sem", "risk"))
    return n, e


def ident_pct(r):
    """Share of all key computations (plans and verifications) served by identity sharing."""
    n = r.get("n_verify", 0) + r.get("n_plan", 0)
    return 100 * (r.get("src_exact", 0) + r.get("n_plan_exact", 0)) / max(1, n)


def verified_pct(r):
    return 100 * r.get("src_dcert", 0) / max(1, r.get("n_verify", 0))


def passes(r, kind="key"):
    """Engine passes per call of this kind that the engine served (the first pass processes the prompt)."""
    return r.get(f"{kind}_passes", 0.0)


def one_pass(r, kind="key"):
    return 100 * r.get(f"{kind}_one_pass", 0.0)


def toks(r, kind="key"):
    """Output tokens per call of this kind that the engine served."""
    return r.get(f"{kind}_out_tokens", 0) / max(1, r.get(f"{kind}_engine_calls", 0))


def passes_all(r, kind="key"):
    """Engine passes per call of this kind, reused calls included (a reused call takes no pass)."""
    n = r.get("n_verify" if kind == "key" else "n_plan", 0)
    return passes(r, kind) * r.get(f"{kind}_engine_calls", 0) / max(1, n)


def cp_hi(k, n):
    return 100 * (beta.ppf(0.975, k + 1, n - k) if k < n else 1.0) if n else float("nan")


def paired(tag, ds, a, b, B=4000):
    qa, qb = qs(tag, ds, a), qs(tag, ds, b)
    if not qa or not qb:
        return None
    ids = sorted(set(qa) & set(qb))
    la = np.array([qa[i]["latency"] for i in ids])
    lb = np.array([qb[i]["latency"] for i in ids])
    ea = np.array([qa[i]["em"] for i in ids])
    eb = np.array([qb[i]["em"] for i in ids])
    rng = np.random.default_rng(0)
    red, dem = [], []
    for _ in range(B):
        j = rng.integers(0, len(ids), len(ids))
        red.append(100 * (1 - la[j].mean() / lb[j].mean()))
        dem.append(100 * (ea[j].mean() - eb[j].mean()))
    n01, n10 = int(((ea == 0) & (eb == 1)).sum()), int(((ea == 1) & (eb == 0)).sum())
    p = binomtest(n10, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    out = dict(red=100 * (1 - la.mean() / lb.mean()), red_lo=np.percentile(red, 2.5), red_hi=np.percentile(red, 97.5),
               dem=100 * (ea.mean() - eb.mean()), dem_lo=np.percentile(dem, 2.5), dem_hi=np.percentile(dem, 97.5),
               p=p, n=len(ids))
    # Greedy decoding in batches is not bit-reproducible: a decision at a numerical tie can send a query down another
    # course in one of the two runs. The same statistic on the queries that take the same course in both runs
    # (the same hops of every agent and the same team answer) compares like with like.
    same = np.array([qa[i].get("hops") == qb[i].get("hops") and qa[i].get("answer") == qb[i].get("answer") for i in ids])
    out["same_n"] = int(same.sum())
    if 0 < same.sum():
        sa, sb = la[same], lb[same]
        rs = []
        for _ in range(B):
            j = rng.integers(0, len(sa), len(sa))
            rs.append(100 * (1 - sa[j].mean() / sb[j].mean()))
        out.update(same_red=100 * (1 - sa.mean() / sb.mean()), same_lo=np.percentile(rs, 2.5),
                   same_hi=np.percentile(rs, 97.5))
    return out


def pair_macros(tag, ds, a, b, prefix):
    st = paired(tag, ds, a, b)
    if st:
        mac(f"{prefix}", f"{st['red']:.0f}")
        mac(f"{prefix}CI", f"[{st['red_lo']:.0f},{st['red_hi']:.0f}]")
        mac(f"{prefix}Lo", f"{st['red_lo']:.0f}")
        mac(f"{prefix}Hi", f"{st['red_hi']:.0f}")
        mac(f"{prefix}EMd", f"{st['dem']:+.1f}")
        mac(f"{prefix}EMdCI", f"[{st['dem_lo']:+.1f},{st['dem_hi']:+.1f}]")
        mac(f"{prefix}EMp", f"{st['p']:.2f}")
        mac(f"{prefix}Dec", f"{st['red']:.1f}")  # one decimal, for effects of a few percent
        mac(f"{prefix}DecCI", f"[{st['red_lo']:.1f},{st['red_hi']:.1f}]")
        mac(f"{prefix}SameN", str(st["same_n"]))
        mac(f"{prefix}OtherN", str(st["n"] - st["same_n"]))
        if "same_red" in st:
            mac(f"{prefix}Same", f"{st['same_red']:.0f}")
            mac(f"{prefix}SameCI", f"[{st['same_lo']:.0f},{st['same_hi']:.0f}]")
            mac(f"{prefix}SameDec", f"{st['same_red']:.1f}")
            mac(f"{prefix}SameDecCI", f"[{st['same_lo']:.1f},{st['same_hi']:.1f}]")
    return st


def em_spread():
    """Exact match of the systems that execute every call themselves, on the main streams. Their decisions are greedy
    outputs of the same prompts, so their answers differ only where a decision at a numerical tie fell the other way;
    a question that recurs in the stream then moves several queries at once. Macros per dataset: the range of EM over
    these systems, and for each of them against vLLM the queries whose EM differs, the distinct questions behind those
    queries and how often the most frequent of them occurs in the stream."""
    exact = ["noshare", "pld", "suffix", U]
    for ds in DS:
        S = load(f"{V}_main", ds)
        k = DSK[ds]
        ems = [S[m]["em"] for m in exact if m in S]
        if not ems:
            continue
        mac(f"{k}EmExactMin", f"{min(ems):.3f}")
        mac(f"{k}EmExactMax", f"{max(ems):.3f}")
        mac(f"{k}EmExactSpread", f"{100 * (max(ems) - min(ems)):.1f}")
        ref = qs(f"{V}_main", ds, "noshare")
        if not ref:
            continue
        pop = {}
        for q in ref.values():
            pop[q["qid"]] = pop.get(q["qid"], 0) + 1
        mac(f"{k}StreamQuestions", str(len(pop)))
        mac(f"{k}StreamTopCount", str(max(pop.values())))
        for m in exact[1:]:
            q = qs(f"{V}_main", ds, m)
            if not q:
                continue
            ids = sorted(set(q) & set(ref))
            diff = [i for i in ids if q[i]["em"] != ref[i]["em"]]
            by = {}
            for i in diff:
                by[q[i]["qid"]] = by.get(q[i]["qid"], 0) + 1
            key = KEYN[m]
            mac(f"{k}EmDiffN{key}", str(len(diff)))
            mac(f"{k}EmDiffQ{key}", str(len(by)))
            mac(f"{k}EmDiffTop{key}", str(max(by.values()) if by else 0))
            mac(f"{k}EmDiffTopStream{key}", str(pop[max(by, key=by.get)]) if by else "0")
            mac(f"{k}CourseDiffN{key}", str(sum(q[i].get("hops") != ref[i].get("hops") or q[i].get("answer") != ref[i].get("answer")
                                                for i in ids)))


def esteps(tag, ds, m):
    p = os.path.join(R, tag, ds, f"{m}.esteps.npy")
    if not os.path.exists(p):
        return None
    E = np.load(p)
    return E if E.ndim == 2 and len(E) else None


# ---------------------------------------------------------------- what a step of the engine costs
def step_model(E, B=1000, seed=0):
    """Least squares of the model's time in a step on its content: t = a + c_p * prompt positions + c_g * requests
    + c_d * draft positions (every request decodes one position; a request admitted in the step also has its
    prompt positions). Returns the coefficients, their bootstrap 95% intervals over steps, R^2 and the step count."""
    y = E[:, 1] - E[:, 7]
    req, pos, dr = E[:, 2], E[:, 3], E[:, 4]
    prm = np.clip(pos - req - dr, 0, None)
    ok = (y > 0) & (y < np.percentile(y, 99.5))
    X = np.c_[np.ones(ok.sum()), prm[ok], req[ok], dr[ok]]
    y = y[ok]
    use = [0, 1, 2] + ([3] if X[:, 3].any() else [])
    co = np.zeros(4)
    co[use] = np.linalg.lstsq(X[:, use], y, rcond=None)[0]
    res = y - X @ co
    r2 = 1 - res.var() / y.var()
    rng = np.random.default_rng(seed)
    bs = np.zeros((B, 4))
    for b in range(B):
        j = rng.integers(0, len(y), len(y))
        bs[b, use] = np.linalg.lstsq(X[j][:, use], y[j], rcond=None)[0]
    return co, np.percentile(bs, 2.5, 0), np.percentile(bs, 97.5, 0), r2, len(y), X, y


def cost_structure():
    """Table tab_cost: per system of the main runs, the engine's steps, positions and time per query, from the
    engine's own step log. Macros: the step-time model with intervals, and where the time of a run goes."""
    prof = []
    for tag in sorted(glob.glob(os.path.join(R, f"{V}_*"))):
        for ds in DS:
            p = os.path.join(tag, ds, "summary.json")
            if os.path.exists(p) and not tag.endswith(("_edit", "_smoke", "_collect", "_collect2")):
                prof += [r["cost"] for r in json.load(open(p)) if isinstance(r.get("cost"), dict)]
    if prof:  # the start-up profile every run takes of its own engine: step = a + delta * B + ctok * D
        for nm, key, sc, fm in (("CostBase", "a", 1e3, "{:.1f}"), ("CostGen", "delta", 1e6, "{:.0f}"),
                                ("CostDraft", "ctok", 1e6, "{:.0f}")):
            v = np.array([r[key] for r in prof]) * sc
            mac(nm, fm.format(np.median(v)))
            mac(nm + "Lo", fm.format(np.percentile(v, 25)))
            mac(nm + "Hi", fm.format(np.percentile(v, 75)))
        mac("CostProfiles", str(len(prof)))
    pooled = []
    with open(os.path.join(P, "tab_cost.tex"), "w") as fo:
        for ds in DS:
            S = load(f"{V}_main", ds)
            k = DSK[ds]
            rows, Es = [], []
            for m, lab in SHORT:
                r, E = S.get(m), esteps(f"{V}_main", ds, m)
                if r is None or E is None:
                    continue
                nq = max(1, r["n_queries"])
                model_s = (E[:, 1].sum() - E[:, 7].sum()) / nq
                draft_ms = 1e3 * (E[:, 6].sum() + E[:, 7].sum()) / nq
                cells = [f"{len(E) / nq:.0f}", f"{E[:, 3].sum() / nq / 1000:.2f}", f"{E[:, 4].sum() / nq:.0f}",
                         f"{1e3 * E[:, 1].mean():.1f}", f2(model_s), f"{draft_ms:.0f}", f2(gpu_s(r))]
                rows.append(f"{lab} & " + " & ".join(cells) + r" \\")
                key = KEYN[m]
                mac(f"{k}XSteps{key}", f"{len(E) / nq:.0f}")
                mac(f"{k}XPos{key}", f"{E[:, 3].sum() / nq / 1000:.2f}")
                mac(f"{k}XDraftPos{key}", f"{E[:, 4].sum() / nq:.0f}")
                mac(f"{k}XStepMs{key}", f"{1e3 * E[:, 1].mean():.1f}")
                mac(f"{k}XModelS{key}", f2(model_s))
                mac(f"{k}XDraftMs{key}", f"{draft_ms:.0f}")
                mac(f"{k}XDraftShare{key}", f"{100 * (E[:, 6].sum() + E[:, 7].sum()) / max(1e-9, E[:, 1].sum() + E[:, 6].sum()):.1f}")
                mac(f"{k}XReq{key}", f"{E[:, 2].mean():.1f}")
                Es.append((m, E))
            if rows:
                fo.write(f"\\multicolumn{{8}}{{l}}{{\\textit{{{DS[ds]}}}}}\\\\\n" + "\n".join(rows) + "\n\\midrule\n")
            if not Es:
                WARN.append(f"no engine step logs for {V}_main/{ds}")
                continue
            allE = np.concatenate([E for _, E in Es])
            pooled.append(allE)
            co, lo, hi, r2, n, _, _ = step_model(allE)
            for j, (nm, sc, fm) in enumerate((("Base", 1e3, "{:.2f}"), ("Prompt", 1e6, "{:.0f}"), ("Gen", 1e6, "{:.0f}"),
                                              ("Draft", 1e6, "{:.0f}"))):
                mac(f"{k}Reg{nm}", fm.format(co[j] * sc))
                mac(f"{k}Reg{nm}CI", "[" + fm.format(lo[j] * sc) + "," + fm.format(hi[j] * sc) + "]")
            mac(f"{k}RegRsq", f2(r2))
            mac(f"{k}RegSteps", f"{n:,}".replace(",", "{,}"))
            for m, E in Es:  # where the model's time of a run goes, by the fitted model
                req, pos, dr = E[:, 2], E[:, 3], E[:, 4]
                prm = np.clip(pos - req - dr, 0, None)
                parts = np.array([co[0] * len(E), co[1] * prm.sum(), co[2] * req.sum(), co[3] * dr.sum()])
                parts = np.clip(parts, 0, None)
                sh = 100 * parts / max(1e-9, parts.sum())
                for nm, v in zip(("Fixed", "Prompt", "Gen", "Draft"), sh):
                    mac(f"{k}Share{nm}{KEYN[m]}", f"{v:.0f}")
    if pooled:
        co, lo, hi, r2, n, _, _ = step_model(np.concatenate(pooled))
        for j, (nm, sc, fm) in enumerate((("Base", 1e3, "{:.2f}"), ("Prompt", 1e6, "{:.0f}"), ("Gen", 1e6, "{:.0f}"),
                                          ("Draft", 1e6, "{:.0f}"))):
            mac(f"Reg{nm}", fm.format(co[j] * sc))
            mac(f"Reg{nm}CI", "[" + fm.format(lo[j] * sc) + "," + fm.format(hi[j] * sc) + "]")
        mac("RegRsq", f2(r2))
        mac("RegSteps", f"{n:,}".replace(",", "{,}"))
        if co[2] > 0:
            mac("RegDraftOverGen", f2(co[3] / co[2]))
        mac("RegBaseOverGen", f"{co[0] / max(1e-12, co[2]):.0f}")
        if co[3] > 0:  # draft positions that cost as much as one pass: measured iteration time over the fitted gamma_d
            for ds in DS:
                it = load(f"{V}_main", ds).get(U, {}).get("nd_iter_ms")
                if it:
                    mac(f"{DSK[ds]}UPassWorth", f"{it * 1e-3 / co[3]:.0f}")


# ---------------------------------------------------------------- main table
def main_table():
    rows = {m: [] for m, _ in MAIN}
    for ds in DS:
        S = load(f"{V}_main", ds)
        k = DSK[ds]
        floor = None
        for m, _ in MAIN:
            r = S.get(m)
            if r is None:
                continue
            n, e = nonexact(r)
            # last column: every delivered verification decision against a re-execution of its own prompt (%)
            rows[m].append(f"{r['lat_mean']:.2f} & {r['lat_p99']:.1f} & {r['em']:.3f} & {gpu_s(r):.2f} & "
                           f"{passes(r):.1f} & {100 * r.get('dec_err_total', 0.0):.2f}")
            if r.get("nd_prompts"):
                floor = (r["nd_decision_diff"], r["nd_prompts"])
            key = KEYN[m]
            mac(f"{k}Lat{key}", f2(r["lat_mean"]))
            mac(f"{k}Pnn{key}", f"{r['lat_p99']:.1f}")
            mac(f"{k}EM{key}", f"{r['em']:.3f}")
            mac(f"{k}Gpu{key}", f2(gpu_s(r)))
            mac(f"{k}Reuse{key}", str(n))
            mac(f"{k}ReuseErr{key}", str(e))
            mac(f"{k}ReuseErrPct{key}", f"{100 * e / n:.1f}" if n else "0")
            mac(f"{k}Ident{key}", f"{ident_pct(r):.0f}")
            mac(f"{k}Verif{key}", f"{verified_pct(r):.0f}")
            mac(f"{k}GenTok{key}", f"{r.get('gen_tokens', 0) / 1000:.0f}")
            mac(f"{k}Iters{key}", f"{r.get('engine_iters', 0):,}".replace(",", "{,}"))
            # every delivered verification decision against an independent re-execution of its own prompt
            nv = r.get("n_verify", 0)
            mac(f"{k}NVerify{key}", f"{nv:,}".replace(",", "{,}"))
            mac(f"{k}DecErrN{key}", str(int(round(r.get("dec_err_total", 0.0) * nv))))
            mac(f"{k}DecErrPct{key}", f"{100 * r.get('dec_err_total', 0.0):.2f}")
            mac(f"{k}FidPlan{key}", f"{100 * r.get('fidelity_plan', 1.0):.1f}")
            for kind, kn in (("key", ""), ("plan", "Plan")):
                mac(f"{k}{kn}Pass{key}", f1(passes(r, kind)))
                mac(f"{k}{kn}One{key}", f"{one_pass(r, kind):.0f}")
                mac(f"{k}{kn}Tok{key}", f1(toks(r, kind)))
                mac(f"{k}{kn}PassAll{key}", f1(passes_all(r, kind)))
                mac(f"{k}{kn}TokPerPass{key}", f1(toks(r, kind) / max(1e-9, passes(r, kind))))
        ns = S.get("suffix") or S.get("noshare")  # the same source as the redundancy table (zipf_table)
        if ns and "redund_passage60" in ns:
            mac(f"{k}RedPass", f"{100 * ns['redund_passage60']:.0f}")
            mac(f"{k}RedIdent", f"{100 * ns['redund_identical']:.0f}")
        if S:  # isolation as measured: co-tenant SM share and the fixed KV capacity of the comparison
            sm = [r.get("gpu_others_sm_mean", -1) for r in S.values() if r.get("gpu_others_sm_mean", -1) >= 0]
            mac(f"{k}CoSm", f"{np.mean(sm):.0f}" if sm else "0")
            kvb = next((r.get("kv_blocks") for r in S.values() if r.get("kv_blocks")), None)
            if kvb:
                mac("KVTokens", f"{16 * kvb // 1000}k")
        if floor:
            mac(f"{k}NDdiff", str(floor[0]))
            mac(f"{k}NDn", str(floor[1]))
            mac(f"{k}NDpct", f"{100 * floor[0] / max(1, floor[1]):.2f}")
        u = S.get(U)
        if u:
            n, e = nonexact(u)
            mac(f"{k}UCertN", str(n))
            mac(f"{k}UCertErr", str(e))
            mac(f"{k}UCertHi", f"{cp_hi(e, n):.2f}" if n else "0")
            mac(f"{k}UProvReq", str(u.get("prov_requests", 0)))
            mac(f"{k}UProvEng", str(u.get("prov_engine", 0)))
            mac(f"{k}UAuditLag", f2(u.get("prov_lag_p50", 0)))
            mac(f"{k}UPlanExact", str(u.get("n_plan_exact", 0)))
            mac(f"{k}UPlanN", str(u.get("n_plan", 0)))
            mac(f"{k}UTeamEarly", str(u.get("n_team_early", 0)))
            mac(f"{k}URoundEarly", str(u.get("n_round_early", 0)))
            mac(f"{k}UExact", str(u.get("src_exact", 0)))
            mac(f"{k}UVerify", str(u.get("n_verify", 0)))
            mac(f"{k}UKeyCalls", str(u.get("key_engine_calls", 0)))
            mac(f"{k}UPlanCalls", str(u.get("plan_engine_calls", 0)))
            if u.get("nd_calls"):
                mac(f"{k}UPrice", f"{u.get('nd_price', 0):.3f}")
                mac(f"{k}UStepPrice", f"{u.get('nd_sprice', 0):.3f}")
                mac(f"{k}UIterMs", f"{u.get('nd_iter_ms', 0):.1f}")
                mac(f"{k}UDrafterS", f2(u.get("nd_seconds", 0)))
                mac(f"{k}UDrafterShare", f"{100 * u.get('nd_seconds', 0) / max(1e-9, u['engine_busy']):.1f}")
            acc = sum(u.get(x, 0) for x in ("acc_pre", "acc_nd", "pacc_pre", "pacc_nd"))
            dra = u.get("acc_drafted", 0) + u.get("pacc_drafted", 0)
            mac(f"{k}UAcc", f"{100 * acc / max(1, dra):.0f}")
            kt = max(1, u.get("key_out_tokens", 0))
            mac(f"{k}UKeyFromPre", f"{100 * u.get('acc_pre', 0) / kt:.0f}")
            mac(f"{k}UKeyFromState", f"{100 * u.get('acc_nd', 0) / kt:.0f}")
            pt = max(1, u.get("plan_out_tokens", 0))
            mac(f"{k}UPlanFromPre", f"{100 * u.get('pacc_pre', 0) / pt:.0f}")
            mac(f"{k}UPlanFromState", f"{100 * u.get('pacc_nd', 0) / pt:.0f}")
        pubs = [(S[m]["lat_mean"], m) for m in PUB if m in S]
        if pubs and u:
            best = min(pubs)[1]
            mac(f"{k}BestPub", dict(SHORT)[best])
            pair_macros(f"{V}_main", ds, U, best, f"{k}LatRedBest")
            gb = min((gpu_s(S[m]), m) for m in PUB if m in S)
            mac(f"{k}GpuRedBest", f"{pct(gpu_s(u), gb[0]):.0f}")
            mac(f"{k}PnnRedBest", f"{pct(u['lat_p99'], S[best]['lat_p99']):.0f}")
            exact_pub = [(S[m]["lat_mean"], m) for m in ("noshare", "pld", "suffix") if m in S]
            if exact_pub:  # the fastest published system that never returns another call's result unchecked
                be = min(exact_pub)[1]
                mac(f"{k}BestExact", dict(SHORT)[be])
                pair_macros(f"{V}_main", ds, U, be, f"{k}LatRedBestExact")
            for b in PUB:
                if b in S:
                    pair_macros(f"{V}_main", ds, U, b, f"{k}LatRed{KEYN[b]}")
                    mac(f"{k}GpuRed{KEYN[b]}", f"{pct(gpu_s(u), gpu_s(S[b])):.0f}")
                    mac(f"{k}PnnRed{KEYN[b]}", f"{pct(u['lat_p99'], S[b]['lat_p99']):.0f}")
                    mac(f"{k}PassRed{KEYN[b]}", f"{pct(passes(u), passes(S[b])):.0f}")
                    mac(f"{k}Speedup{KEYN[b]}", f"{S[b]['lat_mean'] / u['lat_mean']:.1f}")
            if "noshare" in S:
                mac(f"{k}GenRedVllm", f"{pct(u.get('gen_tokens', 0), max(1, S['noshare'].get('gen_tokens', 1))):.0f}")
    with open(os.path.join(P, "tab_main.tex"), "w") as fo:
        for m, lab in MAIN:
            if not rows[m]:
                continue
            if m == U:
                fo.write("\\midrule\n")
            fo.write(f"{lab} & " + " & ".join(rows[m] + ["? & ? & ? & ? & ? & ?"] * (len(DS) - len(rows[m]))) + r" \\" + "\n")


# ---------------------------------------------------------------- three independent streams
SEEDS = [("", "0"), ("_s1", "1"), ("_s2", "2")]


def seeds_table():
    """Mean latency of every system on three streams (other arrival times, another draw and paraphrasing of the
    questions), the mean and standard deviation over the streams, and the range of Unison's reduction."""
    lat = {ds: {} for ds in DS}
    em = {ds: {} for ds in DS}  # per system: the exact-match flags of the queries of all streams, keyed (stream, query)
    for ds in DS:
        for sfx, _ in SEEDS:
            S = load(f"{V}_main{sfx}", ds) if os.path.exists(os.path.join(R, f"{V}_main{sfx}", ds, "summary.json")) else {}
            for m, _ in SHORT:
                lat[ds].setdefault(m, []).append(S[m]["lat_mean"] if m in S else None)
                q = qs(f"{V}_main{sfx}", ds, m) if m in S else None
                for i, x in (q or {}).items():
                    em[ds].setdefault(m, {})[(sfx, i)] = int(x["em"])
    pooled = {}
    for ds in DS:  # exact match pooled over the streams, and an exact McNemar test of every system against vLLM
        k = DSK[ds]
        ref = em[ds].get("noshare", {})
        for m, _ in SHORT:
            e = em[ds].get(m, {})
            if not e:
                continue
            pooled[(ds, m)] = float(np.mean(list(e.values())))
            mac(f"{k}SeedEm{KEYN[m]}", f"{pooled[(ds, m)]:.3f}")
            mac(f"{k}SeedEmN{KEYN[m]}", str(len(e)))
            ids = sorted(set(e) & set(ref))
            if m != "noshare" and ids:
                n10 = sum(e[i] == 1 and ref[i] == 0 for i in ids)
                n01 = sum(e[i] == 0 and ref[i] == 1 for i in ids)
                pv = binomtest(n10, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
                # with its relation, for "$p\Macro$": "=0.62" or "<0.001"
                mac(f"{k}SeedEmP{KEYN[m]}", "<0.001" if pv < 0.001 else (f"={pv:.3f}" if pv < 0.01 else f"={pv:.2f}"))
                mac(f"{k}SeedEmD{KEYN[m]}", f"{100 * (np.mean([e[i] for i in ids]) - np.mean([ref[i] for i in ids])):+.1f}")
                mac(f"{k}SeedEmWin{KEYN[m]}", str(n10))
                mac(f"{k}SeedEmLoss{KEYN[m]}", str(n01))
    with open(os.path.join(P, "tab_seeds.tex"), "w") as fo:
        for m, lab in SHORT:
            cells = []
            for ds in DS:
                v = lat[ds].get(m, [None] * len(SEEDS))
                have = [x for x in v if x is not None]
                cells += [f2(x) if x is not None else "?" for x in v]
                cells.append(f"{np.mean(have):.2f}$\\pm${np.std(have, ddof=1):.2f}" if len(have) > 1 else "?")
                cells.append(f"{pooled[(ds, m)]:.3f}" if len(have) == len(SEEDS) and (ds, m) in pooled else "?")
                if have:
                    mac(f"{DSK[ds]}SeedLat{KEYN[m]}", f2(np.mean(have)))
                    mac(f"{DSK[ds]}SeedSd{KEYN[m]}", f2(np.std(have, ddof=1)) if len(have) > 1 else "?")
            if m == U:
                fo.write("\\midrule\n")
            fo.write(f"{lab} & " + " & ".join(cells) + r" \\" + "\n")
    for ds in DS:
        k = DSK[ds]
        u = lat[ds].get(U, [])
        n = sum(x is not None for x in u)
        mac(f"{k}SeedN", str(n))
        for b in PUB:
            red = [pct(a, c) for a, c in zip(u, lat[ds].get(b, [])) if a is not None and c is not None]
            if red:
                mac(f"{k}SeedRed{KEYN[b]}Min", f"{min(red):.0f}")
                mac(f"{k}SeedRed{KEYN[b]}Max", f"{max(red):.0f}")
        best = []
        for i in range(len(SEEDS)):
            pub = [lat[ds][b][i] for b in PUB if b in lat[ds] and lat[ds][b][i] is not None]
            if pub and u[i] is not None:
                best.append(pct(u[i], min(pub)))
        if best:
            mac(f"{k}SeedRedBestMin", f"{min(best):.0f}")
            mac(f"{k}SeedRedBestMax", f"{max(best):.0f}")
            mac(f"{k}SeedRedBestMean", f"{np.mean(best):.0f}")


def _mpl_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8.5,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 8.0,
        "axes.linewidth": 0.6,
        "grid.linewidth": 0.45,
        "grid.alpha": 0.28,
        "grid.color": "#888888",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    return plt


OI = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "gray": "#595959",
}


# ---------------------------------------------------------------- passes per call, every system
def pass_table():
    """What a call costs in sequential passes of the engine, counted by the engine for every system: output tokens
    and passes per served call, the share of calls that took one pass; verification calls and plans."""
    with open(os.path.join(P, "tab_pass.tex"), "w") as fo:
        for m, lab in SHORT:
            cells = []
            for ds in DS:
                r = load(f"{V}_main", ds).get(m)
                if r is None or not r.get("key_engine_calls"):
                    cells += ["?"] * 6
                    continue
                for kind in ("key", "plan"):
                    cells += [f1(toks(r, kind)), f1(passes(r, kind)), f"{one_pass(r, kind):.0f}"]
            if m == U:
                fo.write("\\midrule\n")
            fo.write(f"{lab} & " + " & ".join(cells) + r" \\" + "\n")
    try:  # distribution of the passes per served call, attribution runs
        plt = _mpl_style()
        fig, ax = plt.subplots(1, 4, figsize=(6.8, 2.1), sharey=True)
        style = {"suffix": ("SuffixDecoding", OI["gray"], ":"), SUF: ("(A) suffix lookup", OI["orange"], "--"),
                 NOPRE: ("(C) drafter after first pass", OI["green"], "-."), U: ("Unison", OI["blue"], "-")}
        any_line = False
        for i, ds in enumerate(DS):
            A, Mn = load(f"{V}_abl", ds), load(f"{V}_main", ds)
            for j, kind in enumerate(("key", "plan")):
                a = ax[2 * i + j]
                for m, (lab, col, ls) in style.items():
                    r = Mn.get(m) if m == "suffix" else A.get(m)
                    h = np.array((r or {}).get(f"{kind}_pass_hist", []), dtype=float)
                    if h.sum() == 0:
                        continue
                    c = np.cumsum(h) / h.sum()
                    a.step(np.arange(len(h)), c, where="post", label=lab, color=col, ls=ls, lw=1.35)
                    any_line = True
                a.set_xscale("log")
                a.set_xlim(1, 64)
                a.set_ylim(0, 1.03)
                a.set_xticks([1, 2, 4, 8, 16, 32, 64])
                a.set_xticklabels(["1", "2", "4", "8", "16", "32", "64"])
                a.minorticks_off()
                a.set_title(f"{DS[ds]}, {'verification' if kind == 'key' else 'plan'}", fontsize=8.5)
                a.set_xlabel("passes", fontsize=8.5)
                a.grid(True, alpha=0.28, lw=0.45, color="#888888")
                a.tick_params(labelsize=8.0, width=0.6, length=2.5)
                for sp in a.spines.values():
                    sp.set_visible(True)
                    sp.set_linewidth(0.6)
        ax[0].set_ylabel("share of calls", fontsize=8.5)
        fig.tight_layout(pad=0.35, rect=(0, 0, 1, 0.86))
        if any_line:  # one legend for the four panels, above them
            hs, ls_ = ax[0].get_legend_handles_labels()
            fig.legend(hs, ls_, fontsize=8.0, frameon=False, loc="upper center", ncol=len(hs),
                       bbox_to_anchor=(0.5, 1.02), handlelength=2.2, columnspacing=1.5)
        fig.savefig(os.path.join(P, "figs", "passes.pdf"))
        plt.close(fig)
    except Exception as e:
        WARN.append(f"passes figure: {e!r}")


# ---------------------------------------------------------------- where the time of a call goes
def ctimes(tag, ds):
    """Per method: time to the first token and after it by call kind and decision, the composition of a verification
    round and of an agent's critical path (code/scripts/call_times.py, from the per-call log and the step log)."""
    p = os.path.join(R, tag, ds, "calltimes.json")
    if not os.path.exists(p):
        WARN.append(f"missing call times {tag}/{ds} (run code/scripts/call_times.py on the server)")
        return {}
    return json.load(open(p))


def ms0(x):
    return "?" if x is None else f"{x:.0f}"


def round_ms(c):
    """Mean duration of a verification round: from its first call to the agent's next plan."""
    a = c.get("agent", {})
    return a["round_ms"] / a["rounds"] if a.get("rounds") and a.get("round_ms") is not None else None


def time_macros(prefix, c):
    v, p, a, rd = c.get("verify", {}), c.get("plan", {}), c.get("agent", {}), c.get("round", {})
    vp, vn = c.get("verify_pos", {}), c.get("verify_neg", {})
    for nm, x in (("VFirst", v.get("ttft_ms")), ("VRest", v.get("after_ms")), ("PFirst", p.get("ttft_ms")),
                  ("PRest", p.get("after_ms")), ("VWait", v.get("wait_ms")), ("VStep", v.get("first_step_ms")),
                  ("Round", round_ms(c)), ("RoundPos", rd.get("pos_ms")), ("RoundNeg", rd.get("neg_ms")),
                  ("AgSpan", a.get("span_ms")), ("AgPlan", a.get("plan_ms")), ("AgRound", a.get("round_ms")),
                  ("AgOther", a.get("other_ms")), ("PosRest", vp.get("after_ms")), ("NegRest", vn.get("after_ms")),
                  ("VStepDrafts", v.get("first_step_drafts")), ("VStepPos", v.get("first_step_positions"))):
        mac(f"{prefix}{nm}", ms0(x))
    if v.get("ttft_ms") is not None and v.get("after_ms") is not None:
        mac(f"{prefix}VCall", ms0(v["ttft_ms"] + v["after_ms"]))
        mac(f"{prefix}VFirstShare", f"{100 * v['ttft_ms'] / (v['ttft_ms'] + v['after_ms']):.0f}")
    if p.get("ttft_ms") is not None and p.get("after_ms") is not None:
        mac(f"{prefix}PCall", ms0(p["ttft_ms"] + p["after_ms"]))
    for nm, x in (("VLater", v.get("later_pass_ms")), ("PLater", p.get("later_pass_ms"))):
        mac(f"{prefix}{nm}", "?" if x is None else f"{x:.0f}")
    mac(f"{prefix}PosShare", f"{100 * c.get('pos_share', 0):.0f}")
    for nm, x in (("PosPass", vp.get("passes")), ("NegPass", vn.get("passes"))):
        mac(f"{prefix}{nm}", "?" if x is None else f1(x))
    for nm, x in (("PosOne", vp.get("one_pass")), ("NegOne", vn.get("one_pass"))):
        mac(f"{prefix}{nm}", "?" if x is None else f"{100 * x:.0f}")
    n = rd.get("pos_n", 0) + rd.get("neg_n", 0)
    mac(f"{prefix}RoundPosShare", f"{100 * rd.get('pos_n', 0) / n:.0f}" if n else "?")
    ca = c.get("cache", {})
    if ca:  # the two cache services: a lookup, a hit from arrival to return, the put after a miss
        for nm, key in (("Look", "look_ms"), ("LookMed", "look_p50_ms"), ("Hit", "hit_ms"), ("Put", "post_ms"),
                        ("Emb", "emb_ms")):
            mac(f"{prefix}{nm}", ms0(ca.get(key)))
        mac(f"{prefix}HitShare", f"{100 * ca['hits'] / max(1, ca['n']):.0f}")
    st = c.get("step", {})
    if st:
        mac(f"{prefix}Busy", f"{100 * st.get('busy_share', 0):.0f}")
        mac(f"{prefix}AdmitMs", ms0(st.get("admit_ms")))
        mac(f"{prefix}OtherMs", ms0(st.get("other_ms")))


def time_table():
    """Main runs, every system: milliseconds to the first token of a served call and after it, for verification
    calls and plans, and the mean duration of a verification round."""
    with open(os.path.join(P, "tab_time.tex"), "w") as fo:
        for m, lab in SHORT:
            cells = []
            for ds in DS:
                c = ctimes(f"{V}_main", ds).get(m)
                if not c:
                    cells += ["?"] * 5
                    continue
                v, p = c["verify"], c["plan"]
                cells += [ms0(v.get("ttft_ms")), ms0(v.get("after_ms")), ms0(p.get("ttft_ms")), ms0(p.get("after_ms")),
                          ms0(round_ms(c))]
                time_macros(f"{DSK[ds]}T{KEYN[m]}", c)
            if m == U:
                fo.write("\\midrule\n")
            fo.write(f"{lab} & " + " & ".join(cells) + r" \\" + "\n")
    for ds in DS:
        C = ctimes(f"{V}_main", ds)
        if U in C:
            for m in PUB:
                if m in C and C[m]["verify"].get("after_ms") and C[U]["verify"].get("after_ms"):
                    mac(f"{DSK[ds]}TRestRatio{KEYN[m]}", f"{C[m]['verify']['after_ms'] / C[U]['verify']['after_ms']:.0f}")


# ---------------------------------------------------------------- attribution
def abl_table():
    """Columns: mean latency, P99, GPU s/query, passes per served verification call, the share that took one pass
    and the milliseconds after the first token; passes and milliseconds after the first token of a plan."""
    with open(os.path.join(P, "tab_abl.tex"), "w") as fo:
        for ds in DS:
            S = load(f"{V}_abl", ds)
            C = ctimes(f"{V}_abl", ds)
            k = DSK[ds]
            fo.write(f"\\multicolumn{{9}}{{l}}{{\\textit{{{DS[ds]}}}}}\\\\\n")
            for m, lab in ABL:
                r = S.get(m)
                if r is None:
                    fo.write(f"{lab} & " + " & ".join(["?"] * 8) + r" \\" + "\n")
                    continue
                c = C.get(m, {})
                cells = [f2(r["lat_mean"]), f"{r['lat_p99']:.1f}", f2(gpu_s(r)),
                         f1(passes(r)), f"{one_pass(r):.0f}", ms0(c.get("verify", {}).get("after_ms")),
                         f1(passes(r, 'plan')), ms0(c.get("plan", {}).get("after_ms"))]
                fo.write(f"{lab} & " + " & ".join(cells) + r" \\" + "\n")
                key = KEYN[m]
                if c:
                    time_macros(f"{k}A{key}", c)
                mac(f"{k}A{key}Lat", f2(r["lat_mean"]))
                mac(f"{k}A{key}Pnn", f"{r['lat_p99']:.1f}")
                mac(f"{k}A{key}Gpu", f2(gpu_s(r)))
                mac(f"{k}A{key}Ident", f"{ident_pct(r):.0f}")
                mac(f"{k}A{key}Verif", f"{verified_pct(r):.0f}")
                mac(f"{k}A{key}Pass", f1(passes(r)))
                mac(f"{k}A{key}One", f"{one_pass(r):.0f}")
                mac(f"{k}A{key}PlanPass", f1(passes(r, "plan")))
                mac(f"{k}A{key}PlanOne", f"{one_pass(r, 'plan'):.0f}")
                mac(f"{k}A{key}Iters", f"{r.get('engine_iters', 0):,}".replace(",", "{,}"))
                mac(f"{k}A{key}EM", f"{r['em']:.3f}")
                mac(f"{k}A{key}GenTok", f"{r.get('gen_tokens', 0) / 1000:.0f}")
                mac(f"{k}A{key}ProvEng", str(r.get("prov_engine", 0)))
                if r.get("x_steps"):
                    mac(f"{k}A{key}XSteps", f"{r['x_steps'] / max(1, r['n_queries']):.0f}")
                    mac(f"{k}A{key}XDraftMs", f"{1e3 * (r.get('x_pre_s', 0) + r.get('x_nd_s', 0)) / max(1, r['n_queries']):.0f}")

            def lat(m):
                return S[m]["lat_mean"] if m in S else None
            pairs = {"Total": (U, SUF), "PreGain": (U, NOPRE), "DrafterGain": (NOPRE, SUF), "LearnGain": (NOPRE, RULE),
                     "RuleGain": (RULE, SUF), "KeyedGain": (SUF, SBD), "PeerGain": (U, "u17-nopeer"),
                     "CtxGain": (U, "u17-noctx"),
                     "StateGain": (U, "u17-noh"), "PlanGain": (U, "u17-noplan"), "IdentGain": (U, "u17-noid"),
                     "DfGain": (U, "u17-nodf")}
            for nm, (a, b) in pairs.items():
                if lat(a) and lat(b):
                    st = pair_macros(f"{V}_abl", ds, a, b, f"{k}Abl{nm}")
                    if st is None:
                        mac(f"{k}Abl{nm}", f"{pct(lat(a), lat(b)):.0f}")
                    mac(f"{k}Abl{nm}Gpu", f"{pct(gpu_s(S[a]), gpu_s(S[b])):.0f}")
                    mac(f"{k}Abl{nm}Pnn", f"{pct(S[a]['lat_p99'], S[b]['lat_p99']):.0f}")
                    mac(f"{k}Abl{nm}Iters", f"{pct(S[a].get('engine_iters', 1), max(1, S[b].get('engine_iters', 1))):.0f}")
                    mac(f"{k}Abl{nm}Pass", f"{pct(passes(S[a]), max(1e-9, passes(S[b]))):.0f}")
                    mac(f"{k}Abl{nm}PlanPass", f"{pct(passes(S[a], 'plan'), max(1e-9, passes(S[b], 'plan'))):.0f}")
                    mac(f"{k}Abl{nm}Inv", f"{100 * (lat(b) / lat(a) - 1):.0f}")  # how much slower b is than a
            for a, b, nm in ((U, SUF, "Total"), (NOPRE, SUF, "Drafter"), (U, NOPRE, "Pre")):  # time after the first token
                ca, cb = C.get(a), C.get(b)
                if ca and cb:
                    for kind, kk in (("verify", "V"), ("plan", "P")):
                        xa, xb = ca[kind].get("after_ms"), cb[kind].get("after_ms")
                        if xa and xb:
                            mac(f"{k}Abl{nm}{kk}Rest", f"{pct(xa, xb):.0f}")
            fo.write("\\midrule\n")


# ---------------------------------------------------------------- offline drafter evaluation
def drafter_table():
    p = os.path.join(R, "drafter", "eval.json")
    if not os.path.exists(p):
        WARN.append("missing drafter/eval.json")
        return
    E = json.load(open(p))
    ev = E["eval"]
    cfg = E.get("cfg", {})
    mac("DrParams", f"{E['parameters'] / 1e6:.1f}")
    mac("DrVocab", f"{E['vocab']:,}".replace(",", "{,}"))
    mac("DrTrainWin", f"{E['n_train_windows'] / 1000:.0f}")
    mac("DrValWin", f"{E['n_val_windows'] / 1000:.0f}")
    mac("DrEpochs", str(E.get("epochs", len(E["log"]))))
    mac("DrEpochKept", f2(E.get("epoch_kept", 0)))
    mac("DrBeta", f2(E.get("beta", 1.0)))
    mac("DrPeerLen", str(cfg.get("peer_len", 48)))
    mac("DrMaxPeers", str(cfg.get("max_peers", 4)))
    mac("DrCtxLen", str(cfg.get("ctx_len", 320)))
    mac("DrHidden", str(cfg.get("d", 384)))
    mac("DrTrainMin", f"{E.get('train_seconds', 0) / 60:.0f}")
    mac("DrOov", f"{100 * E.get('oov_held_out', 0):.1f}")
    mac("DrQTrain", str(E["questions_train"]))
    mac("DrQDev", str(E["questions_dev"]))
    mac("DrQHeld", str(E["questions_held_out"]))
    for nm, key in (("DrCallsVal", "calls_val"), ("DrCallsTrain", "calls_train"), ("DrCallsDev", "calls_dev")):
        mac(nm, f"{E[key]:,}".replace(",", "{,}"))
    prices = E.get("prices", [0.03, 0.1, 0.3])
    lo, mid, hi = prices[0], prices[len(prices) // 2], prices[-1]
    mac("DrPrice", f"{lo:.2f}")
    mac("DrPriceMid", f"{mid:.1f}")
    mac("DrPriceHi", f"{hi:.1f}")
    if "cost" in ev:
        mac("DrCostMsStep", f"{ev['cost']['ms_per_step']:.1f}")
    rows = [("Lookup", "sim|lookup", "Suffix lookup"),
            ("Rule", "sim|rule", "Longest peer continuation (rule)"),
            ("Cd", f"sim|cd@{lo}", "Record drafter after the first pass"),
            ("CdNopeer", f"sim|cd-nopeer@{lo}", r"\quad record without peer outputs"),
            ("CdNoctx", f"sim|cd-noctx@{lo}", r"\quad record without the prompt tail"),
            ("CdNoh", f"sim|cd-noh@{lo}", r"\quad without the consumer state"),
            ("Static", f"sim0|static+cd@{lo}", "Majority key in the first pass, then drafter"),
            ("Pre", f"sim0|cd@{lo}+cd@{lo}", r"\textbf{Record drafter in every pass}"),
            ("PreNopeer", f"sim0|cd-nopeer@{lo}+cd-nopeer@{lo}", r"\quad record without peer outputs"),
            ("PreNoctx", f"sim0|cd-noctx@{lo}+cd-noctx@{lo}", r"\quad record without the prompt tail"),
            ("PreNoh", f"sim0|cd-noh@{lo}+cd-noh@{lo}", r"\quad without the consumer state")]
    with open(os.path.join(P, "tab_drafter.tex"), "w") as fo:
        for name, key, lab in rows:
            cells = []
            for kind in ("verify", "plan", "all"):
                s = ev.get(f"{key}|{kind}")
                kn = kind.capitalize()
                if s is None:
                    cells += ["?"] * (1 if kind == "all" else 2)
                    continue
                cells.append(f2(s["passes"]))
                if kind != "all":
                    cells.append(f"{100 * s['one_pass']:.0f}")
                mac(f"DrSim{name}{kn}Pass", f2(s["passes"]))
                mac(f"DrSim{name}{kn}One", f"{100 * s['one_pass']:.0f}")
                mac(f"DrSim{name}{kn}Drafted", f1(s["drafted"]))
                mac(f"DrSim{name}{kn}Accepted", f1(s["accepted"]))
                mac(f"DrSimTokens{kn}", f1(s["tokens"]))
            s = ev.get(f"{key}|all")
            cells += [f1(s["drafted"]), f1(s["accepted"])] if s else ["?", "?"]
            if name in ("Cd", "Static"):
                fo.write("\\midrule\n")
            fo.write(f"{lab} & " + " & ".join(cells) + r" \\" + "\n")
    for pr, pk in ((lo, "Lo"), (mid, "Mid"), (hi, "Hi")):  # the price sweep of the full drafter
        for kind in ("verify", "plan", "all"):
            s = ev.get(f"sim0|cd@{pr}+cd@{pr}|{kind}")
            if s:
                kn = kind.capitalize()
                mac(f"DrPrice{pk}{kn}Pass", f2(s["passes"]))
                mac(f"DrPrice{pk}{kn}Drafted", f1(s["drafted"]))
                mac(f"DrPrice{pk}{kn}Accepted", f1(s["accepted"]))
                mac(f"DrPrice{pk}{kn}Waste", f1(s["drafted"] - s["accepted"]))
    # the first-pass draft by kind: tokens drafted and accepted, whole keys
    for name, key in (("Pre", f"pre:cd@{lo}"), ("PreStatic", "pre:static"), ("PreRule", "pre:rule"),
                      ("PreNoctx", f"pre:cd-noctx@{lo}"), ("PreNopeer", f"pre:cd-nopeer@{lo}")):
        for kind in ("verify", "plan"):
            r = ev.get(f"{key}|{kind}|all")
            if r:
                kn = kind.capitalize()
                mac(f"Dr{name}{kn}Whole", f"{100 * r['whole']:.1f}")
                mac(f"Dr{name}{kn}Draft", f1(r["draft_per_step"]))
                mac(f"Dr{name}{kn}Acc", f1(r["accepted_per_step"]))
                mac(f"Dr{name}{kn}First", f"{100 * r['first']:.0f}")
                mac(f"Dr{name}{kn}N", f"{r['n']:,}".replace(",", "{,}"))
    # the state-driven draft per decoding step
    for name, key in (("Cd", f"cd@{lo}"), ("CdNopeer", f"cd-nopeer@{lo}"), ("CdNoh", f"cd-noh@{lo}"),
                      ("CdNoctx", f"cd-noctx@{lo}"), ("Rule", "rule"), ("Lookup", "lookup")):
        for kind in ("verify", "plan", "all"):
            r = ev.get(f"{key}|{kind}|all")
            if r:
                kn = kind.capitalize()
                mac(f"Dr{name}{kn}Acc", f"{100 * r['acc_rate']:.0f}")
                mac(f"Dr{name}{kn}Len", f1(r["accepted_per_step"]))
                mac(f"Dr{name}{kn}Draft", f1(r["draft_per_step"]))
                mac(f"Dr{name}{kn}FirstTok", f"{100 * r['first']:.0f}")
    transfer_table(lo)
    if "calibration" in ev:  # reliability of the acceptance estimate, with and without the consumer state
        mac("DrEce", f"{100 * ev['ece']:.1f}")
        mac("DrEcePre", f"{100 * ev.get('ece_pre', 0):.1f}")
        cal = [c for c in ev["calibration"] if c["n"] > 0]
        calp = [c for c in ev.get("calibration_pre", []) if c["n"] > 0]
        pos = ev["by_position"]
        mac("DrPosOneAcc", f"{100 * pos[0]['accepted']:.0f}")
        mac("DrPosEightAcc", f"{100 * pos[-1]['accepted']:.0f}")
        mac("DrCalPositions", f"{sum(c['n'] for c in cal) / 1000:.0f}")
        for nm, cc in (("", cal), ("Pre", calp)):
            hi_ = [c for c in cc if c["bin"] >= 0.8]
            lo_ = [c for c in cc if c["bin"] < 0.2]
            if hi_:
                mac(f"DrCal{nm}HiPred", f"{100 * sum(c['predicted'] * c['n'] for c in hi_) / sum(c['n'] for c in hi_):.0f}")
                mac(f"DrCal{nm}HiAcc", f"{100 * sum(c['accepted'] * c['n'] for c in hi_) / sum(c['n'] for c in hi_):.0f}")
            if lo_:
                mac(f"DrCal{nm}LoPred", f"{100 * sum(c['predicted'] * c['n'] for c in lo_) / sum(c['n'] for c in lo_):.0f}")
                mac(f"DrCal{nm}LoAcc", f"{100 * sum(c['accepted'] * c['n'] for c in lo_) / sum(c['n'] for c in lo_):.0f}")
        try:
            plt = _mpl_style()
            # set beside the passes figure at the same scale (0.335 against 0.655 of the text width)
            fig, ax = plt.subplots(1, 2, figsize=(3.45, 2.1), sharey=True)
            ax[0].plot([0, 1], [0, 1], color=OI["gray"], lw=0.8, ls=":")
            mx = max(x["n"] for x in cal)
            ax[0].scatter([c["predicted"] for c in cal], [c["accepted"] for c in cal],
                          s=[6 + 32 * c["n"] / mx for c in cal], color=OI["blue"], label="with state", zorder=3)
            if calp:
                mxp = max(x["n"] for x in calp)
                ax[0].scatter([c["predicted"] for c in calp], [c["accepted"] for c in calp], marker="s",
                              s=[6 + 32 * c["n"] / mxp for c in calp], color=OI["orange"], alpha=0.88, label="first pass", zorder=3)
            ax[0].set_title("Reliability", fontsize=8.5)
            ax[0].set_xlabel("estimate $q_j$", fontsize=8.5)
            ax[0].set_ylabel("accepted", fontsize=8.5)
            ax[0].set_xlim(0, 1)
            ax[0].set_ylim(0, 1.03)
            ax[0].set_xticks([0, 0.5, 1])
            ax[0].legend(fontsize=7.5, frameon=False, loc="upper left", handletextpad=0.2, borderaxespad=0.15)
            ax[1].plot([p_["position"] for p_ in pos], [p_["accepted"] for p_ in pos], "o-", color=OI["blue"], ms=3.5, lw=1.25,
                       label="accepted")
            ax[1].plot([p_["position"] for p_ in pos], [p_["predicted"] for p_ in pos], "s--", color=OI["red"], ms=3.5, lw=1.25,
                       label="estimated")
            ax[1].set_title("By position", fontsize=8.5)
            ax[1].set_xlabel("draft position $j$", fontsize=8.5)
            ax[1].set_xticks([p_["position"] for p_ in pos][::2] or [1])
            ax[1].legend(fontsize=7.5, frameon=False, loc="lower left", borderaxespad=0.15)
            for a_ in ax:
                for sp in a_.spines.values():
                    sp.set_visible(True)
                    sp.set_linewidth(0.6)
                a_.tick_params(labelsize=8.0, width=0.6, length=2.5)
                a_.grid(True, alpha=0.28, lw=0.45, color="#888888")
            fig.tight_layout(pad=0.35, rect=(0, 0, 1, 0.86))
            fig.savefig(os.path.join(P, "figs", "calibration.pdf"))
            plt.close(fig)
        except Exception as e:  # the figure is optional
            WARN.append(f"calibration figure: {e!r}")


def transfer_table(pr):
    """The drafter trained on one dataset and applied to the other (train_drafter.py --transfer): held-out
    questions of the training dataset next to sampled calls of the dataset it never saw. Passes per key for the
    drafter in every pass, and for suffix lookup on the same calls."""
    rows = []
    for d, src, dst, key in (("drafter_h2m", "HotpotQA", "MuSiQue", "HM"), ("drafter_m2h", "MuSiQue", "HotpotQA", "MH")):
        pi, pt = os.path.join(R, d, "eval.json"), os.path.join(R, d, "eval_transfer.json")
        if not (os.path.exists(pi) and os.path.exists(pt)):
            WARN.append(f"missing {d}")
            continue
        for tag, E, to in (("In", json.load(open(pi))["eval"], f"{src}, held out"),
                           ("Tr", json.load(open(pt))["eval"], f"\\textbf{{{dst}}}")):
            cells = []
            for kind in ("verify", "plan"):
                s = E.get(f"sim0|cd@{pr}+cd@{pr}|{kind}")
                b = E.get(f"sim|lookup|{kind}")
                kn = kind.capitalize()
                if not (s and b):
                    cells += ["?", "?", "?"]
                    continue
                cells += [f2(s["passes"]), f"{100 * s['one_pass']:.0f}", f2(b["passes"])]
                mac(f"Tr{key}{tag}{kn}Pass", f2(s["passes"]))
                mac(f"Tr{key}{tag}{kn}One", f"{100 * s['one_pass']:.0f}")
                mac(f"Tr{key}{tag}{kn}Lookup", f2(b["passes"]))
                mac(f"Tr{key}{tag}{kn}Tokens", f1(s["tokens"]))
            mac(f"Tr{key}{tag}Ece", f"{100 * E.get('ece', 0):.1f}")
            rows.append(f"{src} & {to} & " + " & ".join(cells) + r" \\")
    if rows:
        with open(os.path.join(P, "tab_transfer.tex"), "w") as fo:
            fo.write("\n".join(rows) + "\n")


# ---------------------------------------------------------------- payload width regime
def wide_table():
    with open(os.path.join(P, "tab_wide.tex"), "w") as fo:
        for ds in DS:
            S = load(f"{V}_wide", ds)
            N = load(f"{V}_main", ds)
            k = DSK[ds]
            fo.write(f"\\multicolumn{{7}}{{l}}{{\\textit{{{DS[ds]}}}}}\\\\\n")
            for m, lab in WIDE:
                r, rn = S.get(m), N.get(m)
                if r is None or rn is None:
                    continue
                cells = [f2(rn["lat_mean"]), f2(r["lat_mean"]), f"{r['lat_p99']:.1f}", f"{r['em']:.3f}",
                         f2(gpu_s(r)), f"{r.get('gen_tokens', 0) / 1000:.0f}"]
                fo.write(f"{lab} & " + " & ".join(cells) + r" \\" + "\n")
                key = KEYN[m]
                mac(f"{k}W{key}Lat", f2(r["lat_mean"]))
                mac(f"{k}W{key}Pnn", f"{r['lat_p99']:.1f}")
                mac(f"{k}W{key}Gpu", f2(gpu_s(r)))
                mac(f"{k}W{key}EM", f"{r['em']:.3f}")
                mac(f"{k}W{key}GenTok", f"{r.get('gen_tokens', 0) / 1000:.0f}")
                mac(f"{k}W{key}ProvTok", f"{r.get('prov_tokens', 0) / 1000:.1f}")
                mac(f"{k}W{key}ProvEng", str(r.get("prov_engine", 0)))
                mac(f"{k}W{key}ProvReq", str(r.get("prov_requests", 0)))
                mac(f"{k}W{key}Lag", f2(r.get("prov_lag_p50", 0)))
                mac(f"{k}W{key}LatGrow", f"{100 * (r['lat_mean'] / rn['lat_mean'] - 1):.0f}")
                mac(f"{k}W{key}LatX", f"{r['lat_mean'] / rn['lat_mean']:.1f}")
            if U in S:
                pubs = [(S[m]["lat_mean"], m) for m in PUB if m in S]
                if pubs:
                    best = min(pubs)[1]
                    mac(f"{k}WBestPub", dict(SHORT)[best])
                    pair_macros(f"{V}_wide", ds, U, best, f"{k}WLatRedBest")
                for b in PUB:
                    if b in S:
                        pair_macros(f"{V}_wide", ds, U, b, f"{k}WLatRed{KEYN[b]}")
            fo.write("\\midrule\n")
            # the attribution of the drafter with the wide rationale (same layer, three drafting variants)
            A = load(f"{V}_wabl", ds) if os.path.exists(os.path.join(R, f"{V}_wabl", ds, "summary.json")) else {}
            for m in (SUF, NOPRE, U):
                if m in A:
                    mac(f"{k}WA{KEYN[m]}Lat", f2(A[m]["lat_mean"]))
                    mac(f"{k}WA{KEYN[m]}Pnn", f"{A[m]['lat_p99']:.1f}")
                    mac(f"{k}WA{KEYN[m]}Gpu", f2(gpu_s(A[m])))
                    mac(f"{k}WA{KEYN[m]}Pass", f1(passes(A[m])))
            for nm, (a, b) in {"Total": (U, SUF), "PreGain": (U, NOPRE), "DrafterGain": (NOPRE, SUF)}.items():
                if a in A and b in A:
                    st = pair_macros(f"{V}_wabl", ds, a, b, f"{k}WAbl{nm}")
                    if st is None:
                        mac(f"{k}WAbl{nm}", f"{pct(A[a]['lat_mean'], A[b]['lat_mean']):.0f}")
                    mac(f"{k}WAbl{nm}Gpu", f"{pct(gpu_s(A[a]), gpu_s(A[b])):.0f}")


# ---------------------------------------------------------------- load curve and capacity (narrow and wide payload)
def capacity(pts):
    pts = sorted(pts)
    if not pts or pts[0][2] > SLO:
        return None, False
    best = pts[0][0]
    for (r0, _, p0, _), (r1, _, p1, _) in zip(pts, pts[1:]):
        if p1 <= SLO:
            best = r1
        else:
            return r0 + (r1 - r0) * (SLO - p0) / (p1 - p0), False
    return best, True


def cap_str(c, sat, lowest=0.6):
    return f"$<${lowest}" if c is None else (f"$\\ge${c:.1f}" if sat else f"{c:.2f}")


def load_curve():
    data = {m: [] for m, _ in LOADM}
    for rate in RATES:
        tag = f"{V}_main" if rate == 1.2 else f"{V}_load_r{rate}"
        S = load(tag, "hotpotqa")
        for m, _ in LOADM:
            if m in S:
                data[m].append((rate, S[m]["lat_mean"], S[m]["lat_p99"], gpu_s(S[m])))
                mac(f"L{RATEK[rate]}{KEYN[m]}Lat", f2(S[m]["lat_mean"]))
                mac(f"L{RATEK[rate]}{KEYN[m]}Pnn", f"{S[m]['lat_p99']:.1f}")
        if U in S:
            pubs = [(S[m]["lat_mean"], m) for m in PUB if m in S]
            if pubs:
                best = min(pubs)[1]
                mac(f"L{RATEK[rate]}BestPub", dict(SHORT)[best])
                mac(f"L{RATEK[rate]}LatRedBest", f"{pct(S[U]['lat_mean'], S[best]['lat_mean']):.0f}")
                mac(f"L{RATEK[rate]}PnnRedBest", f"{pct(S[U]['lat_p99'], S[best]['lat_p99']):.0f}")
                mac(f"L{RATEK[rate]}SpeedupBest", f"{S[best]['lat_mean'] / S[U]['lat_mean']:.1f}")
    wdata = {m: [] for m, _ in WIDE}
    for rate in WRATES:
        tag = f"{V}_wide" if rate == 1.2 else f"{V}_wload_r{rate}"
        S = load(tag, "hotpotqa")
        for m in wdata:
            if m in S:
                wdata[m].append((rate, S[m]["lat_mean"], S[m]["lat_p99"], gpu_s(S[m])))
                mac(f"WL{RATEK[rate]}{KEYN[m]}Lat", f2(S[m]["lat_mean"]))
                mac(f"WL{RATEK[rate]}{KEYN[m]}Pnn", f"{S[m]['lat_p99']:.1f}")
    cap, wcap = {}, {}
    with open(os.path.join(P, "tab_load.tex"), "w") as fo:
        for m, lab in LOADM:
            pts = {x[0]: x for x in data[m]}
            if any(r not in pts for r in RATES):
                continue
            cells = [f"{pts[r][1]:.2f}/{pts[r][2]:.1f}" for r in RATES]
            c, sat = capacity(data[m])
            cap[m] = c
            if m == U:
                fo.write("\\midrule\n")
            fo.write(f"{lab} & " + " & ".join(cells) + f" & {cap_str(c, sat)}" + r" \\" + "\n")
            mac(f"Cap{KEYN[m]}", cap_str(c, sat))
    with open(os.path.join(P, "tab_wload.tex"), "w") as fo:
        for m, lab in WIDE:
            pts = {x[0]: x for x in wdata[m]}
            if any(r not in pts for r in WRATES):
                continue
            cells = [f"{pts[r][1]:.2f}/{pts[r][2]:.1f}" for r in WRATES]
            c, sat = capacity(wdata[m])
            wcap[m] = c
            if m == U:
                fo.write("\\midrule\n")
            fo.write(f"{lab} & " + " & ".join(cells) + f" & {cap_str(c, sat, WRATES[0])}" + r" \\" + "\n")
            mac(f"WCap{KEYN[m]}", cap_str(c, sat, WRATES[0]))
    pub = [cap[m] for m in PUB if cap.get(m)]
    if cap.get(U) and pub:
        mac("CapGainPub", f"{cap[U] / max(pub):.1f}")
    exact = [cap[m] for m in ("noshare", "pld", "suffix") if cap.get(m)]  # published systems that check every result
    if cap.get(U) and exact:
        mac("CapGainExact", f"{cap[U] / max(exact):.1f}")
        mac("CapBestExact", f"{max(exact):.2f}")
    wpub = [wcap[m] for m in PUB if wcap.get(m)]
    if wcap.get(U) and wpub:
        mac("WCapGainPub", f"{wcap[U] / max(wpub):.1f}")
    CURVES["ho"], CURVES["wide"] = data, wdata


CURVES = {}


def load_figure():
    """Latency against arrival rate, one panel per stream family: HotpotQA mean and P99, MuSiQue P99 and HotpotQA
    P99 with the wide rationale. One legend above the panels."""
    try:
        plt = _mpl_style()
        fig, ax = plt.subplots(1, 4, figsize=(8.4, 2.1))
        marks = "osv^D*"
        cols = [OI["gray"], OI["purple"], OI["orange"], OI["green"], OI["sky"], OI["blue"]]
        lss = ["--", ":", "-.", "--", "-.", "-"]
        panels = [("ho", 1, "HotpotQA, mean (s)"), ("ho", 2, "HotpotQA, P99 (s)"),
                  ("mu", 2, "MuSiQue, P99 (s)"), ("wide", 2, "HotpotQA wide, P99 (s)")]
        for a, (src, col, title) in zip(ax, panels):
            for (m, lab), mk, cl, ls in zip(LOADM, marks, cols, lss):
                d = sorted(CURVES.get(src, {}).get(m, []))
                if len(d) < 2:
                    continue
                lab2 = lab.replace("\\textbf{", "").replace("}", "")
                a.plot([x[0] for x in d], [x[col] for x in d], marker=mk, color=cl, ls=ls,
                       label=lab2, lw=1.35 if m == U else 1.15, ms=3.8 if m == U else 3.3)
                a.set_xticks([x[0] for x in d])
            if col == 2:
                a.axhline(SLO, color="#666666", ls=":", lw=0.8)
            a.set_title(title, fontsize=8.5)
            a.set_xlabel("arrival rate (queries/s)", fontsize=8.5)
            a.set_yscale("log")
            a.grid(True, alpha=0.28, lw=0.45, color="#888888")
            a.tick_params(labelsize=8.0, width=0.6, length=2.5)
            for sp in a.spines.values():
                sp.set_visible(True)
                sp.set_linewidth(0.6)
        ax[0].set_ylabel("query latency (s)", fontsize=8.5)
        fig.tight_layout(pad=0.35, rect=(0, 0, 1, 0.86))
        hs, ls_ = ax[0].get_legend_handles_labels()
        if hs:
            fig.legend(hs, ls_, fontsize=8.0, frameon=False, loc="upper center", ncol=len(hs),
                       bbox_to_anchor=(0.5, 1.02), handlelength=2.2, columnspacing=1.6)
        fig.savefig(os.path.join(P, "figs", "load.pdf"))
        plt.close(fig)
    except Exception as e:
        WARN.append(f"figure: {e}")


# ---------------------------------------------------------------- the verification price at work
PRICEM = [(SUF, "(A) layer with suffix lookup"), (RULE, "(B) A, decisions by the peer rule"),
          (NOPRE, "(C) record drafter after the first pass"), (U, r"\textbf{(D) Unison}")]
PRATES = [1.2, 2.4, 3.0]


def price_table():
    """Drafting variants on the same layer under a light, a filling and a saturated engine (HotpotQA). Per rate:
    the mean number of requests in a step, the engine's mean iteration time and the price of a draft position that
    the drafter measured; cells: mean latency, P99, GPU s/query, passes per verification call and per plan, tokens
    drafted per served key call."""
    with open(os.path.join(P, "tab_price.tex"), "w") as fo:
        for rate in PRATES:
            tag = f"{V}_abl" if rate == 1.2 else f"{V}_abload_r{rate}"
            S = load(tag, "hotpotqa")
            if not S:
                continue
            rk = RATEK[rate]
            u = S.get(U)
            head = f"{rate} queries/s"
            if u and u.get("nd_calls"):
                mac(f"Pr{rk}B", f1(u.get("x_requests", 0)))
                mac(f"Pr{rk}Price", f2(u.get("nd_price", 0)))
                mac(f"Pr{rk}StepPrice", f2(u.get("nd_sprice", 0)))
                mac(f"Pr{rk}IterMs", f"{u.get('nd_iter_ms', 0):.0f}")
                head += f": $\\bar B={u.get('x_requests', 0):.1f}$, iteration {u.get('nd_iter_ms', 0):.0f}\\,ms, $\\vartheta={u.get('nd_price', 0):.2f}$"
            fo.write(f"\\multicolumn{{7}}{{l}}{{\\textit{{{head}}}}}\\\\\n")
            for m, lab in PRICEM:
                r = S.get(m)
                if r is None:
                    continue
                calls = max(1, r.get("key_engine_calls", 0))
                dpc = r.get("acc_drafted", 0) / calls
                cells = [f2(r["lat_mean"]), f"{r['lat_p99']:.1f}", f2(gpu_s(r)), f1(passes(r)), f1(passes(r, "plan")),
                         f1(dpc)]
                fo.write(f"{lab} & " + " & ".join(cells) + r" \\" + "\n")
                key = KEYN[m]
                mac(f"Pr{rk}{key}Lat", f2(r["lat_mean"]))
                mac(f"Pr{rk}{key}Pnn", f"{r['lat_p99']:.1f}")
                mac(f"Pr{rk}{key}Gpu", f2(gpu_s(r)))
                mac(f"Pr{rk}{key}Pass", f1(passes(r)))
                mac(f"Pr{rk}{key}PlanPass", f1(passes(r, "plan")))
                mac(f"Pr{rk}{key}One", f"{one_pass(r):.0f}")
                mac(f"Pr{rk}{key}Dpc", f1(dpc))
            for a, b, nm in ((U, SUF, "UvsSuf"), (RULE, SUF, "RuleVsSuf"), (U, RULE, "UvsRule"), (U, NOPRE, "UvsNopre"),
                             (NOPRE, SUF, "NopreVsSuf")):
                if a in S and b in S:
                    st = pair_macros(tag, "hotpotqa", a, b, f"Pr{rk}{nm}")
                    if st is None:
                        mac(f"Pr{rk}{nm}", f"{pct(S[a]['lat_mean'], S[b]['lat_mean']):.0f}")
                    mac(f"Pr{rk}{nm}Pnn", f"{pct(S[a]['lat_p99'], S[b]['lat_p99']):.0f}")
                    mac(f"Pr{rk}{nm}Gpu", f"{pct(gpu_s(S[a]), gpu_s(S[b])):.0f}")
                    mac(f"Pr{rk}{nm}Abs", f"{abs(pct(S[a]['lat_mean'], S[b]['lat_mean'])):.0f}")
            fo.write("\\midrule\n")


# ---------------------------------------------------------------- MuSiQue load
MRATES = [0.3, 0.6, 0.9, 1.2]
MRATEK = {0.3: "ZeroThree", 0.6: "Main", 0.9: "ZeroNine", 1.2: "OneTwo"}


def load_table_musique():
    data = {m: [] for m, _ in LOADM}
    for rate in MRATES:
        tag = f"{V}_main" if rate == 0.6 else f"{V}_load_r{rate}"
        S = load(tag, "musique")
        for m, _ in LOADM:
            if m in S:
                data[m].append((rate, S[m]["lat_mean"], S[m]["lat_p99"], gpu_s(S[m])))
    mcap = {}
    with open(os.path.join(P, "tab_loadm.tex"), "w") as fo:
        for m, lab in LOADM:
            pts = {x[0]: x for x in data[m]}
            if any(r not in pts for r in MRATES):
                continue
            cells = [f"{pts[r][1]:.2f}/{pts[r][2]:.1f}" for r in MRATES]
            c, sat = capacity(data[m])
            mcap[m] = c
            if m == U:
                fo.write("\\midrule\n")
            fo.write(f"{lab} & " + " & ".join(cells) + f" & {cap_str(c, sat, MRATES[0])}" + r" \\" + "\n")
            mac(f"MCap{KEYN[m]}", cap_str(c, sat, MRATES[0]))
            for r in MRATES:
                mac(f"ML{MRATEK[r]}{KEYN[m]}Lat", f2(pts[r][1]))
                mac(f"ML{MRATEK[r]}{KEYN[m]}Pnn", f"{pts[r][2]:.1f}")
    pub = [mcap[m] for m in PUB if mcap.get(m)]
    if mcap.get(U) and pub:
        mac("MCapGainPub", f"{mcap[U] / max(pub):.1f}")
    CURVES["mu"] = data
    for rate in MRATES:
        if rate == 0.6:
            continue
        S = load(f"{V}_load_r{rate}", "musique")
        if U in S:
            pubs = [(S[m]["lat_mean"], m) for m in PUB if m in S]
            if pubs:
                best = min(pubs)[1]
                mac(f"ML{MRATEK[rate]}BestPub", dict(SHORT)[best])
                mac(f"ML{MRATEK[rate]}LatRedBest", f"{pct(S[U]['lat_mean'], S[best]['lat_mean']):.0f}")
                mac(f"ML{MRATEK[rate]}PnnRedBest", f"{pct(S[U]['lat_p99'], S[best]['lat_p99']):.0f}")


# ---------------------------------------------------------------- version-safety stress
def edit_table():
    S = load(f"{V}_edit", "hotpotqa")
    with open(os.path.join(P, "tab_edit.tex"), "w") as fo:
        for m, lab in EDITM:
            r = S.get(m)
            if r is None:
                continue
            n, e = nonexact(r)
            fo.write(f"{lab} & {n} & {r.get('stale_reuse', 0)} & {r.get('stale_decisive', 0)} & "
                     f"{r.get('stale_decisive_err', 0)} & {e} \\\\\n")
            key = KEYN[m]
            mac(f"Ed{key}Xv", str(r.get("stale_reuse", 0)))
            mac(f"Ed{key}Dec", str(r.get("stale_decisive", 0)))
            mac(f"Ed{key}DecErr", str(r.get("stale_decisive_err", 0)))
            mac(f"Ed{key}EM", f"{r['em']:.3f}")
            mac(f"Ed{key}Err", str(e))
            mac(f"Ed{key}ErrPct", f"{100 * e / n if n else 0:.1f}")
            mac(f"Ed{key}Reuse", str(n))
            mac(f"Ed{key}Lat", f2(r["lat_mean"]))
        if S:
            mac("EdUpdates", str(next(iter(S.values())).get("n_updates", 0)))


# ---------------------------------------------------------------- redundancy sweep
def zipf_table():
    with open(os.path.join(P, "tab_zipf.tex"), "w") as fo:
        for z, tag in (("0", f"{V}_zipf_0"), ("0.8", f"{V}_main"), ("1.2", f"{V}_zipf_1.2")):
            S = load(tag, "hotpotqa")
            zk = ZIPFK[z]
            ns = S.get("suffix") or S.get("noshare")
            red = f"{100 * ns['redund_identical']:.0f}" if ns else "?"
            fo.write(f"\\multicolumn{{7}}{{l}}{{\\textit{{$s={z}$: {red}\\% of the calls identical to another}}}}\\\\\n")
            mac(f"Z{zk}RedIdent", red)
            for m, lab in ZIPFM:
                r = S.get(m)
                if r is None:
                    continue
                n, e = nonexact(r)
                cells = [f2(r["lat_mean"]), f"{r['lat_p99']:.1f}", f2(gpu_s(r)), f"{ident_pct(r):.0f}",
                         f1(passes_all(r)), f"{100 * r.get('dec_err_total', 0.0):.2f}"]
                fo.write(f"{lab} & " + " & ".join(cells) + r" \\" + "\n")
                key = KEYN[m]
                mac(f"Z{zk}{key}Lat", f2(r["lat_mean"]))
                mac(f"Z{zk}{key}Gpu", f2(gpu_s(r)))
                mac(f"Z{zk}{key}Ident", f"{ident_pct(r):.0f}")
                mac(f"Z{zk}{key}Verif", f"{verified_pct(r):.0f}")
                mac(f"Z{zk}{key}Pass", f1(passes(r)))
                mac(f"Z{zk}{key}PassAll", f1(passes_all(r)))
                mac(f"Z{zk}{key}One", f"{one_pass(r):.0f}")
                mac(f"Z{zk}{key}Err", f"{100 * e / n:.1f}" if n else "0")
            for m in PUB:
                if U in S and m in S:
                    mac(f"Z{zk}LatRed{KEYN[m]}", f"{pct(S[U]['lat_mean'], S[m]['lat_mean']):.0f}")
            fo.write("\\midrule\n")


# ---------------------------------------------------------------- latency by query class
def class_table():
    """Queries split by what preceded them in the stream: first occurrence of the question, a verbatim repeat
    (identity sharing applies to the whole trajectory), or a paraphrased repeat (no prompt of the trajectory is
    identical to an earlier one beyond the passage prefix)."""
    with open(os.path.join(P, "tab_class.tex"), "w") as fo:
        for ds in DS:
            k = DSK[ds]
            ref = qs(f"{V}_main", ds, U)
            if not ref:
                fo.write(f"\\multicolumn{{4}}{{l}}{{\\textit{{{DS[ds]}}}}}\\\\\n" + "\n".join(f"{lab} & ? & ? & ? \\\\" for _, lab in SHORT) + "\n")
                continue
            order = sorted(ref.values(), key=lambda x: x["t_arrive"])
            seen_text, seen_qid, cls = set(), set(), {}
            for x in order:
                cls[x["qi"]] = "first" if x["qid"] not in seen_qid else ("exact" if x["question"] in seen_text else "para")
                seen_text.add(x["question"])
                seen_qid.add(x["qid"])
            groups = {c: [i for i, cc in cls.items() if cc == c] for c in ("first", "exact", "para")}
            for c, ids in groups.items():
                mac(f"{k}ClsN{c.capitalize()}", str(len(ids)))
            fo.write(f"\\textit{{{DS[ds]}, queries}} & \\textit{{{len(groups['first'])}}} & "
                     f"\\textit{{{len(groups['exact'])}}} & \\textit{{{len(groups['para'])}}}\\\\\n")
            for m, lab in SHORT:
                q = qs(f"{V}_main", ds, m)
                if not q:
                    continue
                cells = []
                for c in ("first", "exact", "para"):
                    v = float(np.mean([q[i]["latency"] for i in groups[c] if i in q])) if groups[c] else float("nan")
                    cells.append(f2(v))
                    mac(f"{k}Cls{c.capitalize()}{KEYN[m]}", f2(v))
                fo.write(f"{lab} & " + " & ".join(cells) + r" \\" + "\n")
            fo.write("\\midrule\n")


# ---------------------------------------------------------------- audit quality
def audit_table():
    gmax = []
    with open(os.path.join(P, "tab_audit.tex"), "w") as fo:
        for ds in DS:
            p = os.path.join(R, f"{V}_abl", ds, f"audit_{U}_vs_{SBD}.json")
            pre = DSK[ds] + "Aud"
            if not os.path.exists(p):
                WARN.append(f"missing audit {V}_abl/{ds}")
                fo.write(f"\\multicolumn{{4}}{{l}}{{\\textit{{{DS[ds]}}}}}\\\\\n? & ? & ? & ? \\\\\n")
                continue
            a = json.load(open(p))
            fo.write(r"\multicolumn{4}{l}{\textit{" + DS[ds] + f", {a['n_pairs']} pairs" + r"}}\\" + "\n")

            def r1(x, sign=False):
                """One decimal, halves rounded up (the rates are multiples of 1/n, so the two columns and their
                difference must round the same way); a signed zero prints as 0.0."""
                v = Decimal(str(round(x, 6))).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
                return "0.0" if v == 0 else (f"{v:+}" if sign else f"{v}")
            for q, lab in (("grounded", "Grounded"), ("responsive", "Responsive"), ("accept", "Accept"),
                           ("cited_contains_answer", "Cited sentence contains answer")):
                d, lo, hi = (100 * x for x in a[f"{q}_diff"])
                u, b = 100 * a[q + "_a"], 100 * a[q + "_b"]
                fo.write(f"{lab} & {r1(u)} & {r1(b)} & {r1(d, True)} [{r1(lo, True)}, {r1(hi, True)}]" + r"\\" + "\n")
                mac(f"{pre}{q.replace('_', '')}U", r1(u))
                mac(f"{pre}{q.replace('_', '')}B", r1(b))
                mac(f"{pre}{q.replace('_', '')}D", r1(d, True))
                mac(f"{pre}{q.replace('_', '')}DCI", f"[{r1(lo, True)},{r1(hi, True)}]")
                if q == "grounded":
                    gmax.append(abs(d))
                    mac("AudGroundedMax", r1(max(gmax)))
                    if lo > 0 or hi < 0:
                        WARN.append(f"AUDIT CLAIM: the grounded interval excludes zero on {ds}")
            mac(f"{pre}N", str(a["n_pairs"]))
            mac(f"{pre}Unparsed", f"{100 * max(a.get('unparsed_a', 0), a.get('unparsed_b', 0)):.1f}")
            mac(f"{pre}LenU", f"{a.get('prov_len_a', 0):.0f}")
            mac(f"{pre}LenB", f"{a.get('prov_len_b', 0):.0f}")
            raw = p[:-len(".json")] + ".raw.jsonl"  # the judge's replies, alternating Unison / continuation per pair
            if os.path.exists(raw):
                L = [json.loads(x) for x in open(raw, encoding="utf-8")]
                pairs = list(zip(L[0::2], L[1::2]))
                if len(pairs) != a["n_pairs"] or not all(x["side"] == "a" and y["side"] == "b" for x, y in pairs):
                    WARN.append(f"audit raw replies of {ds} do not pair up")
                mac(f"{pre}Same", str(sum(x["reply"].strip() == y["reply"].strip() for x, y in pairs)))
            else:
                WARN.append(f"missing audit raw replies {V}_abl/{ds}")
            fo.write("\\midrule\n")


for fn in (cost_structure, main_table, em_spread, seeds_table, pass_table, time_table, abl_table, drafter_table, wide_table, load_curve,
           price_table, load_table_musique, load_figure, edit_table, zipf_table, class_table, audit_table):
    try:
        fn()
    except Exception as e:  # one missing or malformed run must not hide the rest
        import traceback
        WARN.append(f"{fn.__name__} failed: {e!r}")
        traceback.print_exc()
def merge_load():
    """tab_loads: the two load tables side by side (HotpotQA: five rates and capacity, MuSiQue: three and capacity)."""
    def rows(name, n):
        out = {}
        p = os.path.join(P, name)
        if os.path.exists(p):
            for line in open(p):
                if "&" in line:
                    lab, rest = line.split("&", 1)
                    out[lab.strip()] = [c.strip() for c in rest.strip().rstrip("\\").strip().split("&")]
        return out, n
    (ho, nh), (mu, nm) = rows("tab_load.tex", 6), rows("tab_loadm.tex", len(MRATES) + 1)
    with open(os.path.join(P, "tab_loads.tex"), "w") as fo:
        for m, lab in LOADM:
            if lab not in ho and lab not in mu:
                continue
            if m == U:
                fo.write("\\midrule\n")
            fo.write(f"{lab} & " + " & ".join(ho.get(lab, ["?"] * nh) + mu.get(lab, ["?"] * nm)) + r" \\" + "\n")


try:
    merge_load()
except Exception as e:
    WARN.append(f"merge_load failed: {e!r}")
# claims of the text that the data must carry (Section 6.1 isolation, Section 6.9 wide capacity)
for (tag, ds), (sm, mib) in sorted(ISO.items()):
    if sm >= 0.05 or mib > 500:  # the text states 0.0% for every reported run
        WARN.append(f"ISOLATION CLAIM: {tag}/{ds} co-tenant SM {sm:.1f}%, co-tenant memory {mib} MiB")
if "$<$" not in M.get("WCapSuffix", "$<$"):
    WARN.append(f"WIDE CAPACITY CLAIM: SuffixDecoding holds the P99 at some rate ({M['WCapSuffix']})")


def _ci(name):
    s = M.get(name, "")
    try:
        lo, hi = s.strip("[]").split(",")
        return float(lo), float(hi)
    except ValueError:
        return None


for r in ("PrTwoFour", "PrThree"):  # Section "Load": statements made in words about the two higher rates
    ci = _ci(r + "UvsNopreDecCI")
    if ci is None or not ci[0] <= 0 <= ci[1]:
        WARN.append(f"LOAD CLAIM: (D) against (C) is not zero within its interval at {r}: {M.get(r + 'UvsNopreDecCI')}")
    ci = _ci(r + "RuleVsSufCI")
    if ci is None or ci[1] >= 0:
        WARN.append(f"LOAD CLAIM: the rule (B) is not slower than suffix lookup (A) at {r}: {M.get(r + 'RuleVsSufCI')}")
if len({M.get(r + "Price") for r in ("PrMain", "PrTwoFour", "PrThree")}) != 1:
    WARN.append("LOAD CLAIM: the price differs between the three rates")
def _num(name):
    try:
        return float(M.get(name, ""))
    except ValueError:
        return None


_caps = {m: _num("Cap" + KEYN[m]) for m in ("noshare", "pld", "suffix", "gptcache", "vcache-official")}
if _caps["gptcache"] is None or any(v is not None and v > _caps["gptcache"] for v in _caps.values()):
    WARN.append(f"LOAD CLAIM: GPTCache is not the published system with the highest capacity: {_caps}")
for nm in ("MCapUnison", "MCapGptcache"):
    if "ge" not in M.get(nm, ""):
        WARN.append(f"LOAD CLAIM: {nm} does not hold the P99 at the highest MuSiQue rate ({M.get(nm)})")
_excl = {(k, q): _ci(f"{k}Aud{q}DCI") for k in ("Ho", "Mu") for q in ("grounded", "responsive", "accept")}
_excl = sorted(kq for kq, ci in _excl.items() if ci and not ci[0] <= 0 <= ci[1])
if _excl != [("Ho", "accept"), ("Mu", "responsive")] or _ci("HoAudacceptDCI")[1] >= 0 or _ci("MuAudresponsiveDCI")[0] <= 0:
    WARN.append(f"AUDIT CLAIM: intervals that exclude zero are {_excl}")
for k in ("Ho", "Mu"):  # Section "Wide rationales": Unison at vLLM's exact match
    if M.get(k + "WUnisonEM") != M.get(k + "WVllmEM"):
        WARN.append(f"WIDE CLAIM: exact match of Unison {M.get(k + 'WUnisonEM')} and vLLM {M.get(k + 'WVllmEM')} differ")
    ne, tie = M.get(k + "UCertErr"), M.get(k + "NDpct")  # "no more than the engine differs from itself"
    n = M.get(k + "UCertN")
    if ne is None or tie is None or n is None or 100 * float(ne) / max(1.0, float(n)) > float(tie):
        WARN.append(f"EXACTNESS CLAIM: {k} non-identity differences {ne} of {n} against a tie rate of {tie}%")
def minus(s):
    """A sign in front of a number is typeset as a minus in text and in math; hyphens inside names stay."""
    return re.sub(r"(?<![\w}\\])-(?=\d)", r"\\ensuremath{-}", s)


for name in os.listdir(P):  # block separators are written after each block; the last one would meet \bottomrule
    if name.startswith("tab_") and name.endswith(".tex"):
        path = os.path.join(P, name)
        txt = open(path).read()
        while txt.rstrip().endswith("\\midrule"):
            txt = txt.rstrip()[: -len("\\midrule")]
        with open(path, "w") as fo:
            fo.write(minus(txt.rstrip()) + "\n")
with open(os.path.join(P, "numbers_auto.tex"), "w") as fo:
    fo.write("% generated by code/scripts/analyze_v17.py -- do not edit\n")
    for k, v in sorted(M.items()):
        fo.write(f"\\providecommand{{\\{k}}}{{{minus(v)}}}\n")
print(f"{len(M)} macros")
for w in dict.fromkeys(WARN):
    print("WARNING", w)
