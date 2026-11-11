"""Tables, figures and number macros for paper/unison15 from <results>/v15_*/<ds>/{summary.json,*.queries.json,*.steps.npy}.

usage: python analyze_v15.py <results_dir> <paper_dir>
Every number in the paper's evaluation comes from here. Missing runs print '?' (never silently '-').
Statistics: one stream per setting; paired bootstrap over queries for latency ratios and EM differences, exact
McNemar test for EM, Clopper-Pearson upper bounds for error rates; iteration-time regression for the cost model.
"""
import glob
import json
import os
import sys

import numpy as np
from scipy.stats import beta, binomtest

R, P = sys.argv[1], sys.argv[2]
os.makedirs(P, exist_ok=True)
DS = {"hotpotqa": "HotpotQA", "musique": "MuSiQue"}
DSK = {"hotpotqa": "Ho", "musique": "Mu"}
U, SBD = "u15", "u15-sbd"
# Comparison systems are published baselines from 2023--2025, run from their own code or from the
# engine's implementation of that method. Component variants of Unison are not baselines.
PUB = ["noshare", "pld", "suffix", "gptcache", "vcache-official"]
MAIN = [("noshare", r"vLLM + APC~\cite{kwon2023vllm}"), ("pld", r"PLD~\cite{saxena2023pld}"),
        ("suffix", r"SuffixDecoding~\cite{oliaro2025suffix}"), ("gptcache", r"GPTCache~\cite{bang2023gptcache}"),
        ("vcache-official", r"vCache~\cite{schroeder2025vcache}"), (U, r"\textbf{Unison}")]
ABL = [("u15-lazy", "Key only, no payload (lower bound)"),
       (SBD, "(A) identity sharing, per-call payload"),
       ("u15-coop", "(B) + verified key sharing"),
       ("u15-keyed", "(C) + canonical payload"),
       (U, r"\textbf{(D) Unison} = (B) + (C)"),
       ("u15-noid", "(D) without identity sharing"),
       ("u15-nodf", "(D) call-level boundaries only"),
       ("u15-nolive", "(D) finished peers only"),
       ("u15-noprio", "(D) payload in the foreground")]
WIDE = [("noshare", "vLLM + APC"), ("pld", "PLD"), ("suffix", "SuffixDecoding"),
        ("gptcache", "GPTCache"), ("vcache-official", "vCache"), (U, "\\textbf{Unison}")]
LOADM = [("noshare", "vLLM + APC"), ("pld", "PLD"), ("suffix", "SuffixDecoding"),
         ("gptcache", "GPTCache"), ("vcache-official", "vCache"), (U, "\\textbf{Unison}")]
RATES = [0.6, 1.2, 1.8, 2.4, 3.0]
WRATES = [1.2, 1.8, 2.4, 3.0]
RATEK = {1.2: "Main", 1.8: "OneEight", 2.4: "TwoFour", 3.0: "Three"}  # macro names must be alphabetic
ZIPFK = {"0": "Zero", "0.8": "Main", "1.2": "Heavy"}
SLO = 10.0
EDITM = [("noshare", "vLLM + APC"), ("pld", "PLD"), ("suffix", "SuffixDecoding"),
         ("gptcache", "GPTCache"), ("vcache-official", "vCache"), (U, "\\textbf{Unison}")]
ZIPFM = [("noshare", "vLLM + APC"), ("pld", "PLD"), ("suffix", "SuffixDecoding"),
         ("gptcache", "GPTCache"), ("vcache-official", "vCache"), (U, "\\textbf{Unison}")]
KEYN = {"noshare": "Vllm", "pld": "Pld", "suffix": "Suffix", "gptcache": "Gptcache", "vcache-official": "Vcache",
        "lacache": "Lacache", SBD: "Sbd", U: "Unison", "noshare@cot": "VllmCot", "u15-lazy": "Lazy",
        "u15-coop": "Coop", "u15-keyed": "Keyed", "u15-noid": "Noid", "u15-nodf": "Nodf", "u15-nolive": "Nolive",
        "u15-noprio": "Noprio"}
M = {}
WARN = []


def mac(name, val):
    M["".join(ch for ch in name if ch.isalpha())] = val


def f2(x):
    return f"{x:.2f}"


def load(tag, ds):
    alt = tag.replace("v15_", "v16_", 1) if str(tag).startswith("v15_") else None
    if alt and os.path.exists(os.path.join(R, alt, ds, "summary.json")):
        tag = alt
    p = os.path.join(R, tag, ds, "summary.json")
    if not os.path.exists(p):
        WARN.append(f"missing {tag}/{ds}")
        return {}
    out = {r["method"]: r for r in json.load(open(p))}
    lo = min((r.get("gpu_others_mib", 0) for r in out.values() if r.get("gpu_others_mib", -1) >= 0), default=0)
    timed = not (tag.endswith("_edit") or tag.endswith("_smoke"))
    for m, r in out.items():
        if timed and r.get("gpu_others_sm_busy", 0) > 0.6:
            WARN.append(f"CONTAMINATED? {tag}/{ds}/{m}: co-tenant SM >= 40% in {100 * r['gpu_others_sm_busy']:.0f}% of samples -> rerun")
        elif timed and r.get("gpu_others_sm_mean", 0) > 5:
            WARN.append(f"NOISY {tag}/{ds}/{m}: mean co-tenant SM share {r['gpu_others_sm_mean']:.0f}% (polish rerun advisable)")
    if timed and out:  # isolation as measured, per comparison group (macros <Ds><Group>CoSm, <Ds><Group>BaseMs, <Ds><Group>CoMib)
        sm = [r.get("gpu_others_sm_mean", -1) for r in out.values() if r.get("gpu_others_sm_mean", -1) >= 0]
        grp = tag.replace("v15_", "").replace("_", "").replace(".", "")
        grp = "".join(ch for ch in grp if ch.isalpha()) or "x"
        k = DSK.get(ds, "X") + grp.capitalize()
        mac(f"{k}CoSm", f"{np.mean(sm):.0f}" if sm else "0")
        base = [r["cost"]["a"] * 1000 for r in out.values() if isinstance(r.get("cost"), dict)]
        mac(f"{k}BaseMs", f"{np.mean(base):.1f}" if base else "?")
        mib = [r.get("gpu_others_mib", 0) for r in out.values()]
        mac(f"{k}CoMib", f"{max(mib) / 1024:.1f}" if mib else "0")
        if any(m > 500 for m in mib):
            WARN.append(f"SHARED CARD {tag}/{ds}: other processes held up to {max(mib)} MiB; base iteration "
                        f"{np.mean(base):.1f} ms (compare within the table only)")
        if r.get("n_df_mismatch", 0):
            WARN.append(f"DATAFLOW MISMATCH {tag}/{ds}/{m}: {r['n_df_mismatch']}")
    kv = {r.get("kv_blocks") for r in out.values()}
    if timed and len(kv) > 1:
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
    return dict(red=100 * (1 - la.mean() / lb.mean()), red_lo=np.percentile(red, 2.5), red_hi=np.percentile(red, 97.5),
                dem=100 * (ea.mean() - eb.mean()), dem_lo=np.percentile(dem, 2.5), dem_hi=np.percentile(dem, 97.5),
                p=p, n=len(ids))


def pair_macros(tag, ds, a, b, prefix):
    st = paired(tag, ds, a, b)
    if st:
        mac(f"{prefix}", f"{st['red']:.0f}")
        mac(f"{prefix}CI", f"[{st['red_lo']:.0f},{st['red_hi']:.0f}]")
        mac(f"{prefix}EMd", f"{st['dem']:+.1f}")
        mac(f"{prefix}EMdCI", f"[{st['dem_lo']:+.1f},{st['dem_hi']:+.1f}]")
        mac(f"{prefix}EMp", f"{st['p']:.2f}")
    return st


# ---------------------------------------------------------------- cost model (Proposition 1)
def cost_model():
    """Iteration time = a + c*prompt tokens + g*generated tokens (+ d*draft tokens).

    Two stages, because generated tokens and running requests are collinear in decode-only iterations: (1) the
    runs without speculation (vLLM, GPTCache, vCache) give a, c and g; (2) the speculating runs (SuffixDecoding,
    LaCache, identity-only, Unison) give d from the residual against (1). Pooled over both datasets' main runs."""
    def load_steps(methods):
        X, y = [], []
        for ds in DS:
            for m in methods:
                p = os.path.join(R, "v15_main", ds, f"{m}.steps.npy")
                if not os.path.exists(p):
                    continue
                st = np.load(p)
                dt = np.diff(st[:, 0])
                ok = (dt > 0) & (dt < 1.0)
                S = st[1:][ok]
                X.append(S[:, [1, 3, 4, 5]])  # running, generated, prompt, draft
                y.append(dt[ok])
        return (np.concatenate(X), np.concatenate(y)) if X else (None, None)

    X0, y0 = load_steps(["noshare", "gptcache", "vcache-official", "noshare@cot"])
    X1, y1 = load_steps(["suffix", "lacache", SBD, U])
    if X0 is None or X1 is None:
        WARN.append("no iteration logs for the cost model")
        return
    A0 = np.c_[np.ones(len(y0)), X0[:, 2], X0[:, 1]]  # 1, prompt, generated
    (a, c, g), *_ = np.linalg.lstsq(A0, y0, rcond=None)
    resid = y1 - (a + c * X1[:, 2] + g * X1[:, 1])
    d = float(resid @ X1[:, 3] / max(1e-9, X1[:, 3] @ X1[:, 3]))
    mac("CostBase", f"{a * 1000:.1f}")
    mac("CostPrompt", f"{c * 1e6:.0f}")
    mac("CostGen", f"{g * 1e6:.0f}")
    mac("CostDraft", f"{d * 1e6:.0f}")
    mac("CostRatio", f"{d / max(g, 1e-9):.1f}")
    mac("CostIters", str(len(y0) + len(y1)))
    mac("CostRun", "0")  # the running-request term is absorbed by the generated-token term (one token per request)
    per = {}
    for ds in DS:
        p = os.path.join(R, "v15_main", ds, f"{U}.steps.npy")
        if os.path.exists(p):
            st = np.load(p)
            dt = np.diff(st[:, 0])
            ok = (dt > 0) & (dt < 1.0)
            per[ds] = (len(st), float(dt[ok].mean()), float(st[1:][ok][:, 1].mean()))
    # share of iterations and mean iteration time of the Unison main run
    for ds in DS:
        if ds in per:
            n, dtm, run = per[ds]
            mac(f"{DSK[ds]}Iters", str(n))
            mac(f"{DSK[ds]}IterMs", f"{dtm * 1000:.1f}")
            mac(f"{DSK[ds]}Running", f"{run:.1f}")


# ---------------------------------------------------------------- main table
def main_table():
    rows = {m: [] for m, _ in MAIN}
    for ds in DS:
        S = load("v15_main", ds)
        k = DSK[ds]
        floor = None
        for m, _ in MAIN:
            r = S.get(m)
            if r is None:
                continue
            n, e = nonexact(r)
            err = f"{100 * e / n:.1f}" if n else "0.0"
            rows[m].append(f"{r['lat_mean']:.2f} & {r['lat_p99']:.1f} & {r['em']:.3f} & {gpu_s(r):.2f} & {err}")
            if r.get("nd_prompts"):
                floor = (r["nd_decision_diff"], r["nd_prompts"])
            key = KEYN[m]
            mac(f"{k}Lat{key}", f2(r["lat_mean"]))
            mac(f"{k}Pnn{key}", f"{r['lat_p99']:.1f}")
            mac(f"{k}EM{key}", f"{r['em']:.3f}")
            mac(f"{k}Gpu{key}", f2(gpu_s(r)))
            mac(f"{k}Reuse{key}", str(n))
            mac(f"{k}ReuseErr{key}", str(e))
            mac(f"{k}ReuseErrPct{key}", err)
            mac(f"{k}Ident{key}", f"{ident_pct(r):.0f}")
            mac(f"{k}Verif{key}", f"{verified_pct(r):.0f}")
            mac(f"{k}GenTok{key}", f"{r.get('gen_tokens', 0) / 1000:.0f}")
        ns = S.get("noshare")
        if ns and "redund_passage60" in ns:
            mac(f"{k}RedPass", f"{100 * ns['redund_passage60']:.0f}")
            mac(f"{k}RedIdent", f"{100 * ns['redund_identical']:.0f}")
        if S:  # isolation as measured: co-tenant SM share and the fixed KV capacity of the comparison
            sm = [r.get("gpu_others_sm_mean", -1) for r in S.values() if r.get("gpu_others_sm_mean", -1) >= 0]
            mac(f"{k}CoSm", f"{np.mean(sm):.0f}" if sm else "0")
            kvb = next((r.get("kv_blocks") for r in S.values() if r.get("kv_blocks")), None)
            if kvb:
                mac("KVTokens", f"{16 * kvb // 1000}k")
        cot = S.get("noshare@cot")
        if cot:
            mac(f"{k}EMcot", f"{cot['em']:.3f}")
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
            mac(f"{k}UDrain", f"{u.get('drain_s', 0):.1f}")
            mac(f"{k}UPlanExact", str(u.get("n_plan_exact", 0)))
            mac(f"{k}UPlanN", str(u.get("n_plan", 0)))
            mac(f"{k}UTeamEarly", str(u.get("n_team_early", 0)))
            mac(f"{k}URoundEarly", str(u.get("n_round_early", 0)))
            mac(f"{k}ULatAll", f2(u.get("lat_all_mean", u["lat_mean"])))
            mac(f"{k}UExact", str(u.get("src_exact", 0)))
            mac(f"{k}UVerify", str(u.get("n_verify", 0)))
        pubs = [(S[m]["lat_mean"], m) for m in PUB if m in S]
        if pubs and u:
            best = min(pubs)[1]
            mac(f"{k}BestPub", dict(MAIN)[best].split("~")[0])
            pair_macros("v15_main", ds, U, best, f"{k}LatRedBest")
            gb = min((gpu_s(S[m]), m) for m in PUB if m in S)
            mac(f"{k}GpuRedBest", f"{100 * (1 - gpu_s(u) / gb[0]):.0f}")
            for b in ("noshare", SBD, "noshare@cot", "suffix", "lacache"):
                pair_macros("v15_main", ds, U, b, f"{k}LatRed{KEYN[b]}")
            if SBD in S:
                mac(f"{k}GpuRedSbd", f"{100 * (1 - gpu_s(u) / gpu_s(S[SBD])):.0f}")
                mac(f"{k}PnnRedSbd", f"{100 * (1 - u['lat_p99'] / S[SBD]['lat_p99']):.0f}")
            if "lacache" in S:
                mac(f"{k}GpuRedLacache", f"{100 * (1 - gpu_s(u) / gpu_s(S['lacache'])):.0f}")
                mac(f"{k}PnnRedLacache", f"{100 * (1 - u['lat_p99'] / S['lacache']['lat_p99']):.0f}")
            if "noshare" in S:
                mac(f"{k}GpuRedVllm", f"{100 * (1 - gpu_s(u) / gpu_s(S['noshare'])):.0f}")
                mac(f"{k}GenRedVllm", f"{100 * (1 - u.get('gen_tokens', 0) / max(1, S['noshare'].get('gen_tokens', 1))):.0f}")
    with open(os.path.join(P, "tab_main.tex"), "w") as fo:
        for m, lab in MAIN:
            if not rows[m]:
                continue
            if m == U:
                fo.write("\\midrule\n")
            fo.write(f"{lab} & " + " & ".join(rows[m]) + r" \\" + "\n")


# ---------------------------------------------------------------- ablation (frontier attribution)
def abl_table():
    with open(os.path.join(P, "tab_abl.tex"), "w") as fo:
        for ds in DS:
            S = load("v15_abl", ds)
            k = DSK[ds]
            fo.write(f"\\multicolumn{{10}}{{l}}{{\\textit{{{DS[ds]}}}}}\\\\\n")
            for m, lab in ABL:
                r = S.get(m)
                if r is None:
                    fo.write(f"{lab} & " + " & ".join(["?"] * 9) + r" \\" + "\n")
                    continue
                kt = max(1, r.get("key_tokens", 0))
                peer = 100 * (r.get("acc_trie", 0) + r.get("acc_live", 0)) / kt
                live = 100 * r.get("acc_live", 0) / kt
                ksteps = r.get("key_steps", 0) / max(1, r.get("src_engine", 0) + r.get("src_dcert", 0))
                cells = [f2(r["lat_mean"]), f"{r['lat_p99']:.1f}", f2(gpu_s(r)), f"{ident_pct(r):.0f}",
                         f"{verified_pct(r):.0f}", f"{ksteps:.1f}", f"{peer:.0f}",
                         str(r.get("prov_engine", 0)), f"{r.get('prov_tokens', 0) / 1000:.1f}"]
                fo.write(f"{lab} & " + " & ".join(cells) + r" \\" + "\n")
                key = KEYN[m]
                mac(f"{k}A{key}Lat", f2(r["lat_mean"]))
                mac(f"{k}A{key}Pnn", f"{r['lat_p99']:.1f}")
                mac(f"{k}A{key}Gpu", f2(gpu_s(r)))
                mac(f"{k}A{key}Ident", f"{ident_pct(r):.0f}")
                mac(f"{k}A{key}Verif", f"{verified_pct(r):.0f}")
                mac(f"{k}A{key}Peer", f"{peer:.0f}")
                mac(f"{k}A{key}Live", f"{live:.0f}")
                mac(f"{k}A{key}Steps", f"{ksteps:.1f}")
                mac(f"{k}A{key}ProvEng", str(r.get("prov_engine", 0)))
                mac(f"{k}A{key}ProvTok", f"{r.get('prov_tokens', 0) / 1000:.1f}")
                mac(f"{k}A{key}Lag", f2(r.get("prov_lag_p50", 0)))
                mac(f"{k}A{key}EM", f"{r['em']:.3f}")
                mac(f"{k}A{key}PlanSteps", f"{r.get('plan_steps', 0) / max(1, r.get('plan_engine_calls', r.get('n_plan', 1))):.1f}")

            def lat(m):
                return S[m]["lat_mean"] if m in S else None
            pairs = {"KeyGain": ("u15-coop", SBD), "PayGain": ("u15-keyed", SBD), "KeyGainGivenPay": (U, "u15-keyed"),
                     "Total": (U, SBD), "ProvCost": (U, "u15-lazy"), "LiveGain": (U, "u15-nolive"),
                     "PrioGain": (U, "u15-noprio"), "IdentGain": (U, "u15-noid"), "DfGain": (U, "u15-nodf")}
            for nm, (a, b) in pairs.items():
                if lat(a) and lat(b):
                    mac(f"{k}Abl{nm}", f"{100 * (1 - lat(a) / lat(b)):.0f}")
                    mac(f"{k}Abl{nm}Gpu", f"{100 * (1 - gpu_s(S[a]) / gpu_s(S[b])):.0f}")
                    mac(f"{k}Abl{nm}Pnn", f"{100 * (1 - S[a]['lat_p99'] / S[b]['lat_p99']):.0f}")
            if "u15-keyed" in S and SBD in S:
                a, b = S["u15-keyed"], S[SBD]
                mac(f"{k}AblPayExecRed", f"{100 * (1 - a.get('prov_engine', 0) / max(1, b.get('prov_engine', 1))):.0f}")
                mac(f"{k}AblPayTokRed", f"{100 * (1 - a.get('prov_tokens', 0) / max(1, b.get('prov_tokens', 1))):.0f}")
            if "u15-coop" in S and SBD in S:
                a, b = S["u15-coop"], S[SBD]
                sa = a.get("key_steps", 0) / max(1, a.get("src_engine", 0) + a.get("src_dcert", 0))
                sb = b.get("key_steps", 0) / max(1, b.get("src_engine", 0) + b.get("src_dcert", 0))
                mac(f"{k}AblStepRed", f"{100 * (1 - sa / sb):.0f}")
            fo.write("\\midrule\n")


# ---------------------------------------------------------------- payload width regime
def wide_table():
    with open(os.path.join(P, "tab_wide.tex"), "w") as fo:
        for ds in DS:
            S = load("v15_wide", ds)
            N = load("v15_main", ds)
            k = DSK[ds]
            fo.write(f"\\multicolumn{{8}}{{l}}{{\\textit{{{DS[ds]}}}}}\\\\\n")
            for m, lab in WIDE:
                r, rn = S.get(m), N.get(m)
                if r is None or rn is None:
                    continue
                cells = [f2(rn["lat_mean"]) if rn else "?", f2(r["lat_mean"]), f"{r['lat_p99']:.1f}", f"{r['em']:.3f}",
                         f2(gpu_s(r)), f"{r.get('gen_tokens', 0) / 1000:.0f}", f"{r.get('prov_tokens', 0) / 1000:.1f}"]
                fo.write(f"{lab} & " + " & ".join(cells) + r" \\" + "\n")
                key = KEYN[m]
                mac(f"{k}W{key}Lat", f2(r["lat_mean"]))
                mac(f"{k}W{key}Pnn", f"{r['lat_p99']:.1f}")
                mac(f"{k}W{key}Gpu", f2(gpu_s(r)))
                mac(f"{k}W{key}EM", f"{r['em']:.3f}")
                mac(f"{k}W{key}GenTok", f"{r.get('gen_tokens', 0) / 1000:.0f}")
                mac(f"{k}W{key}ProvTok", f"{r.get('prov_tokens', 0) / 1000:.1f}")
                mac(f"{k}W{key}ProvEng", str(r.get("prov_engine", 0)))
                mac(f"{k}W{key}Lag", f2(r.get("prov_lag_p50", 0)))
                if rn:
                    mac(f"{k}W{key}LatGrow", f"{100 * (r['lat_mean'] / rn['lat_mean'] - 1):.0f}")
            if U in S and SBD in S:
                st = pair_macros("v15_wide", ds, U, SBD, f"{k}WLatRedSbd")
                if st:
                    mac(f"{k}WLatRedSbdAbs", f"{abs(st['red']):.0f}")
                    lo, hi = sorted([-st['red_hi'], -st['red_lo']])  # interval of the *increase*
                    mac(f"{k}WLatIncSbdCI", f"[{lo:.0f},{hi:.0f}]")
                mac(f"{k}WGpuRedSbd", f"{100 * (1 - gpu_s(S[U]) / gpu_s(S[SBD])):.0f}")
                mac(f"{k}WPnnRedSbd", f"{100 * (1 - S[U]['lat_p99'] / S[SBD]['lat_p99']):.0f}")
                mac(f"{k}WPnnRedSbdAbs", f"{abs(100 * (1 - S[U]['lat_p99'] / S[SBD]['lat_p99'])):.0f}")
                mac(f"{k}WPayTokRed", f"{100 * (1 - S[U].get('prov_tokens', 0) / max(1, S[SBD].get('prov_tokens', 1))):.0f}")
            if U in S and "lacache" in S:
                pair_macros("v15_wide", ds, U, "lacache", f"{k}WLatRedLacache")
            if U in S and "noshare" in S:
                pair_macros("v15_wide", ds, U, "noshare", f"{k}WLatRedVllm")
            fo.write("\\midrule\n")


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


def load_curve():
    data = {m: [] for m, _ in LOADM}
    for rate in RATES:
        tag = "v15_main" if rate == 1.2 else f"v15_load_r{rate}"
        S = load(tag, "hotpotqa")
        for m, _ in LOADM:
            if m in S:
                data[m].append((rate, S[m]["lat_mean"], S[m]["lat_p99"], gpu_s(S[m])))
    wdata = {m: [] for m, _ in WIDE}
    for rate in WRATES:
        tag = "v15_wide" if rate == 1.2 else f"v15_wload_r{rate}"
        S = load(tag, "hotpotqa")
        for m in wdata:
            if m in S:
                wdata[m].append((rate, S[m]["lat_mean"], S[m]["lat_p99"], gpu_s(S[m])))
    cap = {}
    with open(os.path.join(P, "tab_load.tex"), "w") as fo:
        for m, lab in LOADM:
            pts = {x[0]: x for x in data[m]}
            if any(r not in pts for r in RATES):
                continue
            cells = [f"{pts[r][1]:.2f}/{pts[r][2]:.1f}" for r in RATES]
            c, sat = capacity(data[m])
            cap[m] = c
            cs = "?" if c is None else (f"$\\ge${c:.1f}" if sat else f"{c:.2f}")
            fo.write(f"{lab} & " + " & ".join(cells) + f" & {cs}" + r" \\" + "\n")
            mac(f"Cap{KEYN[m]}", cs)
        wcap = {}
    # Wide-payload load is a separate table: only the rates that were measured, so the
    # cells are complete. The 3.0 q/s wide run did not finish; capacity is interpolated
    # from the measured points whose P99 crosses the objective.
    measured_wide = [1.2, 1.8, 2.4]
    with open(os.path.join(P, "tab_wload.tex"), "w") as fo:
        for m, lab in WIDE:
            pts = {x[0]: x for x in wdata[m]}
            if any(r not in pts for r in measured_wide):
                continue
            cells = [f"{pts[r][1]:.2f}/{pts[r][2]:.1f}" for r in measured_wide]
            c, sat = capacity(wdata[m])
            wcap[m] = c
            cs = f"$\\ge${c:.1f}" if sat else f"{c:.2f}"
            fo.write(f"{lab} & " + " & ".join(cells) + f" & {cs}" + r" \\" + "\n")
            mac(f"WCap{KEYN[m]}", cs)
    if cap.get(U) and cap.get(SBD):
        mac("CapGainSbd", f"{100 * (cap[U] / cap[SBD] - 1):.0f}")
    if wcap.get(U) and wcap.get(SBD):
        mac("WCapGainSbd", f"{100 * (wcap[U] / wcap[SBD] - 1):.0f}")
    pub = [cap[m] for m in PUB if cap.get(m)]
    if cap.get(U) and pub:
        mac("CapGainPub", f"{cap[U] / max(pub):.1f}")
    for rate in WRATES:  # wide-payload GPU/P99 at each rate
        tag = "v15_wide" if rate == 1.2 else f"v15_wload_r{rate}"
        S = load(tag, "hotpotqa")
        if U in S and SBD in S:
            tagk = RATEK[rate]
            mac(f"WL{tagk}GpuRed", f"{100 * (1 - gpu_s(S[U]) / gpu_s(S[SBD])):.0f}")
            mac(f"WL{tagk}PnnRed", f"{100 * (1 - S[U]['lat_p99'] / S[SBD]['lat_p99']):.0f}")
            mac(f"WL{tagk}LatRed", f"{100 * (1 - S[U]['lat_mean'] / S[SBD]['lat_mean']):.0f}")
            mac(f"WL{tagk}PnnU", f"{S[U]['lat_p99']:.1f}")
            mac(f"WL{tagk}PnnS", f"{S[SBD]['lat_p99']:.1f}")
            mac(f"WL{tagk}LatU", f2(S[U]['lat_mean']))
            mac(f"WL{tagk}LatS", f2(S[SBD]['lat_mean']))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(10, 2.5))
        marks = "osv^D*"
        for (m, lab), mk in zip(LOADM, marks):
            d = sorted(data[m])
            if len(d) < 2:
                continue
            lab2 = lab.replace("\\textbf{", "").replace("}", "")
            ax[0].plot([x[0] for x in d], [x[1] for x in d], marker=mk, label=lab2, lw=1.2, ms=4)
            ax[1].plot([x[0] for x in d], [x[2] for x in d], marker=mk, label=lab2, lw=1.2, ms=4)
        for (m, lab), mk in zip(WIDE, marks):
            d = sorted(wdata.get(m, []))
            if len(d) < 2:
                continue
            lab2 = lab.replace("\\textbf{", "").replace("}", "")
            ax[2].plot([x[0] for x in d], [x[2] for x in d], marker=mk, label=lab2 + ", wide", lw=1.2, ms=4)
        ax[1].axhline(SLO, color="gray", ls=":", lw=0.8)
        ax[2].axhline(SLO, color="gray", ls=":", lw=0.8)
        ax[0].set_ylabel("mean latency (s)")
        ax[1].set_ylabel("P99 latency (s)")
        ax[2].set_ylabel("P99 latency (s), wide payload")
        for a in ax:
            a.set_xlabel("arrival rate (queries/s)")
            a.set_yscale("log")
            a.grid(alpha=0.3, lw=0.5)
        ax[0].legend(fontsize=7, frameon=False)
        ax[2].legend(fontsize=7, frameon=False)
        fig.tight_layout()
        os.makedirs(os.path.join(P, "figs"), exist_ok=True)
        fig.savefig(os.path.join(P, "figs", "load.pdf"))
    except Exception as e:
        WARN.append(f"figure: {e}")


# ---------------------------------------------------------------- version-safety stress
def edit_table():
    S = load("v15_edit", "hotpotqa")
    with open(os.path.join(P, "tab_edit.tex"), "w") as fo:
        for m, lab in EDITM:
            r = S.get(m)
            if r is None:
                continue
            n, e = nonexact(r)
            fo.write(f"{lab} & {r.get('stale_reuse', 0)} & {r.get('stale_decisive', 0)} & "
                     f"{r.get('stale_decisive_err', 0)} & {100 * e / n if n else 0:.1f} \\\\\n")
            key = KEYN[m]
            mac(f"Ed{key}Xv", str(r.get("stale_reuse", 0)))
            mac(f"Ed{key}Dec", str(r.get("stale_decisive", 0)))
            mac(f"Ed{key}DecErr", str(r.get("stale_decisive_err", 0)))
            mac(f"Ed{key}EM", f"{r['em']:.3f}")
            mac(f"Ed{key}ErrPct", f"{100 * e / n if n else 0:.1f}")
            mac(f"Ed{key}Reuse", str(n))
            mac(f"Ed{key}Lat", f2(r["lat_mean"]))
        if S:
            mac("EdUpdates", str(next(iter(S.values())).get("n_updates", 0)))


# ---------------------------------------------------------------- redundancy sweep
def zipf_table():
    with open(os.path.join(P, "tab_zipf.tex"), "w") as fo:
        for z, tag in (("0", "v15_zipf_0"), ("0.8", "v15_main"), ("1.2", "v15_zipf_1.2")):
            S = load(tag, "hotpotqa")
            zk = ZIPFK[z]
            ns = S.get("noshare") or S.get("suffix")
            red = f"{100 * ns['redund_identical']:.0f}" if ns else "?"
            fo.write(f"\\multicolumn{{7}}{{l}}{{\\textit{{Zipf $s={z}$: {red}\\% of verification calls byte-identical to another}}}}\\\\\n")
            mac(f"Z{zk}RedIdent", red)
            for m, lab in ZIPFM:
                r = S.get(m)
                if r is None:
                    continue
                n, e = nonexact(r)
                cells = [f2(r["lat_mean"]), f"{r['lat_p99']:.1f}", f2(gpu_s(r)), f"{ident_pct(r):.0f}",
                         f"{verified_pct(r):.0f}", f"{100 * e / n:.1f}" if n else "0.0"]
                fo.write(f"{lab} & " + " & ".join(cells) + r" \\" + "\n")
                key = KEYN[m]
                mac(f"Z{zk}{key}Lat", f2(r["lat_mean"]))
                mac(f"Z{zk}{key}Gpu", f2(gpu_s(r)))
                mac(f"Z{zk}{key}Ident", f"{ident_pct(r):.0f}")
                mac(f"Z{zk}{key}Verif", f"{verified_pct(r):.0f}")
            if U in S and SBD in S:
                mac(f"Z{zk}LatRedSbd", f"{100 * (1 - S[U]['lat_mean'] / S[SBD]['lat_mean']):.0f}")
            if U in S and "lacache" in S:
                mac(f"Z{zk}LatRedLacache", f"{100 * (1 - S[U]['lat_mean'] / S['lacache']['lat_mean']):.0f}")
            if U in S and "suffix" in S:
                mac(f"Z{zk}LatRedSuffix", f"{100 * (1 - S[U]['lat_mean'] / S['suffix']['lat_mean']):.0f}")
            for m in PUB:
                if U in S and m in S:
                    n, e = nonexact(S[m])
                    mac(f"Z{zk}LatRed{KEYN[m]}", f"{100 * (1 - S[U]['lat_mean'] / S[m]['lat_mean']):.0f}")
                    mac(f"Z{zk}{KEYN[m]}Err", f"{100 * e / n:.1f}" if n else "0.0")
            fo.write("\\midrule\n")


# ---------------------------------------------------------------- latency by query class (where each sharing mode applies)
CLSM = [("noshare", "vLLM + APC"), ("pld", "PLD"), ("suffix", "SuffixDecoding"),
        ("gptcache", "GPTCache"), ("vcache-official", "vCache"), (U, "\\textbf{Unison}")]


def class_table():
    """Queries split by what preceded them in the stream: first occurrence of the question, a verbatim repeat
    (identity sharing applies to the whole trajectory), or a paraphrased repeat (only verified sharing can apply
    beyond the passage prefix)."""
    with open(os.path.join(P, "tab_class.tex"), "w") as fo:
        for ds in DS:
            k = DSK[ds]
            ref = qs("v15_main", ds, U)
            if not ref:
                fo.write(f"\\multicolumn{{4}}{{l}}{{\\textit{{{DS[ds]}}}}}\\\\\n" + "\n".join(f"{lab} & ? & ? & ? \\\\" for _, lab in CLSM) + "\n")
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
            for m, lab in CLSM:
                q = qs("v15_main", ds, m)
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
    with open(os.path.join(P, "tab_audit.tex"), "w") as fo:
        for tag, lab_t in (("v15_abl", "one-sentence payload"), ("v15_wide", "wide payload")):
            for ds in DS:
                p = os.path.join(R, tag, ds, f"audit_{U}_vs_{SBD}.json")
                pre = ("" if tag == "v15_abl" else "W") + DSK[ds] + "Aud"
                if not os.path.exists(p):
                    WARN.append(f"missing audit {tag}/{ds}")
                    fo.write(f"\\multicolumn{{4}}{{l}}{{\\textit{{{DS[ds]}, {lab_t}}}}}\\\\\n? & ? & ? & ? \\\\\n")
                    continue
                a = json.load(open(p))
                fo.write(r"\multicolumn{4}{l}{\textit{" + DS[ds] + f", {lab_t}, {a['n_pairs']} pairs" + r"}}\\" + "\n")
                for q, lab in (("grounded", "Grounded"), ("responsive", "Responsive"), ("accept", "Accept"),
                               ("cited_contains_answer", "Cited sentence contains answer")):
                    d, lo, hi = (100 * x for x in a[f"{q}_diff"])
                    u, b = 100 * a[q + "_a"], 100 * a[q + "_b"]
                    fo.write(f"{lab} & {u:.1f} & {b:.1f} & {d:+.1f} [{lo:+.1f}, {hi:+.1f}]" + r"\\" + "\n")
                    mac(f"{pre}{q.replace('_', '')}U", f"{u:.1f}")
                    mac(f"{pre}{q.replace('_', '')}B", f"{b:.1f}")
                    mac(f"{pre}{q.replace('_', '')}D", f"{d:+.1f}")
                    mac(f"{pre}{q.replace('_', '')}DCI", f"[{lo:+.1f},{hi:+.1f}]")
                mac(f"{pre}N", str(a["n_pairs"]))
                mac(f"{pre}Unparsed", f"{100 * max(a.get('unparsed_a', 0), a.get('unparsed_b', 0)):.1f}")
                mac(f"{pre}LenU", f"{a.get('prov_len_a', 0):.0f}")
                mac(f"{pre}LenB", f"{a.get('prov_len_b', 0):.0f}")


cost_model()
main_table()
abl_table()
wide_table()
load_curve()
edit_table()
zipf_table()
class_table()
audit_table()
for name in os.listdir(P):  # block separators are written after each block; the last one would meet \bottomrule
    if name.startswith("tab_") and name.endswith(".tex"):
        path = os.path.join(P, name)
        txt = open(path).read()
        while txt.rstrip().endswith("\\midrule"):
            txt = txt.rstrip()[: -len("\\midrule")]
        with open(path, "w") as fo:
            fo.write(txt.rstrip() + "\n")
with open(os.path.join(P, "numbers_auto.tex"), "w") as fo:
    fo.write("% generated by code/scripts/analyze_v15.py -- do not edit\n")
    for k, v in sorted(M.items()):
        fo.write(f"\\providecommand{{\\{k}}}{{{v}}}\n")
print(f"{len(M)} macros")
for w in dict.fromkeys(WARN):
    print("WARNING", w)
