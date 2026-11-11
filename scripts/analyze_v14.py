"""Tables, figure and number macros for paper/unison14 from <results>/v14_*/<ds>/{summary.json,*.queries.json}.

usage: python analyze_v14.py <results_dir> <paper_dir>
Every number in the paper's evaluation comes from here. Missing runs print '?' (never silently '-').
Statistics: single stream per setting; paired bootstrap over queries (same stream, same query index) for
latency ratios and EM differences, exact McNemar test for EM, Clopper-Pearson upper bounds for error rates.
"""
import json
import os
import sys

import numpy as np
from scipy.stats import beta, binomtest

R, P = sys.argv[1], sys.argv[2]
os.makedirs(P, exist_ok=True)
DS = {"hotpotqa": "HotpotQA", "musique": "MuSiQue"}
DSK = {"hotpotqa": "Ho", "musique": "Mu"}
U = "u14"
PUB = ["noshare", "pld", "suffix", "gptcache", "vcache-official", "lacache"]
MAIN = [("noshare", r"vLLM + APC~\cite{kwon2023vllm}"), ("pld", r"PLD~\cite{saxena2023pld}"),
        ("suffix", r"SuffixDecoding~\cite{oliaro2025suffix}"), ("gptcache", r"GPTCache~\cite{bang2023gptcache}"),
        ("vcache-official", r"vCache~\cite{schroeder2025vcache}"), ("lacache", r"LaCache$^\dagger$~\cite{liang2026lacache}"),
        ("u14-sbd", r"Boundary+Defer$^\ddagger$"), (U, r"\textbf{Unison}")]
ABL = [("u14-lazy", "Decision only (no provenance)", "--", "--"),
       ("u14-sbd", "(A) own key, per-call provenance", "self", "call"),
       ("u14-coop", "(B) cooperative key, per-call prov.", "coop", "call"),
       ("u14-keyed", "(C) own key, claim-keyed prov.", "self", "keyed"),
       (U, r"\textbf{(D) Unison}", "coop", "keyed"),
       ("u14-nolive", "(D) w/o in-flight peers", "trie", "keyed"),
       ("u14-noprio", "(D) w/o compute-minimal payload", "coop", "keyed")]
LOADM = [("suffix", "SuffixDecoding"), ("gptcache", "GPTCache"), ("u14-sbd", "Boundary+Defer"), (U, "Unison")]
RATES = [0.6, 1.2, 1.8, 2.4, 3.0]
SLO = 10.0  # P99 latency objective (s) for the capacity column
EDITM = [("gptcache", "GPTCache"), ("vcache-official", "vCache"), ("lacache", "LaCache$^\\dagger$"),
         ("u14-sbd", "Boundary+Defer"), (U, "\\textbf{Unison}")]
KEYN = {"noshare": "Vllm", "pld": "Pld", "suffix": "Suffix", "gptcache": "Gptcache", "vcache-official": "Vcache",
        "lacache": "Lacache", "u14-sbd": "Sbd", U: "Unison", "noshare@cot": "VllmCot", "u14-lazy": "Lazy",
        "u14-coop": "Coop", "u14-keyed": "Keyed", "u14-nolive": "Nolive", "u14-noprio": "Noprio"}
M = {}
WARN = []


def mac(name, val):
    name = "".join(ch for ch in name if ch.isalpha())
    M[name] = val


def load(tag, ds):
    p = os.path.join(R, tag, ds, "summary.json")
    if not os.path.exists(p):
        WARN.append(f"missing {tag}/{ds}")
        return {}
    out = {r["method"]: r for r in json.load(open(p))}
    lo = min((r.get("gpu_others_mib", 0) for r in out.values() if r.get("gpu_others_mib", -1) >= 0), default=0)
    for m, r in out.items():  # co-tenant check: another process grabbed GPU memory during this method's run
        if r.get("gpu_others_mib", 0) - lo > 1000:
            WARN.append(f"CONTAMINATED? {tag}/{ds}/{m}: others={r['gpu_others_mib']} MiB (min {lo}) -> rerun")
    return out


def qs(tag, ds, m):
    p = os.path.join(R, tag, ds, f"{m}.queries.json")
    if not os.path.exists(p):
        return None
    return {q["qi"]: q for q in json.load(open(p))}


def f(x, d=2):
    return "?" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}"


def gpu_s(r):  # engine busy seconds per query, including the drain of background work
    return r["engine_busy"] / max(1, r["n_queries"])


def nonexact(r):  # reuses that are not identical-prompt coalescing: (count, wrong)
    n = sum(r.get(f"n_{s}", 0) for s in ("dcert", "sem", "risk"))
    e = sum(r.get(f"nerr_{s}", 0) for s in ("dcert", "sem", "risk"))
    return n, e


def cp_hi(k, n):
    return 100 * (beta.ppf(0.975, k + 1, n - k) if k < n else 1.0) if n else float("nan")


def paired(tag, ds, a, b, B=4000):
    """Mean-latency reduction of a vs b (%), paired bootstrap 95% CI; EM difference (pts) with CI, McNemar p."""
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


# ---------------------------------------------------------------- main table
def main_table():
    rows = {m: [] for m, _ in MAIN}
    for ds in DS:
        S = load("v14_main", ds)
        k = DSK[ds]
        floor = None
        for m, _ in MAIN:
            r = S.get(m)
            if r is None:
                rows[m].append("? & ? & ? & ?")
                continue
            n, e = nonexact(r)
            err = f"{100 * e / n:.1f}" if n else "--"
            rows[m].append(f"{r['lat_mean']:.2f} & {r['lat_p99']:.1f} & {100 * r['em']:.1f} & {gpu_s(r):.2f}")
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
        ns = S.get("noshare")
        if ns and "redund_passage60" in ns:  # workload redundancy measured on the independent execution
            mac(f"{k}RedPass", f"{100 * ns['redund_passage60']:.0f}")
            mac(f"{k}RedIdent", f"{100 * ns['redund_identical']:.0f}")
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
        pubs = [(S[m]["lat_mean"], m) for m in PUB if m in S]
        if pubs and u:
            best = min(pubs)[1]
            mac(f"{k}BestPub", dict(MAIN)[best].split("~")[0])
            st = paired("v14_main", ds, U, best)
            if st:
                mac(f"{k}LatRedBest", f"{st['red']:.0f}")
                mac(f"{k}LatRedBestCI", f"[{st['red_lo']:.0f},{st['red_hi']:.0f}]")
            gb = min((gpu_s(S[m]), m) for m in PUB if m in S)
            mac(f"{k}GpuRedBest", f"{100 * (1 - gpu_s(u) / gb[0]):.0f}")
            for b in ("noshare", "u14-sbd", "noshare@cot", "suffix"):
                st = paired("v14_main", ds, U, b)
                if st:
                    kk = KEYN[b]
                    mac(f"{k}LatRed{kk}", f"{st['red']:.0f}")
                    mac(f"{k}LatRed{kk}CI", f"[{st['red_lo']:.0f},{st['red_hi']:.0f}]")
                    mac(f"{k}EMd{kk}", f"{st['dem']:+.1f}")
                    mac(f"{k}EMd{kk}CI", f"[{st['dem_lo']:+.1f},{st['dem_hi']:+.1f}]")
                    mac(f"{k}EMp{kk}", f"{st['p']:.2f}")
            if "u14-sbd" in S:
                mac(f"{k}GpuRedSbd", f"{100 * (1 - gpu_s(u) / gpu_s(S['u14-sbd'])):.0f}")
                mac(f"{k}PnnRedSbd", f"{100 * (1 - u['lat_p99'] / S['u14-sbd']['lat_p99']):.0f}")
    with open(os.path.join(P, "tab_main.tex"), "w") as fo:
        for m, lab in MAIN:
            fo.write(f"{lab} & " + " & ".join(rows[m]) + r" \\" + "\n")
            if m == "lacache":
                fo.write("\\midrule\n")


def f2(x):
    return f"{x:.2f}"


# ---------------------------------------------------------------- ablation
def abl_table():
    with open(os.path.join(P, "tab_abl.tex"), "w") as fo:
        records = {m: [] for m in ("u14-sbd", "u14-coop", "u14-keyed", U)}
        for ds in DS:
            S = load("v14_abl", ds)
            k = DSK[ds]
            for m, lab, _, _ in ABL:
                r = S.get(m)
                if r is None:
                    if m in records:
                        records[m].extend(["?"] * 3)
                    continue
                kt = max(1, r.get("key_tokens", 0))
                peer = 100 * (r.get("acc_trie", 0) + r.get("acc_live", 0)) / kt
                live = 100 * r.get("acc_live", 0) / kt
                ksteps = r.get("key_steps", 0) / max(1, r.get("src_engine", 0) + r.get("src_dcert", 0))
                if m in records:
                    records[m].extend([f"{ksteps:.1f}", f"{r['prov_engine']:,}", f"{r['prov_tokens']:,}"])
                key = KEYN[m]
                mac(f"{k}A{key}Lat", f2(r["lat_mean"]))
                mac(f"{k}A{key}Pnn", f"{r['lat_p99']:.1f}")
                mac(f"{k}A{key}Gpu", f2(gpu_s(r)))
                mac(f"{k}A{key}Peer", f"{peer:.0f}")
                mac(f"{k}A{key}Live", f"{live:.0f}")
                mac(f"{k}A{key}Steps", f"{ksteps:.1f}")
                mac(f"{k}A{key}ProvEng", str(r.get("prov_engine", 0)))
                mac(f"{k}A{key}ProvTok", f"{r.get('prov_tokens', 0) / 1000:.1f}")
            # attribution: B-A (key), C-A (payload), D-C (key given keyed payload), D-lazy (cost of provenance)
            def lat(m):
                return S[m]["lat_mean"] if m in S else None
            pairs = {"KeyGain": ("u14-coop", "u14-sbd"), "PayGain": ("u14-keyed", "u14-sbd"),
                     "KeyGainGivenPay": (U, "u14-keyed"), "Total": (U, "u14-sbd"), "ProvCost": (U, "u14-lazy"),
                     "LiveGain": (U, "u14-nolive"), "PrioGain": (U, "u14-noprio")}
            for nm, (a, b) in pairs.items():
                if lat(a) and lat(b):
                    mac(f"{k}Abl{nm}", f"{100 * (1 - lat(a) / lat(b)):.0f}")
                    mac(f"{k}Abl{nm}Gpu", f"{100 * (1 - gpu_s(S[a]) / gpu_s(S[b])):.0f}")
        for m, setting, drafts, unit in [
            ("u14-sbd", "A", "Suffix", "Call"), ("u14-coop", "B", "Suffix + peers", "Call"),
            ("u14-keyed", "C", "Suffix", "Claim"), (U, "D (Unison)", "Suffix + peers", "Claim")
        ]:
            fo.write(" & ".join([setting, drafts, unit] + records[m]) + r" \\" + "\n")



# ---------------------------------------------------------------- load curve and capacity
def load_curve():
    data = {m: [] for m, _ in LOADM}
    for rate in RATES:
        tag = "v14_main" if rate == 1.2 else f"v14_load_r{rate}"
        S = load(tag, "hotpotqa")
        for m, _ in LOADM:
            if m in S:
                data[m].append((rate, S[m]["lat_mean"], S[m]["lat_p99"], gpu_s(S[m])))
    cap = {}
    with open(os.path.join(P, "tab_load.tex"), "w") as fo:
        for m, lab in LOADM:
            pts = {x[0]: x for x in data[m]}
            cells = [f"{pts[r][1]:.2f}/{pts[r][2]:.1f}" if r in pts else "?" for r in RATES]
            c, sat = capacity(data[m])
            cap[m] = c
            cs = "?" if c is None else (f"$\\ge${c:.1f}" if sat else f"{c:.2f}")
            fo.write(f"{lab} & " + " & ".join(cells) + f" & {cs}" + r" \\" + "\n")
            mac(f"Cap{KEYN[m]}", cs)
    if cap.get(U) and cap.get("u14-sbd"):
        mac("CapGainSbd", f"{100 * (cap[U] / cap['u14-sbd'] - 1):.0f}")
    pub = [cap[m] for m in ("suffix", "gptcache") if cap.get(m)]
    if cap.get(U) and pub:
        mac("CapGainPub", f"{cap[U] / max(pub):.1f}")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter
        with plt.rc_context({"font.size": 8, "pdf.fonttype": 42, "axes.spines.top": False,
                             "axes.spines.right": False}):
            fig, axes = plt.subplots(2, 1, sharex=True, figsize=(3.4, 3.95))
            styles = [("#60717c", "s", "-."), ("#9d632c", "^", "--"),
                      ("#604d78", "o", ":"), ("#116567", "x", "-")]
            for (method, label), (color, marker, line) in zip(LOADM, styles):
                points = sorted(data[method])
                for index, axis in enumerate(axes, start=1):
                    axis.plot([p[0] for p in points], [p[index] for p in points], color=color,
                              linestyle=line, marker=marker, markerfacecolor="none",
                              markersize=5.5 if method == "u14-sbd" else 4, lw=1.2, label=label)
            for axis, label in zip(axes, ("Mean latency (s)", "P99 latency (s)")):
                axis.set_yscale("log")
                axis.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
                axis.set_ylabel(label)
                axis.grid(axis="y", alpha=0.25, lw=0.5)
            axes[1].axhline(SLO, color="#555555", ls=(0, (4, 2)), lw=0.8, zorder=0)
            axes[1].set_xticks(RATES)
            axes[1].set_xlabel("Arrival rate (queries/s)")
            fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=2,
                       fontsize=7.5, frameon=False)
            fig.subplots_adjust(left=0.17, bottom=0.12, top=0.85, right=0.98, hspace=0.12)
            os.makedirs(os.path.join(P, "figs"), exist_ok=True)
            fig.savefig(os.path.join(P, "figs", "load.pdf"))
            plt.close(fig)
    except Exception as e:
        WARN.append(f"figure: {e}")


def capacity(pts):
    """Largest arrival rate whose P99 meets the SLO (linear interpolation in rate between measured points)."""
    pts = sorted(pts)
    if not pts or pts[0][2] > SLO:
        return None, False
    best = pts[0][0]
    for (r0, _, p0, _), (r1, _, p1, _) in zip(pts, pts[1:]):
        if p1 <= SLO:
            best = r1
        else:
            return r0 + (r1 - r0) * (SLO - p0) / (p1 - p0), False
    return best, True  # the SLO held at every measured rate: capacity is at least the largest one


# ---------------------------------------------------------------- version-safety stress
def edit_table():
    S = load("v14_edit", "hotpotqa")
    main = {ds: load("v14_main", ds) for ds in DS}

    def disagreements(wrong, checked):
        return f"{wrong:,}/{checked:,}" if checked else "--"

    with open(os.path.join(P, "tab_edit.tex"), "w") as fo:
        for m, lab in EDITM:
            r = S.get(m)
            if m != "u14-sbd":
                cells = []
                for stream in main.values():
                    if m in stream:
                        n, e = nonexact(stream[m])
                        cells.append(disagreements(e, n))
                    else:
                        cells.append("?")
                if r:
                    n, e = nonexact(r)
                    cells.extend([str(r["n_updates"]), disagreements(e, n), str(r["stale_reuse"]),
                                  disagreements(r["stale_decisive_err"], r["stale_decisive"])])
                else:
                    cells.extend(["?"] * 4)
                fo.write(" & ".join([lab] + cells) + r" \\" + "\n")
            if not r:
                continue
            n, e = nonexact(r)
            key = KEYN[m]
            mac(f"Ed{key}Xv", str(r.get("stale_reuse", 0)))
            mac(f"Ed{key}Dec", str(r.get("stale_decisive", 0)))
            mac(f"Ed{key}DecErr", str(r.get("stale_decisive_err", 0)))
            mac(f"Ed{key}EM", f"{r['em']:.3f}")
            mac(f"Ed{key}ErrPct", f"{100 * e / n:.1f}" if n else "--")
        if U in S:
            mac("EdUpdates", str(S[U]["n_updates"]))


# ---------------------------------------------------------------- audit quality
def audit_table():
    with open(os.path.join(P, "tab_audit.tex"), "w") as fo:
        for ds in DS:
            p = os.path.join(R, "v14_abl", ds, "audit_u14_vs_u14-sbd.json")
            if not os.path.exists(p):
                WARN.append(f"missing audit {ds}")
                fo.write(f"{DS[ds]} & ? & ? & ? \\\\\n")
                continue
            a = json.load(open(p))
            fo.write(r"\multicolumn{4}{l}{\textit{" + DS[ds] + f", {a['n_pairs']} pairs" + r"}}\\" + "\n")
            for q, lab in (("grounded","Grounded"), ("responsive","Responsive"), ("accept","Accept"), ("cited_contains_answer","Cited")):
                d, lo, hi = (100 * x for x in a[f"{q}_diff"])
                u, baseline = 100 * a[q + "_a"], 100 * a[q + "_b"]
                if q != "cited_contains_answer":
                    n = a["n_pairs"]
                    yes_u, yes_b = round(a[q + "_a"] * n), round(a[q + "_b"] * n)
                    fo.write(f"{lab} & {yes_u} & {yes_b} & {d:+.2f} [{lo:+.2f}, {hi:+.2f}]" + r"\\" + "\n")
                mac(f"{DSK[ds]}Aud{q.replace('_', '')}U", f"{u:.2f}")
                mac(f"{DSK[ds]}Aud{q.replace('_', '')}B", f"{baseline:.2f}")
            mac(f"{DSK[ds]}AudN", str(a["n_pairs"]))


# ---------------------------------------------------------------- models (optional)
def model_table():
    with open(os.path.join(P, "tab_models.tex"), "w") as fo:
        for mdl, lab in (("qwen2.5-7b", "Qwen2.5-7B"), ("llama3.1-8b", "Llama-3.1-8B")):
            S = load(f"v14_model_{mdl}", "hotpotqa")
            if not S:
                continue
            cells = []
            for m, _ in LOADM:
                r = S.get(m)
                cells.append(f"{r['lat_mean']:.2f} / {r['em']:.3f}" if r else "?")
            fo.write(f"{lab} & " + " & ".join(cells) + " \\\\\n")
            if U in S and "u14-sbd" in S:
                mac(f"Mod{'Qwen' if mdl.startswith('qwen') else 'Llama'}RedSbd",
                    f"{100 * (1 - S[U]['lat_mean'] / S['u14-sbd']['lat_mean']):.0f}")


main_table()
abl_table()
load_curve()
edit_table()
audit_table()
model_table()
with open(os.path.join(P, "numbers_auto.tex"), "w") as fo:
    fo.write("% generated by code/scripts/analyze_v14.py -- do not edit\n")
    for k, v in sorted(M.items()):
        fo.write(f"\\providecommand{{\\{k}}}{{{v}}}\n")
print(f"{len(M)} macros")
for w in WARN:
    print("WARNING", w)
