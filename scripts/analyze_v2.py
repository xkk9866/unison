"""Tables (LaTeX) and figures (PDF) for the paper from results_v2/<tag>/<ds>/summary.json and SSN metrics.

Run locally:  python code/scripts/analyze_v2.py results_v2 paper/unison
"""
import json
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES, PAPER = sys.argv[1], sys.argv[2]
FIG = os.path.join(PAPER, "figs")
os.makedirs(FIG, exist_ok=True)
# reference categorical palette (fixed order) + marker/linestyle as secondary encoding
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MK = ["o", "s", "^", "D", "v", "P", "X", "*"]
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e6e6e3", "grid.linewidth": 0.5, "axes.edgecolor": "#8a8984",
                     "lines.linewidth": 1.6, "lines.markersize": 4, "legend.frameon": False,
                     "pdf.fonttype": 42, "font.family": "DejaVu Sans"})

NAMES = {"noshare": "NoShare (APC)", "exact": "ExactShare", "pld": "PLD", "suffix": "SuffixDecoding",
         "exact+suffix": "Exact+Suffix", "semcache-0.90": "SemCache-0.90", "semcache-0.95": "SemCache-0.95", "semcacheV-0.90": "SemCache-V-0.90",
         "semcrc-0.05": "SemCRC-0.05", "osc-e0.05": "TextLaw-0.05", "unison": "Unison ($\\varepsilon$=0)",
         "unison-e0.02": "Unison-0.02", "unison-e0.05": "Unison-0.05", "unison-e0.1": "Unison-0.10"}


def load(tag, ds):
    p = os.path.join(RES, tag, ds, "summary.json")
    if not os.path.exists(p):
        return {}
    return {d["method"]: d for d in json.load(open(p))}


def f(x, d=2):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def main_table():
    rows = []
    for ds, label in [("hotpotqa", "HotpotQA"), ("musique", "MuSiQue")]:
        S = load("v3_main", ds)
        if not S:
            continue
        base = S["noshare"]
        rows.append(f"\\multicolumn{{10}}{{l}}{{\\textit{{{label}}}}}\\\\")
        for m in ["noshare", "exact", "pld", "suffix", "exact+suffix", "semcache-0.90", "semcache-0.95", "semcrc-0.05",
                  "osc-e0.05", "unison-e0.02", "unison-e0.05", "unison-e0.1"]:
            if m not in S:
                continue
            d = S[m]
            work = d["verify_steps"] / base["verify_steps"]
            err = d.get("dec_err_risk")
            rows.append(" & ".join([NAMES[m], f(d["lat_mean"]), f(d["lat_p99"]), f(d["engine_busy"], 1),
                                    f"{100 * work:.0f}\\%", f"{d['reuse_calls']}", f(100 * (1 - d['fidelity_decision']), 1),
                                    "0.0" if err is None or np.isnan(err) else f(100 * err, 1),
                                    f"{d['stale_err']}/{d['stale_reuse']}", f"{d['em']:.3f}/{d['f1']:.3f}"]) + "\\\\")
        rows.append("\\midrule")
    open(os.path.join(PAPER, "tab_main.tex"), "w").write("\n".join(rows[:-1]) + "\n")


def abl_table():
    rows = []
    for ds, label in [("hotpotqa", "HotpotQA"), ("musique", "MuSiQue")]:
        S = load("v3_abl", ds)
        if not S:
            continue
        rows.append(f"\\multicolumn{{7}}{{l}}{{\\textit{{{label}}}}}\\\\")
        lab = {"unison-e0.05": "Unison-0.05 (full)", "unison-e0.05-noreplay": "-- replay (no sharing law)",
               "unison-e0.05-norace": "-- race with in-flight peers", "unison-e0.05-onecand": "-- decision-class candidates",
               "unison-e0.05-nolaw": "-- learned law ($e^{-\\sum\\sigma}$)", "unison-e0.05-nospec": "-- speculation",
               "osc-e0.05": "text law instead of replay (AHN)", "noshare": "NoShare"}
        for m in lab:
            if m not in S:
                continue
            d = S[m]
            err = d.get("dec_err_risk")
            rows.append(" & ".join([lab[m], f(d["lat_mean"]), f(d["lat_p99"]), f"{d['reuse_calls']}",
                                    "0.0" if err is None or np.isnan(err) else f(100 * err, 1),
                                    f"{d['stale_err']}/{d['stale_reuse']}", f"{d['em']:.3f}"]) + "\\\\")
        rows.append("\\midrule")
    open(os.path.join(PAPER, "tab_abl.tex"), "w").write("\n".join(rows[:-1]) + "\n")


def sweep_table(tags, ds, methods, fname, col):
    rows = []
    for tag, lab in tags:
        S = load(tag, ds)
        if not S:
            continue
        cells = [lab]
        for m in methods:
            d = S.get(m)
            cells.append("--" if d is None else f"{d['lat_mean']:.2f} / {d['lat_p99']:.1f}")
        rows.append(" & ".join(cells) + "\\\\")
    open(os.path.join(PAPER, fname), "w").write("\n".join(rows) + "\n")


def fig_load():
    rates = [0.6, 1.2, 1.8, 2.4]
    tags = {0.6: "v3_load_r0.6", 1.2: "v3_load_r1.2", 1.8: "v3_load_r1.8", 2.4: "v3_load_r2.4"}
    ms = ["noshare", "exact+suffix", "semcache-0.90", "osc-e0.05", "unison-e0.05", "unison-e0.1"]
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.3))
    for i, m in enumerate(ms):
        xs, mean, p99 = [], [], []
        for r in rates:
            d = load(tags[r], "hotpotqa").get(m)
            if d:
                xs.append(r), mean.append(d["lat_mean"]), p99.append(d["lat_p99"])
        if xs:
            axs[0].plot(xs, mean, color=C[i], marker=MK[i], label=NAMES[m])
            axs[1].plot(xs, p99, color=C[i], marker=MK[i], label=NAMES[m])
    for ax, t in zip(axs, ["mean query latency (s)", "P99 query latency (s)"]):
        ax.set_xlabel("arrival rate (questions/s)")
        ax.set_ylabel(t)
        ax.set_yscale("log")
    axs[1].legend(fontsize=6.5, loc="upper left", ncol=1)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "load.pdf"))


def fig_version():
    mus = [0, 0.5, 1.0, 2.0]
    ms = ["semcache-0.90", "semcacheV-0.90", "osc-e0.05", "unison-e0.05"]
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.2))
    rates = []
    for i, m in enumerate(ms):
        xs, se, de = [], [], []
        for mu in mus:
            d = load(f"v3_ver_mu{mu}", "hotpotqa").get(m)
            if d:
                xs.append(mu), se.append(d["stale_err"]), de.append(100 * (1 - d["fidelity_decision"]))
        if xs:
            axs[0].plot(xs, se, color=C[i + 2], marker=MK[i + 2], label=NAMES[m])
            axs[1].plot(xs, de, color=C[i + 2], marker=MK[i + 2], label=NAMES[m])
    axs[0].set_ylabel("stale errors (#calls)")
    axs[1].set_ylabel("decision errors (% of calls)")
    for ax in axs:
        ax.set_xlabel("evidence update rate $\\mu$ (edits/s)")
    axs[0].legend(fontsize=6.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "version.pdf"))


def fig_law():
    """Coverage-risk curves of legality predictors on the calibration pool (deployment statistic)."""
    z = np.load(os.path.join(RES, "ssn", "ssn_calib_scores.npz"))
    agree, cert = z["agree"], z["certified"]
    keys = [("ssn", "SSN (replay trace)"), ("trace-LR", "LogReg (trace stats)"), ("AHN_text", "AHN (text)"),
            ("decision-surprisal", "decision-field surprisal"), ("survival_no-learn", "$e^{-\\sum\\sigma}$ (no learning)"),
            ("cosine_text", "condition cosine")]
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.3))
    met = json.load(open(os.path.join(RES, "ssn", "ssn_metrics.json")))
    for i, (k, lab) in enumerate(keys):
        if k not in z:
            continue
        q = z[k]
        o = np.argsort(-q, kind="stable")
        a = agree[o]
        n = np.arange(1, len(a) + 1)
        risk = np.cumsum(1 - a) / n
        axs[0].plot(n / len(a), 100 * risk, color=C[i], label=lab, lw=1.3)
    axs[0].set_xlabel("coverage (fraction of candidate pairs committed)")
    axs[0].set_ylabel("decision error among commits (%)")
    axs[0].set_ylim(0, 30)
    axs[0].legend(fontsize=6.2, loc="upper left")
    eps = ["0.010", "0.020", "0.050", "0.100"]
    w = 0.13
    names = [("ssn", "SSN"), ("trace-LR", "LogReg"), ("AHN(text)", "AHN"), ("cosine(text)", "cosine")]
    for i, (k, lab) in enumerate(names):
        cov = [100 * met["predictors"][k]["deploy"][e]["coverage"] for e in eps]
        axs[1].bar(np.arange(4) + (i - 1.5) * w * 1.15, cov, w, color=C[[0, 1, 2, 5][i]], label=lab)
    axs[1].set_xticks(range(4), ["1%", "2%", "5%", "10%"])
    axs[1].set_xlabel("risk budget $\\varepsilon$")
    axs[1].set_ylabel("calibrated commit coverage (%)")
    axs[1].legend(fontsize=6.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "law.pdf"))


def fig_eps():
    """Realized risk vs budget across all end-to-end runs (guarantee check)."""
    pts = []
    for tag in sorted(os.listdir(RES)):
        for ds in ("hotpotqa", "musique"):
            for m, d in load(tag, ds).items():
                if m.startswith("unison-e") and "-" not in m[len("unison-e"):]:
                    e = float(m[len("unison-e"):])
                    err = d.get("dec_err_risk")
                    if err is not None and not np.isnan(err):
                        pts.append((e, 100 * err, d["risk_commits"], tag))
    if not pts:
        return
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    for e in sorted({p[0] for p in pts}):
        ys = [p[1] for p in pts if p[0] == e]
        ax.scatter(np.full(len(ys), 100 * e) + np.random.default_rng(0).normal(0, 0.12, len(ys)), ys, s=10,
                   color=C[0], alpha=0.7, edgecolor="white", linewidth=0.5)
    xx = np.linspace(0, 11, 2)
    ax.plot(xx, xx, color="#8a8984", ls="--", lw=1, label="budget $\\varepsilon$")
    ax.set_xlabel("risk budget $\\varepsilon$ (%)")
    ax.set_ylabel("realized decision error (%)")
    ax.legend(fontsize=6.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "eps.pdf"))


def fig_horizon():
    m = json.load(open(os.path.join(RES, "ssn", "law_meta.json")))
    hc = {int(k): v for k, v in m["horizon_curve"].items() if k.isdigit()}
    H = sorted(hc)
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.2))
    axs[0].plot(H, [hc[h]["auc"] for h in H], color=C[0], marker=MK[0], label="AUC")
    axs[0].plot(H, [hc[h]["deploy"]["0.050"]["coverage"] for h in H], color=C[1], marker=MK[1],
                label=r"coverage at $\varepsilon$=5%")
    axs[0].set_xscale("log", base=2)
    axs[0].set_xlabel("replay horizon H (tokens)")
    axs[0].legend(fontsize=6.5)
    axs[1].plot(H, [hc[h]["net"] for h in H], color=C[2], marker=MK[2])
    axs[1].axhline(0, color="#8a8984", lw=0.8)
    axs[1].axvline(m["horizon"], color="#8a8984", lw=0.8, ls="--")
    axs[1].set_xscale("log", base=2)
    axs[1].set_xlabel("replay horizon H (tokens)")
    axs[1].set_ylabel("net saving (decode-token units)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "horizon.pdf"))


def sweep_tab():
    rows = []
    ms = ["noshare", "exact+suffix", "semcache-0.90", "unison-e0.05"]
    spec = [("v3_model_qwen3-4b", "hotpotqa", "Qwen3-4B"), ("v3_model_qwen2.5-7b", "hotpotqa", "Qwen2.5-7B"),
            ("v3_zipf_0", "musique", "Zipf $s$=0"), ("v3_zipf_0.5", "musique", "Zipf $s$=0.5"),
            ("v3_zipf_1.1", "musique", "Zipf $s$=1.1"), ("v3_team_1", "hotpotqa", "team $K$=1"),
            ("v3_team_5", "hotpotqa", "team $K$=5"), ("v3_azure", "hotpotqa", "Azure trace")]
    for tag, ds, lab in spec:
        S = load(tag, ds)
        if not S:
            continue
        cells = [lab]
        for m in ms:
            d = S.get(m)
            cells.append("--" if d is None else f"{d['lat_mean']:.2f}/{d['lat_p99']:.1f}/{100 * (1 - d['fidelity_decision']):.1f}")
        rows.append(" & ".join(cells) + "\\\\")
    open(os.path.join(PAPER, "tab_sweep.tex"), "w").write("\n".join(rows) + "\n")


def numbers():
    out = []
    for ds, k in [("hotpotqa", "Hp"), ("musique", "Mu")]:
        S = load("v3_main", ds)
        for m, mk in [("noshare", "NS"), ("exact+suffix", "ES"), ("semcache-0.90", "Sem"), ("osc-e0.05", "TL"),
                      ("unison-e0.05", "U"), ("unison-e0.1", "Uten"), ("unison-e0.02", "Utwo"), ("semcrc-0.05", "Crc")]:
            d = S.get(m)
            if not d:
                continue
            err = d.get("dec_err_risk")
            for key, val in [("Lat", f"{d['lat_mean']:.2f}"), ("Pnn", f"{d['lat_p99']:.1f}"),
                             ("Vlat", f"{d['vlat_mean']:.3f}"), ("Busy", f"{d['engine_busy']:.1f}"),
                             ("Reuse", f"{d['reuse_calls']}"), ("Risk", "0.0" if err is None or np.isnan(err) else f"{100 * err:.1f}"),
                             ("Derr", f"{100 * (1 - d['fidelity_decision']):.1f}"), ("Stale", f"{d['stale_err']}"),
                             ("EM", f"{d['em']:.3f}"), ("Commits", f"{d['risk_commits']}")]:
                out.append(f"\\newcommand{{\\{k}{mk}{key}}}{{{val}}}")
    open(os.path.join(PAPER, "numbers_auto.tex"), "w").write("\n".join(out) + "\n")


if __name__ == "__main__":
    for fn in (main_table, abl_table, fig_load, fig_version, fig_law, fig_eps, fig_horizon, sweep_tab, numbers):
        try:
            fn()
            print("ok", fn.__name__)
        except Exception as e:  # keep going: partial results are still useful while the queue runs
            print("fail", fn.__name__, repr(e))


def work(d, base):
    return 100.0 * d["verify_steps"] / base["verify_steps"]


def abl_table2():
    lab = [("exact+suffix", "no replay, no law (Exact+Suffix)"), ("semcache-0.90", "similarity instead of replay"),
           ("osc-e0.05", "text law instead of replay"), ("unison-e0.05-nolaw", "replay, un-learned law"),
           ("unison-e0.05-latest", "replay + law, no class prior"), ("unison-e0.05", r"\textbf{Unison-0.05}")]
    rows = []
    for ds, name in [("hotpotqa", "HotpotQA"), ("musique", "MuSiQue")]:
        S = load("v3_abl", ds)
        b = S["noshare"]
        rows.append(rf"\multicolumn{{6}}{{l}}{{\textit{{{name}}}}}\\")
        for m, l in lab:
            d = S[m]
            r = d.get("dec_err_risk")
            r = 0.0 if r is None or np.isnan(r) else 100 * r
            rows.append(f"{l} & {work(d, b):.0f}\% & {d['reuse_calls']} & {r:.1f} & {100 * (1 - d['fidelity_decision']):.1f} & {d['stale_err']}\\\\")
        rows.append("\midrule")
    open(os.path.join(PAPER, "tab_abl.tex"), "w").write("\n".join(rows[:-1]) + "\n")


def sweep_table2():
    ms = ["exact+suffix", "semcache-0.90", "unison-e0.05"]
    spec = [("v3_load_r0.6", "hotpotqa", "load 0.6 q/s"), ("v3_load_r1.8", "hotpotqa", "load 1.8 q/s"),
            ("v3_load_r2.4", "hotpotqa", "load 2.4 q/s"), ("v3_azure", "hotpotqa", "Azure trace"),
            ("v3_zipf_0", "musique", "Zipf $s$=0"), ("v3_zipf_0.5", "musique", "Zipf $s$=0.5"),
            ("v3_zipf_1.1", "musique", "Zipf $s$=1.1"), ("v3_team_1", "hotpotqa", "team $K$=1"),
            ("v3_team_5", "hotpotqa", "team $K$=5"), ("v3_model_qwen3-4b", "hotpotqa", "Qwen3-4B"),
            ("v3_model_qwen2.5-7b", "hotpotqa", "Qwen2.5-7B")]
    rows = []
    for tag, ds, lab in spec:
        S = load(tag, ds)
        b = S["noshare"]
        rows.append(lab + "".join(f" & {work(S[m], b):.0f}\% / {100 * (1 - S[m]['fidelity_decision']):.1f}" for m in ms) + "\\\\")
    open(os.path.join(PAPER, "tab_sweep.tex"), "w").write("\n".join(rows) + "\n")


def fig_version2():
    mus = [0, 0.5, 1.0, 2.0]
    ms = [("semcache-0.90", "SemCache-0.90"), ("osc-e0.05", "TextLaw-0.05"), ("unison-e0.05", "Unison-0.05")]
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.2))
    for i, (m, l) in enumerate(ms):
        S = [load(f"v3_ver_mu{mu}", "hotpotqa")[m] for mu in mus]
        axs[0].plot(mus, [d["stale_reuse"] for d in S], color=C[[1, 3, 0][i]], marker=MK[i], label=l)
        axs[1].plot(mus, [d["stale_err"] for d in S], color=C[[1, 3, 0][i]], marker=MK[i], label=l)
    axs[0].set_ylabel("results reused across versions")
    axs[1].set_ylabel("stale decision errors")
    for ax in axs:
        ax.set_xlabel("evidence edit rate $\mu$ (edits/s)")
    axs[0].legend(fontsize=6.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "version.pdf"))


if __name__ == "__main__":
    for fn in (abl_table2, sweep_table2, fig_version2):
        try:
            fn()
            print("ok", fn.__name__)
        except Exception as e:
            print("fail", fn.__name__, repr(e))


# ------------------------------------------------------------------ revision (v7): seeds + intervals
from scipy.stats import beta as _beta

V7M = [("noshare", "NoShare (vLLM+APC)"), ("exact+suffix", "Exact+Suffix"), ("semcache-0.90", "SemCache-0.90"),
       ("semcache-0.95", "SemCache-0.95"), ("semcrc-0.05", "SemCRC-0.05"), ("osc-e0.05", "TextLaw-0.05"),
       ("unison-shadow", "Draft, no commit"), ("unison-e0.02", "Unison-0.02"), ("unison-e0.05", "Unison-0.05"),
       ("unison-e0.1", "Unison-0.10")]


def _cp(k, n):
    if n == 0:
        return 0.0, 0.0
    lo = _beta.ppf(0.025, k, n - k + 1) if k > 0 else 0.0
    hi = _beta.ppf(0.975, k + 1, n - k) if k < n else 1.0
    return 100 * lo, 100 * hi


def _seeds(ds):
    return [load(f"v7_main_s{s}", ds) for s in range(3) if load(f"v7_main_s{s}", ds)]


def main_v7():
    rows = []
    for ds, name in [("hotpotqa", "HotpotQA"), ("musique", "MuSiQue")]:
        R = _seeds(ds)
        rows.append(rf"\multicolumn{{9}}{{l}}{{\textit{{{name}}} ({len(R)} seeds)}}\\")
        for m, l in V7M:
            D = [r[m] for r in R if m in r]
            B = [r["noshare"] for r in R if m in r]
            lat = np.array([d["lat_mean"] for d in D])
            p99 = np.array([d["lat_p99"] for d in D])
            wk = np.mean([100 * d["verify_steps"] / b["verify_steps"] for d, b in zip(D, B)])
            k = sum(round(d.get("dec_err_risk", 0) * d["risk_commits"]) if d.get("dec_err_risk") == d.get("dec_err_risk") and d.get("dec_err_risk") is not None else 0 for d in D)
            n = sum(d["risk_commits"] for d in D)
            lo, hi = _cp(k, n)
            risk = f"{100 * k / n:.1f} [{lo:.1f},{hi:.1f}]" if n else "0.0 (no commits)"
            de = np.mean([100 * (1 - d["fidelity_decision"]) for d in D])
            st = sum(d["stale_err"] for d in D)
            sr = sum(d["stale_reuse"] for d in D)
            em = np.mean([d["em"] for d in D])
            rows.append(f"{l} & {lat.mean():.2f}$\pm${lat.std():.2f} & {p99.mean():.1f}$\pm${p99.std():.1f} & {wk:.0f}\% & "
                        f"{n} & {risk} & {de:.1f} & {st}/{sr} & {em:.3f}\\\\")
        rows.append("\midrule")
    open(os.path.join(PAPER, "tab_main.tex"), "w").write("\n".join(rows[:-1]) + "\n")


def fig_eps_v7():
    pts = []
    for s in range(3):
        for ds in ("hotpotqa", "musique"):
            for m, d in load(f"v7_main_s{s}", ds).items():
                if m in ("unison-e0.02", "unison-e0.05", "unison-e0.1") and d["risk_commits"]:
                    e = float(m.split("-e")[1])
                    k = round(d["dec_err_risk"] * d["risk_commits"])
                    lo, hi = _cp(k, d["risk_commits"])
                    pts.append((e, 100 * d["dec_err_risk"], lo, hi, ds))
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    rng = np.random.default_rng(0)
    for e, r, lo, hi, ds in pts:
        x = 100 * e + rng.normal(0, 0.15) + (0.25 if ds == "musique" else -0.25)
        c = C[0] if ds == "hotpotqa" else C[1]
        ax.errorbar(x, r, yerr=[[r - lo], [hi - r]], fmt="o", color=c, ms=3.5, lw=0.8, capsize=0)
    xx = np.linspace(0, 11, 2)
    ax.plot(xx, xx, color="#8a8984", ls="--", lw=1)
    ax.scatter([], [], color=C[0], s=10, label="HotpotQA")
    ax.scatter([], [], color=C[1], s=10, label="MuSiQue")
    ax.set_xlabel(r"risk budget $\varepsilon$ (%)")
    ax.set_ylabel("realized commit risk (%)")
    ax.legend(fontsize=6.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "eps.pdf"))


def stress_table():
    s = json.load(open(os.path.join(RES, "ssn", "stress.json")))
    lab = [("law", "Anticipatory law (Unison)"), ("gru-clf", "same-size GRU classifier"),
           ("law-noR", "law without $R_H$"), ("prefix", "16 accepted tokens $\Rightarrow$ commit")]
    rows = []
    for k, l in lab:
        p = s["predictors"][k]
        e = p["eps0.05"]
        ci = f"[{100 * e['ci'][0]:.1f},{100 * e['ci'][1]:.1f}]" if e["n"] else ""
        rk = f"{100 * e['risk']:.1f} {ci}" if e["n"] else "0.0 (no commits)"
        a = e["subgroups"]["answer_after_40"]
        ak = f"{100 * a['risk']:.1f} ({a['err']}/{a['n']})" if a["n"] else "0.0 (no commits)"
        rows.append(f"{l} & {p['auc']:.3f} & {100 * e['coverage']:.1f} & {rk} & {ak}\\\\")
    open(os.path.join(PAPER, "tab_stress.tex"), "w").write("\n".join(rows) + "\n")


if __name__ == "__main__":
    for fn in (main_v7, fig_eps_v7, stress_table):
        try:
            fn()
            print("ok", fn.__name__)
        except Exception as e:
            print("fail", fn.__name__, repr(e))


# ------------------------------------------------------------------ second revision (v8)
def _cluster(name):
    p = os.path.join(RES, "cluster", name)
    return json.load(open(p)) if os.path.exists(p) else {}


def main_v8():
    """Main table with risk among NON-EXACT commits, team-clustered 95% CI (pooled over 3 seeds)."""
    cl = _cluster("cluster_v7.json")
    rows = []
    for ds, name in [("hotpotqa", "HotpotQA"), ("musique", "MuSiQue")]:
        R = _seeds(ds)
        rows.append(rf"\multicolumn{{9}}{{l}}{{\textit{{{name}}} (3 seeds)}}\\")
        for m, l in V7M:
            D = [r[m] for r in R if m in r]
            B = [r["noshare"] for r in R if m in r]
            lat = np.array([d["lat_mean"] for d in D])
            p99 = np.array([d["lat_p99"] for d in D])
            wk = np.mean([100 * d["verify_steps"] / b["verify_steps"] for d, b in zip(D, B)])
            c = cl.get(ds, {}).get("risk", {}).get(m)
            if c and c["n_commits"]:
                n = c["n_commits"]
                risk = f"{100 * c['risk']:.1f} [{100 * c['ci95'][0]:.1f},{100 * c['ci95'][1]:.1f}]"
            else:
                n, risk = 0, "0.0 (exact only)"
            de = np.mean([100 * (1 - d["fidelity_decision"]) for d in D])
            st = sum(d["stale_err"] for d in D)
            sr = sum(d["stale_reuse"] for d in D)
            em = np.mean([d["em"] for d in D])
            rows.append(f"{l} & {lat.mean():.2f}$\pm${lat.std():.2f} & {p99.mean():.1f}$\pm${p99.std():.1f} & {wk:.0f}\% & "
                        f"{n} & {risk} & {de:.1f} & {st}/{sr} & {em:.3f}\\\\")
        rows.append("\midrule")
    open(os.path.join(PAPER, "tab_main.tex"), "w").write("\n".join(rows[:-1]) + "\n")


def fair_v8():
    cl = _cluster("cluster_v8.json")
    lab = [("exact+suffix", "Exact+Suffix"), ("semcache+S-0.90", "SemCache+S-0.90"), ("semcache+S-0.95", "SemCache+S-0.95"),
           ("semcacheV+S-0.90", "SemCache-V+S-0.90"), ("semcrcS-0.05", "SemCRC+S-0.05"), ("unison-e0.05", r"\textbf{Unison-0.05}")]
    rows = []
    for ds, name in [("hotpotqa", "HotpotQA"), ("musique", "MuSiQue")]:
        R = [load(f"v8_fair_s{s}", ds) for s in range(3) if load(f"v8_fair_s{s}", ds)]
        rows.append(rf"\multicolumn{{7}}{{l}}{{\textit{{{name}}} ({len(R)} seeds)}}\\")
        for m, l in lab:
            D = [r[m] for r in R]
            B = [r["noshare"] for r in R]
            wk = np.mean([100 * d["verify_steps"] / b["verify_steps"] for d, b in zip(D, B)])
            dt = np.mean([100 * d["draft_tokens"] / max(1, e["draft_tokens"]) for d, e in zip(D, [r["exact+suffix"] for r in R])])
            c = cl.get(ds, {}).get("risk", {}).get(m)
            if c and c["n_commits"]:
                rk = f"{100 * c['risk']:.1f} ({100 * c['ub95_one_sided']:.1f})"
                n = c["n_commits"]
            else:
                rk, n = "0.0 (exact only)", 0
            de = np.mean([100 * (1 - d["fidelity_decision"]) for d in D])
            st = sum(d["stale_err"] for d in D)
            rows.append(f"{l} & {wk:.0f}\% & {dt:.0f}\% & {n} & {rk} & {de:.1f} & {st}\\\\")
        rows.append("\midrule")
    open(os.path.join(PAPER, "tab_fair.tex"), "w").write("\n".join(rows[:-1]) + "\n")


if __name__ == "__main__":
    for fn in (main_v8, fair_v8):
        try:
            fn()
            print("ok", fn.__name__)
        except Exception as e:
            print("fail", fn.__name__, repr(e))
