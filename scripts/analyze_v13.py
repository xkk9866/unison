"""Tables and macros for the v13 paper (published baselines), from results_v2/v1[23]_*/<ds>/{summary.json,*.queries.json}.
usage: python analyze_v12.py <results_dir> <paper_dir>"""
import json
import os
import sys

import numpy as np
from scipy.stats import beta

R, P = sys.argv[1], sys.argv[2]
DS = {"hotpotqa": "HotpotQA", "musique": "MuSiQue"}
DSK = {"hotpotqa": "Ho", "musique": "Mu"}
MAIN = {"noshare@cot": "vLLM (APC)~\cite{kwon2023vllm}", "pld@cot": "PLD~\cite{saxena2023pld}",
        "suffix@cot": "SuffixDecoding~\cite{oliaro2025suffix}", "gptcache@cot": "GPTCache~\cite{bang2023gptcache}",
        "vcache-official@cot": "vCache~\cite{schroeder2025vcache}", "unison-canon": "\textbf{Unison}"}
BASE = [m for m in MAIN if m != "unison-canon"]
REF = "noshare@cot"
ABL = {"exact+suffix": "Exact coalescing + SuffixDecoding", "unison": "Certification, full note",
       "unison-sbd": "Own boundary, deferred own rationale", "unison-cont": "+ certification",
       "unison-sbdcanon": "+ decision-keyed provenance", "unison-canonnofg": "Unison w/o foreground-first",
       "unison-canon": "\textbf{Unison} (certified keying)", "unison-lazy": "Decision only (no provenance)"}
SWEEP_M = {"suffix@cot": "SuffixDecoding", "gptcache@cot": "GPTCache", "vcache-official@cot": "vCache",
           "unison-canon": "\textbf{Unison}"}
KEY = {"noshare@cot": "Vllm", "pld@cot": "Pld", "suffix@cot": "Suffix", "gptcache@cot": "Gptcache",
       "vcache-official@cot": "Vcache", "unison-canon": "Unison", "exact+suffix": "ExactSuffix",
       "unison": "Certonly", "unison-sbd": "Sbd", "unison-cont": "Cont", "unison-sbdcanon": "Sbdcanon",
       "unison-canonnofg": "Nofg", "unison-lazy": "Lazy"}
M = {}
NUMW = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]


def macro(name, val):
    M[name.replace("0", "Zero").replace("1", "One").replace("2", "Two")] = val


def load(tag, ds):
    p = os.path.join(R, tag, ds, "summary.json")
    return {r["method"]: r for r in json.load(open(p))} if os.path.exists(p) else None


def queries(tag, ds, m):
    p = os.path.join(R, tag, ds, f"{m}.queries.json")
    return {q["qi"]: q["em"] for q in json.load(open(p))} if os.path.exists(p) else None


def MAINTAG(s, ds):
    """Seed 2 was rerun (v12_main_s2r) because the original overlapped a co-tenant on the GPU."""
    return f"v13_main_s{s}"


def gpu(r):
    return r["gen_tokens"] + r["draft_tokens"]


def cp_hi(k, n):
    return 100 * (beta.ppf(0.975, k + 1, n - k) if k < n else 1.0) if n else 0.0


def nonid(r):  # reuses that are not identical-prompt coalescing
    return r.get("src_dcert", 0) + r.get("src_sem", 0) + r.get("src_risk", 0)


def nonid_err(r):
    return r.get("nerr_dcert", 0) + r.get("nerr_sem", 0) + r.get("nerr_risk", 0)


def f2(x):
    return f"{x:.2f}"


def runs_main(ds):
    out = []
    for s in range(3):
        x = load(MAINTAG(s, ds), ds)
        if x:
            out.append(x)
    return out


def main_table():
    rows = []
    for ds in DS:
        runs = runs_main(ds)
        if not runs:
            continue
        k = DSK[ds]
        g0 = np.mean([gpu(x[REF]) for x in runs])
        b0 = np.mean([x[REF]["engine_busy"] for x in runs])
        rows.append(f"\\midrule\n\\multicolumn{{10}}{{l}}{{\\textit{{{DS[ds]}}} ({len(runs)} seeds, 200 queries each)}}\\\\")
        for m, lab in MAIN.items():
            if m not in runs[0]:
                continue
            rr = [x[m] for x in runs]
            lat = [r["lat_mean"] for r in rr]
            p99 = np.mean([r["lat_p99"] for r in rr])
            gt = 100 * np.mean([gpu(r) for r in rr]) / g0
            bz = 100 * np.mean([r["engine_busy"] for r in rr]) / b0
            nre = sum(r["n_reused"] for r in rr)
            ner = sum(r["nerr_reused"] for r in rr)
            nc = sum(nonid(r) for r in rr)
            nce = sum(nonid_err(r) for r in rr)
            st = sum(r["stale_reuse"] for r in rr)
            sd = sum(r.get("stale_decisive", 0) for r in rr)
            se = sum(r["stale_err"] for r in rr)
            val = 100 * np.mean([r["support_valid"] for r in rr])
            em = np.mean([r["em"] for r in rr])
            err = f"{ner}/{nre} [{cp_hi(ner, nre):.2f}]" if nre else "0/0 [0.00]"
            cer = f"{nce}/{nc}" if nc else "0/0"
            rows.append(f"{lab} & {np.mean(lat):.2f}$\\pm${np.std(lat):.2f} & {p99:.1f} & {gt:.0f}\\% & {bz:.0f}\\% & "
                        f"{err} & {cer} & {se}/{st} ({sd}) & {val:.1f} & {em:.3f}\\\\")
            km = KEY[m]
            macro(f"{k}{km}Lat", f2(np.mean(lat)))
            macro(f"{k}{km}Pnn", f"{p99:.1f}")
            macro(f"{k}{km}Gpu", f"{gt:.0f}")
            macro(f"{k}{km}Busy", f"{bz:.0f}")
            macro(f"{k}{km}Em", f"{em:.3f}")
            macro(f"{k}{km}Reuse", f"{nre:,}")
            macro(f"{k}{km}ErrK", f"{ner}")
            macro(f"{k}{km}ErrHi", f"{cp_hi(ner, nre):.2f}")
            macro(f"{k}{km}Cert", f"{nc:,}")
            macro(f"{k}{km}CertErr", f"{nce}")
            macro(f"{k}{km}CertErrPct", f"{100 * nce / max(nc, 1):.1f}")
            macro(f"{k}{km}Stale", f"{st}")
            macro(f"{k}{km}StaleDec", f"{sd}")
            macro(f"{k}{km}StaleErr", f"{se}")
            macro(f"{k}{km}Valid", f"{val:.1f}")
        u = [x["unison-canon"] for x in runs]
        best = min(BASE, key=lambda m: np.mean([x[m]["lat_mean"] for x in runs]))
        macro(f"{k}BestBase", MAIN[best].split("~")[0])
        for ref, nm in [(best, "Best")] + [(m, KEY[m]) for m in BASE]:
            red = [100 * (1 - a["lat_mean"] / x[ref]["lat_mean"]) for a, x in zip(u, runs)]
            mr = 100 * (1 - np.mean([a["lat_mean"] for a in u]) / np.mean([x[ref]["lat_mean"] for x in runs]))
            macro(f"{k}LatRed{nm}", f"{mr:.0f}")
            macro(f"{k}LatRed{nm}Min", f"{min(red):.0f}")
            macro(f"{k}LatRed{nm}Max", f"{max(red):.0f}")
            pr = 100 * (1 - np.mean([a["lat_p99"] for a in u]) / np.mean([x[ref]["lat_p99"] for x in runs]))
            macro(f"{k}PnnRed{nm}", f"{pr:.0f}")
        # nondeterminism floor (two independent replays, different batch composition)
        fd = sum(x[REF].get("nd_decision_diff", 0) for x in runs)
        fn = sum(x[REF].get("nd_prompts", 0) for x in runs)
        ft = sum(x[REF].get("nd_token_diff", 0) for x in runs)
        macro(f"{k}FloorK", f"{fd}")
        macro(f"{k}FloorN", f"{fn:,}")
        macro(f"{k}FloorTok", f"{ft}")
        macro(f"{k}FloorHi", f"{cp_hi(fd, fn):.2f}")
        macro(f"{k}FloorPct", f"{100 * fd / max(fn, 1):.2f}")
    open(os.path.join(P, "tab_main.tex"), "w").write("\n".join(rows) + "\n")


def bg_table():
    """Background accounting of Unison's provenance (all of it is completed before a run ends)."""
    for ds in DS:
        runs = runs_main(ds)
        if not runs:
            continue
        k = DSK[ds]
        u = [x["unison-canon"] for x in runs]
        for f, nm, fmt in (("prov_requests", "ProvReq", "{:.0f}"), ("prov_engine", "ProvEng", "{:.0f}"),
                           ("prov_lag_mean", "ProvLag", "{:.2f}"), ("prov_lag_p99", "ProvLagP", "{:.1f}"),
                           ("drain_s", "Drain", "{:.2f}")):
            macro(f"{k}Unison{nm}", fmt.format(np.mean([r.get(f, 0) for r in u])))
        macro(f"{k}UnisonProvTok", f"{np.mean([r.get('prov_tokens', 0) for r in u]) / 1000:.1f}k")
        macro(f"{k}ProvShare", f"{100 * (1 - np.mean([r['prov_engine'] for r in u]) / max(np.mean([r['prov_requests'] for r in u]), 1)):.0f}")
        macro(f"{k}UnisonBgPct", f"{100 * np.mean([r['prov_tokens'] / max(r['gen_tokens'], 1) for r in u]):.0f}")


def paired_em():
    """Paired bootstrap of EM differences per query (pooled seeds), Unison vs. references."""
    rng = np.random.default_rng(0)
    rows = []
    for ds in DS:
        k = DSK[ds]
        for ref, nm in [(m, KEY[m]) for m in BASE]:
            d = []
            for s in range(3):
                a, b = queries(MAINTAG(s, ds), ds, "unison-canon"), queries(MAINTAG(s, ds), ds, ref)
                if a and b:
                    d += [a[q] - b[q] for q in a if q in b]
            if not d:
                continue
            d = np.asarray(d)
            bs = [rng.choice(d, len(d)).mean() for _ in range(5000)]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            macro(f"{k}EmDiff{nm}", f"{100 * d.mean():+.1f}")
            macro(f"{k}EmDiff{nm}Lo", f"{100 * lo:+.1f}")
            macro(f"{k}EmDiff{nm}Hi", f"{100 * hi:+.1f}")
            rows.append((ds, ref, len(d), 100 * d.mean(), 100 * lo, 100 * hi))
    return rows


def abl_table():
    rows = []
    for ds in DS:
        runs = [x for x in (load(t, ds) for t in ("v13_abl_s0r", "v12_abl_s1")) if x]
        if not runs:
            continue
        k = DSK[ds]
        g0 = np.mean([gpu(x["exact+suffix"]) for x in runs])
        b0 = np.mean([x["exact+suffix"]["engine_busy"] for x in runs])
        rows.append(f"\\midrule\n\\multicolumn{{8}}{{l}}{{\\textit{{{DS[ds]}}} ({len(runs)} seeds)}}\\\\")
        for m, lab in ABL.items():
            if m not in runs[0]:
                continue
            rr = [x[m] for x in runs]
            lat = np.mean([r["lat_mean"] for r in rr])
            p99 = np.mean([r["lat_p99"] for r in rr])
            gt = 100 * np.mean([gpu(r) for r in rr]) / g0
            bz = 100 * np.mean([r["engine_busy"] for r in rr]) / b0
            nre = sum(r["n_reused"] for r in rr)
            ner = sum(r["nerr_reused"] for r in rr)
            em = np.mean([r["em"] for r in rr])
            rows.append(f"{lab} & {lat:.2f} & {p99:.1f} & {gt:.0f}\\% & {bz:.0f}\\% & {ner}/{nre} & "
                        f"{sum(nonid(r) for r in rr)} & {em:.3f}\\\\")
            macro(f"{k}Abl{KEY[m]}Lat", f2(lat))
            macro(f"{k}Abl{KEY[m]}Pnn", f"{p99:.1f}")
            macro(f"{k}Abl{KEY[m]}Gpu", f"{gt:.0f}")
            macro(f"{k}Abl{KEY[m]}Busy", f"{bz:.0f}")
    open(os.path.join(P, "tab_abl.tex"), "w").write("\n".join(rows) + "\n")


SWEEPS = [("v13_load_r0.6", "hotpotqa", "load 0.6/s"), ("v13_load_r1.8", "hotpotqa", "load 1.8/s"),
          ("v13_load_r2.4", "hotpotqa", "load 2.4/s"), ("v13_muload_r1.2", "musique", "MuSiQue 1.2/s"),
          ("v13_ver_mu0.5", "hotpotqa", "$\\mu{=}0.5$"), ("v13_ver_mu2.0", "hotpotqa", "$\\mu{=}2.0$"),
          ("v13_zipf_0", "hotpotqa", "Zipf 0"), ("v13_zipf_1.1", "hotpotqa", "Zipf 1.1"),
          ("v13_team_1", "hotpotqa", "team 1"), ("v13_team_5", "hotpotqa", "team 5"),
          ("v13_azure", "hotpotqa", "Azure trace"), ("v13_lowred", "hotpotqa", "low redundancy")]


def sweep_table(spec, fname, prefix):
    rows, wins, red, xv, xd, xe, n = [], 0, [], 0, 0, 0, 0
    for tag, ds, lab in spec:
        x = load(tag, ds)
        if not x:
            continue
        n += 1
        g0 = gpu(x["suffix@cot"])
        first = True
        for m, ml in SWEEP_M.items():
            if m not in x:
                continue
            r = x[m]
            rows.append(f"{lab if first else ''} & {ml} & {r['lat_mean']:.2f} & {r['lat_p99']:.1f} & "
                        f"{100 * gpu(r) / g0:.0f}\\% & {r['nerr_reused']}/{r['n_reused']} & "
                        f"{r['stale_err']}/{r['stale_reuse']} & {r['em']:.3f}\\\\")
            first = False
        rows.append("\\midrule")
        u = x["unison-canon"]
        best = min(x[m]["lat_mean"] for m in SWEEP_M if m in x and m != "unison-canon")
        wins += u["lat_mean"] < best
        red.append(100 * (1 - u["lat_mean"] / best))
        xv += u["stale_reuse"]
        xd += u.get("stale_decisive", 0)
        xe += u["stale_err"]
    if rows and rows[-1] == "\\midrule":
        rows.pop()
    open(os.path.join(P, fname), "w").write("\n".join(rows) + "\n")
    if n:
        macro(f"{prefix}N", f"{n}")
        macro(f"{prefix}Wins", f"{wins}")
        macro(f"{prefix}RedMin", f"{min(red):.0f}")
        macro(f"{prefix}RedMax", f"{max(red):.0f}")
        macro(f"{prefix}Xver", f"{xv}")
        macro(f"{prefix}XverDec", f"{xd}")
        macro(f"{prefix}XverErr", f"{xe}")


def safety():
    """Pooled cross-version reuse safety for Unison over all v12 runs."""
    xv = xd = xe = xde = 0
    for d in os.listdir(R):
        if not d.startswith("v13_") and d != "v12_abl_s1":
            continue
        for ds in DS:
            x = load(d, ds)
            if x and "unison-canon" in x:
                u = x["unison-canon"]
                xv += u.get("stale_dcert", 0)
                xd += u.get("stale_decisive_dcert", 0)
                xe += u["stale_err"]
                xde += u.get("stale_decisive_err", 0)
    macro("AllXverCert", f"{xv}")
    macro("AllXverCertDec", f"{xd}")
    macro("AllXverErr", f"{xe}")
    macro("AllXverDecErr", f"{xde}")
    macro("AllXverHi", f"{cp_hi(0, max(xd, 1)):.1f}")


if __name__ == "__main__":
    main_table()
    bg_table()
    pe = paired_em()
    abl_table()
    sweep_table(SWEEPS, "tab_sweep.tex", "Sweep")
    sweep_table([("v13_model_qwen3-4b", "hotpotqa", "Qwen3-4B"), ("v13_model_qwen2.5-7b", "hotpotqa", "Qwen2.5-7B")],
                "tab_models.tex", "Model")
    safety()
    with open(os.path.join(P, "numbers_auto.tex"), "w") as f:
        for k, v in M.items():
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    for r in pe:
        print("pairedEM", r)
    print(len(M), "macros")
