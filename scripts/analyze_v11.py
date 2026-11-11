"""Tables and macros for the decision-boundary certification paper, from results_v2/v11_*/<ds>/summary.json.
usage: python analyze_v11.py <results_dir> <paper_dir>"""
import json
import os
import sys

import numpy as np
from scipy.stats import beta

R, P = sys.argv[1], sys.argv[2]
DS = {"hotpotqa": "HotpotQA", "musique": "MuSiQue"}
LABEL = {"noshare@cot": "NoShare, answer-last", "noshare": "NoShare (vLLM+APC)",
         "unison-lazy": "Decision-only (no reasoning)", "unison-bstop": "Decide-then-Lookup",
         "unison@dfirst": "decision-first, no pointer", "unison-defer@dfirst": "Unison, decision-first (no pointer)", "unison-defer@gdf": "Unison, pointer-first", "unison-defer": "\\textbf{Unison}",
         "exact+suffix@cot": "Exact+Suffix, answer-last", "exact+suffix": "Exact+Suffix",
         "semcache+S-0.90": "SemCache+S-0.90", "semcache+S-0.95": "SemCache+S-0.95",
         "semcacheV+S-0.90": "SemCacheV+S-0.90", "semcrcS-0.05": "SemCRC+S-0.05",
         "law-e0.05@cot": "PrefixLaw-0.05, answer-last", "unison": "Certify only (no defer)",
         "unison-nodraft": "w/o trie drafting", "unison-single": "single candidate",
         "unison-note": "certify whole note", "unison@cot": "answer-last format"}
LABEL["unison-defer"] = LABEL.pop("unison-defer")  # headline method last
M = {}
KEY = {"noshare@cot": "NoshareCot", "noshare": "Noshare", "exact+suffix@cot": "ExactSuffixCot",
       "exact+suffix": "ExactSuffix", "semcache+S-0.90": "SemcacheS", "semcache+S-0.95": "SemcacheSHi",
       "semcacheV+S-0.90": "SemcacheVS", "semcrcS-0.05": "Semcrc", "law-e0.05@cot": "Prefixlaw", "unison-lazy": "Lazy", "unison-bstop": "Bstop",
       "unison": "Trie", "unison-defer": "Unison", "unison-defer@dfirst": "UnisonDf", "unison-defer@gdf": "UnisonGdf"}


def load(tag, ds):
    p = os.path.join(R, tag, ds, "summary.json")
    return {r["method"]: r for r in json.load(open(p))} if os.path.exists(p) else None


def gpu(r):
    return r["gen_tokens"] + r["draft_tokens"]


def cp(k, n):
    lo = beta.ppf(0.025, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(0.975, k + 1, n - k) if k < n else 1.0
    return 100 * lo, 100 * hi


REUSE_SRC = ("certified", "exact", "replay-cert", "dcert", "sem", "risk")


def nonexact_err(r):
    """(decision errors, count) over every reused call (coalesced, certified or similarity/risk reuse)."""
    n = sum(r.get(f"src_{s}", 0) for s in REUSE_SRC)
    if n == 0:
        return 0, 0
    e = 0.0
    for s in ("certified", "exact", "replay-cert", "dcert", "sem"):
        v = r.get(f"dec_err_{s}")
        if v is not None and v == v:
            e += v * r.get(f"src_{s}", 0)
    if r.get("src_risk", 0) and r.get("dec_err_risk") == r.get("dec_err_risk"):
        e += r["dec_err_risk"] * r["src_risk"]
    return int(round(e)), n


def macro(name, val):
    M[name] = val


def main_table():
    rows = []
    for ds in DS:
        runs = [load("v11_main_s2r" if (s == 2 and ds == "hotpotqa" and load("v11_main_s2r", ds)) else f"v11_main_s{s}", ds) for s in range(3)]
        runs = [x for x in runs if x]
        if not runs:
            continue
        ns = np.mean([gpu(x["exact+suffix"]) for x in runs])
        nsv = np.mean([x["noshare"]["verify_steps"] for x in runs])
        nb = np.mean([x["exact+suffix"]["engine_busy"] for x in runs])
        rows.append(f"\\midrule\n\\multicolumn{{12}}{{l}}{{\\textit{{{DS[ds]}}} ({len(runs)} seeds, 200 queries each)}}\\\\")
        for m in LABEL:
            if m not in runs[0]:
                continue
            rr = [x[m] for x in runs]
            lat = [r["lat_mean"] for r in rr]
            p99 = [r["lat_p99"] for r in rr]
            k = sum(r.get("nerr_reused", 0) for r in rr)
            n = sum(r.get("n_reused", 0) for r in rr)
            kp = sum(r.get("nerr_reused_prov", 0) for r in rr)
            nc = sum(r.get("n_dcert", 0) + r.get("n_sem", 0) + r.get("n_risk", 0) + r.get("n_replay-cert", 0) for r in rr)
            kc = sum(r.get("nerr_dcert", 0) + r.get("nerr_sem", 0) + r.get("nerr_risk", 0) + r.get("nerr_replay-cert", 0) for r in rr)
            err = f"{k}/{n} [{cp(k, n)[1]:.2f}]" if n else "0/0 [no reuse]"
            st = sum(r.get("stale_err", 0) for r in rr)
            sr = sum(r.get("stale_reuse", 0) for r in rr)
            busy = 100 * np.mean([r["engine_busy"] for r in rr]) / nb
            rows.append(f"{LABEL[m]} & {np.mean(lat):.2f}$\\pm${np.std(lat):.2f} & {np.mean(p99):.1f} & "
                        f"{100 * np.mean([gpu(r) for r in rr]) / ns:.0f}\\% & {busy:.0f}\\% & {n} & {nc} & {err} & {kp} & "
                        f"{st}/{sr} & {100 * np.mean([r.get('support_valid', 1) for r in rr]):.1f} & "
                        f"{np.mean([r['em'] for r in rr]):.3f}\\\\")
            macro(f"{ds[:2].capitalize()}{KEY[m]}Cert", f"{nc}")
            macro(f"{ds[:2].capitalize()}{KEY[m]}CertErr", f"{kc}")
            macro(f"{ds[:2].capitalize()}{KEY[m]}CertHi", f"{cp(kc, nc)[1]:.2f}" if nc else "0.00")
            macro(f"{ds[:2].capitalize()}{KEY[m]}ProvErr", f"{kp}")
            macro(f"{ds[:2].capitalize()}{KEY[m]}Valid", f"{100 * np.mean([r.get('support_valid', 1) for r in rr]):.1f}")
            macro(f"{ds[:2].capitalize()}{KEY[m]}BusyRel", f"{busy:.0f}")
            key = ds[:2].capitalize() + KEY[m]
            macro(f"{key}Lat", f"{np.mean(lat):.2f}")
            macro(f"{key}Pnn", f"{np.mean(p99):.1f}")
            macro(f"{key}Gpu", f"{100 * np.mean([gpu(r) for r in rr]) / ns:.0f}")
            macro(f"{key}Steps", f"{100 * np.mean([r['verify_steps'] for r in rr]) / nsv:.0f}")
            macro(f"{key}Em", f"{np.mean([r['em'] for r in rr]):.3f}")
            macro(f"{key}Reuse", f"{n}")
            macro(f"{key}Err", f"{100 * k / n:.2f}" if n else "0.00")
            macro(f"{key}ErrK", f"{k}")
            macro(f"{key}ErrHi", f"{cp(k, n)[1]:.2f}" if n else "0.00")
            macro(f"{key}Busy", f"{np.mean([r['engine_busy'] for r in rr]):.1f}")
            macro(f"{key}Stale", f"{st}")
            macro(f"{key}StaleN", f"{sr}")
            macro(f"{key}DecErr", f"{100 * np.mean([r.get('dec_err_total', 0) for r in rr]):.2f}")
        # paired latency reduction of Unison vs Exact+Suffix per seed
        d = [1 - x["unison-defer"]["lat_mean"] / x["exact+suffix"]["lat_mean"] for x in runs]
        d99 = [1 - x["unison-defer"]["lat_p99"] / x["exact+suffix"]["lat_p99"] for x in runs]
        g = [1 - gpu(x["unison-defer"]) / gpu(x["exact+suffix"]) for x in runs]
        tag = ds[:2].capitalize()
        db = [1 - x["unison-defer"]["lat_mean"] / x["unison-bstop"]["lat_mean"] for x in runs]
        gb = [1 - gpu(x["unison-defer"]) / gpu(x["unison-bstop"]) for x in runs]
        macro(f"{tag}LatRedB", f"{100 * np.mean(db):.0f}")
        macro(f"{tag}LatRedBMin", f"{100 * min(db):.0f}")
        macro(f"{tag}LatRedBMax", f"{100 * max(db):.0f}")
        macro(f"{tag}GpuRedB", f"{100 * np.mean(gb):.0f}")
        dc = [1 - x["unison-defer"]["lat_mean"] / x["exact+suffix@cot"]["lat_mean"] for x in runs]
        macro(f"{tag}LatRedCot", f"{100 * np.mean(dc):.0f}")
        macro(f"{tag}LatRed", f"{100 * np.mean(d):.0f}")
        macro(f"{tag}LatRedMin", f"{100 * min(d):.0f}")
        macro(f"{tag}LatRedMax", f"{100 * max(d):.0f}")
        macro(f"{tag}PnnRed", f"{100 * np.mean(d99):.0f}")
        macro(f"{tag}GpuRed", f"{100 * np.mean(g):.0f}")
    open(os.path.join(P, "tab_main.tex"), "w").write("\n".join(rows[0:]).replace("\\midrule\n", "", 1) + "\n")


def simple_table(tags, methods, fname, first_col):
    rows = []
    for title, tag, ds in tags:
        x = load(tag, ds)
        if not x:
            continue
        ns = gpu(x["exact+suffix"])
        for m in methods:
            if m not in x:
                continue
            r = x[m]
            a, n = nonexact_err(r)
            rows.append(f"{title if m == methods[0] else ''} & {LABEL.get(m, m)} & {r['lat_mean']:.2f} & "
                        f"{100 * gpu(r) / ns:.0f}\\% & {n} & {100 * a / n if n else 0:.1f} & "
                        f"{r.get('stale_err', 0)}/{r.get('stale_reuse', 0)} & {r['em']:.3f}\\\\")
        rows.append("\\midrule")
    open(os.path.join(P, fname), "w").write("\n".join(rows[:-1]) + "\n")


main_table()
simple_table([(DS[d], "v11_abl", d) for d in DS],
             ["exact+suffix", "unison-bstop", "unison-nodraft", "unison-single", "unison", "unison-defer@dfirst", "unison-defer@gdf", "unison-defer"],
             "tab_abl.tex", "Dataset")
simple_table([("Qwen3-4B", "v11_model_qwen3-4b", "hotpotqa"), ("Qwen2.5-7B", "v11_model_qwen2.5-7b", "hotpotqa")],
             ["exact+suffix", "noshare", "semcache+S-0.90", "unison-bstop", "unison", "unison-defer"], "tab_models.tex", "Model")
sw = [("edit $\\mu{=}0.5$", "v11_ver_mu0.5"), ("edit $\\mu{=}2.0$", "v11_ver_mu2.0"), ("Zipf $s{=}0$", "v11_zipf_0"),
      ("Zipf $s{=}1.1$", "v11_zipf_1.1"), ("team 1", "v11_team_1"), ("team 5", "v11_team_5"),
      ("rate 0.6/s", "v11_load_r0.6"), ("rate 2.4/s", "v11_load_r2.4"), ("Azure trace", "v11_azure"), ("low redundancy", "v11_lowred")]
simple_table([(a, b, "hotpotqa") for a, b in sw], ["exact+suffix", "semcache+S-0.90", "semcacheV+S-0.90", "unison-bstop", "unison-defer"],
             "tab_sweep.tex", "Setting")
# sweep summary macros
red, cnt, xs, xe = [], 0, 0, 0
for _, t in sw:
    x = load(t, "hotpotqa")
    if x:
        red.append(1 - gpu(x["unison-defer"]) / gpu(x["exact+suffix"]))
        cnt += 1
        xs += x["unison-defer"].get("stale_reuse", 0)
        xe += x["unison-defer"].get("stale_err", 0)
if red:
    macro("SweepGpuRedMin", f"{100 * min(red):.0f}")
    macro("SweepGpuRedMax", f"{100 * max(red):.0f}")
    macro("SweepN", str(cnt))
    macro("SweepXver", str(xs))
    macro("SweepXverErr", str(xe))
with open(os.path.join(P, "numbers_auto.tex"), "w") as f:
    for k, v in sorted(M.items()):
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(len(M), "macros")
