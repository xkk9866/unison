"""Team-clustered risk of Unison commits (answers the correlated-decisions concern).

For each (run, method) the non-exact commits are grouped by question team; a cluster bootstrap over
teams (pooled across seeds) gives a 95% interval and a one-sided 95% upper bound of the commit risk.
Also reports the cross-version subgroup and compute accounting (draft tokens, GPU busy, q/GPU-s).
Usage: python scripts/cluster_risk.py v7_main_s0,v7_main_s1,v7_main_s2 [out.json]
"""
import gzip
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from transformers import AutoTokenizer

from unison.common import MODELS, RESULTS, decision_equiv, parse_note

RISK = ("risk", "sem", "race-risk")


def main():
    tags = sys.argv[1].split(",")
    out_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(RESULTS, "cluster_risk.json")
    tok = AutoTokenizer.from_pretrained(MODELS["qwen3-8b"])
    dec = lambda t: parse_note(tok.decode(t, skip_special_tokens=True))
    out = {}
    for ds in ("hotpotqa", "musique"):
        per = defaultdict(lambda: defaultdict(list))  # method -> team-key -> [err...]
        xv = defaultdict(list)
        comp = defaultdict(lambda: defaultdict(float))
        for tag in tags:
            d = os.path.join(RESULTS, tag, ds)
            if not os.path.exists(os.path.join(d, "replay.json.gz")):
                continue
            ref = json.load(gzip.open(os.path.join(d, "replay.json.gz"), "rt"))
            for s in json.load(open(os.path.join(d, "summary.json"))):
                m = s["method"]
                c = comp[m]
                c["draft"] += s.get("draft_tokens", 0)
                c["acc"] += s.get("accepted_tokens", 0)
                c["busy"] += s["engine_busy"]
                c["queries"] += s["n_queries"]
                c["vsteps"] += s["verify_steps"]
                c["runs"] += 1
                f = os.path.join(d, f"{m}.calls.jsonl.gz")
                if not os.path.exists(f):
                    continue
                for line in gzip.open(f, "rt"):
                    r = json.loads(line)
                    if r["kind"] != "verify" or r["source"] not in RISK or r.get("sim", 0.0) >= 2.0:
                        continue  # exact-prompt hits of a semantic cache are certified, not risk commits
                    e = 0 if decision_equiv(dec(r["out"]), dec(ref[r["phash"]])) else 1
                    per[m][f"{tag}-{r['team']}"].append(e)
                    if r.get("src_version", r["version"]) != r["version"]:
                        xv[m].append(e)
        res = {}
        rng = np.random.default_rng(0)
        for m, groups in per.items():
            keys = list(groups)
            errs = np.array([sum(groups[k]) for k in keys], float)
            ns = np.array([len(groups[k]) for k in keys], float)
            point = errs.sum() / ns.sum()
            boot = []
            for _ in range(4000):
                i = rng.integers(0, len(keys), len(keys))
                boot.append(errs[i].sum() / max(1.0, ns[i].sum()))
            boot = np.array(boot)
            x = np.array(xv[m]) if xv[m] else np.zeros(0)
            k, n = int(x.sum()), len(x)
            from scipy.stats import beta
            ub_x = float(beta.ppf(0.95, k + 1, n - k)) if n else 1.0
            res[m] = dict(n_commits=int(ns.sum()), n_teams=len(keys), errors=int(errs.sum()), risk=float(point),
                          ci95=[float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
                          ub95_one_sided=float(np.quantile(boot, 0.95)),
                          xver=dict(n=n, err=k, ub95_one_sided=ub_x))
        cres = {m: dict(draft_tokens=c["draft"] / c["runs"], accepted=c["acc"] / c["runs"],
                        rejected=(c["draft"] - c["acc"]) / c["runs"], busy=c["busy"] / c["runs"],
                        q_per_gpu_s=c["queries"] / c["busy"], vsteps=c["vsteps"] / c["runs"]) for m, c in comp.items()}
        out[ds] = dict(risk=res, compute=cres)
        print(ds, json.dumps(res, indent=0)[:1500], flush=True)
    json.dump(out, open(out_path, "w"), indent=1)


if __name__ == "__main__":
    main()
