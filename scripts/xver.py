"""Cross-version stress test from a high-edit-rate shadow run (mu = 2 edits/s).

Every drafted consumer in shadow mode logs the law's q and computes its own output. Applying the
deployed workload threshold, we count the decisions Unison would have committed, split by whether
the candidate was computed on a different evidence version, and report risk with one-sided 95%
Clopper-Pearson upper bounds. 'Answer-changing edits' are cross-version pairs whose own decision
differs from the candidate's -- exactly the cases a version-blind reuse would get wrong.
"""
import gzip
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch
from scipy.stats import beta

from unison.common import RESULTS, decision_equiv


def ub(k, n):
    return float(beta.ppf(0.95, k + 1, n - k)) if n else 1.0


def main():
    meta = torch.load(os.path.join(RESULTS, "ssn", "law.pt"), map_location="cpu")["__meta__"]
    out = {}
    for ds in ("hotpotqa", "musique"):
        th = meta["thresholds_by"][f"deploy_{ds}"]
        recs = [json.loads(l) for l in gzip.open(os.path.join(RESULTS, "v8_xver", ds, "unison-shadow.calls.jsonl.gz"), "rt")]
        S = [r for r in recs if r["kind"] == "verify" and r.get("cand_dec") is not None and r.get("q") is not None]
        res = {"drafted": len(S), "cross_version_drafted": sum(r["cand_version"] != r["version"] for r in S)}
        xa = [r for r in S if r["cand_version"] != r["version"]]
        res["answer_changing_edits"] = sum(not decision_equiv(r["own_dec"], r["cand_dec"]) for r in xa)
        for eps in ("0.020", "0.050"):
            t = th[eps]
            g = {}
            for name, sub in [("all", S), ("same_version", [r for r in S if r["cand_version"] == r["version"]]),
                              ("cross_version", xa)]:
                com = [r for r in sub if r["q"] >= t]
                k = sum(not decision_equiv(r["own_dec"], r["cand_dec"]) for r in com)
                g[name] = dict(n=len(com), err=k, risk=k / len(com) if com else 0.0, ub95=ub(k, len(com)))
            blocked = [r for r in xa if not decision_equiv(r["own_dec"], r["cand_dec"])]
            g["answer_changing_blocked"] = dict(n=len(blocked), blocked=sum(r["q"] < t for r in blocked))
            res[f"eps{eps}"] = g
        out[ds] = res
        print(ds, json.dumps(res), flush=True)
    json.dump(out, open(os.path.join(RESULTS, "ssn", "xver.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
