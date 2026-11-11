"""Deployment calibration of the commit threshold from lossless (epsilon = 0) runs.

In an epsilon = 0 run every consumer that drafts a candidate still computes its own output, and the
engine logs the law's value q on the very verification pass that would decide a commit. The label is
free: agree = [decision(own output) == decision(candidate)]. Certified passes (full acceptance) are
exact and excluded. Thresholds come from fixed-sequence Learn-then-Test with Clopper-Pearson bounds.

  python scripts/calib_shadow.py --runs calibshadow/hotpotqa,calibshadow/musique --ssn results/ssn/ssn.pt
"""
import argparse
import gzip
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from train_ssn import EPS, calibrate
from unison.common import RESULTS, decision_equiv


def load(run, method):
    with gzip.open(os.path.join(RESULTS, run, f"{method}.calls.jsonl.gz"), "rt") as f:
        recs = [json.loads(l) for l in f]
    out = []
    for r in recs:
        if r["kind"] != "verify" or r.get("cand_dec") is None or r.get("q") is None:
            continue
        out.append((float(r["q"]), int(decision_equiv(r["own_dec"], r["cand_dec"])), r["cand_version"] != r["version"]))
    return out


def main(a):
    S = []
    for run in a.runs.split(","):
        S += load(run, a.method)
    q = np.array([x for x, _, _ in S])
    y = np.array([x for _, x, _ in S])
    xv = np.array([x for _, _, x in S])
    thr = {f"{e:.3f}": calibrate(q, y, e) for e in EPS}
    rep = dict(n=len(S), agree=float(y.mean()), auc=float(roc_auc_score(y, q)) if 0 < y.mean() < 1 else None,
               cross_version=int(xv.sum()), thresholds=thr,
               coverage={k: float((q >= t).mean()) for k, t in thr.items()},
               risk={k: float(1 - y[q >= t].mean()) if (q >= t).any() else 0.0 for k, t in thr.items()})
    print(json.dumps(rep, indent=1))
    st = torch.load(a.ssn, map_location="cpu")
    key = f"deploy_{a.workload}" if a.workload else "deploy"
    st["__meta__"].setdefault("thresholds_by", {})[key] = thr
    st["__meta__"].setdefault("deploy_calib", {})[key] = rep
    torch.save(st, a.ssn)
    np.savez(a.ssn.replace(".pt", f"_shadow{('_' + a.workload) if a.workload else ''}.npz"), q=q, agree=y,
             cross_version=xv)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--runs", required=True)
    p.add_argument("--method", default="unison")
    p.add_argument("--workload", default="")
    p.add_argument("--ssn", default=os.path.join(RESULTS, "ssn", "ssn.pt"))
    main(p.parse_args())
