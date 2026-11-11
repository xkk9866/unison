"""Law vs. plain GRU classifier under model shift, without recalibration.

Both predictors see the 16-token replay prefix, are trained on Qwen3-8B training pairs, and get their
eps=5% threshold from Qwen3-8B calibration pairs (Learn-then-Test). The frozen thresholds are then
applied to replay traces of Qwen3-4B and Qwen2.5-7B: a predictor whose scores keep their meaning
across models keeps its risk near the budget.
"""
import gzip
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch
from scipy.stats import beta
from sklearn.metrics import roc_auc_score

from stress import GRUClf, score, train
from train_ssn import calibrate
from unison.common import RESULTS
from unison.trace import CausalSSN, features

torch.manual_seed(0)


def load(name):
    rows = pickle.load(gzip.open(os.path.join(RESULTS, "ssn", name)))
    return [features(r["lp"], r["lp1"], r["dmask"]) for r in rows], np.array([r["agree"] for r in rows])


def main():
    torch.set_num_threads(8)
    Ft, yt = load("train_qwen3-8b.pkl.gz")
    Fc, yc = load("calib_qwen3-8b.pkl.gz")
    st = torch.load(os.path.join(RESULTS, "ssn", "law.pt"), map_location="cpu")
    st.pop("__meta__")
    law = CausalSSN()
    law.load_state_dict(st)
    law.eval()
    clf = train(GRUClf(), Ft, list(yt))
    out = {}
    for name, net in [("law", law), ("gru-clf", clf)]:
        res = {}
        for eps in (0.02, 0.05):
            t = calibrate(score(net, Fc), yc, eps)
            r = {}
            for m in ("qwen3-8b", "qwen3-4b", "qwen2.5-7b"):
                F, y = load(f"calib_{m}.pkl.gz")
                q = score(net, F)
                com = q >= t
                k, n = int((y[com] == 0).sum()), int(com.sum())
                r[m] = dict(auc=float(roc_auc_score(y, q)), coverage=float(com.mean()), n=n, err=k,
                            risk=k / n if n else 0.0, ub95=float(beta.ppf(0.95, k + 1, n - k)) if n else 0.0)
            res[f"eps{eps}"] = r
        out[name] = res
        print(name, json.dumps(res), flush=True)
    json.dump(out, open(os.path.join(RESULTS, "ssn", "transfer.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
