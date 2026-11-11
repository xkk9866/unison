"""Offline stress tests of the 16-token anticipatory law on full-replay calibration traces.

Every predictor sees the trace prefix of H=16 tokens only (as deployed). Thresholds are fitted by
Learn-then-Test on one half of the calibration pairs (split by passage) and evaluated on the other
half, with Clopper-Pearson 95% intervals. Predictors:
  law        anticipatory survival law (observed survival + learned remainder hazard R_H)
  law-noR    same network without R_H (observed survival only)
  gru-clf    plain GRU classifier, same three inputs, same size, logistic output
  prefix     commit iff all H replayed tokens are accepted (no learning)
Subgroups: cross-version pairs, pairs whose first divergence lies beyond H ("late divergence"),
pairs whose answer span starts after token 40.
"""
import gzip
import json
import os
import pickle
import sys
import zlib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import beta
from sklearn.metrics import roc_auc_score

from train_ssn import calibrate
from unison.common import RESULTS
from unison.trace import CausalSSN, features, pad

H = 16
torch.manual_seed(0)


class NoR(CausalSSN):
    def log_survival_all(self, x, mask):
        ls = super().log_survival_all(x, mask)
        T = x.shape[1]
        pos = torch.arange(T, device=x.device, dtype=torch.float32)[None, :, None].expand(x.shape[0], T, 1) / 64.0
        z = torch.cat([torch.log1p(x[..., :1]), x[..., 1:2] / 10.0, x[..., 2:3], pos], -1)
        h, _ = self.gru(torch.tanh(self.inp(z)))
        rest = nn.functional.softplus(self.fut(torch.cat([h, z], -1)).squeeze(-1))
        return ls + rest  # remove the remainder hazard


class GRUClf(nn.Module):
    def __init__(self, hid=32):
        super().__init__()
        self.inp = nn.Linear(4, hid)
        self.gru = nn.GRU(hid, hid, batch_first=True)
        self.out = nn.Sequential(nn.Linear(hid, hid), nn.GELU(), nn.Linear(hid, 1))

    def log_survival_all(self, x, mask):
        T = x.shape[1]
        pos = torch.arange(T, device=x.device, dtype=torch.float32)[None, :, None].expand(x.shape[0], T, 1) / 64.0
        z = torch.cat([torch.log1p(x[..., :1]), x[..., 1:2] / 10.0, x[..., 2:3], pos], -1)
        h, _ = self.gru(torch.tanh(self.inp(z)))
        return nn.functional.logsigmoid(self.out(h).squeeze(-1))


def train(net, F, y, epochs=25):
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-4)
    yt = torch.tensor(y, dtype=torch.float32)
    for _ in range(epochs):
        perm = torch.randperm(len(F))
        for i in range(0, len(F), 256):
            idx = perm[i:i + 256].tolist()
            x, m = pad([F[j][:H] for j in idx])
            ls = net.log_survival_all(x, m)
            last = (m.sum(1).long() - 1)
            l = ls.gather(1, last[:, None])[:, 0]
            yy = yt[idx]
            loss = -(yy * l + (1 - yy) * torch.log(-torch.expm1(l.clamp(max=-1e-6)))).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return net.eval()


def score(net, F):
    with torch.no_grad():
        x, m = pad([f[:H] for f in F])
        ls = net.log_survival_all(x, m)
        last = (m.sum(1).long() - 1)
        return ls.gather(1, last[:, None])[:, 0].exp().numpy()


def cp(k, n):
    if n == 0:
        return [0.0, 1.0]
    lo = beta.ppf(0.025, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(0.975, k + 1, n - k) if k < n else 1.0
    return [float(lo), float(hi)]


def main():
    torch.set_num_threads(8)
    tr = pickle.load(gzip.open(os.path.join(RESULTS, "ssn", "train_qwen3-8b.pkl.gz")))
    ca = pickle.load(gzip.open(os.path.join(RESULTS, "ssn", "calib_qwen3-8b.pkl.gz")))
    Ft = [features(r["lp"], r["lp1"], r["dmask"]) for r in tr]
    yt = [r["agree"] for r in tr]
    Fc = [features(r["lp"], r["lp1"], r["dmask"]) for r in ca]
    yc = np.array([r["agree"] for r in ca])
    half = np.array([zlib.crc32(f"{r['ds']}{r['pid']}".encode()) % 2 for r in ca])
    st = torch.load(os.path.join(RESULTS, "ssn", "law.pt"), map_location="cpu")
    st.pop("__meta__")
    law = CausalSSN()
    law.load_state_dict(st)
    law.eval()
    preds = {"law": score(law, Fc)}
    preds["law-noR"] = score(train(NoR(), Ft, yt), Fc)
    preds["gru-clf"] = score(train(GRUClf(), Ft, yt), Fc)
    preds["prefix"] = np.array([1.0 if r["acc"] >= min(H, r["n"]) else 0.0 for r in ca])
    late_div = np.array([r["acc"] >= H and r["acc"] < r["n"] for r in ca])
    ans_pos = np.array([int(np.argmax(r["dmask"])) if r["dmask"].any() else 0 for r in ca])
    xver = np.array([r["v_u"] != r["v_w"] for r in ca])
    out = {"n_pairs": len(ca), "agree": float(yc.mean()), "late_divergence_share": float(late_div.mean()),
           "late_divergence_disagree_share": float((late_div & (yc == 0)).sum() / max(1, (yc == 0).sum())),
           "answer_pos_median": float(np.median(ans_pos)), "predictors": {}}
    for name, q in preds.items():
        res = {"auc": float(roc_auc_score(yc, q))}
        for eps in (0.02, 0.05):
            t = calibrate(q[half == 0], yc[half == 0], eps)
            com = (half == 1) & (q >= t)
            k, n = int((yc[com] == 0).sum()), int(com.sum())
            sub = {}
            for g, gm in [("cross_version", xver), ("late_divergence", late_div), ("answer_after_40", ans_pos > 40)]:
                c2 = com & gm
                k2, n2 = int((yc[c2] == 0).sum()), int(c2.sum())
                sub[g] = dict(n=n2, err=k2, risk=k2 / n2 if n2 else 0.0, ci=cp(k2, n2))
            res[f"eps{eps}"] = dict(theta=float(t), coverage=float(com.sum() / (half == 1).sum()), n=n, err=k,
                                    risk=k / n if n else 0.0, ci=cp(k, n), subgroups=sub)
        out["predictors"][name] = res
        print(name, json.dumps(res), flush=True)
    json.dump(out, open(os.path.join(RESULTS, "ssn", "stress.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "predictors"}))


if __name__ == "__main__":
    main()
