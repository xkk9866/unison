"""How much of the replay trace does the law need? Causal SSN trained on trace prefixes.

For horizon m (tokens of the candidate replayed), the law sees trace[0:m] and must predict decision
agreement. Reports per m: AUC, calibrated coverage at eps and the verification cost (m tokens).
"""
import gzip
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from train_ssn import calibrate, deploy
from unison.common import RESULTS
from unison.trace import features, pad


class CausalSSN(nn.Module):
    def __init__(self, hid=32, K=4):
        super().__init__()
        self.inp = nn.Linear(4, hid)
        self.gru = nn.GRU(hid, hid, batch_first=True)
        self.cons = nn.Sequential(nn.Linear(hid + 4, hid), nn.GELU(), nn.Linear(hid, 1))
        self.fut = nn.Sequential(nn.Linear(hid + 4, hid), nn.GELU(), nn.Linear(hid, 1))
        self.a = nn.Parameter(torch.zeros(K))
        self.b = nn.Parameter(torch.linspace(-2, 1, K))

    def forward(self, x, mask):
        """log P(agree | trace[0:t]) for every t: observed survival + learned unobserved remainder."""
        T = x.shape[1]
        pos = torch.arange(T, device=x.device, dtype=torch.float32)[None, :, None].expand(x.shape[0], T, 1) / 64.0
        z = torch.cat([torch.log1p(x[..., :1]), x[..., 1:2] / 10.0, x[..., 2:3], pos], -1)
        h, _ = self.gru(torch.tanh(self.inp(z)))
        hz = torch.cat([h, z], -1)
        a, b = F.softplus(self.a), F.softplus(self.b)
        inten = ((1 - torch.exp(-x[..., :1] * b)) * a).sum(-1)
        lam = F.softplus(self.cons(hz).squeeze(-1)) * inten * mask
        rest = F.softplus(self.fut(hz).squeeze(-1))  # expected hazard of the part not yet replayed
        return -(torch.cumsum(lam, 1) + rest)  # [B, T]


def main():
    dev = "cuda"
    tr = pickle.load(gzip.open(os.path.join(RESULTS, "ssn", "train_qwen3-8b.pkl.gz")))
    ca = pickle.load(gzip.open(os.path.join(RESULTS, "ssn", "calib_qwen3-8b.pkl.gz")))
    Ft = [features(r["lp"], r["lp1"], r["dmask"]) for r in tr]
    Fc = [features(r["lp"], r["lp1"], r["dmask"]) for r in ca]
    yt = torch.tensor([r["agree"] for r in tr], dtype=torch.float32)
    torch.manual_seed(0)
    net = CausalSSN().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-4)
    for ep in range(25):
        perm = torch.randperm(len(tr))
        for i in range(0, len(tr), 256):
            idx = perm[i:i + 256].tolist()
            x, m = pad([Ft[j] for j in idx])
            x, m = x.to(dev), m.to(dev)
            ls = net(x, m)  # supervise every prefix length with the same final label
            y = yt[idx].to(dev)[:, None]
            ll = y * ls + (1 - y) * torch.log(-torch.expm1(ls.clamp(max=-1e-6)))
            loss = -(ll * m).sum() / m.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
    net.eval()
    yc = np.array([r["agree"] for r in ca])
    out = {}
    with torch.no_grad():
        x, m = pad(Fc)
        LS = net(x.to(dev), m.to(dev)).exp().cpu().numpy()
    n = np.array([len(f) for f in Fc])
    for H in [4, 8, 12, 16, 24, 32, 48, 64, 96, 128]:
        q = LS[np.arange(len(Fc)), np.minimum(H, n) - 1]
        cert = np.array([r["acc"] >= min(H, r["n"]) and r["n"] <= H for r in ca])  # full candidate replayed & accepted
        for r, c in zip(ca, cert):
            r["certified_h"] = bool(c)
        saved = {}
        c_, s_, ok_, _ = deploy([dict(r, certified=r["certified_h"]) for r in ca], q)
        res = {}
        for e in (0.02, 0.05, 0.1):
            t = calibrate(s_[~c_], ok_[~c_], e)
            com = ~c_ & (s_ >= t)
            res[f"{e}"] = dict(cov=float((com | c_).mean()), risk=float(1 - ok_[com].mean()) if com.any() else 0.0)
        out[H] = dict(auc=float(roc_auc_score(yc, q)), mean_cost=float(np.minimum(H, n).mean()), **res)
        print(H, json.dumps(out[H]), flush=True)
    json.dump(out, open(os.path.join(RESULTS, "ssn", "horizon.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
