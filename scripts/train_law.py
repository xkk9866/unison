"""Train the anticipatory (causal) Surprisal Survival law, choose the replay horizon, calibrate.

Horizon choice is a cost model, not a tuned constant. Replaying H tokens of a candidate costs
H * gamma of verification; a commit avoids decoding the consumer's remaining output, (n_bar - H) * delta.
For each H the calibrated commit coverage cov_eps(H) on the calibration pool gives the expected net
saving per consumer  cov_eps(H) * (n_bar - H) * delta - H * gamma; H* maximises it (gamma, delta are
the engine constants profiled at start-up, see engine.profile()).

  python scripts/train_law.py                       # train on qwen3-8b replays, calibrate, save ssn_causal.pt
  python scripts/train_law.py --load X --calib Y --out Z   # frozen law, other model: calibrate only
"""
import argparse
import gzip
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from train_ssn import EPS, calibrate, deploy
from unison.common import RESULTS
from unison.trace import CausalSSN, features, pad

GRID = [8, 12, 16, 20, 24, 32, 48, 64, 128]


def prefix_scores(net, F, H, dev):
    out = []
    with torch.no_grad():
        for i in range(0, len(F), 1024):
            b = [f[:H] for f in F[i:i + 1024]]
            x, m = pad(b)
            out.append(net(x.to(dev), m.to(dev)).cpu().numpy())
    return np.concatenate(out)


def surv_scores(F, H):
    return np.array([np.exp(-f[:H, 0].sum()) for f in F])


def table(rows, q, H):
    rr = [dict(r, certified=bool(r["n"] <= H and r["certified"])) for r in rows]
    c, s, ok, _ = deploy(rr, q)
    out = {}
    for e in EPS:
        t = calibrate(s[~c], ok[~c], e)
        com = ~c & (s >= t)
        out[f"{e:.3f}"] = dict(theta=float(t), coverage=float((com | c).mean()), risk_cov=float(com.mean()),
                               risk=float(1 - ok[com].mean()) if com.any() else 0.0)
    return out


def seq_looks(r, C):
    """Look positions of sequential replay with chunk C: after every chunk while the consumer stays
    aligned with the candidate, and once more at the end of the chunk that contains the divergence."""
    n, acc = r["n"], r["acc"]
    looks, L = [], 1
    while L < n:
        L = min(n, L + C)
        looks.append(L)
        if acc < L:          # divergence inside this chunk: its counterfactual tail is observed, stop
            break
        if acc == L and L < n:  # the consumer's own (bonus) token diverges at the boundary: stop
            break
        L += 1               # the bonus token at the boundary was accepted
    return looks


def seq_features(f, r, looks):
    """Trace as the engine sees it: positions not drafted (prefill token, accepted bonus tokens) carry
    sigma = 0 and unobserved log-probability (0)."""
    g = f.copy()
    drafted = np.zeros(len(f), bool)
    L = 1
    for Lk in looks:
        drafted[L:Lk] = True
        L = Lk + 1
    g[~drafted, 0] = 0.0
    g[~drafted, 1] = 0.0
    return g


def seq_scores(net, F, rows, C, dev):
    """max over looks of the law (commit happens at the first look that clears theta), cost, waste."""
    q, cost, waste, first = [], [], [], []
    items = []
    for i, (f, r) in enumerate(zip(F, rows)):
        lk = seq_looks(r, C)
        g = seq_features(f, r, lk)
        for L in lk:
            items.append((i, L, g[:L]))
    vals = np.zeros(len(items))
    with torch.no_grad():
        for j in range(0, len(items), 1024):
            b = items[j:j + 1024]
            x, m = pad([z for _, _, z in b])
            vals[j:j + len(b)] = net(x.to(dev), m.to(dev)).cpu().numpy()
    per = {}
    for (i, L, _), v in zip(items, vals):
        per.setdefault(i, []).append((L, v))
    for i, r in enumerate(rows):
        lst = per.get(i, [(1, 0.0)])
        q.append(max(v for _, v in lst))
        last = lst[-1][0]
        cost.append(last - 1)
        waste.append(max(0, last - max(r["acc"], 1)))
        first.append(lst)
    return np.array(q), np.array(cost), np.array(waste), first


def main(a):
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    ca = pickle.load(gzip.open(a.calib, "rb"))
    Fc = [features(r["lp"], r["lp1"], r["dmask"]) for r in ca]
    yc = np.array([r["agree"] for r in ca])
    net = CausalSSN().to(dev)
    if a.load:
        st = torch.load(a.load, map_location="cpu")
        meta0 = st.pop("__meta__")
        net.load_state_dict(st)
    else:
        tr = pickle.load(gzip.open(a.train, "rb"))
        Ft = [features(r["lp"], r["lp1"], r["dmask"]) for r in tr]
        yt = torch.tensor([r["agree"] for r in tr], dtype=torch.float32)
        opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-4)
        for ep in range(a.epochs):
            perm = torch.randperm(len(tr))
            for i in range(0, len(tr), 256):
                idx = perm[i:i + 256].tolist()
                x, m = pad([Ft[j] for j in idx])
                x, m = x.to(dev), m.to(dev)
                ls = net.log_survival_all(x, m)  # every prefix horizon is supervised by the final label
                y = yt[idx].to(dev)[:, None]
                ll = y * ls + (1 - y) * torch.log(-torch.expm1(ls.clamp(max=-1e-6)))
                loss = -(ll * m).sum() / m.sum()
                opt.zero_grad()
                loss.backward()
                opt.step()
            print("ep", ep, round(loss.item(), 4), flush=True)
    net.eval()
    n_bar = float(np.mean([r["n"] for r in ca]))
    ratio = a.gamma_over_delta
    curve = {}
    for H in ([] if a.sequential else GRID):
        q = prefix_scores(net, Fc, H, dev)
        tab = table(ca, q, H)
        cov = tab["0.050"]["coverage"]
        curve[H] = dict(auc=float(roc_auc_score(yc, q)), cost=float(np.mean([min(H, r["n"]) for r in ca])),
                        net=float(cov * (n_bar - min(H, n_bar)) - ratio * H), deploy=tab)
        print(H, json.dumps({k: v for k, v in curve[H].items() if k != "deploy"}), json.dumps(tab["0.050"]), flush=True)
    if a.sequential:
        # sequential replay: chunk C; only the verification after the divergence is wasted
        for C in [8, 16, 24, 32, 48]:
            q, cost, waste, looks = seq_scores(net, Fc, ca, C, dev)
            tab = table(ca, q, 1 << 30)
            t5 = tab["0.050"]["theta"]
            # tokens of decoding a commit avoids: from the first look that clears theta to the end
            sav = []
            for r, lst in zip(ca, looks):
                hit = [L for L, v in lst if v >= t5]
                sav.append(r["n"] - hit[0] if hit else 0.0)
            net_gain = float(np.mean(sav) - ratio * np.mean(waste))
            curve[f"seq{C}"] = dict(auc=float(roc_auc_score(yc, q)), cost=float(cost.mean()), waste=float(waste.mean()),
                                    saved=float(np.mean(sav)), net=net_gain, deploy=tab)
            print("seq", C, json.dumps({k: v for k, v in curve[f"seq{C}"].items() if k != "deploy"}),
                  json.dumps(tab["0.050"]), flush=True)
        best = a.horizon or max([8, 16, 24, 32, 48], key=lambda c: curve[f"seq{c}"]["net"])
        H = int(best)
        thr = {k: v["theta"] for k, v in curve[f"seq{H}"]["deploy"].items()}
        sv = table(ca, np.array([np.exp(-seq_features(f, r, seq_looks(r, H))[:seq_looks(r, H)[-1], 0].sum())
                                 for f, r in zip(Fc, ca)]), 1 << 30)
        meta = dict(arch=dict(causal=True, hid=32, K=4), horizon=H, sequential=True, n_bar=n_bar,
                    gamma_over_delta=ratio, thresholds=thr,
                    thresholds_by={"offline": thr, "deploy": thr, "survival": {k: v["theta"] for k, v in sv.items()}},
                    horizon_curve={str(k): v for k, v in curve.items()},
                    n_params=sum(p.numel() for p in net.parameters()))
        st = {k: v.cpu() for k, v in net.state_dict().items()}
        st["__meta__"] = meta
        torch.save(st, a.out)
        json.dump(meta, open(a.out.replace(".pt", "_meta.json"), "w"), indent=1)
        print("chunk", H, "thresholds", thr, flush=True)
        return
    H = a.horizon or max(GRID, key=lambda h: curve[h]["net"])
    q = prefix_scores(net, Fc, H, dev)
    thr = {k: v["theta"] for k, v in curve[H]["deploy"].items()}
    sv = table(ca, surv_scores(Fc, H), H)
    meta = dict(arch=dict(causal=True, hid=32, K=4), horizon=int(H), n_bar=n_bar, gamma_over_delta=ratio,
                thresholds=thr, thresholds_by={"offline": thr, "deploy": thr,
                                               "survival": {k: v["theta"] for k, v in sv.items()}},
                horizon_curve={str(k): v for k, v in curve.items()},
                n_params=sum(p.numel() for p in net.parameters()))
    st = {k: v.cpu() for k, v in net.state_dict().items()}
    st["__meta__"] = meta
    torch.save(st, a.out)
    json.dump(meta, open(a.out.replace(".pt", "_meta.json"), "w"), indent=1)
    print("horizon", H, "thresholds", thr, "survival", meta["thresholds_by"]["survival"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train", default=os.path.join(RESULTS, "ssn", "train_qwen3-8b.pkl.gz"))
    p.add_argument("--calib", default=os.path.join(RESULTS, "ssn", "calib_qwen3-8b.pkl.gz"))
    p.add_argument("--out", default=os.path.join(RESULTS, "ssn", "law.pt"))
    p.add_argument("--load", default="")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--horizon", type=int, default=0)
    p.add_argument("--sequential", type=int, default=1)
    p.add_argument("--device", default="")
    p.add_argument("--gamma_over_delta", type=float, default=1.1)  # ctok / delta profiled in main runs
    main(p.parse_args())
