"""Train the Surprisal Survival Network on replay traces, calibrate commit thresholds, evaluate.

Train: replay traces from the train pools (independent of eval and calib pools).
Calibrate: deployment statistic on the calib pool. A consumer whose candidate set contains a certified
replay (zero divergence) is served exactly; for the others the statistic is max_u P(agree | trace_u)
and the outcome is the decision agreement of the argmax candidate. Thresholds come from Learn-then-Test
with Clopper-Pearson bounds (FWER delta).
Also reported on the same calibration pairs/consumers: text-only predictors (AHN, condition cosine),
the un-learned survival exp(-sum sigma), and a logistic regression on hand-made trace statistics.
"""
import argparse
import gzip
import json
import os
import pickle
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import beta
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from unison.common import RESULTS
from unison.trace import SSN, features, pad

EPS = [0.01, 0.02, 0.05, 0.1]
DELTA = 0.05


def calibrate(scores, agree, eps, delta=DELTA):
    """Fixed-sequence Learn-then-Test: largest coverage whose Clopper-Pearson upper bound <= eps."""
    order = np.argsort(-scores, kind="stable")
    s, a = scores[order], agree[order]
    errs = np.cumsum(1 - a)
    best = 1.01
    k0 = int(np.ceil(np.log(delta) / np.log(1 - eps)))
    for k in range(k0, len(s) + 1):
        if k < len(s) and s[k] == s[k - 1]:
            continue  # thresholds are only realisable between distinct scores
        e = int(errs[k - 1])
        ub = beta.ppf(1 - delta, e + 1, k - e) if e < k else 1.0
        if ub > eps:
            break
        best = float(s[k - 1])
    return best


def feats_of(rows):
    return [features(r["lp"], r["lp1"], r["dmask"]) for r in rows]


def hand_stats(f):
    sig, d = f[:, 0], f[:, 2]
    return [np.log1p(sig.sum()), np.log1p(sig.max()), np.log1p((sig * d).sum()), np.log1p((sig > 0).sum()),
            np.log1p(len(sig)), float((sig > 0).argmax() / len(sig)) if (sig > 0).any() else 1.0]


@torch.no_grad()
def predict(net, feats, dev, bs=1024):
    out = []
    for i in range(0, len(feats), bs):
        x, m = pad(feats[i:i + bs])
        out.append(net(x.to(dev), m.to(dev)).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def deploy(rows, q):
    """Group pairs by consumer -> (certified?, max q, agreement of argmax candidate)."""
    g = defaultdict(list)
    for r, s in zip(rows, q):
        g[(r["ds"], r["w"])].append((r, s))
    cert, stat, ok, ds = [], [], [], []
    for (d, _), lst in g.items():
        c = any(r["certified"] for r, _ in lst)
        r, s = max(lst, key=lambda x: x[1])
        cert.append(c)
        stat.append(s)
        ok.append(1 if c else r["agree"])
        ds.append(d)
    return np.array(cert), np.array(stat, np.float64), np.array(ok), np.array(ds)


def risk_table(cert, stat, ok, eps_list, thr=None):
    unc = ~cert
    out = {}
    for e in eps_list:
        t = thr[f"{e:.3f}"] if thr else calibrate(stat[unc], ok[unc], e)
        com = unc & (stat >= t)
        out[f"{e:.3f}"] = dict(theta=float(t), coverage=float((cert | com).mean()), risk_cov=float(com.mean()),
                               risk=float(1 - ok[com].mean()) if com.any() else 0.0, n_commit=int(com.sum()))
    return out


def main(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    with gzip.open(a.train, "rb") as f:
        tr = pickle.load(f)
    with gzip.open(a.calib, "rb") as f:
        ca = pickle.load(f)
    rng = np.random.default_rng(0)
    pids = sorted({(r["ds"], r["pid"]) for r in tr})
    val_p = set(map(tuple, np.array(pids, dtype=object)[rng.random(len(pids)) < 0.1]))
    trn = [r for r in tr if (r["ds"], r["pid"]) not in val_p]
    val = [r for r in tr if (r["ds"], r["pid"]) in val_p]
    Ftr, Fva, Fca = feats_of(trn), feats_of(val), feats_of(ca)
    ytr = torch.tensor([r["agree"] for r in trn], dtype=torch.float32)
    yva = np.array([r["agree"] for r in val])
    yca = np.array([r["agree"] for r in ca])
    print("train", len(trn), "val", len(val), "calib", len(ca), "agree", ytr.mean().item(), yca.mean(), flush=True)

    net = SSN().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-4)
    best, best_state = -1, None
    if a.load:
        st0 = torch.load(a.load, map_location="cpu")
        st0.pop("__meta__")
        net.load_state_dict(st0)
        best_state, best = {k: v.clone() for k, v in st0.items()}, float("nan")
    for ep in range(0 if a.load else a.epochs):
        net.train()
        perm = torch.randperm(len(trn))
        for i in range(0, len(trn), 256):
            idx = perm[i:i + 256].tolist()
            x, m = pad([Ftr[j] for j in idx])
            ls, _ = net.log_survival(x.to(dev), m.to(dev))
            y = ytr[idx].to(dev)
            # Bernoulli likelihood of agreement with log P = log-survival (numerically stable form)
            loss = -(y * ls + (1 - y) * torch.log(-torch.expm1(ls.clamp(max=-1e-6)))).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        pv = predict(net, Fva, dev)
        auc = roc_auc_score(yva, pv)
        if auc > best:
            best, best_state = auc, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        print(f"ep {ep} loss {loss.item():.4f} val_auc {auc:.4f}", flush=True)
    net.load_state_dict(best_state)
    net.to(dev).eval()

    # ---------------------------------------------------------------- predictors on calibration pairs
    P = {"ssn": predict(net, Fca, dev)}
    P["survival(no-learn)"] = np.array([np.exp(-f[:, 0].sum()) for f in Fca])
    P["decision-surprisal"] = np.array([np.exp(-(f[:, 0] * f[:, 2]).sum()) for f in Fca])
    lr = LogisticRegression(max_iter=2000).fit(np.array([hand_stats(f) for f in Ftr]), ytr.numpy())
    P["trace-LR"] = lr.predict_proba(np.array([hand_stats(f) for f in Fca]))[:, 1]
    cert = np.array([r["certified"] for r in ca])
    for k in ("q_ahn", "cos"):
        if k in ca[0]:
            P[{"q_ahn": "AHN(text)", "cos": "cosine(text)"}[k]] = np.array([r[k] for r in ca])
    metrics = dict(n_train=len(trn), n_calib_pairs=len(ca), agree_rate=float(yca.mean()),
                   certified_pairs=float(cert.mean()), val_auc=float(best), predictors={})
    for name, q in P.items():
        unc = ~cert
        auc_all = roc_auc_score(yca, q)
        auc_unc = roc_auc_score(yca[unc], q[unc]) if len(set(yca[unc])) > 1 else float("nan")
        c, s, ok, _ = deploy(ca, q)
        metrics["predictors"][name] = dict(auc=float(auc_all), auc_uncertified=float(auc_unc),
                                           deploy=risk_table(c, s, ok, EPS))
        print(name, json.dumps(metrics["predictors"][name]), flush=True)

    # ---------------------------------------------------------------- validity of the guarantee (split calib)
    c, s, ok, dsv = deploy(ca, P["ssn"])
    val_runs = defaultdict(list)
    for rep in range(100):
        msk = np.random.default_rng(rep).random(len(c)) < 0.5
        for e in EPS:
            t = calibrate(s[msk & ~c], ok[msk & ~c], e)
            com = ~msk & ~c & (s >= t)
            val_runs[f"{e:.3f}"].append((float(1 - ok[com].mean()) if com.any() else 0.0, float((~c[~msk] & (s[~msk] >= t)).mean())))
    metrics["split_validity"] = {k: dict(risk_mean=float(np.mean([x for x, _ in v])),
                                         risk_q95=float(np.quantile([x for x, _ in v], 0.95)),
                                         viol_rate=float(np.mean([x > float(k) for x, _ in v])),
                                         cov_mean=float(np.mean([y for _, y in v]))) for k, v in val_runs.items()}
    print("split_validity", json.dumps(metrics["split_validity"]), flush=True)
    thr = {f"{e:.3f}": calibrate(s[~c], ok[~c], e) for e in EPS}
    metrics["thresholds"] = thr
    c2, s2, ok2, _ = deploy(ca, P["survival(no-learn)"])
    alt = {"survival": {f"{e:.3f}": calibrate(s2[~c2], ok2[~c2], e) for e in EPS}}
    st = {k: v for k, v in net.state_dict().items()}
    st["__meta__"] = dict(arch=dict(hid=32, K=4), thresholds=thr, alt_thresholds=alt, delta=DELTA,
                          thresholds_by={"offline": thr, "survival": alt["survival"], "deploy": thr},
                          n_params=sum(p.numel() for p in net.parameters()))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    torch.save(st, a.out)
    with open(a.out.replace(".pt", "_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=1)
    np.savez(a.out.replace(".pt", "_calib_scores.npz"), **{k.replace("(", "_").replace(")", ""): v for k, v in P.items()},
             agree=yca, certified=cert)
    print("thresholds", thr, "params", st["__meta__"]["n_params"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train", default=os.path.join(RESULTS, "ssn", "train_qwen3-8b.pkl.gz"))
    p.add_argument("--calib", default=os.path.join(RESULTS, "ssn", "calib_qwen3-8b.pkl.gz"))
    p.add_argument("--out", default=os.path.join(RESULTS, "ssn", "ssn.pt"))
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--load", default="")
    main(p.parse_args())
