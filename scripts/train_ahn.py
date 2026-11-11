"""Train the Agreement Hazard Network, calibrate risk-bounded commit thresholds, evaluate predictors."""
import gzip
import json
import math
import os
import random
import sys
import time
import zlib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch
from scipy.stats import beta, spearmanr
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer

from unison.ahn import AHN, Featurizer, collate, survival_loss, token_pieces
from unison.common import AHN_BACKBONE, MODELS, RESULTS
from unison.retriever import Encoder

EPS = [0.005, 0.01, 0.02, 0.05, 0.1]
torch.manual_seed(0)
random.seed(0)


def calibrate(scores, agree, eps, delta=0.05):
    """Learn-then-Test fixed-sequence calibration of the commit threshold.

    Hypotheses H_c: 'selective risk at coverage c exceeds eps', tested in the fixed order
    c = 1%, 2%, ... with Clopper-Pearson p-values at level delta; we return the threshold of the
    last rejected hypothesis, which bounds P(risk(theta) > eps) <= delta (FWER control)."""
    order = np.argsort(-scores)
    s, a = scores[order], agree[order]
    errs = np.cumsum(1 - a)
    best = 1.01
    k0 = int(np.ceil(np.log(delta) / np.log(1 - eps)))  # smallest sample that can certify eps
    for k in range(k0, len(s) + 1, max(1, len(s) // 200)):
        e = int(errs[k - 1])
        ub = beta.ppf(1 - delta, e + 1, k - e) if e < k else 1.0
        if ub > eps:
            break
        best = float(s[k - 1])
    return best


def c_index(pred, true, n=200000):
    rng = np.random.default_rng(0)
    i, j = rng.integers(0, len(pred), n), rng.integers(0, len(pred), n)
    m = true[i] != true[j]
    i, j = i[m], j[m]
    conc = ((pred[i] - pred[j]) * (true[i] - true[j]) > 0).mean()
    ties = (pred[i] == pred[j]).mean()
    return float(conc + 0.5 * ties)


def main():
    llm_tok = AutoTokenizer.from_pretrained(MODELS["qwen3-8b"])
    with gzip.open(os.path.join(RESULTS, "ahn", "pairs.jsonl.gz"), "rt") as f:
        pairs = [json.loads(l) for l in f]
    piece_cache = {}
    for p in pairs:
        key = tuple(p["prod"])
        if key not in piece_cache:
            piece_cache[key] = token_pieces(llm_tok, p["prod"])
        p["pieces"] = piece_cache[key]
    calib_mask = [zlib.crc32(f"{p['ds']}-{p['pid']}".encode()) % 100 < 15 for p in pairs]
    train = [p for p, c in zip(pairs, calib_mask) if not c]
    calib = [p for p, c in zip(pairs, calib_mask) if c]
    print(f"pairs {len(pairs)} train {len(train)} calib {len(calib)} agree-rate {np.mean([p['agree'] for p in pairs]):.3f}"
          f" mean tau/n {np.mean([p['tau'] / max(1, p['n_prod']) for p in pairs]):.3f}", flush=True)
    enc_tok = AutoTokenizer.from_pretrained(AHN_BACKBONE)
    feat = Featurizer(enc_tok)
    model = AHN(AHN_BACKBONE).cuda()
    head_params = [p for n, p in model.named_parameters() if not n.startswith("enc.")]
    opt = torch.optim.AdamW([{"params": model.enc.parameters(), "lr": 4e-5},
                             {"params": head_params, "lr": 5e-4}], weight_decay=0.01)
    bs, epochs = 64, 3
    steps = epochs * math.ceil(len(train) / bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[4e-5, 5e-4], total_steps=steps, pct_start=0.06)
    bce = torch.nn.BCEWithLogitsLoss()
    t0 = time.time()
    model.train()
    for ep in range(epochs):
        random.shuffle(train)
        for b in range(0, len(train), bs):
            chunk = train[b:b + bs]
            batch, taus, nobs, ys = [], [], [], []
            for p in chunk:  # observe a random prefix 40% of the time (in-flight producers)
                m = p["n_prod"] if random.random() < 0.6 else random.randint(0, p["n_prod"])
                batch.append(feat.encode(p["cond_c"], p["cond_p"], p["delta"], p["pieces"][:m]))
                taus.append(p["tau"])
                nobs.append(m)
                ys.append(p["agree"])
            ids, am, tt, gi = [x.cuda() for x in collate(batch, enc_tok.pad_token_id)]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lh, lq = model(ids, am, tt, gi)
            lh, lq = lh.float(), lq.float()
            loss = survival_loss(lh, gi, torch.tensor(taus).cuda(), torch.tensor(nobs).cuda()) + \
                bce(lq, torch.tensor(ys, dtype=torch.float32).cuda())
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            if (b // bs) % 200 == 0:
                print(f"ep {ep} step {b // bs} loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)
    model.eval()
    # ---------------------------------------------------------------- evaluation on calibration split
    qs, A, taus, agrees = [], [], [], []
    with torch.no_grad():
        for b in range(0, len(calib), 128):
            chunk = calib[b:b + 128]
            batch = [feat.encode(p["cond_c"], p["cond_p"], p["delta"], p["pieces"]) for p in chunk]
            ids, am, tt, gi = [x.cuda() for x in collate(batch, enc_tok.pad_token_id)]
            lh, lq = model(ids, am, tt, gi)
            h = torch.sigmoid(lh.float()).cpu().numpy()
            for i, p in enumerate(chunk):
                S = np.cumprod(1 - h[i, :p["n_prod"]])
                A.append(S.sum())
            qs.extend(torch.sigmoid(lq.float()).cpu().numpy().tolist())
            taus.extend(p["tau"] for p in chunk)
            agrees.extend(p["agree"] for p in chunk)
    qs, A, taus, agrees = map(np.array, (qs, A, taus, agrees))
    enc = Encoder()
    ec = enc.encode([p["cond_c"] for p in calib])
    ep_ = enc.encode([p["cond_p"] for p in calib])
    cos = (ec * ep_).sum(1)
    lex = np.array([len(set(p["cond_c"].lower().split()) & set(p["cond_p"].lower().split())) /
                    max(1, len(set(p["cond_c"].lower().split()) | set(p["cond_p"].lower().split()))) for p in calib])
    nprod = np.array([p["n_prod"] for p in calib])
    metrics = dict(
        n_calib=len(calib), agree_rate=float(agrees.mean()),
        auc_ahn=float(roc_auc_score(agrees, qs)), auc_cos=float(roc_auc_score(agrees, cos)),
        auc_lex=float(roc_auc_score(agrees, lex)),
        cidx_ahn=c_index(A, taus), cidx_cos=c_index(cos * nprod, taus), cidx_lex=c_index(lex * nprod, taus),
        spearman_ahn=float(spearmanr(A, taus)[0]), spearman_cos=float(spearmanr(cos * nprod, taus)[0]),
        mae_ahn=float(np.abs(A - taus).mean()), mae_const=float(np.abs(taus.mean() - taus).mean()))
    thr = {f"{e:.3f}": calibrate(qs, agrees, e) for e in EPS}
    cthr = {f"{e:.3f}": calibrate(cos, agrees, e) for e in EPS}
    cover = {f"{e:.3f}": dict(ahn=float((qs >= thr[f"{e:.3f}"]).mean()), cos=float((cos >= cthr[f"{e:.3f}"]).mean()),
                              ahn_err=float(1 - agrees[qs >= thr[f"{e:.3f}"]].mean()) if (qs >= thr[f"{e:.3f}"]).any() else 0,
                              cos_err=float(1 - agrees[cos >= cthr[f"{e:.3f}"]].mean()) if (cos >= cthr[f"{e:.3f}"]).any() else 0)
             for e in EPS}
    metrics.update(thresholds=thr, cos_thresholds=cthr, coverage=cover)
    print(json.dumps(metrics, indent=1), flush=True)
    state = model.state_dict()
    state["__meta__"] = dict(thresholds=thr, cos_thresholds=cthr, metrics=metrics)
    torch.save(state, os.path.join(RESULTS, "ahn", "ahn.pt"))
    np.savez(os.path.join(RESULTS, "ahn", "calib_scores.npz"), q=qs, A=A, tau=taus, agree=agrees, cos=cos, lex=lex,
             nprod=nprod)
    with open(os.path.join(RESULTS, "ahn", "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=1)


if __name__ == "__main__":
    main()
