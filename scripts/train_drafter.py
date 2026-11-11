"""Train and evaluate the consensus drafter from collection runs (run_exp.py --methods collect --collect DIR).

A collection run executes the agent program without sharing; the engine plugin records, for every call, the hidden
state that emitted each output token, and rebuild_prompts.py recovers the call's prompt. Training windows are built
from the run's own timeline: at the moment a call emitted its t-th token, the other calls of its group (same
passage / same question) had emitted the prefixes that the drafter would have seen in the engine; identical-prompt
calls are excluded (identity sharing never issues them) and verification peers are cut at their decision boundary
(a key request ends there). The record of a window is the tail of the call's own prompt plus those peer prefixes.

Questions are split three ways: training, development (the epoch with the lowest development loss is kept) and
held-out (every reported number). No question contributes calls to more than one part.

  python scripts/train_drafter.py --runs v17_collect/hotpotqa v17_collect/musique --out results/drafter
"""
import argparse
import copy
import glob
import gzip
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch

from unison.common import BOUNDARY_FIELD, MODELS, RESULTS
from unison.drafter import ConsensusDrafter, nll
from unison.executors_v14 import decision_end

P, L, K, C = 4, 48, 32, 320  # peers, peer window, draft cap (the engine's per-request cap), own-context window
VARIANTS = {"cd": {}, "cd-nopeer": dict(use_peers=False), "cd-noctx": dict(use_ctx=False), "cd-noh": dict(use_h=False)}
PRE = -1  # window position of a pre-draft: the key is drafted before the call's first pass (no state yet)


def load_run(path, tok, fmt, eos):
    """Calls (with hidden states and prompts) and group timelines of one collection run directory.

    Every call gets its key sequence r["key"], the tokens the engine delivers for it under Unison: a plan is its
    whole output (through its end-of-sequence token); a verification call ends at its decision boundary, so its
    key is the decision followed by the end marker (the payload after the boundary is a separate computation).
    r["dend"] is the number of key tokens a peer can see (the key without the end marker). Only calls whose
    prompt was recovered exactly are drafted from; every call can be a peer."""
    calls = {}
    ns = "/".join(os.path.normpath(path).split(os.sep)[-2:])  # a group belongs to its run: names repeat across runs
    with gzip.open(os.path.join(path, "collect.calls.jsonl.gz"), "rt") as f:
        for r in map(json.loads, f):
            r["group"] = f"{ns}:{r['group']}"
            out = r["out"]
            if r["kind"] == "verify":
                r["dend"] = decision_end(tok, out, BOUNDARY_FIELD.get(fmt, "answer:"))
                r["key"] = out[:r["dend"]] + [eos[0]]
            else:
                r["dend"] = len(out) - (1 if out and out[-1] in eos else 0)
                r["key"] = out
            calls[r["rid"]] = r
    z = np.load(os.path.join(path, "collect.prompts.npz"))
    ids, off = z["ids"], z["off"].tolist()  # an .npz member is read from disk on every access: read each once
    for rid, a, b, ok in zip(z["rids"].tolist(), off[:-1], off[1:], z["exact"].tolist()):
        if ok and rid in calls:
            calls[rid]["prompt"] = ids[a:b]
    print(f"{path}: {len(calls)} calls, {sum('prompt' in r for r in calls.values())} prompts", flush=True)
    H = {}
    for fn in sorted(glob.glob(os.path.join(path, "shards", "shard_*.pt"))):
        for rid, (h, toks) in torch.load(fn).items():
            r = calls.get(rid)
            if r is None or "prompt" not in r:
                continue
            m = min(h.shape[0], len(toks), len(r["out"]))
            if m and list(toks[:m].tolist()) == r["out"][:m]:
                H[rid] = h[:min(m, len(r["key"]))].clone()  # only key positions are ever drafted from
    groups = defaultdict(list)
    for r in calls.values():
        groups[r["group"]].append(r)
    for g in groups.values():
        g.sort(key=lambda r: r["t_first"])
    return calls, H, groups


def distinct(calls, H):
    """A prompt is drafted once: identity sharing serves every later call with the same prompt without a request.
    Only the first call of each distinct prompt is a training or evaluation example; all calls stay peers."""
    first = {}
    for rid in sorted(H, key=lambda r: calls[r]["t_first"]):
        first.setdefault(calls[rid]["phash"], rid)
    return {rid: H[rid] for rid in first.values()}


def peers_at(r, tau, groups):
    """Peer prefixes visible at time tau: other calls of the group that had started, newest first, each cut at
    what it had emitted by tau (linear in time between first and last token) and at its boundary."""
    out = []
    for p in reversed(groups[r["group"]]):
        if p["rid"] == r["rid"] or p["phash"] == r["phash"] or p["t_first"] > tau:
            continue
        span = max(p["t_end"] - p["t_first"], 1e-6)
        n = int(math.ceil(p["n_out"] * min(1.0, (tau - p["t_first"]) / span)))
        n = max(1, min(n, p["dend"], L))
        out.append(p["out"][:n])
        if len(out) >= P:
            break
    return out


def windows(calls, H, groups, rids):
    """(call, position, visible peers). Position t >= 0: the call has emitted key[:t + 1] and its state is the one
    that emitted key[t]. Position PRE: the call has just arrived and the record is all there is."""
    W = []
    for rid in rids:
        r, h = calls[rid], H[rid]
        T = min(h.shape[0], len(r["key"]) - 1)  # positions of the key that have a next key token
        span = r["t_end"] - r["t_first"]
        W.append((rid, PRE, peers_at(r, r["t_arrive"], groups)))
        for t in range(T):
            tau = r["t_first"] + span * (t / max(1, r["n_out"] - 1))
            W.append((rid, t, peers_at(r, tau, groups)))
    return W


class Pack:
    """Windows as tensors, built once: the hidden-state row of every window (one float16 matrix in host memory; a
    pre-draft window has none), its call's prompt tail, its peer tokens and the continuation of its key (the last
    token the consumer holds and the K tokens after it, -1 past the key's end). A batch is then a gather, with no
    per-window Python work."""

    def __init__(self, W, calls, H, dev):
        rids = sorted({rid for rid, _, _ in W})
        off, n, cid = {}, 0, {}
        ctx = np.full((max(1, len(rids)), C), -1, dtype=np.int32)
        for i, rid in enumerate(rids):
            off[rid], cid[rid] = n, i
            n += H[rid].shape[0]
            p = calls[rid]["prompt"][-C:]
            ctx[i, C - len(p):] = p
        self.H = torch.cat([H[rid] for rid in rids]) if rids else torch.zeros(0, 1)
        self.row = torch.tensor([off[rid] + max(t, 0) for rid, t, _ in W], dtype=torch.long)
        self.has = torch.tensor([t >= 0 for _, t, _ in W], dtype=torch.bool, device=dev)
        self.cid = torch.tensor([cid[rid] for rid, _, _ in W], dtype=torch.long, device=dev)
        pe = np.full((len(W), P, L), -1, dtype=np.int32)
        seq = np.full((len(W), K + 1), -1, dtype=np.int32)
        for b, (rid, t, peers) in enumerate(W):
            for i, p in enumerate(peers[:P]):
                p = p[:L]
                pe[b, i, :len(p)] = p
            key = calls[rid]["key"]
            s = key[t:t + K + 1] if t >= 0 else [int(calls[rid]["prompt"][-1])] + key[:K]
            seq[b, :len(s)] = s
        self.ctx = torch.from_numpy(ctx).to(dev)
        self.pe, self.seq, self.dev, self.n = torch.from_numpy(pe).to(dev), torch.from_numpy(seq).to(dev), dev, len(W)

    def batch(self, idx):
        """h, last token, own context, peers, targets and has-a-state flags of the windows `idx` (a LongTensor)."""
        h = self.H[self.row[idx]].to(self.dev)
        idx = idx.to(self.dev)
        seq = self.seq[idx].long()
        kmax = max(1, int((seq[:, 1:] >= 0).sum(1).max()))
        return h, seq[:, 0], self.ctx[self.cid[idx]].long(), self.pe[idx].long(), seq[:, 1:kmax + 1], self.has[idx]


def accepted(draft, truth):
    n = 0
    for a, b in zip(draft, truth):
        if a != b:
            break
        n += 1
    return n


def rule_draft(own_next, peers, t):
    """The v15 rule: the longest peer output that extends the consumer's own prefix (own[:t+1]), continued."""
    best = []
    for p in peers:
        if len(p) > t + 1 and list(p[:t + 1]) == own_next[:t + 1] and len(p) - t - 1 > len(best):
            best = list(p[t + 1:])
    return best


def lookup_draft(prompt, key, peers, t, k=10, max_n=4, min_n=2):
    """Prompt lookup over the same record, without a model: the longest n-gram suffix of what the consumer has in
    front of it (its prompt and the key tokens it has emitted) is searched in the record (the peers' outputs and
    the consumer's own text); the tokens after its latest earlier occurrence are the draft."""
    seq = []
    for p in peers:
        seq.extend(p)
        seq.append(-1)  # no match across two texts
    seq.extend(prompt[-C:].tolist())
    seq.extend(key[:t + 1])
    a = np.asarray(seq, dtype=np.int64)
    for n in range(max_n, min_n - 1, -1):
        if len(a) <= n:
            continue
        win = np.lib.stride_tricks.sliding_window_view(a[:-1], n)
        hit = np.nonzero((win == a[-n:]).all(1))[0]
        if len(hit):
            s = int(hit[-1]) + n
            d = a[s:s + k].tolist()
            if -1 in d:
                d = d[:d.index(-1)]
            return d
    return []


@torch.no_grad()
def evaluate(net, W, calls, H, dev, prices, eos, major=None, bs=256):
    """Offline acceptance per window: drafted / accepted tokens for the learned drafter (full and with one input
    removed) at each verification price, for the peer-copy rule and for n-gram lookup over the record, split by
    call kind and by whether any peer was visible. Keys: "<drafter>@<price>|<kind>|<avail>" for the windows of a
    call that has made its first pass and "pre:<drafter>..." for its pre-draft window (the model-free drafters
    have no price; "static" is the most frequent training key of the call's kind, a pre-draft only)."""
    net.eval()
    net.freeze_lexicon()
    agg = defaultdict(lambda: Counter())
    # calibration of the acceptance estimate: drafted positions of the unpriced drafter (price 0, horizon 8),
    # binned by their cumulative estimate q_j, against whether the position was accepted; also by position.
    # Index 0: windows after the first pass, 1: pre-draft windows
    cal, cal_n = np.zeros((2, 10, 2)), np.zeros((2, 10))
    pos = np.zeros((2, 8, 3))
    per = defaultdict(dict)  # drafter -> (call, position) -> (drafted, accepted)
    pk = Pack(W, calls, H, dev)
    t_draft, n_draft, n_steps = 0.0, 0, 0
    for i in range(0, len(W), bs):
        w = W[i:i + bs]
        h, y, ctx, pe, tgt, has = pk.batch(torch.arange(i, min(i + bs, len(W))))
        outs = {}
        for pr in prices:
            for name, kw in VARIANTS.items():
                if name == "cd" and dev == "cuda":
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                outs[f"{name}@{pr}"] = net.propose(h, y, ctx, pe, K, pr, eos, has_h=has, **kw)
                if name == "cd" and dev == "cuda":
                    torch.cuda.synchronize()
                    t_draft, n_draft, n_steps = t_draft + time.perf_counter() - t0, n_draft + 1, n_steps + net.last_steps
        free, qs = net.propose(h, y, ctx, pe, 8, 0.0, eos, has_h=has, return_q=True)
        for b, (rid, t, peers) in enumerate(w):
            r = calls[rid]
            truth = r["key"][t + 1:]
            ver = r["kind"] == "verify"
            pre = int(t == PRE)
            last = len(r["key"]) - (2 if ver else 1)  # the key is complete once this many of its tokens are accepted
            key = (r["kind"], "peers" if peers else "alone")
            outs_b = {name: d[b] for name, d in outs.items()}
            outs_b["rule"] = rule_draft(r["key"], peers, t)
            outs_b["lookup"] = lookup_draft(r["prompt"], r["key"], peers, t)
            if pre and major:
                outs_b["static"] = list(major.get(r["kind"], []))
            for name, d in outs_b.items():
                if ver and d and d[-1] in eos:  # a key's end marker is the drafter's stop, not a token to verify
                    d = d[:-1]
                a = accepted(d, truth)
                per[name][(rid, t)] = (len(d), a)
                for kk in (key, (r["kind"], "all"), ("all", "all")):
                    c = agg[("pre:" * pre + name,) + kk]
                    c["n"] += 1
                    c["drafted"] += len(d)
                    c["accepted"] += a
                    c["full"] += int(a == len(d) and len(d) > 0)
                    c["first"] += int(a >= 1)
                    c["whole"] += int(a >= last - t - 1)  # the draft completes the key in the pass that verifies it
            a = accepted(free[b], truth)
            for j, qj in enumerate(qs[b]):
                k = min(int(qj * 10), 9)
                cal[pre, k] += (qj, float(j < a))
                cal_n[pre, k] += 1
                pos[pre, j] += (1, float(j < a), qj)
    res = {}
    for (name, kind, avail), c in sorted(agg.items()):
        res[f"{name}|{kind}|{avail}"] = dict(n=c["n"], drafted=c["drafted"], accepted=c["accepted"],
                                             acc_rate=c["accepted"] / max(1, c["drafted"]),
                                             draft_per_step=c["drafted"] / max(1, c["n"]),
                                             accepted_per_step=c["accepted"] / max(1, c["n"]),
                                             full=c["full"] / max(1, c["n"]), first=c["first"] / max(1, c["n"]),
                                             whole=c["whole"] / max(1, c["n"]))
    # decoding a whole key: the first pass processes the prompt, verifies the pre-draft (if any) and emits its
    # accepted tokens plus one; every later pass verifies the draft made at the key's current position and emits
    # its accepted tokens plus one. The passes a key takes are its sequential cost, the drafted tokens its
    # verification work (without drafts a key of n tokens takes n passes).
    cover = defaultdict(list)
    for rid, t, _ in W:
        cover[rid].append(t)

    def simulate(pre, cont):
        sim = defaultdict(Counter)
        for rid, ts in cover.items():
            r = calls[rid]
            last = min(len(r["key"]) - (2 if r["kind"] == "verify" else 1), max(ts) + 1)
            p = steps = drafted = acc = 0
            if pre is not None and (rid, PRE) in pre:
                drafted, acc = pre[(rid, PRE)]
                p = acc
            one = int(p >= last)
            while p < last and (rid, p) in cont:
                d, a = cont[(rid, p)]
                steps, drafted, acc, p = steps + 1, drafted + d, acc + a, p + a + 1
            for kind in (r["kind"], "all"):
                c = sim[kind]
                c["calls"] += 1
                c["steps"] += steps
                c["one"] += one
                c["tokens"] += last + 1
                c["drafted"] += drafted
                c["accepted"] += acc
        return {kind: dict(calls=c["calls"], passes=1 + c["steps"] / max(1, c["calls"]),
                           one_pass=c["one"] / max(1, c["calls"]), tokens=c["tokens"] / max(1, c["calls"]),
                           drafted=c["drafted"] / max(1, c["calls"]), accepted=c["accepted"] / max(1, c["calls"]))
                for kind, c in sim.items()}

    none = {}
    for name, tab in sorted(per.items()):  # no pre-draft: what a drafter that needs the consumer's state can do
        for kind, v in simulate(None, tab).items():
            res[f"sim|{name}|{kind}"] = v
    pairs = [(name, name) for name in sorted(per) if name != "static"]  # the same drafter before and after
    pairs += [(a, b) for b in sorted(per) if b.startswith("cd@") for a in ("static", "rule") if a in per]
    pairs += [(a, b) for a, b in (("static", "lookup"), ("rule", "lookup"), ("static", "none")) if a in per]
    for a, b in pairs:
        for kind, v in simulate(per[a], per.get(b, none)).items():
            res[f"sim0|{a}+{b}|{kind}"] = v
    for st, tag in ((0, ""), (1, "_pre")):
        res["calibration" + tag] = [dict(bin=k / 10, n=int(cal_n[st, k]), predicted=float(cal[st, k, 0] / max(1, cal_n[st, k])),
                                         accepted=float(cal[st, k, 1] / max(1, cal_n[st, k]))) for k in range(10)]
        tot = cal_n[st].sum()
        res["ece" + tag] = float(sum(cal_n[st, k] / max(1, tot) * abs(cal[st, k, 0] - cal[st, k, 1]) / max(1, cal_n[st, k])
                                     for k in range(10)))
        res["by_position" + tag] = [dict(position=j + 1, n=int(pos[st, j, 0]), accepted=float(pos[st, j, 1] / max(1, pos[st, j, 0])),
                                         predicted=float(pos[st, j, 2] / max(1, pos[st, j, 0]))) for j in range(8)]
    if n_draft:  # the drafter's own cost on this device, for a batch of `bs` requests, run without a captured graph
        res["cost"] = dict(batch=bs, ms_per_call=1e3 * t_draft / n_draft, steps_per_call=n_steps / n_draft,
                           ms_per_step=1e3 * t_draft / max(1, n_steps))
    return res


def show(ev, tag=""):
    for k, v in ev.items():
        if k.endswith("|all|all") or k.startswith("sim") or k == "cost" or (k.startswith("pre:") and k.endswith("|all")):
            print(tag + k, json.dumps({kk: round(vv, 3) if isinstance(vv, float) else vv for kk, vv in v.items()}), flush=True)
    print(tag + "ece", round(ev["ece"], 4), "ece_pre", round(ev["ece_pre"], 4), flush=True)


def majority_keys(calls, rids, eos):
    """The most frequent key of each call kind among the calls `rids` (without its end marker)."""
    cnt = defaultdict(Counter)
    for rid in rids:
        r = calls[rid]
        cnt[r["kind"]][tuple(t for t in r["key"] if t not in eos)] += 1
    return {kind: list(c.most_common(1)[0][0]) for kind, c in cnt.items()}


def lexicon_for(net, model, dev):
    """Give the drafter the target model's output embedding (read from the model's checkpoint; frozen)."""
    if net.use_lexicon:
        from unison.drafter import load_lexicon
        net.set_lexicon(load_lexicon(MODELS[model], dev))
    return net


@torch.no_grad()
def fit_calibration(net, pk, eos, bs, horizon=8, stride=3):
    """The exponent beta of the acceptance estimate, fitted on development questions: the drafter drafts freely
    (no price, `horizon` positions) from development windows; every drafted position gives its raw estimate
    prod p_max and whether it was accepted; beta minimizes the cross-entropy of estimate^beta against acceptance
    (one scalar, golden-section search). Returns beta and the cross-entropy before and after."""
    net.eval()
    net.freeze_lexicon()
    net.beta.fill_(1.0)
    Q, A = [], []
    for i in range(0, pk.n, bs * stride):
        h, y, ctx, pe, tgt, has = pk.batch(torch.arange(i, min(i + bs, pk.n)))
        drafts, qs = net.propose(h, y, ctx, pe, horizon, 0.0, eos, has_h=has, return_q=True)
        truth = tgt.tolist()
        for d, q, t in zip(drafts, qs, truth):
            a = accepted(d, t)
            Q.extend(q)
            A.extend(j < a for j in range(len(q)))
    Q = np.clip(np.asarray(Q, dtype=np.float64), 1e-6, 1 - 1e-6)
    A = np.asarray(A, dtype=np.float64)

    def f(beta):
        p = np.clip(Q ** beta, 1e-6, 1 - 1e-6)
        return float(-(A * np.log(p) + (1 - A) * np.log(1 - p)).mean())

    g = (math.sqrt(5) - 1) / 2
    a, b = 0.25, 4.0
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(40):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = f(d)
    beta = (a + b) / 2
    net.beta.fill_(beta)
    return beta, f(1.0), f(beta), len(Q)


def dev_loss(net, pk, bs, amp):
    net.eval()
    vl, vn = 0.0, 0
    with torch.no_grad():
        for i in range(0, pk.n, bs):
            h, y, ctx, pe, tgt, has = pk.batch(torch.arange(i, min(i + bs, pk.n)))
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                logp, tix = net(h, y, ctx, pe, tgt, has_h=has)
            vl += float(nll(logp, tix))
            vn += 1
    return vl / max(1, vn)


def main(a):
    from transformers import AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    amp = dev == "cuda"
    if dev == "cpu":  # the drafter is small enough to be trained without a GPU
        torch.set_num_threads(a.threads)
    tok = AutoTokenizer.from_pretrained(MODELS[a.model])
    eos = sorted({tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")} - {None})
    calls, H, groups = {}, {}, {}
    for run in a.runs:
        path = os.path.join(RESULTS, run)
        c, h, g = load_run(path, tok, a.note, eos)
        print(f"{run}: {len(c)} calls, {len(h)} with hidden states and prompt, {len(g)} groups", flush=True)
        calls.update(c)
        H.update(h)
        groups.update(g)
    n_all = len(H)
    H = distinct(calls, H)
    print(f"distinct prompts: {len(H)} of {n_all} calls with hidden states", flush=True)
    # whole questions are held out: every call of a development or held-out question is unseen in training
    rng = random.Random(0)
    qids = sorted({str(calls[rid].get("qid")) for rid in H})
    rng.shuffle(qids)
    n_part = max(1, len(qids) // 10)
    held, devq = set(qids[:n_part]), set(qids[n_part:2 * n_part])
    part = lambda rid: "held" if str(calls[rid].get("qid")) in held else "dev" if str(calls[rid].get("qid")) in devq else "train"
    train = sorted(rid for rid in H if part(rid) == "train")
    dv = sorted(rid for rid in H if part(rid) == "dev")
    val = sorted(rid for rid in H if part(rid) == "held")
    print(f"questions: {len(qids) - 2 * n_part} train, {n_part} development, {n_part} held out; "
          f"calls: {len(train)} train, {len(dv)} development, {len(val)} held out", flush=True)
    Wtr, Wdv, Wva = (windows(calls, H, groups, x) for x in (train, dv, val))
    print(f"windows: train {len(Wtr)}, development {len(Wdv)}, held out {len(Wva)}", flush=True)
    # the vocabulary holds the tokens that recur across questions (the form of a key); what is specific to a
    # question can only be drafted by pointing at the record
    seen = defaultdict(set)
    for rid in train:
        q = str(calls[rid].get("qid"))
        for t in set(calls[rid]["key"]):
            seen[t].add(q)
    major = majority_keys(calls, train, eos)
    if a.eval_only:  # the trained drafter of --out on the same held-out questions, at other prices
        ck = torch.load(os.path.join(a.out, "cd.pt"), map_location="cpu")
        cfg = dict(ck["cfg"])
        cfg.setdefault("lexicon", False)
        cfg.setdefault("read", False)
        net = lexicon_for(ConsensusDrafter(ck["vocab_ids"], **cfg).to(dev), a.model, dev)
        state = dict(ck["state"])
        state.setdefault("beta", torch.ones(()))
        net.load_state_dict(state)
        ev = evaluate(net, Wva, calls, H, dev, a.prices, eos, major)
        show(ev)
        with open(os.path.join(a.out, f"eval_{a.eval_only}.json"), "w") as f:
            json.dump(dict(eval=ev, prices=a.prices, n_val_windows=len(Wva), calls_val=len(val), runs=a.runs), f, indent=1)
        return
    vocab_ids = sorted(t for t, qs in seen.items() if len(qs) >= a.min_questions)
    cfg = dict(h_dim=a.h_dim, d=a.d, heads=8, ctx_len=C, max_peers=P, peer_len=L, dropout=a.dropout,
               lexicon=bool(a.lexicon), read=bool(a.read))
    net = lexicon_for(ConsensusDrafter(vocab_ids, **cfg).to(dev), a.model, dev)
    n_par = sum(p.numel() for p in net.parameters())
    cover = Counter()
    for rid in val:
        for t in calls[rid]["key"]:
            cover["in" if t in seen and len(seen[t]) >= a.min_questions else "out"] += 1
    print(f"vocab {net.vocab.size}, parameters {n_par / 1e6:.1f}M, held-out key tokens outside the vocabulary "
          f"{100 * cover['out'] / max(1, sum(cover.values())):.1f}%", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
    steps = a.epochs * math.ceil(len(Wtr) / a.bs) + 1
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.05)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    os.makedirs(a.out, exist_ok=True)
    log, best = [], None
    t_pack = time.time()
    if a.finish:
        # the run was interrupted after some of its development checks: the state kept at the best of them is in
        # cd.pt, and its output (--finish) lists every check made; calibration and evaluation continue from that
        # state
        ck = torch.load(os.path.join(a.out, "cd.pt"), map_location="cpu")
        assert list(ck["vocab_ids"]) == list(vocab_ids), "the stored drafter was trained on other runs"
        log = [json.loads(l) for l in open(a.finish) if l.startswith('{"epoch"')]
        assert log[:len(ck["log"])] == list(ck["log"]), "the output is not that of the run that wrote cd.pt"
        kept = min(log, key=lambda e: e["dev_loss"])
        assert abs(kept["epoch"] - ck["epoch"]) < 1e-6, "cd.pt is not the best state of its own log"
        state = dict(ck["state"])
        state.setdefault("beta", torch.ones(()))
        net.load_state_dict(state)
        best = (kept["dev_loss"], ck["epoch"], copy.deepcopy(net.state_dict()))
        print(f"finishing an interrupted run: {len(log)} development checks up to epoch {log[-1]['epoch']:.2f}",
              flush=True)
        ptr, pdv = None, Pack(Wdv, calls, H, dev)
    else:
        ptr, pdv = Pack(Wtr, calls, H, dev), Pack(Wdv, calls, H, dev)
    print(f"packed in {time.time() - t_pack:.0f}s; device {dev}", flush=True)
    gen = torch.Generator().manual_seed(0)
    t_train = time.time() - (log[-1]["seconds"] if a.finish else 0.0)
    n_it = math.ceil(len(Wtr) / a.bs)
    check = {round(n_it * (k + 1) / a.checks) for k in range(a.checks)}  # development checks within an epoch
    for ep in range(0 if a.finish else a.epochs):
        net.train()
        perm = torch.randperm(ptr.n, generator=gen)
        t0, tot, nb, seg, ns = time.time(), 0.0, 0, 0.0, 0
        for i in range(0, ptr.n, a.bs):
            h, y, ctx, pe, tgt, has = ptr.batch(perm[i:i + a.bs])
            # each input is dropped for a share of the batches, so that the same model can be measured without it
            u = rng.random()
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                logp, tix = net(h, y, ctx, pe, tgt, has_h=has & (not u < 0.1), use_ctx=not 0.1 <= u < 0.2,
                                use_peers=not 0.2 <= u < 0.3)
            loss = nll(logp, tix)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot, seg, nb, ns = tot + float(loss), seg + float(loss), nb + 1, ns + 1
            if nb % 200 == 0:
                print(f"ep {ep} it {nb} loss {tot / nb:.3f} ({time.time() - t0:.0f}s)", flush=True)
            if nb in check:  # the passes over the training questions are few: the best state can lie inside one
                dl = dev_loss(net, pdv, a.bs, amp)
                net.train()
                log.append(dict(epoch=ep + nb / n_it, train_loss=seg / max(1, ns), dev_loss=dl,
                                seconds=time.time() - t_train))
                seg, ns = 0.0, 0
                print(json.dumps(log[-1]), flush=True)
                if best is None or dl < best[0]:
                    best = (dl, ep + nb / n_it, copy.deepcopy(net.state_dict()))
                    torch.save(dict(vocab_ids=vocab_ids, cfg=cfg, state=best[2], log=log, epoch=best[1]),
                               os.path.join(a.out, "cd.pt"))
    train_seconds = time.time() - t_train
    net.load_state_dict(best[2])
    net.train()  # drops what was derived from other weights (the projected lexicon)
    print(f"kept epoch {best[1]:.2f} (development loss {best[0]:.4f})", flush=True)
    beta, l0, l1, n_cal = fit_calibration(net, pdv, eos, a.bs)
    print(f"calibration exponent {beta:.3f} on {n_cal} development draft positions: cross-entropy {l0:.4f} -> {l1:.4f}",
          flush=True)
    torch.save(dict(vocab_ids=vocab_ids, cfg=cfg, state=net.state_dict(), log=log, epoch=best[1]),
               os.path.join(a.out, "cd.pt"))
    ev = evaluate(net, Wva, calls, H, dev, a.prices, eos, major)
    show(ev)
    with open(os.path.join(a.out, "eval.json"), "w") as f:
        json.dump(dict(eval=ev, log=log, n_train_windows=len(Wtr), n_dev_windows=len(Wdv), n_val_windows=len(Wva),
                       vocab=net.vocab.size, parameters=n_par, prices=a.prices, cfg=cfg, runs=a.runs, device=dev,
                       questions_train=len(qids) - 2 * n_part, questions_dev=n_part, questions_held_out=n_part,
                       calls_train=len(train), calls_dev=len(dv), calls_val=len(val), epoch_kept=best[1],
                       epochs=log[-1]["epoch"] if a.finish else a.epochs, interrupted=bool(a.finish),
                       dev_loss=best[0], beta=beta, cal_ce_raw=l0, cal_ce=l1, cal_positions=n_cal,
                       calls_distinct=len(H), calls_with_state=n_all,
                       train_seconds=train_seconds, oov_held_out=cover["out"] / max(1, sum(cover.values()))), f, indent=1)
    if a.transfer:  # runs the drafter never saw (another dataset): every call of them is evaluated
        del H, Wtr, Wva, Wdv, ptr, pdv
        tc, tH, tg = {}, {}, {}
        for run in a.transfer:
            c, h, g = load_run(os.path.join(RESULTS, run), tok, a.note, eos)
            print(f"transfer {run}: {len(c)} calls, {len(h)} with hidden states and prompt", flush=True)
            tc.update(c)
            tH.update(h)
            tg.update(g)
        tH = distinct(tc, tH)
        rids = sorted(tH)
        random.Random(1).shuffle(rids)
        Wt = windows(tc, tH, tg, rids[:a.transfer_calls])
        evt = evaluate(net, Wt, tc, tH, dev, a.prices, eos, major)
        show(evt, "transfer ")
        with open(os.path.join(a.out, "eval_transfer.json"), "w") as f:
            json.dump(dict(eval=evt, n_windows=len(Wt), prices=a.prices, runs=a.runs, transfer=a.transfer), f, indent=1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", default=["v17_collect/hotpotqa", "v17_collect/musique"])
    p.add_argument("--out", default=os.path.join(RESULTS, "drafter"))
    p.add_argument("--model", default="qwen3-8b")
    p.add_argument("--note", default="dfp")
    p.add_argument("--h_dim", type=int, default=4096)
    p.add_argument("--d", type=int, default=384)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--checks", type=int, default=4, help="development-loss checks per epoch (the best state is kept)")
    p.add_argument("--lexicon", type=int, default=1,
                   help="token identity: 1 = the target model's output embedding (frozen), 0 = a learned hashed table")
    p.add_argument("--bs", type=int, default=256)
    p.add_argument("--lr", type=float, default=7e-4)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--threads", type=int, default=32, help="CPU threads when no GPU is visible")
    p.add_argument("--min_questions", type=int, default=4,
                   help="a token enters the vocabulary when the keys of this many training questions contain it")
    p.add_argument("--read", type=int, default=1,
                   help="1 = a consumer without a state is given the drafter's reading of the record, 0 = nothing")
    p.add_argument("--prices", type=float, nargs="+", default=[0.03, 0.1, 0.3],
                   help="verification prices (break-even acceptance) at which the offline horizon is evaluated")
    p.add_argument("--transfer", nargs="*", default=[], help="collection runs that are evaluated but never trained on")
    p.add_argument("--transfer_calls", type=int, default=3000)
    p.add_argument("--eval_only", default="", help="name: evaluate the drafter stored in --out, write eval_<name>.json")
    p.add_argument("--finish", default="",
                   help="output of an interrupted run: keep the best state stored in --out, calibrate and evaluate it")
    main(p.parse_args())
