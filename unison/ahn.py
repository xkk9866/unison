"""Agreement Hazard Network (AHN).

Inputs (four, all necessary): consumer condition c_w, producer condition c_u, the version delta
between the evidence they read, and the (possibly in-flight, possibly empty) producer output y_u.

Outputs, from one shared encoder:
  * token hazards h_t = P(consumer's greedy output diverges at t | agrees on y_u[:t]) for every
    observed LLM token of y_u, giving the survival S(t) = prod_{s<=t}(1-h_s);
  * the decision-agreement probability q = P(decision(y_w) == decision(y_u)), where a decision is
    the (relevant, answer) pair the agent consumes.
Both are trained self-supervised from execution logs: tau = LCP(y_w, y_u) is exactly the number
of draft tokens greedy verification accepts, and the decision label comes from parsing outputs.
"""
import difflib

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

MAX_LEN = 320


def version_delta(old_text, new_text):
    if old_text == new_text:
        return "none"
    a, b = old_text.split(), new_text.split()
    ops = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if tag != "equal":
            ops.append(f"{' '.join(a[i1:i2])} -> {' '.join(b[j1:j2])}")
    return "; ".join(ops)[:200] or "none"


def token_pieces(llm_tok, ids):
    """Per-token surface strings (their concatenation is the text the AHN reads)."""
    return [llm_tok.decode([t], skip_special_tokens=True) for t in ids]


class AHN(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.enc = AutoModel.from_pretrained(backbone)
        d = self.enc.config.hidden_size
        self.ctx = nn.Linear(d, d)
        self.haz = nn.Sequential(nn.Linear(d, d // 2), nn.GELU(), nn.Linear(d // 2, 1))
        self.agree = nn.Sequential(nn.Linear(d, d // 2), nn.GELU(), nn.Linear(d // 2, 1))

    def forward(self, input_ids, attention_mask, token_type_ids, gather_idx):
        H = self.enc(input_ids=input_ids, attention_mask=attention_mask,
                     token_type_ids=token_type_ids).last_hidden_state
        cls = H[:, 0]
        g = torch.gather(H, 1, gather_idx.clamp(min=0).unsqueeze(-1).expand(-1, -1, H.size(-1)))
        logit_h = self.haz(g + self.ctx(cls).unsqueeze(1)).squeeze(-1)
        logit_q = self.agree(cls).squeeze(-1)
        return logit_h, logit_q


class Featurizer:
    def __init__(self, enc_tok):
        self.tok = enc_tok

    def encode(self, cond_c, cond_p, delta, pieces):
        """pieces: per-LLM-token strings of the observed producer output (may be a prefix or empty)."""
        seg_a = f"{cond_c} [SEP] {cond_p} [SEP] {delta}"
        text = "".join(pieces) or "(running)"
        enc = self.tok(seg_a, text, truncation="only_second", max_length=MAX_LEN, return_offsets_mapping=True)
        seq = enc.sequence_ids()
        b_pos = [i for i, s in enumerate(seq) if s == 1]
        b_off = [enc["offset_mapping"][i] for i in b_pos]
        idx, j, c = [], 0, 0
        for p in pieces:
            s = c + (len(p) - len(p.lstrip())) if p.strip() else c
            c += len(p)
            while j < len(b_off) and b_off[j][1] <= s:
                j += 1
            idx.append(b_pos[j] if j < len(b_pos) else (b_pos[-1] if b_pos else 0))
        return enc["input_ids"], enc.get("token_type_ids", [0] * len(enc["input_ids"])), idx


def collate(batch, pad_id):
    L = max(len(b[0]) for b in batch)
    T = max(1, max(len(b[2]) for b in batch))
    ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
    tt = torch.zeros((len(batch), L), dtype=torch.long)
    am = torch.zeros((len(batch), L), dtype=torch.long)
    gi = torch.full((len(batch), T), -1, dtype=torch.long)
    for i, (a, t, g) in enumerate(batch):
        ids[i, :len(a)] = torch.tensor(a)
        tt[i, :len(t)] = torch.tensor(t)
        am[i, :len(a)] = 1
        if g:
            gi[i, :len(g)] = torch.tensor(g)
    return ids, am, tt, gi


def survival_loss(logit_h, gi, tau, n_obs):
    """Discrete-time survival NLL; observations beyond n_obs (or after a full match) are censored."""
    T = logit_h.size(1)
    pos = torch.arange(T, device=logit_h.device).unsqueeze(0)
    valid = gi >= 0
    log_s = nn.functional.logsigmoid(-logit_h)
    log_h = nn.functional.logsigmoid(logit_h)
    surv = valid & (pos < tau.unsqueeze(1)) & (pos < n_obs.unsqueeze(1))
    event = valid & (pos == tau.unsqueeze(1)) & (tau.unsqueeze(1) < n_obs.unsqueeze(1))
    nll = -(log_s * surv).sum(1) - (log_h * event).sum(1)
    return (nll / n_obs.clamp(min=1)).mean()


class AHNScorer:
    """Inference wrapper used by the online planner."""

    def __init__(self, ckpt, backbone, llm_tok, device="cuda"):
        self.enc_tok = AutoTokenizer.from_pretrained(backbone)
        state = torch.load(ckpt, map_location="cpu")
        self.meta = state.pop("__meta__", {}) if isinstance(state, dict) else {}
        self.model = AHN(backbone)
        self.model.load_state_dict(state)
        self.model.to(device).eval().half()
        self.feat = Featurizer(self.enc_tok)
        self.llm_tok = llm_tok
        self.device = device
        from .batcher import AsyncBatcher
        self._ab = AsyncBatcher(self._score_many)

    def _score_many(self, reqs):
        """reqs: list of (cond_c, cands); one forward over all their pairs."""
        flat, spans = [], []
        for cond_c, cands in reqs:
            spans.append((len(flat), len(cands)))
            flat.extend((cond_c, c) for c in cands)
        out = []
        for i in range(0, len(flat), 128):
            chunk = flat[i:i + 128]
            out.extend(self._score_pairs(chunk))
        return [out[s:s + n] for s, n in spans]

    async def ascore(self, cond_c, cands):
        if not cands:
            return []
        return await self._ab((cond_c, cands))

    @torch.no_grad()
    def _score_pairs(self, pairs):
        batch = [self.feat.encode(cc, cp, d, pcs) for cc, (cp, d, pcs) in pairs]
        ids, am, tt, gi = [x.to(self.device) for x in collate(batch, self.enc_tok.pad_token_id)]
        lh, lq = self.model(ids, am, tt, gi)
        h = torch.sigmoid(lh.float()).cpu().numpy()
        q = torch.sigmoid(lq.float()).cpu().numpy()
        res = []
        for i, (_, (_, _, pcs)) in enumerate(pairs):
            n = len(pcs)
            res.append((np.cumprod(1.0 - h[i, :n]) if n else np.zeros(0), float(q[i])))
        return res

    def threshold(self, eps):
        th = self.meta.get("thresholds", {})
        return th.get(f"{eps:.3f}", 1.01)

    @torch.no_grad()
    def score(self, cond_c, cands):
        """cands: list of (cond_p, delta, pieces). Returns list of (S over observed tokens, q)."""
        if not cands:
            return []
        batch = [self.feat.encode(cond_c, cp, d, pcs) for cp, d, pcs in cands]
        ids, am, tt, gi = [x.to(self.device) for x in collate(batch, self.enc_tok.pad_token_id)]
        lh, lq = self.model(ids, am, tt, gi)
        h = torch.sigmoid(lh.float()).cpu().numpy()
        q = torch.sigmoid(lq.float()).cpu().numpy()
        out = []
        for i, (_, _, pcs) in enumerate(cands):
            n = len(pcs)
            S = np.cumprod(1.0 - h[i, :n]) if n else np.zeros(0)
            out.append((S, float(q[i])))
        return out
