"""Surprisal traces of consumer-conditioned replay and the Surprisal Survival Network (SSN).

Replaying a producer output y_u under the consumer's prompt gives, at every position t,
  sigma_t = log p_w(argmax) - log p_w(y_u[t])  >= 0   (0 iff the consumer would emit y_u[t] itself)
  ell_t   = -log p_w(y_u[t])
The SSN maps the trace to the probability that the consumer's own (independent) execution reaches the
same decision as y_u:

  P(agree | trace) = exp( - sum_t c_t * h(sigma_t) ),   h(0) = 0, h increasing,  c_t >= 0

h is the per-token divergence intensity and c_t the *consequence* of a divergence at t, read by a
small bidirectional GRU from the trace itself and the output schema's decision-field indicator d_t.
Because h(0) = 0, a trace with no divergence has P = 1 exactly: the replay certificate is the
boundary case of the learned law, not a separate rule.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DECISION_FIELD = "Relevant:"


def decision_mask(tok, ids):
    """d_t = 1 for tokens of the decision fields (the 'Relevant:' / 'Answer:' lines of the note)."""
    pieces = [tok.decode([t], skip_special_tokens=True) for t in ids]
    text = "".join(pieces)
    j = text.find(DECISION_FIELD)
    m = np.zeros(len(ids), np.uint8)
    if j < 0:
        return m
    pos = 0
    for t, p in enumerate(pieces):
        if pos + len(p) > j:
            m[t] = 1
        pos += len(p)
    return m


def features(lp, lp1, dmask):
    lp = np.asarray(lp, np.float32)
    sig = np.maximum(np.asarray(lp1, np.float32) - lp, 0.0)
    return np.stack([sig, np.minimum(-lp, 30.0), np.asarray(dmask, np.float32)], 1)  # [n, 3]


class SSN(nn.Module):
    def __init__(self, hid=32, K=4):
        super().__init__()
        self.inp = nn.Linear(3, hid)
        self.gru = nn.GRU(hid, hid, batch_first=True, bidirectional=True)
        self.cons = nn.Sequential(nn.Linear(2 * hid + 3, hid), nn.GELU(), nn.Linear(hid, 1))
        self.a = nn.Parameter(torch.zeros(K))
        self.b = nn.Parameter(torch.linspace(-2, 1, K))

    def intensity(self, sig):
        a, b = F.softplus(self.a), F.softplus(self.b)
        return ((1 - torch.exp(-sig.unsqueeze(-1) * b)) * a).sum(-1)  # h(0)=0, increasing

    def log_survival(self, x, mask):
        """x [B, T, 3] (sigma, -logp, d); mask [B, T]. Returns log P(agree) [B] and per-token hazard."""
        z = torch.stack([torch.log1p(x[..., 0]), x[..., 1] / 10.0, x[..., 2]], -1)
        h, _ = self.gru(torch.tanh(self.inp(z)))
        c = F.softplus(self.cons(torch.cat([h, z], -1)).squeeze(-1))
        lam = c * self.intensity(x[..., 0]) * mask
        return -lam.sum(1), lam

    def forward(self, x, mask):
        return self.log_survival(x, mask)[0].exp()


class CausalSSN(nn.Module):
    """Anticipatory survival law over a replay *prefix*.

    log P(agree | trace[0:t]) = - sum_{s<=t} c_s h(sigma_s)  -  R_t
    The first term is the observed survival of the replayed prefix (as in SSN, with a causal GRU so
    that it never looks ahead); R_t >= 0 is the learned hazard of the part of the candidate that has
    not been replayed yet. One pass gives the law at every horizon t, so the replay can stop early.
    """

    def __init__(self, hid=32, K=4, causal=True):
        super().__init__()
        self.inp = nn.Linear(4, hid)
        self.gru = nn.GRU(hid, hid, batch_first=True)
        self.cons = nn.Sequential(nn.Linear(hid + 4, hid), nn.GELU(), nn.Linear(hid, 1))
        self.fut = nn.Sequential(nn.Linear(hid + 4, hid), nn.GELU(), nn.Linear(hid, 1))
        self.a = nn.Parameter(torch.zeros(K))
        self.b = nn.Parameter(torch.linspace(-2, 1, K))

    def log_survival_all(self, x, mask):
        T = x.shape[1]
        pos = torch.arange(T, device=x.device, dtype=torch.float32)[None, :, None].expand(x.shape[0], T, 1) / 64.0
        z = torch.cat([torch.log1p(x[..., :1]), x[..., 1:2] / 10.0, x[..., 2:3], pos], -1)
        h, _ = self.gru(torch.tanh(self.inp(z)))
        hz = torch.cat([h, z], -1)
        a, b = F.softplus(self.a), F.softplus(self.b)
        inten = ((1 - torch.exp(-x[..., :1] * b)) * a).sum(-1)
        lam = F.softplus(self.cons(hz).squeeze(-1)) * inten * mask
        return -(torch.cumsum(lam, 1) + F.softplus(self.fut(hz).squeeze(-1)))  # [B, T]

    def forward(self, x, mask):
        """P(agree) at the last valid position of each row."""
        ls = self.log_survival_all(x, mask)
        last = (mask.sum(1).long() - 1).clamp(min=0)
        return ls.gather(1, last[:, None])[:, 0].exp()


def build_law(meta):
    arch = dict(meta.get("arch", {}))
    if arch.pop("causal", False):
        return CausalSSN(**arch)
    return SSN(**arch)


def pad(feats):
    T = max(len(f) for f in feats)
    x = np.zeros((len(feats), T, 3), np.float32)
    m = np.zeros((len(feats), T), np.float32)
    for i, f in enumerate(feats):
        x[i, :len(f)] = f
        m[i, :len(f)] = 1
    return torch.from_numpy(x), torch.from_numpy(m)


class SSNScorer:
    def __init__(self, path, device="cpu"):
        torch.set_num_threads(1)  # a 15k-parameter law; never contend with the serving engine for cores
        st = torch.load(path, map_location="cpu")
        self.meta = st.pop("__meta__")
        self.net = build_law(self.meta)
        self.net.load_state_dict(st)
        self.net.eval().to(device)
        self.device = device

    @torch.no_grad()
    def score(self, feats):
        if not feats:
            return []
        x, m = pad(feats)
        return self.net(x.to(self.device), m.to(self.device)).cpu().numpy().tolist()
