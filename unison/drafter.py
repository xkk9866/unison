"""Consensus drafter: the learned component of Unison.

A key computation (a plan or a verification decision) is decoded by the consumer's own model with speculative
verification. In an agent program the tokens of a key are almost always already written somewhere the system
holds: in the evidence and the question the consumer was given, or in what a peer concluded from the same evidence.
The drafter is a pointer into that record, driven by the consumer's state:

  record  the tokens in front of the consumer and its peers: the tail of the consumer's own prompt (evidence and
          question) and the outputs of other calls on the same evidence (verification) or the same question (plan),
          finished or still running
  state   once the consumer has made a pass, its hidden state at its last accepted token, exactly the state the
          model used to emit it, and that token. Before its first pass the consumer has no state, and the drafter
          uses its own reading of the record in its place

One softmax ranges over the record's slots and a small vocabulary of the tokens that recur across questions (the
format of a note or a plan). Anything else can only be drafted by pointing, so the drafter reads content and
generates form. After a token is emitted, the decoder is given the memory of the slot that follows the place the
token was read from: continuing a span is the default, leaving it is a decision of the state.

Because the drafter does not need the consumer's state, a key can be drafted before the consumer's prompt is
processed and verified in the very pass that processes it: a call whose draft holds takes one pass.

The target model verifies every proposal, so the drafter never changes an output; it changes how many iterations
and how much verification work an output costs. Its own confidence sets the draft horizon: a token is drafted
while the probability that the whole draft so far is accepted exceeds the engine's verification price at the
current load (unison_plugin, _verification_price).

A draft is short and the network is small, so what drafting costs is not arithmetic but the kernels the host
launches for every position. DraftProgram runs the decoding loop as a device program of fixed shape, captured
once and replayed with a single launch per `chunk` positions.
"""
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PAD, UNK = 0, 1  # vocabulary rows: no target / a token that is neither in the vocabulary nor in the record
N_SPECIAL = 2
BUCKETS = 16384  # identity embedding of any token without the lexicon: its id modulo BUCKETS


class Vocab:
    """Restricted output vocabulary: raw token ids that recur across questions, plus the reserved rows."""

    def __init__(self, ids, raw_size=160000):
        self.ids = [int(t) for t in ids]
        self.size = N_SPECIAL + len(self.ids)
        table = torch.full((raw_size,), UNK, dtype=torch.long)
        table[torch.tensor(self.ids, dtype=torch.long)] = torch.arange(N_SPECIAL, self.size)
        self.table = table
        self.raw_of = torch.tensor([-1] * N_SPECIAL + self.ids, dtype=torch.long)


def load_lexicon(model_dir, device="cpu"):
    """The target model's output embedding (the rows of its output head) as a frozen float16 table
    [vocabulary, h_dim]. Inside the engine the drafter reads the model's own tensor instead (no copy)."""
    import json
    import os
    from safetensors import safe_open
    idx = os.path.join(model_dir, "model.safetensors.index.json")
    fn = json.load(open(idx))["weight_map"]["lm_head.weight"] if os.path.exists(idx) else "model.safetensors"
    with safe_open(os.path.join(model_dir, fn), "pt") as f:
        return f.get_tensor("lm_head.weight").to(torch.float16).to(device)


class ConsensusDrafter(nn.Module):
    def __init__(self, vocab_ids, h_dim=4096, d=384, heads=8, ctx_len=320, max_peers=4, peer_len=48, dropout=0.1,
                 lexicon=True, buckets=BUCKETS, read=True):
        super().__init__()
        self.vocab = Vocab(vocab_ids)
        self.h_dim, self.d, self.buckets = h_dim, d, buckets
        self.ctx_len, self.max_peers, self.peer_len = ctx_len, max_peers, peer_len
        V = self.vocab.size
        # token identity (record slots and the decoder's input). With the lexicon, a token is the row of the target
        # model's own output head for it, the vector the consumer's state is compared with when the model emits
        # that token: the drafter learns one projection of that space and no per-token parameter, so a token it
        # never saw in training is as readable as any other. Without it: a learned table of hashed token ids.
        self.use_lexicon, self._lex, self._table = lexicon, None, None
        if lexicon:
            self.e_norm = nn.LayerNorm(h_dim)
            self.e_proj = nn.Linear(h_dim, d)
        else:
            self.emb = nn.Embedding(buckets, d)
        self.pos = nn.Embedding(max(ctx_len, peer_len), d)
        self.seg = nn.Embedding(2, d)                    # 0: the consumer's own context, 1: a peer's output
        self.local = nn.Conv1d(d, d, 3)                  # a slot sees its two predecessors (n-gram context)
        self.enc = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=dropout, batch_first=True, norm_first=True)
        self.h_norm = nn.LayerNorm(h_dim)
        self.h_proj = nn.Linear(h_dim, d)
        # the state of a consumer that has not made a pass yet: one attention read of the record
        self.use_read = read
        if read:
            self.read_q = nn.Parameter(torch.randn(1, 1, d) / math.sqrt(d))
            self.read = nn.MultiheadAttention(d, heads, batch_first=True)
            self.r_norm = nn.LayerNorm(d)
        self.s0 = nn.Linear(2 * d, d)
        self.cell = nn.GRUCell(3 * d, d)
        self.q_ptr = nn.Linear(d, d)
        self.k_mem = nn.Linear(d, d)
        self.out = nn.Linear(d, V)
        self.drop = nn.Dropout(dropout)
        vmask = torch.zeros(V)
        vmask[PAD] = float("-inf")
        # one exponent on the acceptance estimate, fitted on development questions after training: the estimate
        # has to mean what the verification price assumes on questions the drafter has not seen
        self.register_buffer("beta", torch.ones(()))
        self.register_buffer("vmask", vmask, persistent=False)
        self.register_buffer("vtable", self.vocab.table, persistent=False)
        self.register_buffer("vraw", self.vocab.raw_of, persistent=False)
        self._programs = {}

    def train(self, mode=True):
        if mode:  # the projected lexicon and the captured programs hold the weights they were built from
            self._table, self._programs = None, {}
        return super().train(mode)

    # ------------------------------------------------------------------ the record
    def encode(self, tok, kind):
        """tok: LongTensor [N, T] of raw token ids (-1 = none) of one segment kind -> slot memories [N, T, d]."""
        T = tok.shape[1]
        valid = tok >= 0
        x = self.embed(tok) + self.pos.weight[:T] + self.seg.weight[kind]
        x = x * valid.unsqueeze(-1).to(x.dtype)
        x = x + self.local(F.pad(x.transpose(1, 2), (2, 0))).transpose(1, 2)
        m = valid.clone()
        m[:, 0] |= ~valid.any(1)  # an empty segment keeps one (unused) position so that attention is defined
        return self.enc(x, src_key_padding_mask=~m)

    @staticmethod
    def _successor(mem, valid):
        """Memory of the slot that follows each slot of its segment (zero after the segment's last token)."""
        nxt = torch.zeros_like(mem)
        nxt[:, :-1] = mem[:, 1:] * valid[:, 1:].unsqueeze(-1).to(mem.dtype)
        return nxt

    def record(self, ctx, peers, mem_ctx=None):
        """ctx: [B, C] raw ids of the consumer's prompt tail, right-aligned (-1 = none); peers: [B, P, L] raw ids
        (-1 = none). Returns what decoding needs: successor memories [B, M, d], pointer keys [B, M, d], the slot
        mask [B, M], the scatter index of the joint distribution [B, V + M] (a slot adds its probability to its
        token's vocabulary row, or to the first slot holding the same token), the raw token of every joint index
        [B, V + M], the slots' raw tokens [B, M] and the slot memories [B, M, d]. mem_ctx: precomputed
        encode(ctx, 0)."""
        B, P, L = peers.shape
        mc = self.encode(ctx, 0) if mem_ctx is None else mem_ctx
        pf = peers.reshape(B * P, L)
        mp = self.encode(pf, 1)
        nxt = torch.cat([self._successor(mc, ctx >= 0), self._successor(mp, pf >= 0).reshape(B, P * L, self.d)], 1)
        mem = torch.cat([mc, mp.reshape(B, P * L, self.d)], 1)
        slots = torch.cat([ctx, peers.reshape(B, P * L)], 1)
        mask = slots >= 0
        V = self.vocab.size
        rows = self.vtable[slots.clamp(min=0)]
        same = (slots[:, :, None] == slots[:, None, :]) & mask[:, None, :]
        first = same.to(torch.uint8).argmax(-1)  # first slot that holds the same token
        slot_ext = torch.where((rows == UNK) & mask, V + first, rows)
        idx_all = torch.cat([torch.arange(V, device=slots.device).expand(B, V), slot_ext], 1)
        raw_all = torch.cat([self.vraw.expand(B, V), slots], 1)
        return nxt, self.k_mem(mem), mask, idx_all, raw_all, slots, mem

    def target_index(self, tgt, slots, mask):
        """Joint index of every target token [B, K]: its vocabulary row, else the first slot that holds it, else
        UNK (a token the drafter cannot produce: the calibrated answer there is to stop). PAD where tgt < 0."""
        V = self.vocab.size
        rows = self.vtable[tgt.clamp(min=0)]
        same = (slots[:, None, :] == tgt[:, :, None]) & mask[:, None, :]
        first = same.to(torch.uint8).argmax(-1)
        idx = torch.where((rows == UNK) & same.any(-1), V + first, rows)
        return torch.where(tgt < 0, torch.full_like(idx, PAD), idx)

    # ------------------------------------------------------------------ the consumer's state
    def consumer(self, h, has_h, mem, mask):
        """What the decoder knows of the consumer [B, d]: its hidden state where it has one (has_h [B]), else the
        drafter's reading of the record (one attention read with a learned query)."""
        dt = self.h_proj.weight.dtype
        hp = self.h_proj(self.h_norm(h.to(dt)))
        if self.use_read:
            keep = mask.clone()
            keep[:, self.ctx_len - 1] = True  # the last prompt token: the read is defined on an empty record
            r, _ = self.read(self.read_q.expand(h.shape[0], 1, self.d).to(mem.dtype), mem, mem,
                             key_padding_mask=~keep, need_weights=False)
            r = self.r_norm(r.squeeze(1)).to(hp.dtype)
        else:
            r = torch.zeros_like(hp)
        return torch.where(has_h.unsqueeze(-1), hp, r)

    def set_lexicon(self, table):
        """table: [vocabulary, h_dim], the target model's output embedding. It stays where it is (the engine's own
        tensor, or load_lexicon's copy), is never trained and is not part of the drafter's parameters."""
        self._lex, self._table = table.detach(), None

    @torch.no_grad()
    def freeze_lexicon(self, chunk=16384):
        """A trained drafter projects every row of the lexicon once: token identity becomes one table read."""
        if not self.use_lexicon or self._lex is None:
            return self
        w = self.e_proj.weight
        out = torch.empty(self._lex.shape[0], self.d, dtype=w.dtype, device=w.device)
        for i in range(0, out.shape[0], chunk):
            out[i:i + chunk] = self.e_proj(self.e_norm(self._lex[i:i + chunk].to(w.device, w.dtype)))
        self._table = out
        return self

    def embed(self, raw):
        raw = raw.clamp(min=0)
        if not self.use_lexicon:
            return self.emb(raw % self.buckets)
        if self._table is not None and not self.training:
            return self._table[raw]
        dt = self.e_proj.weight.dtype
        if raw.numel() > 32768:  # a batch of records repeats its tokens: every distinct token is projected once
            u, inv = torch.unique(raw, return_inverse=True)
            return self.e_proj(self.e_norm(self._lex[u].to(dt)))[inv]
        return self.e_proj(self.e_norm(self._lex[raw].to(dt)))

    def init_state(self, hp, y_last):
        return torch.tanh(self.s0(torch.cat([hp, self.embed(y_last)], -1)))

    # ------------------------------------------------------------------ one decoding step
    def dist(self, s, kmem, mask, idx_all):
        """Joint distribution over the vocabulary and the record's slots [B, V + M], and the same distribution
        with every token's probability collected at one index [B, V + M]."""
        lv = self.out(self.drop(s)) + self.vmask
        lp = torch.bmm(kmem, self.q_ptr(s).unsqueeze(-1)).squeeze(-1) / math.sqrt(self.d)
        joint = torch.cat([lv, lp.masked_fill(~mask, float("-inf"))], 1).float().softmax(-1)
        return torch.zeros_like(joint).scatter_add_(1, idx_all, joint), joint

    def advance(self, s, tok, joint, slots, nxt, hp):
        """State after emitting `tok`: the token's identity and the memory that follows the place it was read."""
        V = self.vocab.size
        w = (joint[:, V:] * (slots == tok.unsqueeze(-1))).to(nxt.dtype)
        c = torch.bmm(w.unsqueeze(1), nxt).squeeze(1)
        return self.cell(torch.cat([self.embed(tok), c, hp], -1).to(s.dtype), s)

    # ------------------------------------------------------------------ training
    def forward(self, h, y_last, ctx, peers, tgt, has_h=None, use_ctx=True, use_peers=True):
        """Teacher-forced log-probability of every target token [B, K] (tgt: raw ids, -1 past the key's end), and
        the targets' joint indices (PAD where there is no target). has_h [B]: the windows whose consumer has a
        state (all, if omitted)."""
        if not use_ctx:
            ctx = torch.full_like(ctx, -1)
        if not use_peers:
            peers = torch.full_like(peers, -1)
        if has_h is None:
            has_h = torch.ones(y_last.shape[0], dtype=torch.bool, device=y_last.device)
        nxt, kmem, mask, idx_all, _, slots, mem = self.record(ctx, peers)
        hp = self.consumer(h, has_h, mem, mask)
        tix = self.target_index(tgt, slots, mask)
        s = self.init_state(hp, y_last)
        out = []
        for t in range(tgt.shape[1]):
            ext, joint = self.dist(s, kmem, mask, idx_all)
            out.append(torch.log(ext.gather(1, tix[:, t:t + 1]) + 1e-9))
            s = self.advance(s, tgt[:, t], joint, slots, nxt, hp)
        return torch.cat(out, 1), tix

    # ------------------------------------------------------------------ drafting
    @torch.no_grad()
    def propose(self, h, y_last, ctx, peers, k, price, eos, has_h=None, use_h=True, use_ctx=True, use_peers=True,
                step_price=0.0, mem_ctx=None, return_q=False, look=8):
        """Greedy drafts for a batch of any size (DraftProgram.run without a captured graph). has_h [B]: the
        requests that have a state; use_h=False withholds it from all."""
        B = y_last.shape[0]
        if not use_ctx:
            ctx, mem_ctx = torch.full_like(ctx, -1), None
        if not use_peers:
            peers = torch.full_like(peers, -1)
        if has_h is None:
            has_h = torch.ones(B, dtype=torch.bool, device=y_last.device)
        if not use_h:
            has_h = torch.zeros_like(has_h)
        prog = DraftProgram(self, B, eos, chunk=look, graph=False)
        prog.load(h, has_h, y_last, ctx, self.encode(ctx, 0) if mem_ctx is None else mem_ctx, peers)
        out = prog.run(B, k, price, step_price, return_q)
        self.last_steps, self.last_seconds = prog.last_steps, prog.last_seconds
        return out

    def program(self, n, eos, sizes=(8, 32)):
        """The captured program with the fewest rows that holds n requests (the largest, if none does)."""
        rows = next((r for r in sizes if r >= n), sizes[-1])
        if rows not in self._programs:
            self._programs[rows] = DraftProgram(self, rows, eos)
        return self._programs[rows]


class DraftProgram:
    """The drafter's decoding loop for `rows` requests as a program of fixed shape on the device.

    Inputs live in the program's own tensors (load); the record is assembled and `chunk` positions are decoded by
    one captured graph, further positions by a second one that continues from the state the first left. The host
    reads the result of a chunk once and applies the horizon: for every request the positions whose cumulative
    acceptance estimate q_j = (prod_{i<=j} p_max(i))^beta stayed at or above the price of a draft position. That
    price is `price` (verifying the token) plus `step_price` divided by the number of drafts still being extended
    (the drafter's own decoding step, shared by them). Without a graph (graph=False, or no CUDA device) the same
    code runs directly."""

    def __init__(self, net, rows, eos, chunk=8, graph=True):
        w = net.out.weight
        dev, dt = w.device, w.dtype
        self.net, self.rows, self.chunk = net, rows, chunk
        self.graph = bool(graph and w.is_cuda)
        C, P, L = net.ctx_len, net.max_peers, net.peer_len
        self.h = torch.zeros(rows, net.h_dim, dtype=dt, device=dev)
        self.has_h = torch.zeros(rows, dtype=torch.bool, device=dev)
        self.y = torch.zeros(rows, dtype=torch.long, device=dev)
        self.ctx = torch.full((rows, C), -1, dtype=torch.long, device=dev)
        self.ctx[:, -1] = 0  # a row that holds no request still has a defined record
        self.mctx = torch.zeros(rows, C, net.d, dtype=dt, device=dev)
        self.pe = torch.full((rows, P, L), -1, dtype=torch.long, device=dev)
        self.eos = torch.tensor(sorted(eos), dtype=torch.long, device=dev)
        self.s = torch.zeros(rows, net.d, dtype=dt, device=dev)
        self.beta = float(net.beta)
        self.last_steps, self.last_seconds = 0, 0.0
        if self.graph:
            with torch.no_grad():
                for _ in range(2):  # lazy initializations happen outside the capture
                    self._start()
                    self._more()
                torch.cuda.synchronize()
                self.g0 = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.g0):
                    self.out0 = self._start()
                self.g1 = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.g1):
                    self.out1 = self._more()

    def load(self, h, has_h, y, ctx, mctx, pe):
        """Requests 0..n-1: hidden states [n, h_dim] (any row for a request without one; None if no request has
        one), has_h [n], last tokens [n], prompt tails [n, C] with their slot memories [n, C, d], peers [n, P, L]."""
        n = y.shape[0]
        if h is not None:
            self.h[:n].copy_(h)
        self.has_h[:n].copy_(has_h)
        self.y[:n].copy_(y)
        self.ctx[:n].copy_(ctx)
        self.mctx[:n].copy_(mctx)
        self.pe[:n].copy_(pe)
        return n

    def _start(self):
        net = self.net
        nxt, kmem, mask, idx_all, raw_all, slots, mem = net.record(self.ctx, self.pe, self.mctx)
        hp = net.consumer(self.h, self.has_h, mem, mask)
        self.fix = (nxt, kmem, mask, idx_all, raw_all, slots, hp, torch.isin(raw_all, self.eos))
        self.s.copy_(net.init_state(hp, self.y))
        return self._more()

    def _more(self):
        """`chunk` greedy positions from the stored state. Returns one tensor [3 * chunk, rows]: the raw tokens,
        their flags (1: a token the drafter can emit, 2: an end marker) and their probabilities."""
        net = self.net
        nxt, kmem, mask, idx_all, raw_all, slots, hp, eos_all = self.fix
        s, toks, flags, ps = self.s, [], [], []
        for _ in range(self.chunk):
            ext, joint = net.dist(s, kmem, mask, idx_all)
            pmax, idx = ext.max(-1)
            ix = idx.unsqueeze(-1)
            raw = raw_all.gather(1, ix).squeeze(-1)
            toks.append(raw)
            flags.append((idx >= N_SPECIAL).long() + 2 * eos_all.gather(1, ix).squeeze(-1).long())
            ps.append(pmax)
            s = net.advance(s, raw, joint, slots, nxt, hp)
        self.s.copy_(s)
        return torch.cat([torch.stack(toks + flags).double(), torch.stack(ps).double()])

    @torch.no_grad()
    def run(self, n, k, price, step_price=0.0, return_q=False):
        """Drafts of the first n loaded requests, at most k positions each (lists of raw token ids); with
        return_q also the estimate q_j of every drafted position. last_steps and last_seconds: the positions the
        device decoded and the time the whole call took."""
        t_in = time.perf_counter()
        S = self.chunk
        q, alive, length = np.ones(n), np.ones(n, dtype=bool), np.zeros(n, dtype=np.int64)
        T, Q, steps, done = [], [], 0, 0
        while steps < k and alive.any():
            if self.graph:
                (self.g1 if done else self.g0).replay()
                out = (self.out1 if done else self.out0).cpu().numpy()
            else:
                out = (self._more() if done else self._start()).cpu().numpy()
            done += S
            tok, fl, p = out[:S, :n], out[S:2 * S, :n].astype(np.int64), out[2 * S:, :n]
            T.append(tok)
            for j in range(min(S, k - steps)):
                q = q * p[j] ** self.beta
                emit = alive & (fl[j] & 1 > 0) & (q >= price + step_price / max(1, int(alive.sum())))
                length += emit
                alive = emit & (fl[j] < 2)
                Q.append(q)
                steps += 1
        toks = np.concatenate(T).astype(np.int64) if T else np.zeros((0, n), dtype=np.int64)
        drafts = [toks[:length[b], b].tolist() for b in range(n)]
        self.last_steps, self.last_seconds = done, time.perf_counter() - t_in
        if return_q:
            qs = np.stack(Q) if Q else np.zeros((0, n))
            return drafts, [qs[:length[b], b].tolist() for b in range(n)]
        return drafts


# ---------------------------------------------------------------------- data
def peer_tensor(peer_lists, max_peers, peer_len):
    """[P, L] LongTensor of raw ids (-1 = none) from up to `max_peers` token lists, most relevant first."""
    out = torch.full((max_peers, peer_len), -1, dtype=torch.long)
    for i, p in enumerate(peer_lists[:max_peers]):
        p = list(p)[:peer_len]
        if p:
            out[i, :len(p)] = torch.tensor(p, dtype=torch.long)
    return out


def ctx_tensor(prompt_ids, ctx_len):
    """[C] LongTensor: the last `ctx_len` tokens of the consumer's prompt, right-aligned (-1 = none)."""
    out = torch.full((ctx_len,), -1, dtype=torch.long)
    p = torch.as_tensor(prompt_ids[-ctx_len:], dtype=torch.long)
    if p.numel():
        out[-p.numel():] = p
    return out


def nll(logp, tix):
    """Mean negative log-likelihood over the positions that have a target."""
    keep = tix != PAD
    return -(logp * keep).sum() / keep.sum().clamp(min=1)
