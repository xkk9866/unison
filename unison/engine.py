"""AsyncLLM (vLLM 0.8.5 V1) wrapper: streaming generation, per-step statistics, cost profiling."""
import asyncio
import os
import time
import uuid

import numpy as np

os.environ.setdefault("VLLM_USE_V1", "1")
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.inputs import TokensPrompt
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.metrics.loggers import StatLoggerBase

K_SPEC = int(os.environ.get("UNISON_K_SPEC", "128"))  # engine cap; history speculators use k=32 per request

# Replay requests only need the raw (token, logprob) rows of the replayed span. vLLM's frontend would
# otherwise build one Python dict per prompt position on the event loop; keep the raw arrays instead.
from vllm.v1.engine import logprobs as _lpm


def _raw_prompt_logprobs(self, tensors):
    self.prompt_logprobs = ("raw", tensors.logprob_token_ids.numpy(), tensors.logprobs.numpy())


_lpm.LogprobsProcessor._update_prompt_logprobs = _raw_prompt_logprobs


class StepLog(StatLoggerBase):
    """Collects one record per engine iteration."""
    records = []

    def __init__(self, vllm_config, engine_index=0):
        pass

    def record(self, scheduler_stats, iteration_stats):
        if iteration_stats is None:
            return
        sd = scheduler_stats.spec_decoding_stats
        pc = getattr(scheduler_stats, "prefix_cache_stats", None)  # tokens looked up / served by the prefix cache
        StepLog.records.append((iteration_stats.iteration_timestamp, scheduler_stats.num_running_reqs,
                                scheduler_stats.num_waiting_reqs, iteration_stats.num_generation_tokens,
                                iteration_stats.num_prompt_tokens,
                                sd.num_draft_tokens if sd else 0, sd.num_accepted_tokens if sd else 0,
                                getattr(pc, "queries", 0) if pc else 0, getattr(pc, "hits", 0) if pc else 0))

    def log_engine_initialized(self):
        pass


class Engine:
    def __init__(self, model_path, gpu_util=0.78, max_num_seqs=256, k_spec=K_SPEC, kv_blocks=None):
        # kv_blocks: fixed KV-cache capacity (blocks of 16 tokens) so that every run has the same capacity
        # whatever the memory situation of the card; None lets vLLM size it from gpu_util.
        kv_blocks = kv_blocks if kv_blocks is not None else os.environ.get("UNISON_KV_BLOCKS")
        self.kv_blocks = int(kv_blocks) if kv_blocks else None
        args = AsyncEngineArgs(
            model=model_path, gpu_memory_utilization=gpu_util, max_model_len=4096,
            enable_prefix_caching=True, max_num_seqs=max_num_seqs, max_num_batched_tokens=8192,
            disable_log_stats=False, disable_log_requests=True, seed=0,
            num_gpu_blocks_override=self.kv_blocks,
            speculative_config={"method": "ngram", "num_speculative_tokens": k_spec,
                                "prompt_lookup_max": 4, "prompt_lookup_min": 2})
        self.llm = AsyncLLM.from_engine_args(args, stat_loggers=[StepLog])
        self.k_spec = k_spec
        self.tok = None
        self.eos = None
        self.n_running = 0
        self.step_ewma = 0.03
        self.cost = dict(a=0.02, delta=1e-4, ctok=2e-5)

    async def init(self):
        self.tok = await self.llm.get_tokenizer()
        self.eos = sorted({self.tok.eos_token_id, self.tok.convert_tokens_to_ids("<|im_end|>"),
                           self.tok.convert_tokens_to_ids("<|endoftext|>"),
                           self.tok.get_vocab().get("<|eot_id|>")} - {None})  # <|eot_id|>: Llama-3 only

    def chat_ids(self, messages):
        kw = {"enable_thinking": False}
        return self.tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, **kw)

    async def generate(self, prompt_ids, max_tokens, extra_args=None, rid=None, on_tokens=None):
        """Greedy generation with token streaming. Returns (rid, tokens, timing dict)."""
        rid = rid or uuid.uuid4().hex
        ea = dict(extra_args or {})
        ea["eos"] = self.eos
        sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, output_kind=RequestOutputKind.DELTA,
                            extra_args=ea, skip_special_tokens=True)
        toks, steps = [], 0
        t0 = time.time()
        t_first = None
        last = t0
        self.n_running += 1
        try:
            async for out in self.llm.generate(TokensPrompt(prompt_token_ids=prompt_ids), sp, request_id=rid):
                new = list(out.outputs[0].token_ids)
                now = time.time()
                if new:
                    if t_first is None:
                        t_first = now
                    elif len(new) == 1:  # a plain decode step
                        self.step_ewma = 0.95 * self.step_ewma + 0.05 * (now - last)
                    steps += 1
                    last = now
                    toks.extend(new)
                    if on_tokens is not None:
                        on_tokens(toks)
        finally:
            self.n_running -= 1
        t1 = time.time()
        # rid: the engine's per-request accounting (passes, accepted drafts) is joined to the call's record by it
        return rid, toks, dict(t_submit=t0, t_first=t_first or t1, t_end=t1, steps=steps, rid=rid)

    async def replay(self, prompt_ids, cand, rid=None):
        """Consumer-conditioned replay: one teacher-forced pass of a producer output `cand` under the
        consumer prompt. The consumer prompt prefix comes from the prefix cache; only `cand` is
        computed. Returns (lp, lp1, top1): per position t the log-probability of cand[t], of the
        consumer's own argmax, and that argmax token, plus timing."""
        rid = rid or uuid.uuid4().hex
        P = len(prompt_ids)
        sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=1, output_kind=RequestOutputKind.FINAL_ONLY,
                            extra_args={"mode": "none", "replay_from": P - 1, "eos": self.eos}, detokenize=False)
        t0 = time.time()
        self.n_running += 1
        try:
            async for out in self.llm.generate(TokensPrompt(prompt_token_ids=list(prompt_ids) + list(cand)), sp,
                                               request_id=rid):
                final = out
        finally:
            self.n_running -= 1
        _, ids_, lps_ = final.prompt_logprobs
        rows = slice(P - 1, P - 1 + len(cand))  # row j scores prompt token j+1
        lp = lps_[rows, 0].astype(np.float32)   # column 0: the replayed token, column 1: the argmax
        lp1 = np.maximum(lps_[rows, 1].astype(np.float32), lp)
        top1 = ids_[rows, 1].astype(np.int64)
        top1 = np.where(lps_[rows, 1] >= lps_[rows, 0], top1, np.asarray(cand, dtype=np.int64))
        nxt = list(final.outputs[0].token_ids)
        return lp, lp1, top1, (nxt[0] if nxt else None), dict(t_submit=t0, t_end=time.time())

    def decode(self, toks):
        return self.tok.decode(toks, skip_special_tokens=True)

    async def reset(self):
        await self.llm.reset_prefix_cache()

    async def profile(self):
        """Fit step_time = a + delta*B + ctok*D (B running sequences, D draft tokens in the step)."""
        rng = np.random.default_rng(0)
        base = self.chat_ids([{"role": "user", "content": "Describe the history of the printing press in detail."}])
        rows = []
        # The step time of a configuration is the decode span of its requests divided by their decode steps (every
        # step emits one token: junk drafts are rejected). A span covers ~47 iterations, so the delivery jitter of
        # the frontend, which distorts the gap between two consecutive iterations, cancels out.
        for B, d in [(1, 0), (8, 0), (32, 0), (64, 0), (128, 0), (32, 8), (32, 24), (64, 16), (8, 32), (128, 8)]:
            ea = {"mode": "junk", "k": d} if d else {"mode": "none"}
            best = None
            for _ in range(2):
                ids = [base + [int(x) for x in rng.integers(1000, 30000, 4)] for _ in range(B)]
                outs = await asyncio.gather(*[self.generate(p, 48, ea) for p in ids])
                dts = [(tm["t_end"] - tm["t_first"]) / (len(toks) - 1) for _, toks, tm in outs if len(toks) >= 24]
                if len(dts) >= max(1, B // 2):
                    t = float(np.median(dts))
                    best = t if best is None else min(best, t)
            if best is not None:
                rows.append((B, B * d, best))
        X = np.array([[1.0, b, dd] for b, dd, _ in rows])
        y = np.array([t for _, _, t in rows])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        self.cost = dict(a=float(coef[0]), delta=float(max(coef[1], 1e-6)), ctok=float(max(coef[2], 1e-7)))
        self.step_ewma = float(coef[0] + coef[1] * 16)
        await self.reset()
        return self.cost, rows
