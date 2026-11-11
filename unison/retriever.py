"""Dense retrieval with bge-base-en-v1.5 (CLS pooling, cosine)."""
import asyncio
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from .batcher import AsyncBatcher
from .common import EMBED_MODEL, read_jsonl

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Encoder:
    def __init__(self, path=EMBED_MODEL, device="cuda"):
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModel.from_pretrained(path, torch_dtype=torch.float16).to(device).eval()
        self.device = device
        self.pool = ThreadPoolExecutor(max_workers=1)  # one thread owns the model and tokenizer
        self.aencode_one = AsyncBatcher(lambda texts: list(self.encode(texts)), pool=self.pool)
        # A semantic cache keys on the sub-question the retriever has just embedded for its search. When a cache
        # executor sets this to a dict, the retriever's pass also embeds the bare sub-question (the cache's key,
        # without the retrieval instruction) and leaves the vector here, so a lookup needs no encoder pass.
        self.keys = None

    @torch.no_grad()
    def encode(self, texts, batch_size=128, max_length=512):
        outs = []
        for i in range(0, len(texts), batch_size):
            b = self.tok(texts[i:i + batch_size], padding=True, truncation=True, max_length=max_length,
                         return_tensors="pt").to(self.device)
            h = self.model(**b).last_hidden_state[:, 0]
            outs.append(torch.nn.functional.normalize(h.float(), dim=-1).cpu().numpy())
        return np.concatenate(outs) if outs else np.zeros((0, 768), np.float32)

    async def aencode(self, texts):
        return np.stack([await self.aencode_one(t) for t in texts]) if len(texts) == 1 else \
            np.stack(await _gather([self.aencode_one(t) for t in texts]))


async def _gather(cs):
    import asyncio
    return await asyncio.gather(*cs)


class Retriever:
    def __init__(self, pool_dir, encoder):
        self.corpus = read_jsonl(os.path.join(pool_dir, "corpus.jsonl"))
        self.emb = torch.from_numpy(np.load(os.path.join(pool_dir, "corpus_emb.npy"))).to(encoder.device).half()
        self.enc = encoder
        self.k = 4
        self.asearch_one = AsyncBatcher(self._search_batch, pool=encoder.pool)
        # cuBLAS allocates a handle and workspace per thread on first use. Doing it on the pool thread here keeps
        # that allocation out of the run, where the card may have no memory left for it.
        encoder.pool.submit(self._search_batch, ["warm-up " * 48] * 32).result()

    @torch.no_grad()
    def _search_batch(self, queries):
        texts, keys = [QUERY_PREFIX + x for x in queries], self.enc.keys
        new = [x for x in dict.fromkeys(queries) if x not in keys] if keys is not None else []
        e = self.enc.encode(texts + new)
        for x, v in zip(new, e[len(texts):]):
            keys[x] = v
        q = torch.from_numpy(e[:len(texts)]).to(self.emb.device).half()
        return (q @ self.emb.T).topk(self.k, dim=-1).indices.cpu().tolist()

    def search(self, queries, k):
        self.k = k
        return self._search_batch(queries)

    async def asearch(self, query, k):
        self.k = k
        return await self.asearch_one(query)
