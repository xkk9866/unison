"""Embed each pool's corpus with bge-base-en-v1.5 (dense retrieval index)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

from unison.common import DATA, read_jsonl
from unison.retriever import Encoder

enc = Encoder()
for ds in ["hotpotqa", "musique"]:
    for split in os.environ.get("SPLITS", "eval,train").split(","):
        d = os.path.join(DATA, ds, split)
        corpus = read_jsonl(os.path.join(d, "corpus.jsonl"))
        emb = enc.encode([f"{c['title']}. {c['text']}" for c in corpus])
        np.save(os.path.join(d, "corpus_emb.npy"), emb.astype(np.float32))
        print(ds, split, emb.shape, flush=True)
