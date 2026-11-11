"""Calibrate commit thresholds on the statistic the planner actually thresholds.

Replays held-out (never trained on) passages of the independent-execution logs in arrival order:
for every verification call, the candidate producers are the earlier calls on the same passage
(identical prompts excluded -- they take the certified path); the planner commits to the
argmax-q producer, so the calibration sample is (max q, decision agreement with that producer).
The same replay with condition cosine gives the SemCRC baseline thresholds.
"""
import gzip
import json
import os
import sys
import zlib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch
from transformers import AutoTokenizer

from train_ahn import EPS, calibrate
from unison.agents import World
from unison.ahn import AHNScorer, token_pieces, version_delta
from unison.common import AHN_BACKBONE, DATA, MODELS, RESULTS, normalize_answer, parse_note
from unison.retriever import Encoder

MAX_CANDS = 12


def main():
    tok = AutoTokenizer.from_pretrained(MODELS["qwen3-8b"])
    ck = os.path.join(RESULTS, "ahn", "ahn.pt")
    ahn = AHNScorer(ck, AHN_BACKBONE, tok)
    enc = Encoder()
    Q, QA, C, CA = [], [], [], []
    for ds in ["hotpotqa", "musique"]:
        split, tag = os.environ.get("CALIB_SPLIT", "train"), os.environ.get("CALIB_TAG", "ahndata")
        world = World(os.path.join(DATA, ds, split))
        with gzip.open(os.path.join(RESULTS, tag, ds, "noshare.calls.jsonl.gz"), "rt") as f:
            recs = sorted((json.loads(l) for l in f), key=lambda r: r["t_arrive"])
        recs = [r for r in recs if r["kind"] == "verify"]
        hist = {}
        for r in recs:
            text = tok.decode(r["out"], skip_special_tokens=True)
            rel, ans = parse_note(text)
            r["dec"] = (rel, normalize_answer(ans))
            prior = [u for u in hist.get(r["pid"], [])[-MAX_CANDS:] if u["phash"] != r["phash"]]
            same = any(u["phash"] == r["phash"] for u in hist.get(r["pid"], []))
            held_out = split != "train" or zlib.crc32(f"{ds}-{r['pid']}".encode()) % 100 < 15
            if prior and held_out and not same:
                items = [(u["cond"], version_delta(world.text(r["pid"], u["version"]), world.text(r["pid"], r["version"])),
                          u.setdefault("pieces", token_pieces(tok, u["out"]))) for u in prior]
                sc = ahn._score_pairs([(r["cond"], it) for it in items])
                j = int(np.argmax([q for _, q in sc]))
                Q.append(sc[j][1])
                QA.append(int(prior[j]["dec"] == r["dec"]))
                e = enc.encode([r["cond"]] + [u["cond"] for u in prior])
                sims = e[1:] @ e[0]
                j = int(np.argmax(sims))
                C.append(float(sims[j]))
                CA.append(int(prior[j]["dec"] == r["dec"]))
            hist.setdefault(r["pid"], []).append(r)
        print(ds, "decisions", len(Q), flush=True)
    Q, QA, C, CA = map(np.array, (Q, QA, C, CA))
    thr = {f"{e:.3f}": calibrate(Q, QA, e) for e in EPS}
    cthr = {f"{e:.3f}": calibrate(C, CA, e) for e in EPS}
    cover = {k: dict(ahn=float((Q >= thr[k]).mean()), ahn_err=float(1 - QA[Q >= thr[k]].mean()) if (Q >= thr[k]).any() else 0.0,
                     cos=float((C >= cthr[k]).mean()), cos_err=float(1 - CA[C >= cthr[k]].mean()) if (C >= cthr[k]).any() else 0.0)
             for k in thr}
    state = torch.load(ck, map_location="cpu")
    meta = state["__meta__"]
    meta.setdefault("pair_thresholds", meta["thresholds"])
    meta.setdefault("pair_cos_thresholds", meta["cos_thresholds"])
    meta["thresholds"], meta["cos_thresholds"] = thr, cthr
    meta["deploy_calib"] = dict(n=len(Q), agree_rate=float(QA.mean()), coverage=cover)
    torch.save(state, ck)
    np.savez(os.path.join(RESULTS, "ahn", "deploy_calib.npz"), q=Q, qa=QA, c=C, ca=CA)
    print(json.dumps(dict(n=len(Q), agree=float(QA.mean()), thresholds=thr, cos_thresholds=cthr, coverage=cover), indent=1))


if __name__ == "__main__":
    main()
