"""Collect summaries from results/<tag>/<ds>/summary.json into tables and figures (runs locally)."""
import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RES = sys.argv[1] if len(sys.argv) > 1 else "results"
FIG = sys.argv[2] if len(sys.argv) > 2 else "figs"
os.makedirs(FIG, exist_ok=True)

NAMES = {"noshare": "NoShare", "exact": "ExactShare", "pld": "PLD", "suffix": "SuffixDecoding",
         "exact+suffix": "Exact+Suffix", "unison": "Unison (ε=0)", "semcache-0.90": "SemCache-0.90",
         "semcache-0.95": "SemCache-0.95", "semcache-0.85": "SemCache-0.85", "semcache-0.98": "SemCache-0.98",
         "semcacheV-0.90": "SemCache-V-0.90", "semcrc-0.05": "SemCRC-0.05", "semcrc-0.02": "SemCRC-0.02",
         "semcrc-0.1": "SemCRC-0.10", "unison-e0.05": "Unison (ε=0.05)", "unison-e0.02": "Unison (ε=0.02)",
         "unison-e0.01": "Unison (ε=0.01)", "unison-e0.1": "Unison (ε=0.10)"}


def load(tag, ds):
    p = os.path.join(RES, tag, ds, "summary.json")
    if not os.path.exists(p):
        return {}
    return {s["method"]: s for s in json.load(open(p))}


def main_table(tag="main"):
    rows = []
    for ds in ["hotpotqa", "musique"]:
        S = load(tag, ds)
        if not S:
            continue
        base = S.get("noshare")
        for m, s in S.items():
            rows.append((ds, NAMES.get(m, m), s["throughput"], s["lat_p50"], s["lat_p99"], s["vlat_p50"],
                         s["vlat_p99"], s["engine_busy"], base["engine_busy"] / s["engine_busy"] if base else 1,
                         s["reuse_calls"] / max(1, s["n_verify"]), s.get("fidelity_verify", float("nan")),
                         s.get("fidelity_decision", float("nan")), s["em"], s["f1"]))
    hdr = "ds method thr p50 p99 vp50 vp99 busy speedup reuse tokfid decfid em f1".split()
    print("\t".join(hdr))
    for r in rows:
        print("\t".join(x if isinstance(x, str) else f"{x:.3f}" for x in r))
    return rows


if __name__ == "__main__":
    main_table(sys.argv[3] if len(sys.argv) > 3 else "main")
