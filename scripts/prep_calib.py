"""Calibration pools: 500 further training-split questions per dataset, disjoint from the AHN train pools."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from prep_data import N_TRAIN, build_pool, hotpot_rows, musique_rows
from unison.common import DATA, read_jsonl

for ds, rows in [("hotpotqa", hotpot_rows("train")), ("musique", musique_rows("train"))]:
    used = {q["qid"] for q in read_jsonl(os.path.join(DATA, ds, "train", "questions.jsonl"))}
    rows = [r for r in rows if r["qid"] not in used]
    build_pool(rows, N_TRAIN, 2, os.path.join(DATA, ds, "calib"))
