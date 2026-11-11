"""Build question pools and shared open-domain corpora (IRCoT-style) for HotpotQA and MuSiQue."""
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pandas as pd

from unison.common import DATA, read_jsonl, write_jsonl

HF = os.path.join(DATA, "hf")
N_EVAL, N_TRAIN = 800, 500


def hotpot_rows(split):
    d = os.path.join(HF, "hotpot_qa/distractor")
    files = sorted(f for f in os.listdir(d) if f.startswith(split))
    df = pd.concat([pd.read_parquet(os.path.join(d, f)) for f in files])
    rows = []
    for r in df.itertuples():
        paras = [(t, "".join(s).strip()) for t, s in zip(r.context["title"], r.context["sentences"])]
        rows.append(dict(qid=r.id, question=r.question, answers=[r.answer], qtype=r.type,
                         gold_titles=sorted(set(r.supporting_facts["title"])), paras=paras))
    return rows


def musique_rows(split):
    p = os.path.join(HF, f"MuSiQue/musique_ans_v1.0_{split}.jsonl")
    if not os.path.exists(p):
        return None
    rows = []
    for r in read_jsonl(p):
        if not r.get("answerable", True):
            continue
        paras = [(x["title"], x["paragraph_text"].strip()) for x in r["paragraphs"]]
        gold = sorted({x["title"] for x in r["paragraphs"] if x["is_supporting"]})
        rows.append(dict(qid=r["id"], question=r["question"],
                         answers=[r["answer"]] + list(r.get("answer_aliases", [])),
                         qtype=r["id"].split("__")[0], gold_titles=gold, paras=paras,
                         decomposition=[d["question"] for d in r["question_decomposition"]]))
    return rows


def build_pool(rows, n, seed, out_dir):
    rng = random.Random(seed)
    rows = rng.sample(rows, min(n, len(rows)))
    pid_of, corpus = {}, []
    for r in rows:
        pids = []
        for title, text in r.pop("paras"):
            key = (title, text)
            if key not in pid_of:
                pid_of[key] = len(corpus)
                corpus.append(dict(pid=len(corpus), title=title, text=text))
            pids.append(pid_of[key])
        r["context_pids"] = pids
    write_jsonl(os.path.join(out_dir, "questions.jsonl"), rows)
    write_jsonl(os.path.join(out_dir, "corpus.jsonl"), corpus)
    print(out_dir, len(rows), "questions", len(corpus), "passages")


def main():
    hp_dev, hp_train = hotpot_rows("validation"), hotpot_rows("train")
    build_pool(hp_dev, N_EVAL, 0, os.path.join(DATA, "hotpotqa/eval"))
    build_pool(hp_train, N_TRAIN, 1, os.path.join(DATA, "hotpotqa/train"))
    mq_dev, mq_train = musique_rows("dev"), musique_rows("train")
    if mq_train is None:  # fall back to a disjoint slice of dev
        rng = random.Random(7)
        rng.shuffle(mq_dev)
        mq_train, mq_dev = mq_dev[:N_TRAIN], mq_dev[N_TRAIN:]
        print("MuSiQue train split missing: using a disjoint dev slice for AHN training")
    build_pool(mq_dev, N_EVAL, 0, os.path.join(DATA, "musique/eval"))
    build_pool(mq_train, N_TRAIN, 1, os.path.join(DATA, "musique/train"))


if __name__ == "__main__":
    main()
