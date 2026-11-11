"""Build self-supervised AHN pairs from independent-execution logs (train pools only).

Every pair (w, u) reads the same passage. Labels: tau = LCP(y_w, y_u) (accepted draft length under
greedy verification) and agree = [decision(y_w) == decision(y_u)]. To cover version shifts, a sample
of logged calls is re-executed on the other versions of its passage (same condition).
"""
import gzip
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vllm import LLM, SamplingParams

from unison.agents import World
from unison.ahn import version_delta
from unison.common import DATA, MODELS, RESULTS, normalize_answer, parse_note, verify_messages

AUG_FRAC, MAX_PAIRS_PER_PID = 0.35, 60


def lcp(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def decision(text):
    rel, ans = parse_note(text)
    return rel, normalize_answer(ans)


def main():
    llm = LLM(model=MODELS["qwen3-8b"], gpu_memory_utilization=0.85, max_model_len=4096, enable_prefix_caching=True)
    tok = llm.get_tokenizer()
    rng = random.Random(0)
    out_rows = []
    for ds in ["hotpotqa", "musique"]:
        world = World(os.path.join(DATA, ds, "train"))
        with gzip.open(os.path.join(RESULTS, "ahndata", ds, "noshare.calls.jsonl.gz"), "rt") as f:
            recs = [json.loads(l) for l in f]
        items = [dict(pid=r["pid"], version=r["version"], cond=r["cond"], out=r["out"]) for r in recs
                 if r["kind"] == "verify"]
        # dedup identical (pid, version, cond): those are certified-equivalent, not uncertain pairs
        uniq = {}
        for it in items:
            uniq.setdefault((it["pid"], it["version"], it["cond"]), it)
        items = list(uniq.values())
        aug_prompts, aug_meta = [], []
        for it in items:
            vt = world.vtexts.get(it["pid"])
            if vt is None or rng.random() > AUG_FRAC:
                continue
            for v in rng.sample([v for v in range(len(vt)) if v != it["version"]], k=min(2, len(vt) - 1)):
                if (it["pid"], v, it["cond"]) in uniq:
                    continue
                c = world.corpus[it["pid"]]
                msgs = verify_messages(c["title"], world.text(it["pid"], v), it["cond"])
                aug_prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                           enable_thinking=False))
                aug_meta.append(dict(pid=it["pid"], version=v, cond=it["cond"]))
        outs = llm.generate(aug_prompts, SamplingParams(temperature=0.0, max_tokens=128))
        for m, o in zip(aug_meta, outs):
            m["out"] = list(o.outputs[0].token_ids)
            items.append(m)
        by_pid = {}
        for it in items:
            it["text"] = tok.decode(it["out"], skip_special_tokens=True)
            it["dec"] = decision(it["text"])
            by_pid.setdefault(it["pid"], []).append(it)
        n_pairs, deltas = 0, {}
        for pid, grp in by_pid.items():
            if len(grp) < 2:
                continue
            cand = [(i, j) for i in range(len(grp)) for j in range(len(grp)) if i != j]
            rng.shuffle(cand)
            for i, j in cand[:MAX_PAIRS_PER_PID]:
                w, u = grp[i], grp[j]
                key = (pid, u["version"], w["version"])
                if key not in deltas:
                    deltas[key] = version_delta(world.text(pid, u["version"]), world.text(pid, w["version"]))
                out_rows.append(dict(ds=ds, pid=pid, cond_c=w["cond"], cond_p=u["cond"], v_c=w["version"],
                                     v_p=u["version"], delta=deltas[key], prod=u["out"],
                                     tau=lcp(w["out"], u["out"]), n_prod=len(u["out"]),
                                     agree=int(w["dec"] == u["dec"])))
                n_pairs += 1
        print(ds, "items", len(items), "augmented", len(aug_meta), "pairs", n_pairs, flush=True)
    os.makedirs(os.path.join(RESULTS, "ahn"), exist_ok=True)
    with gzip.open(os.path.join(RESULTS, "ahn", "pairs.jsonl.gz"), "wt") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
