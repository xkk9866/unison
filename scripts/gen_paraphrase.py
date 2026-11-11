"""Generate user paraphrases of each question (CacheBlend-style 'extended' workloads)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vllm import LLM, SamplingParams

from unison.common import DATA, MODELS, read_jsonl, write_jsonl

N_PARA = 3
llm = LLM(model=MODELS["qwen3-8b"], gpu_memory_utilization=0.85, max_model_len=2048, enable_prefix_caching=True)
tok = llm.get_tokenizer()
for ds in ["hotpotqa", "musique"]:
    for split in os.environ.get("SPLITS", "eval,train").split(","):
        d = os.path.join(DATA, ds, split)
        qs = read_jsonl(os.path.join(d, "questions.jsonl"))
        prompts = []
        for q in qs:
            for i in range(N_PARA):
                msg = [{"role": "user", "content":
                        f"Rewrite the following question the way a different user might ask it (variant {i + 1}). "
                        f"Keep exactly the same meaning and all names. Output only the rewritten question.\n\n"
                        f"Question: {q['question']}"}]
                prompts.append(tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True,
                                                       enable_thinking=False))
        outs = llm.generate(prompts, SamplingParams(temperature=0.8, top_p=0.95, max_tokens=80, seed=0))
        rows = []
        for j, q in enumerate(qs):
            paras = []
            for i in range(N_PARA):
                t = outs[j * N_PARA + i].outputs[0].text.strip().split("\n")[0].strip().strip('"')
                if t and t.lower() != q["question"].lower() and t not in paras:
                    paras.append(t)
            rows.append(dict(qid=q["qid"], paraphrases=paras))
        write_jsonl(os.path.join(d, "paraphrases.jsonl"), rows)
        print(ds, split, sum(len(r["paraphrases"]) for r in rows), "paraphrases")
