import os, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")
from vllm import LLM, SamplingParams
m = os.environ.get("MODEL", "/home/yaokang/models/Qwen3-8B-AWQ")
spec = os.environ.get("SPEC", "1") == "1"
kw = dict(model=m, gpu_memory_utilization=0.80, max_model_len=4096, enable_prefix_caching=True, max_num_seqs=128)
if spec:
    kw["speculative_config"] = {"method": "ngram", "num_speculative_tokens": 16, "prompt_lookup_max": 4, "prompt_lookup_min": 2}
llm = LLM(**kw)
tok = llm.get_tokenizer()
msgs = [{"role": "user", "content": "Passage: The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named after the engineer Gustave Eiffel, whose company designed and built the tower from 1887 to 1889.\n\nQuestion: Who designed the Eiffel Tower? Quote the evidence then answer."}]
p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
sp = SamplingParams(temperature=0, max_tokens=96, extra_args={"x": 1})
t = time.time(); outs = llm.generate([p] * 64, sp); dt = time.time() - t
print("TIME", dt, "tokens", sum(len(o.outputs[0].token_ids) for o in outs))
print(repr(outs[0].outputs[0].text))
print(len(set(o.outputs[0].text for o in outs)), "distinct outputs")
