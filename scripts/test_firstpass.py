"""First-pass verification on a live engine, in one process: exactness, passes and bookkeeping.

usage: CUDA_VISIBLE_DEVICES=<g> python test_firstpass.py --model <dir> [--drafter <checkpoint>] [--n 32]

A small model is served by vLLM's V1 engine core inside this process with the Unison plugin. The same prompts are
decoded greedily without drafts (the reference) and with drafts handed to the engine before a call's first pass,
so that every case of the bookkeeping is exercised against the reference:
  oracle   the reference output itself: every call has to finish in one pass
  half     the first half of the reference followed by tokens the model does not produce: the accepted part is
           kept, the rest is decoded token by token
  wrong    tokens the model does not produce: everything is rejected
  chain    half, then the reference again in every later pass (drafts before and after the first pass)
  drafter  the consensus drafter's own code path with an untrained drafter of the model's width (captured
           programs, record, peers, accounting); what it drafts is verified like any draft
Calls arrive in groups over several engine steps, share a long prompt prefix (prefix-cache hits at admission) and
every second one ends at a decision boundary. Every delivered output has to equal the reference token by token.
Prints FIRSTPASS_OK or FIRSTPASS_BAD.
"""
import argparse
import json
import os
import sys
import tempfile

os.environ.setdefault("VLLM_USE_V1", "1")
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"  # the engine core runs in this process

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "unison_plugin"))

CITIES = ["Paris", "Tokyo", "Cairo", "Lima", "Oslo", "Delhi", "Rome", "Quito", "Hanoi", "Dakar", "Perth", "Kyoto",
          "Lagos", "Seoul", "Miami", "Sofia", "Tunis", "Minsk", "Dhaka", "Accra", "Berne", "Porto", "Sucre", "Osaka",
          "Milan", "Kabul", "Amman", "Doha", "Riga", "Baku", "Turin", "Graz"]
RULES = ("You are a geography clerk who files one record per city. Records are read by a program, so the format is "
         "strict. For the city the user names, write exactly four lines and nothing else. The first line starts "
         "with 'Country: ' followed by the country the city is in. The second line starts with 'Continent: ' "
         "followed by the continent. The third line starts with 'Language: ' followed by the main language spoken "
         "there. The fourth line starts with 'Note: ' followed by one short sentence about the city. Do not add "
         "greetings, explanations, markup or blank lines. Use the common English names of countries, continents "
         "and languages. If a city name is ambiguous, file the largest city of that name. Records that break the "
         "format are discarded by the program, so follow it exactly. " * 2)
MAXTOK = 48
JUNK = 11  # a token the model does not produce in these records


def boundary(toks, nlset, n_nl):
    """Number of tokens through the n_nl-th newline (None if the output has fewer lines)."""
    cnt = 0
    for j, t in enumerate(toks):
        cnt += t in nlset
        if cnt >= n_nl:
            return j + 1
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--drafter", default="")
    p.add_argument("--n", type=int, default=32)
    p.add_argument("--kv_blocks", type=int, default=700)
    p.add_argument("--need_mib", type=int, default=6500)
    a = p.parse_args()
    logdir = tempfile.mkdtemp(prefix="firstpass_")
    os.environ["UNISON_CCR_LOGDIR"] = logdir

    # an untrained drafter of this model's width (the drafter variant runs the real code path with it)
    from unison.drafter import ConsensusDrafter
    hidden = json.load(open(os.path.join(a.model, "config.json")))["hidden_size"]
    if a.drafter:
        ck = torch.load(a.drafter, map_location="cpu")
        cfg, vocab_ids = dict(ck["cfg"]), list(ck["vocab_ids"])
    else:
        cfg, vocab_ids = {}, list(range(1000, 1400))
    cfg["h_dim"] = hidden
    torch.manual_seed(0)
    net = ConsensusDrafter(vocab_ids, **cfg)
    dpath = os.path.join(logdir, "cd_untrained.pt")
    torch.save(dict(cfg=cfg, vocab_ids=vocab_ids, state=net.state_dict()), dpath)
    del net
    os.environ["UNISON_DRAFTER"] = dpath

    import unison_vllm_patch as up
    up.register()
    from vllm.v1.core.sched import scheduler as sm
    PASS = {}
    plugin_schedule = sm.Scheduler.schedule

    def schedule(self):
        out = plugin_schedule(self)
        for rid in out.num_scheduled_tokens:
            PASS[rid] = PASS.get(rid, 0) + 1
        return out

    sm.Scheduler.schedule = schedule
    errors = []

    def spy(*args, **kw):  # the plugin reports a failed draft with a print and carries on: here it is a failure
        if args and isinstance(args[0], str) and args[0].startswith("[unison]") and "error" in args[0]:
            errors.append(" ".join(str(x) for x in args))
        print(*args, **kw)

    up.print = spy

    free, total = torch.cuda.mem_get_info()
    util = round(min(0.97, (total - free + a.need_mib * 2 ** 20) / total), 2)
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.sampling_params import RequestOutputKind
    llm = LLM(model=a.model, gpu_memory_utilization=util, max_model_len=4096, enable_prefix_caching=True,
              max_num_seqs=64, max_num_batched_tokens=8192, seed=0, num_gpu_blocks_override=a.kv_blocks,
              speculative_config={"method": "ngram", "num_speculative_tokens": 128, "prompt_lookup_max": 4,
                                  "prompt_lookup_min": 2})
    eng = llm.llm_engine
    tok = llm.get_tokenizer()
    eos = sorted({tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>"),
                  tok.convert_tokens_to_ids("<|endoftext|>")} - {None})
    nl = [i for i in range(len(tok)) if "\n" in tok.decode([i])]
    nlset = set(nl)

    def prompt(i):
        city = CITIES[i % len(CITIES)] + ("" if i < len(CITIES) else f" (file {i})")
        msgs = [{"role": "system", "content": RULES}, {"role": "user", "content": f"City: {city}"}]
        return tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, enable_thinking=False)

    prompts = [prompt(i) for i in range(a.n)]
    print(f"prompts: {a.n}, {len(prompts[0])} tokens each (shared prefix), gpu_memory_utilization {util}", flush=True)

    def run(name, eas, arrive=4, reset=False):
        """Requests arrive `arrive` per engine step. Returns the outputs and the passes of every request."""
        if reset:
            eng.reset_prefix_cache()
        outs, todo, steps = {}, list(range(a.n)), 0
        while todo or len(outs) < a.n:
            for i in todo[:arrive]:
                sp = SamplingParams(temperature=0.0, max_tokens=MAXTOK, output_kind=RequestOutputKind.FINAL_ONLY,
                                    extra_args=dict(eas[i], run=name, eos=eos), skip_special_tokens=True)
                eng.add_request(f"{name}-{i}", TokensPrompt(prompt_token_ids=prompts[i]), sp)
            todo = todo[arrive:]
            for o in eng.step():
                if o.finished:
                    outs[o.request_id] = list(o.outputs[0].token_ids)
            steps += 1
            if steps > 20000:
                raise RuntimeError("the engine does not finish")
        for _ in range(2):  # the runner writes a call's accounting in the step after its end
            eng.step()
        acc = {}
        path = os.path.join(logdir, f"{name}.acc.jsonl")
        if os.path.exists(path):
            acc = {r["rid"]: r for r in map(json.loads, open(path))}
        return ([outs[f"{name}-{i}"] for i in range(a.n)], [PASS.get(f"{name}-{i}", 0) for i in range(a.n)],
                [acc.get(f"{name}-{i}", {}) for i in range(a.n)], steps)

    def strip(t):
        return t[:-1] if t and t[-1] in eos else t

    # ---- reference: no drafts, no boundary
    ref, ref_pass, _, steps = run("ref", [{"mode": "none"}] * a.n)
    print(f"reference: {steps} engine steps, {sum(map(len, ref)) / a.n:.1f} tokens per call, "
          f"{sum(ref_pass) / a.n:.1f} passes per call", flush=True)
    print("example:", repr(tok.decode(ref[0])), flush=True)
    ref2, _, _, _ = run("ref2", [{"mode": "none"}] * a.n, arrive=7, reset=True)
    noise = sum(x != y for x, y in zip(ref, ref2))
    print(f"reference repeated with other batches: {noise} of {a.n} outputs differ (numerical floor)", flush=True)

    N_NL = 3
    bnd = [boundary(r, nlset, N_NL) if i % 2 else None for i, r in enumerate(ref)]  # every second call: a key

    def base(i):
        ea = dict(mode="nd", pre=1, k=64, astar=0.0, group=f"g{i % 4}")
        if bnd[i] is not None:
            ea.update(nl=nl, n_nl=N_NL)
        return ea

    def key(i):  # what the call has to deliver: its reference, up to its boundary if it has one
        return ref[i][:bnd[i]] if bnd[i] is not None else ref[i]

    def provider(net_, H, ents, B, k_engine):
        """Test drafts in place of the drafter: t_pre before the first pass, the rest of t_post after any pass."""
        out = {}
        for req_id, ea, store, y, row, tail in ents:
            if row < 0:
                out[req_id] = list(ea.get("t_pre") or [])
            elif ea.get("t_post"):
                own = store.live[req_id].tolist()
                post = ea["t_post"]
                if post[:len(own)] == own:
                    out[req_id] = post[len(own):len(own) + 16]
        return out

    class Dummy:
        ctx_len = 8

    real = (up._nd_drafts, up._drafter)
    bad = []

    def check(name, eas, expect_one_pass=False, reset=False, arrive=4):
        outs, passes, acc, steps = run(name, eas, arrive=arrive, reset=reset)
        wrong = 0
        for i, o in enumerate(outs):
            o, r = strip(o), strip(ref[i])
            if bnd[i] is not None:  # ends at its boundary (or later, if the stop was seen a step late)
                ok = o == r[:len(o)] and len(o) >= bnd[i]
            else:
                ok = o == r
            if not ok:
                wrong += 1
                if wrong <= 3:
                    print(f"  MISMATCH {name}-{i}: got {tok.decode(o)!r}\n    expected {tok.decode(r)!r}", flush=True)
        pre = sum(x.get("pre", 0) for x in acc)
        nd = sum(x.get("nd", 0) for x in acc)
        drafted = sum(x.get("drafted", 0) for x in acc)
        acc_pass = [x.get("passes", -1) for x in acc]
        one = sum(x == 1 for x in passes)
        print(f"{name}: {steps} engine steps, passes per call {sum(passes) / a.n:.2f} (reference {sum(ref_pass) / a.n:.2f}), "
              f"one-pass calls {one}/{a.n}, drafted {drafted}, accepted before the first pass {pre}, after {nd}, "
              f"outputs differing from the reference {wrong}", flush=True)
        if wrong > noise:
            bad.append(f"{name}: {wrong} outputs differ from the reference")
        if acc_pass != passes:
            bad.append(f"{name}: the plugin's pass count differs from the scheduler's ({acc_pass[:6]} vs {passes[:6]})")
        if expect_one_pass and one < a.n - noise:
            bad.append(f"{name}: {a.n - one} calls needed more than one pass")
        if up._ND.get("mem"):
            bad.append(f"{name}: {len(up._ND['mem'])} record memories left behind")
        return outs, passes, acc

    up._ND.setdefault("mem", {})
    up._nd_drafts, up._drafter = provider, (lambda runner=None: Dummy)
    check("nodraft", [base(i) for i in range(a.n)])
    check("oracle", [dict(base(i), t_pre=key(i)) for i in range(a.n)], expect_one_pass=True)
    check("oracle-cold", [dict(base(i), t_pre=key(i)) for i in range(a.n)], expect_one_pass=True, reset=True, arrive=9)
    half = [key(i)[:len(key(i)) // 2] + [JUNK] * 4 for i in range(a.n)]
    check("half", [dict(base(i), t_pre=half[i]) for i in range(a.n)], reset=True)
    check("wrong", [dict(base(i), t_pre=[JUNK] * 8) for i in range(a.n)])
    check("chain", [dict(base(i), t_pre=half[i], t_post=key(i)) for i in range(a.n)], reset=True, arrive=3)
    check("post", [dict(base(i), pre=0, t_post=key(i)) for i in range(a.n)])
    up._nd_drafts, up._drafter = real
    up._ND.pop("mem", None)
    # short drafts: an untrained drafter at price 0 drafts to the cap, and every position costs a row of logits
    _, _, acc = check("drafter", [dict(base(i), k=8) for i in range(a.n)], reset=True)
    if not sum(x.get("drafted", 0) for x in acc):
        bad.append("drafter: the untrained drafter drafted nothing (its code path was not exercised)")
    check("drafter-warm", [dict(base(i), k=8) for i in range(a.n)], arrive=16)
    net = up._ND.get("net")
    if net is not None:
        print(f"drafter programs captured: rows {sorted(net._programs)}; drafting time "
              f"{1e3 * up._ND['t'][0] / max(1, up._ND['t'][1]):.3f} ms per decoded position", flush=True)
    bad += errors
    print("FIRSTPASS_OK" if not bad else "FIRSTPASS_BAD " + " | ".join(bad), flush=True)
    return int(bool(bad))


if __name__ == "__main__":
    sys.exit(main())
