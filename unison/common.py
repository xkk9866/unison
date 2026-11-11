"""Shared paths, prompt templates and QA metrics."""
import json
import os
import re
import string
from collections import Counter

ROOT = os.environ.get("UNISON_ROOT", "/home/yaokang/evishare_fast")
DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")
MODELS = {
    "qwen3-8b": "/home/yaokang/models/Qwen3-8B-AWQ",
    "qwen3-4b": "/home/yaokang/models/Qwen3-4B-AWQ",
    "qwen2.5-7b": os.path.join(ROOT, "models/Qwen2.5-7B-Instruct-AWQ"),
    # second model family (optional generalization table): hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4
    "llama3.1-8b": os.path.join(ROOT, "models/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"),
}
EMBED_MODEL = os.path.join(ROOT, "models/bge-base-en-v1.5-hf")
AHN_BACKBONE = os.path.join(ROOT, "models/bge-small-en-v1.5-hf")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- prompts
VERIFY_SYSTEM = ("You are an evidence verification agent. You read one passage and check "
                 "whether it answers a question.")
VERIFY_HEAD = ("Read the passage and write a short verification note for the question.\n"
               "Format:\n"
               "Evidence: <quote the most relevant sentence from the passage, or \"none\">\n"
               "Reasoning: <one sentence on whether and how the evidence answers the question>\n"
               "Relevant: <yes or no>\n"
               "Answer: <a short answer span, or \"none\">\n\n")

# decision-first note: the same four fields, with the decision serialized before its provenance
VERIFY_HEAD_DF = ("Read the passage and write a short verification note for the question.\n"
                  "Format:\n"
                  "Relevant: <yes or no>\n"
                  "Answer: <a short answer span, or \"none\">\n"
                  "Evidence: <quote the most relevant sentence from the passage, or \"none\">\n"
                  "Reasoning: <one sentence on whether and how the evidence answers the question>\n\n")
# grounded decision-first note: an evidence *pointer* (sentence number) precedes the decision, so the
# certified prefix fixes both the decision and the sentence that supports it; the quoted sentence is
# materialized from the consumer's own passage version, and only the reasoning follows the boundary
VERIFY_HEAD_GDF = ("Read the numbered passage and write a short verification note for the question.\n"
                   "Format:\n"
                   "Support: <the number of the sentence that answers the question, or \"none\">\n"
                   "Relevant: <yes or no>\n"
                   "Answer: <a short answer span, or \"none\">\n"
                   "Reasoning: <one sentence on whether and how that sentence answers the question>\n\n")
# grounded reason-then-decide note: pointer, one sentence of reasoning, then the decision, which closes the
# note; the certified prefix is the whole note (reasoning precedes and conditions the decision)
VERIFY_HEAD_GCOT = ("Read the numbered passage and write a short verification note for the question.\n"
                    "Format:\n"
                    "Support: <the number of the sentence that answers the question, or \"none\">\n"
                    "Reasoning: <one sentence on whether and how that sentence answers the question>\n"
                    "Relevant: <yes or no>\n"
                    "Answer: <a short answer span, or \"none\">\n\n")
# decision-first with a certified evidence pointer: the decision is stated before the pointer (no
# grounding-first conservatism), and the boundary closes the pointer line, so decision and supporting
# sentence are certified together; the reasoning follows the boundary
VERIFY_HEAD_DFP = ("Read the numbered passage and write a short verification note for the question.\n"
                   "Format:\n"
                   "Relevant: <yes or no>\n"
                   "Answer: <a short answer span, or \"none\">\n"
                   "Support: <the number of the sentence that contains the answer, or \"none\">\n"
                   "Reasoning: <one sentence on whether and how that sentence answers the question>\n\n")
# decision-first with pointer and a *wide* payload: the audit record quotes the supporting sentence and gives a
# multi-sentence justification. Same key (Relevant, Answer, Support) as dfp; only the payload width differs.
VERIFY_HEAD_DFPW = ("Read the numbered passage and write a verification note for the question.\n"
                    "Format:\n"
                    "Relevant: <yes or no>\n"
                    "Answer: <a short answer span, or \"none\">\n"
                    "Support: <the number of the sentence that contains the answer, or \"none\">\n"
                    "Evidence: <quote the supporting sentence verbatim, or \"none\">\n"
                    "Reasoning: <two or three sentences on how the evidence answers the question, "
                    "what the question asks for, and any caveat>\n\n")
NOTE_FORMATS = {"cot": VERIFY_HEAD, "dfirst": VERIFY_HEAD_DF, "gdf": VERIFY_HEAD_GDF, "gcot": VERIFY_HEAD_GCOT,
                "dfp": VERIFY_HEAD_DFP, "dfpw": VERIFY_HEAD_DFPW}
BOUNDARY_FIELD = {"dfp": "support:", "dfpw": "support:"}  # the last line of the certified prefix (default: answer)
NUMBERED_FORMATS = ("gdf", "gcot", "dfp", "dfpw")
PAYLOAD_TOKENS = {"dfpw": 160}  # payload budget per note format (default 64: one Reasoning line)
PAYLOAD_FIRST_FIELD = {"dfpw": "Evidence:"}  # the field that opens the payload (default: Reasoning:)


# decision-keyed provenance: the reasoning that justifies an evidence claim (answer a is stated by sentence k
# of this passage version) is a function of the claim and the evidence only, not of how the question was
# worded. Its prompt is therefore identical for every call that reached the same claim on the same version,
# and identical-prompt coalescing shares it exactly.
PROV_SYSTEM = "You are an evidence verification agent. You justify an evidence claim about one passage."
PROV_HEAD = ("Read the numbered passage. Claim: sentence [{k}] states the answer \"{a}\".\n"
             "Write one line:\n"
             "Reasoning: <one sentence on how sentence [{k}] states this answer>\n\n")


def provenance_messages(title, text, answer, k):
    user = PROV_HEAD.format(k=k, a=answer) + f"Passage (title: {title}):\n{number_sentences(text)}"
    return [{"role": "system", "content": PROV_SYSTEM}, {"role": "user", "content": user}]


def provenance_ids(engine, title, text, answer, k, fmt="dfp"):
    """Claim-keyed provenance program that shares the evidence prefix of the key program.

    The prompt is the verification prompt of the same note format up to and including the passage
    (system, instructions, numbered passage: identical tokens, hence a prefix-cache hit on the evidence KV
    that the key computations already materialized), followed by the claim instead of the question. The
    assistant turn is pre-filled with the claim's decision fields, so the program decodes exactly the
    Reasoning field of a note whose decision is the claim. It contains neither the question nor any peer
    output: equal claims on equal passage versions give byte-identical programs."""
    body = number_sentences(text) if fmt in NUMBERED_FORMATS else text
    where = f"sentence [{k}]" if k > 0 else "the passage"
    user = (NOTE_FORMATS[fmt] + f"Passage (title: {title}):\n{body}\n\n"
            + f"Claim: {where} states the answer \"{answer}\". Write the note for this claim.")
    msgs = [{"role": "system", "content": VERIFY_SYSTEM}, {"role": "user", "content": user}]
    sup = str(k) if k > 0 else "none"
    head = f"Relevant: yes\nAnswer: {answer}\nSupport: {sup}\n" + PAYLOAD_FIRST_FIELD.get(fmt, "Reasoning:")
    return list(engine.chat_ids(msgs)) + list(engine.tok.encode(head, add_special_tokens=False))


def payload_tokens(fmt):
    return PAYLOAD_TOKENS.get(fmt, 64)


def split_sentences(text):
    return [x for x in re.split(r"(?<=[.!?])\s+", text.strip()) if x]


def number_sentences(text):
    return " ".join(f"[{i}] {x}" for i, x in enumerate(split_sentences(text), 1))


def parse_support(text):
    m = re.search(r"(?i)support\s*:\s*\[?(\d+)", text)
    return int(m.group(1)) if m else 0

PLAN_SYSTEM = "You are a research agent that solves a multi-hop question step by step."
STRATEGIES = [
    "Break the question into simple single-hop sub-questions and ask them one at a time.",
    "First identify the entity you must find, then ask a precise factual question about it.",
    "Ask the most direct factual question that brings you one hop closer to the answer.",
    "Work backwards from what the question asks and query the missing fact first.",
    "Ask short, specific questions that a single encyclopedia paragraph can answer.",
]


def verify_messages(title, text, question, fmt="cot"):
    # condition-invariant content (instructions + evidence) precedes the condition
    if fmt in NUMBERED_FORMATS:
        text = number_sentences(text)
    user = NOTE_FORMATS[fmt] + f"Passage (title: {title}):\n{text}\n\nQuestion: {question}"
    return [{"role": "system", "content": VERIFY_SYSTEM}, {"role": "user", "content": user}]


def plan_messages(strategy, question, steps, force_final=False):
    lines = [STRATEGIES[strategy % len(STRATEGIES)], f"Question: {question}"]
    for i, (sq, ans) in enumerate(steps, 1):
        lines.append(f"Sub-question {i}: {sq}")
        lines.append(f"Finding {i}: {ans}")
    if force_final:
        lines.append('Give your best final answer now. Reply with one line: "Final answer: <short answer>".')
    else:
        n = len(steps) + 1
        lines.append('If the findings are sufficient, reply with "Final answer: <short answer>". '
                     f'Otherwise reply with "Sub-question {n}: <the next single-hop question>". '
                     "Reply with one line only.")
    return [{"role": "system", "content": PLAN_SYSTEM}, {"role": "user", "content": "\n".join(lines)}]


def parse_plan(text):
    t = text.strip().split("\n")[0].strip()
    m = re.match(r"(?i)final answer\s*:\s*(.*)", t)
    if m:
        return "final", m.group(1).strip().strip('."')
    m = re.match(r"(?i)sub-question\s*\d*\s*:\s*(.*)", t)
    if m and m.group(1).strip():
        return "ask", m.group(1).strip()
    if t.endswith("?"):
        return "ask", t
    return "final", t.strip('."')


def parse_note(text):
    rel = re.search(r"(?i)relevant\s*:\s*(yes|no)", text)
    ans = re.search(r"(?i)answer\s*:\s*(.*)", text)
    relevant = bool(rel and rel.group(1).lower() == "yes")
    a = ans.group(1).strip().strip('."') if ans else "none"
    if a.lower() in ("none", "n/a", "unknown", ""):
        relevant, a = False, "none"
    return relevant, a


# ---------------------------------------------------------------- metrics
def normalize_answer(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1_score(pred, gold):
    p, g = normalize_answer(pred).split(), normalize_answer(gold).split()
    common = Counter(p) & Counter(g)
    ns = sum(common.values())
    if ns == 0:
        return 0.0
    pr, rc = ns / len(p), ns / len(g)
    return 2 * pr * rc / (pr + rc)


def em_f1(pred, golds):
    em = max(float(normalize_answer(pred) == normalize_answer(g)) for g in golds)
    f1 = max(f1_score(pred, g) for g in golds)
    return em, f1


def decision_equiv(d1, d2):
    """Decision equivalence: same relevance and the same answer span after SQuAD normalization."""
    (r1, a1), (r2, a2) = d1, d2
    return bool(r1) == bool(r2) and normalize_answer(a1) == normalize_answer(a2)
