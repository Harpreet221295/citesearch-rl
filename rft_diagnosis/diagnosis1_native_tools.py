"""Diagnosis 1, native-tool-calling variant — a direct A/B against the custom
bracket-format ReAct prompt (diagnosis1.py).

Question this answers: Qwen2.5-Instruct has a REAL, instruction-tuned-in tool-calling
format (Hermes-style `<tool_call>{"name":...,"arguments":...}</tool_call>`, verified
directly against the installed tokenizer's chat_template, not assumed from docs — see
`/tmp/qwen_full_template.txt` captured 2026-08-25). Our custom bracket format
(`Action: search[query]`) is something the model only ever sees 3 times in-context; the
native format is something it was actually TRAINED to use. This script tests: does
giving the model its OWN native tool-calling format (via `tools=` in
apply_chat_template, proper JSON-schema tool definitions, real `role="tool"` responses)
produce MORE RELIABLE multi-hop trajectories than the custom bracket format did —
tested WITHOUT hand-crafted worked examples, since native tool-calling is supposed to
need little/no in-context teaching if the model already knows the format.

Same probe pool, same temperature (0.9, GRPO's own rollout temp), same underlying
DocStore/tools.execute/citations.py/metrics.py scoring as diagnosis1.py, so the
comparison is apples-to-apples. Reuses trajectory.py's Trajectory/Step/ToolCall so
classify_trajectory-equivalent scoring is identical.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import config as dr_config
import citations
import metrics as dr_metrics
import tools as dr_tools
from trajectory import Trajectory, Step, ToolCall
from evaluate import build_vllm_engine, close_vllm_engine
from diagnosis1 import load_pool, aggregate_breakdown


TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "search",
        "description": "Search the document corpus; returns the top-k passage titles and snippets.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "natural-language search query"},
        }, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "read",
        "description": "Read the full text of one passage by its exact title.",
        "parameters": {"type": "object", "properties": {
            "title": {"type": "string", "description": "the exact passage title, as returned by search"},
        }, "required": ["title"]}}},
    {"type": "function", "function": {
        "name": "answer",
        # 2026-08-26: the terseness rule was previously ONLY in `_INSTRUCTIONS` (system
        # prompt prose). The 2026-08-25 A/B found native-format `correct_rate` = 0% NOT
        # from a knowledge gap but from answer VERBOSITY — the model embedded the right
        # fact in a full restated sentence ("...released in 2010, a song written...")
        # that strict EM can't credit. Hypothesis under test: in native-calling mode the
        # model attends to the tool schema's own description more reliably than to the
        # surrounding system-prompt prose. So state the rule HERE too, first, before the
        # citation instruction. See RFT_PLAN_AND_MODEL_DIAGNOSIS.md's A/B section.
        "description": ("Commit your final answer. Answer with the SHORTEST possible "
                        "phrase - usually 1-3 words: the exact entity, name, number, or "
                        "'yes'/'no'. No sentence, no explanation. "
                        "Cite every passage that supports it, "
                        "by its exact title in square brackets, e.g. "
                        "'American [Blue Harvest (film)] [Jane Doe (director)]'. "
                        "Do not cite a passage you did not read. This ends the episode."),
        "parameters": {"type": "object", "properties": {
            "text": {"type": "string", "description": "the shortest possible answer phrase (1-3 words), with [Title] citations appended"},
        }, "required": ["text"]}}},
]

_INSTRUCTIONS = (
    "You are a research agent. Answer the question by searching a document corpus, "
    "reading the most relevant passages, then giving a short, grounded final answer. "
    "Base your answer only on what you retrieve.\n"
    "Answer with the SHORTEST possible phrase — usually 1-3 words: the exact entity, "
    "name, number, or 'yes'/'no'. No sentence, no explanation.\n"
    "Ground your answer ONLY in passages you retrieved. Do not cite a passage you did "
    "not read."
)

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_STOP = ["<tool_response>", "<|im_start|>"]


def _parse_tool_call(text: str):
    """Extract the FIRST <tool_call>{...}</tool_call> block. Returns (name, args_dict,
    ok). Mirrors env._parse_react_action's contract for classify-compatibility."""
    m = _TOOL_CALL_RE.search(text or "")
    if not m:
        return "", {}, False
    try:
        obj = json.loads(m.group(1))
    except json.JSONDecodeError:
        return "", {}, False
    name = str(obj.get("name", "")).strip().lower()
    args = obj.get("arguments", {})
    if isinstance(args, str):          # some models emit arguments as a JSON string
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return name, {}, False
    if not isinstance(args, dict):
        return name, {}, False
    return name, args, True


def run_one_episode(task, docstore, cfg, llm, tokenizer, sampling_params) -> Trajectory:
    history = [{"role": "user", "content": _INSTRUCTIONS + f"\n\nQuestion: {task.question}"}]
    steps: list[Step] = []
    final_answer = None
    done = False
    for _round in range(cfg.max_turns + 1):
        prompt = tokenizer.apply_chat_template(
            history, tools=TOOL_SCHEMAS, add_generation_prompt=True, tokenize=False)
        out = llm.generate([prompt], sampling_params, use_tqdm=False)[0]
        raw = out.outputs[0].text
        name, args, ok = _parse_tool_call(raw)

        if not ok or name not in dr_tools.TOOLS:
            steps.append(Step(thought="", call=None,
                              observation="error: no valid tool_call found", ok=False,
                              error="parse", parse_ok=False))
            history.append({"role": "assistant", "content": raw})
            history.append({"role": "user", "content":
                            "error: no valid <tool_call> found. Call exactly one of "
                            "search, read, or answer."})
            if _round >= cfg.max_turns:
                break
            continue

        if name == "answer":
            final_answer = str(args.get("text", "")).strip()
            steps.append(Step(thought="", call=ToolCall(name, args, raw=raw),
                              observation=final_answer, ok=True, parse_ok=True))
            done = True
            break

        if name == "search" and "k" not in args:
            args = {**args, "k": cfg.search_k}
        obs, tok_ok, err, titles = dr_tools.execute(docstore, name, args)
        obs = obs[: cfg.max_obs_chars]
        steps.append(Step(thought="", call=ToolCall(name, args, raw=raw), observation=obs,
                          ok=tok_ok, error=err, parse_ok=True, retrieved_titles=titles))
        history.append({"role": "assistant", "tool_calls": [
            {"function": {"name": name, "arguments": args}}]})
        history.append({"role": "tool", "content": obs})
        if _round >= cfg.max_turns:
            break

    return Trajectory(
        task_id=task.task_id, query=task.question, tools=list(cfg.tools),
        gold_answer=task.gold_answer, gold_aliases=list(task.gold_aliases),
        supporting_titles=list(task.supporting_titles), steps=steps,
        final_answer=final_answer, done=done)


def classify(task, traj: Trajectory, cfg) -> dict:
    n_reads = sum(1 for c in traj.tool_calls if c.name == "read")
    n_searches = traj.n_searches
    parse_failures = sum(1 for s in traj.steps if not s.parse_ok)
    pred = citations.strip_citations(traj.final_answer or "")
    correct = dr_metrics.exact_match(pred, task.answers)
    creport = citations.verify_citations(
        traj, backend=cfg.citation_backend, align_threshold=cfg.citation_align_threshold,
        gold_titles=task.supporting_titles)
    if not correct:
        bucket = "wrong_answer"
    elif not creport.any_citation:
        bucket = "correct_uncited"
    elif creport.f1 < 0.999:
        bucket = "correct_miscited"
    else:
        bucket = "correct_and_cited"
    return {
        "task_id": task.task_id, "parse_ok": parse_failures == 0,
        "n_parse_failures": parse_failures, "calls_read": n_reads > 0,
        "n_reads": n_reads, "n_searches": n_searches,
        "n_tool_calls": len(traj.tool_calls), "terminated_cleanly": bool(traj.done),
        "correct": correct, "n_citations": creport.n_citations,
        "cite_f1": creport.f1, "cite_precision": creport.precision,
        "cite_recall": creport.recall, "bucket": bucket,
    }


def main():
    cfg = dr_config.Config.cloud_preset()
    tasks = load_pool(cfg, "train", n=16, offset=0)
    k = 4
    temperature = cfg.temperature   # 0.9, GRPO's own rollout temp — same comparison point as v2

    from dotenv import load_dotenv
    load_dotenv(_HERE / ".env")
    from transformers import AutoTokenizer
    from vllm import SamplingParams
    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    sp = SamplingParams(temperature=temperature, top_p=cfg.top_p,
                        max_tokens=cfg.max_new_tokens, stop=_STOP)

    print(f"Diagnosis 1 (native tool-calling): {len(tasks)} questions x k={k} "
          f"at temperature={temperature} on {cfg.model_name} (base, no adapter, "
          f"NO hand-crafted examples — testing the model's OWN trained format)")

    llm = build_vllm_engine(cfg, enable_lora=False)
    rows = []
    try:
        for task in tasks:
            docstore = task.docstore()
            for _ in range(k):
                traj = run_one_episode(task, docstore, cfg, llm, tok, sp)
                rows.append(classify(task, traj, cfg))
    finally:
        close_vllm_engine(llm)

    breakdown = aggregate_breakdown(rows)
    print("\n=== Diagnosis 1 — NATIVE tool-calling format ===")
    for k_, v in breakdown.items():
        print(f"  {k_}: {v}")

    # Write to a NEW path: diagnosis1_native_results.json is the 2026-08-25 baseline
    # this run's whole purpose is to be compared against — overwriting it would destroy
    # the comparison point. Override with DIAG1_NATIVE_OUT for further variants.
    out_name = os.environ.get("DIAG1_NATIVE_OUT", "diagnosis1_native_results_v2_terse.json")
    out_path = _HERE / "rft_diagnosis" / out_name
    out_path.write_text(json.dumps({"breakdown": breakdown, "raw_rows": rows}, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
