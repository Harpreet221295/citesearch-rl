"""Diagnosis 1, native tool-calling — RE-RUN on the fixed, batched harness.

Supersedes `diagnosis1_native_tools.py`. That script's numbers measured six harness
bugs, not the model (`verify_format.py` found them; `native_rollout.py`'s docstring
lists each one and its fix). Its headline result — native format `correct_rate` 0.0 —
is an artifact: in a 4-question probe the model emitted the EXACT gold answer on 2 of
them and the harness discarded both.

Same comparison point as the runs already in RFT_PLAN_AND_MODEL_DIAGNOSIS.md's A/B
table, so before/after is apples-to-apples:
    16 questions (train pool, offset 0), k=4, temperature=0.9 (GRPO's own rollout temp),
    Qwen2.5-3B-Instruct base, no adapter.
Scoring is `diagnosis1.classify_trajectory` UNCHANGED — the same code training uses, so
"correct" and "well-cited" keep one meaning across every stage.

    python rft_diagnosis/diagnosis1_native_v2.py                 # the standard 16 x 4
    python rft_diagnosis/diagnosis1_native_v2.py --n 4 --k 2      # quick smoke
    python rft_diagnosis/diagnosis1_native_v2.py --answer-tool    # A/B: advertise `answer`

Beyond the old breakdown this reports two things the old one could not, both of which
exist because the old harness could not tell these cases apart:
  * finish_mode  — plain_text / answer_tool / round_limit / dead. How the episode ENDED.
  * n_retries    — replies that were genuinely unreadable, now that "the model answered
                   in prose" is no longer miscounted as a parse failure.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import citations
import config as dr_config
import metrics as dr_metrics
from evaluate import build_vllm_engine, close_vllm_engine
from diagnosis1 import load_pool, classify_trajectory, aggregate_breakdown
from native_rollout import (
    run_batched_native_rollouts, build_sampling_params, tool_schemas,
    system_prompt, STOP_STRINGS,
)


def _make_transcriber(tdir: Path, n_questions: int):
    """One file per QUESTION, written to be read top-to-bottom like a conversation.

    Requested by Harpreet 2026-08-26. The first version of this dumped the full prompt
    on every round — but the prompt GROWS each round (it is the whole conversation so
    far), so a 9-round episode repeated everything nine times and came to 40KB of
    near-duplicate text. Unreadable, and it buried the one line that mattered.

    So: the opening prompt (system message + tool schemas + question) is printed ONCE,
    verbatim, because it is identical every round. After that only what is NEW appears —
    what the model wrote, and what we sent back. That is the actual conversation.

    Only sample #0 of each question is captured, so one question means one file.
    """
    buffers: dict[str, list[str]] = {}
    tracked: dict[str, object] = {}

    def bar(ch: str, label: str = "", width: int = 92) -> str:
        if not label:
            return ch * width
        label = f" {label} "
        pad = max(0, width - len(label))
        return ch * (pad // 2) + label + ch * (pad - pad // 2)

    def indent(text: str, prefix: str = "    ") -> str:
        return "\n".join(prefix + ln for ln in (text or "").splitlines()) or prefix + "(empty)"

    def on_round(rnd, active, prompts, raws, outs):
        for ep, prompt, raw, out in zip(active, prompts, raws, outs):
            if ep.sample_idx != 0:                 # one file per question, not per sample
                continue
            key = ep.task.task_id
            if key not in buffers:
                if len(buffers) >= n_questions:
                    continue
                tracked[key] = ep
                T = buffers[key] = []
                T.append(bar("="))
                T.append(f"QUESTION : {ep.task.question}")
                T.append(f"GOLD     : {ep.task.gold_answer!r}")
                T.append(f"EVIDENCE : {ep.task.supporting_titles}")
                T.append(f"task_id  : {ep.task.task_id}")
                T.append(bar("="))
                T.append("")
                T.append("PART 1 - WHAT WE SEND THE MODEL AT THE START.")
                T.append("This is identical on every round (later rounds just have the")
                T.append("conversation below appended to it), so it is shown once, verbatim.")
                T.append("")
                T.append(bar("-"))
                T.append(prompt)
                T.append(bar("-"))
                T.append("")
                T.append("")
                T.append("PART 2 - THE CONVERSATION. Only what is new each round.")
                T.append("")
            T = buffers[key]
            o = out.outputs[0]
            T.append(bar("-", f"ROUND {rnd}"))
            T.append("")
            T.append("  MODEL WROTE:")
            T.append(indent(raw, "      "))
            T.append("")
            T.append(f"  [ended: {o.finish_reason}"
                     + (f" at {o.stop_reason!r}" if o.stop_reason else " (end-of-turn token)")
                     + f", {len(o.token_ids)} tokens]")
            # What the harness did with it is appended by _note_action below, which runs
            # after _advance so it can report the real outcome rather than predict it.
            ep._transcript = T                     # noqa: SLF001 - deliberate side channel

    def note_action(ep, text: str) -> None:
        T = getattr(ep, "_transcript", None)
        if T is None:
            return
        T.append("")
        T.append("  WE DID:")
        T.append(indent(text, "      "))
        T.append("")

    def flush():
        tdir.mkdir(exist_ok=True)
        written = []
        for key, T in buffers.items():
            ep = tracked[key]
            # Score the transcript with the SAME code the metrics use — strip citations
            # first, then exact_match against gold + aliases. A naive raw-string compare
            # here printed "exact match: no" for 'RCD Mallorca [RCD Mallorca]' while the
            # breakdown correctly counted it right; a transcript that contradicts the
            # metrics is worse than no transcript.
            pred = citations.strip_citations(ep.final_answer or "")
            correct = dr_metrics.exact_match(pred, ep.task.answers)
            T.append(bar("="))
            T.append("HOW IT ENDED")
            T.append(f"  ended because : {ep.finish_mode}")
            T.append(f"  final answer  : {ep.final_answer!r}")
            T.append(f"  ...citations stripped -> scored as: {pred!r}")
            T.append(f"  gold answer   : {ep.task.gold_answer!r}  (aliases: {list(ep.task.gold_aliases)})")
            T.append(f"  CORRECT       : {'YES' if correct else 'no'}   "
                     "(same check the metrics use: strip_citations + exact_match)")
            T.append(f"  tools called  : {[c.name for c in ep.to_trajectory(dr_config.Config()).tool_calls]}")
            T.append(f"  unreadable replies: {ep.n_retries}")
            T.append(bar("="))
            path = tdir / f"v2_{ep.task.task_id}.txt"
            path.write_text("\n".join(T))
            written.append(path)
        return written

    return on_round, note_action, flush


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16, help="questions from the probe pool")
    ap.add_argument("--k", type=int, default=4, help="samples per question")
    ap.add_argument("--temperature", type=float, default=None, help="default: cfg.temperature (0.9)")
    ap.add_argument("--answer-tool", action="store_true",
                    help="advertise the `answer` tool (A/B against the native convention)")
    ap.add_argument("--examples", action="store_true",
                    help="prepend the 3 worked trajectories from diagnosis1._RICH_EXAMPLES, "
                         "translated to native conversation turns. THE MISSING 2x2 CELL: the "
                         "bracket arm always had these, the native arm never did")
    ap.add_argument("--think", action="store_true",
                    help="ask for one sentence of reasoning before each call (fix 8 A/B). "
                         "MEASURED 2026-08-26: without this the model writes NO reasoning "
                         "at all - 0 of 8 tool-call replies had any text before the call")
    ap.add_argument("--transcripts", type=int, default=3, help="episodes to dump verbatim")
    ap.add_argument("--out", default=None, help="results filename under rft_diagnosis/")
    args = ap.parse_args()

    from dotenv import load_dotenv
    load_dotenv(_HERE / ".env")
    from transformers import AutoTokenizer

    cfg = dr_config.Config.cloud_preset()
    temperature = cfg.temperature if args.temperature is None else args.temperature
    tasks = load_pool(cfg, "train", n=args.n, offset=0)
    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    sp = build_sampling_params(cfg, temperature)

    print(f"Diagnosis 1 v2 (native tool-calling, FIXED batched harness)")
    print(f"  model        : {cfg.model_name} (base, no adapter)")
    print(f"  probe        : {len(tasks)} questions x k={args.k} = {len(tasks)*args.k} episodes")
    print(f"  temperature  : {temperature}   max_new_tokens: {cfg.max_new_tokens}")
    answer_tool_desc = ("ADVERTISED" if args.answer_tool else
                        "not advertised (plain text = final answer, the native convention)")
    print(f"  answer tool  : {answer_tool_desc}")
    print(f"  think first  : {args.think}")
    print(f"  worked examples: {args.examples} (3 native-format trajectories)")
    print(f"  stop strings : {STOP_STRINGS}")
    print(f"  tools        : {[t['function']['name'] for t in tool_schemas(args.answer_tool)]}")
    print(f"  batching     : lockstep — ONE generate() per round over all active episodes")
    print()

    tdir = _HERE / "rft_diagnosis" / "transcripts"
    on_round, note_action, flush = _make_transcriber(tdir, args.transcripts)

    llm = build_vllm_engine(cfg, enable_lora=False)
    try:
        episodes = run_batched_native_rollouts(
            tasks, cfg, llm, tok, sp, k=args.k, think=args.think,
            examples=args.examples,
            include_answer_tool=args.answer_tool,
            on_round=on_round, on_action=note_action)
    finally:
        close_vllm_engine(llm)

    rows = []
    for ep in episodes:
        row = classify_trajectory(ep.task, ep.to_trajectory(cfg), cfg)
        row["finish_mode"] = ep.finish_mode
        row["n_retries"] = ep.n_retries
        row["n_thoughts"] = ep.n_thoughts
        row["final_answer"] = ep.final_answer     # keep the text: the 2026-08-25 session
        row["gold_answer"] = ep.task.gold_answer  # had to re-run to get this back
        rows.append(row)

    breakdown = aggregate_breakdown(rows)
    breakdown["finish_mode_pct"] = {k: v / len(rows) for k, v in
                                    Counter(r["finish_mode"] for r in rows).items()}
    breakdown["mean_n_retries"] = sum(r["n_retries"] for r in rows) / len(rows)
    breakdown["zero_tool_call_rate"] = sum(
        1 for r in rows if r["n_searches"] == 0 and r["n_reads"] == 0) / len(rows)
    # fix 8: does the model reason at all? Without --think the measured answer is "no".
    breakdown["mean_n_thoughts"] = sum(r["n_thoughts"] for r in rows) / len(rows)
    breakdown["any_thought_rate"] = sum(1 for r in rows if r["n_thoughts"]) / len(rows)

    print("\n=== Diagnosis 1 v2 — NATIVE format, fixed harness ===")
    for k_, v in breakdown.items():
        print(f"  {k_}: {v}")

    print("\n--- baseline for comparison (diagnosis1_native_results.json, BROKEN harness) ---")
    old_path = _HERE / "rft_diagnosis" / "diagnosis1_native_results.json"
    if old_path.exists():
        old = json.loads(old_path.read_text())["breakdown"]
        for key in ("correct_rate", "mean_cite_f1", "calls_read_rate",
                    "terminated_cleanly_rate", "parse_ok_rate"):
            print(f"  {key}: {old.get(key)}  ->  {breakdown.get(key)}")

    out_name = args.out or ("diagnosis1_native_v2"
                            + ("_answertool" if args.answer_tool else "")
                            + ("_examples" if args.examples else "")
                            + ("_think" if args.think else "") + ".json")
    out_path = _HERE / "rft_diagnosis" / out_name
    out_path.write_text(json.dumps({
        "breakdown": breakdown, "raw_rows": rows,
        "setup": {"model": cfg.model_name, "n_questions": len(tasks), "k": args.k,
                  "temperature": temperature, "max_new_tokens": cfg.max_new_tokens,
                  "answer_tool_advertised": args.answer_tool,
                  "think": args.think, "examples": args.examples,
                  "stop_strings": STOP_STRINGS,
                  "system_prompt": system_prompt(args.think)},
    }, indent=2))
    print(f"\nwrote {out_path}")
    for p in flush():
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
