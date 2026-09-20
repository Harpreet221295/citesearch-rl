"""Audit the BRACKET-format harness the same way the native one was audited.

WHY (2026-08-26): the native harness turned out to have six bugs that made its
`correct_rate=0.0` meaningless. The bracket arm has never had that scrutiny, and it
carries the same warning signature that started the whole investigation:

    parse_ok_rate            0.000   <- EVERY trajectory has >=1 parse failure
    zero-tool-call           36/64   <- 56% never emit a single valid Action
    correct when it DID use a tool   12/28 = 42.9%

That 42.9% is currently the best number this project has, and the entire "which format
should RFT collect with" decision would rest on it. It should not be trusted until it
has been looked at directly rather than through an aggregate — that is exactly the
mistake that let the native bugs survive four separate readings of the same data.

Two questions this answers, both by measurement, neither by assumption:

Q1. DOES THE MODEL REASON IN THE BRACKET FORMAT? (Harpreet, 2026-08-26)
    In the native format it does not: 0 of 8 tool-call replies had any text before the
    call, and the answers ran 2-5 tokens. The bracket format is ReAct and has an
    explicit `Thought:` channel, and `env._parse_react_action` DOES capture it into
    `Step.thought` (env.py:238) — but `classify_trajectory` never reported it, so the
    stored results carry no thought data at all and the question has never actually
    been answered for this arm. Here it is measured directly.

Q2. WHAT ARE THE 36 ZERO-TOOL-CALL TRAJECTORIES ACTUALLY DOING?
    "Never emitted a valid Action" is a symptom with several very different causes, and
    the aggregate cannot tell them apart:
      - the model answered straight away out of parametric memory (a real behaviour)
      - it wrote a valid-looking action our regex rejects (a harness bug, like native's)
      - it produced prose that was discarded (native's exact bug, in a new place)
      - generation was cut by a stop sequence before the Action line (a boundary bug)
    Same failure-cause taxonomy the native audit used, so the two are comparable.

    python rft_diagnosis/audit_bracket.py                # 16 x 4, temp 0.9, transcripts
    python rft_diagnosis/audit_bracket.py --n 4 --k 2    # smoke

Uses the SAME rollout path the real bracket results came from
(`evaluate.run_batched_rollouts` under `patched_rich_prompt`), so this audits the thing
that produced the number rather than a reimplementation of it. Batched, per
RUNPOD_PLAYBOOK pattern #3.
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
import tools as dr_tools
from evaluate import build_vllm_engine, close_vllm_engine
from diagnosis1 import (
    load_pool, classify_trajectory, aggregate_breakdown,
    patched_rich_prompt, _TURN_STOP_SEQUENCES,
)
from env import DeepResearchEnv


def rollout_keeping_envs(tasks, cfg, llm, tokenizer, sampling_params, k):
    """Byte-for-byte evaluate.run_batched_rollouts, except the env objects are RETURNED
    instead of discarded — batched lockstep, same stop sequences, same everything.

    Needed because of a real observability gap in production `env.py`: on a parse
    failure `_run_tool` receives the model's raw text as `raw` but stores
    `Step(call=None, ...)`, dropping it (env.py:122). So a rejected reply leaves NO
    record of what the model actually said — which is precisely why 277 rejected steps
    could sit in the results unnoticed. `env._history` does keep every assistant turn,
    so holding the envs recovers the text without touching production code or
    perturbing the metrics (adding a raw field to those Steps would change
    `traj.tool_calls` counting and silently move every existing number).
    """
    expanded = [t for t in tasks for _ in range(k)]
    envs = [DeepResearchEnv.from_dict({"task": t, "cfg": cfg, "judge": None})
            for t in expanded]
    for env in envs:
        env.reset()
    done = [False] * len(envs)
    for _round in range(int(getattr(cfg, "max_turns", 8)) + 1):
        active = [i for i, d in enumerate(done) if not d]
        if not active:
            break
        prompts = [tokenizer.apply_chat_template(envs[i]._history,
                                                 add_generation_prompt=True, tokenize=False)
                   for i in active]
        outs = llm.generate(prompts, sampling_params, use_tqdm=False)
        for i, out in zip(active, outs):
            _obs, _r, d, _info = envs[i].step(out.outputs[0].text)
            done[i] = d
    return list(zip(expanded, envs))


def assistant_turns(env) -> list[str]:
    """Every raw assistant reply, in order — the text the Steps threw away."""
    return [m["content"] for m in env._history if m.get("role") == "assistant"]


def _step_cause(step, raw: str = "") -> str:
    """Why did this step fail to parse? The bracket analogue of the native audit's
    cause buckets, so the two arms can be compared rather than just eyeballed.

    `raw` must come from env._history (see rollout_keeping_envs) — a failed Step does
    not carry it. Falling back to step.observation, as an earlier version of this did,
    classifies the ERROR MESSAGE we sent rather than what the model wrote, and every
    bucket comes out wrong.
    """
    if not raw and step.call is not None and getattr(step.call, "raw", None):
        raw = step.call.raw
    if step.parse_ok:
        return "parsed_ok"
    if "Action:" in raw:
        return "action_line_present_but_rejected"   # a harness-side rejection
    if "Thought:" in raw:
        return "thought_only_no_action"             # reasoned, never acted
    if raw.strip():
        return "prose_no_react_syntax"              # native's bug, in a new place
    return "model_returned_empty_string"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--transcripts", type=int, default=3)
    ap.add_argument("--out", default="audit_bracket_results.json")
    args = ap.parse_args()

    from dotenv import load_dotenv
    load_dotenv(_HERE / ".env")
    from transformers import AutoTokenizer
    from vllm import SamplingParams

    cfg = dr_config.Config.cloud_preset()
    T = cfg.temperature if args.temperature is None else args.temperature
    tasks = load_pool(cfg, "train", n=args.n, offset=0)
    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    sp = SamplingParams(temperature=T, top_p=cfg.top_p, max_tokens=cfg.max_new_tokens,
                        stop=_TURN_STOP_SEQUENCES)

    print("Bracket-format AUDIT (same rollout path as diagnosis1_results_v2.json)")
    print(f"  model       : {cfg.model_name} (base, no adapter)")
    print(f"  probe       : {len(tasks)} questions x k={args.k} = {len(tasks)*args.k} episodes")
    print(f"  temperature : {T}   max_new_tokens: {cfg.max_new_tokens}")
    print(f"  stop        : {_TURN_STOP_SEQUENCES}")
    print(f"  prompt      : rich_opening_prompt (3 worked examples), via patched_rich_prompt")
    print()

    llm = build_vllm_engine(cfg, enable_lora=False)
    try:
        with patched_rich_prompt():
            pairs = rollout_keeping_envs(tasks, cfg, llm, tok, sp, args.k)
    finally:
        close_vllm_engine(llm)

    rows, causes, per_ep = [], Counter(), []
    for task, env in pairs:
        traj = env._to_dr_trajectory()
        raws = assistant_turns(env)
        row = classify_trajectory(task, traj, cfg)
        thoughts = [s.thought for s in traj.steps if (s.thought or "").strip()]
        row["n_thoughts"] = len(thoughts)
        row["mean_thought_chars"] = (sum(len(t) for t in thoughts) / len(thoughts)) if thoughts else 0
        row["final_answer"] = traj.final_answer
        row["gold_answer"] = task.gold_answer
        # Persist every raw assistant reply. The whole reason this audit exists is that
        # rejected replies left no record; writing only the aggregate again would
        # reproduce the exact blind spot being fixed.
        row["raw_replies"] = raws
        row["step_causes"] = [_step_cause(s_, raws[i] if i < len(raws) else "")
                              for i, s_ in enumerate(traj.steps)]
        for i, s in enumerate(traj.steps):
            causes[_step_cause(s, raws[i] if i < len(raws) else "")] += 1
        rows.append(row)
        per_ep.append((task, traj, row, raws))

    b = aggregate_breakdown(rows)
    n = len(rows)
    b["mean_n_thoughts"] = sum(r["n_thoughts"] for r in rows) / n
    b["any_thought_rate"] = sum(1 for r in rows if r["n_thoughts"]) / n
    b["mean_thought_chars"] = (sum(r["mean_thought_chars"] for r in rows if r["n_thoughts"])
                               / max(1, sum(1 for r in rows if r["n_thoughts"])))
    b["zero_tool_call_rate"] = sum(1 for r in rows if r["n_tool_calls"] == 0) / n
    b["step_cause_counts"] = dict(causes)

    print("\n=== Bracket audit — full breakdown ===")
    for k_, v in b.items():
        if k_ != "raw_rows":
            print(f"  {k_}: {v}")

    # Q1 — reasoning, the direct comparison against the native arm's 0.031 / 0.547.
    print("\n--- Q1: does the model reason in the bracket format? ---")
    print(f"  episodes writing >=1 Thought : {b['any_thought_rate']:.1%}")
    print(f"  mean Thoughts per episode    : {b['mean_n_thoughts']:.2f}")
    print(f"  mean Thought length (chars)  : {b['mean_thought_chars']:.0f}")
    print("  native arm, for comparison   : 0.031 plain / 0.547 with --think / 0.750 with examples")

    # Q2 — what the zero-tool-call trajectories really are.
    zero = [x for x in per_ep if x[2]["n_tool_calls"] == 0]
    print(f"\n--- Q2: the zero-tool-call trajectories ({len(zero)}/{n}) ---")
    print(f"  step-level causes across ALL episodes: {dict(causes)}")
    zt = sum(1 for x in zero if x[2]["n_thoughts"])
    print(f"  of the zero-tool-call ones, how many still WROTE a Thought: {zt}/{len(zero)}")
    print("  (a high number means the model reasoned and then failed to act — a very")
    print("   different diagnosis from 'it never engaged at all')")
    print("\n  first 5 zero-tool-call episodes, verbatim final_answer:")
    for t, tr, r, raws in zero[:3]:
        print(f"    [{t.task_id}] gold={t.gold_answer!r} final={tr.final_answer!r} "
              f"n_steps={len(tr.steps)} thoughts={r['n_thoughts']}")
        for j, rw in enumerate(raws[:3]):
            print(f"        round {j} RAW: {rw[:300]!r}")

    # Transcripts: prefer a spread — some zero-tool-call, some correct.
    tdir = _HERE / "rft_diagnosis" / "transcripts"
    tdir.mkdir(exist_ok=True)
    picked, seen = [], set()
    for pool in (zero, [x for x in per_ep if x[2]["correct"]], per_ep):
        for t, tr, r, raws in pool:
            if len(picked) >= args.transcripts:
                break
            if t.task_id in seen:
                continue
            seen.add(t.task_id)
            picked.append((t, tr, r, raws))

    written = []
    for t, tr, r, raws in picked:
        L = ["=" * 92,
             f"QUESTION : {t.question}",
             f"GOLD     : {t.gold_answer!r}",
             f"EVIDENCE : {t.supporting_titles}",
             f"task_id  : {t.task_id}",
             "=" * 92, "",
             "BRACKET (ReAct) FORMAT. Each round the model is supposed to write:",
             "    Thought: <reasoning>",
             "    Action: <tool>[<argument>]",
             "Below is what it ACTUALLY wrote each round, verbatim, and what we did.", ""]
        for i, s in enumerate(tr.steps):
            raw = raws[i] if i < len(raws) else ""
            L.append("-" * 40 + f" ROUND {i} " + "-" * 40)
            L.append("")
            L.append("  MODEL WROTE:")
            L.append("\n".join("      " + ln for ln in raw.splitlines()) or "      (nothing captured)")
            L.append("")
            L.append(f"  thought captured : {s.thought!r}")
            L.append(f"  parsed OK        : {s.parse_ok}")
            L.append(f"  action           : {s.call.name if s.call else None}"
                     f"{'(' + str(s.call.args) + ')' if s.call else ''}")
            L.append(f"  cause bucket     : {_step_cause(s, raw)}")
            L.append("")
            L.append("  WE SENT BACK:")
            L.append("\n".join("      " + ln for ln in (s.observation or "").splitlines()[:14]))
            L.append("")
        pred = citations.strip_citations(tr.final_answer or "")
        L += ["=" * 92, "HOW IT ENDED",
              f"  final answer  : {tr.final_answer!r}",
              f"  ...citations stripped -> scored as: {pred!r}",
              f"  gold answer   : {t.gold_answer!r}  (aliases: {list(t.gold_aliases)})",
              f"  CORRECT       : {'YES' if dr_metrics.exact_match(pred, t.answers) else 'no'}",
              f"  tools called  : {[c.name for c in tr.tool_calls]}",
              f"  thoughts      : {r['n_thoughts']}",
              "=" * 92]
        path = tdir / f"bracket_{t.task_id}.txt"
        path.write_text("\n".join(L))
        written.append(path)

    out = _HERE / "rft_diagnosis" / args.out
    b_out = {k: v for k, v in b.items() if k != "raw_rows"}
    out.write_text(json.dumps({"breakdown": b_out, "raw_rows": rows,
                               "setup": {"model": cfg.model_name, "n_questions": len(tasks),
                                         "k": args.k, "temperature": T,
                                         "stop": _TURN_STOP_SEQUENCES,
                                         "prompt": "rich_opening_prompt (3 worked examples)"}},
                              indent=2))
    print(f"\nwrote {out}")
    for p in written:
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
