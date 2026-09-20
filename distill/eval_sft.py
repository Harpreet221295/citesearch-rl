"""The SFT gate: base vs tuned on held-out questions. Generation, not loss.

    python distill/eval_sft.py --adapter distill/runs/sft/final --split sft_dev --n 128
    python distill/eval_sft.py --adapter distill/runs/sft/final --split heldout_eval --n 500

WHAT THIS IS FOR. A falling training loss says the model memorised the teacher's tokens.
It says nothing about whether the agent now searches, reads before citing, and stops
cleanly. Those only show up in real multi-turn rollouts, which is why early stopping and
the final gate both run through here rather than through a validation loss.

THREE THINGS IT GETS RIGHT ON PURPOSE:

1. SAME PROMPT AS TRAINING. `include_worked_example=False`, so the model is evaluated
   under exactly the prompt it was fine-tuned under and will be rolled out under during
   RL. Evaluating with the worked example still in the prompt would flatter the model and
   make the result unreadable — the whole point of SFT here is to move that demonstration
   into the weights (SFT_RL_PLAN.md §2).

2. ONE ENGINE, LoRA HOT-SWAPPED. Base and tuned run in the SAME resident vLLM engine
   (`lora_request=None` vs a LoRARequest), so the comparison cannot drift on engine
   settings, seed, or sampling params — and it does not pay to load the model twice.
   Batched throughout (RUNPOD_PLAYBOOK pattern #3); never batch-1 in a loop.

3. THE SPLIT METRIC. `cite_f1` alone is conjunctive (cited AND gold AND read) and cannot
   say which half moved. `title_f1` (did it pick the right sources?) and
   `read_before_cite_rate` (did it verify them?) are reported separately, because
   read-before-cite is the specific behaviour this SFT stage exists to install and we
   need to see it move independently of answer correctness.

ANTI-HACKING PROBES (CLAUDE.md: "reward went up" is not a result). A model can improve
`cite_f1` by pasting more citations, or improve nothing while inflating length. Both are
reported next to the headline numbers so a hollow win is visible rather than inferred.

THREE ARMS, not two — because two would answer the wrong question.

    base_noex    base model, no worked example    <- what SFT started from
    base_ex      base model, WITH worked example  <- what prompting alone can do
    tuned        SFT model, no worked example     <- what we built

base_noex vs tuned measures what fine-tuning added, and both use the same prompt so it is
apples-to-apples. But on its own it is a soft target: the base model is being denied a
demonstration it used to receive, so the delta flatters SFT.

base_ex vs tuned is the question a skeptic actually asks: **was fine-tuning worth it, or
would showing the model a worked example have done the same job for free?** If tuned does
not beat base_ex, SFT is still needed (RL needs the short prompt and a policy that behaves
without the crutch) — but it is NOT a capability win and must not be reported as one.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

_HERE = Path(__file__).resolve().parent.parent
for p in (str(_HERE), str(_HERE / "rft_diagnosis"), str(_HERE / "distill")):
    if p not in sys.path:
        sys.path.insert(0, p)

import citations
import config as dr_config
import splits as dr_splits
from diagnosis1 import classify_trajectory, aggregate_breakdown, _TURN_STOP_SEQUENCES
from evaluate import build_vllm_engine, close_vllm_engine, run_batched_rollouts


def score_all(results, cfg) -> list[dict]:
    rows = []
    for task, traj, _ri in results:
        r = classify_trajectory(task, traj, cfg)
        r["final_answer"] = traj.final_answer
        r["gold_answer"] = task.gold_answer
        r["answer_words"] = len((citations.strip_citations(traj.final_answer or "")).split())
        r["n_steps"] = len(traj.steps)
        rows.append(r)
    return rows


def summarize(rows: list[dict]) -> dict:
    b = aggregate_breakdown(rows)
    n = len(rows)
    b["zero_tool_call_rate"] = sum(1 for r in rows
                                   if r["n_searches"] + r["n_reads"] == 0) / n
    b["mean_answer_words"] = sum(r["answer_words"] for r in rows) / n
    b["mean_n_citations"] = sum(r.get("n_citations", 0) for r in rows) / n
    b["mean_n_steps"] = sum(r["n_steps"] for r in rows) / n
    return b


HEADLINE = [
    ("correct_rate",            "correct (exact match)"),
    ("title_f1",                "  picked the right sources"),
    ("read_before_cite",        "  verified them (read first)"),
    ("cite_f1",                 "  both -> the reward's number"),
    ("calls_read_rate",         "called read at least once"),
    ("terminated_cleanly_rate", "finished cleanly"),
    ("zero_tool_call_rate",     "never used a tool (lower better)"),
]
KEYMAP = {"title_f1": "mean_title_f1", "read_before_cite": "mean_read_before_cite",
          "cite_f1": "mean_cite_f1"}
PROBES = [
    ("mean_answer_words",  "answer length (words)"),
    ("mean_n_citations",   "citations per answer"),
    ("mean_n_steps",       "turns per episode"),
    ("mean_n_reads",       "reads per episode"),
]


def table(base: dict, base_ex: dict, tuned: dict) -> None:
    print(f"\n{'':38}{'base':>9}{'base+ex':>9}{'tuned':>9}{'vs base':>9}{'vs +ex':>9}")
    print(f"  {'':36}{'no ex':>9}{'prompted':>9}{'no ex':>9}")
    print("-" * 84)
    for key, label in HEADLINE:
        k = KEYMAP.get(key, key)
        b, e, t = base.get(k, 0.0), base_ex.get(k, 0.0), tuned.get(k, 0.0)
        print(f"  {label:36}{b:>9.3f}{e:>9.3f}{t:>9.3f}{t-b:>+9.3f}{t-e:>+9.3f}")
    print(f"\n  ANTI-HACKING PROBES (a hollow win shows up here, not above)")
    print("-" * 84)
    for key, label in PROBES:
        b, e, t = base.get(key, 0.0), base_ex.get(key, 0.0), tuned.get(key, 0.0)
        print(f"  {label:36}{b:>9.2f}{e:>9.2f}{t:>9.2f}{t-b:>+9.2f}{t-e:>+9.2f}")
    print(f"\n  buckets base   : {base.get('bucket_pct')}")
    print(f"  buckets base+ex: {base_ex.get('bucket_pct')}")
    print(f"  buckets tuned  : {tuned.get('bucket_pct')}")
    print(f"  hops base+ex   : {base_ex.get('hop_count_distribution')}")
    print(f"  hops tuned     : {tuned.get('hop_count_distribution')}")


def gate(base: dict, base_ex: dict, tuned: dict) -> bool:
    """The SFT gate from SFT_RL_PLAN.md §4, written down BEFORE the numbers existed.

    The point of this SFT stage is PROCESS, so the gate is on process: reading and
    verifying must improve substantially, correctness must not regress, and the model
    must still finish cleanly. Correctness improving is welcome, not required.
    """
    checks = [
        ("read_before_cite improves >= 0.10",
         tuned["mean_read_before_cite"] - base["mean_read_before_cite"] >= 0.10),
        ("calls_read_rate improves >= 0.10",
         tuned["calls_read_rate"] - base["calls_read_rate"] >= 0.10),
        ("correct_rate does not regress (>= base - 0.02)",
         tuned["correct_rate"] >= base["correct_rate"] - 0.02),
        ("terminated_cleanly >= base",
         tuned["terminated_cleanly_rate"] >= base["terminated_cleanly_rate"] - 0.02),
        ("no citation-count inflation (< 2x base, unless title_f1 also rose)",
         tuned["mean_n_citations"] <= 2 * max(0.5, base["mean_n_citations"])
         or tuned["mean_title_f1"] > base["mean_title_f1"]),
    ]
    print(f"\n  SFT GATE (fixed before the run — see SFT_RL_PLAN.md §4)")
    print("-" * 68)
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    passed = all(ok for _, ok in checks)
    print(f"\n  => {'GATE PASSED' if passed else 'GATE FAILED'}")

    # Reported, deliberately NOT a gate condition: SFT is required for RL regardless
    # (short prompt, no crutch). But whether it beat prompting is the difference between
    # a capability win and an expensive re-implementation of a prompt.
    beat_r = tuned["mean_read_before_cite"] > base_ex["mean_read_before_cite"]
    beat_c = tuned["correct_rate"] > base_ex["correct_rate"]
    print(f"\n  DID FINE-TUNING BEAT JUST PROMPTING? (the skeptic's question)")
    print("-" * 84)
    print(f"    read_before_cite : tuned {tuned['mean_read_before_cite']:.3f} vs "
          f"prompted {base_ex['mean_read_before_cite']:.3f}   {'YES' if beat_r else 'NO'}")
    print(f"    correct_rate     : tuned {tuned['correct_rate']:.3f} vs "
          f"prompted {base_ex['correct_rate']:.3f}   {'YES' if beat_c else 'NO'}")
    if not (beat_r or beat_c):
        print("    -> A worked example in the prompt does the same job. SFT is still")
        print("       needed for RL, but this is NOT a capability win.")
    return passed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, help="path to the LoRA adapter directory")
    ap.add_argument("--split", default="sft_dev",
                    choices=["sft_dev", "heldout_eval", "sft_collect", "rl_train", "musique_dev"])
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="0 = greedy. The gate should be deterministic; sample only to "
                         "study variance")
    ap.add_argument("--lora-rank", type=int, default=32,
                    help="must be >= the adapter's r, or vLLM refuses to load it")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from dotenv import load_dotenv
    load_dotenv(_HERE / ".env")
    from transformers import AutoTokenizer
    from vllm import SamplingParams
    from vllm.lora.request import LoRARequest

    # The prompt the model was TRAINED under, and the one RL will roll out under.
    cfg = replace(dr_config.Config.cloud_preset(),
                  include_worked_example=False, lora_r=args.lora_rank)
    tasks = dr_splits.get_split(args.split, cfg, limit=args.n)
    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    sp = SamplingParams(temperature=args.temperature,
                        top_p=1.0 if args.temperature == 0 else cfg.top_p,
                        max_tokens=cfg.max_new_tokens, stop=_TURN_STOP_SEQUENCES)

    print(f"SFT gate — base vs tuned, one engine, LoRA hot-swapped")
    print(f"  split       : {args.split} ({len(tasks)} questions)")
    print(f"  adapter     : {args.adapter}")
    print(f"  prompt      : include_worked_example=False (SAME as training and RL)")
    print(f"  sampling    : temperature={args.temperature} "
          f"({'greedy' if args.temperature == 0 else 'sampled'}), "
          f"max_new_tokens={cfg.max_new_tokens}")

    llm = build_vllm_engine(cfg, enable_lora=True)
    try:
        print("\n  [1/3] BASE, no worked example ...", flush=True)
        base_rows = score_all(run_batched_rollouts(tasks, cfg, llm, tok, sp), cfg)

        # Same weights, same engine — only the prompt changes.
        print("  [2/3] BASE, WITH worked example (what prompting alone does) ...",
              flush=True)
        cfg_ex = replace(cfg, include_worked_example=True)
        tasks_ex = dr_splits.get_split(args.split, cfg_ex, limit=args.n)
        base_ex_rows = score_all(
            run_batched_rollouts(tasks_ex, cfg_ex, llm, tok, sp), cfg_ex)

        print("  [3/3] TUNED, no worked example ...", flush=True)
        lora = LoRARequest("sft", 1, args.adapter)
        tuned_rows = score_all(
            run_batched_rollouts(tasks, cfg, llm, tok, sp, lora_request=lora), cfg)
    finally:
        close_vllm_engine(llm)

    base = summarize(base_rows)
    base_ex = summarize(base_ex_rows)
    tuned = summarize(tuned_rows)
    table(base, base_ex, tuned)
    passed = gate(base, base_ex, tuned)

    out = Path(args.out or (_HERE / "distill" / f"eval_{args.split}.json"))
    out.write_text(json.dumps({
        "split": args.split, "n": len(tasks), "adapter": args.adapter,
        "temperature": args.temperature, "gate_passed": passed,
        "base_noex": base, "base_with_example": base_ex, "tuned": tuned,
        "base_rows": base_rows, "base_ex_rows": base_ex_rows,
        "tuned_rows": tuned_rows,
    }, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
