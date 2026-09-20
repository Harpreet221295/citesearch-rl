"""Collected trajectories -> a tokenized, masked SFT dataset. No GPU, no API calls.

    python distill/build_sft.py                    # tier A only (the plan's default)
    python distill/build_sft.py --tiers AB         # add process-good/answer-wrong
    python distill/build_sft.py --cite-bar 0.5     # loosen tier A's citation bar

Three things happen here, and two of them are silent if wrong:

1. TIERING (SFT_RL_PLAN.md §3)
     A  correct AND cite_f1 >= bar        -> grade every assistant turn
     B  read >= half the gold passages,
        answer wrong                      -> grade the process turns, SKIP the answer
     C  never found the gold evidence     -> dropped
   Episodes where the API gave up mid-run are dropped outright: an empty turn caused by
   a rate limit is not a demonstration of anything.

2. PROMPT RE-RENDER — the silent one.
   The teacher was collected under a prompt containing a full worked example, plus
   teacher-only instructions ("be terse", "read before citing", "use Action: answer[...]").
   The student must be trained under the prompt it will actually see at RL and eval time:
   `include_worked_example=False`, no teacher suffix. So turn 0 is REBUILT here rather
   than taken from the stored history.
   Nothing crashes if this is skipped — the prompt is masked, so the loss is identical —
   but the model would be conditioned on instructions it never receives in deployment,
   and the entire SFT stage would be quietly devalued. See SFT_RL_PLAN.md §2.

3. MASKING, asserted per example against the real tokenizer, not assumed.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

_HERE = Path(__file__).resolve().parent.parent
for p in (str(_HERE), str(_HERE / "rft_diagnosis"), str(_HERE / "distill")):
    if p not in sys.path:
        sys.path.insert(0, p)

import config as dr_config
import env as env_mod
from sft_data import encode_trajectory, assert_masking_correct, summarize

JSONL = _HERE / "distill" / "teacher_trajectories.jsonl"
OUT = _HERE / "distill" / "sft_dataset.pt"


def _norm(t: str) -> str:
    return " ".join((t or "").lower().split())


def read_titles(history: list[dict]) -> list[str]:
    out = []
    for m in history or []:
        if m.get("role") != "assistant":
            continue
        for line in (m.get("content") or "").splitlines():
            line = line.strip()
            if line.startswith("Action: read[") and line.endswith("]"):
                out.append(line[len("Action: read["):-1])
    return out


def is_process_clean(rec: dict) -> bool:
    """2026-09-07 — the PROCESS filter ("tier P"), Harpreet's framing: distillation
    literature (sequence-level KD, on-policy distillation) imitates the teacher's
    behaviour whether or not the answer is right; filtering on correctness is a
    rejection-sampling choice, and the strict tier-A gate silently mixed the two,
    discarding 70% of demonstrations — the HARD questions, where the teacher's
    search/read behaviour is most informative and where "commit to an answer anyway"
    is actually demonstrated (the 418/1,210-example adapters never see that, hence F12).

    What the correctness gate was doing FOR us by accident: 21% of teacher episodes cite
    a passage they never opened (read_before_cite_rate < 1), and read-before-cite is the
    one behaviour this SFT stage exists to install (finding F1: a citation without a read
    is unscoreable for RL). So filter on PROCESS, not outcome: the teacher reached an
    answer, cited something, and cited ONLY what it read. Wrong answers are kept and
    their answer turn IS graded. Measured on the 4,000: 2,692 pass (2,128 correct, 564
    wrong); 1,105 drop for citing-without-reading, 189 for never answering."""
    return (bool(rec.get("history")) and not rec.get("api_gave_up")
            and bool(rec.get("terminated_cleanly")) and rec.get("n_citations", 0) > 0
            and rec.get("read_before_cite_rate", 0.0) >= 0.999)


def tier_of(rec: dict, cite_bar: float) -> str:
    if rec.get("api_gave_up"):
        return "drop_api"
    if not rec.get("history"):
        return "drop_nohistory"
    if rec.get("correct") and rec.get("cite_f1", 0.0) >= cite_bar:
        return "A"
    gold = {_norm(t) for t in rec.get("supporting_titles", [])}
    got = {_norm(t) for t in read_titles(rec["history"])}
    hit = len(gold & got) / max(1, len(gold))
    return "B" if hit >= 0.5 else "C"


def student_prompt(question: str, cfg) -> str:
    """The prompt the STUDENT will see at SFT, RL rollout, and eval time — identical at
    all three, which is the requirement. `_opening_prompt` only reads `task.question`."""
    lean = replace(cfg, include_worked_example=False)
    return env_mod._opening_prompt(SimpleNamespace(question=question), lean)


def rebuild_history(rec: dict, cfg) -> list[dict]:
    h = [dict(m) for m in rec["history"]]
    h[0] = {"role": "user", "content": student_prompt(rec["question"], cfg)}
    return h


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default=str(JSONL))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--tiers", default="A", choices=["A", "AB", "P"],
                    help="A: correct+perfectly cited; AB: + process-good/answer-wrong with the "
                         "answer turn SKIPPED; P: every process-clean episode (cited only what "
                         "it read, reached an answer) with EVERY turn graded, right or wrong")
    ap.add_argument("--cite-bar", type=float, default=0.999)
    ap.add_argument("--max-len", type=int, default=3072)
    args = ap.parse_args()

    cfg = dr_config.Config.cloud_preset()
    recs = [json.loads(l) for l in Path(args.jsonl).read_text().splitlines() if l.strip()]
    print(f"Building SFT dataset from {len(recs)} collected episodes")
    print(f"  tiers={args.tiers}  cite_bar={args.cite_bar}  max_len={args.max_len}")

    tiers = Counter(tier_of(r, args.cite_bar) for r in recs)
    print(f"  tiering: {dict(tiers)}")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.model_name)

    # Verify the re-render actually removed the teacher's scaffolding before building
    # thousands of examples on top of it.
    sample = student_prompt(recs[0]["question"], cfg)
    stored = recs[0]["history"][0]["content"]
    print(f"\n  prompt re-render check:")
    # Anchor on text that is ACTUALLY in the collection prompt. An earlier version
    # checked for a sentence that had been accidentally deleted from the teacher prompt,
    # so the assertion passed vacuously and verified nothing.
    SUFFIX_MARK = "Write your Thought as ONE short sentence"
    print(f"    stored (collection) : {len(stored):5} chars, "
          f"worked example={'Worked example' in stored}, "
          f"teacher suffix={SUFFIX_MARK in stored}")
    print(f"    rebuilt (student)   : {len(sample):5} chars, "
          f"worked example={'Worked example' in sample}, "
          f"teacher suffix={SUFFIX_MARK in sample}")
    assert SUFFIX_MARK in stored, (
        "the collection prompt does not contain the teacher suffix — this check is "
        "vacuous and is not verifying the re-render")
    assert "Worked example" not in sample, "re-render failed: example still present"
    assert SUFFIX_MARK not in sample, "re-render failed: teacher suffix still present"

    keep_tiers = set(args.tiers)
    if args.tiers == "P":
        n_p = sum(1 for r in recs if is_process_clean(r))
        n_pw = sum(1 for r in recs if is_process_clean(r) and not r.get("correct"))
        print(f"  tier P (process-clean, all turns graded): {n_p} episodes, "
              f"{n_pw} of them with a WRONG final answer (kept on purpose)")
    examples, kept_meta, dropped = [], [], Counter()
    for rec in recs:
        if args.tiers == "P":
            if not is_process_clean(rec):
                dropped["not_process_clean"] += 1
                continue
            t = "P" if rec.get("correct") else "P_wrong"
        else:
            t = tier_of(rec, args.cite_bar)
            if t not in keep_tiers:
                dropped[t] += 1
                continue
        h = rebuild_history(rec, cfg)
        ex = encode_trajectory(tok, h, task_id=rec["task_id"], max_len=args.max_len,
                               skip_final_answer=(t == "B"))
        if ex is None:
            dropped["too_long_or_unencodable"] += 1
            continue
        try:
            assert_masking_correct(tok, ex, h)
        except AssertionError as e:                               # noqa: BLE001
            dropped["masking_violation"] += 1
            print(f"    !! masking violation, dropped: {e}")
            continue
        examples.append(ex)
        kept_meta.append({"task_id": rec["task_id"], "tier": t,
                          "correct": rec.get("correct"), "cite_f1": rec.get("cite_f1")})

    print(f"\n  kept    : {len(examples)}")
    print(f"  dropped : {dict(dropped)}")
    if not examples:
        raise SystemExit("nothing kept — loosen --cite-bar or collect more")

    print("\n=== DATASET SHAPE ===")
    for k, v in summarize(examples).items():
        print(f"  {k:26} {v:.3f}" if isinstance(v, float) else f"  {k:26} {v}")
    lens = sorted(len(e.input_ids) for e in examples)
    print(f"  seq len p50/p90/p99/max   {lens[len(lens)//2]} / {lens[int(len(lens)*.9)]} / "
          f"{lens[int(len(lens)*.99)]} / {lens[-1]}")
    print(f"  tiers kept                {dict(Counter(m['tier'] for m in kept_meta))}")
    print(f"  distinct questions        {len({m['task_id'] for m in kept_meta})}")

    import torch
    torch.save({"input_ids": [e.input_ids for e in examples],
                "labels": [e.labels for e in examples],
                "meta": kept_meta,
                "config": {"model": cfg.model_name, "tiers": args.tiers,
                           "cite_bar": args.cite_bar, "max_len": args.max_len,
                           "prompt": "include_worked_example=False (student prompt)"}},
               args.out)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
