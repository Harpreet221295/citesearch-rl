"""Diagnosis 1 — in-context capability check (RFT_PLAN_AND_MODEL_DIAGNOSIS.md).

Question this answers: given RICH worked examples (well beyond the single
search->answer snippet in env.py's real training prompt), can the model follow
multi-hop tool use at all — call `read`, not just `search`, terminate cleanly, and
answer correctly with valid citations? This is a pure CAPABILITY probe: no
training, no gradient updates. If the model can't do this even with every possible
in-context help, GRPO has nothing to sharpen and Diagnosis 2 (RFT) is unlikely to
help either — see the plan doc's decision gate.

Deliberately reuses, unchanged, everything already built and verified:
  - `evaluate.run_batched_rollouts` — the batched-vLLM multi-turn rollout loop
    (no HF `.generate()`, per RUNPOD_PLAYBOOK.md pattern #3).
  - `citations.verify_citations` / `metrics.exact_match` — the SAME correctness/
    citation scoring GRPO's reward uses, so "correct" means the same thing here as
    it will in training.
  - `env.DeepResearchEnv` — the richer prompt below is injected via a temporary,
    scoped monkey-patch of `env._opening_prompt` (see `patched_rich_prompt`), NOT a
    permanent replacement of env.py's real prompt — this is a stronger prompt than
    what's committed for training, used only for this diagnostic. (env.py itself
    DID get two real fixes on 2026-08-25, found running this diagnostic against the
    real model: `_ACTION_RE`'s greedy-match-across-lines bug, and an explicit
    "stop after one action" instruction added to the real prompt too — see env.py's
    own inline comments for both.)

Usage (pod, real model):
    python rft_diagnosis/diagnosis1.py                          # probe pool, cloud preset (Qwen2.5-3B)
    python rft_diagnosis/diagnosis1.py --pool heldout            # the frozen eval set (run LAST, see plan doc)
    python rft_diagnosis/diagnosis1.py --n-questions 32 --k 8 --temperatures 0.3,0.7,0.9,1.0
    python rft_diagnosis/diagnosis1.py --adapter runs/.../checkpoint   # test a checkpoint instead of base

Offline (no GPU): `pytest -q rft_diagnosis/tests/` exercises the pure-Python pieces
(the rich prompt construction, the classify/aggregate logic) via the SAME scripted-
policy pattern tests/test_evaluate.py already uses — no model needed to trust the
mechanics before spending GPU time on the real thing.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

_HERE = Path(__file__).resolve().parent.parent   # .../deep_research_agent (this file lives in rft_diagnosis/)
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import config as dr_config
import data as dr_data
import citations
import metrics as dr_metrics
import env as env_mod
from evaluate import build_vllm_engine, close_vllm_engine, make_lora_request, run_batched_rollouts


# --------------------------------------------------------------------------- #
# The rich, multi-example prompt — 3 worked trajectories, varying hop-count and
# failure/recovery shape. Goes well beyond env.py's real training prompt (one
# search->answer snippet as of the 2026-08-24 fix; still just ONE worked example).
# --------------------------------------------------------------------------- #
# All three use the SAME fictional universe as env.py's existing example (Blue
# Harvest / Jane Doe) so nothing here is new "content" the model could piggyback
# on knowing — only the demonstrated PATTERN varies. Fictional entities deliberately
# distinct per example so they don't collide with any real corpus passage titles.
_RICH_EXAMPLES = (
    # Example A — 2-hop, clean. Every Action is followed by the ACTUAL tool-response
    # turn the model would really receive (verbatim in the format tools.py produces:
    # "search results:\n[1] Title: snippet..." / "[Title]\n<full text>") before the
    # next Thought — not skipped, the way an earlier draft of this prompt (and
    # env.py's real committed example) did. Showing the real turn shape matters: the
    # model needs to see what it's reacting to, not just narration of the outcome.
    "Worked example 1, a 2-hop question "
    "(\"What nationality is the director of the film Blue Harvest?\"):\n"
    "Thought: First I need to find who directed Blue Harvest.\n"
    "Action: search[who directed the film Blue Harvest]\n"
    "search results:\n"
    "[1] Blue Harvest (film): Blue Harvest is a 2009 drama film. It was directed by "
    "Jane Doe and premiered at the Toronto festival. The film follows a family farm "
    "over one summer.\n"
    "[2] Blue Harvest (album): Blue Harvest is a 2015 studio album by the Norwegian "
    "band Fjord. It reached number three on the domestic chart.\n"
    "[3] John Roe (producer): John Roe is a British film producer who has worked on "
    "several independent dramas.\n"
    "Thought: [1] is the film, and it names Jane Doe as director. Reading it to "
    "confirm before citing.\n"
    "Action: read[Blue Harvest (film)]\n"
    "[Blue Harvest (film)]\n"
    "Blue Harvest is a 2009 drama film. It was directed by Jane Doe and premiered at "
    "the Toronto festival. The film follows a family farm over one summer.\n"
    "Thought: Confirmed — Jane Doe directed it. Now I need her nationality.\n"
    "Action: search[Jane Doe director nationality]\n"
    "search results:\n"
    "[1] Jane Doe (director): Jane Doe is an American film director, born in Ohio in "
    "1971. She is known for independent dramas including Blue Harvest and Winter "
    "Wheat.\n"
    "[2] Blue Harvest (film): Blue Harvest is a 2009 drama film. It was directed by "
    "Jane Doe and premiered at the Toronto festival.\n"
    "[3] John Roe (producer): John Roe is a British film producer who has worked on "
    "several independent dramas.\n"
    "Thought: [1] says she is American. Reading it to confirm before citing.\n"
    "Action: read[Jane Doe (director)]\n"
    "[Jane Doe (director)]\n"
    "Jane Doe is an American film director, born in Ohio in 1971. She is known for "
    "independent dramas including Blue Harvest and Winter Wheat.\n"
    "Thought: Confirmed. I have both facts, each grounded in a passage I read.\n"
    "Action: answer[American [Blue Harvest (film)] [Jane Doe (director)]]\n\n"

    # Example B — 3-hop, AND the FULL read text corrects what the SNIPPET alone
    # implied (the snippet is deliberately truncated before the correcting fact,
    # exactly like tools.py's real snippet() truncation — the read reveals more).
    "Worked example 2, a 3-hop question where reading the full passage changes the "
    "answer a snippet alone would have given "
    "(\"In what year did the founder of the company that manufactures the "
    "'Halcyon Drive' bicycle first found a company?\"):\n"
    "Thought: First I need to find who manufactures the Halcyon Drive bicycle.\n"
    "Action: search[manufacturer of Halcyon Drive bicycle]\n"
    "search results:\n"
    "[1] Halcyon Drive: Halcyon Drive is a carbon-frame touring bicycle manufactured "
    "by Ridgeline Cycles, released in 2018 and marketed toward long-distance riders.\n"
    "[2] Halcyon (mythology): In Greek mythology, the halcyon is a bird associated "
    "with calm seas during the winter solstice.\n"
    "[3] Meridian Bikes: Meridian Bikes is an unrelated bicycle brand based in "
    "Denver, known for mountain bikes.\n"
    "Thought: [1] says it's made by Ridgeline Cycles. Reading to confirm before "
    "moving on.\n"
    "Action: read[Halcyon Drive]\n"
    "[Halcyon Drive]\n"
    "Halcyon Drive is a carbon-frame touring bicycle manufactured by Ridgeline "
    "Cycles, released in 2018 and marketed toward long-distance riders.\n"
    "Thought: Confirmed — Ridgeline Cycles. Now I need who founded Ridgeline Cycles.\n"
    "Action: search[founder of Ridgeline Cycles]\n"
    "search results:\n"
    "[1] Ridgeline Cycles: Ridgeline Cycles is a bicycle manufacturer headquartered "
    "in Portland, founded in 2010 by Tom Ashford. The company specializes in "
    "touring and gravel bikes, including the Halcyon Drive mod…\n"
    "[2] Tom Ashford (engineer): Tom Ashford is an American mechanical engineer "
    "known for frame-design patents filed in the early 2000s.\n"
    "[3] Portland Cycling Co-op: A nonprofit organization supporting cyclists in "
    "Portland, unrelated to Ridgeline Cycles.\n"
    "Thought: [1] names Tom Ashford as founder, in 2010. But the question asks when "
    "he FIRST founded a company — the snippet is truncated ('…'), so 2010 might just "
    "be when Ridgeline itself started, not his first company. Reading the FULL "
    "passage to check, not just trusting the snippet.\n"
    "Action: read[Ridgeline Cycles]\n"
    "[Ridgeline Cycles]\n"
    "Ridgeline Cycles is a bicycle manufacturer headquartered in Portland, founded "
    "in 2010 by Tom Ashford. The company specializes in touring and gravel bikes, "
    "including the Halcyon Drive model. Before starting Ridgeline, Ashford had "
    "founded Ashford Metalworks in 2003, a small custom frame-building shop that "
    "later closed when he shifted focus to full bicycle manufacturing.\n"
    "Thought: The full passage reveals a fact the snippet cut off: Ashford founded "
    "Ashford Metalworks in 2003, before Ridgeline in 2010. The question asks for his "
    "FIRST company, so the answer is 2003, not what the snippet alone implied.\n"
    "Action: answer[2003 [Halcyon Drive] [Ridgeline Cycles]]\n\n"

    # Example C — first search misses (an unrelated album, not the novel);
    # REFORMULATE and recover, don't give up or answer from a bad hit. Also
    # resolves a same-name distractor (a different, unrelated Priya Anand), and
    # grounds the final geography claim in a passage actually read rather than
    # assuming it — "base your answer only on what you retrieve" applies to every
    # hop, not just the named entity.
    "Worked example 3, a question where the first search doesn't find the right "
    "passage and needs reformulating "
    "(\"What continent is the birthplace city of the author who wrote 'Coral Bell "
    "Testament' located on?\"):\n"
    "Thought: I need to find who wrote 'Coral Bell Testament'.\n"
    "Action: search[author of Coral Bell Testament]\n"
    "search results:\n"
    "[1] Coral Bell Testament (album): Coral Bell Testament is a 2016 concept album "
    "by the band Wintervale, exploring themes of memory and loss.\n"
    "[2] Testament Records: An independent record label based in Bristol, known for "
    "folk and ambient releases.\n"
    "[3] Coral Reef Conservation Trust: A nonprofit focused on marine ecosystem "
    "preservation, unrelated to the album or novel.\n"
    "Thought: None of these name an author of a NOVEL — [1] is an album, [2] and [3] "
    "are unrelated. My query was too generic. Searching more specifically for the "
    "novel.\n"
    "Action: search[novel Coral Bell Testament writer]\n"
    "search results:\n"
    "[1] Coral Bell Testament (novel): Coral Bell Testament is a 2011 novel written "
    "by Priya Anand, following three generations of a coastal family through decades "
    "of change.\n"
    "[2] Coral Bell Testament (album): Coral Bell Testament is a 2016 concept album "
    "by the band Wintervale, exploring themes of memory and loss.\n"
    "[3] Priya Anand (poet): Priya Anand is an unrelated poet based in Mumbai, known "
    "for a 2019 collection of verse.\n"
    "Thought: [1] is the novel, naming Priya Anand as author. Note [3] is a "
    "DIFFERENT Priya Anand (a poet) — I need the novelist, not her. Reading [1] to "
    "confirm before moving on.\n"
    "Action: read[Coral Bell Testament (novel)]\n"
    "[Coral Bell Testament (novel)]\n"
    "Coral Bell Testament is a 2011 novel written by Priya Anand, following three "
    "generations of a coastal family through decades of change.\n"
    "Thought: Confirmed — Priya Anand (the novelist). Now I need her birthplace.\n"
    "Action: search[Priya Anand novelist birthplace]\n"
    "search results:\n"
    "[1] Priya Anand (author): Priya Anand is a novelist born in Accra. She has "
    "written several works exploring family and migration, including Coral Bell "
    "Testament.\n"
    "[2] Priya Anand (poet): Priya Anand is an unrelated poet based in Mumbai, known "
    "for a 2019 collection of verse.\n"
    "[3] Accra: Accra is the capital and largest city of Ghana, located on the Gulf "
    "of Guinea coast.\n"
    "Thought: [1] is the right Priya Anand (matches the novel) and says she was born "
    "in Accra. Reading to confirm before citing.\n"
    "Action: read[Priya Anand (author)]\n"
    "[Priya Anand (author)]\n"
    "Priya Anand is a novelist born in Accra. She has written several works "
    "exploring family and migration, including Coral Bell Testament.\n"
    "Thought: Confirmed — born in Accra. I should ground the CONTINENT in a passage "
    "too, not just assume it. Reading the Accra passage from the search results.\n"
    "Action: read[Accra]\n"
    "[Accra]\n"
    "Accra is the capital and largest city of Ghana, located on the Gulf of Guinea "
    "coast in West Africa.\n"
    "Thought: Confirmed — Accra is in Africa, and I've now read every passage I'm "
    "citing.\n"
    "Action: answer[Africa [Coral Bell Testament (novel)] [Priya Anand (author)] "
    "[Accra]]\n"
)


def rich_opening_prompt(task, cfg) -> str:
    """A STRONGER variant of env._opening_prompt for this diagnostic only — 3 worked
    examples (varying hop-count, a correcting-read case, a search-miss-then-recover
    case) instead of the one real training uses. Mirrors env._opening_prompt's
    preamble verbatim (not imported, since that function isn't parameterized for
    this) so the only variable being tested is "how many/how rich the examples are",
    not any wording drift in the rest of the instructions."""
    from tools import render_toolset
    if not getattr(cfg, "require_citations", True):
        # Rich multi-example citation demos don't have a natural no-cite analogue
        # worth maintaining separately — fall back to the real prompt in this case.
        return env_mod._opening_prompt(task, cfg)
    cite_rule = (
        "Ground your answer ONLY in passages you retrieved. CITE EVERY passage that "
        "supports your answer — usually ONE PER FACT/HOP — by its exact title in square "
        "brackets. Multi-hop answers need MULTIPLE citations, e.g. "
        "answer[American [Blue Harvest (film)] [Jane Doe (director)]]. Do not cite a "
        "passage you did not read.\n"
    )
    return (
        "You are a research agent. Answer the question by SEARCHING a document "
        "corpus, READING the most relevant passages, then giving a short, grounded "
        "final answer. Base your answer only on what you retrieve.\n"
        "Answer with the SHORTEST possible phrase — usually 1-3 words: the exact "
        "entity, name, number, or 'yes'/'no'. No sentence, no explanation.\n"
        + cite_rule + "\n"
        "Tools:\n" + render_toolset(list(cfg.tools)) + "\n\n"
        "Format each turn as:\nThought: <your reasoning>\nAction: <tool>[<argument>]\n"
        "Write EXACTLY ONE Thought/Action pair, then STOP — do not write what the "
        "tool returns yourself; the real result will be given to you before your "
        "next turn.\n\n"
        + _RICH_EXAMPLES +
        f"\nQuestion: {task.question}"
    )


@contextlib.contextmanager
def patched_rich_prompt():
    """Temporarily swap env._opening_prompt for the rich version, for the duration
    of this diagnostic run only. Does NOT touch env.py or affect GRPO training's
    actual prompt — the patch is undone in `finally` even on error."""
    original = env_mod._opening_prompt
    env_mod._opening_prompt = rich_opening_prompt
    try:
        yield
    finally:
        env_mod._opening_prompt = original


# --------------------------------------------------------------------------- #
# Data hygiene: the probe pool is a deterministic slice of the TRAIN split
# (train_seed), distinct from the frozen held-out eval set (eval_seed, already
# the project's existing convention — see data.load_tasks). Loading once with a
# large-enough count then slicing in Python keeps `offset` slices reproducible;
# calling load_tasks twice with different `n` does NOT give a stable prefix,
# because the final shuffle's permutation depends on the pre-shuffle list length.
# --------------------------------------------------------------------------- #
def load_pool(cfg, split: str, n: int, offset: int = 0):
    """A deterministic, offset-stable slice of `split` ("train" or "eval"). Used to
    carve the Diagnosis-1 probe pool and (later) Diagnosis-2's collection pool out
    of the SAME train draw without overlapping — pass distinct `offset`s."""
    big_cfg = replace(cfg, num_train_examples=offset + n, num_eval_examples=offset + n)
    tasks = dr_data.load_tasks(big_cfg, split)
    return tasks[offset: offset + n]


# --------------------------------------------------------------------------- #
# Per-trajectory classification — the full breakdown, not just pass/fail. Reuses
# the SAME scoring code as training/eval (citations.py, metrics.py) — "correct"
# and "well-cited" mean the same thing here as everywhere else in this lab.
# --------------------------------------------------------------------------- #
def classify_trajectory(task, traj, cfg) -> dict:
    n_reads = sum(1 for c in traj.tool_calls if c.name == "read")
    n_searches = traj.n_searches
    parse_failures = sum(1 for s in traj.steps if not s.parse_ok)
    pred = citations.strip_citations(traj.final_answer or "")
    correct = dr_metrics.exact_match(pred, task.answers)
    creport = citations.verify_citations(
        traj, backend=cfg.citation_backend,
        align_threshold=cfg.citation_align_threshold,
        gold_titles=task.supporting_titles)

    if not correct:
        bucket = "wrong_answer"
    elif not creport.any_citation:
        bucket = "correct_uncited"
    elif creport.f1 < 0.999:                    # cited something, but not perfectly
        bucket = "correct_miscited"
    else:
        bucket = "correct_and_cited"

    return {
        "task_id": task.task_id,
        "parse_ok": parse_failures == 0,
        "n_parse_failures": parse_failures,
        "calls_read": n_reads > 0,
        "n_reads": n_reads,
        "n_searches": n_searches,
        "n_tool_calls": len(traj.tool_calls),
        "terminated_cleanly": bool(traj.done),
        "correct": correct,
        "n_citations": creport.n_citations,
        "cite_f1": creport.f1,
        "cite_precision": creport.precision,
        "cite_recall": creport.recall,
        # 2026-08-26 split (see citations.CitationReport): cite_f1 above is conjunctive
        # (cited AND gold AND read) so a zero cannot say WHICH half failed. These two
        # separate "picked the right sources" from "verified them first".
        "title_f1": creport.title_f1,
        "read_before_cite_rate": creport.read_before_cite_rate,
        "bucket": bucket,
    }


def aggregate_breakdown(rows: list[dict]) -> dict:
    """The full diagnostic summary for one temperature's sample — rates + bucket
    breakdown + hop-count distribution, not a single pass/fail number."""
    n = len(rows)
    if n == 0:
        return {"n": 0}
    bucket_counts = Counter(r["bucket"] for r in rows)
    hop_counts = Counter(r["n_tool_calls"] for r in rows)
    return {
        "n": n,
        "parse_ok_rate": sum(r["parse_ok"] for r in rows) / n,
        "calls_read_rate": sum(r["calls_read"] for r in rows) / n,
        "terminated_cleanly_rate": sum(r["terminated_cleanly"] for r in rows) / n,
        "correct_rate": sum(r["correct"] for r in rows) / n,
        "mean_n_reads": sum(r["n_reads"] for r in rows) / n,
        "mean_n_searches": sum(r["n_searches"] for r in rows) / n,
        "mean_cite_f1": sum(r["cite_f1"] for r in rows) / n,
        # The split. Read them together: a high title_f1 with a low read_before_cite_rate
        # means the model KNOWS the right sources and simply is not reading them — a very
        # different problem from not knowing which passages matter.
        "mean_title_f1": sum(r.get("title_f1", 0.0) for r in rows) / n,
        "mean_read_before_cite": sum(r.get("read_before_cite_rate", 0.0) for r in rows) / n,
        "bucket_pct": {k: v / n for k, v in bucket_counts.items()},
        "hop_count_distribution": dict(sorted(hop_counts.items())),
    }


# --------------------------------------------------------------------------- #
# Sampling — k trajectories per task, reusing run_batched_rollouts UNCHANGED by
# simply repeating each task k times in the input list. vLLM's own batched
# generate() call at each round already batches across every active env,
# including the k copies, so this needs no new rollout machinery.
# --------------------------------------------------------------------------- #
def sample_k_per_task(tasks, cfg, llm, tokenizer, sampling_params, k, lora_request=None):
    expanded = [t for t in tasks for _ in range(k)]
    return run_batched_rollouts(expanded, cfg, llm, tokenizer, sampling_params,
                                lora_request=lora_request)


# 2026-08-25: without this, generation doesn't stop at the true turn boundary —
# confirmed directly against the real model (see git log / TRAINING_HISTORY_LOG.md
# for the raw transcripts): the model sometimes free-runs past its first `Action:`
# and writes an entire hallucinated continuation (fake tool results, a fake second
# turn, a fake final answer) in one completion, imitating the worked examples' full
# multi-turn SHAPE rather than stopping after one step. env._parse_react_action's
# MULTILINE fix (same date) makes this safe to PARSE (extracts only the first real
# action), but stopping generation here is still worth doing — it's the mechanical
# enforcement of "one action per turn", not just a parser workaround, and it saves
# real tokens/GPU time (no point generating 200+ tokens of hallucinated content the
# parser is just going to discard). Every fake-continuation shape starts with one of
# these three strings immediately after the real Action line.
_TURN_STOP_SEQUENCES = ["\nThought:", "\nsearch results:", "\n["]


def run_diagnosis1(tasks, cfg, llm, tokenizer, k, temperatures, lora_request=None):
    """Sweep temperatures, sample k/task at each, classify + aggregate. Returns
    {temperature: {**aggregate_breakdown, "raw_rows": [...]}}."""
    from vllm import SamplingParams
    per_temp = {}
    for T in temperatures:
        sp = SamplingParams(temperature=T, top_p=cfg.top_p, max_tokens=cfg.max_new_tokens,
                            stop=_TURN_STOP_SEQUENCES)
        results = sample_k_per_task(tasks, cfg, llm, tokenizer, sp, k, lora_request)
        rows = [classify_trajectory(task, traj, cfg) for task, traj, _ in results]
        breakdown = aggregate_breakdown(rows)
        breakdown["raw_rows"] = rows
        per_temp[T] = breakdown
    return per_temp


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def print_report(per_temp: dict, cfg) -> None:
    print("\n=== Diagnosis 1 — in-context capability check ===")
    grpo_temp = getattr(cfg, "temperature", 0.9)
    for T in sorted(per_temp):
        b = per_temp[T]
        flag = "  <-- GRPO's own rollout temperature" if abs(T - grpo_temp) < 1e-6 else ""
        print(f"\n-- temperature={T}{flag} (n={b['n']}) --")
        print(f"  parse_ok_rate:          {b['parse_ok_rate']:.2f}")
        print(f"  calls_read_rate:        {b['calls_read_rate']:.2f}  "
              f"(did it call read at least once, not just search+answer?)")
        print(f"  terminated_cleanly_rate:{b['terminated_cleanly_rate']:.2f}  "
              f"(emitted a parsed answer vs hit max_turns)")
        print(f"  correct_rate (EM):      {b['correct_rate']:.2f}")
        print(f"  mean_cite_f1:           {b['mean_cite_f1']:.2f}")
        print(f"  mean_n_reads:           {b['mean_n_reads']:.2f}   "
              f"mean_n_searches: {b['mean_n_searches']:.2f}")
        print(f"  hop_count_distribution: {b['hop_count_distribution']}")
        print(f"  outcome buckets:        {b['bucket_pct']}")
    print("\n(pass/fail threshold is Harpreet's call — see RFT_PLAN_AND_MODEL_DIAGNOSIS.md "
          "'Diagnosis 1' section. This report is the full picture to judge it from, not a verdict.)")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Diagnosis 1 — in-context capability check")
    ap.add_argument("--preset", default="cloud", choices=["sanity", "default", "cloud"],
                    help="which Config preset's MODEL to test (default: cloud = the real "
                         "Qwen2.5-3B target — testing the sanity stand-in tells you nothing "
                         "about whether the actual model can do this)")
    ap.add_argument("--pool", choices=["probe", "heldout"], default="probe",
                    help="'probe' (default) = a Diagnosis-1-only slice of the train pool, "
                         "safe to iterate on repeatedly. 'heldout' = the frozen eval set — "
                         "per the plan doc's data-hygiene rule, run this once, at the end, "
                         "not while iterating on prompt variants.")
    ap.add_argument("--pool-offset", type=int, default=0,
                    help="offset into the train pool (keep DISTINCT from whatever offset "
                         "Diagnosis 2's collection pass uses, so the two never overlap)")
    ap.add_argument("--n-questions", type=int, default=16)
    ap.add_argument("--k", type=int, default=4, help="sampled trajectories per question per temperature")
    ap.add_argument("--temperatures", default="0.3,0.7,0.9",
                    help="comma-separated sweep; include cfg.temperature (0.9 by default) "
                         "to get the number that actually predicts GRPO viability")
    ap.add_argument("--adapter", default=None, help="test a checkpoint instead of the base model")
    ap.add_argument("--out", default=None, help="write the full breakdown + per-trajectory rows as JSON here")
    args = ap.parse_args()

    from dotenv import load_dotenv
    load_dotenv(_HERE / ".env")

    cfg = {"sanity": dr_config.Config.sanity_preset,
           "default": dr_config.Config.default,
           "cloud": dr_config.Config.cloud_preset}[args.preset]()

    if args.pool == "heldout":
        tasks = dr_data.load_tasks(cfg, "eval")[: args.n_questions]
    else:
        tasks = load_pool(cfg, "train", args.n_questions, offset=args.pool_offset)

    temperatures = [float(t) for t in args.temperatures.split(",")]

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.model_name)

    print(f"Diagnosis 1: {len(tasks)} questions x k={args.k} x {len(temperatures)} "
          f"temperatures {temperatures} on {cfg.model_name}"
          f"{' + adapter ' + args.adapter if args.adapter else ' (base)'}, pool={args.pool}")

    llm = build_vllm_engine(cfg, enable_lora=bool(args.adapter))
    lora_req = make_lora_request(args.adapter) if args.adapter else None
    try:
        with patched_rich_prompt():
            per_temp = run_diagnosis1(tasks, cfg, llm, tok, args.k, temperatures,
                                       lora_request=lora_req)
    finally:
        close_vllm_engine(llm)

    print_report(per_temp, cfg)

    if args.out:
        serializable = {
            str(T): {k: v for k, v in b.items() if k != "raw_rows"}
            for T, b in per_temp.items()
        }
        for T, b in per_temp.items():
            serializable[str(T)]["raw_rows"] = b["raw_rows"]
        Path(args.out).write_text(json.dumps(serializable, indent=2))
        print(f"\nwrote full breakdown + raw rows to {args.out}")


if __name__ == "__main__":
    main()
