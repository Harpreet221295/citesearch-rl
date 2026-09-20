"""Eval gate — base vs tuned on a FROZEN held-out set, with localized metrics and the
anti-hacking probes (spec in NOTES.md). CLAUDE.md: eval is part of the task, never optional.

Key design: the CORE (rollout_episode/evaluate_set/probes/gate) is FRAMEWORK-AGNOSTIC — it
drives DeepResearchEnv with a `policy_fn` (conversation -> next assistant text) one episode
at a time, so it runs offline in tests with a SCRIPTED policy (no model, no GPU). That core
is unchanged and still what tests/test_evaluate.py exercises.

The POD-FACING path (`main`) does NOT use that serial per-episode loop or HF `generate()` —
per project convention (RUNPOD_PLAYBOOK.md pattern #3: "batched generation is THE cost
lever"; no inefficient HF methods for eval/inference), it drives ALL held-out episodes
CONCURRENTLY through one shared vLLM engine (`run_batched_rollouts`), batching every round's
"next assistant turn" across every still-active episode into a single `LLM.generate()` call.
Base vs tuned reuse the SAME resident base-model weights via vLLM's LoRA hot-swap
(`lora_request=None` for base, a `LoRARequest` for tuned) — no second model copy loaded.
The judge (if requested) is likewise vLLM-batched (judge.py `judge_backend="vllm"`), loaded
AFTER the policy engine is freed (`close_vllm_engine`) so both phases fit one GPU.

Usage (pod):
    python evaluate.py --adapter runs/deep_research_agent_cloud/best             # tuned vs base
    python evaluate.py --adapter ... --judge vllm                                # + judge cross-check
    python evaluate.py --adapter ... --judge vllm --wandb                        # + log to W&B

Metrics (all rule-based unless --judge): answer EM/F1, retrieval hit-rate, citation-F1
(precision/recall), fabricated-cite rate, avg #steps, answer length. Probes compare base
vs tuned and flag the named hacks (judge-vs-grounded gap, research-decoupling, fabrication,
length inflation).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import config as dr_config
import data as dr_data
import metrics as dr_metrics
import reward as dr_reward
import citations
from env import DeepResearchEnv


# --------------------------------------------------------------------------- #
# rollout: drive the env with a policy_fn (framework-agnostic)
# --------------------------------------------------------------------------- #
def rollout_episode(task, cfg, policy_fn, judge=None):
    """Run one eval episode. policy_fn(history: list[{role,content}]) -> assistant text.
    Returns (trajectory, RewardInfo). Greedy generation is the caller's choice inside policy_fn."""
    env = DeepResearchEnv.from_dict({"task": task, "cfg": cfg, "judge": judge})
    env.reset()
    done = False
    reward, info = 0.0, {}
    while not done:
        action = policy_fn(env._history)
        _obs, reward, done, info = env.step(action)
    traj = env._to_dr_trajectory()
    # score with the SAME reward code (single source of truth), judge optional at eval
    rinfo = dr_reward.grounded_outcome(task, traj, judge, cfg)
    return traj, rinfo


def evaluate_set(tasks, cfg, policy_fn, judge=None) -> dict:
    """Roll every task once and aggregate the localized metric row."""
    rows = []
    for task in tasks:
        traj, ri = rollout_episode(task, cfg, policy_fn, judge)
        rows.append(_row_from_result(task, traj, ri))
    return dr_metrics.aggregate(rows)


def _row_from_result(task, traj, ri) -> dict:
    """Same per-example metric row evaluate_set builds, factored out so the batched
    vLLM path (below) and the serial framework-agnostic path score identically —
    single source of truth for what a 'row' is."""
    pred = citations.strip_citations(traj.final_answer or "")
    return {
        "em": float(ri.correct),
        "f1": dr_metrics.token_f1(pred, task.answers),
        "hit_rate": ri.hit_rate,
        "cite_f1": ri.cite_f1,
        "cite_precision": ri.cite_precision,
        "cite_recall": ri.cite_recall,
        "fabricated_rate": 1.0 if ri.cite_fabricated > 0 else 0.0,
        "avg_steps": float(ri.n_steps),
        "judge": ri.judge,
        "overlap": ri.overlap,
        "answer_len": float(len(pred.split())),
    }


# --------------------------------------------------------------------------- #
# BATCHED multi-episode rollout via vLLM (the pod-facing eval/inference path)
# --------------------------------------------------------------------------- #
# Project convention (RUNPOD_PLAYBOOK.md pattern #3, and explicit instruction): the
# eval/inference generation step must NOT use HF `model.generate()`. This drives every
# held-out episode CONCURRENTLY through one shared vLLM engine, advancing them in
# lockstep and batching each round's "next assistant turn" across every still-active
# episode into a single `LLM.generate()` call — the same cost lever training rollouts
# use, applied to eval. rollout_episode/evaluate_set above are UNCHANGED (still the
# offline-testable, framework-agnostic reference the tests exercise); this is an
# ADDITIONAL path used only by `main()` on the pod.
def build_vllm_engine(cfg, enable_lora: bool = False):
    """One vLLM engine for the POLICY model. With enable_lora=True, base (no adapter)
    and tuned (adapter) generation share this SAME resident engine via LoRA hot-swap
    per-call (`lora_request=None` vs a `LoRARequest`) — no second model copy loaded."""
    from vllm import LLM
    kwargs = dict(
        model=cfg.model_name,
        dtype="bfloat16" if cfg.bf16 else "auto",
        gpu_memory_utilization=getattr(cfg, "verl_gpu_mem_util", 0.6),
        tensor_parallel_size=getattr(cfg, "verl_tensor_parallel", 1),
        trust_remote_code=True,
        seed=cfg.eval_seed,
    )
    if enable_lora:
        kwargs.update(enable_lora=True, max_loras=1, max_lora_rank=cfg.lora_r)
    return LLM(**kwargs)


def close_vllm_engine(llm, timeout_s: float = 90.0) -> None:
    """Free a vLLM engine's GPU memory before loading a DIFFERENT one in the same
    process (e.g. policy engine -> judge engine).

    vLLM V1 runs the actual engine in a SEPARATE SUBPROCESS ("EngineCore") —
    deleting the Python-level `LLM` handle does NOT synchronously free that
    subprocess's GPU memory. Verified empirically: a naive `del` + `gc.collect()` +
    `torch.cuda.empty_cache()` left ~48GB still resident seconds later, and the next
    `LLM(...)` call failed with vLLM's own "Free memory on device ... is less than
    desired GPU memory utilization" error — `LLMEngine.__del__` doesn't call
    `engine_core.shutdown()` at all (checked the installed vllm==0.22.1 source).
    Fix: explicitly call the engine-core client's `shutdown()`, THEN POLL
    `torch.cuda.mem_get_info()` until the memory is actually back (don't trust any
    particular vLLM version's shutdown call to be synchronous)."""
    import gc
    import time
    import torch
    try:
        llm.llm_engine.engine_core.shutdown()
    except Exception:
        pass   # best-effort: internal path; older/newer vLLM may not expose it
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    total = torch.cuda.mem_get_info()[1]
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        free, _ = torch.cuda.mem_get_info()
        if free / total > 0.85:      # the previous engine-core subprocess released it
            return
        time.sleep(2.0)
    free, _ = torch.cuda.mem_get_info()
    print(f"[warn] close_vllm_engine: only {free / total:.0%} GPU memory free after "
          f"{timeout_s}s — the previous engine's subprocess may not have fully exited. "
          f"Proceeding anyway; the next engine load may fail or OOM.")


def make_lora_request(adapter_dir: str, name: str = "tuned"):
    from vllm.lora.request import LoRARequest
    return LoRARequest(name, 1, adapter_dir)


def run_batched_rollouts(tasks, cfg, llm, tokenizer, sampling_params, lora_request=None):
    """Roll every task's episode CONCURRENTLY via one shared vLLM engine.
    Returns list[(task, trajectory, RewardInfo)] in the same order as `tasks`.
    Judge=None here ALWAYS — reward.grounded_outcome's judge.score() is a single-
    trajectory call, so folding a judge in here would score one-at-a-time and defeat
    batching. Judge scoring is a separate batched pass — see score_with_judge below."""
    envs = [DeepResearchEnv.from_dict({"task": t, "cfg": cfg, "judge": None}) for t in tasks]
    for env in envs:
        env.reset()
    done = [False] * len(envs)
    max_rounds = int(getattr(cfg, "max_turns", 8)) + 1   # +1: the forced-stop round
    for _round in range(max_rounds):
        active = [i for i, d in enumerate(done) if not d]
        if not active:
            break
        prompts = [tokenizer.apply_chat_template(envs[i]._history, add_generation_prompt=True,
                                                  tokenize=False) for i in active]
        outs = llm.generate(prompts, sampling_params, lora_request=lora_request, use_tqdm=False)
        for i, out in zip(active, outs):
            _obs, _reward, d, _info = envs[i].step(out.outputs[0].text)
            done[i] = d
    results = []
    for task, env in zip(tasks, envs):
        traj = env._to_dr_trajectory()
        rinfo = dr_reward.grounded_outcome(task, traj, None, cfg)
        results.append((task, traj, rinfo))
    return results


def score_with_judge(results: list, judge) -> None:
    """Batch-score every trajectory's groundedness through `judge` — ONE generate()
    call for the WHOLE set (judge.score_batch), never per-trajectory — and fold the
    score into each RewardInfo IN PLACE (RewardInfo is a plain mutable dataclass)."""
    if judge is None or not results:
        return
    tasks_ = [t for t, _, _ in results]
    trajs_ = [tr for _, tr, _ in results]
    scores = judge.score_batch(tasks_, trajs_)
    for (_, _, ri), s in zip(results, scores):
        ri.judge = float(s)


def aggregate_results(results: list) -> dict:
    rows = [_row_from_result(task, traj, ri) for task, traj, ri in results]
    return dr_metrics.aggregate(rows)


def sample_from_results(results: list, n: int = 3) -> list:
    return [{
        "question": task.question,
        "pred": citations.strip_citations(traj.final_answer or ""),
        "gold": task.gold_answer,
        "correct": bool(ri.correct),
        "n_steps": ri.n_steps,
        "cite_f1": ri.cite_f1,
    } for task, traj, ri in results[:n]]


def evaluate_set_batched(tasks, cfg, llm, tokenizer, sampling_params, lora_request=None,
                          n_samples: int = 3) -> tuple[list, list]:
    """The batched-vLLM analogue of evaluate_set — rollout ONLY (no judge; that's a
    separate pass, see score_with_judge/aggregate_results). Returns (results, samples)."""
    results = run_batched_rollouts(tasks, cfg, llm, tokenizer, sampling_params,
                                    lora_request=lora_request)
    return results, sample_from_results(results, n_samples)


# --------------------------------------------------------------------------- #
# probes + gate  (NOTES.md spec)
# --------------------------------------------------------------------------- #
def probes(base: dict, tuned: dict, judge_used: bool) -> list[str]:
    """Return a list of FIRED probe warnings (empty = clean). Each maps to a named hack."""
    fired = []
    # 1. judge-vs-grounded gap (only meaningful if a judge ran)
    if judge_used and (tuned.get("judge", 0) - tuned.get("cite_f1", 0)) > 0.25:
        fired.append(f"JUDGE-GAMING: judge {tuned['judge']:.2f} >> citation-F1 "
                     f"{tuned['cite_f1']:.2f} (looks grounded to a judge, isn't)")
    # 2. research-decoupling: EM up but retrieval flat, or steps collapsed
    if tuned["em"] > base["em"] + 0.02 and tuned["hit_rate"] <= base["hit_rate"] + 0.01:
        fired.append("MEMORY-GUESSING: EM rose but retrieval hit-rate did not — "
                     "answering from memory, not research")
    if tuned["avg_steps"] < 1.5:
        fired.append(f"TOOL-AVOIDANCE: avg steps {tuned['avg_steps']:.1f} < 1.5 — "
                     "short-circuiting the multi-hop loop")
    # 3. fabrication spike
    if tuned["fabricated_rate"] > base["fabricated_rate"] + 0.05:
        fired.append(f"FABRICATION: fabricated-cite rate {tuned['fabricated_rate']:.2f} "
                     f"up from {base['fabricated_rate']:.2f}")
    # 4. length inflation without accuracy gain
    if tuned["answer_len"] > base["answer_len"] * 1.5 and tuned["f1"] <= base["f1"] + 0.02:
        fired.append("LENGTH-INFLATION: answers got much longer without an F1 gain")
    return fired


def gate(base: dict, tuned: dict, judge_used: bool, margin: float = 0.05) -> tuple[bool, list[str]]:
    """PASS only if tuned beats base by `margin` on EM AND no probe fired AND groundedness
    didn't regress. Returns (passed, reasons)."""
    reasons = []
    em_gain = tuned["em"] - base["em"]
    if em_gain < margin:
        reasons.append(f"EM gain {em_gain:+.3f} < required margin {margin}")
    if tuned["cite_f1"] < base["cite_f1"] - 0.02:
        reasons.append(f"citation-F1 regressed ({base['cite_f1']:.2f} -> {tuned['cite_f1']:.2f})")
    fired = probes(base, tuned, judge_used)
    reasons += fired
    return (len(reasons) == 0), reasons


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
_ROW_KEYS = ["em", "f1", "hit_rate", "cite_f1", "cite_precision", "cite_recall",
             "fabricated_rate", "avg_steps", "answer_len", "judge"]


def print_report(base: dict, tuned: dict, judge_used: bool, margin: float = 0.05):
    print("\n=== held-out eval: base vs tuned ===")
    print(f"{'metric':<18}{'base':>10}{'tuned':>10}{'Δ':>10}")
    for k in _ROW_KEYS:
        b, t = base.get(k, 0.0), tuned.get(k, 0.0)
        print(f"{k:<18}{b:>10.3f}{t:>10.3f}{t-b:>+10.3f}")
    passed, reasons = gate(base, tuned, judge_used, margin)
    print(f"\nGATE: {'PASS ✅' if passed else 'FAIL ❌'}")
    for r in reasons:
        print(f"  - {r}")
    return passed


# --------------------------------------------------------------------------- #
# W&B logging — RUNPOD_PLAYBOOK.md pattern #8: log more than a pass/fail line.
# --------------------------------------------------------------------------- #
def log_eval_to_wandb(cfg, base: dict, tuned: dict, passed: bool, reasons: list[str],
                       base_samples: list, tuned_samples: list, run_name: str | None = None):
    """Log the full eval gate to W&B: the base/tuned/delta metric table (scalars, so
    they show up alongside the training curves), the gate verdict + fired-probe
    reasons, and a sample table of real generations (question/pred/gold/correct) so
    you can eyeball BEHAVIOR, not just the aggregate numbers. Safe to call standalone
    (own run) or you could pass a live run's id in `run_name` to log into the SAME
    dashboard as training — kept as a fresh run by default so a repeated eval pass
    (e.g. from the checkpoint-picker) doesn't clobber the training curves."""
    import wandb
    run = wandb.init(project=cfg.wandb_project, name=run_name or f"{cfg.run_name}_eval",
                      job_type="eval", config={"margin": getattr(cfg, "eval_margin", 0.05)},
                      reinit="finish_previous")
    try:
        scalars = {}
        for k in _ROW_KEYS:
            scalars[f"eval/base/{k}"] = base.get(k, 0.0)
            scalars[f"eval/tuned/{k}"] = tuned.get(k, 0.0)
            scalars[f"eval/delta/{k}"] = tuned.get(k, 0.0) - base.get(k, 0.0)
        scalars["eval/gate_passed"] = int(passed)
        scalars["eval/n_probes_fired"] = len(reasons)
        run.log(scalars)
        run.summary["gate_passed"] = passed
        run.summary["gate_reasons"] = reasons

        cols = ["split", "question", "pred", "gold", "correct", "n_steps", "cite_f1"]
        table = wandb.Table(columns=cols)
        for s in base_samples:
            table.add_data("base", s["question"], s["pred"], s["gold"], s["correct"],
                           s["n_steps"], s["cite_f1"])
        for s in tuned_samples:
            table.add_data("tuned", s["question"], s["pred"], s["gold"], s["correct"],
                           s["n_steps"], s["cite_f1"])
        run.log({"eval/samples": table})
    finally:
        run.finish()


# --------------------------------------------------------------------------- #
# main — vLLM-driven, batched across the whole held-out set (see module docstring)
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None, help="path to the tuned LoRA adapter")
    ap.add_argument("--preset", default="cloud", choices=["sanity", "default", "cloud"])
    ap.add_argument("--judge", default=None, choices=["mock", "vllm", "hf"],
                    help="run a judge for the judge-vs-grounded probe (eval-only). "
                         "'vllm' is the batched, non-HF path — prefer it on the pod.")
    ap.add_argument("--margin", type=float, default=0.05)
    ap.add_argument("--wandb", action="store_true", help="log the gate + samples to W&B")
    ap.add_argument("--run-name", default=None, help="W&B run name override")
    args = ap.parse_args()

    from dotenv import load_dotenv
    load_dotenv(_HERE / ".env")            # HF_TOKEN, WANDB_API_KEY

    cfg = {"sanity": dr_config.Config.sanity_preset,
           "default": dr_config.Config.default,
           "cloud": dr_config.Config.cloud_preset}[args.preset]()
    if args.judge:
        cfg.judge_backend = args.judge
    tasks = dr_data.load_tasks(cfg, "eval")            # frozen held-out (eval_seed)

    from transformers import AutoTokenizer
    from vllm import SamplingParams
    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=cfg.max_new_tokens)

    print(f"evaluating {len(tasks)} held-out questions (batched vLLM, base + tuned share "
          f"one engine via LoRA hot-swap)...")
    llm = build_vllm_engine(cfg, enable_lora=bool(args.adapter))
    base_results, base_samples = evaluate_set_batched(tasks, cfg, llm, tok, sampling_params,
                                                       lora_request=None)
    tuned_results, tuned_samples = evaluate_set_batched(
        tasks, cfg, llm, tok, sampling_params,
        lora_request=make_lora_request(args.adapter) if args.adapter else None)
    close_vllm_engine(llm)   # free the policy engine before the judge (if any) loads its own

    judge_used = bool(args.judge)
    if judge_used:
        import judge as judge_mod
        judge = judge_mod.make_judge(cfg)
        score_with_judge(base_results, judge)     # ONE batched generate() call per set
        score_with_judge(tuned_results, judge)
        if hasattr(judge, "close"):
            judge.close()

    base = aggregate_results(base_results)
    tuned = aggregate_results(tuned_results)
    passed = print_report(base, tuned, judge_used=judge_used, margin=args.margin)
    _, reasons = gate(base, tuned, judge_used, args.margin)
    if args.wandb:
        log_eval_to_wandb(cfg, base, tuned, passed, reasons, base_samples, tuned_samples,
                          run_name=args.run_name)
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
