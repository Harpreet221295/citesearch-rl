"""Entry point: train the deep-research agent with rLLM (agent layer) + veRL (engine).

    python train_dr.py --dry-run cloud   # NO rllm needed: prints resolved config +
                                         # veRL hydra overrides + dataset stats. DO FIRST.
    python train_dr.py sanity            # 0.5B version/import spike (a few steps)
    python train_dr.py cloud             # Qwen2.5-3B real run on the A100
    python train_dr.py --mask-check sanity  # run ONE rollout, verify veRL's loss mask

Design: rLLM's AgentExecutionEngine runs DeepResearchEnv episodes (rollouts) and hands
trajectories to veRL, which runs the GRPO update. This file (a) loads a Config preset,
(b) builds the train/val datasets that inject each DRTask into env_args, (c) translates
our Config knobs into veRL hydra overrides (the "Rosetta map"), and (d) constructs the
AgentTrainer.

⚠️  VERIFY everything under the `rllm`/`verl` seams against your INSTALLED versions —
    import paths, AgentTrainer signature, dataset format, hydra key names, and how the
    global training step reaches the env — all move across releases. `--dry-run` needs
    none of them, so use it to sanity the config first. See HANDOFF.md step 1.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import config as dr_config
import data as dr_data
from env import DeepResearchEnv


# --------------------------------------------------------------------------- #
# 1. dataset — DRTask -> a parquet file veRL's own RLHFDataset loader reads
# --------------------------------------------------------------------------- #
# RESOLVED 2026-08-23 (was the sibling `# VERIFY` to the Workflow port — HANDOFF §2
# item 3), confirmed by reading rllm.engine.agent_workflow_engine.execute_tasks_verl:
#   tasks = batch.non_tensor_batch["extra_info"].tolist()
# i.e. the `task` dict DeepResearchWorkflow.run/reset receives is EXACTLY the
# `extra_info` column of whatever parquet file `data.train_files`/`data.val_files`
# point at (verl's own RLHFDataset convention — see rllm_workflow.py's module
# docstring for the full chain). A `prompt` column must also exist (the loader
# requires it) even though we never use it — DeepResearchEnv builds its own ReAct
# opening prompt from the reconstructed DRTask, not from verl's prompt-templating.
def build_dataset(cfg, split: str) -> list:
    """Load multi-hop tasks for `split` ("train" | "eval"). Returns raw DRTask objects
    (dry-run's sample display and run_mask_check want the live object);
    write_verl_dataset below converts them to the parquet file veRL's loader reads.

    2026-09-07: when cfg.train_split / cfg.eval_split name a splits.py partition, draw
    from THAT instead of the historical `data.load_tasks` draw. This is what keeps the
    RL pool disjoint from what SFT trained on (splits.py asserts it against real
    task_ids) and keeps `heldout_eval` untouched — the periodic-eval side is `sft_dev`,
    never the held-out set, because anything you early-stop on is no longer held out.
    None on both fields = the exact historical behaviour every pre-SFT run used."""
    name = cfg.train_split if split == "train" else cfg.eval_split
    if name is None:
        return dr_data.load_tasks(cfg, split)
    import splits as dr_splits
    if name == "heldout_eval":
        raise SystemExit("refusing to train/early-stop on heldout_eval — it is reported "
                         "against ONCE, at the end (SFT_RL_PLAN.md §1).")
    limit = None if split == "train" else (cfg.num_eval_examples or None)
    return dr_splits.get_split(name, cfg, limit=limit)


def _task_to_extra_info(t) -> dict:
    """DRTask -> the plain, JSON/parquet-safe dict that becomes `extra_info` (and
    therefore the `task` dict the Workflow receives — see rllm_workflow.py's
    `_task_dict_to_drtask`, the exact inverse of this)."""
    return {
        "task_id": t.task_id,
        "question": t.question,
        "gold_answer": t.gold_answer,
        "gold_aliases": list(t.gold_aliases),
        "supporting_titles": list(t.supporting_titles),
        "passages": [[str(title), str(text)] for title, text in t.passages],   # bug #12, 2026-09-07
        # pyarrow can't infer a schema for a struct column with ZERO observed child
        # fields across every row (hit for real: the bundled fixture's meta={} on
        # every row -> "Cannot write struct type 'meta' with no child field").
        # Never let it be truly empty.
        "meta": dict(t.meta) or {"_placeholder": ""},
    }


def write_verl_dataset(tasks: list, path: Path) -> Path:
    """Write `tasks` to a parquet file in the shape verl's RLHFDataset expects."""
    import pandas as pd
    rows = [{
        "prompt": [{"role": "user", "content": t.question}],   # required by the loader; unused otherwise
        "data_source": "deep_research_agent",
        "extra_info": _task_to_extra_info(t),
    } for t in tasks]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path)
    return path


# --------------------------------------------------------------------------- #
# 2. Rosetta map — our Config -> veRL hydra overrides (list[str])
# --------------------------------------------------------------------------- #
def verl_overrides(cfg, train_path: Path | None = None, val_path: Path | None = None) -> list[str]:
    """Translate the Config into veRL command-line (hydra) overrides.

    RESOLVED 2026-08-23: the `trainer.*`/`data.*` (non-dataset) keys below are
    CONFIRMED against the real installed verl==0.9.0 config schema
    (verl/trainer/config/ppo_trainer.yaml) — total_training_steps, project_name,
    experiment_name, logger, save_freq, test_freq, default_local_dir, resume_mode
    all match verbatim. Still genuinely `# VERIFY` (not yet checked against source):
    the `actor_rollout_ref.*` LoRA/rollout keys, and `ppo_mini_batch_size`'s unit
    convention (prompts vs sequences) flagged in HANDOFF.md/RUNPOD_PLAYBOOK.md.
    `data.train_files`/`val_files` point at the parquet files write_verl_dataset
    wrote — see that function + rllm_workflow.py for the full dataset chain."""
    rollouts_per_step = cfg.group_size * cfg.prompts_per_step * cfg.grad_accum
    ov = [
        # --- algorithm: GRPO (group-relative, no critic) ---
        f"algorithm.adv_estimator={cfg.verl_adv_estimator}",          # VERIFY
        f"algorithm.kl_ctrl.kl_coef={cfg.kl_coef}",                   # VERIFY (or actor.kl_loss_coef)
        # --- model ---
        f"actor_rollout_ref.model.path={cfg.model_name}",            # VERIFY
        f"actor_rollout_ref.model.lora_rank={cfg.lora_r}",           # VERIFY (LoRA support varies)
        f"actor_rollout_ref.model.lora_alpha={cfg.lora_alpha}",      # VERIFY
        # PROBE 2026-08-23: fused Triton kernels for the actor's fwd/bwd —
        # configurable via cfg.verl_use_liger (default False, off until
        # measured). See ONE_STEP_TUNING_VERL_RLLM.md §11. Requires the
        # `liger-kernel` pip package when True — verl lazy-imports it only
        # in that branch (transformer_impl.py), so leaving this False (the
        # default) never requires the package to be installed.
        f"actor_rollout_ref.model.use_liger={cfg.verl_use_liger}",
        # RESOLVED 2026-08-23: verl's FSDPCheckpointManager defaults to saving the
        # FULL model state_dict (frozen base + LoRA) on every trainer.save_freq hit
        # unless checkpoint.save_lora_only is explicitly True (confirmed by reading
        # checkpoint_manager.py::should_save_lora_only, default False; the key
        # itself is absent from actor.yaml's defaults so hydra needs `+` to add it,
        # not plain override). Without this, a 3B model would write a ~6GB+
        # full-state checkpoint every `checkpoint_every` steps — with 25GB free
        # disk that's a real risk over a 1000-step run. With this flag verl only
        # writes the LoRA adapter tensors (tens of MB), matching what
        # push_checkpoints()/hub.py already assume they're merging from.
        "+actor_rollout_ref.actor.checkpoint.save_lora_only=True",
        # --- optim ---
        f"actor_rollout_ref.actor.optim.lr={cfg.lr}",                # VERIFY
        # SUPERSEDED 2026-08-23 (attempt-1 cloud run, real measured problem): the
        # micro_batch_size_per_gpu=1 / mini_batch_size=1 pair below (kept in the
        # comment for the record) forced the ref-logprob, rollout-logprob, AND
        # actor-update passes to each process ~1074 rows ONE SEQUENCE AT A TIME —
        # confirmed via a killed real run: rollout gen for 1024 trajectories took
        # ~4min (fine, vLLM batched gen working as intended), then the log-prob/
        # update phase sat at 14% GPU util with no log output for 5+ min, pure
        # per-sequence Python/kernel-launch overhead dominating actual compute on
        # an 80GB A100 running a 3B LoRA model that barely uses the card at
        # batch=1. Fix: use_dynamic_bsz packs many sequences per forward pass by
        # TOKEN BUDGET instead of a fixed row count — verl's own designed
        # mechanism for exactly this. Confirmed via source
        # (verl/workers/config/actor.py::__post_init__ + verl/utils/config.py):
        # setting actor.use_dynamic_bsz=True (a) makes ppo_micro_batch_size_per_gpu
        # unnecessary (its __post_init__ assert is gated on `not use_dynamic_bsz`),
        # (b) REMOVES the real_train_batch_size % minimal_bsz divisibility assert
        # entirely (was the root class of the mini_batch_size=1 workaround), and
        # (c) the ref/rollout log-prob passes AUTO-INHERIT it — confirmed in the
        # resolved config dump: both carry
        # 'log_prob_use_dynamic_bsz': '${oc.select:actor_rollout_ref.actor.use_dynamic_bsz,false}'
        # — one flag fixes all three passes, no per-worker overrides needed.
        # ppo_max_token_len_per_gpu (actor) / log_prob_max_token_len_per_gpu (ref,
        # rollout — both interpolate from the actor's value) default to 16384,
        # which at our sequence lengths (observed 300-1500 tokens/trajectory in
        # the mask-diagnostic log) already packs ~15-30x more per forward pass
        # than batch=1 — kept at the default rather than guessed higher; retune
        # from measured timing_s/* if still not fast enough.
        "actor_rollout_ref.actor.use_dynamic_bsz=True",
        # PROBE 2026-08-23 (real measured headroom): mini_batch_size=64 got
        # update_actor 264s->135.8s and confirmed we're now closer to compute-
        # bound (135.8s is only ~2.2x old_log_prob's 61.2s for the SAME token
        # volume, forward-only — a plausible fwd+bwd+optim ratio). But
        # perf/max_memory_allocated_gb was only 23.2 GB / 25.1 GB reserved
        # against 80 GB during that same actor phase — ppo_max_token_len_per_gpu
        # (governs the dynamic_bsz forward-pass chunk size for BOTH old_log_prob
        # and update_actor's forward, ref/rollout inherit it too) was left at
        # the untested 16384 default. Raising it means fewer, bigger chunks —
        # less per-chunk Python/kernel-launch overhead for the same total
        # compute. 4x jump (65536) tested directly via a full-scale probe
        # before committing — real headroom, not guessed.
        f"actor_rollout_ref.actor.ppo_max_token_len_per_gpu={cfg.verl_ppo_max_token_len_per_gpu}",
        # SUPERSEDED 2026-08-23 (2nd real measured problem, same day): even with
        # use_dynamic_bsz packing the FORWARD/BACKWARD compute efficiently, dynamic_bsz
        # only packs sequences WITHIN one mini-batch — a mini_batch_size=1 mini-batch
        # has nothing to pack, so it still forced ~1074 SEPARATE optimizer steps
        # (one per collected row), each paying its own overhead. Measured via a
        # dedicated full-scale (cloud_preset() rollout volume, 1024 rollouts,
        # 591917 tokens) 1-step probe: update_actor took 264s vs old_log_prob's
        # 61.8s for the IDENTICAL token volume (forward-only, same dynamic_bsz
        # chunking) — the ~200s gap is per-mini-batch overhead (FSDP grad
        # sync + optimizer.step() + Python loop) repeated ~1074 times, not raw
        # compute. At 360s/step (measured, real) x 1000 steps that's ~100 hours —
        # nowhere near the 6h budget. Fix: raise ppo_mini_batch_size to 64 (~17
        # mini-batch SGD steps/training-step instead of ~1074), cutting most of
        # that repeated overhead while still doing multiple PPO mini-batch
        # updates per step (not full-batch gradient descent — kept that PPO
        # design property deliberately, only the batch-safety-patch, not this
        # value, is what makes any leftover indivisibility non-fatal).
        # PROBE 2026-08-23: 64 measured 264s->135.8s for update_actor (real, see
        # WORKFLOW_PORT_NOTES.md). Pushing further to 256 — testing directly
        # rather than assuming diminishing returns; combined with the token-
        # budget raise above in the same probe to check total headroom.
        "actor_rollout_ref.actor.ppo_mini_batch_size=256",
        f"actor_rollout_ref.actor.grad_clip={cfg.max_grad_norm}",    # VERIFY
        # 2026-09-07: KL to the reference as a LOSS term. Found reading verl 0.9.0's
        # resolved config: `algorithm.kl_ctrl.kl_coef` (set above from cfg.kl_coef) is
        # only consumed when `algorithm.use_kl_in_reward=True`, whose default is False —
        # so every historical run in TRAINING_HISTORY_LOG.md trained with NO KL term.
        # Off by default here too (keeps those runs reproducible); rl_from_sft turns it
        # on, where the reference (LoRA-disabled actor = the MERGED SFT weights) is the
        # policy we actually want to stay close to.
        f"actor_rollout_ref.actor.use_kl_loss={cfg.verl_use_kl_loss}",
        f"actor_rollout_ref.actor.kl_loss_coef={cfg.verl_kl_loss_coef}",
        f"actor_rollout_ref.actor.kl_loss_type={cfg.verl_kl_loss_type}",
        # --- rollout engine (vLLM), group size, sampling ---
        f"actor_rollout_ref.rollout.name={cfg.verl_rollout_engine}",             # VERIFY
        f"actor_rollout_ref.rollout.n={cfg.group_size}",             # G rollouts/prompt  VERIFY
        f"actor_rollout_ref.rollout.temperature={cfg.temperature}",  # VERIFY
        f"actor_rollout_ref.rollout.top_p={cfg.top_p}",              # VERIFY
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={cfg.verl_tensor_parallel}",  # VERIFY
        f"actor_rollout_ref.rollout.gpu_memory_utilization={cfg.verl_gpu_mem_util}",          # VERIFY
        # NOTE: NO actor_rollout_ref.rollout.multi_turn.max_turns override — RESOLVED
        # 2026-08-23: that key doesn't exist in the real config schema (hydra rejects
        # it, "Could not override... use +key to append"), AND it was never actually
        # needed — DeepResearchWorkflow.run() (rllm_workflow.py) owns the turn loop
        # itself in Python (reads cfg.max_turns directly via workflow_args), it
        # doesn't delegate multi-turn control to veRL's config at all.
        # --- data / batch (CONFIRMED keys; train_files/val_files RESOLVED 2026-08-23) ---
        f"data.train_batch_size={rollouts_per_step}",               # VERIFY semantics
        f"data.max_prompt_length={cfg.max_len}",                    # VERIFY
        # F18 (2026-09-07): this is the WHOLE merged multi-turn response budget, not the
        # per-turn cap — see Config.verl_max_response_length. Per-turn max_tokens is
        # passed explicitly by DeepResearchWorkflow.run.
        f"data.max_response_length={cfg.verl_max_response_length or cfg.max_new_tokens}",
        f"data.train_files={train_path}",
        f"data.val_files={val_path}",
        # --- trainer: steps, eval, checkpoint (CONFIRMED against ppo_trainer.yaml) ---
        f"trainer.total_training_steps={cfg.steps}",
        f"trainer.test_freq={cfg.eval_every or 999999}",
        f"trainer.save_freq={cfg.checkpoint_every or 999999}",
        f"trainer.default_local_dir={cfg.save_dir}/{cfg.run_name}",
        f"trainer.project_name={cfg.wandb_project}",
        f"trainer.experiment_name={cfg.run_name}",
        f"trainer.logger={_logger(cfg)}",
        f"trainer.resume_mode={'auto' if cfg.resume else 'disable'}",
        # RESOLVED 2026-08-23: veRL's default config assumes an 8-GPU node
        # (trainer.n_gpus_per_node=8) — found on the 5th sanity spike attempt
        # ("Total available GPUs 1.0 is less than total desired GPUs 8"). We run
        # single-GPU (colocated policy+rollout on one A100 — HANDOFF §5).
        "trainer.n_gpus_per_node=1",
        "trainer.nnodes=1",
        # RESOLVED 2026-08-23 (7th sanity spike attempt): with rejection sampling ON
        # (agent_workflow_trainer.py's default), partial-solve groups get accumulated
        # ACROSS multiple dataloader batches until train_batch_size is met — at our
        # tiny 4-question sanity scale this produces a non-deterministic total row
        # count (9, 10, ... — NOT a clean multiple of group_size) that can't reliably
        # satisfy make_iterator's `total % mini_batch_size == 0` assert no matter how
        # small mini_batch_size is set. Confirms the HANDOFF-flagged uncertainty for
        # real: ppo_mini_batch_size (set in PROMPT units above) gets auto-multiplied
        # by rollout.n (group_size) before this check. Disable rejection sampling for
        # the sanity spike — one dataloader batch = train_batch_size*group_size rows,
        # deterministic, divides evenly. Real cloud run can revisit this (rejection
        # sampling is a sample-efficiency optimization, not needed to prove the
        # wiring works).
        "rllm.rejection_sample.enable=False",
        # NEW 2026-08-23: rLLM's built-in episode logger (agent_workflow_trainer.py
        # checks trainer.log_episodes) dumps every completed Episode's full trajectory
        # (chat_completions, model_output token ids, reward) to JSON per step under
        # trainer.episode_log_dir/episodes/. This is the masking-gate re-verification
        # mechanism (HANDOFF.md step 4 / item 6) — pull a real episode from here and
        # check trajectory.is_cumulative() + real model_output fields, rather than
        # hand-rolling a standalone rollout engine. Cheap; worth having on by default.
        "trainer.log_episodes=True",
        f"trainer.episode_log_dir={cfg.save_dir}/{cfg.run_name}/episode_logs",
        # DIAGNOSTIC 2026-08-23 (8th attempt still hit a non-deterministic total —
        # 9, 10, 11 across otherwise-identical runs, disabling rejection sampling
        # didn't fix it): testing whether the pre-training validation pass
        # (trainer.val_before_train, on by default) is leaking stray episodes into
        # the training batch's row count via the shared AgentWorkflowEngine. Not
        # needed for a 2-step sanity spike regardless.
        "trainer.val_before_train=False",
    ]
    return ov


def _logger(cfg) -> str:
    return "[console,wandb]" if cfg.log_backend == "wandb" else "[console]"


def resolve_verl_config(overrides: list[str]):
    """Compose veRL's default hydra config (rllm/trainer/config/agent_ppo_trainer.yaml
    — the same file `@hydra.main(config_path="../config", config_name=
    "agent_ppo_trainer")` in train_agent_ppo.py loads when run via CLI) with our
    override list, producing a real resolved OmegaConf DictConfig.

    RESOLVED 2026-08-23 — a real bug hit on the first sanity spike attempt:
    `AgentTrainer(config=...)`'s own docstring claims it accepts "a list of strings
    (e.g. ['data.train_batch_size=8'])" and applies them "to the default config" —
    but its actual `__init__` (checked the source) just does `self.config = config`
    and NEVER resolves/merges anything; `_train_verl` passes that raw list straight
    to `TaskRunner.run(config=...)`, which immediately calls
    `OmegaConf.to_container(config)` expecting a real DictConfig and raises
    `ValueError: Input cfg is not an OmegaConf config object (list)` on a plain list.
    The docstring describes intended behavior this rLLM version doesn't actually
    implement — so WE compose the config ourselves via hydra's own `compose()`
    (the same mechanism `@hydra.main` uses under the hood), replicating exactly what
    running `train_agent_ppo.py` from the CLI with these overrides would produce."""
    import os

    from hydra import compose, initialize_config_dir
    import rllm.trainer.verl.train_agent_ppo as _tap

    config_dir = os.path.abspath(os.path.join(os.path.dirname(_tap.__file__), "..", "config"))
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        return compose(config_name="agent_ppo_trainer", overrides=overrides)


# --------------------------------------------------------------------------- #
# 3. mask check — run ONE rollout and verify veRL's loss mask (the silent bug)
# --------------------------------------------------------------------------- #
def run_mask_check(cfg):
    """UPDATED 2026-08-23 (rllm_workflow.py's module docstring has the full story):
    the masking mechanism is NOT hand-built anymore, so this check's ORIGINAL design
    (roll one rollout, decode where input_ids has mask==0/1, assert passage text is
    only in the ignored span — env.assert_verl_masking_matches) doesn't apply to the
    real API. veRL's rllm.trainer.verl.transform._process_trajectory builds the mask
    automatically from each Step's `model_output.prompt_ids`/`.completion_ids` via a
    cumulative-prefix merge — correct by construction as long as DeepResearchAgent
    (rllm_workflow.py) builds every Step via `Step.from_model_output(...)` with a
    REAL ModelOutput from the rollout engine, which it does.

    What's still worth checking empirically once a real rollout runs (not yet wired):
    pull one Episode's trajectory.steps out of a live run and confirm (a) each step's
    model_output.prompt_ids/completion_ids are non-empty and (b) trajectory.is_cumulative()
    is True (rllm.types.Trajectory has this exact helper) — that's the precondition
    _process_trajectory needs for the auto-mask merge to fire as ONE segment rather
    than silently degrading to multiple rows. Not yet wired to a live engine; do this
    once the sanity training spike (train_dr.py sanity) is running end-to-end."""
    raise NotImplementedError(
        "See the docstring above — the masking mechanism changed (now automatic via "
        "Step.from_model_output + Trajectory.is_cumulative(), not hand-built), so the "
        "original hand-verification design doesn't apply. Wire a live rollout's "
        "Episode out and check is_cumulative() once `train_dr.py sanity` runs.")


# --------------------------------------------------------------------------- #
# checkpoint push / best-tracking (hub.py mechanics + evaluate.py's batched scorer)
# --------------------------------------------------------------------------- #
def score_checkpoint(step: int, cfg, run_dir, llm, tok, sampling_params, tasks) -> tuple[float, "Path"]:
    """Merge ONE veRL checkpoint step to a standard adapter (hub.merge_checkpoint)
    and score it on `tasks` via the SAME shared vLLM engine as base/tuned eval
    (LoRA hot-swapped per checkpoint — no reload of the base model per candidate).
    Returns (em_score, adapter_dir)."""
    import evaluate as ev
    import hub
    merged_dir = run_dir / "_merged" / f"step{step}"
    adapter_dir = hub.merge_checkpoint(run_dir / f"global_step_{step}", merged_dir)
    lora_req = ev.make_lora_request(str(adapter_dir), name=f"step{step}")
    results, _ = ev.evaluate_set_batched(tasks, cfg, llm, tok, sampling_params, lora_request=lora_req)
    row = ev.aggregate_results(results)
    return row.get("em", 0.0), adapter_dir


def push_checkpoints(cfg) -> dict:
    """Orchestration entry point: find every local veRL checkpoint step under
    `save_dir/run_name`, score each on a small held-out slice via the batched vLLM
    path (one shared engine, LoRA hot-swapped per candidate), keep the top
    `keep_best_k` by EM + always the LATEST step, and push each to
    `push_to_hub_repo` under one timestamped run branch
    (RUNPOD_PLAYBOOK.md pattern #5b: `checkpoint/` = latest, `best/step<N>_em<S>/` =
    best — two separate artifacts, never conflate them). Safe to call repeatedly
    (e.g. periodically during a long run, or once at the end) — re-scores whatever
    steps exist each time; cheap relative to training since `periodic_eval_n` is small.
    """
    import hub
    from transformers import AutoTokenizer
    from vllm import SamplingParams
    import evaluate as ev

    if not cfg.push_to_hub_repo:
        raise SystemExit("cfg.push_to_hub_repo is not set — nothing to push to.")

    run_dir = Path(cfg.save_dir) / cfg.run_name
    steps = hub.list_checkpoint_steps(run_dir)
    if not steps:
        print(f"[push] no checkpoints found under {run_dir} (global_step_*/actor) — nothing to do.")
        return {}

    branch = hub.run_branch_name(cfg, run_dir)
    all_tasks = build_dataset(cfg, "eval")          # honours cfg.eval_split (2026-09-07)
    tasks = all_tasks[: cfg.periodic_eval_n] if cfg.periodic_eval_n else all_tasks
    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=cfg.max_new_tokens,
                                     stop=list(cfg.rollout_stop_sequences) or None)
    llm = ev.build_vllm_engine(cfg, enable_lora=True)

    print(f"[push] scoring {len(steps)} checkpoint(s) on {len(tasks)} held-out questions "
          f"(branch {branch})...")
    scored: list[tuple[int, float, Path]] = []
    try:
        for step in steps:
            em, adapter_dir = score_checkpoint(step, cfg, run_dir, llm, tok, sampling_params, tasks)
            print(f"[push]   step {step}: em={em:.3f}")
            scored.append((step, em, adapter_dir))
    finally:
        ev.close_vllm_engine(llm)

    latest_step, latest_em, latest_adapter = max(scored, key=lambda s: s[0])
    hub.push_folder_to_hub(latest_adapter, cfg.push_to_hub_repo, branch, "checkpoint",
                           private=True)

    best_k = sorted(scored, key=lambda s: s[1], reverse=True)[: max(1, cfg.keep_best_k)]
    for step, em, adapter_dir in best_k:
        hub.push_folder_to_hub(adapter_dir, cfg.push_to_hub_repo, branch,
                               f"best/step{step}_em{em:.3f}", private=True)

    report = {
        "branch": branch, "repo": cfg.push_to_hub_repo,
        "latest": {"step": latest_step, "em": latest_em},
        "best": [{"step": s, "em": e} for s, e, _ in best_k],
    }
    (run_dir / "push_report.json").write_text(json.dumps(report, indent=2))
    print(f"[push] done — {json.dumps(report, indent=2)}")
    return report


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("preset", nargs="?", default="cloud",
                     choices=["sanity", "default", "cloud",
                              "probe_beta0", "probe_beta_ramp_fast", "probe_additive",
                              "probe_cite_gated",
                              "rl_from_sft", "probe_rl_from_sft"])
    ap.add_argument("--dry-run", action="store_true",
                    help="print resolved config + veRL overrides + dataset stats; no rllm needed")
    ap.add_argument("--mask-check", action="store_true",
                    help="run one rollout and verify veRL's loss mask, then exit")
    ap.add_argument("--push-checkpoints", action="store_true",
                    help="merge+score existing local checkpoints and push latest+best "
                         "to HF Hub (hub.py), then exit — no training. Also runs "
                         "automatically at the end of training if cfg.push_checkpoints.")
    args = ap.parse_args()

    cfg = {"sanity": dr_config.Config.sanity_preset,
           "default": dr_config.Config.default,
           "cloud": dr_config.Config.cloud_preset,
           "probe_beta0": dr_config.Config.probe_beta0,
           "probe_beta_ramp_fast": dr_config.Config.probe_beta_ramp_fast,
           "probe_additive": dr_config.Config.probe_additive,
           "probe_cite_gated": dr_config.Config.probe_cite_gated,
           "rl_from_sft": dr_config.Config.rl_from_sft,
           "probe_rl_from_sft": dr_config.Config.probe_rl_from_sft}[args.preset]()

    if args.push_checkpoints:
        from dotenv import load_dotenv
        load_dotenv(_HERE / ".env")            # HF_TOKEN, WANDB_API_KEY
        push_checkpoints(cfg)
        return

    run_dir = Path(cfg.save_dir) / cfg.run_name

    # 2026-09-07: rl_from_sft points model_name at a LOCAL merged-SFT directory. Fail
    # loudly here rather than 3 minutes later inside a Ray worker's model load.
    # (A Hub id like "Qwen/Qwen2.5-3B-Instruct" also contains a slash — the first version
    # of this check keyed on `os.sep` and broke `--dry-run sanity` inside setup_pod.sh.)
    _is_local = os.path.isabs(cfg.model_name) or cfg.model_name.startswith((".", "~"))
    if _is_local and not (Path(cfg.model_name) / "config.json").exists():
        raise SystemExit(
            f"cfg.model_name={cfg.model_name!r} is a local path but has no config.json — "
            f"build it first: `python distill/merge_sft.py` (downloads the SFT adapter and "
            f"merges it into the base weights).")

    if args.dry_run:
        # dataset build may need `datasets`/network (cloud preset uses real HotpotQA);
        # don't let that hide the config/overrides we came to inspect.
        try:
            train_tasks = build_dataset(cfg, "train")
            val_tasks = build_dataset(cfg, "eval")
            train_path = write_verl_dataset(train_tasks, run_dir / "data" / "train.parquet")
            val_path = write_verl_dataset(val_tasks, run_dir / "data" / "eval.parquet")
            ds_line = f"train rows: {len(train_tasks)}  val rows: {len(val_tasks)}"
            sample = (f"sample question: {train_tasks[0].question!r}\n"
                      f"sample corpus size: {len(train_tasks[0].passages)} passages\n"
                      f"train parquet: {train_path}\nval parquet: {val_path}"
                      if train_tasks else "(no rows)")
        except Exception as e:
            train_path = val_path = None
            ds_line = f"(dataset not loadable locally: {type(e).__name__}: {e})"
            sample = "  → needs `datasets` + network on the pod (use_fixture=True works offline)"
        print(f"=== preset: {args.preset} ===")
        print(cfg.to_json())
        print(f"\n=== dataset ===\n{ds_line}\n{sample}")
        print(f"\n=== veRL hydra overrides ===")
        for o in verl_overrides(cfg, train_path, val_path):
            print("  " + o)
        print(f"\nrollouts/step = G×prompts×accum = "
              f"{cfg.group_size}×{cfg.prompts_per_step}×{cfg.grad_accum} = "
              f"{cfg.group_size*cfg.prompts_per_step*cfg.grad_accum}")
        return

    if args.mask_check:
        run_mask_check(cfg)
        return

    train_tasks = build_dataset(cfg, "train")
    val_tasks = build_dataset(cfg, "eval")
    train_path = write_verl_dataset(train_tasks, run_dir / "data" / "train.parquet")
    val_path = write_verl_dataset(val_tasks, run_dir / "data" / "eval.parquet")

    # --- real training ---
    from dotenv import load_dotenv
    load_dotenv(_HERE / ".env")            # HF_TOKEN, WANDB_API_KEY
    try:
        from rllm.trainer.agent_trainer import AgentTrainer

        from rllm_workflow import DeepResearchWorkflow
    except ImportError as e:
        raise SystemExit(
            f"rLLM/verl not importable ({e}). Install first (see setup_pod.sh / "
            f"RLLM_VERL_INSTALL_NOTES.md), or use --dry-run. ")

    resolved_config = resolve_verl_config(verl_overrides(cfg, train_path, val_path))

    # RESOLVED 2026-08-23 — the batch-divisibility bug (WORKFLOW_PORT_NOTES.md bug
    # 11/12): occasional trajectory-splitting means the collected batch size isn't
    # reliably predictable, and verl's make_iterator hard-crashes if it doesn't
    # divide the configured mini_batch_size — measured to be a near-certain crash
    # at cloud-scale group_size, not just a rare edge case. The fix
    # (rllm_workflow._patch_make_iterator_for_ragged_batches) has to run inside the
    # WorkerDict/ActorRolloutRefWorker Ray actor processes that actually call
    # make_iterator — those are spawned by verl's own worker-group machinery and
    # never import rllm_workflow.py themselves, so the patch is registered as a Ray
    # `worker_process_setup_hook` (runs in every new worker process at startup) via
    # our OWN `ray.init()` call here, BEFORE AgentTrainer gets a chance to call its
    # own (guarded `if not ray.is_initialized()`, so ours wins if we go first).
    import ray
    if not ray.is_initialized():
        from rllm.trainer.ray_init_utils import get_ray_init_settings
        from rllm.trainer.verl.ray_runtime_env import get_ppo_ray_runtime_env

        from rllm_workflow import _ray_worker_setup_hook
        runtime_env = get_ppo_ray_runtime_env()
        runtime_env["worker_process_setup_hook"] = _ray_worker_setup_hook
        # RESOLVED 2026-08-23 — cloudpickle ships the setup hook BY REFERENCE
        # ("import rllm_workflow, call _ray_worker_setup_hook"), not by embedding
        # its bytecode. A fresh worker process's sys.path does NOT automatically
        # include this lab's directory the way train_dr.py's own top-of-file
        # `sys.path.insert` does for THIS process — hit for real:
        # "ModuleNotFoundError: No module named 'rllm_workflow'" deserializing the
        # hook, which killed the whole TaskRunner actor (ActorDiedError). Forward
        # PYTHONPATH explicitly so any new worker process can import our modules.
        runtime_env.setdefault("env_vars", {})
        existing_pp = os.environ.get("PYTHONPATH", "")
        runtime_env["env_vars"]["PYTHONPATH"] = (
            str(_HERE) + (os.pathsep + existing_pp if existing_pp else "")
        )
        ray.init(runtime_env=runtime_env, **get_ray_init_settings(resolved_config))

    # A frozen judge is NOT needed for the TRAINING reward (gold-based citation-F1 —
    # see NOTES.md 'no judge model in training'); leave judge=None at train.
    #
    # RESOLVED 2026-08-23 — a real bug hit on the 4th sanity spike attempt:
    # train_dataset/val_dataset are deliberately NOT passed below. AgentTrainer.
    # __init__ does `self.config.data.train_files = train_dataset.get_verl_data_path()`
    # whenever a train_dataset IS given and self.config has a `.data` attr (true now
    # that resolve_verl_config gives it a real DictConfig) — OVERWRITING the
    # data.train_files/val_files override we just set correctly with None, because
    # Dataset.get_verl_data_path() only resolves to something real for a dataset
    # registered via DatasetRegistry (name+split), and rllm.data.Dataset(data=...)
    # here has neither. Passing None here skips that whole branch (its guard is
    # `if train_dataset is not None`) — the real dataset wiring already happened via
    # the data.train_files/val_files hydra overrides (resolve_verl_config above).
    trainer = AgentTrainer(
        workflow_class=DeepResearchWorkflow,
        workflow_args={"cfg": cfg, "judge": None},
        config=resolved_config,
        train_dataset=None,
        val_dataset=None,
        backend="verl",
    )
    # RESOLVED 2026-08-24: periodic mid-training HF push (latest checkpoint only, best-
    # effort, CPU-only merge — safe alongside the training GPU workload) so a crash/kill
    # doesn't lose everything before the end-of-run push_checkpoints() call below ever
    # runs. See rllm_workflow.register_periodic_push / hub.push_latest_snapshot's own
    # docstrings for the full reasoning.
    from rllm_workflow import register_periodic_push
    register_periodic_push(cfg, run_dir)
    trainer.train()

    if cfg.push_checkpoints:
        # ISSUE #16 (2026-09-08, hit at the end of the first full RL run): calling
        # push_checkpoints() HERE builds a vLLM engine inside the training process, whose
        # CUDA context Ray/torch already initialised -> vLLM V1's forked EngineCore dies
        # with "Cannot re-initialize CUDA in forked subprocess". Every checkpoint was safe
        # (periodic push had already mirrored them), but best-selection never ran. Run
        # the scoring in a FRESH process instead — same code path as the manual CLI.
        import subprocess
        print("[push] scoring + pushing checkpoints in a fresh process (see issue #16)...")
        subprocess.run([sys.executable, str(_HERE / "train_dr.py"), "--push-checkpoints",
                        args.preset], check=False)


if __name__ == "__main__":
    main()
