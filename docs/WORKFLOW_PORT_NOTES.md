# Porting DeepResearchEnv to a real rLLM Workflow — API discoveries + every bug hit

**Date:** 2026-08-23 (same day as `RLLM_VERL_INSTALL_NOTES.md` — read that one first, this
picks up right after the install). **Task:** HANDOFF.md step 2 — wire `AgentTrainer` for
real, which the earlier scaffold left as `agent_class`/`env_class`/`env_args` (an API that
was removed from this rLLM version). This file is the sibling to
`RLLM_VERL_INSTALL_NOTES.md`: that one is "getting the packages installed," this one is
"getting `AgentTrainer.train()` to actually run" — a **different, deeper** layer of
integration (Ray, hydra, veRL's PPO trainer, its dataset loader, its resource-pool manager),
hit for the first time in this repo. Six real, distinct bugs so far, each found from an
actual traceback, not guessed.

**Status: PASSING.** `python train_dr.py sanity` completed both sanity steps end-to-end on
the 9th attempt (7 real bugs found and fixed along the way — bugs 6–12 below), exit code 0.
Real evidence from the log, not just an exit code: `training/global_step:1`/`2`,
`timing_s/update_actor:7.9s`, `timing_s/update_weights:3.7s` (LoRA hot-swapped back into the
vLLM rollout engine after the gradient step), 4 rollouts completing per step via our own
`TerminationReason.ENV_DONE` (confirms `DeepResearchEnv`'s terminal logic fired correctly
through the full Ray/veRL/vLLM stack), a final validation pass logging
`val/deep_research_agent/pass@1`. Reward was 0.0 throughout — expected and NOT a red flag:
2 steps on a 0.5B stand-in model against a hard multi-hop task is a wiring check, not a
convergence test (`batch/max_response_length_exceeded:0.5` — half the tiny sanity model's
160-token budget got hit, another reason near-zero reward here is unsurprising and not
diagnostic of anything wrong with the pipeline).
**What this validates:** the full chain works — `DeepResearchEnv` → `DeepResearchAgent` →
`DeepResearchWorkflow` → `AgentTrainer` → Ray → veRL's PPO trainer → vLLM rollout → GRPO
actor update → weight sync → validation. HANDOFF.md step 2 (wire `AgentTrainer` for real) is
DONE.

**Update, same day, masking gate + transcript check — also DONE.** Enabled `trainer.
log_episodes=True` (a built-in rLLM episode logger, dumps every `Episode` to JSON —
`train_dr.write_verl_dataset`'s sibling override, see `verl_overrides`) and pulled real
output from a training step:
- **Masking, concretely re-verified** (not just mechanism-understood): grepped the full
  training log for `_process_trajectory`'s `"has no valid model_output, skipping"` warning —
  it's the ONE thing that would fire if a step's `model_output.prompt_ids`/`.completion_ids`
  were missing at training time. It never fired across 8 real trajectories in the step. Note:
  the episode-logger JSON itself does NOT include `model_output` (checked `episode_logger.py`
  directly — it's a deliberately human-readable subset), so don't expect to see token IDs
  there; that's normal, use the grep-the-training-log approach instead.
- **Chat-template / ReAct fidelity, checked honestly**: the untrained 0.5B sanity model does
  NOT follow the format cleanly — it often rambles through multiple pseudo `Thought:/Action:`
  pairs in one generation (imitating the few-shot demo's own shape), which means many actions
  don't parse. This is HANDLED correctly (env returns a recoverable `"error: unknown or
  unparsed action..."` observation, not a crash), and the model usually manages a well-formed
  `answer[...]` by its last turn (episodes do terminate via `TerminationReason.ENV_DONE`).
  Expected for an untrained tiny model with a 160-token sanity budget and no SFT warm-start —
  same characteristic finqa_agent already documented — not a red flag, but worth re-checking
  after the real 3B run rather than assuming it improves.
- **Found + fixed a real (cosmetic, not correctness) gap while doing this**: `DeepResearchAgent
  .update_from_env` never backfilled the most-recent `Step`'s `observation`/`reward`/`done`
  fields (only `Step.from_model_output` sets fields, and it only knows about the MODEL's own
  output, not the env's response to it) — every step showed `done=False`/empty `observation`
  even on a correctly-terminated episode. Confusing to debug from until traced to the source
  (masking/training itself was never affected — `_process_trajectory` never reads these
  fields, and the real reward/termination logic lives in `trajectory.reward` + the raised
  `TerminationEvent` in `DeepResearchWorkflow.run()`, both correct). Fixed — see the updated
  docstring on `update_from_env`.

**Not yet done:** the real Qwen2.5-3B cloud run (HANDOFF step 5) — everything above was the
0.5B sanity preset. `run_mask_check` is still an unimplemented stub in `train_dr.py`, but the
file-log-plus-grep approach above is a perfectly good standalone substitute — implementing the
stub isn't a blocker anymore.

**Update, same day, one more real blocker found + fixed — the batch-divisibility bug (bugs
11/12 below).** A near-certain crash at cloud-scale `group_size` (measured: ~31% trajectory
break rate, `42 % 16 != 0` on the first isolated test) — found and root-caused via direct
instrumentation, NOT assumed, and fixed with a verified-working patch (confirmed via the
patch's own log line actually firing inside the correct Ray worker process, not just "it
didn't crash"). Two earlier fix attempts failed for real, instructive reasons (wrong process;
then a `PYTHONPATH` gap) — see the full arc below, don't skip it if revisiting this. This was
caught and fixed BEFORE the real cloud run, which is the entire point of a sanity spike.

---

## Part 1 — the API discoveries (read this before touching `rllm_workflow.py` again)

All confirmed by reading the REAL installed `rllm@9beb6e0` source directly (grep/Read on
`.venv-deep-research/lib/python3.11/site-packages/rllm/`), not docs — this session's
established rule (`RLLM_VERL_INSTALL_NOTES.md`: three doc pages disagreed with each other
and with reality). Cite the exact file if you need to re-verify any of this against a newer
rLLM version.

### `Workflow` still wraps `BaseEnv`/`BaseAgent` — the scaffold wasn't as dead as it looked
`rllm/workflows/workflow.py`'s `Workflow` ABC takes a `rollout_engine`/`executor` and is
driven by `AgentWorkflowEngine`; concrete workflows (e.g. the built-in `MultiTurnWorkflow`)
still construct `agent_cls(**agent_args)` / `env_cls(**env_args)` where `agent_cls` is a
`rllm.agents.agent.BaseAgent` subclass and `env_cls` is a `rllm.environments.base.base_env.
BaseEnv` subclass. **`env.py`'s existing `DeepResearchEnv(BaseEnv)` needed ZERO changes** —
only a new `DeepResearchAgent(BaseAgent)` + `DeepResearchWorkflow(Workflow)` adapter layer
was needed (`rllm_workflow.py`).

`rllm.trainer.env_agent_mappings.ENV_CLASSES`/`AGENT_CLASSES` are now empty dicts (a
docstring there says "After the cleanup of the Agent+Environment+AgentExecutionEngine stack,
the env and agent maps are empty") — this is a **string-registry** lookup deprecation only
(`AGENT_CLASS_MAPPING[agent_cls] if isinstance(agent_cls, str) else agent_cls` — skipped
entirely when you pass real class objects, which we do). Don't read it as "BaseEnv/BaseAgent
don't work anymore" — they do; only the "look this class up by name" convenience path is gone.

**Didn't reuse `MultiTurnWorkflow`/`SingleTurnWorkflow` verbatim** despite them looking like
an exact fit (same `agent_cls`/`env_cls`/`agent_args`/`env_args` shape) — their `run()`
does `response = output.text; action = self.agent.update_from_model(response)`, discarding
the rest of `output` (a `ModelOutput` with the real token ids). That silently breaks masking
(next section) — the built-in workflow appears to work (runs, produces text) but the
resulting `Step`s would have no `model_output`, and `transform.py`'s mask-builder **silently
drops any step without one** (`logger.warning(...); continue` — not an error, just missing
training signal). Wrote `DeepResearchWorkflow.run()` as a small variant that passes the full
`ModelOutput` through instead. If you're building a DIFFERENT agent later and are tempted to
reach for `MultiTurnWorkflow` directly: don't, for this same reason, unless you've confirmed
it's been fixed upstream.

### Masking is now AUTOMATIC — the old design (`assert_verl_masking_matches`) is obsolete
This is the single most important discovery, worth re-stating plainly. The OLD assumed API
(what `env.assert_verl_masking_matches` was built to check) expected: rLLM/veRL hands you a
rollout's raw `(input_ids, loss_mask)`, and you verify the mask is correct by decoding
mask==0/1 spans and checking retrieved-passage text only appears in the ignored one. **That
model does not apply to the real API.** Instead (confirmed by reading
`rllm/trainer/verl/transform.py::_process_trajectory`, not guessed):

- veRL walks `trajectory.steps`. Each `Step` needs `step.model_output.prompt_ids` /
  `.completion_ids` — the REAL tokens the rollout engine tokenized-and-generated (we never
  tokenize anything ourselves; the engine does it when you call
  `rollout_engine.get_model_response(messages)`).
- If step N's `prompt_ids` is a **prefix-extension** of step N−1's
  (`prompt_ids + completion_ids`), the DELTA between them is auto-masked **0** (it's an
  "observation" — tool text / injected context inserted between turns) and each step's own
  `completion_ids` stays masked **1** (the model's own generation). This is a
  **cumulative-prefix merge** — correct by construction, no hand-verification needed, AS
  LONG AS every `Step` is built via `Step.from_model_output(model_output, messages=...,
  action=...)` (the sanctioned factory in `rllm/types.py`) using a REAL `ModelOutput`.
- `rllm.types.Trajectory.is_cumulative()` is a ready-made helper that checks this
  prefix-chain property holds — useful for a lighter-weight sanity check than the old
  design once a live rollout exists (not yet wired; `train_dr.run_mask_check`'s docstring
  has the concrete next step).

Concretely for us: retrieved-passage text only ever enters the trajectory as a `role=user`
message between two `role=assistant` turns — it can never land inside any step's
`completion_ids` — so it's masked out automatically, by the shape of the data, not by an
explicit check we wrote.

### The dataset contract: `task` comes from a parquet file's `extra_info` column
The OTHER sibling `# VERIFY` from HANDOFF §2 item 3 ("what container does rLLM want — a
list[dict]? an HF Dataset? a parquet path?"). Confirmed by reading
`rllm/engine/agent_workflow_engine.py::execute_tasks_verl`:
```python
tasks = batch.non_tensor_batch["extra_info"].tolist()
```
The `task: dict` that `Workflow.run(task=..., uid=...)` receives is **exactly** the
`extra_info` column of whatever parquet file `data.train_files`/`data.val_files` point at —
verl's own `RLHFDataset` convention (also requires a `prompt` column to exist, even though we
never use its contents — `DeepResearchEnv` builds its own ReAct opening prompt). This means:
- `train_dr.write_verl_dataset` writes one parquet row per `DRTask`, with `extra_info` =
  `train_dr._task_to_extra_info(t)` (a plain, JSON/parquet-safe dict).
- `rllm_workflow.DeepResearchWorkflow.reset()` reconstructs a `DRTask` from that dict via
  `_task_dict_to_drtask` (the exact inverse).
- **Real bug hit here:** pandas/pyarrow round-trip list-typed columns (`gold_aliases`,
  `supporting_titles`, `passages`) as **numpy arrays**, not plain Python lists. A naive
  `task.get("gold_aliases") or []` raises `ValueError: truth value of an array... is
  ambiguous` — `or` as an empty-check on something that MIGHT be a numpy array is a trap.
  Fixed with explicit `is None` checks + `list()`/`dict()` conversion (works on both plain
  values and numpy arrays). **General lesson:** any value that survives a parquet/pandas
  round-trip should be treated as "maybe a numpy array," not "definitely a Python list."
- **Real bug hit here too:** PyArrow can't infer a schema for a struct column with ZERO
  observed child fields across every row — hit for real because the bundled fixture's
  `DRTask.meta` is `{}` for all 4 rows ("Cannot write struct type 'meta' with no child
  field"). Fixed by never letting it be truly empty: `dict(t.meta) or {"_placeholder": ""}`.

---

## Part 2 — every `AgentTrainer.train()` bug hit, in order (bugs #1–6 of this phase)

Numbering continues from `RLLM_VERL_INSTALL_NOTES.md`'s install-time bugs (1–5) since these
are the same "verify against source, iterate on real errors" methodology applied one layer
deeper. Each was found from a real traceback on an actual `python train_dr.py sanity` run —
none were guessed or fixed preemptively.

### Bug 6 — `AgentTrainer(config=...)` doesn't actually resolve a `list[str]` of overrides
**Symptom:** `ValueError: Input cfg is not an OmegaConf config object (list)`, raised from
`OmegaConf.to_container(config)` inside `rllm/trainer/verl/train_agent_ppo.py`.
**Root cause:** `AgentTrainer.__init__`'s own docstring claims `config` "can be... a list of
strings (e.g. `["data.train_batch_size=8"]`)" and that these get applied "to the default
config." **Checked the actual `__init__` body — it does not do this.** It just does
`self.config = config` and stores it verbatim; `_train_verl()` passes that raw list straight
to `TaskRunner.run(config=...)`, which immediately calls `OmegaConf.to_container(config)`
expecting a real `DictConfig`. The docstring describes intended/aspirational behavior this
rLLM version doesn't implement.
**Fix:** compose the config ourselves via Hydra's own `compose()` API (the same mechanism
`@hydra.main` uses under the hood) — `train_dr.resolve_verl_config()`:
```python
from hydra import compose, initialize_config_dir
with initialize_config_dir(config_dir=<rllm/trainer/config/>, version_base=None):
    return compose(config_name="agent_ppo_trainer", overrides=overrides)
```
Pass the RESULT of this (a real `DictConfig`) as `AgentTrainer(config=...)`, not the raw
override list.

### Bug 7 — a scaffold-era hydra key doesn't exist (and was never actually needed)
**Symptom:** `hydra.errors.ConfigCompositionException: Could not override
'actor_rollout_ref.rollout.multi_turn.max_turns'. To append... use +key=value`.
**Root cause:** the original scaffold's Rosetta map guessed at
`actor_rollout_ref.rollout.multi_turn.max_turns` for controlling turn count. It isn't a real
key in the installed verl's config schema (hydra's OmegaConf "struct" mode rejects setting
brand-new keys without a `+` prefix). More importantly: **it was never needed at all** —
`DeepResearchWorkflow.run()` (rllm_workflow.py) owns the turn loop directly in Python (reads
`cfg.max_turns` via `workflow_args`), it never delegates multi-turn control to veRL's config.
**Fix:** deleted the override entirely, with a comment explaining why it was never needed
(not just that it was wrong) — so nobody re-adds it "to be safe."

### Bug 8 — two missing required micro-batch-size keys
**Symptom (first instance):** `AssertionError: [actor] Please set at least one of
'actor.ppo_micro_batch_size' or 'actor.ppo_micro_batch_size_per_gpu' if use_dynamic_bsz is
not enabled.` — from `verl.workers.config.actor.FSDPActorConfig.__post_init__`.
**Symptom (second instance, next attempt):** the SAME class of error for
`actor_rollout_ref.ref.log_prob_micro_batch_size(_per_gpu)` and
`actor_rollout_ref.rollout.log_prob_micro_batch_size(_per_gpu)` — a `check_mutually_exclusive`
helper in `verl/utils/config.py` that fires for THREE separate config sections (actor's PPO
update, the reference policy's log-prob pass, the rollout's log-prob pass).
**Root cause:** these are genuinely separate, required knobs from `ppo_mini_batch_size`
(which controls something else — the outer PPO batch, not the per-GPU forward/backward chunk
size) — the scaffold's Rosetta map only had the mini-batch key. **Lesson applied on the
SECOND instance:** rather than fix-and-rerun one key at a time again, grepped
`verl/utils/config.py` for every `check_mutually_exclusive(` call site up front and fixed all
three matching override keys in one pass — worth doing sooner next time a "you're missing
config X" error fires: check if the same validation HELPER function fires more than once.
**Fix:** added all three as `=1` (conservative default; matches RUNPOD_PLAYBOOK's
"loss_chunk_size" bound-the-forward-pass philosophy — tune up on the real cloud run once
timing is measured, not before):
```
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
```

### Bug 9 — `AgentTrainer` silently overwrites `data.train_files`/`val_files` with `None`
**Symptom:** `TypeError: 'NoneType' object is not subscriptable` deep inside
`verl.utils.fs.copy_local_path_from_hdfs`'s `assert src[-1] != "/"` — `src` was `None`.
Confusing because `resolve_verl_config()` tested standalone showed `data.train_files` set
correctly to our real parquet path.
**Root cause:** `AgentTrainer.__init__` has this (present the whole time, but INERT until
Bug 6 was fixed — `hasattr(self.config, "data")` was False when `self.config` was a bare
`list[str]`, so this branch never ran before):
```python
if train_dataset is not None and self.config is not None and hasattr(self.config, "data"):
    self.config.data.train_files = train_dataset.get_verl_data_path()
```
Once `self.config` became a real `DictConfig` (Bug 6's fix), `hasattr(..., "data")` turned
True, and this line OVERWRITES our correctly-set `data.train_files` with whatever
`train_dataset.get_verl_data_path()` returns. That method only resolves to something real for
a dataset **registered** via `rllm.data.DatasetRegistry` (needs `name`+`split`); our
`Dataset(data=[...])` has neither, so it returns `None` — silently clobbering a value we'd
already set correctly via the hydra overrides.
**Fix:** pass `train_dataset=None, val_dataset=None` to `AgentTrainer` — the guard is
`if train_dataset is not None`, so passing `None` skips the whole clobbering branch. The real
dataset wiring already happened via `data.train_files`/`data.val_files` in
`resolve_verl_config`; we don't need `AgentTrainer`'s dataset-object auto-wiring at all since
we're driving the file-based path directly.
**Lesson, same shape as the install-time bugs:** a framework feature "not applying" isn't the
same as it "not existing" — Bug 6's fix (making `self.config` a real DictConfig) silently
ACTIVATED a dormant code path (this overwrite) that had been a no-op the whole time. Fixing
one bug can un-mask a different one that was there all along but inert. Don't assume a
component you haven't touched is safe just because it hasn't caused a visible problem yet.

### Bug 10 — veRL's default config assumes an 8-GPU node
**Symptom:** `ValueError: Total available GPUs 1.0 is less than total desired GPUs 8`.
**Root cause:** `trainer.n_gpus_per_node` defaults to 8 in veRL's own `ppo_trainer.yaml`
(written for a typical multi-GPU cluster). We run single-GPU, colocated policy+rollout on
one A100 (HANDOFF.md §5 — a deliberate design choice, not a limitation to work around).
**Fix:** `trainer.n_gpus_per_node=1`, `trainer.nnodes=1`.

### Bug 11 — got all the way to the PPO actor update, then a batch-divisibility assert
**Symptom:** `AssertionError: 10 % 4 != 0` (next attempt: `9 % 2 != 0`) from
`verl/utils/tensordict_utils.py::make_iterator` — `assert tensordict.batch_size[0] %
mini_batch_size == 0`. This is INSIDE `WorkerDict.actor_rollout_update_actor` /
`train_mini_batch` — meaning rollouts had already generated, rewards/advantages had already
computed, and this was the LAST step before an actual gradient update. Two real findings
bundled into diagnosing this one:
1. **Confirmed a HANDOFF-flagged uncertainty for real**: `ppo_mini_batch_size` (set in
   PROMPT units in our Rosetta map) gets auto-multiplied by `rollout.n` (our `group_size`)
   before `make_iterator` sees it — setting `ppo_mini_batch_size=1` produced an observed
   `mini_batch_size=2` in the assertion (`group_size=2`), not 1. So the OLD scaffold
   comment "ppo_mini_batch_size unit convention (prompts vs sequences, auto-×rollout.n?)"
   — yes, confirmed, it auto-multiplies.
2. **The bigger issue**: the actual accumulated row total (10, then 9 on the next run —
   non-deterministic between otherwise-identical runs) is NOT simply
   `data.train_batch_size × group_size` (would be a deterministic 4×2=8). Root cause:
   `agent_workflow_trainer.py`'s `fit_agent()` loop has REJECTION SAMPLING on by default
   (`config.rllm.rejection_sample.enable`) — when a task's rollout group is either
   all-correct or all-incorrect, it's dropped; when a batch doesn't yet have
   `solve_partial >= train_batch_size` "mixed" groups, it ACCUMULATES across MULTIPLE
   dataloader iterations before proceeding (`if solve_partial < train_batch_size: continue`).
   At our tiny 4-question sanity scale this produces a row count that isn't a clean multiple
   of `group_size`, so no choice of `mini_batch_size` reliably divides it every run.
**Fix attempted (incomplete on its own):** `rllm.rejection_sample.enable=False`. Necessary
but NOT sufficient — the very next attempt still hit the same class of error (`11 % 2 != 0`),
proving rejection sampling wasn't the (only) source of the variable total. Real fix was
Bug 12 below; keep this override too, it's still a reasonable sanity-scale simplification.

### Bug 12 — `trainer.val_before_train`'s validation pass was leaking into the training
### batch's row count (the ACTUAL fix for the divisibility assert)
**Symptom:** same `make_iterator` assert as Bug 11, persisting across 3 different attempts
with a different non-deterministic total each time (10, 9, 11 — always ≥ the nominal
`train_batch_size × group_size = 8`, never below) even after disabling rejection sampling.
**Diagnosis:** `trainer.val_before_train` (default `True`, inherited from `ppo_trainer.yaml`)
runs a full validation pass through the SAME `AgentWorkflowEngine` BEFORE the training loop
starts. Hypothesis: validation episodes/rows were leaking into the training batch's
accounting (uid/episode bookkeeping shared across the async engine) — never fully root-caused
at the source-code level (didn't trace the exact leak path; ran out of productive leads
purely from static reading and made the call to test it empirically instead, per this
session's whole "iterate on real errors, verify empirically" discipline).
**Fix attempted:** `trainer.val_before_train=False`. **CORRECTION (2026-08-23, later the same
day) — this was NOT actually the fix, walking back an earlier overclaim.** The 9th sanity
spike attempt (with this override) passed clean, and at the time that read as confirmation.
A LATER rerun — same config, same override, only change was an unrelated cosmetic fix in
`rllm_workflow.py` (`DeepResearchAgent.update_from_env` backfilling `Step.observation`/
`done` — see `HANDOFF.md` item 7) — hit the **exact same class of failure again**
(`9 % 2 != 0`, `make_iterator`), with `val_before_train=False` still in place. That's direct
proof the total row count genuinely IS non-deterministic run-to-run for reasons independent
of validation leaking — `val_before_train=False` may have reduced how OFTEN this fires, or
its apparent fix may have been coincidental (one clean run isn't proof against a
non-deterministic bug — should have said "worked this time," not "confirmed," the first time
around). **Root cause is still genuinely open.** The leading hypothesis remains occasional
non-cumulative trajectory splits (`_process_trajectory`'s cumulative-prefix merge failing for
some steps, contributing extra rows) — NOT yet directly observed, only inferred from the
total exceeding the nominal `train_batch_size × group_size = 8` by a variable amount (9, 10,
11 seen across different runs). Keep `val_before_train=False` (still a reasonable
simplification, just not proven to be THE fix) but **do not trust this is resolved** — see
the follow-up bug entry below for the actual instrumented investigation.

**If picking this up for the cloud run:** do NOT assume this is fixed. At minimum, re-run
the instrumented diagnostic below (or something equivalent) at something closer to cloud
scale before trusting a multi-hour run not to crash on this partway through.

### Bug 11/12 — RESOLVED FOR REAL (2026-08-23, later the same day) — the complete arc
Root-caused, measured, and fixed with a verified patch. Full chain, each step evidence-based:

1. **Instrumented diagnosis** (`rllm_workflow._diagnose_cumulative_break`, mirrors
   `_process_trajectory`'s exact token-level check): the cumulative-prefix break IS real and
   directly observable — fires reliably, exact count matches the batch-total excess
   (10 fires → 42 rows vs 32 nominal = +10, precisely).
2. **Leading hypothesis (`max_new_tokens` truncation) DISPROVEN with direct evidence**: added
   `finish_reason` to the diagnostic output — every break observed had `finish_reason='stop'`
   (natural completion), never `'length'`. Truncation is not the cause.
3. **Real root cause (well-evidenced, not just plausible)**: BPE tokenizer non-prefix-
   stability when retokenizing a growing conversation from scratch each turn (the rollout
   engine's own approach — `rollout_engine.get_model_response(messages)` retokenizes the FULL
   history every call, doesn't extend incrementally). Divergence points cluster near the
   assistant-turn/role-boundary seam — exactly where a chat template's closing/opening
   markers meet raw generated text, a known hard case for greedy BPE merging. Confirmed as a
   known challenge class (web-search corroboration, not just our own reasoning): "after
   applying the chat template and tokenizing the full message list, it's hard to identify
   which tokens belong to assistant messages" is a documented multi-turn-RL tokenization
   issue, not specific to our code.
4. **Severity, MEASURED not assumed**: at sanity's `group_size=2` (divisor 2) the failure is
   roughly coin-flip odds per step — annoying, not blocking. At cloud's `group_size=16`
   (divisor 16, since `ppo_mini_batch_size` gets auto-multiplied by `rollout.n` — see bug 11's
   original note) a dedicated isolated test (2 prompts × 16 rollouts, tiny model) hit
   `42 % 16 != 0` on the very first attempt, with 10 breaks out of 32 nominal rollouts (~31%
   break rate). At that rate, hitting an exact multiple of 16 by chance is rare — **this would
   have crashed the real Qwen2.5-3B cloud run almost every step**, not occasionally. Confirmed
   this is a real blocker before it cost real money, not after.
5. **The fix**: `rllm_workflow._patch_make_iterator_for_ragged_batches` — when
   `verl.utils.tensordict_utils.make_iterator`'s requested `mini_batch_size` doesn't evenly
   divide the actual collected batch, fall back to ONE full-batch mini-batch instead of
   crashing. Confirmed SAFE, not a correctness compromise: `mini_batch_size` only controls how
   many PPO mini-batch gradient steps happen per epoch over an already-collected,
   already-correct batch (verified this divisibility check is a DELIBERATE verl SPMD design
   choice via web search, not a removable/buggy assert) — one large mini-batch is a legitimate
   PPO/GRPO configuration, doesn't touch masking/advantages/gradient correctness, and every
   data-parallel worker still sees the identical whole-batch shape.
6. **First patch attempt FAILED SILENTLY** — applied at `rllm_workflow.py`'s module-import
   time, which patches the module in whatever process imports it. Verification run still
   crashed with the exact same assertion, patch never printing — traced to: `make_iterator`
   actually executes inside `WorkerDict`/`ActorRolloutRefWorker`, a SEPARATE Ray actor spawned
   by verl's own worker-group machinery (`ray.remote(actor_rollout_cls)` in
   `TaskRunner.add_actor_rollout_worker`) that **never imports `rllm_workflow.py` at all** —
   the exact same class of process-boundary issue as the global-step investigation (HANDOFF.md
   item 5), hit again in a new spot.
7. **Second patch attempt FAILED LOUDLY** — registered the patch as a Ray
   `worker_process_setup_hook` (runs in every new worker process at startup, the correct
   mechanism for reaching a framework-owned process) via `train_dr.py`'s own `ray.init()` call
   (going first, before `AgentTrainer`'s own guarded `if not ray.is_initialized()` call).
   Crashed differently: `ActorDiedError` — `ModuleNotFoundError: No module named
   'rllm_workflow'` while Ray tried to deserialize the hook. Cloudpickle ships a module-level
   function BY REFERENCE ("import X, call X.f"), not by embedding bytecode — a fresh worker
   process's `sys.path` doesn't automatically include this lab's directory the way
   `train_dr.py`'s own top-of-file `sys.path.insert` does for that process alone.
8. **Final fix, VERIFIED working**: forward `PYTHONPATH` explicitly in the Ray
   `runtime_env["env_vars"]` so any new worker process can import our modules. Re-ran the
   isolated `group_size=16` test: `GROUP16_TEST_SUCCESS`, exit code 0, **`[batch-safety-patch]`
   fired 2 times** (direct proof the patch engaged inside the actual `WorkerDict` process this
   time — not inferred from "it didn't crash"), `[mask-diagnostic]` still fired 19 times
   (confirming the underlying tokenizer non-determinism is still happening at its real rate —
   this patch doesn't eliminate the CAUSE, it makes the framework robust TO it, which is the
   correct scope: we don't own the tokenizer or the chat template). All 3 steps of the
   isolated test completed, including final validation.

**Status: this is genuinely resolved**, not just worked-around-and-hoped. Both the
`_patch_make_iterator_for_ragged_batches` function and its `worker_process_setup_hook`
registration + `PYTHONPATH` forwarding are in `rllm_workflow.py`/`train_dr.py`'s `main()`
respectively — they run unconditionally on every real-training invocation (not gated behind
`group_size`), so this protects the sanity path too, not just the eventual cloud run.
**Lesson worth carrying forward**: this whole bug took THREE separate wrong turns before the
real fix (disproven truncation hypothesis, patch-in-wrong-process, patch-can't-deserialize) —
each one caught by DEMANDING direct proof (a diagnostic firing, a log line, an exit code)
rather than accepting "it didn't crash this time" as confirmation. The batch total's own
non-determinism is exactly what made the earlier `val_before_train` overclaim possible in the
first place (bug 11's correction, above) — a pattern worth remembering: when a fix's evidence
is "it worked once," for a bug that's inherently non-deterministic, that's not evidence yet.

---

## Part 3 — files touched this phase

- **NEW** `rllm_workflow.py` — `DeepResearchAgent(BaseAgent)` + `DeepResearchWorkflow
  (TimingTrackingMixin, Workflow)`. Reuses `env.py`/`reward.py`/`data.py` completely
  unchanged; this is purely the adapter layer.
- `train_dr.py` — `build_dataset` now returns raw `DRTask`s; new `_task_to_extra_info` /
  `write_verl_dataset` (the parquet writer); new `resolve_verl_config` (Bug 6's fix);
  `verl_overrides` gained `data.train_files`/`val_files` + the Bug 7/8/10/11/12 fixes and
  lost the Bug 7 dead key; `main()`'s `AgentTrainer` construction fixed per Bug 9;
  `run_mask_check`'s docstring updated to reflect the automatic-masking discovery (not yet
  reimplemented — see its docstring for the concrete next step).
- `HANDOFF.md` — updated: step 2 marked done, per-file table reflects the sanity spike pass.

**If you're picking this up fresh:** read Part 1 in full before changing anything in
`rllm_workflow.py` — the masking-is-automatic discovery in particular changes what "correct"
means compared to the original scaffold's design. Then check this file's status line at the
top for where the sanity spike actually landed.
