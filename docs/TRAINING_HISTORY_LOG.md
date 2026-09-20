# Training history log — deep_research_agent

A chronological log of every real cloud training attempt: what config was run, what was
observed, what got changed and why, and the outcome. **This is the narrative trail** —
W&B has the raw metrics, `NOTES.md` has settled design decisions, `HANDOFF.md` has
current status/gotchas, `ONE_STEP_TUNING_VERL_RLLM.md` has the throughput-tuning story
specifically. This doc is "what happened, in order," meant to survive across sessions —
append to the bottom as new attempts happen, don't rewrite history above.

**Sibling doc (2026-08-26):** `rft_diagnosis/FORMAT_INVESTIGATION_LOG.md` is the same
"what happened, in order" trail for work that is NOT a training run — the session that
audited how the agent talks to the model and found that two sessions of results had been
measuring our own harness. Nothing was trained that day, so it has no entry below, but
it invalidated conclusions that several entries here fed into.

## Quick reference: every W&B run

| Run | W&B URL | lr | reward_mode | Outcome |
|---|---|---|---|---|
| Attempt 1 | [njpxc1cn](https://wandb.ai/happy22harpreet/deep_research_agent/runs/njpxc1cn) | 1e-5 | gated | Killed step 31 — flat reward, dead-groups climbing |
| Attempt 2 | [lchavjqh](https://wandb.ai/happy22harpreet/deep_research_agent/runs/lchavjqh) | 1e-4 | gated | Killed step 28 — real collapse |
| Attempt 3 | [vgzv9byy](https://wandb.ai/happy22harpreet/deep_research_agent/runs/vgzv9byy) | 5e-5 | gated | Killed step 36 — healthy, but citation-avoidance discovered |
| Attempt 4 | [qyzrepqk](https://wandb.ai/happy22harpreet/deep_research_agent/runs/qyzrepqk) | 5e-5 | gated + beta ramp (20→120) | Killed step 38 — healthy, beta ramp too slow to show effect |
| Probe 1 | [k5zymqoc](https://wandb.ai/happy22harpreet/deep_research_agent/runs/k5zymqoc) | 5e-5 | gated, beta=0.0 static | Killed step 10 — reward-signal crushed to ~0 for all episodes |
| Probe 2 | [5e0jmvje](https://wandb.ai/happy22harpreet/deep_research_agent/runs/5e0jmvje) | 5e-5 | gated + beta ramp (1→30, fast) | Completed — healthy, `pass@1=0.309`, groundedness never moved |
| Probe 3 | [i4ve6cf9](https://wandb.ai/happy22harpreet/deep_research_agent/runs/i4ve6cf9) | 5e-5 | additive | Completed — healthy, `pass@1=0.262`, citation attempts collapsed 43%→0.2% |
| Probe 4 | [wdd1i4na](https://wandb.ai/happy22harpreet/deep_research_agent/runs/wdd1i4na) | 5e-5 | cite_gated (hard zero, no ramp) | Completed — `pass@1=0.258`, attempt-rate fixed (0.45→0.95+), but citation stuck at ~1/rollout, groundedness regressed to 0 |

| **SFT (LoRA)** — 2026-08-26 | no W&B (local, ~7 min) | 1e-4 | n/a — supervised, not RL | **Completed. Gate PASSED. correct-and-cited 0.3% -> 30.7% held-out.** Shipped: `harpreet22happy/deep-research-agent-sft` |
| **SFT `sft_correct_only`** — 2026-09-07 | no W&B (local, 14 min) | 1e-4 | n/a — supervised | 1,209 correct+perfectly-cited teacher episodes, 2 epochs. sft_dev n=128: correct 0.625, read-before-cite 0.859, cite_f1 0.770, **correct-and-cited 48.4%**, capped 13.3%. **The RL starting point.** Hub: `…-sft-correct-only` |
| **SFT `sft_imitate_all`** — 2026-09-07 | no W&B (local, 29 min) | 1e-4 | n/a — supervised | 2,682 process-clean teacher episodes incl. 558 wrong answers, all turns graded. sft_dev: correct 0.656, cite_f1 0.712, correct-and-cited 33.6%, capped 10.2%. Commits more, cites less (F19). Hub: `…-sft-imitate-all` |
| Probe (RL from SFT, broken) | [n2em5ok7](https://wandb.ai/happy22harpreet/deep_research_agent/runs/n2em5ok7) | 5e-5 | cite_gated + KL 0.01 | Killed step 4 — **found F18**: `max_response_length=256` truncated 87% of trajectories and zeroed their reward (`critic/rewards/mean` 0.05 vs real 0.55) |
| Probe (RL from SFT, F18 fixed) | [mchhqpc6](https://wandb.ai/happy22harpreet/deep_research_agent/runs/mchhqpc6) | 5e-5 | cite_gated + KL 0.01 | Killed step 4 — wiring confirmed: clip_ratio 0, rewards/mean 0.55–0.62, 184–191 s/step at 512 traj/step |
| **RL from SFT (`rl_from_sft_correct_only`)** — 2026-09-07/08 | [t2n91x71](https://wandb.ai/happy22harpreet/deep_research_agent/runs/t2n91x71) | 5e-5 | cite_gated + KL-to-SFT 0.01, trajectory budget 4096 | **COMPLETED 99 steps** (~5.2 h, 190 s/step). Training reward 0.58 → 0.70, capped 0.074 → 0.034, reads flat. Best ckpt step 75: dev c&c 0.484 → **0.570**; **held-out 0.380 → 0.393 (inside noise)**, read-before-cite 0.860 → 0.895. No step-100 ckpt (#17). `RL_FROM_SFT_LOG.md` §4.20–4.35 |

> **2026-09-07 — read before interpreting ANY row above the SFT rows:** finding **F18**
> (`RL_FROM_SFT_LOG.md` §4.18). Every GRPO run from Attempt 1 to Probe 4 ran with
> `data.max_response_length=256` applied to the MERGED multi-turn response: trajectories
> longer than 256 tokens were truncated to their first turn and given reward 0. The
> "2-turn collapse" those runs share is the policy learning the only shape that was ever
> paid. Their reward-design conclusions (beta ramps, additive, cite_gated) were drawn on a
> signal that was structurally biased toward short episodes, on top of F1.

*(The SFT row is a different KIND of run from the eight above — supervised fine-tuning on
teacher trajectories, no reward, no rollouts during training. It is in this table because
the GRPO run that follows starts FROM it, so anyone reading the RL numbers needs to know
what the policy was initialised with. Full narrative:
[`distill/SFT_HISTORY_LOG.md`](../distill/SFT_HISTORY_LOG.md).)*

*(Probe 3's first launch attempt, [qv6kreru](https://wandb.ai/happy22harpreet/deep_research_agent/runs/qv6kreru),
was killed at step 0 before completing a step — restarted as i4ve6cf9 above to pick up
the `pct_rollouts_with_citation` metric. Probe 4's log line above is updated as it runs.)*

---

**Entry template** (copy for each new attempt):
```
## Attempt N — <date> — <one-line summary>
- **Config changed from previous:** <field>=<old> -> <new> (or "none, first real attempt")
- **W&B run:** <url> (renamed to <final-name> if aborted)
- **Ran:** steps 1-<N> before <killed manually / completed / crashed>
- **Observed:** <the real numbers that mattered>
- **Decision:** <what changed and why, or "kept running">
- **Commit:** <short sha + link>
```

---

## Attempt 1 — 2026-08-24 — `lr=1e-5` (base `Config` default), killed at step 31/252
- **Config:** first real attempt after all throughput tuning — `grad_accum=2`,
  `use_liger=True`, `mini_batch=256`, `token_len=65536`, `lambda_eff` ramp 150→230,
  `lr=1e-5` (never explicitly set — the base `Config` dataclass default, inherited
  by `cloud_preset()` without override).
- **W&B run:** [njpxc1cn](https://wandb.ai/happy22harpreet/deep_research_agent/runs/njpxc1cn)
  — renamed to `deep_research_agent_cloud_lr1e-5_ABORTED_step31`.
- **Ran:** steps 1-31, ~40 minutes of the 6h budget.
- **Observed:** reward growth flattening early — 10-step-window averages
  `0.048 → 0.092 → 0.099` (decile 2→3 gain much smaller than 1→2). Dead-group fraction
  (`batch/solve_none + batch/solve_all`) climbing steadily: `28% → 59%` over steps 5-21.
  Entropy plateaued around 0.10-0.15 without collapsing to zero. Step timing matched
  the tuned steady-state (~85-90s/step) — the throughput work held up on the real run.
- **Decision:** killed. This shape (early gains, then flattening) matches
  `assignments/finqa_agent/last_runpod_session.md`'s own documented `lr=1e-5` failure on
  a sibling agentic-RL lab (flat reward, FAILED its eval gate at 88/1000 steps) — that
  session's own A/B probe found `lr=1e-4` gave real movement where `1e-5` was flat.
  Tried `lr=1e-4` next, following that precedent.
- **Commit:** [011f206](https://github.com/Harpreet221295/Agentic-RL-Alignment-Path/commit/011f206)

## Attempt 2 — 2026-08-24 — `lr=1e-4`, killed at step 28/252 — real collapse
- **Config changed from previous:** `lr` 1e-5 → 1e-4 (10x jump, matching finqa_agent's
  own precedent). Everything else unchanged.
- **W&B run:** [lchavjqh](https://wandb.ai/happy22harpreet/deep_research_agent/runs/lchavjqh)
  — renamed to `deep_research_agent_cloud_lr1e-4_ABORTED_step28_collapse`.
- **Ran:** steps 1-28, ~50 minutes.
- **Observed:** NOT just underperformance — a real collapse. `batch/solve_none` hit
  `96-100%` for 4 straight steps (24-27) — nearly every group all-wrong.
  `response_length/clip_ratio` steady at `96-98%` — the model stopped terminating
  cleanly, hitting `max_new_tokens=256` on nearly every rollout instead of emitting a
  clean `answer` action. The step-25 periodic eval showed `val/pass@1 = 0.0` (vs `0.25`
  for the `lr=1e-5` run at the same step). **Notably, `grad_norm` and `ppo_kl` looked
  stable throughout** (no spike, no obvious blow-up signal) — the collapse showed up in
  generation *behavior* (response length, termination), not in the gradient-level
  metrics that would normally flag instability. Real lesson: "no gradient spike" isn't
  the same as "training is healthy" — watch behavioral metrics, not just optimizer ones.
- **Decision:** killed. `1e-4` was too aggressive a jump for this model/task/single-epoch-
  GRPO combination. Tried `5e-5` next — halfway between the two failed extremes.
  Also, prompted directly by this diagnosis: added `reward_components/*` (outcome,
  groundedness, `cite_f1`, tolls) and `batch/dead_groups_pct` logging to W&B
  (`rllm_workflow.py::_patch_tracking_log_for_extra_metrics`) — the existing metrics
  eventually caught this (clip_ratio, solve_none), but a faster/more direct read on
  reward composition was a real gap surfaced by this attempt.
- **Commit:** [585471b](https://github.com/Harpreet221295/Agentic-RL-Alignment-Path/commit/585471b)
  (lr change), [39a6379](https://github.com/Harpreet221295/Agentic-RL-Alignment-Path/commit/39a6379)
  (the new logging, committed just before this restart)

## Attempt 3 — 2026-08-24 — `lr=5e-5`, in progress
- **Config changed from previous:** `lr` 1e-4 → 5e-5 (halfway between the two failed
  extremes in log-space, closer to the failed `1e-4`). Everything else unchanged.
- **W&B run:** [vgzv9byy](https://wandb.ai/happy22harpreet/deep_research_agent/runs/vgzv9byy)
- **What's being watched differently this time:** `response_length/clip_ratio` and
  `batch/solve_none`/`dead_groups_pct` from the START, not just entropy/reward — those
  two caught Attempt 2's real collapse where grad_norm/entropy alone didn't. Also now
  have `reward_components/*` available for the first time (didn't exist during Attempts
  1-2) — can directly check whether outcome or groundedness is the one lagging, instead
  of only seeing the combined gated scalar.
- **Observed so far (steps 1-29, real numbers pulled via `wandb.Api().history()`):**
  - **No Attempt-2-style collapse.** `response_length/clip_ratio` stayed in a `0.16-0.68`
    band throughout (vs. Attempt 2's `96-98%` for 4 straight steps) — the model is
    terminating cleanly, not just hitting `max_new_tokens`. `outcome` (the EM component)
    stayed in a healthy `0.24-0.46` band, no downward trend. `hit_rate` (retrieval)
    stayed `0.63-0.80` — retrieval itself is working fine and not degrading.
    Step-25 periodic eval: `val/pass@1=0.234` — real, non-zero, in line with training-time
    `outcome` (vs. Attempt 2's `val/pass@1=0.0` at the same step).
  - **Citation avoidance — a real, growing pattern, not noise.** `reward_components/
    groundedness`, `cite_precision`, `cite_recall` all start near-zero at step 1
    (`groundedness=0.004`) and by step 7 onward are essentially flat at `~0.000-0.005`
    for the rest of the run (steps 13, 16-19, 21, 26, 28, 29 are exact `0.0`).
    Crucially, `cite_fabricated` is ALSO dropping over the same window
    (`0.346`@step1 → `~0.00-0.03` by step 10+) — so this is not "the model tries to cite
    and gets it wrong," it's "the model has learned to stop citing almost entirely."
    Matches the `beta=0.5` gate-floor mechanism discussed in chat: a correct-but-uncited
    answer still banks `outcome × 0.5` — a safe, zero-variance payoff GRPO's group-relative
    advantage can favor over the noisier "try to cite, risk the fabrication toll" path.
  - **New finding, not previously flagged in chat: `batch/dead_groups_pct` trending UP
    over the run**, `0.25-0.47` in steps 1-9 → mostly `0.56-0.84` from step 14 onward
    (peak `0.844` at step 21). Worth watching — rising dead-group fraction means a
    shrinking fraction of groups are contributing real GRPO advantage signal, independent
    of the citation question. Not yet at Attempt-2-collapse levels (that was 96-100%), but
    the trend direction is the wrong one and merits a longer look.
  - **Real bug found and fixed in the logging itself, not the training:** `reward_components/*`
    keys are absent from wandb's stored history at step 25 specifically (`test_freq=25` and
    `checkpoint_every=25` coincide there) — confirmed via direct `wandb.Api()` query, not a
    console-print artifact. Root cause: the `is_val_call` skip-gate in
    `rllm_workflow.py::_patch_tracking_log_for_extra_metrics` triggered on ANY `val/`-prefixed
    key, which also matched the single mixed train+val `Tracking.log()` call verl makes on a
    coincident step — silently skipping the injection (though not losing the data outright;
    the accumulator wasn't cleared, so those episodes just got folded into step 26's average
    instead of reported on their own). Fixed same day by switching to a positive check
    (`actor/`/`batch/`-prefixed key present) instead of the negative one. Takes effect on the
    next process restart, not on this already-running process.
- **Decision:** kept running — no collapse signal, worth letting it continue past step 29 to
  see whether citation-avoidance and rising dead-groups are transient or entrenched. Proposed
  (not yet implemented — pending Harpreet's decision): a ramped `beta` in the gated reward
  (mirroring `lambda_eff_at`'s pattern — start `~0.5` for early-training bootstrapping, decay
  toward `~0.15-0.2` as training matures) to directly shrink the safe-uncited-answer payoff
  that appears to be driving the avoidance.
- **Commit:** [def3f4a](https://github.com/Harpreet221295/Agentic-RL-Alignment-Path/commit/def3f4a)
  (logging-bug fix + backfilled real numbers)

## Attempt 4 — 2026-08-24 — `lr=5e-5` + ramped `beta`, in progress
- **Config changed from previous:** `lr` unchanged at 5e-5 (Attempt 3 showed no
  collapse — not what's being re-tested here). New: `beta_ramp_start=20`,
  `beta_ramp_end=120` (beta itself and `beta_min=0.15` left at Config defaults) —
  `Config.beta_at(step)` now ramps `beta` DOWN 0.5→0.15 over steps 20-120, instead of
  the static `beta=0.5` used throughout Attempts 1-3. See `config.py`'s `beta_at`
  docstring and Attempt 3's entry above for the full citation-avoidance evidence
  motivating this.
- **Why kill Attempt 3 instead of letting it keep running:** the code change only takes
  effect on a fresh process start (module-level config, not hot-patchable) — same
  reasoning as the lambda_eff/logging fixes earlier. Attempt 3 was killed at step 36,
  still healthy on outcome/hit_rate/clip_ratio grounds, no data lost (latest checkpoint
  `global_step_25` already on HF).
- **Attempt 3 final disposition:** W&B run [vgzv9byy](https://wandb.ai/happy22harpreet/deep_research_agent/runs/vgzv9byy)
  renamed to `deep_research_agent_cloud_lr5e-5_ABORTED_step36_citation_avoidance`.
- **W&B run:** [qyzrepqk](https://wandb.ai/happy22harpreet/deep_research_agent/runs/qyzrepqk)
- **What's being watched:** same as Attempt 3 (`clip_ratio`, `dead_groups_pct`,
  `val/pass@1`) PLUS now specifically `reward_components/groundedness` /
  `cite_precision` / `cite_recall` / `cite_fabricated` trending UP instead of flat-zero,
  and `reward_components/beta` itself to confirm the ramp is actually engaging
  (same "verify the wiring with real numbers, don't assume" discipline as lambda_eff).
- **Observed (steps 1-37, killed to make room for the reward-design probes below):**
  `beta` confirmed moving exactly on formula (0.5 at step 20 boundary -> 0.44 by step 37,
  matching `beta_at()` step-by-step). Despite that real movement, `groundedness`/
  `cite_precision`/`cite_recall` stayed at EXACTLY 0 from step 5 through step 37 — the
  slow 100-step-wide ramp hadn't moved `beta` far enough yet to change the incentive
  (0.44 vs. the 0.15 floor is still most of the way to the original 0.5). `outcome`
  trended genuinely upward over the run (many steps >0.4, several >0.5, vs. mostly
  0.25-0.42 in Attempt 3) — `val/pass@1=0.305` at step 25 vs. Attempt 3's 0.234 at the
  same step, a real improvement, though confounded with normal run-to-run noise given
  only one seed each. `dead_groups_pct` stayed noisy in the 0.44-0.88 band, no clear
  monotonic trend either way. `clip_ratio` healthy throughout (0.14-0.62), no collapse.
- **Decision:** killed at step 38 (not because it was unhealthy — it wasn't) to run three
  short 25-step reward-design comparison probes instead (below) — Harpreet wants to
  isolate which reward SHAPE actually fixes citation-avoidance before committing another
  100+ step run to one guess. Also, separately: Harpreet caught a real reasoning gap in
  `beta_ramp_start=20` — the value was picked by loose analogy to `lambda_eff_ramp_start`
  (\"give some early runway\"), but that analogy doesn't hold for citation-avoidance
  (`groundedness≈0` from step 1, no exploration-runway case to make the way there is for
  tool-use). The probes below test a `ramp_start=0` schedule directly.
- **Final W&B disposition:** run [qyzrepqk](https://wandb.ai/happy22harpreet/deep_research_agent/runs/qyzrepqk)
  renamed to `deep_research_agent_cloud_lr5e-5_betaramp_ABORTED_step38_for_probes`.

---

## Reward-design probes — 2026-08-24 — 25 steps each, same lr=5e-5, fresh from base model

Motivation: rather than bet 100+ steps on one reward-design guess, run three short,
independent, apples-to-apples 25-step probes and compare `reward_components/*`,
`batch/dead_groups_pct`, `response_length/clip_ratio` trends directly. Each starts FRESH
from the base model (`resume=False`, distinct `run_name` per probe — no chaining off each
other's checkpoints), `lr=5e-5` held fixed (the only non-collapsing value found so far —
isolates the reward-design variable). No periodic eval/checkpoint push (25 steps is too
short for either to matter). Config: `Config.probe_beta0()` / `probe_beta_ramp_fast()` /
`probe_additive()` in `config.py`, CLI via `train_dr.py probe_beta0` etc. See
`config.py`'s "reward-design comparison probes" section for the exact knobs.

### Probe 1 — `beta=0.0` static (most aggressive gate: uncited answer = zero credit)
- **W&B run:** [k5zymqoc](https://wandb.ai/happy22harpreet/deep_research_agent/runs/k5zymqoc)
  — renamed `deep_research_agent_probe_beta0_KILLED_step10_reward_crushed`.
- **Ran:** steps 1-9 (killed after step 10 started; mechanism was already conclusive,
  more steps unlikely to change the qualitative finding).
- **Observed:** Harpreet's hypothesis, confirmed with real numbers: with `beta=0`,
  `base = outcome * groundedness`. Since the model has essentially no citation skill
  (`groundedness≈0.007-0.041` throughout), `base` stayed pinned at `0.001-0.023` for
  ALL 512 episodes/step — regardless of `outcome` being a healthy `0.31-0.40` (correct
  ~1/3 of the time). The correctness signal gets multiplied away to near-zero before
  GRPO ever sees it — the model can't tell from its reward whether it answered right,
  only whether it accidentally cited something (which it almost never does). Downstream
  symptoms: `clip_ratio` climbed to `0.81-0.93` (vs. Attempt 4's healthy 0.14-0.62 —
  genuinely in Attempt-2-collapse territory), entropy elevated and not settling
  (0.30->0.43, noisy plateau instead of convergence — consistent with the policy
  chasing a near-flat, noise-dominated reward surface rather than learning anything
  coherent), `dead_groups_pct` stayed high (0.66-0.97). Note: `hit_rate` (retrieval)
  actually kept improving (0.77->0.86) — that skill is unaffected by the beta gate,
  confirming the damage is specific to the reward's correctness signal, not a general
  model breakdown.
- **Decision:** `beta=0.0` static is too aggressive as a starting point — it erases the
  correctness signal before the model has any citation skill to unlock it with. Confirms
  the opposite-direction risk flagged before launching. Moving to Probe 2 (ramped, starts
  at 0.5 so this crush doesn't happen from step 1).

### Probe 2 — `beta` ramped 0.5→0.0 over steps 1→30 (fast/early, vs. Attempt 4's slow 20→120)
- **W&B run:** [5e0jmvje](https://wandb.ai/happy22harpreet/deep_research_agent/runs/5e0jmvje)
  — renamed `deep_research_agent_probe_beta_ramp_fast_COMPLETE_step25_no_citation_recovery`.
- **Ran:** all 25 steps to completion, clean finish (final val eval ran automatically at
  step 25 despite `eval_every=0` — verl runs one regardless at the true end of training).
- **Observed:** no collapse, unlike Probe 1 — `clip_ratio` stayed healthy the whole run
  (0.13-0.41, no Probe-1-style spike to 0.8+), `outcome` trended genuinely upward
  (several steps >0.4, two >0.5 by steps 20-21), `hit_rate` stayed healthy (0.51-0.78),
  final `val/pass@1=0.309` — essentially matching Attempt 4's 0.305 at the same step and
  clearly better than Attempt 3's 0.234. **But citation-avoidance did NOT resolve**:
  `groundedness` stayed at ~0 for the entire run (max single-step value seen: 0.0013)
  even as `beta` dropped from 0.483 all the way to 0.10 by step 24 — nearly its full
  range. `cite_fabricated` dropped from 0.28 (step 1) to ~0 by step 11 onward and stayed
  there — confirming (via the new `n_citations` metric's predecessor signal) this is
  "stopped attempting to cite," not "attempts citation, gets it wrong." The avoidance
  locked in within the first ~10 steps, while `beta` was still >=0.33, and never
  recovered even as the incentive kept shrinking toward zero.
- **Decision:** ramping `beta` faster/earlier than Attempt 4 does NOT fix
  citation-avoidance — the avoidance pattern isn't sensitive to ramp speed or how low
  `beta` eventually goes. This is a real, useful negative result: it shifts the leading
  hypothesis from "the incentive isn't strong enough yet" toward "the model doesn't have
  the citation skill to unlock regardless of incentive strength" — the kind of finding
  that would point toward `TENTATIVE_FUTURE_EXPERIMENTS.md`'s cold-start SFT/RFT path if
  Probe 3 (additive, no gate at all) also fails to move `groundedness`. Also prompted
  adding a direct `reward_components/n_citations` metric (raw citation-attempt count,
  correct or not) so future probes don't have to infer "no attempt" indirectly via
  `cite_fabricated` — see `reward.py`/`rllm_workflow.py` commit for Probe 3 onward.

### Probe 3 — `reward_mode="additive"` (no gate/floor — outcome and groundedness summed independently)
- **W&B run:** [qv6kreru](https://wandb.ai/happy22harpreet/deep_research_agent/runs/qv6kreru)
  — killed at step 0 (before step 1 completed) and restarted to pick up a new metric,
  renamed `deep_research_agent_probe_additive_KILLED_step0_restarted_for_new_metric`.
  Real run continues under a new W&B URL:
  [i4ve6cf9](https://wandb.ai/happy22harpreet/deep_research_agent/runs/i4ve6cf9).
- **New metric added mid-launch:** `reward_components/pct_rollouts_with_citation` —
  fraction of a step's episodes with `n_citations>0`, distinct from the plain
  `n_citations` AVERAGE (which conflates "half cite twice, half cite zero" with
  "everyone cites once"). Harpreet's ask, directly answers "how many rollouts even
  tried to cite." See `rllm_workflow.py`'s `_patch_tracking_log_for_extra_metrics`.
- **Note:** `additive` mode gives FULL outcome credit even at zero groundedness
  (`base = w_outcome*outcome + w_ground*groundedness`, citing is a pure bonus on top,
  not something that gates/unlocks the outcome credit) — a genuinely different
  incentive shape from the gated formula's multiplicative floor, not just a `beta=0`
  equivalent. Also the first probe to log `reward_components/n_citations` directly
  (raw citation-attempt count) instead of inferring "no attempt" via `cite_fabricated`.
- **Harpreet's pre-registered hypothesis (before seeing any Probe 3 data):** this will
  NOT move `groundedness` either — if the model genuinely lacks the citation SKILL (not
  just facing the wrong incentive shape), a bonus it doesn't know how to earn is just as
  unclaimed as a floor it can't unlock. Consistent with Probes 1-2 both failing to move
  `groundedness` despite very different incentive shapes.
- **Ran:** killed/restarted once at step 0 (see above), then all 25 steps to completion
  clean (final val eval fired automatically at step 25 despite `eval_every=0`).
- **Observed — hypothesis confirmed, decisively:** `pct_rollouts_with_citation` (the new
  metric) collapses from `0.434` at step 1 to `0.002-0.021` by steps 18-24 — citation
  attempts essentially eliminated by the back half of the run, under a design that NEVER
  penalizes skipping citation and gives a pure bonus for getting it right. `groundedness`
  sits at exactly 0 for nearly the entire run (only 5 of 25 steps show any nonzero value,
  all <0.016). Meanwhile `outcome` stayed healthy and trended UP late (0.40-0.48 in steps
  18-21, several of the best values in the whole run), `hit_rate` stayed healthy
  (0.65-0.78), `clip_ratio` healthy throughout (0.15-0.50, actually improving/dropping in
  later steps) — no collapse anywhere, this is a clean, healthy run that simply never
  learns to cite. Final `val/pass@1=0.262` (lower than Probe 2's 0.309 and Attempt 4's
  0.305, though only one seed each — not a strong claim either way on outcome quality).
- **Decision:** confirms Harpreet's hypothesis exactly. Three structurally different
  incentive shapes (`beta=0` static, `beta` ramped fast to 0, additive/no-gate) ALL
  converged on the same citation-abandonment failure — strong evidence this is a genuine
  missing-skill problem, not an incentive-shape problem. Directly motivated adding
  Option D (a direct zero-citation penalty) to `TENTATIVE_FUTURE_EXPERIMENTS.md` as the
  next real candidate, and raises the odds that even a well-tuned reward may not be
  enough — the cold-start SFT/RFT path in that same doc may be genuinely warranted,
  pending Harpreet's call.

---

## Cross-probe synthesis — 2026-08-24, all four reward-design probes complete

| Probe | `groundedness` (final steps) | `pct_rollouts_with_citation` trend | `outcome`/`clip_ratio` | Collapse? |
|---|---|---|---|---|
| 1: `beta=0.0` static | ~0 (base reward crushed) | not measured (metric added after) | healthy `outcome`, `clip_ratio` degraded to 0.81-0.93 | Yes — reward-signal collapse |
| 2: `beta` ramped fast 0.5→0.0 | ~0 throughout | not measured (metric added after) | both healthy | No |
| 3: `reward_mode="additive"` | ~0 throughout | 0.434 → 0.002-0.021 (measured directly) | both healthy | No |
| 4: `cite_gated` (hard zero) | ~0.02-0.03 briefly (steps 8-13), then back to exactly 0 (steps 17-24) | **0.45 → 0.91-0.99, sustained** — the only probe where this DIDN'T collapse | both healthy, `pass@1=0.258` | No |

**Probes 1-3: citation-avoidance is NOT sensitive to the reward's incentive shape** — not
the gate strength, not the ramp speed, not whether a gate exists at all. Probe 1
additionally revealed a DIFFERENT failure mode (an overly-strong static floor can crush
the correctness signal itself, unrelated to citation) — a separate lesson, not the main
finding.

**Probe 4 changed the picture, partially.** Making non-attempts strictly worse than bad
attempts (instead of equally safe) DID fix the attempt-rate — the only one of four
designs where citation didn't collapse toward zero. But it revealed a SECOND, smaller
exploit underneath the first: `n_citations` stayed flat at ~1.0-1.3 the whole run — the
model learned to paste roughly one citation marker per answer, just enough to clear the
gate, not to cite thoroughly or correctly for genuinely multi-hop questions.
`groundedness` briefly moved (the only time across all four probes) before regressing
back to exactly 0. Net picture: **attempt-rate is fixable via reward design;
attempt-QUALITY has resisted every reward shape tried so far** — closing one exploit
surfaced a smaller one right next to it, rather than teaching real citation discipline.

This is real evidence for treating citation-quality (not just citation-presence) as a
missing SKILL rather than a missing INCENTIVE, and reinforces a live discussion during
Probe 4: GRPO's within-group advantage normalization makes large, easy-to-learn
contrasts (attempt vs. not) fast to pick up, but dilutes small contrasts (correct vs.
incorrect citation) when outcome-variance dominates a group — a structural limit on what
reward-shaping alone can fix, not just a matter of finding the right formula. Two
distinct reward-exploits found across four iterations this session is itself a pattern
worth noting: pure GRPO reward-design is proving to be real whack-a-mole. See
`TENTATIVE_FUTURE_EXPERIMENTS.md`'s Options A-C (cold-start SFT / RFT / preference
stage) — RFT specifically flagged as the next candidate worth trying (cheapest to
build, and this session's data — `hit_rate` consistently 0.6-0.8, `groundedness` briefly
hit 0.03 — suggests real correctly-cited examples likely already occur by chance in
sampled rollouts, ready to be mined). Decision on which to pursue: deferred to a future
session per Harpreet ("will do that in new session another time").

---

## Probe 4 — 2026-08-24 — `reward_mode="cite_gated"` (hard zero for genuine non-attempts)

**Motivation:** Probes 1-3 revealed "gated" mode's specific exploit — `beta` gives the
SAME partial credit to a rollout that never attempted a citation as to one that tried
and got it wrong, so not-trying is exactly as safe as trying-badly (and strictly safer
once `fab_toll` risk is considered). `cite_gated` closes that loophole directly:
`base = 0 if n_citations==0 else outcome*(cite_gated_floor + (1-cite_gated_floor)*g)` —
a hard zero for genuine non-attempts, same quality-scaled formula as "gated" mode for
anyone who DID cite something. Harpreet caught a real flaw in an earlier, simpler
version of this idea (a pure "cited anything, get credit" gate) — that would only
incentivize pasting garbage citations, not correct ones; this version keeps
`groundedness` fully in control of the credit for the citing subset. `cite_gated_floor`
is fixed (0.5), not ramped — the ramp existed to avoid crushing EVERYONE's reward before
the model could cite; here the crush only ever hits genuine non-attempts, which is
exactly what we want to punish from step 1.

- **Config changed from previous:** `reward_mode` -> `"cite_gated"` (new mode, not
  `"gated"` or `"additive"`). `lr=5e-5` unchanged. `cite_gated_floor=0.5` (Config
  default, not yet tuned — a first real data point).
- **W&B run:** [wdd1i4na](https://wandb.ai/happy22harpreet/deep_research_agent/runs/wdd1i4na)
- **Verified before launch:** config construction, formula ordering property directly
  (`not-cite=0.0 < bad-cite=0.175 < half-cite=0.26 < good-cite=0.35` using
  `floor=0.5, outcome=0.35`), all three touched files parse cleanly.
- **Ran:** all 25 steps to completion, clean finish. W&B run renamed
  `deep_research_agent_probe_cite_gated_COMPLETE_step25_attempt_fixed_quality_not`.
- **Observed — a real, partial success, cleanly split into two separate findings:**
  - **The attempt problem is genuinely fixed.** `pct_rollouts_with_citation` climbed from
    `0.45` (step 1) to a stable plateau of `0.91-0.99` from step 15 onward — a completely
    different trajectory from Probes 1-3, all of which collapsed toward zero. This
    directly confirms the mechanism: making abstention strictly worse than a bad attempt
    (instead of equally safe) removes the incentive to abandon citation.
  - **But a NEW, smaller exploit replaced it — Harpreet's own catch, confirmed with data
    after the fact.** `n_citations` (average count) stayed flat at `~1.0-1.3` for the
    entire run, never scaling toward the multi-citation behavior the system prompt
    explicitly teaches for multi-hop questions. The model learned to paste roughly ONE
    citation marker per answer — just enough to clear the `n_citations>0` gate — not to
    cite thoroughly or correctly. `groundedness` tracked this: it briefly rose to `0.02-
    0.03` around steps 8-13 (the ONLY time in the whole probe series groundedness moved
    meaningfully off zero), then regressed back to EXACTLY 0 for 8 straight steps (17-24)
    despite near-universal citation attempts by then. Net effect: "cite something" got
    solved; "cite correctly/thoroughly" did not.
  - No collapse anywhere: `outcome` stayed healthy and improved late (several steps
    >0.4, peak `0.527` at step 20), `hit_rate` stayed healthy (0.64-0.82), `clip_ratio`
    settled to 0.21-0.44 by the back half (an earlier mid-run rise to 0.48-0.60 around
    steps 8-12 resolved on its own — investigated directly via raw episode-log
    inspection, NOT the final-answer-length theory originally guessed; that theory was
    disproven — 511/512 episodes at step 13 had short, complete final answers, only 1
    genuinely hit the length cap — so `clip_ratio`'s real driver is an intermediate
    turn, not the citation-laden answer; not resolved further, logged as an open item
    below). Final `val/pass@1=0.258`, comparable to Probe 3's `0.262`.
- **Decision:** `cite_gated` is a real, partial win — worth keeping the "hard zero for
  non-attempts" mechanism, but it needs a SECOND term to also punish under-citing (not
  just non-citing) for genuinely multi-hop questions, e.g. scaling the penalty by how
  many gold-supporting passages exist vs. how many were cited, not just a binary
  attempt-or-not check. Not built this session. Combined with GRPO's within-group
  credit-assignment limits discussed live (attempt-vs-not is a large, easy-to-learn
  contrast; correct-vs-incorrect citation is a much smaller one, easily diluted by
  outcome-variance within the same group), this strengthens the case — beyond Probes
  1-3's evidence alone — for the cold-start SFT/RFT/preference path in
  `TENTATIVE_FUTURE_EXPERIMENTS.md`. Harpreet's read, live during the run: pure GRPO's
  reward-shaping is proving to be a real whack-a-mole (two distinct exploits found
  across four reward-design iterations this session) in a way supervised/preference
  methods structurally avoid, since they don't require a scalar reward robust to every
  strategy the policy might discover — worth trying RFT next specifically, since it's
  the cheapest to build and this session's data (hit_rate consistently 0.6-0.8,
  groundedness briefly hit 0.03) suggests some real correctly-cited examples likely
  already occur by chance in sampled rollouts, ready to be mined.
- **Open item, not resolved:** the real driver of `response_length/clip_ratio`'s mid-run
  rise (an intermediate search/read/thought turn hitting the per-turn 256-token cap,
  not the final answer) — worth a future session's direct look if it recurs.
- **Commit:** [reward.py/config.py/train_dr.py "cite_gated" commit — see git log]

---

## Real finding, discovered post-hoc via raw episode logs — 2026-08-24: eval-time turn count

Prompted by "what's the average number of turns during eval," checked directly against
real per-episode eval logs (`runs/<run_name>/episode_logs/episodes/val_step_25_epoch_0/`)
rather than an aggregate metric — no such metric existed for eval specifically (the
`reward_components/*` patch only fires on train-carrying `Tracking.log` calls, so eval
episodes' `n_steps` never got a clean isolated average logged; this was found by
reading the raw JSON directly).

**Every healthy run this session converges on essentially exactly 2 turns per episode
at eval time, near-zero variance:**

| Run | Avg eval turns | Distribution (n=256 or 512) |
|---|---|---|
| Probe 2 (beta ramp fast) | 2.000 | 256/256 at exactly 2 |
| Probe 3 (additive) | 2.008 | 254 at 2, 2 at 3 |
| Probe 4 (cite_gated) | 2.000 | 256/256 at exactly 2 |
| Attempt 4 (cloud, healthy) | 2.154 | 492/512 at 2, small tail to 8 |
| **Attempt 2 (`lr=1e-4`, the collapsed run)** | **8.000** | **256/256 pinned at `max_turns=8`** |

Two things this confirms/reveals:

1. **Independent confirmation of Attempt 2's collapse** (documented earlier from
   `response_length/clip_ratio` alone): every episode ran out the full 8-turn budget
   without ever emitting a clean `answer` action. The raw turn-count data matches the
   clip_ratio story exactly.
2. **The bigger, new finding: in every HEALTHY run, the agent does essentially ONE
   search, then answers directly** — 2 turns = one tool call (almost certainly
   `search`) + the `answer` action, no `read` call, no second hop. This held true
   ACROSS every reward design tried (gated, additive, cite_gated) — the reward-shape
   experiments never touched this, because it isn't a reward-shape problem.

**Root cause found by inspecting the actual system prompt (`env.py::_opening_prompt`)
directly, at Harpreet's prompt — "we didn't provide it any rich examples... it doesn't
even know to use multi-turn tools":** confirmed exactly right. The prompt's ENTIRE
worked example is:
```
For example: Action: search[who directed the film Blue Harvest]
When you know the answer: Action: answer[American [Blue Harvest (film)] [Jane Doe (director)]]
```
The prompt tells the model in prose to "SEARCH... READ... then give a final answer,"
but the only two demonstrated action syntaxes are `search` and `answer` — there is NO
`read[...]` example anywhere, and no full multi-hop worked trajectory
(`search → read → search → read → answer`) showing what the actual research loop looks
like. The model was instructed in words to do something it was never shown how to do.

**Why this likely matters more than anything found in Probes 1-4:** this plausibly
explains several things treated as separate findings today — `n_citations` stuck at
~1/rollout (only one source ever gets meaningfully engaged with), `groundedness≈0`
(search-RESULT SNIPPETS alone likely don't give the model what it needs to cite
correctly — `read` exists specifically to get full passage text), and why NO reward
design (across 4 very different shapes) changed this behavior — a tool-use pattern the
model was never shown can't be taught by reshaping the reward around it; the model
doesn't know the move exists to reinforce.

**Not yet fixed. Concrete, cheap next thing to try — cheaper than RFT/SFT/preference
methods, worth doing FIRST:** add a `read[...]` example and/or a full worked multi-hop
trajectory to `_opening_prompt` (or a few-shot block), see if turn count and
`groundedness` move at all before reaching for a heavier training-stage intervention.
This is pure prompt engineering, not a training run — near-zero cost to test, and
directly addresses the mechanism just found rather than another reward-design variant.
- **Ran:** _(pending)_
- **Observed:** _(pending)_
- **Decision:** _(pending)_


## SFT stage — 2026-08-26 — the citation problem was never a reward problem

**Not an RL run.** Supervised fine-tuning on trajectories from a GPT teacher, done because
all eight runs above hit the same wall. Recorded here because the next GRPO run starts from
this checkpoint, and because it explains what those eight runs were actually fighting.

Full narrative: [`distill/SFT_HISTORY_LOG.md`](../distill/SFT_HISTORY_LOG.md). Where the data
came from: [`distill/DATA_COLLECTION_LOG.md`](../distill/DATA_COLLECTION_LOG.md).

### The finding that reinterprets Attempts 1-4 and Probes 1-4

`citations.verify_citations` scores a citation only if the agent actually called `read` on
that passage:

```python
tp_titles = distinct_cited & gold & read_titles
```

Deliberate ("cite-what-you-read", NOTES.md 2026-08-14), verified by hand, not a bug.

Now combine that with this log's own "eval-time turn count" entry: **every healthy run
converged on ~2 turns — one search, then answer, never a read.** So for those runs, a
citation was **unscoreable by construction**. `groundedness ~= 0` was not the model
refusing to cite; it was the metric correctly reporting that nothing citable had been read.

**Four reward designs (gated, beta-ramped, additive, cite_gated) were tuning the incentive
on an action the policy never emitted.** That is why none of them moved groundedness, and
why the cross-probe synthesis above ("attempt-QUALITY has resisted every reward shape
tried") reads the way it does. It was never a reward-shape problem.

Measured directly during the teacher pilot: the teacher cited BOTH gold titles from search
snippets and scored `cite_f1 = 0.000`. Adding one instruction — read every passage you
intend to cite — took it to 1.000.

### What was done

- **Teacher:** `gpt-4.1-mini` as the POLICY inside the real `DeepResearchEnv` (it picks
  actions, our corpus answers), NOT asked to write transcripts — which would have produced
  invented retrieval. Non-reasoning model chosen deliberately: a 3B student imitating long
  hidden chains learns to start reasoning it cannot finish.
- **Data:** 2,198 episodes, ~$4.63, 30% strict-gate yield. Trained on 418 tier-A
  trajectories (correct AND perfectly cited), all distinct questions.
- **Masking:** loss on the model's own turns only; system prompt and every tool response
  masked. 13.9% of tokens graded. Verified per example against the real tokenizer.
- **Prompt:** trained and evaluated WITHOUT the worked example (`include_worked_example=
  False`), so the demonstration lives in the weights. This is also what the RL stage must
  use — training and rollout prompts have to match.

### Results — held-out, 300 questions, greedy, never trained or tuned on

| | base | base + worked example | tuned |
|---|---|---|---|
| correct | 0.030 | 0.193 | **0.417** |
| picked right sources | 0.007 | 0.291 | **0.744** |
| verified them (read first) | 0.003 | 0.198 | **0.810** |
| citation-F1 (the reward) | 0.002 | 0.110 | **0.710** |
| never used a tool | 0.897 | 0.107 | **0.000** |
| **correct AND properly cited** | **0.0%** | **0.3%** | **30.7%** |

The middle column exists so the win cannot be overclaimed: **most of the raw correctness
gain is available from prompting alone.** The adapter's distinctive contribution is
read-before-cite (0.810 vs 0.198) and the correct-and-cited bucket, which had been 0 in
every configuration measured that day.

**The 2-turn collapse is gone.** Tool calls per episode peak at 5 (59%) =
search->read->search->read->answer; searches 2.28 vs reads 2.31, near-balanced. The
prompted model still peaks at 2-3 — the historical pattern this log records.

### The failure the RL stage should target

**16% of episodes (47/300) never terminate** — they search until the turn budget runs out
and return nothing (correct 0.064, and only 19% produce any answer at all). Stuck episodes search **2.4x**
more than successful ones (4.36 vs 1.81) and read 1.53x more than needed: a failure to
COMMIT, not slowness. Cause: trained only on trajectories where the teacher SUCCEEDED, so
the model never saw concluding under uncertainty.

Excluding them: **0.482 correct / 0.817 cite_f1**.

**Why this matters for the next GRPO run:** the eight runs above asked RL to *discover* a
behaviour from sparse reward, and it could not. This time the shape already exists 84% of
the time, and a zero outcome-reward for never answering is exactly the contrast GRPO's
group-relative advantage handles well. That is the premise the whole RFT plan rested on,
and it is now actually satisfied.
