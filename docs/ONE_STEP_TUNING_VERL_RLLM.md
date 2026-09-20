# One-Step Throughput Tuning — verl/rLLM GRPO on a Single A100

**Date:** 2026-08-23
**Context:** `deep_research_agent` capstone (Capstone 6, Branch B) — GRPO-trained
multi-hop QA agent, rLLM (agent/Workflow layer) + veRL 0.9.0 (GRPO engine).
**Hardware:** 1× RunPod A100 80GB SXM4, colocated mode (vLLM rollout engine +
FSDP actor training time-share the *same* GPU — no separate GPU per role).
**Model:** `Qwen/Qwen2.5-3B-Instruct` + LoRA (`rank=16, alpha=32`), bf16.
**Config baseline:** `config.cloud_preset()` — `group_size=16`,
`prompts_per_step=1`, `grad_accum=4` → `rollouts_per_step=64` (as
`data.train_batch_size`, i.e. 64 *prompts*/step) × `rollout.n=16` = **1024
raw trajectories/step**, `max_turns=8`, `max_new_tokens=256`,
`max_prompt_length=4096`.

This document is the log of a real, measured throughput-debugging session —
every number below came from an actual run on the pod, not an estimate,
unless explicitly marked "estimated." See `WORKFLOW_PORT_NOTES.md` for the
earlier (same-day) batch-divisibility bug this tuning builds on top of.

---

## 1. Why this mattered

The first real cloud training attempt was launched and looked healthy at a
glance — rollout generation for 1024 trajectories finished in ~4 minutes, no
crash. But the run then sat quiet for 5+ minutes at 14% GPU utilization
before any per-step metric printed. Judged (correctly, in hindsight) as a
real throughput problem rather than "just wait" — a 6-hour training budget
(`max_train_hours=6.0`) with 1000 configured steps needs roughly
**≤21.6s/step** to actually reach step 1000, and even reaching
`lambda_eff_ramp_start=60` (see `config.py`) needs the run to survive long
enough to get there. The run was killed deliberately to fix this properly
instead of burning cloud spend on a run that might not even clear step 60.

## 2. Method: measure, don't guess

Every fix below was validated by a **dedicated throughput probe** — a
standalone script (`probe_throughput_fullscale.py`, not part of the regular
CLI) that runs the exact `cloud_preset()` rollout volume (1024 trajectories)
for **exactly 1 step**, console-only logging (no stray W&B runs), capped
`max_train_hours=0.5` as a safety net, then exits. This gives the real
`timing_s/*` breakdown veRL logs per step without paying for a full run.
A second, smaller probe (`probe_throughput.py`, 8×1×1=8 prompts/64 raw
rollouts, 3 steps) was used earlier for a first fast signal before committing
pod time to full-scale runs.

Why this matters as a practice, not just for this session: guessing at
config values and relaunching the full 6-hour run to "see if it's faster" is
how you burn a rental budget without learning anything. A 1-step probe at
real scale costs ~3-4 minutes of GPU time (mostly fixed vLLM/FSDP warm-up)
and gives an exact, attributable answer.

## 3. Baseline: what was actually slow

First full-scale probe, using the config as originally written
(`ppo_micro_batch_size_per_gpu=1`, `ppo_mini_batch_size=1` — both set during
earlier correctness-first debugging of a batch-divisibility crash, see
`WORKFLOW_PORT_NOTES.md` bug 11/12):

| phase | time | % of step |
|---|---|---|
| `generate_trajectories` (vLLM rollout) | 26.4s | 7% |
| `old_log_prob` | 61.8s | 17% |
| `adv` | 0.07s | ~0% |
| **`update_actor`** | **264.1s** | **73%** |
| `update_weights` (FSDP→vLLM sync) | 5.5s | 2% |
| **total `timing_s/step`** | **360.6s** | |

Total tokens processed that step: `perf/total_num_tokens: 591917`.

**Key finding #1 — rollout generation was never the bottleneck.** vLLM's
batched generation for 1024 multi-turn ReAct trajectories took only 26s. The
instinct "maybe 1024 trajectories is too many, lower it" would have been
treating the wrong symptom — the volume itself wasn't slow, what came *after*
generation was.

**Key finding #2 — `update_actor` (73% of the step) was the real target.**
`old_log_prob` is a forward-only pass over the *same* token volume as
`update_actor`'s forward+backward — comparing the two isolates whether the
slowness is inherent compute cost or repeated per-call overhead. 61.8s
(forward only) vs. 264.1s (forward+backward+optimizer) is a >4x ratio, well
above what forward-vs-forward+backward+optimizer should cost on its own
(normally ~2-3x). That gap is overhead, not compute.

## 4. Fix 1 — `use_dynamic_bsz` (fixed micro-batching → token-budget packing)

**Root cause of the overhead:** `ppo_micro_batch_size_per_gpu=1` forces
every forward/backward micro-batch to contain exactly **one sequence**,
regardless of how short it is. Our DR-agent trajectories vary hugely in
length (observed 10–1466+ tokens per trajectory in the mask-diagnostic log),
so batch=1 either wastes the GPU on short sequences or, if raised naively,
risks OOM on long ones — the original conservative choice.

**The fix, verified from verl source before applying** (not guessed):
`verl/workers/config/actor.py::ActorConfig.__post_init__` gates the
"must set `ppo_micro_batch_size_per_gpu`" assertion on `not use_dynamic_bsz`
— setting `use_dynamic_bsz=True` removes that requirement entirely. It also
removes the `real_train_batch_size % minimal_bsz == 0` divisibility assert
in `verl/utils/config.py` (the root class of problem that forced
`mini_batch_size=1` as a workaround earlier the same day).

With `use_dynamic_bsz=True`, instead of "N sequences per micro-batch," veRL
packs sequences into a chunk **by total token count**: it keeps adding
sequences until the running total would exceed `ppo_max_token_len_per_gpu`
(default 16384), then starts a new chunk. Many short trajectories share one
forward pass; a few long ones might get their own. Same total compute, far
fewer separate forward/backward calls.

**One more thing confirmed before relying on this:** does it actually reach
all three passes (`old_log_prob` for the rollout-time re-check, the ref
policy's log-prob, and `update_actor`)? Checked the resolved hydra config
directly — `actor_rollout_ref.ref.log_prob_use_dynamic_bsz` and
`actor_rollout_ref.rollout.log_prob_use_dynamic_bsz` are both defined as
`${oc.select:actor_rollout_ref.actor.use_dynamic_bsz,false}` — i.e. they
**auto-inherit** from the actor's setting. One override, all three passes
fixed. (Also confirmed separately: `ref_in_actor` — verl auto-detects
`lora_rank > 0` and computes the reference policy's log-probs by disabling
the LoRA adapter on the *same* loaded actor model rather than running a
second full model, so this was really 2 serial passes to fix, not 3.)

**Override applied** (`train_dr.py::verl_overrides`):
```
actor_rollout_ref.actor.use_dynamic_bsz=True
```
(`ppo_micro_batch_size_per_gpu` overrides removed — no longer needed/used.)

**Measured result** (small-scale probe first, 8 prompts/64 raw rollouts,
sanity-checking the mechanism before spending full-scale pod time):
full step (generate+logprob+update+sync) completed in **34.9s**, where the
same config had previously not even finished within 5+ minutes at full
scale. Confirmed no crash, no regression on the existing sanity spike.

## 5. Fix 2 — raise `ppo_mini_batch_size` (1 → 64 → 256)

**Why `use_dynamic_bsz` alone wasn't enough:** dynamic_bsz packs sequences
*within* a mini-batch. `ppo_mini_batch_size=1` means each mini-batch
contains exactly one row — there's nothing to pack, and (critically) each
row still gets its **own separate optimizer step** (its own FSDP gradient
sync + `optimizer.step()` + Python loop iteration). With ~1024–1082
collected rows/step (row count is data-dependent — occasional trajectories
don't cleanly merge via veRL's cumulative-prefix match and contribute extra
rows, see `WORKFLOW_PORT_NOTES.md` bug 11/12), that's ~1000+ separate
optimizer steps per training step.

Confirmed via the full-scale probe with `use_dynamic_bsz=True` alone (mini
batch still 1): `update_actor` was still 264.1s — the dynamic_bsz fix by
itself did essentially nothing for `update_actor`, exactly as reasoned
(there was nothing to pack at mini-batch size 1). `old_log_prob`, which has
no per-mini-batch optimizer-step concept (it's a plain forward pass over the
dynamic-bsz-packed chunks), stayed similarly fast — confirming the two
passes have genuinely different bottleneck mechanisms.

**Fix:** raise `ppo_mini_batch_size` off `1`. The earlier concern that
motivated `=1` (veRL's hard divisibility assert crashing on an unpredictable
row count) is **already mitigated** by `_patch_make_iterator_for_ragged_batches`
(`rllm_workflow.py`, registered as a Ray `worker_process_setup_hook` — see
`WORKFLOW_PORT_NOTES.md` bug 11/12) — it falls back to one full-batch
mini-batch instead of crashing whenever the collected count doesn't divide
evenly. That safety net is what makes raising this value low-risk.

**Measured, full cloud-preset scale (1024 rollouts, ~590K tokens/step):**

| `ppo_mini_batch_size` | `update_actor` | `old_log_prob` | `timing_s/step` |
|---|---|---|---|
| 1 | 264.1s | 61.8s | 360.6s |
| 64 | **135.8s** (−49%) | 61.2s (flat, expected) | 232.8s (−35%) |
| 256 | see §6 (combined with token budget) | | |

At `mini_batch_size=64`, `update_actor` (135.8s) is only ~2.2x
`old_log_prob` (61.2s, same token volume) — much closer to a plausible
forward-vs-forward+backward+optimizer ratio, i.e. now closer to
compute-bound than overhead-bound.

**Design note, not just a speed knob:** `ppo_mini_batch_size` also controls
how "on-policy" each mini-batch's gradient step is. `ppo_epochs=1` means the
collected batch is used exactly once (never replayed in a future training
step — the on-policy property that matters most is preserved regardless).
But *within* that one pass, more mini-batches means more sequential
gradient steps before the batch is exhausted, and each step nudges the
policy slightly further from where the data was actually sampled — bounded
by PPO's clip range, but non-zero. Raising `mini_batch_size` from 1 to 64
*reduces* that intra-step drift (17 mini-batches instead of ~1000+), so this
change is a win on both throughput and on-policy-ness, not a tradeoff
between them. Going all the way to `mini_batch_size` = full collected batch
(~1 mini-batch, pure full-batch gradient descent) would be the most
on-policy option but the noisiest gradient estimate — not tested, deferred
in favor of the memory-budget lever below, which had cleaner measured
headroom.

## 6. Fix 3 — raise `ppo_max_token_len_per_gpu` (16384 → 65536)

**What it is:** the token-count ceiling per dynamic_bsz chunk. Both
`old_log_prob` and `update_actor`'s forward pass are chunked by this budget
(`ref`/`rollout` inherit it too, confirmed via the same `${oc.select:...}`
interpolation pattern as `use_dynamic_bsz`). A bigger budget means fewer,
larger chunks for the *same* total tokens — less repeated per-chunk
Python-loop/kernel-launch/(for the actor) FSDP-sync overhead, no change in
total FLOPs.

**Why raise it, with evidence first:** the `mini_batch_size=64` probe
reported `perf/max_memory_allocated_gb: 23.2` / `max_memory_reserved_gb:
25.1` against 80GB available during the FSDP actor phase specifically (a
different, much lower number than vLLM's own peak of ~65GB during *its*
phase — colocated mode, the two don't peak simultaneously). Real, measured
headroom, not assumed — so raising the token budget 4x (16384→65536) was a
reasoned bet, then verified directly rather than trusted blindly.

**Measured, combined with `ppo_mini_batch_size=256`** (tested together for
pod-time efficiency; the per-phase breakdown still attributes each effect —
`old_log_prob` has no mini-batch-count dependency, so its improvement is
attributable to the token-budget change alone):

| setting | `update_actor` | `old_log_prob` | `timing_s/step` | peak mem (actor phase) | MFU |
|---|---|---|---|---|---|
| mini=64, token=16384 | 135.8s | 61.2s | 232.8s | 23.2 / 25.1 GB | 0.29 |
| mini=256, token=65536 | **96.7s** (−29%) | **40.4s** (−34%) | **176.8s** (−24%) | 54.9 / 57.8 GB | **0.42** |

No OOM (57.8GB reserved vs. 80GB total — ~22GB margin remaining). Model
FLOPs Utilization (`mfu`) rose from 0.29 → 0.42, a genuine efficiency gain,
not merely "spend more memory for the same work."

The batch-safety-patch (`_patch_make_iterator_for_ragged_batches`) fired
again at this setting — `"1082 collected rows not divisible by requested
mini_batch_size=4096"`. `4096 = 256 (ppo_mini_batch_size) × 16 (group_size)`
— this looks like veRL aligning mini-batch boundaries to whole GRPO groups
(so one prompt's 16 rollouts never split across two gradient steps, which
would corrupt the group-relative advantage baseline) rather than a new bug.
The patch's non-fatal fallback (one full-batch mini-batch instead of
crashing) covered it as designed — not further investigated, not urgent.

## 7. Fix 4 — reduce rollout volume (`grad_accum` 4 → 2)

**Why this is a different category from fixes 1-3:** everything above was
pure waste removal — no change to what gets trained on, only to how
efficiently the same work gets done. This one *does* shrink the actual
GRPO batch (statistically noisier gradient estimate per step), so it was
held back until the waste-removal levers were fully explored, and flagged
explicitly rather than applied silently.

**Signal that waste-removal was running out of room:** MFU (Model FLOPs
Utilization) rose from 0.29 → 0.42 with fix 3 — a real efficiency gain, and
also evidence we were becoming genuinely *compute-bound*: further batching
tricks have a shrinking ceiling once you're mostly paying for real FLOPs
rather than repeated overhead. At that point the only lever with real
headroom left is doing less total compute per step.

**Change:** `grad_accum` 4→2, which roughly halves `rollouts_per_step`
(`group_size × prompts_per_step × grad_accum`) and therefore total
tokens/step (591K → ~295K).

**Measured, full cloud-scale probe** (`ppo_mini_batch_size=256`,
`ppo_max_token_len_per_gpu=65536` held constant from fix 3):

| | grad_accum=4 (1024 rollouts, 591K tok) | grad_accum=2 (512 rollouts, 295K tok) |
|---|---|---|
| generate_trajectories | 29.2s | 20.9s |
| old_log_prob | 40.4s | 21.7s |
| update_actor | 96.7s | **48.8s** |
| update_weights | 7.1s | 6.9s (flat — fixed cost, not volume-dependent) |
| **timing_s/step** | **176.8s** | **100.6s** (−43%) |
| peak mem (actor phase) | 54.9 / 57.8 GB | 54.2 / 57.0 GB (flat — bounded by chunk *size*, not total volume) |
| MFU | 0.42 | 0.41 (flat — confirms this was a "less work," not "more efficient," win) |

Near-linear scaling with token volume (tokens halved, most phases roughly
halved) — further confirmation we're compute-bound, not overhead-bound,
at this point.

## 8. Real vs. measured: step-1 warmup is not the steady-state number

All probes above report **step 1** of a fresh process. `use_torch_compile`
is on by default (`actor_rollout_ref.actor.fsdp_config.use_torch_compile:
True`) — if compilation happens once and is reused, step 1 pays a warmup
tax that steady-state steps don't. Worth checking directly rather than
assuming the step-1 number is what a real multi-step run would see.

**3-step probe at the grad_accum=2 setting** (`PROBE_STEPS=3` — note:
`cfg.steps=3` on veRL triggers an automatic *final validation pass* at the
end, which consumed what would have been "step 3"'s training timing, so
only 2 real training-step samples were obtained — still enough to answer
the question):

| step | `timing_s/step` |
|---|---|
| 1 | 102.2s |
| 2 | **88.6s** (−13%) |

The drop is concentrated in `generate_trajectories` (21.2s → 11.5s, likely
vLLM request-scheduling/prefix-cache warmup) and a smaller amount in
`old_log_prob` (22.4s → 19.9s). `update_actor` stayed flat (48.9s → 48.6s)
— whatever compile cost it pays is apparently already absorbed before step
1 finishes (plausibly during the initial FSDP→vLLM weight-sync/setup phase
that happens before step 1 is even timed), so there's no further reduction
to expect from steps 3+ onward. **88.6s/step, not 102.2s, is the honest
number to extrapolate a real run from.**

## 9. Fix 5 — Liger kernels (`use_liger`)

**What it is:** fused Triton kernels (RMSNorm, RoPE, SwiGLU, cross-entropy)
from [linkedin/Liger-Kernel](https://github.com/linkedin/Liger-Kernel),
patched into the actor's forward+backward via a lazy import in
`verl/workers/engine/fsdp/transformer_impl.py` (only imports
`liger_kernel` when `model.use_liger=True`, so it's not a hard dependency
of this lab). A genuine compute-cost reduction, not a batching trick — the
first lever in this document that isn't just "do the same FLOPs more
efficiently."

**Made configurable, not hardcoded:** `config.py`'s `verl_use_liger: bool
= False` (default off, until measured) → `train_dr.py::verl_overrides()`
→ `actor_rollout_ref.model.use_liger`. `probe_throughput_fullscale.py`
gained a `PROBE_LIGER=1` env-var toggle to test either state without
editing files.

**Compatibility checked before installing, not assumed:** confirmed via
[Liger's own docs](https://github.com/linkedin/Liger-Kernel) that it
supports transformers ≥4.52.0 including v5 (`ONE_STEP_TUNING`-session
environment: transformers 5.5.4) and has long-standing Qwen2/2.5 support
(their newer work targets Qwen3.5, meaning 2.5 is well-trodden). Installed
`liger-kernel==0.8.2` — confirmed it didn't touch the existing
torch/transformers/verl versions.

**Correctness before scale:** a small sanity run (`group_size=4`,
`grad_accum=2`, 2 steps, tiny 17,640-token step) confirmed no crash and
sane loss/grad_norm values (`pg_loss: 0.0042`, `grad_norm: 0.44` — same
order of magnitude as every non-Liger run) before spending full-scale pod
time on a timing probe.

**Measured, full cloud scale (`grad_accum=2`, steady-state step 2):**

| phase | Liger OFF | Liger ON | change |
|---|---|---|---|
| generate_trajectories | 11.5s | 11.3s | flat (expected — Liger patches the FSDP training model, not vLLM's separate rollout engine) |
| old_log_prob | 19.9s | 18.6s | −6.4% |
| update_actor | 48.6s | **42.1s** | **−13.3%** |
| update_weights | 7.0s | 7.0s | flat (expected — unrelated to Liger) |
| **timing_s/step** | **88.6s** | **85.7s** | **−3.3%** |
| MFU | 0.415 | **0.482** | genuine efficiency gain |
| peak mem (actor phase) | 57.0GB | 57.2GB | flat (not a memory lever) |

**Honest caveat on interpretation:** a first look at step-1-only numbers
(87.7s vs. 102.2s, "14% faster") overstated the effect — that comparison
rode some of the same `generate_trajectories` warmup noise identified in
§8, which is unrelated to Liger. The real, steady-state, attributable win
is concentrated in `update_actor` (a genuine −13.3%) and a smaller
`old_log_prob` win; total step time only drops 3.3% because those two
phases are ~71% of the budget, not all of it. Still a real, free,
no-downside win (no loss/grad_norm anomalies, no OOM, memory unchanged) —
worth keeping on, just not the dramatic win the step-1-only comparison
suggested.

## 10. Cumulative result

| stage | `timing_s/step` | cumulative change from baseline |
|---|---|---|
| baseline (mini=1, token=16384, fixed micro-batch=1) | 360.6s | — |
| + `use_dynamic_bsz=True` alone (small-scale sanity) | (34.9s at 8-prompt scale — mechanism confirmed) | |
| + `ppo_mini_batch_size=64` | 232.8s | −35% |
| + `ppo_mini_batch_size=256`, `ppo_max_token_len_per_gpu=65536` | 176.8s | −51% |
| + `grad_accum` 4→2 (half rollout volume) | 100.6s | −72% |
| steady-state (step 2+, warmup excluded) | 88.6s | −75.4% |
| + `use_liger=True` | **85.7s** | **−76.2%** (4.2x speedup) |

## 11. One more memory increment tested — confirmed diminishing returns

Before finalizing, one more data point: `ppo_max_token_len_per_gpu` 65536→
98304 (+50%), on top of `grad_accum=2` + `use_liger=True`, measured at full
cloud scale (steady-state step 2):

| | token=65536 | token=98304 (+50%) |
|---|---|---|
| `timing_s/step` | 85.7s | 83.1s (**only −3%**) |
| peak mem reserved | 57.2GB | **69.0GB** (+21%) |
| margin left (of 80GB) | ~23GB | **~11GB** |
| MFU | 0.482 | 0.503 |

No OOM, so not unsafe outright — but a bad risk/reward trade: 50% more
memory footprint (margin cut roughly in half) for 3% more speed. This is
the compute-bound ceiling the rising MFU trend had been signaling since
§6. **Decision: stop here, keep `ppo_max_token_len_per_gpu=65536`, not
98304** — confirmed by measurement, not left as an open question.

## 12. Where this leaves the real run — honest math, not spin, and what got applied

**Applied to `cloud_preset()` in `config.py` as of 2026-08-24:**
`grad_accum=2` and `verl_use_liger=True`. `verl_ppo_max_token_len_per_gpu`
stays at its `Config` default of 65536 (§11's finding — not 98304).
Verified via `python train_dr.py --dry-run cloud`: resolved config matches
what was measured exactly (`data.train_batch_size=32` → 32×16=512 raw
rollouts, `ppo_mini_batch_size=256`, `ppo_max_token_len_per_gpu=65536`,
`use_liger=True`, `use_dynamic_bsz=True`).

At 85.7s/step (half rollout volume vs. the original `cloud_preset()`, plus
Liger), `1000 steps × 85.7s ≈ 23.8 hours` — **still beyond** the
`max_train_hours=6.0` cap, though far closer than the 100-hour estimate at
the start of this document. In 6 hours: `21600s / 85.7s ≈ 252 steps`. That
comfortably clears both `lambda_eff_ramp_start=60` *and*
`lambda_eff_ramp_end=200` (the efficiency-toll ramp, see `config.py`) with
room to spare, but still falls short of the full 1000-step schedule.

**What's left, if more runway is ever wanted (none applied, all deliberate
stops rather than open questions):**

1. **Reduce rollout volume further** (`grad_accum` 2→1, `group_size`
   16→8) — same tradeoff class as §7, smaller/noisier GRPO batch either
   way, deliberately not applied — the user's explicit call was to hold
   at `grad_accum=2` to preserve training signal.
2. **Extend `max_train_hours`** beyond 6 — a budget/cost decision for the
   user, not a technical one; noted here only as an option, not a
   recommendation.
3. **More memory headroom** — tested and declined, §11. Not a live option
   unless the risk/reward calculus changes (e.g. a different model size).

## 13. Debugging tools built along the way (kept in the repo)

- `probe_throughput.py` — small-scale (8 prompts/64 rollouts, 3 steps) fast
  signal, console-only logging, capped `max_train_hours=0.3`.
- `probe_throughput_fullscale.py` — exact `cloud_preset()` rollout volume,
  console-only, capped `max_train_hours=0.5`. This is the one that produced
  every full-scale number in this document. Env vars: `PROBE_STEPS` (default
  1 — use ≥3 to get a real step-2+ steady-state sample, since veRL's
  automatic final-eval consumes whatever the *last* configured step would
  have been, see §8), `PROBE_GRAD_ACCUM` (override `cfg.grad_accum`),
  `PROBE_LIGER=1` (force `cfg.verl_use_liger=True`).

Both are debugging tools, not part of the regular CLI (`train_dr.py`) — they
mirror its real-training code path (same `AgentTrainer`/`DeepResearchWorkflow`/
`ray.init` wiring) exactly, just at reduced scope. Delete them once the real
cloud run's throughput is confirmed healthy, or keep them if the tuning
knobs need re-validating after a future config change (e.g. if `max_turns`,
`max_new_tokens`, or the model size changes materially, sequence-length
distribution changes and these numbers should be re-measured, not assumed
to still hold).

## 14. What actually changed in `train_dr.py::verl_overrides` / `config.py`

```
actor_rollout_ref.actor.use_dynamic_bsz=True
actor_rollout_ref.actor.ppo_mini_batch_size=256          # was 1
actor_rollout_ref.actor.ppo_max_token_len_per_gpu=65536  # was default 16384
```
Removed: `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1`,
`actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1`,
`actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1` — all
unnecessary once `use_dynamic_bsz=True` (the underlying assert that required
them is gated on `not use_dynamic_bsz`).

**Applied to `cloud_preset()` in `config.py` (2026-08-24), after the
explicit hold in §7/§9 and the user's explicit sign-off:**
```
grad_accum=2              # was 4 — the one signal-quality tradeoff in this doc
verl_use_liger=True       # was False — free win, no tradeoff
verl_ppo_max_token_len_per_gpu=65536   # default, unchanged — §11 declined 98304
```
`grad_accum` 4→2 is the one change in this document that trades
training-signal quality for wall-clock speed rather than removing pure
waste — the user's explicit call was to hold here (not go to 2→1) to
preserve training signal. `verl_use_liger` requires `pip install
liger-kernel` (installed on this pod: `0.8.2`) — verl only imports it
lazily when the flag is True, so it was never a hard dependency while the
default was False.

Verified end-to-end via `python train_dr.py --dry-run cloud`: resolved
config matches the measured 85.7s/step run exactly.
