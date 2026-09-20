# RL-from-SFT — running log (2026-09-07 session)

**What this file is.** A chronological, in-progress log of the GRPO-on-top-of-SFT stage:
what is being built, why, the reasoning behind each choice, every issue hit and how it was
fixed, results as they land, and speculation clearly marked as such. Written as the
session goes, not reconstructed afterwards. Companion to `TRAINING_HISTORY_LOG.md` (one
entry per training run, terse) and `distill/SFT_HISTORY_LOG.md` (the previous stage).
Conclusions get promoted to `START_HERE.md` / `HANDOFF.md` at the end; this is the trail.

Legend: **[decision]** a choice and its reasoning · **[issue]** something that broke ·
**[fix]** what was done about it · **[result]** a measured number · **[speculation]** a
belief not yet backed by a measurement.

---

## 0. Where the project stood at session start

Read in full: `START_HERE.md`, `HANDOFF.md` (all five sections), `distill/SFT_RL_PLAN.md`,
`distill/RESULTS.md`, `distill/SFT_HISTORY_LOG.md`, `distill/DATA_COLLECTION_LOG.md`,
`distill/BATCH_SIZE_TUNING.md`, `TRAINING_HISTORY_LOG.md`, `FRESH_POD_SETUP_AND_SANITY_CHECK.md`,
`RLLM_VERL_INSTALL_NOTES.md`, `ONE_STEP_TUNING_VERL_RLLM.md` (skim), `WORKFLOW_PORT_NOTES.md`
(skim), and the code: `config.py`, `train_dr.py`, `rllm_workflow.py`, `env.py`, `reward.py`,
`citations.py`, `evaluate.py`, `splits.py`, `data.py`, `hub.py`, `distill/{sft_train,
sft_data,build_sft,eval_sft}.py`, `rft_diagnosis/diagnosis1.py` (the classifier + stop
sequences).

The state, in one paragraph: a LoRA SFT adapter (`harpreet22happy/deep-research-agent-sft`,
r=32) trained on 418 GPT-4.1-mini teacher trajectories took the 3B student from 0.3% to
30.7% correct-and-properly-cited on a locked 300-question held-out set, and installed
read-before-cite (0.198 → 0.810). Its dominant failure (F12): 16% of episodes never commit
to an answer — they search until the 8-call budget runs out. Every GRPO run before the
SFT stage (Attempts 1–4, Probes 1–4) was confounded by F1 (a citation only scores if the
passage was `read`, and those policies never read), so none of them says anything about
GRPO on top of a policy that does read. RL has never been run on the SFT checkpoint.

**The goal of this session:** GRPO from the SFT policy, on questions SFT never saw, and an
honest measurement of whether it beats the SFT model — with the specific risk that the
outcome reward teaches "answer immediately" and undoes read-before-cite.

Machine: a fresh RunPod A100-SXM4-80GB (driver 580.126, CUDA 13.0), 128 vCPU, 79 GB local
disk on `/`, `/workspace` is a network mount (per the standing rule, nothing is built
there). No venv, no adapter, no model on disk. `.env` has `HF_TOKEN` + `WANDB_API_KEY`.

---

## 1. Design — decisions made before writing code

### 1.1 [decision] How the SFT policy gets into GRPO: merge the adapter, train a fresh LoRA

Two ways to start verl's GRPO from a LoRA checkpoint:

| option | mechanism | what the KL reference is |
|---|---|---|
| (a) `actor_rollout_ref.model.lora_adapter_path` | verl loads the adapter into the actor (`verl/workers/engine/fsdp/transformer_impl.py:323`) | **the base model** — with LoRA, verl's `ref_in_actor=True` computes the reference as the actor with the adapter *disabled* (`verl/trainer/ppo/ray_trainer.py:356-360`) |
| (b) merge the adapter into the weights (`peft merge_and_unload`), point `model.path` at the merged dir, train a fresh LoRA on top | standard HF model dir; verl/vLLM see an ordinary Qwen2 | **the SFT policy** (LoRA disabled = merged weights) |

Read from the installed-version source (downloaded the verl 0.9.0 wheel and the pinned
rLLM commit), not docs. (a) is the trap: the KL term would pull the policy *toward* the
model that never reads before citing — the exact thing SFT installed. Chose (b).
A fresh LoRA's B matrix is zero-initialised, so step 0 of GRPO is byte-for-byte the SFT
policy. Cost: ~6 GB on disk, rebuildable in a minute (`distill/merge_sft.py`, which also
verifies the merge by comparing logits from disk against base+adapter).

Consequence for eval: the RL model is `merged-SFT + RL-LoRA`, so the eval engine is built
on the merged weights and the "SFT" arm is that engine with no adapter. Base-model arms
already exist in `distill/eval_sft_A_*.json` and are not re-run.

### 1.2 [issue → decision] The historical runs had NO KL term at all

`config.kl_coef=0.02` maps to `algorithm.kl_ctrl.kl_coef`, which verl only consumes when
`algorithm.use_kl_in_reward=True` — default `False` (`ppo_trainer.yaml:98`). `actor.use_kl_loss`
also defaults `False`. So Attempts 1–4 and Probes 1–4 trained with no KL regularisation.
Not a bug that hurt them (they were confounded by F1 anyway), but for RL-from-SFT the
reference is the policy we want to stay near, so `rl_from_sft` sets
`actor.use_kl_loss=True, kl_loss_coef=0.01, kl_loss_type=low_var_kl`. 0.01 is 10× verl's
default — **[speculation]** first data point, chosen because the failure mode being guarded
against (behaviour collapse to 2-turn episodes) is exactly what a KL-to-SFT term resists;
if `actor/kl_loss` dominates `actor/pg_loss` in the probe it comes down.

### 1.3 [decision] Stop sequences in TRAINING rollouts, matching eval

`distill/eval_sft.py` (and every diagnosis) sampled with
`stop=["\nThought:", "\nsearch results:", "\n["]`, so the SFT numbers were measured with the
model cut off the moment it starts writing a fake tool result. The GRPO rollouts had no
stop strings. Two reasons to add them:
1. **Train/eval consistency** — the policy is evaluated under them, so it should be
   optimised under them.
2. **Masking hygiene** — a model that free-runs past `Action:` and writes
   `search results: [1] ...` has that hallucination in its *completion tokens*, i.e.
   GRADED as its own output. The `MULTILINE` parser ignores it for the tool call, but
   the loss doesn't.

Verified the plumbing end to end by reading the pinned sources: `Workflow.timed_llm_call(**kwargs)`
→ `RolloutEngine.get_model_response(**kwargs)` → `VerlEngine.get_token_output_from_token_input`
does `sampling_params.update(kwargs)` → verl's vLLM server builds
`SamplingParams(max_tokens=..., **sampling_params)`. `stop` is a native vLLM field, so it
passes straight through. Wired as `cfg.rollout_stop_sequences` (empty = historical).

**[speculation]** A stop-string-truncated completion has no `<|im_end|>`, so the next
turn's re-tokenised prompt appends one — that is still a prefix-extension of
`prompt_ids+completion_ids`, so the cumulative-prefix merge should hold. If
`[mask-diagnostic] NON-CUMULATIVE` fires far more often than the historical ~31%, this is
the first suspect.

### 1.4 [decision] Reward: `cite_gated`, everything else as `cloud_preset`

`START_HERE.md` names the risk: outcome reward pays for answering, so the cheapest fix for
F12's zeros is to answer immediately and stop reading (Probe 3's shape). Three counters:
`cite_gated` (no citation → hard 0, and a citation only *scores* if the passage was read,
so `read` stays load-bearing), the KL term above, and per-step logging of
`n_steps / n_reads / n_searches / answered / capped` under `reward_components/` so a slide
toward 2-turn episodes is visible in the first ~20 steps. `lr=5e-5` (the only value that
neither flat-lined nor collapsed on this stack). Efficiency-toll ramp left at 150→230, i.e.
effectively off for a run this length — **[decision]** do not add a turn toll on top of the
outcome contrast until the outcome contrast alone has been observed.

### 1.5 [decision] Data: `rl_train` for rollouts, `sft_dev` for early-stop, `heldout_eval` once

`train_dr.build_dataset` now honours `cfg.train_split` / `cfg.eval_split` (splits.py
partitions, disjoint by asserted `task_id`s) and refuses `heldout_eval` on either side.
One correction to the docs' worry: `data.train_batch_size = G×prompts×accum = 32` is
*prompts per step*, so a 250-step run touches ~8,000 distinct questions of the 14,500 pool,
not 250. `prompts_per_step=1` was never the binding constraint.

`max_prompt_length` 4096 → 6144: the SFT policy averages 5.4 tool calls/episode and the
conversation grows with every `read`; 4096 was sized for 2-turn episodes.

---

## 2. Build log — what was written

| file | change |
|---|---|
| `config.py` | fields `train_split`, `eval_split`, `rollout_stop_sequences`, `verl_use_kl_loss/_coef/_type`; presets `rl_from_sft()` and `probe_rl_from_sft()` (12 steps, no push) |
| `train_dr.py` | split-aware `build_dataset`; KL-loss overrides; CLI dispatch; `push_checkpoints` scores on the eval split with the same stop sequences; a loud guard when `model_name` is a local dir that doesn't exist yet |
| `rllm_workflow.py` | `stop=` passed into every `timed_llm_call`; five behaviour counters added to `_REWARD_COMPONENT_KEYS` |
| `env.py` | `_terminal_reward` info dict gains `n_reads / n_searches / n_tool_calls / answered / capped` |
| `distill/merge_sft.py` | NEW — download the Hub adapter, merge into base, save, reload from disk, compare logits vs base+adapter, write `MERGE_INFO.json` |
| `distill/eval_rl.py` | NEW — SFT vs SFT+RL on one engine (merged base, LoRA hot-swap), reusing eval_sft's classifier/summary/probes; adds `capped_rate`, `answered_rate`, `correct_and_cited_rate`; RL gate from SFT_RL_PLAN §4 + a turn-collapse check |
| `tests/test_rl_from_sft.py` | NEW — 7 offline tests (preset contract, override map, heldout refusal, env counters) |
| `setup_pod.sh` | bug #7 below |
| `.gitignore` | `sft_merged/`, `sft_adapter/` |

---

## 3. Issues hit, in order

### 3.1 [issue] setup_pod.sh died at its own pytest gate on a fresh pod — bug #7
`tests/test_splits.py` imports `dotenv`, which step 4 installs, but step 3's
`python -m pytest -q tests/` runs first. 6 `ModuleNotFoundError`s → non-zero exit →
`set -euo pipefail` killed the script. Only visible because the log was checked rather
than the process assumed alive. **[fix]** `python-dotenv` added to step 2's foundation
install. Lesson: the same shape as bugs #3/#4/#6 — a dependency that landed *later* than
the first thing that needed it.

### 3.2 [issue] My own guard broke the same gate — bug #8 (self-inflicted)
Added a check in `train_dr.main` refusing to start when `model_name` is a local directory
without `config.json`, keyed on `os.sep in model_name`. A Hub id (`Qwen/Qwen2.5-0.5B-Instruct`)
also contains a slash, so `--dry-run sanity` (step 3 of setup) died with my own error
message. **[fix]** key on `os.path.isabs()` / a leading `.`/`~` instead. Setup relaunched
a third time. Cost: ~2 minutes each. Lesson: a guard that runs inside someone else's gate
needs to be tested against the inputs *that* gate uses, not just the new one.

---

## 4. Run log

_(appended as things happen)_

### 4.0 [decision] Plan change (Harpreet, mid-session): retrain SFT on all 1,210 first

The shipped adapter trained on 418 tier-A trajectories because collection was still
running when that SFT started; 1,210 tier-A are now on the Hub (2.9x). Harpreet's call:
retrain on the full set first, RL from *that*, push the new adapter to a NEW Hub repo
(`harpreet22happy/deep-research-agent-sft-full`) and leave `deep-research-agent-sft`
untouched. During RL, treat the first ~30 steps as an explicit checkpoint: is it doing
anything at all (turns/reads holding, capped-rate falling, outcome rising) before the
hours are spent.

Sequence: `validate.py` ($0, re-derives every tool call) → `build_sft.py --tiers A` →
`sft_train.py --batch-size 2 --grad-accum 8 --epochs 2 --run-name sft_full` (2 epochs, not
3: the 418-run's val loss flattened after epoch 1) → `eval_sft.py --split sft_dev --n 128`
→ push → `merge_sft.py --adapter distill/runs/sft_full/final` → probe → RL.

**[speculation]** 2.9x more successes should raise correct/cite numbers a few points and
may reduce the never-commit rate somewhat (less memorisation, more varied stopping
points), but it cannot fix F12 on its own — every example still ends in a confident
answer. That stays RL's job. The held-out number for this new adapter is measured ONCE,
alongside the RL result, at the end.

### 4.1 [decision] Distillation vs rejection sampling — an A/B on the SFT data, Harpreet's framing

> "even in distillation literature, I don't think they care whether teacher is generating
> correct response or not, they just train — offline KD or on-policy distillation"

Right: sequence-level KD / GKD / MiniLLM imitate the teacher's distribution; filtering on
correctness is the rejection-sampling family (STaR, ReST, RFT). The plan's strict tier-A
gate mixed the two without saying so, and it discarded 70% of the demonstrations — the
hard questions, where "commit to an answer anyway" is actually demonstrated (the 418- and
1,210-example adapters never see it; that is F12's cause).

The one thing the correctness gate did FOR us by accident: 21% of teacher episodes cite a
passage they never opened, and read-before-cite is what this stage exists to install (F1).
So the new recipe filters on **process**, not outcome. Measured on the 4,000:

| filter | kept | of which wrong answer |
|---|---|---|
| tier A (correct AND cite_f1=1.0) | 1,210 | 0 |
| **tier P** (reached an answer, cited ≥1, cited ONLY what it read; every turn graded) | **2,692** | **564** |
| dropped by P | 1,105 cite-without-read · 189 never answered · 13 uncited · 1 API give-up | |

**[decision]** A/B: train `sft_A1210` and `sft_P2692`, both 2 epochs, compare on
`sft_dev` (n=128, greedy) with `eval_sft.py`; RL starts from the winner. Both adapters get
their own new Hub repos; `deep-research-agent-sft` (the 418 one) is not touched.
Pre-registered expectation **[speculation]**: P lowers the capped rate (it demonstrates
concluding on hard questions) and may cost a point or two of correctness on easy ones by
imitating 564 wrong answers; if P's capped-rate drop is real the trade is worth it, since
those episodes score zero today.

### 4.2 [result] Environment rebuilt — third launch of setup_pod.sh succeeded
~22 min wall-clock from bare pod (after bugs #7/#8 above): flash-attn compiled with ~35
parallel nvcc jobs (ninja + MAX_JOBS working), zero `Killed` lines, `torch 2.11.0+cu128 /
vllm 0.22.1 / verl 0.9.0 / flash_attn / rllm` all import with CUDA available.
`distill/run_sft_ab.sh` launched 19:50 — tests → sanity → validate → build A/P → train
both → eval both on sft_dev, unattended.

### 4.3 [result] `train_dr.py sanity` PASSES on this pod (first time since the SFT stage)
Real GRPO step end to end: `step:1` with the full `reward_components/*` block INCLUDING the
five new counters (`n_reads 2.0, n_searches 1.5, n_tool_calls 4.0, answered 0.5, capped
0.5` on the 0.5B stand-in), `[batch-safety-patch]` fired as designed, `step:2` final
validation, zero tracebacks. 77/77 tests.

### 4.4 [issue] Bug #9 — my driver killed itself right after the passing sanity spike
`grep -c traceback | xargs -I{} stamp ...` — `stamp` is a bash *function*; `xargs` execs a
binary and cannot see it → exit 127 → `set -euo pipefail` ended the chain. Found when
Harpreet asked "status?" and the GPU was idle. **[fix]** plain variable + `|| true`; added
`SKIP_TO=3` so the relaunch skips the two stages that already passed. Cost: ~35 min of
idle A100. Lesson (same as #8): a driver that runs unattended must be dry-run past every
line, not just past the lines that do the work.

### 4.5 [issue] Bug #10 (pre-existing) — `validate.py` reloaded the task pool per record
`check_grounding` called `data.load_tasks(cfg, "train")` inside the per-record loop: a
2,048-question draw (with two dataset shuffles) rebuilt 4,000 times. 19 minutes in and
not done. Worse, that draw is `cloud_preset`'s train sample, not the `sft_collect` split
the records came from, so most lookups would have ended in the soft
`task_not_found_in_pool` warning — the grounding re-derivation was largely a no-op on
this data. Never noticed because it had only been run on small early batches.
**[fix]** load `splits.get_split("sft_collect")` once into a `task_id -> DRTask` dict.
This is also the first time the full 4,000 get their tool calls genuinely re-derived.

### 4.6 [issue → result] Bug #11, then validate PASSES on all 4,000 for the first time
With bug #10 fixed, grounding ran on real records for the first time and hit an
`IndexError`: two teacher turns have `Action:` mid-line rather than at a line start, so
the "first line starting with Action:" lookup returned "" and the split indexed past the
end. **[fix]** skip + soft-warn (`action_not_at_line_start`, 2 records).

**[result]** `PASS — 4000/4000 records are structurally sound, grounded in the real corpus,
and SFT-encodable with correct masking.` Every stored search/read observation is
byte-identical to what the corpus returns today (no invented retrieval anywhere), 0
prefix mismatches, mean seq 1,705 tokens (max 3,381), 4.55 graded spans/episode, 10.3% of
tokens graded. Two answers longer than 12 words. This is the first time the full
collection was genuinely re-derived; earlier runs of validate.py were on small batches
and, per bug #10, mostly skipping the grounding check.

Note for the P build: `--max-len 3072` will drop the handful of episodes above it
(max is 3,381); batch 2 was tuned at 2,240 tokens, so the driver's OOM fallback to
batch 1 x accum 16 may fire on the P run.

### 4.7 [result] The two SFT datasets
| | A (tier A) | P (process-clean, all turns graded) |
|---|---|---|
| examples | 1,209 (1 over max_len) | 2,682 = 2,124 correct + 558 wrong (10 over max_len) |
| seq len p50 / p99 / max | 1,067 / 1,842 / 2,292 | 1,031 / 1,904 / 2,292 |
| graded fraction | 0.173 | 0.168 |
Both re-rendered under the student prompt (no worked example, no teacher suffix), masking
asserted per example, zero violations. Training A then P, 2 epochs each, batch 2 x accum 8.

### 4.8 [result] SFT step time (protocol: measure before committing hours)
A run: ~5.3 s per optimizer step (batch 2 x accum 8 = 16 sequences), 144 steps → ~13 min;
47 GB resident at 100% util, matching `BATCH_SIZE_TUNING.md`'s batch-2 figure. P run
projected 336 steps → ~30 min. The RL probe's `timing_s/step` is measured the same way
before the 6h run is sized.

### 4.9 [decision] Naming — the tier letters are retired
Harpreet: "what is P, try to use better names". From here on:

| old | new | what it is |
|---|---|---|
| tier A / `sft_A1210` / `sft_dataset_A.pt` | **`sft_correct_only`** | rejection-sampled: the 1,209 teacher episodes that were correct AND perfectly cited (same recipe as the shipped adapter, 2.9x the data) |
| tier P / `sft_P2692` / `sft_dataset_P.pt` | **`sft_imitate_all`** | distillation-style: all 2,682 teacher episodes with clean process (reached an answer, cited ≥1, cited ONLY what it read); wrong answers included and graded |

Run dirs, eval JSONs and Hub repos are renamed after the running chain exits (a bash
script is read incrementally, so it is not edited mid-run). Hub repos:
`harpreet22happy/deep-research-agent-sft-correct-only` and `...-sft-imitate-all`;
`deep-research-agent-sft` (the 418 one) is not touched.

### 4.10 [result] SFT training runs — both 2 epochs, LoRA r=32/alpha=64, lr 1e-4 cosine, batch 2 x accum 8

| run | examples | opt. steps | wall | val loss ep0 → ep1 |
|---|---|---|---|---|
| shipped adapter (2026-08-26, 3 epochs) | 418 | — | ~7 min | 0.2722 → 0.2565 → 0.2563 |
| `sft_correct_only` | 1,209 | 144 | ~14 min | **0.2491 → 0.2408** |
| `sft_imitate_all` | 2,682 | 336 | ~29 min | 0.2812 → 0.2692 |

Reading: correct_only's val loss is below the shipped adapter's at every epoch and still
falling at epoch 2 — the extra data helps. imitate_all's val loss is NOT comparable to
either: each run holds out its own 5% slice, and imitate_all's slice contains the hard
questions and wrong answers, which are inherently less predictable. Loss comparisons
across these two datasets say nothing; only the generation eval does.

### 4.11 [result] `sft_correct_only` on `sft_dev` — n=128, greedy, one vLLM engine, 3 arms

| metric | base (no example) | base + worked example | **correct_only** |
|---|---|---|---|
| correct (exact match) | 0.039 | 0.375 | **0.625** |
| picked the right sources (title_f1) | 0.010 | 0.305 | **0.773** |
| verified them (read before cite) | 0.000 | 0.203 | **0.859** |
| citation-F1 (the reward) | 0.000 | 0.121 | **0.770** |
| called `read` at least once | 0.039 | 0.391 | **0.984** |
| finished cleanly | 0.570 | 0.922 | 0.875 |
| never used a tool | 0.898 | 0.133 | **0.000** |
| **correct AND properly cited** | 0.0% | 0.8% | **48.4%** |
| answer length (words) | 1.94 | 1.37 | 1.61 |
| citations per answer | 0.27 | 0.95 | 1.67 |
| tool calls per episode | 6.08 | 3.18 | 5.27 |
| reads per episode | 0.06 | 0.61 | 2.16 |
| hit the 8-call cap | — | 5/128 | **17/128 = 13.3%** |

Pre-registered SFT gate: PASSED (all five). Hop distribution for tuned:
`{3: 2, 4: 23, 5: 77, 6: 7, 7: 2, 8: 17}` — peaks at 5 = search→read→search→read→answer.

**Versus the shipped 418-example adapter on the same split (n=96, RESULTS.md §2):**
correct 0.594 → 0.625, read-before-cite 0.807 → 0.859, cite_f1 0.714 → 0.770,
correct-and-cited 44.8% → 48.4%, capped 16.7% → 13.3%. Consistent gains of 3–6 points
across the board from 2.9x data, well inside what F8's noise floor (~5pp at n=64) allows
as real at n=128 for the aggregate pattern, though no single line is individually
significant. The never-commit failure is still there at 13%: as predicted in 4.0, more
successes shrink it but cannot remove it, because every example still ends in a
confident answer.

### 4.12 [result] `sft_imitate_all` on `sft_dev` (n=128, greedy) — and the head-to-head

| metric | base | base + example | **imitate_all** | correct_only (4.11) |
|---|---|---|---|---|
| correct (exact match) | 0.023 | 0.398 | **0.656** | 0.625 |
| picked the right sources (title_f1) | 0.014 | 0.289 | 0.720 | **0.773** |
| verified them (read before cite) | 0.000 | 0.176 | **0.875** | 0.859 |
| citation-F1 (the reward) | 0.000 | 0.111 | 0.712 | **0.770** |
| called `read` at least once | 0.023 | 0.383 | 0.992 | 0.984 |
| finished cleanly | 0.516 | 0.938 | **0.906** | 0.875 |
| **correct AND properly cited** | 0.0% | 0.8% | 33.6% | **48.4%** |
| correct but partially cited | 2.3% | 28.9% | 31.3% | 14.1% |
| hit the 8-call cap | — | 3/128 | **13/128 = 10.2%** | 17/128 = 13.3% |
| tool calls / reads / citations per episode | 6.07 / 0.03 / 0.23 | 3.15 / 0.59 / 0.90 | 4.89 / 1.74 / 1.48 | 5.27 / 2.16 / 1.67 |
| hop distribution | | | `{3:17, 4:33, 5:55, 6:6, 7:4, 8:13}` | `{3:2, 4:23, 5:77, 6:7, 7:2, 8:17}` |

Both pass the pre-registered SFT gate. Capped-episode breakdown, both adapters: capped
episodes are 0.000 correct and only 6–8% of them answer at all; among episodes that
finish, correct_only is 0.721 correct / 0.878 cite_f1 and imitate_all 0.730 / 0.788.
Per-question: 76 questions both get right, 4 only correct_only, 8 only imitate_all, 40
neither — the two adapters mostly agree; the difference is at the margin.

**Analysis.** The 4.1 prediction was half right. imitate_all DOES commit more (capped
13.3% → 10.2%, finished-cleanly 0.875 → 0.906, +3 correct) — the 558 wrong-but-concluded
examples teach "answer with what you have". But the cost landed on citation
*completeness*, not correctness: correct-but-partially-cited went 14% → 31% and cite_f1
fell 0.770 → 0.712. Cause, in the data: the process filter required "cited ⊆ read" but
not "cited ⊇ gold", so the teacher's 1,344 `correct_miscited` episodes (typically one of
two gold passages cited, cite_recall 0.5) were admitted, and the student learned to read
less (2.16 → 1.74 reads) and cite less (1.67 → 1.48). Distillation imitates the teacher's
*whole* behaviour, including the half-citing.

**[decision] RL starts from `sft_correct_only`.** It wins on the project's headline
(correct-and-properly-cited 48.4% vs 33.6%), on the reward's own number (cite_f1 0.770 vs
0.712; mean `cite_gated` reward on dev 0.597 vs 0.593 — essentially tied, so the reward alone would not separate them; the headline and citation metrics do), and its failure mode (13%
never-commit, zero reward) is precisely the contrast GRPO is being pointed at.
imitate_all's failure (partial citation) is also RL-addressable (cite_f1 varies within a
group), but starting from the higher citation bar and asking RL to fix commitment is the
cleaner experiment. **[speculation]** a third recipe — "perfectly cited regardless of
correctness" (correct_only's citation gate with the correctness gate dropped, adding the
wrong-but-fully-cited teacher episodes) — would likely get imitate_all's commitment gain
without its citation cost; it is cheap (build + 15 min train) and is the natural next SFT
experiment if RL's commitment gain turns out small. Not run today.

Both adapters pushed to their own private repos with generated cards:
`harpreet22happy/deep-research-agent-sft-correct-only`, `…-sft-imitate-all`.
`deep-research-agent-sft` (418) untouched. Training curves copied to
`distill/run_histories/`. The sft_dev eval JSONs are committed as
`distill/eval_sft_{correct_only,imitate_all}_dev.json`.

### 4.13 [issue → result] Merging the adapter: the logit check, and what actually decides it
`distill/merge_sft.py` (correct_only adapter → `sft_merged/`, 5.76 GiB) compares the
merged-from-disk model's logits against the un-merged bf16 base + bf16 adapter on a real
student prompt (326 positions × 152k vocab): max |Δlogit| 2.125, mean 0.066, argmax
agreement 0.991. My first threshold (max < 2.0) called that a FAIL; redoing the merge in
fp32 and casting once gave the *identical* numbers, so the delta is not merge-rounding
error but the intrinsic difference between "one bf16 weight" and "bf16 weight + bf16
low-rank product applied separately" — a tail statistic over 50M logits. Threshold
loosened to what it should have been (argmax ≥ 0.98, max < 4), and the DECIDING check is
behavioural: `eval_rl.py` with no adapter on the merged engine must reproduce the
adapter's `eval_sft.py` numbers on the same 128 dev questions (4.14).

### 4.14 [result] Merge confirmed behaviourally — and this row is the RL baseline
`eval_rl.py`, merged weights, no adapter, same 128 dev questions, greedy, same stop seqs:

| | adapter via eval_sft.py | merged via eval_rl.py |
|---|---|---|
| correct | 0.625 | 0.633 |
| title_f1 / read-before-cite / cite_f1 | 0.773 / 0.859 / 0.770 | 0.766 / 0.840 / 0.762 |
| correct AND properly cited | 0.484 | 0.484 |
| hit the turn cap | 17/128 (0.133) | 19/128 (0.148) |
| turns / reads / citations | 5.27 / 2.16 / 1.67 | 5.31 / 2.20 / 1.64 |

Within 1–2 points everywhere (greedy decoding flips a few tail decisions when the weights
differ at the 3rd significant digit). Merge is good. **Every RL checkpoint is compared
against the `merged, no adapter` column** — same engine, same weights, same prompt —
so the comparison cannot drift on anything but the RL LoRA.

### 4.15 [run] Probe: `python train_dr.py probe_rl_from_sft` — 12 steps, checkpoint at 10
What it must answer before the 6h run: does the merged model load in vLLM + FSDP; do the
stop sequences apply (`response_length/clip_ratio` should be low, and no fake
"search results:" in completions); is `actor/kl_loss` logged and how big vs `pg_loss`;
`timing_s/step` at ~5 turns/episode; and — the thing to read — do
`reward_components/n_steps`, `n_reads`, `answered`, `capped` hold or drift within 12 steps.

### 4.16 [issue] Bug #12 — an integer passage title broke the verl parquet write
First probe launch died in `write_verl_dataset`: `ArrowInvalid: Could not convert
'Jacques Cassini' with type str: tried to convert to int64` for `extra_info`. Scanned
the 14,500 `rl_train` rows field by field: exactly ONE passage title (of 144,645) is an
`int` — a 2Wiki context whose title is a bare year — and pyarrow's struct inference locked
onto it. The historical 2,048-question draw never contained it. The same title would also
never match a `read[<title>]` string lookup. **[fix]** `str()` on titles/text in both
loaders (`data.py`) and defensively in `train_dr._task_to_extra_info`.
Verified the fix directly: `write_verl_dataset` on `rl_train` (14,500 rows → 38 MiB
parquet) and `sft_dev[:128]` both succeed. (A relaunch-watcher then reported the OLD
run's traceback because my launch line had backgrounded `pytest && nohup train …` as one
unit, so the new process only started after the 90 s split tests — not a new failure.)
Probe relaunched 22:09 (PID 27623).

---

## 5. Issues register — every defect hit this session, for future sessions

Numbering continues `FRESH_POD_SETUP_AND_SANITY_CHECK.md`'s table (#1–#6 were the
2026-08-25/26 fresh-pod bugs). "Self" = I introduced it this session. Each row has the
symptom you would actually see, the real cause, the fix, and the transferable lesson.

| # | where | symptom | root cause | fix | lesson |
|---|---|---|---|---|---|
| **7** | `setup_pod.sh` step 3 | `45 passed, 6 errors` then the script silently stops (`set -euo pipefail`) | `tests/test_splits.py` imports `dotenv`, installed only in step 4 | `python-dotenv` in step 2 | Same class as #3/#4/#6: a dependency that lands *after* the first thing needing it. On a fresh pod, run the gates in the order the script runs them, not the order you think of them. |
| **8** (self) | `train_dr.py` guard | setup dies again at `--dry-run sanity` with *my own* error message ("local path has no config.json") | guard keyed on `os.sep in model_name`; Hub ids (`Qwen/Qwen2.5-…`) contain a slash too | key on `os.path.isabs()` / leading `.`/`~` | A guard that sits inside someone else's gate must be tested against the inputs *that gate* uses, not only the new case it was written for. |
| **9** (self) | `distill/run_sft_ab.sh` | GPU idle for 35 min; log ends `xargs: stamp: No such file or directory` right after a PASSING sanity spike | `grep … \| xargs -I{} stamp …` — `stamp` is a bash *function*; `xargs` execs a binary and cannot see it → exit 127 → `set -e` | plain variable + `\|\| true`; `SKIP_TO=N` to resume | An unattended driver must be dry-run past *every* line, including the cosmetic ones. `bash -n` catches syntax, not this. Watch the GPU, not just the log. |
| **10** (pre-existing) | `distill/validate.py` | "validate" runs 19+ min single-core with no output; earlier sessions thought it was fine | `check_grounding` called `data.load_tasks(cfg,"train")` **per record** (a 2,048-question draw + two shuffles, 4,000 times) — and that draw is not the `sft_collect` split, so lookups mostly fell through to a soft warning: the grounding check was largely a no-op on this data | load `splits.get_split("sft_collect")` once into a dict | A check that has only ever run on small batches has not been validated at scale. "PASS" from a check that silently skips its own core step is worse than no check. Look at CPU time and *what* it is doing, not just that it is running. |
| **11** (pre-existing, unmasked by #10) | `distill/validate.py` | `IndexError: list index out of range` in `check_grounding` | two teacher turns have `Action:` mid-line, not at a line start → the line lookup returns `""` → `.split(...)[1]` | skip + soft warning | Fixing a check that was skipping work will surface the bugs it was hiding. Budget for that. |
| **12** | `train_dr.write_verl_dataset` | `ArrowInvalid: Could not convert 'Jacques Cassini' with type str: tried to convert to int64` for column `extra_info` | ONE of 144,645 passage titles in `rl_train` is an `int` (a 2Wiki context titled with a bare year); pyarrow's struct inference locked onto it. Never seen on the 2,048-question historical draw | `str()` on titles/text in both loaders and in the writer | Nested-struct inference is only as robust as the weirdest row. Type-scan every field before trusting a schema inferred from data (the one-off scan script is in §4.16). Also: an `int` title could never match `read[<title>]` — a data bug, not just a serialization bug. |
| **13** (self, ops) | probe relaunch | watcher reported the *old* run's traceback; probe looked dead but was alive | `python -m pytest … \| tail -1 && nohup python train_dr.py … &` — the `&` backgrounds the whole `&&` chain, so the probe started 90 s later, after the slow split tests, while the watcher read the not-yet-truncated old log | check `kill -0 PID` and the log's mtime/line count before believing a traceback | When relaunching, `>`-truncate the log *first* as its own command, then launch. And never chain a slow foreground step in front of the thing you background. |
| **14** (self, thresholds) | `distill/merge_sft.py` | `FAIL` on a merge that was actually fine | I set max \|Δlogit\| < 2.0 by feel; the true value is 2.125 with mean 0.066 and 99.1% argmax agreement, and an fp32 merge gives the identical number — it is the intrinsic bf16 "merged weight vs weight + separate low-rank product" difference, a tail statistic over 50M logits | argmax ≥ 0.98, max < 4; and the *deciding* check is behavioural (§4.14) | A verification threshold invented without a reference will fire on good artifacts. Pick the metric that maps to what you care about (greedy behaviour → argmax agreement / an actual eval), not the one that is easiest to print. |

| **15** | `config.py` | `max_train_hours` looked like a budget cap | never read by any code | budget via `steps`, from a measured `timing_s/step` | A config field is a promise only if something consumes it. `grep` the consumer before trusting a knob. |
| **16** | `train_dr.main` end-of-run push | `Cannot re-initialize CUDA in forked subprocess` after training finished | vLLM engine built inside the training process whose CUDA context Ray/torch already owned | run `--push-checkpoints` in a fresh subprocess | Anything that spins up a second CUDA engine after training belongs in a new process. |
| **17** | rLLM trainer loop | 99 steps trained, no `global_step_100` checkpoint | `global_steps >= total_training_steps` is checked BEFORE the step; saves only on multiples of `save_freq` within 1..99 | `steps = wanted + 1` (or `save_freq` \| `steps − 1`) | Dry-run the finish line (`steps=2, checkpoint_every=1`) before a multi-hour run; the last checkpoint is the one you cannot re-create. |

**Non-bug things that cost time or could mislead, also worth knowing:**
- `SKIP_TO` in the driver only guarded stages 1–2; stage 3 re-ran once (2 min). Guard every
  stage when you add a resume knob.
- SFT `val_loss` is per-run (each run holds out its own 5%); it is meaningless *across*
  datasets (4.10). Only the generation eval compares recipes.
- `data.train_batch_size` is prompts/step (32 here), not 1 — the docs' "252 distinct
  questions" worry was arithmetic, not a real constraint (§1.5).
- rLLM's `val/…/pass@1` is `reward > 0`; under `cite_gated` a correct-but-uncited answer
  counts as a failure there. Read `eval_rl.py`'s numbers, not that one, for correctness.
- The Hub datasets (HotpotQA 90k, 2Wiki 167k) load from cache in ~1 s once pulled; the
  first `splits` call in any process still takes ~40 s to draw and shuffle the 20k pool.

**Operational protocol that held up:** measure the step time before committing hours
(§4.8); read the raw table, not the headline; keep every eval on the same engine / prompt /
stop sequences; one adapter per Hub repo, never overwrite a shipped one; log as you go.

### 4.17 [analysis] Did imitating the teacher's failures teach the student to conclude?

Harpreet's question: *"did training with all teacher's trajectories (even where the
answer was incorrect) help… I hope it now concludes rather than taking turns till max
budget?"* Answer: partly, and less than hoped.

| | correct_only (1,209) | imitate_all (2,682, 558 wrong answers) |
|---|---|---|
| hit the 8-call cap (never concluded) | 13.3% (17/128) | **10.2%** (13/128) |
| produced an answer | 0.875 | **0.906** |
| correct | 0.625 | **0.656** |
| tool calls / reads per episode | 5.27 / 2.16 | **4.89** / 1.74 |
| correct AND properly cited | **48.4%** | 33.6% |
| citation-F1 | **0.770** | 0.712 |
| correct but partially cited | 14.1% | 31.3% |

It commits more — four fewer capped episodes, more answers, fewer turns, +3 correct —
but at n=128 a 17→13 change is inside the run-to-run noise this project has measured
(F8: ~5pp swings at n=64). Read it as "moves the right way", not "fixed". One episode in
ten still burns the budget.

**Why it did not fully fix commitment.** The 558 wrong-answer examples teach "answer with
what you have" — but so do the 2,124 correct ones. Both recipes are still 100%
"the teacher concluded". What the student never sees is a trajectory that *searched a
lot and then concluded anyway*: its capped episodes search 2.4x more than its normal
ones (RESULTS.md §3), and no demonstration shows what to do at turn 6 or 7 with weak
evidence. The teacher, being stronger, rarely needed that many turns. That is precisely
the case RL's zero-reward-for-never-answering is aimed at, and why RL starts from
`correct_only` rather than trading 15 points of citation quality for a marginal
commitment gain.

**Why it cost citation quality.** The process filter was "cited ⊆ read", not "cited ⊇
gold". The teacher's 1,344 `correct_miscited` episodes (typically one of two gold
passages cited) came in and the student imitated the half-citing: reads 2.16 → 1.74,
citations 1.67 → 1.48. Distillation imitates the *whole* behaviour, the good and the
lazy.

**The recipe that should get both [speculation, not run]:** keep the citation gate
(cite_f1 = 1.0, i.e. every gold passage cited AND read) and drop the correctness gate,
so wrong-but-fully-cited teacher episodes join the 1,209. Small addition (the
`wrong_answer` bucket is 709 episodes, and only the fully-cited subset qualifies), cheap
to build and train (~15 min), and it targets commitment without importing the
half-citing. Try it if RL's commitment gain turns out small. Beyond that, the only way to
show the student "search a lot, then conclude" is to *collect* such trajectories
(teacher on harder questions / a turn budget), or let RL find them — which is the
current run.

### 4.18 [FINDING F18] `max_response_length` truncated every multi-turn trajectory and zeroed its reward

**What the probe showed (steps 1–3, 512 trajectories/step):** `timing_s/step` ≈ 127 s;
behaviour counters healthy (turns 5.2, reads 2.1, answered 0.92, capped 0.08, cite_f1
0.76, outcome 0.60–0.69); rollout-level rewards from the `Rollout completed` lines
average ≈ 0.55 (693 of 1,674 at 1.0, 336 at 0.8, 480 at 0.0). But
`critic/rewards/mean` = 0.035–0.083, `response_length/mean` ≈ 232 with
`response_length/max` = 256 and `clip_ratio` 0.85–0.89, and `perf/total_num_tokens`
370k for 512 rows (≈ 724 tokens/row) when a real SFT trajectory is ≈ 1,700 tokens.

**The mechanism, from `rllm/trainer/verl/transform.py` at the pinned commit:**
1. `_process_trajectory` merges a cumulative multi-turn trajectory into ONE row whose
   `response` = `[action0, obs1, action1, obs2, …, answer]` with mask 1 on actions, 0
   on observations (the masking is right — that part was verified in August).
2. `_batch_tensors_and_build_data_proto` pads *and right-truncates* that response to
   `data.max_response_length` (`_pad_sequence_batch`: `batch[:, :max_length]`).
3. `_build_step_and_trajectory_rewards` writes the trajectory reward at position
   `resp_len - 1` **only if `resp_len <= max_response_length`**, where `resp_len` is
   the UN-truncated length. Otherwise the row's reward tensor stays all zeros.
4. `agent_workflow_trainer.py:398` sets `token_level_scores = traj_rewards`, and GRPO's
   group-relative advantage is computed from that.

We set `data.max_response_length = cfg.max_new_tokens = 256`, a PER-TURN number, from
the very first scaffold. So for this probe: ~87% of trajectories were (a) trained on
their first action plus a fragment of the first observation and nothing else — no
answer turn, no read turns — and (b) given score 0 regardless of the reward the env
computed. The only trajectories that carried their real reward were the ones short
enough to fit 256 merged tokens.

**Why this reinterprets the history (again).** Every GRPO run in
`TRAINING_HISTORY_LOG.md` (Attempts 1–4, Probes 1–4) ran with the same 256. With the
base model's 2-turn episodes, `action + search-observation + answer` fits 256 tokens
sometimes and not others — exactly Attempt 3/4's `clip_ratio` 0.16–0.68. Trajectories
that searched twice, or read a passage (a 1,200-char observation ≈ 300 tokens), could
NEVER fit, so they were always scored 0. The "eval-time turn count" finding — every
healthy run converging on exactly 2 turns, one search then answer, never a read,
regardless of reward design — is the policy learning the only shape that ever got paid.
It was treated as a prompt problem (no `read` demonstration) and then a skill problem
(F1: never reads → cannot cite); both were real, but underneath them the training
signal itself was structurally biased toward the shortest possible episode. Attempt 2's
"collapse" (`clip_ratio` 96–98%, 8 turns every episode) is the same mechanism from the
other side: once episodes got long, every row scored zero and the policy had no signal
at all. `response_length/clip_ratio` was read all along as "hitting the per-turn
token cap"; it was "the fraction of trajectories whose reward was thrown away".

**Fix:** two knobs, not one. `cfg.max_new_tokens=256` stays the per-turn generation cap
and is now passed explicitly as `max_tokens` on every LLM call (VerlEngine defaults it
to `data.max_response_length` otherwise). New `cfg.verl_max_response_length` is the
whole-trajectory budget: 4096 for `rl_from_sft` (mean ≈ 1,300 response tokens, 8-turn
worst case ≈ 3,000; vLLM `max_model_len` becomes 6144 + 4096). `None` keeps the
historical value so old presets resolve byte-for-byte. Test added. Expected on the
relaunch: `response_length/clip_ratio` ≈ 0, `response_length/mean` ≈ 1,300,
`critic/rewards/mean` ≈ the rollout mean (~0.55), `perf/total_num_tokens` ≈ 850k/step,
and `timing_s/step` up from 127 s (more tokens through `update_actor`).

**[speculation]** With the reward actually attached to every trajectory for the first
time, the GRPO signal is now "answer correctly with citations" vs "don't", instead of
"be short" vs "be long". The turn-collapse risk in START_HERE.md was, in part, this
bug's signature; it is still worth watching, because the outcome reward still pays the
same for a 3-turn correct answer as a 5-turn one.

### 4.19 [result] Probe after the F18 fix (W&B `mchhqpc6`, killed at step 4 — its questions were answered)

| | broken probe (`n2em5ok7`) | fixed probe (`mchhqpc6`) |
|---|---|---|
| `response_length/clip_ratio` | 0.85–0.89 | **0.0** |
| `critic/rewards/mean` (what GRPO actually optimises) | 0.03–0.08 | **0.53–0.62** = the rollout-level mean |
| `response_length/mean` (merged multi-turn response) | 232 (truncated at 256) | 680–704 |
| `perf/total_num_tokens` per step | 370k | 600–650k |
| `mfu` | 0.06 (sanity) → n/a | 0.50 |
| `timing_s/step` (512 trajectories, ~5.2 turns each) | 127 s | **184–191 s** (generate 35 s, update_actor 83–89 s) |
| turns / reads / answered / capped (steps 1–3) | 5.2 / 2.1 / 0.92 / 0.08 | 5.2 / 2.1 / 0.92–0.93 / 0.07–0.08 |
| outcome / cite_f1 | 0.60–0.69 / 0.75–0.77 | 0.58–0.68 / 0.77 |
| dead groups (all-right or all-wrong) | 0.38–0.44 | 0.38–0.47 |

The merged response averages ~690 tokens, not the ~1,300 I estimated in 4.18 — the
1,705 figure in validate.py's scan includes the ~380-token prompt and the teacher's
episodes are longer than the student's. 4096 is ample. Everything else moved exactly as
predicted. Per-turn stop sequences are in effect (no fake "search results:" in
completions — every rollout terminated `ENV_DONE`, 0 length terminations).

**[issue #15]** `cfg.max_train_hours` is a config field that nothing reads — grep'd
`train_dr.py`, `rllm_workflow.py`, `hub.py`: only `config.py` mentions it. Every
historical cloud run was killed by hand. Budget therefore set via `steps=100`
(≈ 5.3 h at 190 s), with checkpoint + sft_dev eval every 25.

### 4.20 [run] THE RUN — `python train_dr.py rl_from_sft` (`deep_research_agent_rl_from_sft_correct_only`)
Launched 22:37. Policy: merged `sft_correct_only` + fresh LoRA r=16/α=32; 32 prompts ×
16 rollouts = 512 trajectories/step from `rl_train`; T=0.9; `cite_gated` reward; KL loss
to the SFT reference (coef 0.01, low_var_kl); lr 5e-5; per-turn 256 tokens, trajectory
budget 4096; stop sequences on; efficiency toll effectively off (ramp starts at 150).
100 steps, checkpoints at 25/50/75/100 (LoRA-only, periodic best-effort push to
`harpreet22happy/deep-research-agent-grpo` on a timestamped branch), verl val on
`sft_dev[:128]` at the same steps (`val/pass@1` = reward > 0).

**What decides "is it doing anything" at ~step 30 (Harpreet's checkpoint):**
`critic/rewards/mean` trending up from ~0.55; `reward_components/capped` trending down
from ~0.08 and `answered` up from ~0.92 (the F12 target); `n_steps` staying ≥ ~4.5 and
`n_reads` ≥ ~1.8 (NOT sliding toward 2 / 0 — the collapse signature); `cite_f1` not
falling; `actor/kl_loss` small relative to `pg_loss`; `dead_groups_pct` not climbing
toward 0.8+. Step-25 `val/pass@1` vs the SFT policy's own would-be value (≈ the fraction
of dev episodes with reward > 0 ≈ 0.55–0.6) is the first held-out read.

### 4.21 [result] The run, steps 1–4 (W&B `t2n91x71`)
| step | s/step | `critic/rewards/mean` | outcome | cite_f1 | turns | reads | answered | capped | dead groups | kl_loss |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 186 | 0.578 | 0.656 | 0.750 | 5.23 | 2.15 | 0.906 | 0.094 | 0.50 | — |
| 2 | 186 | 0.560 | 0.602 | 0.771 | 5.09 | 2.12 | 0.934 | 0.066 | 0.47 | — |
| 3 | 191 | 0.503 | 0.607 | 0.748 | 5.19 | 2.13 | 0.914 | 0.086 | 0.34 | — |
| 4 | 202 | 0.613 | 0.658 | 0.742 | 5.20 | 2.12 | 0.908 | 0.092 | 0.34 | 0.0013 |
`response_length/clip_ratio` = 0 at every step (F18 fix holding); `actor/kl_loss`
0.0013 at coef 0.01 vs `pg_loss` ≈ −0.45 — the KL term is on and not dominating;
entropy ≈ 0.26; GPU 66 GB / 90%. Four steps is noise; the step-25 checkpoint + sft_dev
val (~23:57) and the 30-step read (~00:15) are the first real signal.

### 4.22 [result] How many groups are actually dead — the logged metric overstates it
`batch/dead_groups_pct` = `solve_none + solve_all` where "solved" is reward > 0. It was
built (2026-08-24) for an effectively 0/1 outcome reward. Under `cite_gated` a group whose
16 rollouts are all "solved" still ranks them (1.0 fully cited, 0.5–0.9 partial), so
"all-solved" is usually NOT zero-advantage. Recomputed from the per-episode logs
(`runs/…/episode_logs/episodes/train_step_N`), 32 groups × 16 rollouts:

| step | solve_none / solve_all / partial | logged dead % | **zero-variance groups** | std < 0.1 | median within-group std |
|---|---|---|---|---|---|
| 1 | 9% / 41% / 50% | 50% | **7 (22%)** | 8 | 0.16 |
| 2 | 16% / 31% / 53% | 47% | **5 (16%)** | 10 | 0.18 |
| 3 | 16% / 19% / 66% | 34% | **5 (16%)** | 3 | 0.30 |
| 4 | 3% / 31% / 66% | 34% | **4 (12%)** | 7 | 0.22 |
| 5 | — | — | **6 (19%)** | 5 | 0.24 |

So ~4–7 of 32 groups per step contribute nothing and ~20 carry a real contrast — far
healthier than the 56–84% the historical runs recorded (which were themselves inflated
by F18 zeroing rewards on long trajectories). **[todo, after this run]** log a true
`batch/zero_variance_groups_pct` (and median within-group reward std) in
`_patch_tracking_log_for_extra_metrics`; cannot be changed mid-run.

### 4.23 [result] Steps 1–24 — the first real trend (training-batch metrics, 32 questions/step)
| | steps 1–10 | steps 15–24 |
|---|---|---|
| `critic/rewards/mean` | 0.582 | **0.686** |
| outcome (correct) | 0.646 | **0.740** |
| cite_f1 | 0.762 | **0.813** |
| answered | 0.926 | **0.953** |
| capped (never concluded) | 0.074 | **0.047** |
| reads / turns per episode | 2.10 / 5.08 | 2.15 / 4.99 |
| entropy | ~0.26 | ~0.27 |
| s/step | 186–202 | 172–193 |

Every reward-side line up, the F12 rate roughly halved, and reads/turns flat — i.e. the
gain is NOT "answer sooner, read less" (the collapse signature), it is "conclude
instead of looping, and cite what you read more completely". These are on-policy
training batches at T=0.9, not held-out greedy numbers; the step-25 `val/pass@1` on
`sft_dev[:128]` and the later `eval_rl.py` runs are the honest read.

### 4.24 [result] Step 25 — first held-out read, and the checkpoint chain proven
- **`val/pass@1` = 0.633** on `sft_dev[:128]`, greedy (verl's val is `do_sample=False`).
  `pass@1` here = fraction with reward > 0 = correct AND ≥1 citation. The SFT policy's
  greedy correct rate on the same 128 questions (4.14, merged engine) is 0.633 — so the
  held-out greedy number is at PARITY with SFT at step 25, while on-policy training
  metrics at T=0.9 rose (4.23). Two readings, not yet separable: (a) RL is first
  tightening the sampled distribution (fewer loops, more commits at T=0.9) before it
  moves the greedy argmax; (b) 25 steps × 32 questions is too little. The full greedy
  breakdown (capped, cite_f1, reads) of the step-25 adapter needs a vLLM engine and the
  training run holds 62 GB, so it waits for the run to end.
- **Checkpoint chain proven on a real RL checkpoint:** verl LoRA-only checkpoint
  (`global_step_25/actor`, `save_lora_only`) → `hub.merge_checkpoint` on CPU
  (`CUDA_VISIBLE_DEVICES=""`) → 57 MB adapter, r=16/α=32, all 7 projections; the known
  non-zero-exit quirk of `verl.model_merger` handled as documented. The periodic-push
  poller pushed it to `harpreet22happy/deep-research-agent-grpo` (`checkpoint/` on this
  run's timestamped branch) within a minute.

### 4.25 [analysis] Why 190 s/step and not the tuned 85.7 s — Harpreet's challenge
Same rLLM/verl path and the same tuning (dynamic bsz, mini-batch 256, 65k-token chunks,
Liger). Per-phase, steps 20–24 vs `ONE_STEP_TUNING_VERL_RLLM.md`'s steady state:

| phase | 85.7 s run (base, 2-turn) | now (SFT policy, ~5-turn) | why |
|---|---|---|---|
| generate_trajectories | 11.3 s | 31–36 s | 5 sequential rounds/episode, not 2 |
| old_log_prob | 18.6 s | 33–37 s | ~linear in tokens |
| **ref log-prob** | **0** (no KL term, F22) | **26–29 s** | new: KL loss needs a reference forward |
| update_actor | 42.1 s | 86–95 s | ~linear in tokens |
| update_weights | ~4 s | 4.6 s | |
| tokens/step | ~290k | 620–690k | F18: old rows were cut at 256 response tokens; now ~700 mean, real 5-turn episodes |
| **tokens/second** | **~3.4k** | **~3.4k** | identical throughput; MFU 0.50 |

So 190 s is what an un-truncated 5-turn trajectory costs at this throughput. Levers if
more steps are wanted: drop the KL pass (−27 s; kept for this first run), or
`grad_accum=1` (256 traj/step, ~100 s, 2x steps at half the batch).

### 4.26 [result] The historical runs, pulled from W&B — F18 confirmed with their own numbers
Harpreet: "you have the W&B key, fetch previous logs". Means over steps ≥ 2 per run
(`wandb.Api().run(...).scan_history`):

| run | s/step | generate | old_log_prob | ref | update_actor | tokens/step | response mean | clip_ratio | `critic/rewards/mean` (what GRPO saw) | `reward_components/base` (what env computed) | turns |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Attempt 1 `njpxc1cn` | 83.6 | 12.7 | 18.4 | – | 43.0 | 302k | 214 | 0.40 | 0.080 | – (not logged yet) | – |
| Attempt 3 `vgzv9byy` | 80.4 | 10.1 | 18.0 | – | 42.4 | 297k | 211 | 0.30 | 0.134 | 0.178 | 2.37 |
| Attempt 4 `qyzrepqk` | 78.0 | 7.7 | 18.1 | – | 42.6 | 293k | 211 | 0.30 | 0.134 | 0.183 | 2.09 |
| Probe 4 `wdd1i4na` | 80.0 | 9.7 | 18.0 | – | 42.6 | 298k | 214 | 0.38 | 0.077 | 0.145 | 2.22 |
| **now `t2n91x71`** | 188 | 33.1 | 33.5 | 27 | 88.2 | 639k | 677 | **0.00** | **0.630** | **0.644** | 5.06 |

1. Same pipeline, same throughput: `update_actor` 42 s / 300k tokens then, 88 s / 640k
   now — identical tokens/s. The extra `ref` 27 s is the KL reference pass the old
   runs never had (F22).
2. F18, retroactively: 30–40% of every historical run's trajectories were truncated,
   and GRPO optimised only 53–75% of the reward the environment computed (0.134 vs
   0.178, 0.077 vs 0.145). Now the two agree to within the tolls (0.630 vs 0.644).
3. "211 response tokens" and "2.1–2.4 turns" are the same fact: the only shape that
   fit under 256 tokens, and therefore the only shape that was ever fully rewarded.

---

## 6. This run vs every earlier agentic-RL run in this lab — the differences, in one place

Harpreet asked for this to be written down explicitly (2026-09-08). Everything below is
measured or read from source; the "earlier" column is Attempts 1–4 / Probes 1–4
(2026-08-24), W&B numbers from §4.26.

| | earlier GRPO runs (Aug 24) | this run (`rl_from_sft_correct_only`, Sep 7–8) |
|---|---|---|
| **starting policy** | base Qwen2.5-3B-Instruct, fresh LoRA r=16 | base with the `sft_correct_only` adapter **merged in**, fresh LoRA r=16 on top (step 0 == the SFT policy) |
| **prompt** | with the worked multi-hop example (~800 tokens) | example-free student prompt (the one SFT trained under) |
| **what the policy does at step 0** | 1 search then answer, never reads (2.1–2.4 turns) | search→read→search→read→answer (~5 turns, 2.1 reads), cite_f1 0.76 |
| **training questions** | `cloud_preset` draw of 2,048 from the train split | `rl_train`: 14,500 questions disjoint (asserted) from everything SFT saw |
| **early-stop / val set** | 256 from the validation split | `sft_dev[:128]`; `heldout_eval` untouched until the end |
| **trajectories per step** | 512 (32 prompts × 16) | 512 (32 × 16) — same |
| **response budget** | `data.max_response_length=256` — meant per turn, applied to the MERGED trajectory (F18) | 4096 for the merged trajectory; per-turn 256 passed explicitly as `max_tokens` |
| **trajectories truncated per step** | 30–40% (`clip_ratio`) | 0% |
| **share of env-computed reward that reached GRPO** | 53–75% (`critic/rewards/mean` vs `base`) | ~98% (0.630 vs 0.644; the rest is tolls) |
| **assistant turns inside the training window** | first turn, sometimes a second | all of them |
| **tokens through the optimizer per step** | ~300k | ~640k |
| **reward mode** | gated / additive / cite_gated (four probes) | `cite_gated` (hard 0 for no citation; 0.5–1.0 scaled by citation-F1; a citation only scores if the passage was `read`) |
| **KL to a reference** | none (`kl_coef` mapped to an unused key, F22) | `use_kl_loss=True`, low_var_kl, coef 0.01, reference = the merged SFT weights |
| **stop sequences in rollouts** | none (a model free-running past `Action:` had its fake tool output graded) | `\nThought:`, `\nsearch results:`, `\n[` — same as every eval |
| **efficiency toll** | ramp 150→230 (never reached) | same, effectively off |
| **lr** | 1e-5 (flat) / 1e-4 (collapse) / 5e-5 (healthy) | 5e-5 |
| **behaviour logging** | outcome, groundedness, cite terms, tolls, dead-groups | + turns, reads, searches, answered, capped per step |
| **s/step** | 78–84 | ~188 (same tokens/s; 2.1x tokens + a 27 s reference pass) |
| **run length** | killed by hand at 25–38 steps | 100 steps, ckpt + val every 25 (`max_train_hours` was never enforced, F21) |
| **step-25 held-out greedy `val/pass@1`** | 0.23–0.31 (base-model runs) | 0.633 (parity with the SFT policy it started from) |
| **what the run is asking RL to do** | invent read-then-cite from sparse reward (it never did) | sharpen commitment on a policy that already reads and cites 84–92% of the time |

The three items in bold-face type below are the ones that change what the earlier
runs' conclusions mean:
- **F18** — the earlier runs rewarded short trajectories and zeroed long ones. Their
  reward-design results (beta ramps, additive vs gated, cite_gated) were measured on a
  signal structurally biased toward 2-turn episodes.
- **F1** — the earlier policies never read, so a citation was unscoreable; groundedness ≈ 0
  was a measurement of that, not of citation skill.
- **F22** — no KL term existed; "ppo_kl looked stable" in those logs was the PPO clip
  statistic, not a KL penalty.

### 4.27 [build] Out-of-distribution eval set added while the run trains — MuSiQue
Harpreet: "are there any other datasets on which we can do generalisation testing?" →
`GENERALIZATION_EVAL.md`. Loader (`data.musique_row_to_task` / `_load_musique`),
`splits.get_split("musique_dev")` (300 from MuSiQue validation, `eval_seed`),
`--split musique_dev` in both eval scripts, per-source and per-hop slices in
`eval_rl.py`, offline tests. Measured: 20 passages/question (vs 10), 2.73 gold, hops
2/3/4 = 141/98/61, and BM25 recall@3 of gold from the raw question **0.437 vs 0.705**
on heldout — one search is not enough there, so it tests the re-search-after-reading
process rather than recall. Reported once, after `heldout_eval`; never selected on.

### 4.28 [result] Steps 1–34 by 10-step window (training batches, T=0.9)
| steps | reward | correct | cite_f1 | answered | capped | reads | turns | entropy |
|---|---|---|---|---|---|---|---|---|
| 1–10 | 0.582 | 0.646 | 0.762 | 0.926 | 0.074 | 2.10 | 5.08 | 0.264 |
| 11–20 | 0.668 | 0.721 | 0.795 | 0.942 | 0.058 | 2.13 | 5.02 | 0.264 |
| 21–30 | 0.644 | 0.701 | 0.795 | 0.937 | 0.063 | 2.23 | 5.12 | 0.302 |
| 31–34 | 0.640 | 0.683 | 0.809 | 0.922 | 0.078 | 2.31 | 5.17 | 0.323 |
A real jump in the first ~20 steps, then a plateau at reward ≈ 0.64–0.67; cite_f1 still
creeping up and reads slowly RISING (2.10 → 2.31) — the opposite of the collapse
signature. Entropy drifting up 0.26 → 0.32, to watch. **[speculation]** the plateau is
where the KL term and the lr start to balance; step 50's `val/pass@1` (greedy, held-out)
decides whether the jump moved the argmax policy or only the sampled one (cf. 4.24's
parity at step 25).

### 4.29 [result] Steps 1–97 by 10-step window (training batches, T=0.9) — the whole run
| steps | reward | correct | cite_f1 | answered | capped | reads | turns | entropy |
|---|---|---|---|---|---|---|---|---|
| 1–10 | 0.582 | 0.646 | 0.762 | 0.926 | 0.074 | 2.10 | 5.08 | 0.264 |
| 11–20 | 0.668 | 0.721 | 0.795 | 0.942 | 0.058 | 2.13 | 5.02 | 0.264 |
| 21–30 | 0.644 | 0.701 | 0.795 | 0.937 | 0.063 | 2.23 | 5.12 | 0.302 |
| 31–40 | 0.654 | 0.696 | 0.805 | 0.921 | 0.079 | 2.32 | 5.27 | 0.311 |
| 41–50 | 0.664 | 0.720 | 0.791 | 0.925 | 0.075 | 2.22 | 5.20 | 0.278 |
| 51–60 | 0.641 | 0.699 | 0.793 | 0.946 | 0.054 | 2.15 | 5.06 | 0.280 |
| 61–70 | 0.681 | 0.729 | 0.813 | 0.958 | 0.042 | 2.12 | 5.04 | 0.266 |
| 71–80 | 0.624 | 0.677 | 0.807 | 0.930 | 0.070 | 2.18 | 5.19 | 0.291 |
| 81–90 | 0.670 | 0.716 | 0.829 | 0.947 | 0.053 | 2.18 | 5.14 | 0.300 |
| **91–97** | **0.701** | **0.751** | **0.830** | **0.966** | **0.034** | 2.15 | 5.04 | 0.299 |
Slow, noisy climb after the step-20 jump: reward +0.12, cite_f1 +0.07, capped 0.074 →
0.034 over the run; reads and turns flat (2.1 / 5.1) throughout — no collapse. Entropy
stayed in 0.26–0.32. ~190 s/step held for all 97 steps. The post-run eval pipeline
(`distill/run_rl_eval.sh`: merge all checkpoints → sft_dev all → heldout ONCE → MuSiQue)
is armed to start when the process exits.

### 4.30 [result] Held-out `val/pass@1` during training (`sft_dev[:128]`, greedy, reward > 0)
| step | val/pass@1 |
|---|---|
| SFT policy (start, = its greedy correct rate on the same 128, 4.14) | 0.633 |
| 25 | 0.633 |
| 50 | **0.734** |
| 75 | 0.727 |
The greedy held-out policy moved +10 points between steps 25 and 50 and held. This is the
first number in the project where GRPO improved a held-out metric over its starting
policy. The eval pipeline's full breakdown (capped, cite_f1, reads, per-source) says
what the +10 is made of.

### 4.31 [issue] How the run ended — two defects at the finish line (#16, #17)
Training completed (99 optimizer steps logged, final validation `val/pass@1` = 0.734,
zero tracebacks during training, 190 s/step throughout). Then:

- **#17 — no final checkpoint.** rLLM's `AgentWorkflowPPOTrainer.fit` checks
  `global_steps >= total_training_steps` *before* running a step, so `steps=100` yields
  99 training steps + the final val, and checkpoints fire only at
  `global_steps % save_freq == 0` — 25, 50, 75. The policy from steps 76–99 (the
  best-looking window on training batches, 4.29) was never written to disk and cannot
  be recovered (FSDP optimizer/adapter state lives only in the process). **Fix for next
  time:** set `steps = wanted + 1` so the last multiple of `save_freq` is actually
  trained, i.e. `steps=101` for a 100-step run with saves every 25; or make
  `save_freq` divide `steps − 1`.
- **#16 — the end-of-run `push_checkpoints()` crashed.** It builds a vLLM engine inside
  the training process; Ray/torch had already initialised CUDA there, so vLLM V1's
  forked `EngineCore` died with `Cannot re-initialize CUDA in forked subprocess`. Nothing
  lost — the periodic poller had already mirrored 25/50/75 to the Hub — but the
  best-checkpoint selection never ran. **[fix]** `train_dr.main` now runs the scoring in
  a fresh process (`subprocess.run([python, train_dr.py, "--push-checkpoints", preset])`),
  the same path as the manual CLI. Best-selection for THIS run is done by
  `distill/run_rl_eval.sh` instead (fuller breakdown than EM-only anyway).

Both are the kind of defect that only shows at the end of a 5-hour run — worth a
1-step dry run with `steps=2, checkpoint_every=1, push_checkpoints=True` on the sanity
preset before the next long run, specifically to exercise the finish line.

### 4.32 [issue] #18 (self) — the eval pipeline died on a stdout capture, unnoticed for 30 min
`ADAPTERS=$(python …)` captured everything the merge block printed, and
`hub.merge_checkpoint` prints its known "model_merger exited non-zero but the adapter
was written" note to **stdout**, so `eval_rl.py` was handed that sentence as arguments
(`error: unrecognized arguments: [hub] verl.model_merger …`). The monitor's filter had
`Error` (capitalised) and argparse writes `error:` — silent. Found on Harpreet's
"status". **[fix]** diagnostics redirected to stderr inside the block, a sanity check on
the captured string, and (lesson for every monitor) case-insensitive failure patterns.
Same family as #9/#13: unattended glue that was never exercised end-to-end.

### 4.33 [result] Model selection on `sft_dev` (n=128, greedy, one vLLM engine on the merged SFT weights, LoRA hot-swap)
| | SFT (no adapter) | step 25 | step 50 | **step 75** |
|---|---|---|---|---|
| correct (exact match) | 0.641 | 0.688 | 0.688 | **0.719** |
| picked the right sources (title_f1) | 0.779 | 0.820 | 0.810 | **0.833** |
| verified them (read before cite) | 0.859 | 0.926 | 0.883 | 0.922 |
| citation-F1 (the reward's term) | 0.771 | 0.818 | 0.806 | **0.829** |
| called `read` ≥ 1 | 0.984 | 0.992 | 0.992 | 0.992 |
| finished cleanly / produced an answer | 0.883 | 0.938 | 0.891 | 0.938 |
| **correct AND properly cited** | 0.484 | 0.516 | 0.547 | **0.570** |
| hit the turn cap (F12) | 0.125 | 0.078 | 0.125 | **0.086** |
| answer words / citations per answer | 1.68 / 1.69 | 1.77 / 1.80 | 1.73 / 1.72 | 1.74 / 1.79 |
| turns / reads / searches | 5.25 / 2.15 / 2.22 | 5.21 / 2.22 / 2.05 | 5.34 / 2.27 / 2.18 | 5.19 / 2.16 / 2.09 |
| by source — 2Wiki (n=72) correct / c&c | 0.708 / 0.500 | 0.792 / 0.569 | 0.764 / 0.569 | **0.806 / 0.611** |
| by source — HotpotQA (n=56) correct / c&c | 0.554 / 0.464 | 0.554 / 0.446 | 0.589 / 0.518 | **0.607 / 0.518** |
| RL gate (6 checks) | — | PASS | PASS | PASS |

Hop distribution of step 75: `{3: 2, 4: 15, 5: 95, 7: 5, 8: 11}` — 74% of episodes are
the canonical search→read→search→read→answer, up from 60% for SFT, and capped episodes
16 → 11. **Reading:** +8 correct, +9 correct-and-properly-cited, cite_f1 +0.06, capped
−31%, on BOTH sources, with turns / reads / answer length / citation count unchanged —
no collapse, no length or citation inflation. Selected: **step 75** (also the last
checkpoint that exists, 4.31). The `sft` column differs from 4.14's by 1 point on
`correct` (0.633 → 0.641): same weights, same greedy decoding — vLLM batching
non-determinism at the third digit; F8's noise-floor caveat applies to every 1–2-point
difference in this table, not to the 8–9-point ones.

### 4.34 [result] THE HELD-OUT REPORT — `heldout_eval` n=300, greedy, once. SFT vs RL step 75
| | SFT (merged, no adapter) | **RL step 75** | paired Δ, 95% bootstrap CI |
|---|---|---|---|
| correct (exact match) | 0.517 | 0.520 | +0.003 [−0.047, +0.050], P(Δ≤0)=0.46 |
| picked the right sources (title_f1) | 0.775 | 0.770 | |
| verified them (read before cite) | 0.860 | **0.895** | |
| citation-F1 | 0.749 | 0.761 | |
| **correct AND properly cited** | 0.380 | 0.393 | +0.013 [−0.027, +0.053], P(Δ≤0)=0.29 |
| correct but partially cited | 0.137 | 0.127 | |
| hit the turn cap (F12) | 0.117 (35) | 0.097 (29) | |
| finished cleanly / produced an answer | 0.913 | 0.913 | |
| answer words / citations / turns / reads / searches | 2.11 / 1.78 / 5.16 / 2.15 / 2.10 | 2.10 / 1.78 / 5.18 / 2.18 / 2.09 | unchanged |
| hop distribution | `{3:19, 4:57, 5:160, 6:19, 7:10, 8:35}` | `{3:10, 4:37, 5:206, 6:12, 7:6, 8:29}` | 53% → 69% canonical 5-call episodes |
| by source — HotpotQA (n=152) correct / c&c / capped | 0.474 / 0.355 / 0.092 | **0.520 / 0.414 / 0.066** | +4.6 / +5.9 |
| by source — 2Wiki (n=148) correct / c&c / capped | 0.561 / 0.405 / 0.142 | 0.520 / 0.372 / 0.128 | −4.1 / −3.3 |
| RL gate (6 checks) | | PASS | |

Per question: 128 both correct, 27 only SFT, 28 only RL, 117 neither. Correct-and-cited:
98 both, 16 only SFT, 20 only RL. Of SFT's 35 capped episodes, RL freed 16 and answered
7 of those correctly.

**Honest verdict.** The dev-set gain (+8 correct, +9 correct-and-cited, 4.33) did NOT
replicate at that size. Held-out: +0.3 correct, +1.3 correct-and-cited, both inside the
noise; the gate passes on direction, not on magnitude. What DID carry over is the
process: read-before-cite +3.5 points, capped episodes −17%, and the hop distribution
tightening onto the canonical search→read→search→read→answer — with no length,
citation, or turn inflation. RL made the policy *more disciplined* on held-out; it did
not make it measurably *more correct* on held-out.

**[FINDING F23] The held-out set is a harder distribution than training and dev.**
HotpotQA in `heldout_eval` is 100% `hard`-level (152/152); in `sft_dev` it is 16% hard
(9/56) and in `rl_train` 18% (182/996). This is because `heldout_eval` draws from the
datasets' *validation* splits and HotpotQA's validation set is hard-only, while
`sft_dev`/`rl_train` are slices of the train-split pool. Consequences: (1) SFT itself
scores 0.64 on dev vs 0.52 held-out — not overfitting, a harder set; (2) RL's gain
was learned and selected on the easier distribution, and on held-out it shows up on
the source that DID get harder (HotpotQA-hard +5.9 c&c) while 2Wiki, whose train and
validation splits are alike, moved −3.3 (n=148, inside noise); (3) every earlier
dev-vs-heldout comparison in this lab (SFT stage included) carries the same offset.
Not a bug — the split design was right to keep validation-split questions untouched —
but it must be stated next to every dev number from now on.

**[speculation] Why the dev gain was bigger.** Three non-exclusive causes: selection
(step 75 was chosen on dev; but all three checkpoints were +5 to +9 there, so
selection alone does not explain it); distribution (dev is easy/medium-heavy like
`rl_train`, so RL improved what it practised on); and n (128 vs 300, CI ±5). The 2Wiki
dev slice was where most of the dev gain sat (0.708 → 0.806, n=72) and it is exactly
the slice that reversed on held-out — the signature of a noisy small-n win.

### 4.35 [result] OUT-OF-DISTRIBUTION — `musique_dev` n=300, greedy, once. SFT vs RL step 75
Never trained or collected on; 20 passages/question, 2–4 hops (`GENERALIZATION_EVAL.md`).

| | SFT | **RL step 75** | paired Δ, 95% CI |
|---|---|---|---|
| correct (exact match) | 0.220 | **0.260** | **+0.040 [+0.003, +0.077]** |
| picked the right sources (title_f1) | 0.344 | 0.400 | |
| verified them (read before cite) | 0.465 | **0.585** | |
| citation-F1 | 0.335 | 0.394 | |
| correct AND properly cited | 0.140 | 0.147 | +0.007 [−0.017, +0.030] |
| **hit the turn cap** | 0.530 | **0.443** | **−0.087 [−0.143, −0.030]** |
| produced an answer | 0.493 | 0.593 | |
| called `read` ≥ 1 | 0.937 | 0.997 | |
| answer words / citations | 1.10 / 0.96 | 1.21 / 1.16 | |
| turns / **reads** / **searches** | 6.65 / 2.18 / 3.97 | 6.58 / **2.85** / **3.14** | reads up, searches down, turns flat |
| hop distribution | `{3:8, 4:22, 5:71, 6:24, 7:16, 8:159}` | `{3:1, 4:12, 5:106, 6:8, 7:40, 8:133}` | |
| by hops — 2 (n=141) correct / capped | 0.348 / 0.383 | 0.369 / 0.369 | |
| by hops — 3 (n=98) | 0.143 / 0.571 | 0.184 / 0.439 | |
| by hops — 4 (n=61) | 0.049 / 0.803 | **0.131 / 0.623** | |
| RL gate | | PASS | |

**Reading.** This is the strongest evidence in the session that RL changed the *process*
and that the change generalises: on a harder, unseen dataset the RL policy answers
+10 points more often, gets stuck 8.7 points less often (significant), and is +4 correct
(significant), with the gain concentrated on 3- and 4-hop questions — exactly where one
search is not enough. The mechanism is visible in the tool counts: **reads 2.18 → 2.85,
searches 3.97 → 3.14 at constant turns** — it reads what it found instead of
re-searching, which is the anti-loop behaviour RL was rewarded for (zero for never
answering). Correct-and-properly-cited is flat because citing ALL of 3–4 gold passages
perfectly is ~never achieved by either policy on MuSiQue (0.000 for 3+ hops in both).

Absolute numbers are low (0.26 correct) — as pre-registered in `GENERALIZATION_EVAL.md`
(0.25–0.40 expected). Aliases make EM slightly lenient here; both arms get the same
leniency.

### 4.36 [verdict] The session, in three lines
1. **Training signal:** every earlier GRPO run in this lab was optimising a truncated,
   reward-zeroed signal (F18). This is the first run where GRPO saw whole trajectories
   and their real rewards; it trained stably for 99 steps with no collapse.
2. **In-distribution:** dev +8 correct / +9 correct-and-cited; **held-out +0.3 / +1.3,
   inside noise.** Process transferred (read-before-cite +3.5, capped −17%), correctness
   did not measurably. Held-out is a harder distribution than training (F23).
3. **Out-of-distribution (MuSiQue):** +4.0 correct and −8.7 capped, both significant;
   reads up, searches down. RL taught "read, then commit" — a generalising process
   change whose *correctness* payoff is largest where the task is hardest and smallest
   on the in-distribution held-out set.

Cheapest next experiments, in order: train RL on the hard slice (F23); the fully-cited-
regardless-of-correctness SFT recipe (4.17); a longer / 2-GPU run. All in `HANDOFF.md`.

### 4.37 [artifact] Full step-75 checkpoint pushed for resuming
The Hub had only the 57 MB LoRA adapter (enough to RUN the model, not to continue
training). Pushed the complete verl checkpoint (LoRA + optimizer + RNG/extra state +
dataloader state, 355 MB) with `run_meta.json`, the resolved Config, and a README with
the copy-back steps, to `harpreet22happy/deep-research-agent-grpo` @
`deep_research_agent_rl_from_sft_correct_only__07_09_2026__23_59_34` under
`FULL_VERL_CHECKPOINT_FOR_RESUME__rl_from_sft_correct_only__global_step_75/`.
Decision recorded (Harpreet): the next RL run resumes from this, on a hard-question
slice, rather than starting from SFT again.
