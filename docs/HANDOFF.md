# HANDOFF → pod Claude Code session (deep_research_agent — Capstone 6, Branch B)

## ⚡ CURRENT HANDOFF — 2026-09-08, end of the RL-from-SFT session

**Read [`START_HERE.md`](START_HERE.md) first, then [`RL_FROM_SFT_LOG.md`](RL_FROM_SFT_LOG.md)**
(the running log of this session: design, every issue, every result, the this-run-vs-earlier
table, and the issues register §5). Everything below this section is history.

### Where we are

| | |
|---|---|
| **SFT** | retrained two ways on the full teacher set (`sft_correct_only` 1,209 / `sft_imitate_all` 2,682), both on the Hub; `correct_only` is the RL start (F19). The 2026-08-26 `deep-research-agent-sft` adapter is untouched. |
| **RL** | **DONE, once.** `deep_research_agent_rl_from_sft_correct_only` (W&B `t2n91x71`), 99 GRPO steps, ~5.2 h. Best checkpoint step 75 on the Hub: `deep-research-agent-grpo`, branch `deep_research_agent_rl_from_sft_correct_only__07_09_2026__23_59_34`, `best/step75_dev_cc0.570` — a LoRA on top of the MERGED correct_only weights, not the bare base. |
| **Result** | dev (n=128): correct-and-properly-cited 0.484 → **0.570**, correct 0.641 → 0.719. **Held-out (n=300, once): 0.380 → 0.393 and 0.517 → 0.520 — inside noise.** Process transferred (read-before-cite 0.860 → 0.895, capped 0.117 → 0.097, no hacking). **MuSiQue (OOD, n=300): correct 0.220 → 0.260 and capped 0.530 → 0.443, both significant; reads up, searches down.** See F23: held-out is hard-only HotpotQA. |
| **F18** | the training-signal bug that affected every August GRPO run — fixed (`verl_max_response_length`). Read it before trusting any older run's conclusion. |
| **Pod** | safe to terminate; code on GitHub, adapters + checkpoints on the Hub, curves on W&B, eval JSONs committed. |

### Do this, in order, next time

```bash
bash setup_pod.sh && source .venv-deep-research/bin/activate
python -m pytest -q tests/ distill/tests/          # 80 passed
python train_dr.py sanity                           # still the wiring proof
# rebuild the merged SFT base RL sits on (gitignored, ~1 min):
hf download harpreet22happy/deep-research-agent-sft-correct-only --local-dir ./sft_adapter_correct_only
python distill/merge_sft.py --adapter ./sft_adapter_correct_only
```

### What to do next — options, cheapest first, each with its reason

1. **Dry-run the finish line before any long run:** `steps=2, checkpoint_every=1,
   push_checkpoints=True` on the sanity preset. Issues #16/#17 only showed at the end of a
   5-hour run (no step-100 checkpoint; end-of-run push crashed). Both fixed; unverified.
2. **Resume from step 75** — the FULL verl checkpoint (optimizer state included) is on the
   Hub: `deep-research-agent-grpo` @ the run branch,
   `FULL_VERL_CHECKPOINT_FOR_RESUME__rl_from_sft_correct_only__global_step_75/` (its README
   has the exact copy-back steps). Harpreet's leaning (2026-09-08): resume from 75 on a
   HARD training slice (item 4) rather than from scratch; `steps = 75 + wanted + 1`.
   Recovering just the lost tail (`steps=101`, ~80 min) is the cheaper variant. The training-batch curve was still rising at 91–97 (4.29) and held-out
   `val/pass@1` was flat 50→100, so expect little — but it is the cheapest experiment.
3. **The SFT recipe that should get commitment without losing citation** (4.17):
   correct_only's citation gate with the correctness gate dropped (wrong-but-fully-cited
   teacher episodes added). ~15 min. Then RL from it.
4. **Train RL on harder questions.** F23: `rl_train` HotpotQA is 18% hard, held-out is
   100% hard. A `rl_train_hard` slice (HotpotQA `level=="hard"` + 2Wiki) puts the RL
   signal where the held-out set lives. Cheap: a filter in `splits.py`.
5. **Longer / bigger RL** (2 GPUs, §4.25 estimates ~100 s/step): the honest read of this
   run is "direction right, effect small at 99 steps × 32 questions"; scale is the
   obvious next lever, but only after 3 or 4 changes what the signal is.
6. **MuSiQue** (`GENERALIZATION_EVAL.md`) — the OOD report is in `RL_FROM_SFT_LOG.md` §4.35;
   re-run it for any new checkpoint with `eval_rl.py --split musique_dev`.

**Do not** report the dev numbers as the result; the held-out line is the result.

---

## ⚡ (previous) CURRENT HANDOFF — 2026-08-26, end of session

**Read [`START_HERE.md`](START_HERE.md) FIRST, then this section.** START_HERE has the
status, the 17-finding register (each marked CONFIRMED / KILLED / OPEN), navigation, and
what to check before RL. **This section is the baton** — the exact commands, in order.

**Everything below this section is history** (2026-08-24 / -08-25 / -08-26 early). Four
older "START HERE" sections, each superseded by the one above it. Worth reading for the
reasoning at each point in time, including conclusions later overturned — but they are
**not instructions**.

### Where we are

| | |
|---|---|
| **SFT** | **DONE and shipped.** `harpreet22happy/deep-research-agent-sft` (private LoRA, r=32) |
| **Held-out result** (300 q, locked, greedy) | correct-and-properly-cited **0.3% → 30.7%**; read-before-cite **0.198 → 0.810** (vs the base model *with* a worked example, not just the bare base) |
| **Teacher data** | **COMPLETE — all 4,000 `sft_collect` questions**, 1 trajectory each, `harpreet22happy/deep-research-agent-trajectories` (private). 1,210 pass the strict gate. $8.88. |
| **RL** | **Never run on the SFT checkpoint.** This is the next stage. |
| **Pod** | Safe to terminate. Everything is on GitHub or the Hub. |

### Do this, in order

```bash
# 1. rebuild the env (fresh pod; FRESH_POD_SETUP_AND_SANITY_CHECK.md has the detail)
bash setup_pod.sh && source .venv-deep-research/bin/activate
python -m pytest -q tests/ distill/tests/        # expect 69 passed

# 2. THE CHECK THAT HAS NEVER BEEN RUN ON ANY POD. Do not skip — step 4 depends on it.
python train_dr.py sanity

# 3. (optional but recommended) retrain SFT on the FULL data — the shipped adapter used
#    418 examples; 1,210 are now available, 2.9x more.
hf download harpreet22happy/deep-research-agent-trajectories --repo-type dataset \
   --local-dir distill/ --include teacher_trajectories.jsonl
python distill/validate.py                       # $0, re-derives every tool call
python distill/build_sft.py --tiers A            # -> distill/sft_dataset.pt
python distill/sft_train.py --batch-size 2 --grad-accum 8 --epochs 2 --run-name sft_full
python distill/eval_sft.py --adapter distill/runs/sft_full/final --split sft_dev --n 128

# 4. RL from the SFT adapter
```

**On step 3's `--epochs 2`, not 3:** the shipped run's val loss flattened after epoch 1
(0.2722 → 0.2565 → 0.2563). More epochs bought memorisation, not generalisation.

### The RL stage — config, and the one risk to watch from step 1

- **Start from the SFT adapter, not the base model.**
- `include_worked_example=False`. The student was trained under the short prompt and must
  be rolled out under it — **training and rollout prompts must match** or the SFT stage is
  wasted. This is silent if wrong: nothing crashes.
- `rl_train` split (14,500 questions, disjoint from SFT; `splits.py` asserts it, and the
  collected 4,000 were verified 100% inside `sft_collect` with zero leakage).
  `cloud_preset` uses `prompts_per_step=1`, so a 252-step run touches only ~252 distinct
  questions — raise it rather than assuming the pool size does the work.
- Early-stop on `sft_dev`; report on `heldout_eval` **once**, at the end.
- Reuse `distill/eval_sft.py` (3-arm base / base+example / tuned). Do not rebuild it.

**THE TARGET (finding F12):** 16% of SFT episodes never terminate — they search until the
budget runs out. Held-out: 0.064 correct, only 19% produce any answer. They **search 2.4x**
more than successful episodes: a failure to *commit*, not to read. Cause: trained only on
trajectories where the teacher SUCCEEDED, so the model never saw concluding under
uncertainty. Excluding them the model is **0.482 correct / 0.817 cite_f1** — the headline
understates it.

This is a good RL target: zero outcome-reward for never answering is exactly the contrast
GRPO sharpens, and **unlike Attempts 1-4 / Probes 1-4 the behaviour already exists 84% of
the time** rather than needing to be invented. That was the premise the whole RFT plan
rested on, and it is now actually satisfied.

**THE RISK, and it is the same shape as Probe 3:** the outcome reward pays for answering,
so the cheapest way to stop collecting zeros is to **answer immediately** — which collapses
straight back to the 2-turn pattern and undoes the read-before-cite behaviour this SFT
stage just bought.

> **Watch turns-per-episode and read-rate in the first ~20 steps.** If turns trend toward
> 2, it is hacking the outcome reward — strengthen the citation gate before spending hours.
> Probe 4's `cite_gated` design is the counter, and unlike then, **the model can now
> actually cite.** That combination has never been tested.

### Two plan defects found by the result — do not follow the plan blindly

1. **`distill/SFT_RL_PLAN.md` §3 is wrong.** It says tier-B trajectories should grade the
   process turns and SKIP the final answer turn. Given F12 that teaches process while never
   demonstrating conclusion — the exact broken habit. **Re-test with the answer turn
   graded.** With 1,210 tier-A and ~2,000 tier-B now available, this is a cheap A/B.
2. **`rft_diagnosis/RFT_PLAN_AND_MODEL_DIAGNOSIS.md`'s Diagnosis 2 is dead** (rejection-
   sampling from the student's own rollouts — zero of 320 rollouts were correct-and-cited,
   nothing to mine). Its A/B table is stamped SUPERSEDED. The fallback is what ran.

### Open, not blocking

**~40 GiB of GPU memory is unexplained** in the SFT trainer (`distill/BATCH_SIZE_TUNING.md`
§5). Both hypotheses were tested and neither accounts for it: checkpointing on/off is
45.8 vs 46.2 GiB, and Liger fused CE recovered only 5 GiB. The current setting is safe;
this is upside. A `torch.cuda.memory._record_memory_history()` snapshot would likely settle
it in one run.

---

## ⚡⚡⚡⚡ START HERE — 2026-08-26 (LATE) — SFT IS DONE AND SHIPPED, RL IS NEXT

**Deliverable: `harpreet22happy/deep-research-agent-sft` (private).** A LoRA adapter that
teaches read-before-cite. Held-out (300 q, never trained on): correct-and-properly-cited
**0.3% -> 30.7%**, read-before-cite **0.198 -> 0.810** (vs the base model *with* a worked
example in its prompt, not just vs the bare base). All five pre-registered gate conditions
passed. Full detail: `distill/SFT_RL_PLAN.md`, `BUILD_LOG.md`'s 2026-08-26b entry, and the
model card on the Hub.

**Artifacts on the Hub (both private, both verified by fresh download + sha256):**
- model: `harpreet22happy/deep-research-agent-sft` — the LoRA adapter (228 MB, r=32)
- data:  `harpreet22happy/deep-research-agent-trajectories` — 2,709 teacher trajectories
  (841 pass the strict correct-AND-cited gate). The JSONL is gitignored, so the Hub copy
  is the ONLY copy that survives pod termination.

**Every number in one place:** [`distill/RESULTS.md`](../distill/RESULTS.md) — both eval
sets, all three arms, the bimodal failure on each, tool-call distributions, buckets,
anti-hacking probes, the training curve, and the teacher-data stats. Generated from the
eval JSONs, so it is the reference appendix for the eventual write-up.

**The four logs for this stage** (conclusions live in the plan doc; these are the paths):
- [`distill/SFT_HISTORY_LOG.md`](../distill/SFT_HISTORY_LOG.md) — the SFT stage in order:
  decisions, the eight defects hit while building, the results and what they honestly say.
- [`distill/DATA_COLLECTION_LOG.md`](../distill/DATA_COLLECTION_LOG.md) — the teacher data:
  why a teacher at all, the six defects found during collection, cost, and what the
  2,198 episodes look like.
- [`rft_diagnosis/FORMAT_INVESTIGATION_LOG.md`](../rft_diagnosis/FORMAT_INVESTIGATION_LOG.md)
  — the earlier harness audit that made any of this measurable.
- [`distill/BATCH_SIZE_TUNING.md`](../distill/BATCH_SIZE_TUNING.md) — batch tuning on the
  A100, the three bugs it exposed, and **an unexplained ~40 GiB of memory use left as an
  open exercise** (§5). Not a blocker; the current setting is safe. If it turns out to be
  a real bug, SFT gets several times faster and GRPO likely benefits too.

**DO FIRST, in order:**
1. `python train_dr.py sanity` — the end-to-end GRPO check from
   `FRESH_POD_SETUP_AND_SANITY_CHECK.md` §5. **It has never been run on any pod this
   session.** Everything below depends on it and nothing so far did.
2. Read the KNOWN FAILURE below before designing the RL reward.

**➡️ THE RL STAGE, and it has an unusually well-defined target.**
16% of the SFT model's episodes (47/300) **never terminate** — they search until the turn
budget runs out and usually return nothing (correct 0.064; only 19% answer at all). Measured, not guessed: stuck episodes
search **2.4x** more than successful ones (4.36 vs 1.81) and read 1.53x more than the
question needs. It is a failure to COMMIT, not slowness. Cause: the adapter was trained
only on trajectories where the teacher SUCCEEDED, so it never saw concluding under
uncertainty.

Excluding those, the model is **0.482 correct / 0.817 cite_f1**. So the shape is already
right 84% of the time, and a zero outcome-reward for never answering is exactly the signal
GRPO sharpens well. Unlike Attempts 1-4 and Probes 1-4, RL is now being asked to sharpen a
behaviour that EXISTS rather than to invent one — which is the premise the whole RFT plan
rested on.

**Config for the RL run:**
- Start from the SFT adapter, NOT the base model.
- `include_worked_example=False` — the student was trained under the short prompt and must
  be rolled out under it. Training and rollout prompts must match.
- `rl_train` split (14,500 questions, disjoint from SFT — `splits.py` asserts it).
  NOTE `cloud_preset` uses `prompts_per_step=1`, so a 252-step run touches ~252 distinct
  questions; raise `prompts_per_step` rather than assuming the pool size does the work.
- Report against `heldout_eval` (500) at the end only; use `sft_dev` for early stopping.
  `distill/eval_sft.py` already does the 3-arm comparison — reuse it, do not rebuild.

**One plan change this result forces:** `SFT_RL_PLAN.md` §3 recommends tier B trajectories
grade the process turns and SKIP the final answer turn. Given that never-concluding is the
failure mode, that now looks wrong — it teaches process while never demonstrating
conclusion. Re-test with the answer turn graded before accepting the plan as written.

**Teacher collection** was still running at session end (~2,100 of 3,782 episodes, ~$4.4).
`distill/collect.py --batch N` resumes and skips what is done; `--inspect` re-scores for
$0. More data is the cheapest way to grow the SFT set if the RL stage wants a stronger
starting point.

---

## ⚡⚡⚡ 2026-08-26 (earlier) — harness audit

**The 2026-08-25 section's "THE VERY NEXT THING TO DO" — the terseness fix in
`diagnosis1_native_tools.py` — is CANCELLED. Do not run it.** It was built, then
abandoned once the harness was audited: it targeted answer-verbosity, a symptom that
only appeared in the biased 10/64 sample that survived a broken code path. The model is
not verbose; it answers `no` in two tokens.

**What happened instead:** Harpreet asked to confirm the format and turn boundaries
before running anything. That turned up **eight real bugs in the native tool-calling
harness** — the largest being that Qwen finishes by writing plain text, and our code
scored that as a parse error and discarded it (28/36 rounds; two exact gold answers
thrown away). The bracket arm was then audited the same way and has the same disease in
a smaller dose. `RFT_PLAN_AND_MODEL_DIAGNOSIS.md`'s A/B table is stamped SUPERSEDED.

**Read, in this order:**
1. `rft_diagnosis/RFT_PLAN_AND_MODEL_DIAGNOSIS.md` → **"Harness audit — 2026-08-26"**
   (the last section). Everything current is there.
2. A transcript or two under `rft_diagnosis/transcripts/` — `v2_hotpot-train-48025.txt`
   is a clean 4-step multi-hop success; `bracket_hotpot-train-49699.txt` shows the
   reject-loop. Reading real transcripts is what found all of this; aggregates hid it
   through four separate readings.
3. `BUILD_LOG.md`'s 2026-08-26 entry for the short version.
4. [`rft_diagnosis/FORMAT_INVESTIGATION_LOG.md`](../rft_diagnosis/FORMAT_INVESTIGATION_LOG.md)
   — the chronological trail of HOW this was found: what was checked in what order, why,
   and the six things that were got wrong along the way. Read this one if you are about
   to investigate something similar; the method transfers further than the individual
   bugs do.

**Environment:** rebuilt clean on a fresh A100-80GB. 38 tests pass; torch 2.11.0+cu128 /
vllm 0.22.1 / verl 0.9.0 import fine. `setup_pod.sh` gained bug-fix #6 (`ninja` — without
it flash-attn compiles SERIALLY, ~3.5h instead of ~12min, and `MAX_JOBS` does nothing).

**➡️ THE VERY NEXT THING — an open question, not a queued command.**
Why is the bracket format ~43% correct when it engages, while the native format is
~5-8% however it is configured? It is **not** worked examples and **not** reasoning —
the native arm now has both and still loses. Something else differs that these runs do
not isolate. Suggested first cuts, cheapest first:
  - Per-question overlap: do the two arms succeed on the SAME questions? If they do,
    difficulty dominates and the format gap is smaller than it looks.
  - Read the bracket arm's 12 winning transcripts against the native arm's 3. The gap is
    large enough that the mechanism should be visible in raw text.
  - Raise n or sweep seeds first: at n=64 the run-to-run swing is ~5pp (0.188/0.156/0.141
    on identical config), so do not chase differences smaller than that.
**Do not design Diagnosis 2's collection pass around either format until this is
settled** — the choice determines what every mined trajectory looks like.

Commands (all batched vLLM; never batch-1 in a loop):
```bash
source .venv-deep-research/bin/activate
python rft_diagnosis/verify_format.py template      # offline, no GPU
python rft_diagnosis/diagnosis1_native_v2.py --n 16 --k 4 [--think] [--examples]
python rft_diagnosis/audit_bracket.py --n 16 --k 4
```

---

## ⚡⚡ START HERE — 2026-08-25 end-of-session handoff (read THIS section first — supersedes 2026-08-24 below)

**This pod is being TERMINATED, not stopped** (Harpreet's explicit policy going forward
— see [`FRESH_POD_SETUP_AND_SANITY_CHECK.md`](FRESH_POD_SETUP_AND_SANITY_CHECK.md),
written this session specifically for this). The 2026-08-24 section below (stop/resume
on pod `m4j1fsfj84767c`) is now fully historical — that pod, its `/workspace`, and
everything on it are gone or about to be. **You are starting from a genuinely fresh
pod.** Do this, in order:

1. **Read [`FRESH_POD_SETUP_AND_SANITY_CHECK.md`](FRESH_POD_SETUP_AND_SANITY_CHECK.md)
   first** — the fast-path checklist for bare-pod → verified-working environment,
   written after a real session that hit 5 distinct infra bugs doing exactly this
   (slow network-mounted `/workspace`, venv-move-breaks-shebangs, unpinned
   `flash-attn` needing `psutil`, `ninja` over-parallelism OOM-ing against the
   container's real cgroup limit, `pkill -f` self-matching and killing its own
   shell). All 5 are now fixed/documented; that doc is the checklist, `setup_pod.sh`
   has the fixes baked in.
2. Clone straight to local disk (`/root` or equivalent) — **never `/workspace`, no
   exceptions** (Harpreet's explicit call: every pod is terminated between sessions
   now, so there's nothing to gain from it — don't relitigate this per-pod). Re-supply
   `.env` (`HF_TOKEN`,
   `WANDB_API_KEY` — gitignored, gone on a fresh pod, not something broken).
3. Run `setup_pod.sh`, then the full sanity chain in that doc's §5, ending in the
   real proof: `python train_dr.py sanity` (a genuine end-to-end GRPO step, not just
   imports). Don't skip this — "the doc says it worked last time" isn't the same as
   "it works now."
4. **Then read [`rft_diagnosis/RFT_PLAN_AND_MODEL_DIAGNOSIS.md`](../rft_diagnosis/RFT_PLAN_AND_MODEL_DIAGNOSIS.md)**
   in full — this is where the actual research state lives now, not the sections
   below. Skim `BUILD_LOG.md`'s two most recent entries (2026-08-25, phase 5 and
   phase 6) for the narrative if you want the short version first.

### Where the research actually stands (as of 2026-08-25)

Last session's plan (`HANDOFF.md`'s old "very next thing: run `probe_cite_gated`")
is **superseded**. This session paused GRPO entirely to run a capability diagnosis
first (`rft_diagnosis/RFT_PLAN_AND_MODEL_DIAGNOSIS.md`'s "Diagnosis 1") — verify the
base model can do multi-hop tool use at all before spending more RL compute on it.
It was actually RUN, for real, against Qwen2.5-3B-Instruct:

- **Along the way, found and fixed 3 real bugs in production `env.py`** (not just a
  diagnostic script) — a greedy-regex parser bug that silently absorbed hallucinated
  model continuations into tool-call arguments, a missing generation stop-sequence,
  and a prompt that never explicitly told the model to stop after one action. All
  three are fixed, tested (`tests/test_mechanics.py` has 2 new regression tests),
  and already pushed. If you're about to touch `env._parse_react_action` or
  `env._opening_prompt`, read that doc's "Diagnosis 1 — results" section first —
  the reasoning behind each fix matters, not just the diff.
- **The finding:** the model genuinely CAN do valid multi-hop tool use — one probed
  question scored 4/4 correct-and-cited across every sampled trajectory, real proof,
  not inferred. But it's **bimodal, not reliable**: roughly a third of probed
  questions collapse to a zero-tool-call response across all samples instead of
  engaging at all, with no clean question-type explanation found (checked directly,
  not guessed).
- **A same-session A/B** tested Qwen2.5-Instruct's own NATIVE Hermes-style
  tool-calling format (`<tool_call>{...}</tool_call>`, verified against the real
  installed tokenizer's `chat_template`) against the custom bracket format
  (`rft_diagnosis/diagnosis1_native_tools.py`). Different, complementary failure
  mode: engagement got dramatically better (zero-tool-call rate 56%→8%) but
  `correct_rate` dropped to 0% — investigated directly (pulled real answer text) and
  found it's an answer-VERBOSITY problem (the model embeds the right fact in a full
  sentence, which strict exact-match can't credit), not a knowledge gap.

**➡️ THE VERY NEXT THING TO DO** (recommended, ~$0.30-0.50, same scale as what was
just done — try this BEFORE Diagnosis 2, its result decides which format Diagnosis 2
collects with):

`diagnosis1_native_tools.py`'s `TOOL_SCHEMAS`, the `"answer"` entry (currently ~line
55), has this `description`:
```python
"description": ("Commit your final answer. Cite every passage that supports it, "
                "by its exact title in square brackets, e.g. "
                "'American [Blue Harvest (film)] [Jane Doe (director)]'. "
                "Do not cite a passage you did not read. This ends the episode."),
```
Add the terseness rule that currently only lives in the system prompt (`_INSTRUCTIONS`)
directly into THIS description — e.g. prepend `"Answer with the SHORTEST possible
phrase — usually 1-3 words: the exact entity, name, number, or 'yes'/'no'. No sentence,
no explanation. "` before "Cite every passage...". The hypothesis (from this session's
finding, not a guess): the model attends to the tool schema's own description more
reliably than the surrounding system-prompt prose when in native-calling mode — that's
why `correct_rate` was 0% despite the model clearly knowing correct facts (see the
plan doc's A/B section for the literal `"...released in 2010, a song written..."`
example that should have just been `"2010"`).

Then re-run the exact same comparison point:
```bash
cd assignments/deep_research_agent
source .venv-deep-research/bin/activate
python rft_diagnosis/diagnosis1_native_tools.py
```
(Same 16-question probe pool, `k=4`, `temperature=0.9` — hardcoded in `main()`, matches
`diagnosis1_results_v2.json`'s bracket-format run for a clean before/after comparison.)
Compare the new `correct_rate`/`mean_cite_f1`/bucket breakdown against the numbers
already logged in `RFT_PLAN_AND_MODEL_DIAGNOSIS.md`'s A/B table. If `correct_rate`
comes up meaningfully from 0% while the engagement numbers (zero-tool-call rate,
`calls_read_rate`) stay good, native format + this fix is the clear choice for
Diagnosis 2's collection pass. If it doesn't move, fall back to the bracket format
(which already has a proven, if less frequent, correct-and-cited case) and proceed to
Diagnosis 2 with that instead — don't spend a third round chasing the native format
further without a new concrete hypothesis.

Whichever format wins, the plan doc's existing "Data hygiene" section (frozen held-out
set, pilot-yield-before-scaling, dual-gate correct+cited filtering) still applies to
Diagnosis 2 — don't skip it because this feels like "just one more probe."

---

## ⚡ START HERE — 2026-08-24 end-of-session handoff (read this section FIRST, before anything below)

**The pipeline is fully built, tested, and has been running real cloud GRPO training for
two full days now.** Everything below §0 (version matrix, wiring the `# VERIFY` seams,
sanity spike, masking gate) is **already done** — do NOT redo it, do NOT re-derive it.
This project is now mid-way through an active hyperparameter/reward-design research
question, not a setup task. Skipping straight to "what do I do next" below will save you
real time.

### 1. Get the environment back — ~30 seconds, no reinstall needed
The pod (`m4j1fsfj84767c`) is being **stopped** (not terminated) at the end of this
session — `/workspace` (repo + built venv) survives a stop.
```bash
# SSH back in (RunPod dashboard -> pod -> Connect tab; port/IP likely changed since a
# stop reassigns them — re-copy the connect string). Once connected:
cd /workspace/Agentic-RL-Alignment-Path/assignments/deep_research_agent
source .venv-deep-research/bin/activate
python -m pytest -q tests/                      # ~10s, confirms it came back clean
python -c "import torch; print(torch.cuda.is_available())"
git log --oneline -5                             # confirm you're at today's final commits
```
Full detail if anything above doesn't check out clean (unlikely): [`POD_LIFECYCLE.md`](POD_LIFECYCLE.md).

**If instead you're on a genuinely fresh pod** (this one got terminated after all, or a
new one entirely) — verified honestly, not assumed, so read this carefully rather than
trust an earlier looser claim in this doc's edit history:
- **Code + all docs**: on GitHub (`Harpreet221295/Agentic-RL-Alignment-Path`, branch
  `master`) — but the repo is **private** (confirmed: an unauthenticated clone/curl gets
  a 404). You (Harpreet) need to supply a fresh git credential (PAT or SSH key) to clone
  it — this can't be recovered from inside the repo itself, nor should it be (committing
  a credential would be a real security mistake, not a fix). `pip-freeze.txt` IS
  committed, so `setup_pod.sh` can reuse the already-resolved rllm/verl/vllm/torch
  version matrix instead of re-resolving it from scratch (the single most
  time-consuming part of the original setup — see `RLLM_VERL_INSTALL_NOTES.md`).
- **Secrets** (`HF_TOKEN`, `WANDB_API_KEY` in `.env`): gitignored, never committed, by
  design — same as any project's secrets. Gone on a real terminate. You'll need to
  re-supply them (a fresh `.env` in this folder) before `push_checkpoints`/W&B logging
  work again — this is normal setup, not something broken.
- **The best checkpoint** (`global_step_25` from Attempt 4, `lr=5e-5`, BEFORE any of
  today's four reward-design probes or the prompt fix — it predates all of them):
  on the HF Hub at `harpreet22happy/deep-research-agent-grpo`, but on a **timestamped
  branch, NOT `main`** (`main` is genuinely empty — that's `hub.py`'s normal push
  design, not a bug). Pull the LATEST one specifically:
  ```bash
  hf download harpreet22happy/deep-research-agent-grpo \
    --revision deep_research_agent_cloud__24_08_2026__04_11_47 --local-dir ./resumed_checkpoint
  ```
  (`huggingface-cli` is deprecated in this venv's installed version — use `hf`, verified
  against the actual installed CLI, not assumed from memory.)
  (List `api.list_repo_refs(...).branches` if that exact revision string ever changes —
  don't assume it stays this one forever.) Note this is NOT a checkpoint from the probes
  or with the prompt fix baked in — resuming from it means resuming BEFORE today's
  citation-discipline findings, not continuing them; the code/docs are what actually
  carry today's progress forward, not this specific checkpoint.
- **Raw per-episode debug logs** (`runs/*/episode_logs/`): never pushed anywhere, real
  loss on a terminate. Several findings today (the eval-turn-count discovery, disproving
  the clip_ratio theory) came from reading these directly — that specific historical
  data won't be recoverable, only what's already written into the docs.

### 2. Build context — read in THIS order, don't read everything
1. **[`TRAINING_HISTORY_LOG.md`](TRAINING_HISTORY_LOG.md)** — start with its "Quick
   reference" table at the top (every training run this project has done, one line each),
   then read the most recent entries in full (Attempt 4 through Probe 4, plus the
   "eval-time turn count" finding at the very bottom). This is the real state of the
   project — what's been tried, what worked, what didn't, with real numbers, not guesses.
2. **[`TENTATIVE_FUTURE_EXPERIMENTS.md`](TENTATIVE_FUTURE_EXPERIMENTS.md)** — the "maybe
   later" ideas pile, now with real evidence behind it (see "FINAL UPDATE 2026-08-24" at
   the top) — this is where the cold-start SFT/RFT/preference-stage discussion lives.
3. Skim **[`NOTES.md`](NOTES.md)** only if you need the ORIGINAL design reasoning behind
   the reward (why gated vs. additive, why citation-F1, etc.) — settled decisions, not
   today's active research.
4. Everything else in §7's file map is reference material — pull it up only if something
   specific sends you there (e.g. a real error matching something in
   `RLLM_VERL_INSTALL_NOTES.md`).

### 3. Where today's session left off, and the exact next action

**The core problem, in one sentence:** the model retrieves relevant evidence fine
(`hit_rate` 0.6-0.8 all session) but doesn't reliably cite it correctly
(`groundedness`≈0 in nearly every configuration tried).

**Four reward-design probes were run today** (each a short, fast, 25-step diagnostic —
NOT full training runs), searching for a reward shape that fixes this:
1. `beta=0.0` static — crushed the reward signal for ALL episodes, real collapse.
2. `beta` ramped fast (0.5→0.0) — healthy training, groundedness never moved.
3. `reward_mode="additive"` — healthy, but citation ATTEMPTS collapsed toward zero
   (model learned to stop trying to cite at all).
4. `reward_mode="cite_gated"` (hard zero for zero-citation rollouts) — **fixed the
   attempt-rate cleanly** (0.45→0.95+ citation-attempt rate), but the model found a
   SMALLER exploit right next to it: pasting exactly ~1 citation per answer regardless of
   how many the question actually needed, so `groundedness` still regressed to ~0 by the
   end.

**Then a likely bigger, separate root cause was found** by directly inspecting raw eval
episode logs (not aggregate metrics): every healthy run this whole session converges on
**exactly ~2 turns per episode** (one `search`, then answer directly — never `read`,
never a second hop), regardless of reward design. Checked the actual system prompt
(`env.py::_opening_prompt`) and found why: it tells the model in prose to
"search→read→answer" but the ONLY worked examples ever shown are isolated `search[...]`
and `answer[...]` syntax — **no `read[...]` example anywhere, no full multi-hop worked
trajectory.** The model may simply never have been shown the move it's being asked to
make. Full writeup: `TRAINING_HISTORY_LOG.md`'s "eval-time turn count" entry (bottom of
the file).

**Fixed already, verified, committed, NOT yet run on the pod** (the pod was closing —
this only needed a code edit + offline verification, no GPU time): `_opening_prompt` now
includes a full worked example (`search→read→search→read→answer`, citations tied back to
`read()` calls) — see the git log entry "Add worked multi-hop example... to prompt".

**➡️ THE VERY NEXT THING TO DO, before touching reward design again:**
```bash
cd /workspace/Agentic-RL-Alignment-Path/assignments/deep_research_agent
source .venv-deep-research/bin/activate
nohup python train_dr.py probe_cite_gated > run_probe_prompt_fix.log 2>&1 &
```
(Reuses the `cite_gated` reward design — the best-performing one so far on attempt-rate —
so the ONLY variable that changes from Probe 4 is the new prompt. Tail the log / check
W&B for `reward_components/groundedness` and the raw eval turn count, same way Probes
1-4 were watched — see `TRAINING_HISTORY_LOG.md` for the exact monitoring pattern.) If
turn count and `groundedness` move at all, a good chunk of today's citation problem may
turn out to have been a demonstration gap, not a fundamental RL sample-efficiency limit —
worth ruling out before reaching for the heavier cold-start SFT/RFT path in
`TENTATIVE_FUTURE_EXPERIMENTS.md`.

### 4. Full command cheatsheet — every preset + the launch/monitor/rename pattern used all session

**Every available preset** (`config.py`'s static methods, dispatched via `train_dr.py`'s
CLI `preset` arg):
```bash
python train_dr.py cloud                    # the real, full ~252-step/6h run (Config.cloud_preset())
python train_dr.py probe_beta0               # Probe 1: beta=0.0 static (DON'T rerun — real collapse, see log)
python train_dr.py probe_beta_ramp_fast      # Probe 2: beta ramped 0.5->0.0, steps 1-30
python train_dr.py probe_additive            # Probe 3: reward_mode="additive"
python train_dr.py probe_cite_gated          # Probe 4 / the queued next action above
python train_dr.py sanity                    # tiny stand-in model, offline, no GPU needed — CI-style check
python train_dr.py --dry-run <preset>        # print resolved config + veRL overrides, no rllm/GPU needed
```
Each probe preset is `steps=25, eval_every=0, checkpoint_every=0, push_checkpoints=False`
(see `config.py`) — fast, cheap diagnostics, not meant to produce a usable checkpoint on
their own. To build a NEW probe variant: copy one of the `probe_*` staticmethods in
`config.py`, change what you need to isolate, add it to `train_dr.py`'s CLI dispatch dict
(two one-line edits — search `probe_cite_gated` in both files for the exact pattern).

**The launch/monitor/rename pattern used for every run today** (copy this shape):
```bash
# 1. Launch in the background, tail to a log file:
cd /workspace/Agentic-RL-Alignment-Path/assignments/deep_research_agent
source .venv-deep-research/bin/activate
nohup python train_dr.py <preset> > run_<name>.log 2>&1 &
echo "PID: $!"   # note this — you'll need it to kill cleanly later

# 2. Watch for the W&B run URL (appears within ~30-60s):
grep "wandb: Run data is saved" run_<name>.log
# -> .../wandb/run-<timestamp>-<RUN_ID> — the RUN_ID is what you query via wandb.Api()

# 3. Pull REAL numbers via the W&B API, don't trust console-line guessing (a real bug was
#    found this session doing exactly that — see TRAINING_HISTORY_LOG.md's logging-bug entry):
python -c "
import wandb
from dotenv import load_dotenv
import os
load_dotenv('.env')
api = wandb.Api(api_key=os.environ.get('WANDB_API_KEY'))
run = api.run('happy22harpreet/deep_research_agent/<RUN_ID>')
hist = run.history(pandas=False)
for row in hist: print(row)
"

# 4. To kill cleanly and free the GPU (single A100 — only one run at a time):
kill -TERM <PID>; sleep 8; nvidia-smi --query-gpu=memory.used --format=csv

# 5. Rename the W&B run to record its final disposition (so TRAINING_HISTORY_LOG.md's
#    links resolve to something legible, not "Untitled Run"):
python -c "
import wandb
from dotenv import load_dotenv
import os
load_dotenv('.env')
api = wandb.Api(api_key=os.environ.get('WANDB_API_KEY'))
run = api.run('happy22harpreet/deep_research_agent/<RUN_ID>')
run.name = '<descriptive_name>_COMPLETE_or_KILLED_stepN_<one-line-outcome>'
run.update()
"
```
**Key metrics to check every time, all under `reward_components/*` and `batch/*` in
W&B** (added this session specifically because console-line inspection missed real
findings): `reward_components/groundedness`, `reward_components/pct_rollouts_with_
citation`, `reward_components/n_citations`, `reward_components/outcome`,
`batch/dead_groups_pct`, `response_length/clip_ratio`. Full mechanism for each in
`rllm_workflow.py`'s `_patch_tracking_log_for_extra_metrics` docstring.

**After any run finishes (or you kill it), update `TRAINING_HISTORY_LOG.md`** — copy the
entry template at its top, fill in real numbers pulled from W&B (not guessed), commit +
push. This is the doc that makes the NEXT handoff after yours possible — keep it current.

---

**You are a Claude Code session on a rented RunPod GPU (target: 1× A100 80GB).** A local
session built this lab end-to-end and handed it to you via git. Your job: resolve the
rLLM/veRL version matrix, wire the framework seams marked `# VERIFY`, run the sanity spike,
pass the masking gate, then do the real run + eval gate. **Read this whole file first.**

> **Historical note:** the paragraph above and everything through §2 describes the
> ORIGINAL handoff, from before this project had ever run real training. All of that is
> now done — kept below for reference/onboarding only, not as your next task list. Your
> actual next task is in the ⚡ START HERE section above.

> Mode note: the local build was **full-build** (Harpreet said "just build it, it goes
> straight to RunPod"). So — unlike the finqa handoff — the *core learning logic is already
> written and tested*: the reward, the trajectory/credit-assignment seam, the eval gate, and
> a runnable masking verifier. **You are NOT waiting on `TODO(harpreet)` stubs.** What remains
> is genuinely framework plumbing + running. If Harpreet appears and wants to rewrite the
> reward himself, that's his call; otherwise proceed.

---

> **2026-08-24 update — read this if you're on pod `m4j1fsfj84767c`:** the repo now lives
> at `/workspace/Agentic-RL-Alignment-Path/...` (Volume Disk, survives a pod stop), NOT
> `/root/...` anymore (Container Disk, wiped on stop). If you're picking this pod back up
> after a stop/start, read [`POD_LIFECYCLE.md`](POD_LIFECYCLE.md) FIRST — it has the resume
> steps and what does/doesn't survive. If you're on a fresh pod instead, this doesn't apply
> — just follow the rest of this file + `setup_pod.sh` normally.

## 0. Context (why this exists)
- **Capstone brief:** [`../../projects/06-deep-research-agent/README.md`](../README.md).
  Train a small model to do multi-step **search→read→synthesize** with a **grounded, cited**
  answer, via GRPO. Branch B = offline corpus (HotpotQA + 2WikiMultiHopQA).
- **Stack:** rLLM (agent layer, `AgentExecutionEngine`) + veRL (engine layer, GRPO update).
  Rationale + API primer: root [`../../VERL_RLLM_PRIMER.md`](reference/VERL_RLLM_PRIMER.md).
  Ops patterns (checkpoint/resume/HF-mirror/W&B/tmux): root [`../../RUNPOD_PLAYBOOK.md`](reference/RUNPOD_PLAYBOOK.md).
- **Success bar:** beat a no-training ReAct baseline by a clear margin on held-out multi-hop
  QA (aim for a meaningful share of Search-R1's ~+20% at 3B) **and** pass the groundedness
  gate (no judge-gaming / fabrication). One A100 is sufficient (see §5).
- **Every design decision** made during the build is logged in [`NOTES.md`](NOTES.md) — read it;
  it explains *why* the reward is shaped the way it is. Current-SOTA references (Proof-of-Use,
  Tree-GRPO, etc.) are in the lab [`README.md`](../README.md) "current-landscape check".

---

## 1. What's already built + TESTED (don't rewrite this)
All of this passes `pytest -q tests/` = **36 tests**, offline, no GPU:

| file | status |
|---|---|
| `corpus.py` | DocStore + pure-Python BM25 retriever (deterministic, offline). ✅ done |
| `data.py` | HotpotQA/2Wiki loaders + 4-question offline fixture + `selfcheck()`. ✅ (real path needs `datasets`; `# VERIFY` the 2Wiki mirror schema) |
| `tools.py` | `search`/`read`/`answer` ReAct tools over DocStore. ✅ done |
| `trajectory.py` | Trajectory/Step + `model_mask` + `retrieval_hit_rate()`. ✅ done |
| `metrics.py` | EM/F1 + groundedness floor. ✅ done |
| `citations.py` | Proof-of-Use citation verify: gold-membership + **citation-F1** (precision/recall) + fabrication. ✅ done |
| `judge.py` | groundedness judge (mock/**vllm**/hf) — **eval-only** (not used in training). ✅ done; vllm backend verified end-to-end on the pod 2026-08-23 |
| `reward.py` | **the layered reward** — `reward_deep_research` (gated: `o·(β+(1-β)·g) − tolls`). ✅ done + tested |
| `env.py` | `DeepResearchEnv` + ReAct parser + `_to_dr_trajectory` + `_terminal_reward` + **runnable** `assert_verl_masking_matches`. ✅ core done; `BaseEnv` import path CONFIRMED (`rllm.environments.base.base_env.BaseEnv`) 2026-08-23 |
| `config.py` | sanity/default/cloud presets + reward knobs + veRL Rosetta map. ✅ most hydra keys CONFIRMED against real installed verl (`ppo_trainer.yaml`) 2026-08-23 — `trainer.*` keys are right; `ppo_mini_batch_size` unit convention still `# VERIFY` |
| `train_dr.py` | trainer entry: `--dry-run`, Rosetta→hydra overrides, `--mask-check`, `--push-checkpoints`. ✅ **`python train_dr.py sanity` PASSES end-to-end** (2026-08-23): hydra config now properly composed (`resolve_verl_config`), dataset written to parquet + wired via `data.train_files`/`val_files`, `AgentTrainer` now uses `workflow_class=DeepResearchWorkflow`. `main()` now also calls `ray.init()` itself (with a `worker_process_setup_hook` + forwarded `PYTHONPATH`) before `AgentTrainer` gets the chance — required for the batch-safety patch below to reach the right Ray process; see WORKFLOW_PORT_NOTES.md bug 11/12 for why. |
| `rllm_workflow.py` | **NEW** — `DeepResearchAgent(BaseAgent)` + `DeepResearchWorkflow(Workflow)`, the HANDOFF-step-2 port. ✅ validated live: real GRPO step ran (rollouts→reward→actor update→weight sync) at both sanity (`group_size=2`) AND cloud-like (`group_size=16`) scale. See its module docstring for the masking-is-automatic discovery, and `_patch_make_iterator_for_ragged_batches` + `_ray_worker_setup_hook` for a real, VERIFIED-fixed blocker found this session: veRL's mini-batch divisibility assert crashes ~31% of steps at cloud-scale `group_size` (measured, not assumed) due to occasional trajectory-splitting; patched to gracefully fall back instead of crashing — confirmed working via the patch's own log line firing inside the correct Ray worker process. |
| `evaluate.py` | **framework-agnostic** eval CORE (rollout_episode/evaluate_set/probes/gate) unchanged + tested offline. Pod-facing path now **batched vLLM** (`run_batched_rollouts`, LoRA hot-swap for base/tuned) + **W&B eval logging** (`log_eval_to_wandb`) — no HF `generate()` anywhere. ✅ verified end-to-end on the pod 2026-08-23 |
| `hub.py` | **NEW** — checkpoint merge (verl's own `model_merger`, LoRA-aware) + HF push, branch = `{run_name}__{DD_MM_YYYY__HH_MM_SS}`, latest+best per RUNPOD_PLAYBOOK pattern #5b. ✅ mechanics done; not yet run against a real trained checkpoint (blocked on the AgentTrainer/Workflow fix above) |
| `setup_pod.sh` / `launch_cloud.sh` | pod bootstrap + tmux launcher. ✅ now also fixes the vllm `libcudart.so.13` linker-path bug (RLLM_VERL_INSTALL_NOTES.md bug 4) |

Sanity-check it yourself first thing:
```bash
cd assignments/deep_research_agent
python -m pytest -q tests/                 # expect 36 passed
python -c "import data; data.selfcheck()"  # BM25 surfaces gold evidence (offline)
python train_dr.py --dry-run cloud         # prints resolved config + veRL overrides
```

---

## 2. What's LEFT — updated 2026-08-23 after the Workflow port + sanity spike pass
Read [`WORKFLOW_PORT_NOTES.md`](WORKFLOW_PORT_NOTES.md) for the full story on everything
below marked DONE — it has the exact root cause + fix for each, not just the verdict.

1. **Version matrix.** ✅ DONE — [`RLLM_VERL_INSTALL_NOTES.md`](RLLM_VERL_INSTALL_NOTES.md).
2. **rLLM imports / base class.** ✅ DONE — `BaseEnv` = `rllm.environments.base.base_env.
   BaseEnv` (unchanged, still works); `AgentTrainer` = `rllm.trainer.agent_trainer.
   AgentTrainer` but needs `workflow_class`/`workflow_args`, not `agent_class`/`env_class`/
   `ToolAgent` (that API was removed — see `rllm_workflow.py`, the new adapter layer).
3. **Dataset format.** ✅ DONE — confirmed via source: the `task` dict a Workflow receives is
   the `extra_info` column of a parquet file `data.train_files`/`val_files` point at (verl's
   `RLHFDataset` convention). `train_dr.write_verl_dataset` writes it; `rllm_workflow.
   _task_dict_to_drtask` reconstructs a `DRTask` on the other end.
4. **veRL hydra keys.** ✅ DONE for everything the sanity spike actually exercises — every key
   in `train_dr.verl_overrides` is now either confirmed-correct-by-running or fixed after a
   real error (7 bugs, WORKFLOW_PORT_NOTES.md bugs 6–12). Genuinely still unverified: anything
   the CLOUD preset touches that sanity doesn't (e.g. `actor_rollout_ref.rollout.
   tensor_model_parallel_size>1` behavior, `push_checkpoints`/`keep_best_k` interacting with
   veRL's own checkpoint format for real — hub.py's merger expects `global_step_N/actor`, not
   yet run against an actual saved checkpoint).
5. **Global-step → env** (for the ramped efficiency toll). **✅ RESOLVED 2026-08-24 — the
   earlier "STILL OPEN, needs a Ray-actor-level monkeypatch" conclusion below (2026-08-23)
   was WRONG, found by reading the source more carefully, not by inventing new machinery.**
   `AgentWorkflowPPOTrainer.fit()` (`rllm/trainer/verl/agent_workflow_trainer.py`) already
   calls `self.agent_execution_engine.set_training_step(self.global_steps, ...)` every
   iteration — a real, existing hook, previously found but dismissed as only reaching
   episode-logging. Turns out `AgentWorkflowEngine` is constructed as a plain in-process
   attribute of the trainer (`init_workers()`), and workflow instances run via local
   `asyncio`/`ThreadPoolExecutor` — **same process as the trainer's own step loop**, not
   the cross-actor problem the batch-divisibility patch needed to solve. Fix:
   `rllm_workflow.py`'s `_patch_agent_workflow_engine_step_tracking` wraps
   `set_training_step` to also write into a module-level counter;
   `DeepResearchWorkflow.reset()` reads that instead of the never-set `self._global_step`.
   Plain import-time application — no Ray `worker_process_setup_hook` needed for this one.
   **Verified end-to-end with a real run** (temporary diagnostic prints, removed after
   confirming): `set_training_step` fired 1→2→3→4 (train AND the automatic final-eval val
   pass), and `DeepResearchEnv.from_dict` received the matching `global_step` on every
   episode, not just the patch's write side. Also revisited the ramp TIMING itself while
   fixing this (Harpreet's call, from prior-lab experience: penalizing turn-count before
   the policy has learned to use tools can suppress tool-calling early) —
   `cloud_preset()` now sets `lambda_eff_ramp_start=150, lambda_eff_ramp_end=230` (was the
   base default 60/200), leaving ~22 steps at full toll strength within the ~252-step/6h
   budget rather than firing too early.
   ~~The obsolete "STILL OPEN" writeup, kept for the record of what was actually checked:~~
   Traced the actual paths, no guesswork: `AgentWorkflowEngine.set_training_step()` only
   wrote `self.current_step` on the ENGINE instance (for episode *logging*, per its own
   docstring) — `process_task_with_retry` never passed it into `workflow.run(...)`'s kwargs,
   so there was NO OBVIOUS built-in path from there to a `Workflow` instance (missed that the
   engine and workflow share a process regardless). Checked the obvious alternative proxy
   too: `RolloutEngine.weight_version` exists on the BASE class but is never incremented by
   `verl_engine.py` (the concrete engine our veRL backend actually uses) — genuinely a dead
   end, unrelated to the real fix above.
6. **Masking gate — ✅ DONE, concretely re-verified 2026-08-23** (not just mechanism
   understanding anymore). Enabled rLLM's built-in `trainer.log_episodes=True` (dumps every
   Episode to JSON — see `train_dr.verl_overrides`) and inspected a full training step's real
   output. **The one signal that actually matters checked out directly**: `transform.py`'s
   `_process_trajectory` prints `"has no valid model_output, skipping"` whenever it has to
   drop a step for a missing/empty `model_output` — grepped the full training log for that
   string across all 8 real trajectories in the step; **it never fired**. That's direct
   evidence every step's `model_output.prompt_ids`/`.completion_ids` were genuinely populated
   at training time, not inferred from "training completed without error." (Note: the
   episode-logger JSON itself does NOT include `model_output`/`prompt_ids` — it's a
   deliberately human-readable subset, checked its source directly rather than assume; don't
   expect to see token IDs in those files, that's normal, not a sign of a problem.)
   `train_dr.run_mask_check` is still an unimplemented stub — this file-log + grep approach
   is a perfectly good STANDALONE substitute; a full `run_mask_check` implementation isn't
   required to trust this gate anymore.
7. **Chat template — checked for real, findings are honest and mixed.** Pulled real generated
   transcripts from the same episode logs. The tiny UNTRAINED 0.5B sanity stand-in does NOT
   follow the ReAct format cleanly: it frequently rambles through multiple pseudo `Thought:/
   Action:` pairs in one generation (imitating the few-shot example's own shape rather than
   committing to one action), which often means the trailing action doesn't parse — but this
   is HANDLED correctly, not a bug: `env._parse_react_action` returns `parse_ok=False`, and
   the env feeds back a proper recoverable error observation (`"error: unknown or unparsed
   action. Use one of: search, read, answer."`), which the model then sees on the next turn.
   By the LAST turn the model usually does manage a well-formed `answer[...]` action and the
   episode terminates cleanly via `TerminationReason.ENV_DONE` (confirmed: `done=False` on
   most intermediate steps was ALSO a real gap found+fixed this session — see
   `rllm_workflow.DeepResearchAgent.update_from_env`, it wasn't backfilling per-step
   `observation`/`reward`/`done`, purely cosmetic, didn't affect training). **Read as**:
   expected behavior for an untrained tiny model with a tight 160-token sanity budget and no
   SFT warm-start (same characteristic finqa_agent already documented — pure in-context
   learning off the demo, nothing has explicitly trained the format yet) — not evidence of a
   real problem, but also not something to assume improves at 3B without checking again after
   the real run.

---

## 3. Your steps (in order)

### Step 1 — resolve the version matrix — DONE 2026-08-23, read this first
**Resolved.** `setup_pod.sh` step 4 has the exact working install commands baked in
(`rllm@main` pinned to a commit SHA + `verl==0.9.0` override + `torch==2.11.0` + `vllm==0.22.1`
+ flash-attn built from source). Full blow-by-blow of the three real install bugs hit and
fixed (flash-attn's build-isolation/torch chicken-egg problem, a stale numpy<2 pin in rLLM's
`verl==0.8.0` conflicting with vllm, a hatchling build-dep casualty of `--no-build-isolation`)
plus a CUDA-version-mismatch red herring that cost real investigation time before the actual
cause (transient build contention) was found:
[`RLLM_VERL_INSTALL_NOTES.md`](RLLM_VERL_INSTALL_NOTES.md). **Read it before re-deriving
anything here** — if `bash setup_pod.sh` still fails, versions have likely drifted further
(these frameworks move fast); that file's failure-MODE lessons are still the fastest path to
the next fix.
- Historical note (kept for context): rLLM now also supports a **Tinker** backend — we use
  **veRL**. Search-R1 (`PeterGriffinJin/Search-R1`) is veRL-based and does this exact
  offline-corpus multi-turn RL — its pins were a useful cross-check during resolution.

### Step 2 — wire the `# VERIFY` seams (§2 items 2–5, 7)
Smallest changes that make `python train_dr.py sanity` construct the trainer without error.
Use `--dry-run` to iterate on the overrides without launching a run.

### Step 3 — SANITY spike (0.5B, a few steps)
```bash
bash launch_cloud.sh sanity        # uses config.sanity_preset(): 0.5B, fixture, 2 steps
```
Pure "does the rLLM↔veRL↔vLLM stack train one step together" check. Green = proceed.

### Step 4 — THE MASKING GATE (non-optional)
```bash
python train_dr.py --mask-check sanity
```
Wire `run_mask_check` to pull one rollout's `(input_ids, loss_mask, trajectory)` from rLLM/veRL
and call `env.assert_verl_masking_matches(tok, ids, mask, traj)`. This confirms veRL grades ONLY
model-generated tokens, never the retrieved passages. **Here the observations are long passages,
so a mask bug trains the model mostly on copied text and is invisible in the loss curve.** Do not
run the real job until this passes.

### Step 5 — real run (Qwen2.5-3B + LoRA, 1× A100)
```bash
bash launch_cloud.sh cloud         # config.cloud_preset(): 3B, real HotpotQA/2Wiki, GRPO
```
Follow `RUNPOD_PLAYBOOK.md`: `max_train_hours` cap (preset = 6h), `checkpoint_every=25`,
`push_checkpoints=True` (HF mirror survives a pod wipe), W&B dashboards, `keep_best_k=2`
(RL eval peaks then regresses — keep the best, not the last). Watch: mean reward, KL,
citation-F1, retrieval hit-rate, avg turns, dead-group fraction.

### Step 6 — eval gate (base vs tuned, honest)
```bash
python evaluate.py --adapter runs/deep_research_agent_cloud/best --judge hf
```
Prints the localized metric table + anti-hacking probes + PASS/FAIL. The gate passes only if
tuned beats base by the margin on EM **and** no probe fires **and** citation-F1 didn't regress.
Report base-vs-tuned honestly (see §6).

### Step 7 — report back
Append a root `BUILD_LOG.md` entry (resolved version matrix, eval numbers, what was hard),
update this folder's `README.md` worklog, commit + push so the local session syncs.

---

## 4. The reward (so you understand what you're training) — details in NOTES.md
Gated form (default), per finished trajectory on question q:
```
   r = outcome·(β + (1−β)·g)  −  w_fab·fab  −  λ_fmt·fmt  −  λ_eff(step)·n_steps
```
- `outcome` = EM of the citation-stripped answer vs gold (RLVR, verifiable). `reward_kind="f1"`
  is the denser fallback if too many GRPO dead groups.
- `g` = **citation-F1** vs the gold supporting set (`citation_backend="gold"`): precision
  punishes distractor/fabricated cites, recall punishes incomplete hops. This is the
  groundedness signal; **it's rule-based, so NO judge model is needed in training** (that's
  why the training reward needs zero extra GPU for a judge).
- `β` (0.5) = credit a right-but-ungrounded answer keeps → grounding *unlocks* outcome credit.
- `λ_eff` **ramps** (`config.lambda_eff_at`) — flat-from-0 trains tool-avoidance (finqa lesson).

The judge/NLI are **eval-only**, for the judge-vs-grounded anti-hacking probe.

---

## 5. Compute (back-of-envelope — details in NOTES.md)
Qwen2.5-3B (or 7B) + LoRA fits **1× A100 80GB** (~30–40GB used: 6GB base weights shared,
<1GB LoRA+optim, 6–10GB acts/grads chunked, 15–25GB vLLM rollout). LoRA gives the KL
**reference model for free** (adapters-off). No judge model at train time. veRL **colocated**
mode time-shares the GPU between rollout and training. Multi-GPU is a *speed* choice
(data-parallel rollouts / full-FT), **not** a fit requirement — start on one A100.

---

## 6. Gotchas (don't relearn the hard way)
- **Masking is the silent killer** — §4/Step 4. Long retrieved passages = lots of tokens to leak.
- **Dead groups**: all-wrong group → all EM=0 → zero advantage. If early training stalls,
  switch `reward_kind="f1"` (denser) or lean on the citation-F1 term (varies even when EM=0).
  Also confirm `temperature≈0.9` (greedy rollouts kill diversity → dead groups).
  **2026-08-24: now a real logged metric, not just a concept** — `batch/dead_groups_pct`
  (= `solve_none` + `solve_all`) in W&B, alongside a full `reward_components/*` breakdown
  (outcome, groundedness, cite_f1, tolls) averaged per step — see NOTES.md's matching entry
  for the mechanism. Observed for real on the current cloud run: dead-group fraction climbed
  from ~28% to ~59% over the first ~20 steps of the (aborted, `lr=1e-5`) attempt — this
  gotcha is not theoretical, it happened here.
- **max_turns**: preset is 8 (multi-hop needs search→read→search→read→answer ≥5). Too low
  retrains the finqa tool-avoidance regression. Don't lower it below the task's real hop count.
- **Hybrid-thinking models** (if you switch off Qwen2.5): pass the no-think flag at the
  chat-template call, or it emits `<think>` tokens. Qwen2.5-Instruct is not hybrid — fine.
- **2Wiki loader** (`data._load_2wiki`) — the HF mirror id + field names vary; `# VERIFY` and fix.
  HotpotQA (`hotpot_qa`, distractor) is the safer first dataset; you can train on it alone.
- **Secrets**: `HF_TOKEN`/`WANDB_API_KEY` in `.env` (gitignored, this folder). Never commit.
  W&B key is 40 chars from wandb.ai/authorize.
- **`ppo_mini_batch_size` unit convention** (prompts vs sequences, auto-×`rollout.n`?) — the
  finqa migration flagged this; VERIFY against installed veRL.
- **`verl.model_merger merge` on a `save_lora_only=True` checkpoint exits non-zero even
  when it succeeds** — RESOLVED 2026-08-24, hit for real while testing the periodic HF
  push (see `hub.py::merge_checkpoint`'s docstring for the full mechanism: it correctly
  writes the LoRA adapter, THEN unconditionally tries to also save a full dense model
  from what's left in `state_dict` — empty, since a LoRA-only checkpoint never had base
  weights — and its own validation raises on that empty save). `hub.merge_checkpoint`
  now checks for `lora_adapter/adapter_model.safetensors` directly instead of trusting
  the subprocess exit code. This affects BOTH `push_checkpoints()` (end-of-run) and the
  new periodic push — same underlying function.
- **`lr=1e-5` (the base `Config` default) is too low for LoRA-RL** — RESOLVED 2026-08-24,
  killed a real cloud run over this at step 31/252. `cloud_preset()` now sets `lr=1e-4`
  explicitly (see its own inline comment in `config.py` for the full evidence: reward
  growth flattened early on the real run, alongside the dead-groups climb above — and
  this exactly matches `assignments/finqa_agent/last_runpod_session.md`'s own documented
  `lr=1e-5` failure on a sibling agentic-RL lab, whose A/B probe found `lr=1e-4` gave real
  movement where `1e-5` was flat). LoRA generally wants a meaningfully higher LR than full
  fine-tuning (only a small param fraction updates) — ~1e-4-3e-4 is the common range: if
  you ever touch `cfg.lr` for this lab, don't default back down to the base `1e-5` without
  re-reading this.

---

## 7. Map of files & docs
```
   THIS LAB (assignments/deep_research_agent/)
     README.md   — worklog + design + current-landscape check
     NOTES.md    — every design DECISION + eval/probe spec + backlog   ← read this
     HANDOFF.md  — you are here
     FRESH_POD_SETUP_AND_SANITY_CHECK.md — ← READ THIS FIRST on any new pod (2026-08-25
                              policy: terminate between sessions, no stop/resume). The
                              fast-path checklist from bare pod to verified-working env,
                              with the 5 real infra bugs hit doing exactly that.
     RLLM_VERL_INSTALL_NOTES.md — the resolved version matrix + every install bug hit/fixed
                              (2026-08-23's bug set — FRESH_POD_SETUP_AND_SANITY_CHECK.md's
                              §4 has 5 MORE, found 2026-08-25 on a genuinely fresh pod)
     WORKFLOW_PORT_NOTES.md — the Workflow/AgentTrainer port: API discoveries (masking is
                              automatic!) + every bug hit getting `train_dr.py sanity` to pass
     INFERENCE_SERVING.md — how to pull a pushed checkpoint + serve it (vLLM/SGLang) later
     ONE_STEP_TUNING_VERL_RLLM.md — real-cloud-scale throughput tuning: every lever tried,
                              measured before/after, what got applied to cloud_preset()
     POD_LIFECYCLE.md — largely HISTORICAL as of 2026-08-25 (was for a stop/resume
                              workflow on the old pod m4j1fsfj84767c; current policy is
                              terminate, not stop — see FRESH_POD_SETUP_AND_SANITY_CHECK.md).
                              Still has real bug write-ups worth reading (the venv-move-
                              breaks-shebangs gotcha), just not the active plan anymore.
     TRAINING_HISTORY_LOG.md — chronological log of every real training attempt: config,
                              what was observed, what changed and why — read this to see
                              the actual lr search (and any future hyperparameter changes)
                              in order, without re-deriving it from W&B or git log
     TENTATIVE_FUTURE_EXPERIMENTS.md — speculative, NOT scheduled: cold-start SFT / RFT /
                              a DPO-SimPO-KTO preference stage targeting citation
                              discipline specifically, if pure GRPO keeps showing the
                              near-zero-groundedness bottleneck first seen in Attempt 3
     rft_diagnosis/ — NEW 2026-08-25, its own subfolder (kept separate so diagnosis
                              scripts/data don't clutter the main lab files):
                              RFT_PLAN_AND_MODEL_DIAGNOSIS.md — the plan to verify the
                              base model can do multi-hop tool use at all BEFORE resuming
                              GRPO — supersedes "run probe_cite_gated next" below until
                              this pipeline runs. Read this before launching any new GRPO
                              training on this lab.
     rllm_workflow.py — DeepResearchAgent + DeepResearchWorkflow (the HANDOFF-step-2 port)
     hub.py      — checkpoint merge (verl's model_merger) + HF push, branch/timestamp versioned
     *.py        — see §1 table;  tests/ = 36 passing offline tests

   REPO-LEVEL DOCS (root)
     VERL_RLLM_PRIMER.md  — how rLLM+veRL fit together (first-principles)
     RUNPOD_PLAYBOOK.md   — cheap/watchable/crash-proof rented-GPU run
     CLAUDE.md            — project rules (eval is non-optional; report to BUILD_LOG)
     projects/06-deep-research-agent/README.md — the capstone brief

   REFERENCE IMPLEMENTATION (the template this lab mirrors)
     assignments/finqa_agent/verl/  — finqa's rLLM+veRL scaffold + HANDOFF_VERL.md
     assignments/finqa_agent/masking.py — the masking discipline to cross-check
```

Start at §1 (verify the 36 tests pass), then §3 step 1. Good luck — the hard thinking (reward,
grounding, eval) is done; this is a plumbing-and-running job now.
