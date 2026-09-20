# SFT → RL plan: data, evals, and the in-context-examples decision

**Date:** 2026-08-26. **Status:** plan, agreed in outline with Harpreet; open calls marked.
**Supersedes** `RFT_PLAN_AND_MODEL_DIAGNOSIS.md`'s Diagnosis 2 procedure, which assumed
rejection-sampling from the *student's own* rollouts. That is dead: across 320 student
rollouts, **zero** trajectories were both correct and correctly cited, so there was
nothing to mine. We are on that doc's own documented fallback — a stronger teacher.

---

> **Status update 2026-08-26 (late): the SFT half of this plan is DONE and shipped.**
> What actually happened, in order, is in
> [`SFT_HISTORY_LOG.md`](SFT_HISTORY_LOG.md) and [`DATA_COLLECTION_LOG.md`](DATA_COLLECTION_LOG.md).
> **One decision below was proven wrong by the result** — §3's "tier B skips the final
> answer turn". The trained model's dominant failure is that 16% of episodes never
> conclude, and skipping the answer turn teaches process while never demonstrating
> conclusion. Re-test with it graded before following §3 as written.

## 0. Why this document exists

Two things have to be decided before any training, and both are easy to get wrong in ways
that invalidate everything downstream:

1. **Which questions belong to which stage**, enforced in code rather than remembered.
2. **What prompt the student sees** at SFT time, at RL time, and at eval time — which must
   be *the same prompt*, and which is where the in-context-examples question actually
   lands.

This session already produced one live example of (1) going wrong: a teacher-vs-student
comparison was reported as controlled when the real question overlap was 0/16. That is
what `splits.py` now prevents.

---

## 1. Data — who gets which questions

Defined in `splits.py`, disjointness asserted against real `task_id`s (not index
arithmetic — an off-by-one would pass an arithmetic check and silently leak SFT questions
into RL, invisible in any training curve).

| split | n | used by | may it ever be trained on? |
|---|---|---|---|
| `sft_collect` | 4,000 | teacher generates trajectories → SFT training data | **yes** |
| `sft_dev` | 500 | early-stopping + model selection, for BOTH SFT and RL | no |
| `rl_train` | 14,500 | GRPO rollouts | **yes** (RL only) |
| `heldout_eval` | 500 | the final reported numbers, once | **never** |
| `reserve` | 1,000 | unplanned future stage | — |

Upstream there are ~105,000 questions (HotpotQA 90,447 + 2Wiki 15,000), so none of these
sizes is constrained by data. They are constrained by cost and by run length.

**`rl_train` is deliberately 3.6× `sft_collect`.** If RL trains on questions the policy
memorised during SFT, every rollout in a group produces the same answer, the group's
advantage is zero, and it contributes no gradient — exactly the `dead_groups_pct` climbing
to 84% that `TRAINING_HISTORY_LOG.md` already records. RL needs questions the policy finds
genuinely uncertain.

> **Note when configuring GRPO:** `cloud_preset` uses `prompts_per_step=1`, so a 252-step
> run consumes far fewer distinct questions than 14,500. The headroom is free and worth
> keeping, but "lots of questions for RL" is satisfied long before the pool is exhausted —
> the binding constraint is `prompts_per_step × steps`, and that is worth raising instead
> of assuming the pool size is doing the work.

### Why `sft_dev` and `heldout_eval` are separate

`sft_dev` is used for every *decision*: when to stop SFT, which checkpoint to keep, when to
stop RL. Anything you select on is no longer an unbiased estimate of performance, however
carefully it was held out.

`heldout_eval` is touched **once**, at the end, to report base vs SFT vs SFT+RL. Because
nothing is ever selected on it, it stays a clean number, and because all three stages report
against the same 500 questions, the three numbers are comparable to each other.

**Consequence to accept up front:** Diagnosis 1's numbers (measured on train-pool questions)
are *not* comparable to anything reported from here on. Do not put them in the same table.

---

## 2. The in-context-examples decision — Harpreet's call, and I agree

> *"you have to plan if you're gonna be putting the in-context examples back in model's
> system prompt during SFT or not, I believe not since we rely on model's weights to absorb
> that behaviour from data"*

**Decision: no worked example in the student's prompt, at any stage after SFT.**

That is the right call and it is also the actual experiment. `RFT_PLAN_AND_MODEL_DIAGNOSIS.md`
step 4 says the same thing for the same reason: with the example still in the prompt, a good
result cannot distinguish "the weights learned it" from "it copied the example sitting in
front of it". Removing it is what makes the result mean something.

There is a second, practical reason. The current `env._opening_prompt` is 3,136 chars, and
**the worked example is roughly two-thirds of it** (lines 14–36 of 38). Dropping it shortens
every prompt of every rollout for the whole RL run — real savings in a context where
`max_new_tokens` is 256 and long observations already compete for room.

### The part that is easy to miss: three prompts currently exist

| | prompt | contains |
|---|---|---|
| `P_collect` | what the **teacher** saw | env prompt **+ worked example** + teacher-only suffix (be terse, read-before-cite, exact `Action: answer[...]` form) |
| `P_train` | what SFT **conditions on** | to be decided ← this document |
| `P_infer` | what RL rollouts and eval use | `env._opening_prompt`, **currently with the worked example** |

**`P_train` and `P_infer` must be byte-identical.** Training the model to expect one prompt
and then rolling it out under another wastes the SFT stage — the model conditions on the
prompt even though the prompt tokens are masked out of the loss.

**Canonical student prompt (`P_student`), used for `P_train`, RL rollouts, and eval:**

- KEEP lines 0–12 of the current prompt: the task instruction, the terseness rule, the
  citation rule ("Do not cite a passage you did not read"), the tool menu, and the
  `Thought:/Action:` format spec. This is *task specification* — the model cannot act
  without knowing which tools exist or what syntax to use.
- DROP the worked example (lines 14–36). This is *demonstration*, and demonstration is
  what we are moving into the weights.
- DROP the teacher-only suffix entirely. In particular **do not** add the teacher's explicit
  "you MUST read before citing" instruction to the student. That behaviour is exactly what
  SFT is supposed to install; putting it in the prompt would re-introduce the crutch through
  the back door and make the result unreadable again.

### The mechanic this requires

Trajectories are collected under `P_collect`, but must be **encoded under `P_student`**. So
`encode_trajectory` has to *replace turn 0's content* rather than use what was stored. Since
the prompt is masked, this changes no loss — but it changes what the model conditions on,
which is the whole point. Getting this wrong is silent: the loss curve looks identical.

**Open (small) call:** the teacher's trajectories were *produced* under a prompt containing
a worked example. That is fine and intended — it is how we got good demonstrations — but it
does mean the teacher's behaviour is mildly example-conditioned. Not worth correcting; worth
writing down.

---

## 3. Which trajectories to train on — tiering

From the discussion on whether to use questions the teacher failed. Measured on the first
16 collected: **0 of 16 had bad process**; even the failures searched, read the right
passages, and cited. And of 3 wrong answers, one was our own format bug (since fixed) and
one was `Nairn, Scotland` scored against gold `Nairn`.

| tier | condition | graded |
|---|---|---|
| **A** | correct **and** `cite_f1 ≥ 0.999` | every assistant turn |
| **B** | read ≥ half the gold passages, answer wrong | search/read/reasoning turns; **final answer turn skipped** |
| **C** | never found the gold evidence | excluded entirely |

Also excluded from grading everywhere: any turn the environment rejected (already
implemented — otherwise we would train the student to emit a format its own env refuses).

**Why B is included:** keeping only solved questions teaches the format only on *easy*
questions. The hard ones — where careful searching and reading matter most — would be absent.
That is the narrowness trap the plan doc's data-hygiene section warns about.

**Why B's answer turn is not graded:** it is ~5 tokens out of ~170, but it is the
highest-stakes span in the trajectory, and training on it teaches a confident wrong fact
*with citations attached* — the exact reward-hacking shape. It would also hurt RL: a policy
that confidently emits one memorised wrong answer produces zero-variance groups and no
gradient.

**Measure, don't assume:** train A-only and A+B, compare on `sft_dev`. Cheap, and it settles
whether B helps or dilutes.

---

## 4. Evaluation plan

Every stage reports the **same breakdown**, using `diagnosis1.classify_trajectory` unchanged,
so numbers mean the same thing across stages:

`correct_rate` · `title_f1` (right sources) · `read_before_cite_rate` (verified them) ·
`cite_f1` (both, the reward's own number) · `calls_read_rate` · `terminated_cleanly_rate` ·
hop-count distribution · the wrong/uncited/miscited/correct-and-cited buckets.

Reporting `title_f1` and `read_before_cite_rate` separately is now non-negotiable: the single
conjunctive `cite_f1` is what disguised "the model never reads" as "the model cannot cite"
for four reward-design probes.

| when | on | purpose |
|---|---|---|
| before SFT | `heldout_eval` | **base-model baseline.** Must exist before training, or there is nothing to compare to |
| during SFT, every N steps | `sft_dev` (128-question slice, for speed) | early stopping. Watch **generation metrics, not loss** — falling train loss with flat held-out metrics IS the overfitting signal |
| after SFT | `heldout_eval` | base vs SFT, no ICL examples. **The gate for whether RL is worth starting** |
| during RL, periodically | `sft_dev` | the training curve + early stop |
| final | `heldout_eval` | base vs SFT vs SFT+RL, one table |

**Anti-hacking probes at the final gate** (per CLAUDE.md — reward going up is not a result):
response-length inflation, citation-count inflation (Probe 4's "paste exactly one citation"
exploit), and the LLM judge as an independent cross-check on groundedness.

### Gates, written down before looking at results

- **SFT gate (must pass to justify RL):** on `heldout_eval`, versus base — `calls_read_rate`
  and `read_before_cite_rate` both improve substantially, `correct_rate` does not regress,
  and `terminated_cleanly_rate` ≥ base. The *point* of SFT here is process, so process is
  what the gate is on.
- **RL gate:** `correct_rate` on `heldout_eval` improves over the SFT checkpoint without
  `cite_f1` regressing (this is precisely where Probes 1–4 failed) and without triggering an
  anti-hacking probe.
- **Numbers are noisy at these sample sizes.** Measured this session: three identical runs
  gave `correct_rate` 0.188 / 0.156 / 0.141 at n=64. `heldout_eval` at n=500 is tighter, but
  do not treat sub-2-point differences as real without more samples.

---

## 5. Order of operations, with costs

| # | step | cost | gate before proceeding |
|---|---|---|---|
| 1 | finish `sft_collect` (4,000 q, k=1), in batches, validating between | ~$8 | `validate.py` passes; yield and diversity look sane |
| 2 | build the SFT dataset (tiering + `P_student` re-render) | $0 | masking tests pass on real data |
| 3 | base-model baseline on `heldout_eval` | GPU only | — |
| 4 | sanity-overfit ~16 examples | GPU, minutes | loss → ~0; proves the loss/masking wiring |
| 5 | real LoRA SFT, early-stop on `sft_dev` | GPU, ~1h | held-out metrics improving, not just loss |
| 6 | SFT gate on `heldout_eval` | GPU | the SFT gate above |
| 7 | GRPO from the SFT checkpoint on `rl_train` | GPU, hours | the RL gate above |

**Not yet built:** the SFT trainer itself. This lab has no SFT harness — only GRPO via
`train_dr.py`. `assignments/sft_min` is the reference pattern for the masked loss and is to
be followed, not reinvented.

**Still not run, and it should be:** `python train_dr.py sanity`, the end-to-end GRPO check
from `FRESH_POD_SETUP_AND_SANITY_CHECK.md` §5. Nothing measured so far depended on it, but
step 7 does. Run it before step 7, ideally before step 5.

---

## 6. Open calls for Harpreet

1. **`k` for collection.** k=1 over 4,000 questions maximises question diversity; k=2 over
   2,000 gives per-question variety at the same price. Diversity across questions is more
   valuable here (the plan doc's narrowness warning), so the plan assumes **k=1** — say if
   you'd rather have variety per question.
2. **Tier B in or out of the first SFT run.** Plan is to train both and compare on `sft_dev`;
   if you'd rather just pick one, A-only is the conservative choice.
3. **The citation bar for tier A.** Currently `cite_f1 ≥ 0.999` (perfect). At ~35% strict
   yield that gives ~1,400 trajectories from 4,000 questions — enough. Loosening to ≥ 0.5
   roughly doubles it but admits partially-cited multi-hop answers.
4. **Whether to strip the worked example from `env.py` itself**, or keep `env.py` unchanged
   and pass `P_student` as a config-level prompt variant. The second is safer for
   reproducing historical runs; the first is simpler. Plan assumes a **config-level variant**.
