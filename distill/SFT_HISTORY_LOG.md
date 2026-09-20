# SFT history log — the stage, in order

**Date:** 2026-08-26. **Outcome:** a shipped LoRA adapter
(`harpreet22happy/deep-research-agent-sft`) that moved correct-and-properly-cited from
**0.3% → 30.7%** on a locked held-out set, with one known failure mode documented rather
than buried.

Companion to `TRAINING_HISTORY_LOG.md` (which logs RL runs), `DATA_COLLECTION_LOG.md`
(where the data came from), and `FORMAT_INVESTIGATION_LOG.md` (the harness audit earlier
the same day). This one is the SFT stage: what was built, in what order, what broke, and
what the numbers actually say.

---

## 0. Where this stage started

Earlier the same day, an audit found eight bugs in the agent harness and established that
`correct_rate = 0` had been measuring our own code. With a trustworthy harness, the real
picture was:

- the model retrieves fine but almost never `read`s
- **zero** of 320 rollouts were both correct and correctly cited
- four GRPO reward designs had failed to change this

Then the load-bearing discovery, from the very first teacher smoke run: **a citation only
scores if the agent actually called `read` on that passage.** Every healthy training run
converged on search→answer and never read — so a citation without a read is unscoreable
*by construction*. Four reward probes had been tuning the incentive on an action that never
happened.

That reframed the whole stage. The target was no longer "teach citation" but **"teach
read-before-cite"**, and SFT is the right tool for installing a behaviour that does not yet
exist.

**Scope, Harpreet's call:** SFT only today, RL next session on a fresh pod. That removed
`train_dr.py sanity` from the critical path (it belongs to the RL session) — noted because
it still has not been run on any pod.

---

## 1. Decisions, and why

### 1.1 No in-context examples during SFT — Harpreet's call

> *"you have to plan if you're gonna be putting the in-context examples back in model's
> system prompt during SFT or not, I believe not since we rely on model's weights to absorb
> that behaviour from data"*

Right, and it is also the *only* way the result means anything: with the worked example
still in the prompt, a good score cannot separate "the weights learned it" from "it copied
what was in front of it". Implemented as `cfg.include_worked_example` (default `True`, so
every historical run still reproduces byte-for-byte). Stripping it took the prompt from
3,136 → 1,238 chars, **61% shorter**, which also shortens every rollout of the eventual RL
run.

**The subtle part, which is silent if wrong:** three prompts exist — what the teacher saw
(with example + teacher-only instructions), what SFT trains on, and what RL/eval use. The
last two **must be byte-identical**, or the model is conditioned on instructions it never
receives. Since the prompt is masked, nothing crashes and the loss looks identical either
way. So `build_sft.py` **re-renders turn 0** rather than using the stored collection
prompt.

Corollary followed deliberately: the teacher's explicit "you MUST read before citing" line
is **not** added to the student's prompt. That behaviour is what SFT is installing;
putting it in the prompt would smuggle the crutch back in.

### 1.2 Tiering — use the teacher's failures too

Harpreet: *"are you gonna train smaller model with trajectories where even teacher wasn't
able to solve... our main aim is to teach the model the good format."*

Measured first: **0 of 16** trajectories had bad process — even the failures searched, read
the right passages, and cited. And of 3 wrong answers, one was our own format bug and one
was `Nairn, Scotland` scored against gold `Nairn`.

| tier | condition | graded |
|---|---|---|
| A | correct AND `cite_f1 ≥ 0.999` | every assistant turn |
| B | read ≥ half the gold passages, answer wrong | process turns; answer turn **skipped** |
| C | never found the evidence | excluded |

Rationale for B: keeping only solved questions teaches the format only on *easy* ones.
Rationale for skipping B's answer: ~5 tokens of ~170, but the highest-stakes span — a
confident wrong fact *with citations attached* is the reward-hacking shape.

**This decision turned out to be wrong, and the final result is why — see §5.**

### 1.3 Library: `peft` + a plain torch loop

Asked directly, including about unsloth. Answered with a dry-run rather than an opinion:
unsloth leaves torch/vllm/verl/flash-attn alone (my first concern was overstated) but
downgrades `datasets` 5.0.1→4.3.0 and `transformers` 5.5.4→5.5.0 and adds `xformers`. At
~1.4k examples × ~900 tokens we are not compute-bound, so it would trade a re-verification
of the RL stack for ~15 minutes.

TRL/HF `Trainer` rejected for a different reason: they want to own tokenization and
collation, and **the masking is the one component we have actually verified**. Also, early
stopping needs *generation* metrics (does it read? does it cite what it read?), which a
loss-over-a-dataset eval loop cannot produce.

### 1.4 Data splits, enforced in code

`splits.py`: `sft_collect` 4,000 / `sft_dev` 500 / `rl_train` 14,500 / `heldout_eval` 500 /
`reserve` 1,000. Disjointness asserted against real `task_id`s.

`sft_dev` and `heldout_eval` are deliberately separate: anything you *select* on stops
being an honest estimate. Every decision uses `sft_dev`; `heldout_eval` is touched once, at
the end.

Built after a real error — a teacher-vs-student comparison had been reported as
"the same 16 questions" when the true overlap was **0/16**, because `load_pool` re-shuffles
for each `n`. `diagnosis1.py`'s docstring warns about this; I read it and did it anyway.

---

## 2. What was built

| file | job |
|---|---|
| `teacher.py` | GPT as the policy inside the real env; threads, cost cap, resumable |
| `sft_data.py` | multi-turn trajectory → masked example (prefix-delta boundaries) |
| `build_sft.py` | tiering + prompt re-render + per-example masking assertion |
| `sft_train.py` | LoRA training, batch tuner, overfit sanity |
| `eval_sft.py` | three-arm generation eval + pre-registered gate |
| `validate.py` | re-derives stored trajectories against the real corpus |
| `splits.py` | the stage partition, with disjointness asserted |

**18 masking tests**, including two that deliberately break the mask to prove the checker
fires.

### The masking, demonstrated rather than asserted

Harpreet: *"hope you're taking care of masking properly, model should not be trained on the
system prompt or the tool responses."* From a real training example:

```
[MASKED  315t] <|im_start|>system You are Qwen...              (the prompt)
[GRADED   35t] Thought: Search for the 2013 OSN Cup stadium... | Action: search[...]
[MASKED  208t] <|im_start|>user search results: [1] ...        (TOOL OUTPUT)
[GRADED   41t] Thought: Read the 2013 OSN Cup passage...       | Action: read[...]
[MASKED  205t] <|im_start|>user [2013 OSN Cup] The 2013 ...    (TOOL OUTPUT)
[GRADED   27t] Action: answer[Pearl of Stadiums [2013 OSN Cup] [King Fahd...]]
```

13.9% of tokens carry loss. Boundaries are derived from the tokenizer (prefix-delta), not
from string-matching an assistant marker, so they cannot drift if the chat template
changes.

---

## 3. Defects found while building

### 3.1 My own masking checker raised a FALSE POSITIVE

It substring-matched `"search results:"` in graded text — and fired on a perfectly good
trajectory where the teacher wrote, in its own reasoning, *"Both films have their countries
of origin in the search results: ..."*.

Flagging correct data as corrupt is its own bug: it nearly had me "fixing" a working
collector, and a checker that cries wolf gets switched off. Replaced with a **structural**
invariant — decode each contiguous graded run and require it to be text the model actually
produced. No phrase can spoof it.

Then tightened again: the first structural version also accepted the graded run being a
*superset* of a model turn (for tokenizer slack) — which is exactly the dangerous
direction, since a run extended backwards into a tool response still contains the model's
turn. Now one-directional, with tests for both failure shapes.

> Memory/batch-size work has its own doc:
> [`BATCH_SIZE_TUNING.md`](BATCH_SIZE_TUNING.md), including an unexplained ~40 GiB left
> as an open exercise.

### 3.2 Gradient checkpointing was never active

Enabled *before* `get_peft_model`, so the wrapper never inherited it: **24.2 GiB at batch
size 1** on a 3B LoRA, OOM at batch 4 on an 80 GiB card. Found by printing peak memory
rather than just recording the largest batch that survived.

### 3.3 The loss materialised [B, T, 152k] logits in float32 at once

Qwen's vocab is ~152k. Now chunked, summed over graded tokens and divided **once** — a
per-chunk mean would silently overweight short chunks.

### 3.4 My batch tuner was wrong in the way its own docstring warned about

It probed at p99 (2,038 tokens) while the dataset max is 2,240, and recommended batch 4 at
75/79 GiB. Re-probing at the true max, **batch 4 OOMs** — the "dies late in an epoch"
failure. Now probes worst case and selects on **throughput with headroom**, not
largest-that-fits (the biggest batch is also *slower per token* here, because the big-vocab
loss dominates).

| batch | peak | % card | tok/s |
|---|---|---|---|
| 1 | 26.1 GiB | 33% | 3,753 |
| **2** | **46.2 GiB** | **58%** | **5,195** ← chosen |
| 4 | OOM at max length | | |

### 3.5 Gradient accumulation was a mean of means — Harpreet caught this

> *"you're gonna use gradient accumulation then?"*

The textbook line `(loss / grad_accum).backward()` where `loss` is already a per-batch
*mean* weights every micro-batch equally regardless of how many graded tokens it holds.
That matters here: graded-token counts vary a lot (2-turn vs 7-turn trajectories, and
tier B drops the answer turn), so short trajectories' tokens would count for more.

Fixed to exact token-level averaging: each micro-batch backwards its **sum**, and the
accumulated gradient is divided once by the total graded tokens in the window.

### 3.6 An overfit check that FAILED for the wrong reason

After that change, the sanity check reported FAIL. Cause: 16 examples at effective batch 16
= **1 optimizer step per epoch**, 6 total, learning rate annealed to 8e-10 by step 5. The
wiring was fine; the test was misconfigured. That is the kind of thing that sends you
debugging a healthy system, so the trainer now refuses an overfit check with <20 steps and
says what to change.

**Overfit sanity, properly configured: 0.679 → 0.0066. PASS.**

---

## 4. The training run

418 tier-A trajectories (30% strict-gate yield from 1,406 collected), all distinct
questions. LoRA r=32/alpha=64 on all attention + MLP projections (59.9M params, 1.9%),
lr 1e-4 cosine, batch 2 × grad-accum 8, 3 epochs, ~7 minutes.

```
epoch 0   train 0.31   val 0.2722
epoch 1   train 0.25   val 0.2565
epoch 2   train 0.21   val 0.2563
```

Train loss falls steadily; **val flattens after epoch 1** (0.2565 → 0.2563). Mild
memorisation setting in — 2 epochs is probably enough at this dataset size, and it is
exactly why the plan puts early stopping on held-out *metrics* rather than train loss.

---

## 5. Results, and what they honestly say

> **Full tables — both eval sets, all three arms, every breakdown — are in
> [`RESULTS.md`](RESULTS.md).** This section is the interpretation; that one is the data.

**Held-out set, 300 questions, greedy, never trained or tuned on:**

| | base | base + worked example | **tuned** |
|---|---|---|---|
| correct (exact match) | 0.030 | 0.193 | **0.417** |
| picked the right sources | 0.007 | 0.291 | **0.744** |
| **verified them (read first)** | 0.003 | 0.198 | **0.810** |
| citation-F1 (the reward) | 0.002 | 0.110 | **0.710** |
| called `read` at least once | 0.070 | 0.360 | **0.993** |
| finished cleanly | 0.537 | **0.927** | 0.873 |
| never used a tool | 0.897 | 0.107 | **0.000** |
| **correct AND properly cited** | **0.0%** | **0.3%** | **30.7%** |

All five gate conditions passed — and they were written into the code *before* any adapter
existed, so they could not be fitted to the result.

### The third arm was the most valuable design decision here

Harpreet approved adding "base **with** the worked example" as a third arm — the question a
skeptic asks: *was fine-tuning worth it, or would a prompt have done the same job for free?*

It immediately changed the interpretation. **Most of the raw correctness gain is available
from prompting alone** (0.030 → 0.193). Fine-tuning adds 0.193 → 0.417 on top — real, but
not the whole story. The adapter's *distinctive* win is narrow and specific:

- read-before-cite **0.810 vs 0.198** — 4× what prompting achieves
- correct-and-properly-cited **30.7% vs 0.3%**

So the honest claim is: **SFT did not teach the model to answer better so much as to verify
its sources.** Exactly what it was aimed at. Without that arm, the correctness delta would
have been reported as a general capability win, which it is not.

### The 2-turn collapse is gone

Tool calls per episode now peak at **5 (59% of episodes)** = search→read→search→read→answer.
Searches 2.28 vs reads 2.31, near-perfectly balanced. The prompted model still peaks at 2–3
— the exact pattern every run in `TRAINING_HISTORY_LOG.md` converged to. Reads per episode:
0.11 → 0.59 → **2.28**.

### The known failure — 16% never terminate

Harpreet asked whether to be concerned. The answer was yes, and the data is unambiguous:

| | n | correct | cite_f1 | produced an answer |
|---|---|---|---|---|
| finished under the cap | 253 | **0.482** | **0.817** | 100% |
| hit the 8-call cap | **47 (16%)** | **0.064** | **0.131** | 19% |

The capped episodes are total losses, not slow ones — they return nothing.

Harpreet's hypothesis (*"it could be a case model is unnecessarily reading"*) was partly
right and the data refined it: stuck episodes read 1.53× what the question needs, but
**search 2.4× more** (4.36 vs 1.81). It is a failure to *commit*, not to read.

**Cause:** trained only on trajectories where the teacher **succeeded**, so the model never
saw a demonstration of concluding under uncertainty. Train on successes and you teach the
happy path with no fallback.

**Consequence for the plan:** §1.2's decision to skip tier B's answer turn now looks wrong.
It teaches process while never demonstrating conclusion — precisely the broken habit.
Re-test with the answer turn graded before accepting the plan as written.

---

## 6. Things I got wrong this stage

Kept deliberately; a log of only wins teaches the wrong method.

1. Reported a teacher-vs-student comparison as controlled when question overlap was 0/16.
2. Deleted the read-before-cite sentence from the teacher prompt while patching something
   else (cost: read_before_cite 0.938 → 0.79 until restored).
3. Wrote a masking check that raised a false positive on good data.
4. Wrote a re-render assertion that was vacuous — it checked for a string that no longer
   existed.
5. Enabled gradient checkpointing before the peft wrap, so it never applied.
6. Built a batch tuner that probed p99 instead of max and recommended a size that OOMs.
7. Used mean-of-means gradient accumulation.
8. Overstated the risk of unsloth before checking (the dry-run was far milder than my
   framing).

Four of these were surfaced by Harpreet asking a plain question — go incremental, validate
the data, which library, are you using gradient accumulation, is it over-reading. None was
a code review; each was a question about intent that made a defect visible.

---

## 7. For the next session

- **`train_dr.py sanity` has never been run on any pod today.** RL depends on it.
- Start GRPO from **this adapter**, not the base model, with
  `include_worked_example=False` (training and rollout prompts must match).
- Use `rl_train` (14,500 questions, disjoint — asserted). Note `cloud_preset` uses
  `prompts_per_step=1`, so a 252-step run touches ~252 distinct questions; raise that
  rather than assuming the pool size does the work.
- **The RL target is unusually well-defined:** the non-termination failure. Zero outcome
  reward for never answering is exactly the signal GRPO sharpens well, and unlike Attempts
  1–4 and Probes 1–4, the behaviour already exists 84% of the time rather than needing to
  be invented.
- Reuse `eval_sft.py` for the three-arm comparison; do not rebuild it.
