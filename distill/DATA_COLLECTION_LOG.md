# Teacher data collection — decisions, defects, and what the data looks like

**Date:** 2026-08-26. **What:** generating multi-hop research trajectories with a
closed-source teacher, to fine-tune Qwen2.5-3B. **Companion docs:**
`SFT_HISTORY_LOG.md` (the training stage), `SFT_RL_PLAN.md` (the plan this executes),
`FORMAT_INVESTIGATION_LOG.md` (the harness audit that preceded it).

---

## 0. Why a teacher at all

Rejection-sampling RFT from the student's own rollouts was the plan. It died on a
measurement: across **320 student rollouts** (five format variants, 64 each),
**zero** trajectories were both correct and correctly cited. Best citation-F1 among the
19 correct answers was 0.667. There was nothing to mine.

`RFT_PLAN_AND_MODEL_DIAGNOSIS.md` had already written down the fallback for exactly this
case — seed from a stronger model. Harpreet's call to invoke it: *"do you want to use
closed source openAI model to build some dataset to SFT this model using bracket format?"*

---

## 1. Decisions made, and the reasoning

### 1.1 The teacher drives the REAL environment; it does not write transcripts

The single most important design choice. Asking GPT to *write* trajectories would have it
invent search results that do not exist in our corpus, and we would fine-tune the student
on fiction — undetectable in a loss curve.

Instead GPT is a drop-in replacement for the policy inside the same `DeepResearchEnv` the
student uses: **it chooses the action, our corpus answers, `env.step()` does the
bookkeeping.** Every observation is real, and every trajectory is scorable by the same
`citations.py` / `metrics.py` that scores the student.

`validate.py` enforces this rather than trusting it: it re-runs every stored tool call
against the corpus and byte-compares the result to what was recorded.

### 1.2 `gpt-4.1-mini`, deliberately not the strongest model

Harpreet's constraint: *"I don't want you to try out some very strong model also, since
don't want to inject too much reasoning as well (too verbose reasoning, I don't think our
small model could do)."*

This is the core constraint of distillation and the instinct was right. The GPT-5.x family
are *reasoning* models emitting long hidden chains; a 3B student imitating those learns to
start a chain it cannot finish. `gpt-4.1-mini` is non-reasoning, strong at tool use, and
$0.40/$1.60 per 1M tokens. The teacher prompt tightens it further: *"Write your Thought as
ONE short sentence, at most 20 words."*

Model list was read from the API (126 available) rather than recalled, and pricing fetched
from the pricing page rather than assumed.

### 1.3 Bracket/ReAct format, not OpenAI native function calling

GPT would do better with native calling. But `env.py` speaks bracket format and GRPO will
roll out in bracket format — training the student on a syntax it will never be rolled out
in wastes the entire stage.

### 1.4 Threads across episodes, sequential within one

The opposite of the vLLM path, deliberately. vLLM wants lockstep batching because one
engine serves all sequences; the OpenAI API is network-bound per request, so the win is
many episodes in flight. Turns *inside* an episode are inherently sequential — turn N+1
depends on turn N's tool result.

### 1.5 Incremental, resumable, cost-capped

Harpreet: *"maybe we should not attempt complete generation in one go, and proceed step by
step, fixing any issues coming along without wasting too much money."*

- appends to JSONL as each episode lands (an interrupted run keeps what it paid for)
- resumes by default; `--batch N` means N questions not yet done
- `--max-cost` fires **mid-run** (episodes are submitted in waves) — a cap checked after
  everything is queued is not a cap
- `--inspect` re-scores what is on disk for $0, the between-batches step

This paid for itself twice within the first two batches (see §2).

### 1.6 Which questions — `sft_collect` only

From `splits.py`: 4,000 questions reserved for collection, disjoint from `sft_dev` (500),
`rl_train` (14,500), and `heldout_eval` (500). Disjointness is asserted against real
`task_id`s, not index arithmetic.

---

## 2. Defects found during collection — all six, in order

Every one was found by looking at stored bytes rather than at a summary.

### 2.1 `history: null` on every single record — would have wasted the whole run

The collector recovered the conversation via `getattr(traj, "_history", [])`, and
`Trajectory` has no such attribute. `sft_data.encode_trajectory` consumes exactly that
field, so **every collected record was unusable for SFT** while looking perfectly healthy
in the summary output.

Found only because Harpreet asked to go incrementally instead of running 4,000 at once.
A single big run would have produced a complete, expensive, worthless collection.

### 2.2 The citation metric appeared broken — it was not

First smoke run: the teacher cited **both** gold titles and scored `cite_f1 = 0.000`.
Inspected by hand before changing anything. Not a bug:

```python
tp_titles = distinct_cited & gold & read_titles
```

A citation counts only if the agent actually called `read` on that passage — deliberate
"cite-what-you-read" (`citations.py` docstring, `NOTES.md` 2026-08-14). The teacher had
cited from search snippets without opening them.

**This is the single most valuable thing the collection turned up**, and it is about the
GRPO history, not the teacher: every healthy training run converged on search→answer and
never read, so **a citation without a read is unscoreable by construction.** Four reward
designs were tuning the incentive on an action that never happened. Adding one instruction
to the teacher took the smoke run from `cite_f1` 0.000 to 1.000.

### 2.3 The teacher dropped the `Action: answer[...]` wrapper on the final turn

5 of 7 parse failures were `Answer: Kevin Smith [Silent Bob Speaks]` instead of
`Action: answer[Kevin Smith [Silent Bob Speaks]]` — correct syntax on every search and
read, wrong on the last turn only. One episode never recovered and scored wrong.

Cost worse than a wasted turn: a rejected turn is still an *assistant* turn, so it would
be **graded**, training the student to emit a format its own environment rejects. Two
fixes: an explicit instruction (`parse_ok_rate` 0.500 → **1.000**, measured), and
`encode_trajectory` now refuses to grade any turn the env rejected while keeping it in
context so the recovery stays learnable.

### 2.4 API give-ups could become training examples

16 workers produced 60 transient errors in 268 calls (a tokens-per-minute limit, not a
request limit). When a call finally gives up it returns `""`, which the env treats as a
parse failure — a silent network failure becoming a demonstration of emitting nothing.

Now tracked end-to-end as `api_gave_up` and dropped by the dataset builder. Workers
lowered 16 → 8 → 6. **Final count: 1 give-up in 4,000 episodes.**

### 2.5 I deleted the most important instruction in the teacher prompt

While patching in the answer-format fix (§2.3), the edit **replaced** the read-before-cite
sentence instead of adding alongside it. That sentence is the most load-bearing line in
the prompt — cite-what-you-read is the entire behaviour this stage exists to install.

Measured cost: `read_before_cite_rate` **0.938** (pilot, sentence present) → **0.79–0.84**
(later batches, sentence gone). Restored, with a comment saying why it must not be edited
away again.

### 2.6 The check that should have caught 2.5 was vacuous

`build_sft.py` asserted the teacher suffix was absent from the rebuilt prompt by searching
for a string that no longer existed anywhere. It passed for the wrong reason. Now anchored
to text genuinely present in the collection prompt, *plus* an assertion that the anchor IS
there — **a check that cannot fail is not a check.**

---

## 3. What the data actually looks like

**Final — collection COMPLETE.** All 4,000 questions in the `sft_collect` split, one trajectory each:

```
episodes 4,000   distinct questions 4,000   (k=1: breadth over per-question variety)
correct            0.759
cite_f1            0.665
read_before_cite   0.789
api_gave_up            1

buckets:  {'correct_miscited': 1808, 'correct_and_cited': 1210, 'wrong_answer': 964, 'correct_uncited': 18}
```

**Strict-gate yield = 30%** (1,210/4,000 correct AND perfectly cited). The shipped adapter
used only 418 of these; **1,210 are now available**, so retraining on the full set is free upside.

### Cost

| | |
|---|---|
| spent | **$8.88** for 4,000 episodes |
| per episode | ~$0.0021 |
| tokens | 22.2M in (3.39M cached, 15%) / 622k out |
| calls | 17,265 |

Cost is **measured**, not estimated: token counts come from each response's own `usage`
field. The only assumed number is the cached-input discount multiplier. Reported as an
estimate, never as billing truth.

**Rate limiting dominated the wall clock**, not cost: 4,126 transient errors across 17,265
calls. All retried successfully with backoff. We are TPM-limited, so fewer workers does not
reduce throughput much — the retries *are* the throttle. 4,000 episodes took ~1h45m at ~33/min.

---

## 4. Notes for whoever collects next

- **`--inspect` before buying more.** It re-scores everything on disk for $0.
- **Look at stored bytes, not the summary.** Every defect above was invisible in the
  aggregate and obvious in the raw record. `validate.py` exists for this.
- **k=1 over more questions beat k=2 over fewer**, for the same money: ~1,400 distinct
  questions covered versus ~1,160, and question diversity is what the plan doc's
  narrowness warning is about.
- **Do not raise workers past ~8.** It does not help (TPM-bound) and it raises the
  give-up rate, which quietly degrades trajectories.
- **The teacher prompt is fragile.** Two of six defects were edits to it that broke
  something else. Read `teacher.py`'s `_THOUGHT_BUDGET` comments before touching it; each
  sentence is there for a measured reason.
