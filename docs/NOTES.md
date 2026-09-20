# deep_research_agent — design notes & backlog

Running log of design decisions and ideas we've deliberately *deferred* — so a
"maybe later" doesn't get forgotten. Newest on top.

## Decisions (settled)

### Reward-component breakdown + dead-groups roll-up now logged to W&B — 2026-08-24
Real gap found while diagnosing a flat/declining reward trend on the actual cloud run
(not hypothetical): `reward.py`'s per-episode breakdown (`outcome`, `groundedness`,
`cite_f1`/precision/recall, the toll components — see `env.py::_terminal_reward`'s
`info_dict`) was computed every episode but never aggregated into anything visible in
W&B, only the final scalar `reward`. Made "is outcome improving while groundedness lags,
or vice versa" answerable from the logs instead of guessable — genuinely needed given
the gated reward design below multiplicatively couples the two axes, so a flat scalar
reward doesn't tell you WHICH axis is stuck.
Also added `batch/dead_groups_pct` (= `batch/solve_none` + `batch/solve_all`, both
already logged by rLLM itself) — the DEAD GROUPS concern flagged in the outcome-reward
decision below (all-wrong or all-correct group → zero GRPO advantage) now has a single
number instead of manually adding two metrics by hand every check-in.
Implementation: `rllm_workflow.py`'s `_patch_tracking_log_for_extra_metrics` — wraps
`rllm.utils.tracking.Tracking.log` (the exact call site right before verl forwards
metrics to wandb/console) so the injected keys land on the identical step axis as every
other metric, no separate `wandb.log()` call. See that function's docstring for the full
mechanism (accumulator pattern, val-mode-call skip logic).

### Compute: 1× A100 80GB; NO judge model in training — 2026-08-14
Qwen2.5-3B (or even 7B) + LoRA fits comfortably on ONE A100 80GB (est. ~30-40GB used:
6GB base weights shared, <1GB LoRA+optim, 6-10GB acts/grads chunked, 15-25GB vLLM rollout).
LoRA gives the KL REFERENCE model for free (adapters-off = ref, adapters-on = policy — same
weights, no 2nd copy). Multi-GPU is a SPEED choice (parallel rollouts / full-FT), not a fit
requirement. **Consequence of citation-F1-on-gold: the entire TRAINING reward (EM +
gold-citation-F1 + format + efficiency) is rule-based → NO judge/LLM needed at train time.**
The LLM judge is only used at EVAL as a cross-check/anti-hacking probe. Saves ~6GB + many
generate() calls vs finqa (which ran an HF judge every step). Stack: rLLM (rollouts) + veRL
(GRPO update), colocated mode on the single A100.

### Groundedness term = citation-F1 (not coverage-only) — Harpreet, 2026-08-14
Groundedness `g(τ)` = citation-**F1** vs the gold supporting set, over DISTINCT cited
titles (cite-what-you-read): `TP` = cited & gold & read, `FP` = every other cited title
(distractor / fabricated / gold-but-unread), `FN` = gold not cited-and-read. `P=TP/(TP+FP)`,
`R=TP/(TP+FN)`, `g=F1`. This folds coverage (recall) AND over-citing junk + fabrication
(precision) into one [0,1] number — strictly better than my earlier coverage-only
`d_v/|S_q|` (recall only). Fabrication is now subsumed as an FP (can keep a small EXTRA
penalty for never-*retrieved* cites if desired, since that's worse than citing a real
distractor). Implemented as `CitationReport.precision/recall/f1` + `cite_tp/fp/fn`.

### Outcome reward = EM (log F1 as a metric) — Harpreet, 2026-08-14
`reward_kind="em"` (0/1, ungameable, matches Search-R1). Safe here because the prompt forces
1-3 word answers so pred/gold share a space. Log F1 as a metric. Lever if too many GRPO
DEAD GROUPS (all-wrong group → all EM=0 → no advantage): switch to F1 for a denser signal,
or lean on the groundedness term (varies even when EM=0) to break ties. `reward_kind` knob stays.

### System prompt must teach multi-citation — Harpreet, 2026-08-14
Prompt + `answer` tool now show a TWO-citation example and say multi-hop needs MULTIPLE
citations, one per fact/hop. Reason: with a single-citation example the agent might never
learn it can cite >1 (and citation-F1 recall would be capped). Fixed in `env._opening_prompt`.

### Branch-B training citation reward = gold-membership ONLY — Harpreet, 2026-08-14
For the HotpotQA/2Wiki training reward, verify citations by gold-title membership
(`citation_backend="gold"`) and NOTHING else. Do **not** add an answer-in-passage floor
to the citation term. Reason (Harpreet): in a multi-hop question only ONE gold passage
actually contains the final answer span — the other gold passage(s) support intermediate
hops. An answer-in-passage check per citation would wrongly reject the perfectly valid
hop-1 citation (e.g. cite [Blue Harvest (film)] for the "who directed" hop, which does NOT
contain "American"). So answer-in-passage is not just unnecessary, it's WRONG at the
per-citation level for multi-hop. Keep `metrics.answer_recall_in_context` only as a logged
diagnostic (trajectory-level, any passage), never as a reward term. Alignment backends
(overlap/nli) stay for the no-gold branches (A / SEC-10K), not for Branch-B training.

## Eval gate + anti-hacking probes (spec for evaluate.py — build this)

**Gate:** held-out set (frozen, `eval_seed=9999`, zero overlap with train). Score BASE
(untrained Instruct, prompted ReAct) vs TUNED (after GRPO). PASS only if TUNED beats BASE by
a clear margin on multi-hop QA **AND** groundedness holds (no probe below fires). Never report
one number.

**Localized metric row (base vs tuned):**
| metric | source | reads as |
|---|---|---|
| answer EM | `metrics.exact_match(strip_citations(ans), golds)` | got it right? |
| answer F1 | `metrics.token_f1` | partial-credit correctness |
| retrieval hit-rate | `traj.retrieval_hit_rate()` (gold titles) | did it FIND the evidence? |
| citation-F1 | `citations.verify_citations(...).f1` | did it CITE the right evidence? |
| avg # steps | `traj.n_turns` | actually multi-hop researching? |
| fabricated-cite rate | `report.n_fabricated` / cites | hallucinating citations? |

**Anti-hacking probes (each maps to a named hack; fire = gate fails):**
1. **Judge-vs-grounded gap** — run the hackable LLM judge AND ungameable citation-F1/NLI on
   eval; large `judge − citationF1` gap ⇒ judge-gaming. (Use a DIFFERENT groundedness method
   at eval than the one trained on — Branch B trains on gold-membership, so eval with NLI/overlap
   + judge for an independent read.)
2. **Fabrication / length inflation** — fabricated-cite rate ↑, or answer length ↑ while EM flat
   ⇒ padding/verbosity to farm F1 or fool the judge.
3. **Research-decoupling** — EM ↑ but retrieval-hit-rate flat ⇒ answering from memory (tools are
   theater); EM ↑ but avg-steps → 1 ⇒ tool-avoidance / short-circuiting (the finqa regression).
4. **No-fewshot / honest-RL read** — if the eval prompt carries demos, also eval WITHOUT them to
   separate "what RL taught" from "what the prompt scaffolding gave" (finqa precedent).

Philosophy: don't ask "did reward go up?" — ask "did it go up for the RIGHT reasons?", one
probe per wrong reason.

## Backlog (ideas to try if/when we need them)

### Citation verification via gold-title membership (not just fuzzy alignment) — Harpreet, 2026-08-14
**Idea:** on Branch B we HAVE `gold_supporting_titles` per question. So a citation can be
verified by clean set membership — `cited_title ∈ gold_supporting_titles` — instead of (or
alongside) the heuristic word/entailment `align()`. Cleaner, exact, no paraphrase noise.
**Harpreet is right that for the Branch-B TRAINING reward this is a stronger, simpler signal**
than word-overlap alignment. Using gold in the reward is consistent with RLVR (outcome reward
already uses the gold answer).
**Where alignment still earns its place:** (1) transfer — gold titles DON'T exist at real
deployment / Branch A (live web) / a custom SEC-10K corpus, so alignment is the only
groundedness signal that generalizes; (2) claim-level faithfulness for long/report answers
(gold-title membership is passage-level only); (3) catching "cited a gold passage but the
claim it supports is actually wrong" — though outcome≈0 usually already handles that.
**Recommendation:** add `citation_backend="gold"` (verify by gold-title membership) as the
DEFAULT for Branch B, keep `"overlap"` / `"nli"` as the no-gold fallback for Branch A / SEC-10K.
Small change in `citations.verify_citations` (pass gold titles; when backend=="gold", verified
= resolved-and-in-gold). **Status: NOT built — offered during tutor; implement when Harpreet
starts the reward.** Note: citation-reward-via-gold then correlates with retrieval-hit-rate;
the extra thing citation adds over hit-rate = the agent explicitly COMMITTED to the passage as
evidence, not just incidentally retrieved it.


### Query-aware in-document reranking before truncation  — proposed by Harpreet, 2026-08-14
**Idea:** when a passage is longer than the observation cap, don't cut blindly from
the top. First **rerank the passage's sentences/paragraphs by relevance to the current
query**, put the most-relevant chunks first, *then* truncate. Truncation stops
discarding signal, because what's left is the part that actually matters for the query.

**Why it's good:** cheap (BM25/sentence-overlap rerank, no model needed), and it turns
truncation from "lose the tail" into "keep the relevant bit." A middle ground between
the current generous-cap and full chunked reading.

**Status:** NOT built. Trigger to build it = we see an eval-perf drop that traces to the
agent reading a passage but missing the fact (i.e. the fact was past the cut). Until
then, HotpotQA paragraphs are short enough that the cap rarely bites — don't pre-optimize.

**Where it'd live:** `tools._tool_read` (rerank sentences by query before rendering) —
which means `read` would need access to the current query (it doesn't today; the env
has it). Small plumbing change. Keep the plain `read` as the default; make rerank a
config flag (`read_rerank: bool`).

**Related:** the "instrument truncation" note below — do that first (cheap), this second.

### Instrument truncation (make it loud, not silent)  — 2026-08-14
When `read` truncates, append a visible marker (`[truncated — N more chars; call read
again to continue]`) and log how often it fires. Silent truncation reads as "covered
everything" when it didn't. ~5-line change in `tools._tool_read`. Do this before the
rerank idea above.
