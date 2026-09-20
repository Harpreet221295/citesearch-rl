# Tentative future experiments — deep_research_agent

**Status: speculative, not scheduled.** Ideas worth trying if the current pure-GRPO
approach keeps showing the same bottleneck, not a commitment to build any of this now.
Contrast with `TRAINING_HISTORY_LOG.md` (what actually happened) and `NOTES.md` (settled
decisions) — this doc is explicitly the "maybe later" pile, per its own naming.

---

## The motivating observation (real, from Attempt 3's `reward_components/*` logging)

Step 1 of the `lr=5e-5` run (2026-08-24): `hit_rate≈0.77` (the agent finds/retrieves the
right evidence ~77% of the time — retrieval itself is not the bottleneck) but
`groundedness≈0.004` (essentially zero) and `cite_fabricated≈0.35` (fabricates citations
in ~35% of episodes). The gated reward's own `base` term confirms this is actively
suppressing credit: `outcome(0.33) × (0.5 + 0.5×0.004 groundedness) ≈ 0.16` — the model
is finding the right passages and then not citing them correctly in its final answer.

**The concern this raises**: if citation discipline is a *specific, mechanical skill*
(not a reasoning/exploration problem) that the base model simply doesn't have, pure GRPO
may be a slow/expensive way to teach it — RL's sparse, per-episode reward signal has to
discover "cite correctly" through trial and error across the SAME rollout that also has
to get the multi-hop reasoning right, retrieval strategy right, AND efficiency right.
Harpreet's instinct: teach the citation mechanic *first*, cheaply, via supervised/
preference methods, THEN let RL focus on the harder exploration-requiring parts (search
strategy, multi-hop reasoning, efficiency) on top of a policy that already knows the
citation format — rather than asking RL to discover all of it from scratch at once.

**Not yet decided this is actually needed** — Attempt 3 might still show groundedness
recovering over more steps once GRPO gets more gradient signal on it (the F1-based
citation reward isn't THAT sparse — it's continuous in [0,1], not 0/1 like EM). Revisit
this doc's urgency once we have more steps of real data, not just step 1.

**FINAL UPDATE 2026-08-24, after all four probes: this doc's urgency is now confirmed,
not speculative.** Every reward-design tried — including Probe 4's `cite_gated`, which
DID fix citation-attempt-rate cleanly — still failed to teach citation CORRECTNESS.
Probe 4 additionally surfaced a live, concrete mechanism for why: GRPO's within-group
advantage normalization makes large, easy contrasts (attempt vs. not) fast to learn but
dilutes small ones (correct vs. incorrect citation) when outcome-variance dominates a
group. That's not a tunable-away reward-design problem — it's closer to a structural
limit on what pure RL credit-assignment can efficiently teach for a rare, specific skill.
Harpreet's read, live during Probe 4: RFT is the most promising next step specifically
because this session's own data (`hit_rate` consistently 0.6-0.8, `groundedness` briefly
reaching 0.03 in Probe 4) suggests real correctly-cited examples likely already occur by
chance in sampled rollouts often enough to mine — Option B below, not A or C, is the
recommended starting point.

**UPDATE 2026-08-24, no longer just step-1 speculation**: three 25-step reward-design
probes (see `TRAINING_HISTORY_LOG.md`) all converged on the SAME failure independently —
`beta=0.0` static, `beta` ramped fast 0.5→0.0 (steps 1-30), and `reward_mode="additive"`
(no gate at all) — none moved `groundedness` off ~0. Probe 3 (additive) added a new
direct metric, `reward_components/pct_rollouts_with_citation`, which showed the
mechanism explicitly instead of inferring it: 43% of rollouts attempted a citation at
step 1, dropping to 17% by step 4 — the model is actively LEARNING to abandon citation
attempts even under a design with zero penalty for not citing (additive mode's citation
term is a pure bonus). This is real evidence the citation gap is closer to "the model
doesn't have the skill to reliably earn the reward" than "the incentive shape is wrong" —
three structurally different incentive shapes all produced the same abandonment.

---

## Option D — a direct zero-citation penalty (Harpreet's proposal, 2026-08-24)

**Idea:** none of the reward designs tried so far (gated with any `beta`, additive) ever
make *not citing at all* actively costly — they only modulate the reward FOR correct
citation (the gate/bonus) or the toll for WRONG citation (`fab_toll`, only fires on
fabricated cites). Absence itself has always been free, or even the SAFEST option once
groundedness is low (any citation attempt risks a fabrication toll; zero attempts risk
nothing). Add a dedicated toll that fires specifically when `n_citations==0`:
`zero_cite_toll = w_zero_cite * (1 if n_citations == 0 else 0)`, orthogonal to
`fab_toll` — this removes the "cite nothing" free-lunch equilibrium all three probes
converged to, regardless of what the outcome/groundedness terms are doing.

**Why this is different from what's already been tried:** `beta` and `reward_mode`
control how much credit CORRECT citation earns; this controls whether NO citation is
itself penalized. A model could theoretically satisfy every design tried so far by
citing zero times and taking the (safe, if suboptimal) outcome-only credit — this term
closes that specific door rather than adjusting the reward around it.

**Open design questions, not yet resolved:** how large should `w_zero_cite` be relative
to `fab_toll` (get the ratio wrong and the model may prefer fabricating over abstaining,
which is worse) — and whether this should also ramp (start at 0 so early training isn't
punished for a skill it hasn't found yet, similar reasoning to `lambda_eff_at`/`beta_at`)
or apply from step 1, since the citation-avoidance pattern in all three probes locked in
within the first ~10 steps regardless of ramp speed.

**STATUS UPDATE 2026-08-24: built and run as Probe 4 (`reward_mode="cite_gated"`) — a
hard zero for `n_citations==0` rather than a separate toll, same effect.** Real result
(see `TRAINING_HISTORY_LOG.md`): fixed the attempt-rate cleanly (`pct_rollouts_with_
citation` climbed to a stable 0.91-0.99, the only probe of four where this didn't
collapse toward zero). But it surfaced a SECOND, smaller exploit — the model learned to
paste roughly ONE citation per answer (`n_citations` flat at ~1.0-1.3 the whole run),
just enough to clear the gate, not enough for genuinely multi-hop questions that need
multiple citations. `groundedness` briefly moved (steps 8-13, the only time across all
four probes it left ~0) then regressed to exactly 0 for the last 8 steps. **A natural
Option D-2, not yet built:** scale the penalty by how many gold-supporting passages
exist vs. how many were actually cited (not just a binary "cited anything" check) —
would need `n_gold` (already logged as a component) compared against `n_citations`
per-episode.

## Option A — Cold-start SFT before RL

Train the base model via standard supervised next-token prediction on a set of
`(question, correctly-cited trajectory)` examples before ever starting GRPO. Where the
data comes from is the real design question:
- **Distillation from a stronger teacher** — have a bigger/better model (GPT-5-class,
  Claude, or similar) generate high-quality cited ReAct trajectories on the SAME training
  questions, SFT on those. Real cost: API spend for teacher generations, and the risk of
  the student picking up stylistic quirks that don't transfer to its own retrieval
  behavior (the teacher might "know" the answer already and construct plausible-looking
  citations post-hoc rather than genuinely reasoning from retrieved evidence).
- **Self-distillation via rejection sampling** — see Option B; cheaper, no external
  teacher needed, but bounded by what the base model can ALREADY occasionally do right.

This is conceptually the "cold-start" stage several published RL-for-reasoning recipes
use before their RL stage specifically to avoid unstable/slow early RL — worth a light
literature check before committing to a specific recipe rather than reinventing the
staging from scratch (not done here — this doc is ideas, not a verified survey).

## Option B — RFT (rejection-sampling fine-tuning)

Sample many completions (e.g. K=16-64) per training question from the CURRENT policy,
filter to keep only the ones that scored well on a citation-specific criterion (high
`cite_f1`, zero `cite_fabricated` — stricter than the full gated reward, isolating JUST
the citation skill rather than overall episode quality), then do plain SFT on that
filtered set. This is the cheapest option to build — reuses the reward function and
rollout infrastructure already built for GRPO, no new teacher model or preference-pair
construction needed. Given `hit_rate≈77%` (retrieval already mostly works) but
`groundedness≈0`, there should be a real, if modest, population of "found the right
evidence AND cited it correctly" trajectories already happening by chance in existing
rollouts — RFT would just be "find those, amplify them via SFT." Directly testable using
data we're already generating (could even mine it from ALREADY-COLLECTED rollouts/episode
logs from Attempt 3, if it's not too late by the time this is tried).

## Option C — A DPO/SimPO/KTO stage targeting citation preference specifically

Construct paired `(chosen, rejected)` trajectories that differ SPECIFICALLY in citation
quality (same question, same-ish retrieved evidence, one cites correctly and one
fabricates/omits) and train via a preference objective before RL. Two ways to build the
pairs:
- **Self-generated pairs**: from rejection-sampling, pick a high-cite_f1 completion as
  "chosen" and a low-cite_f1 completion (from the SAME prompt, different rollout) as
  "rejected." Cheap, reuses existing infra, but the pair isn't perfectly controlled (the
  two completions might differ in other ways besides citation quality — outcome
  correctness, reasoning path — muddying what the preference signal is actually teaching).
- **Synthetically corrupted pairs**: take a genuinely good trajectory, programmatically
  strip/replace its citations to build a matched "rejected" twin that's IDENTICAL except
  for citation quality — a cleaner, more controlled preference signal isolating just the
  citation dimension, at the cost of needing to build that corruption pipeline.

DPO/SimPO/KTO differ mainly in whether they need a reference model (DPO does, SimPO/KTO
don't) and pair-vs-unpaired data requirements (KTO can work with unpaired good/bad
labels, not strict pairs) — worth picking based on which data construction path (self-
generated vs. synthetic) ends up easier to build, not a strong prior either way yet.

## A staged pipeline, if this ends up warranted

**This is a bootstrap for RL, not a replacement for it.** Worth being explicit about the
division of labor, since it's easy to over-claim what SFT/RFT/preference methods can do
here: they're good at teaching a narrow, consistent, mechanical HABIT (cite the passage
you read, in the right format) because that's imitation-learnable — a stable pattern to
match. They are NOT well-suited to teaching outcome correctness or search strategy,
which require trying different reasoning/search paths and reinforcing what works — an
inherently exploration-driven problem that stays RL's job, not imitation's, no matter
how good the bootstrap stage is. RFT specifically is also bounded by what the CURRENT
policy already does by chance — it can amplify a correct-citation pattern that
occasionally occurs, not invent a genuinely novel reasoning-plus-citation combination
the policy never samples. And a narrow bootstrap stage risks overfitting to the specific
citation FORMAT it was trained on rather than genuine grounding discipline, narrowing
the response diversity the SUBSEQUENT RL stage needs for its own exploration (see
"Honest risks" below — this was already flagged, worth taking seriously given the stage
below depends on RL still doing real work afterward).

1. **Stage 0 (current)**: pure GRPO from the base model — where we are now, hit the
   citation bottleneck (or didn't — re-check before proceeding).
2. **Stage 1 (proposed)**: a citation-focused bootstrapping pass — RFT (Option B, cheapest
   to build) or a preference stage (Option C) — specifically to raise the citation-
   discipline FLOOR (not solve citation completely) before RL has to also discover it
   from a near-zero base rate.
3. **Stage 2**: resume/redo GRPO on top of the citation-primed checkpoint. This is where
   the actual outcome-correctness and search-strategy learning still has to happen — RL
   isn't optional or reduced to a formality here, it's spending its (expensive,
   sparse-signal) budget on the genuinely exploration-requiring parts instead of ALSO
   having to bootstrap citation mechanics from a near-zero base rate at the same time.

## Honest risks / things not to skip if this gets built for real

- **Extra engineering + compute cost** — a whole new training stage, new data pipeline
  (rejection sampling or pair construction), likely another round of the same
  "correctness before scale" sanity-run discipline this whole project has followed.
- **Distribution shift risk** — a cold-start/preference stage trained on a NARROW slice
  of behavior (just citation correctness) could bias the model in unwanted ways — e.g.
  overfitting to a specific citation FORMAT rather than genuine grounding discipline, or
  reducing the response diversity RL needs for its own exploration (a low-entropy,
  over-confident starting point for RL is its own known failure mode — see the
  `lr` search entries in `TRAINING_HISTORY_LOG.md` for how entropy/diversity concerns
  already came up once this session).
- **Don't skip the eval gate for whatever stage gets built** — same rule as everywhere
  else in this project (per `CLAUDE.md`): a cold-start/preference stage needs its OWN
  before/after check (did it actually raise citation quality without regressing outcome
  correctness or diversity), not just an assumption that it worked.
