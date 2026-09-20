"""Reward — the layered signal that turns one trajectory into a scalar the GRPO
update maximizes. THIS IS THE CORE LEARNING LOGIC OF THE CAPSTONE — the capstone
brief calls the trajectory + reward design "the genuinely hard, interesting part".

STATUS: IMPLEMENTED (full-build 2026-08-14). Originally a COACH-mode TODO, but Harpreet
opted into full-build ("it goes straight to RunPod"). The design below was worked out
WITH him in a tutor session and every decision is recorded in NOTES.md — this is that
design in code, not a black box. The one structural choice (gated vs additive) and all
weights are config knobs, so he can still retune the policy without editing this file.

The design you're implementing (knobs in config.py, policy here):

    r(traj) = w_outcome * outcome(traj)          # did it answer correctly?  (EM or F1)
            + w_ground  * groundedness(traj)      # is the answer supported by retrieval?
            - lambda_format * format_error(traj)  # well-formed ReAct / valid tool calls?
            - lambda_eff(step) * n_steps(traj)    # efficiency toll, RAMPED (config.lambda_eff_at)

Three things make this a *deep-research* reward, not a generic QA reward — and each
is a decision you have to make:

  1. OUTCOME vs GROUNDEDNESS are different axes. A right answer the agent never
     retrieved evidence for (retrieval_hit_rate == 0, EM == 1) is a *lucky guess*,
     not research. Decide how (whether) to couple them — e.g. gate/att­enuate the
     outcome reward by whether the agent actually retrieved the gold evidence, or
     keep them additive and let the eval expose the gap. This choice IS the capstone.

  2. GROUNDEDNESS is partly ungameable (metrics.answer_recall_in_context — token
     overlap with retrieved text) and partly hackable (the judge). Decide the mix.
     Leaning on the judge alone invites judge-gaming (verbose, citation-shaped,
     unsupported answers) — the exact failure evaluate.py probes for.
     >> CURRENT SOTA to aim at (web-checked 2026-08): "Proof-of-Use: Mitigating
     Tool-Call Hacking in Deep Research Agents" (arXiv 2510.10931). Its reward makes
     credit CONTINGENT on genuine evidence use: require the answer to CITE the
     passages that support it, then VERIFY each citation actually entails the claim
     (passage-claim alignment via overlap/embedding/NLI), and reward coverage across
     DISTINCT sources (multi-hop → shouldn't lean on one passage). This is the
     capstone's "citation faithfulness" standout hook — building a citation-verified
     groundedness term here (not just judge + overlap) is the differentiator. The
     agent already retrieves by title, so a citation is cheap to require + check
     against traj.retrieved_title_set / the passage text.

  3. The EFFICIENCY toll must RAMP (config.lambda_eff_at(step)), or the agent
     collapses to "answer immediately, never search" to dodge the per-step penalty
     — the tool-avoidance regression you already hit in finqa_agent. Don't apply a
     flat toll from step 0.

You have the ingredients ready to call:
  * metrics.exact_match / token_f1(pred, task.answers)   — outcome
    >> score citations.strip_citations(traj.final_answer), NOT the raw text, or the
       answer's own `[Title]` markers tank EM/F1.
  * citations.verify_citations(traj, backend, threshold) — Proof-of-Use CitationReport:
       .verified_frac              cited passages that actually support the claim
       .distinct_verified_sources  coverage across sources (multi-hop shouldn't lean on one)
       .n_fabricated               cited a passage never retrieved — the sharpest hacking tell
       .n_uncited_claims           asserted-without-evidence count
    This is the STRONG groundedness signal (aim here — the standout hook). It's
    measurement; YOU decide the policy: how to weight verified_frac vs coverage, and
    how hard to punish a fabricated citation (a cite to text the agent never saw).
  * metrics.answer_recall_in_context(pred, retrieved)    — ungameable groundedness floor (weak)
  * judge.score(task, traj)                              — soft groundedness (hackable)
  * traj.retrieval_hit_rate()                            — did it find the gold evidence?
  * traj.n_turns / traj.n_searches                       — efficiency
  * config.lambda_eff_at(step)                           — the ramped toll

Return a RewardInfo so train/eval can log the *breakdown*, not just the scalar —
you need the components to tell a real gain from reward hacking.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import metrics
from trajectory import Trajectory


@dataclass
class RewardInfo:
    """The reward AND its breakdown — logged per trajectory so the eval can separate
    real gains from hacking (a rising total with rising judge but flat outcome = a
    red flag)."""
    reward: float
    outcome: float = 0.0            # EM (0/1) or F1 of the final answer
    correct: bool = False           # EM hit (for solve-rate)
    groundedness: float = 0.0       # combined groundedness in [0,1]
    judge: float = 0.0              # raw judge score in [0,1]
    overlap: float = 0.0            # metrics.answer_recall_in_context (ungameable floor)
    hit_rate: float = 0.0           # traj.retrieval_hit_rate()
    # --- Proof-of-Use citation grounding (from citations.CitationReport) ---
    cite_f1: float = 0.0            # citation-F1 vs gold (the groundedness term, Branch B)
    cite_precision: float = 0.0     # cited-right / cited (punishes distractors+fabrication)
    cite_recall: float = 0.0        # cited-right / gold (punishes incomplete hops)
    cite_fabricated: int = 0        # cites to never-retrieved passages (extra hacking tell)
    cite_uncited_claims: int = 0    # asserted-without-evidence count
    format_error: float = 0.0
    n_steps: int = 0
    components: dict[str, Any] = field(default_factory=dict)


def _retrieved_text(traj: Trajectory) -> str:
    """The evidence text the agent surfaced — for the ungameable groundedness floor."""
    return "\n".join(s.observation for s in traj.steps
                     if s.call is not None and s.call.name in ("search", "read") and s.ok)


def format_error(traj: Trajectory) -> float:
    """A cheap, mechanical well-formedness signal in [0,1] (0 = clean). Provided so
    your reward can dock malformed ReAct without you re-deriving it: fraction of
    turns whose action failed to parse into a legal tool call, plus a penalty if the
    episode never emitted an `answer`. Pure mechanics — tweak the policy in the TODO."""
    if not traj.steps:
        return 1.0
    bad = sum(1 for s in traj.steps if not s.parse_ok)
    frac_bad = bad / len(traj.steps)
    no_answer = 0.0 if traj.done else 0.5
    return min(1.0, frac_bad + no_answer)


# ============================================================================ #
#  THE CORE REWARD  (full-build 2026-08-14 — see module note below)             #
# ----------------------------------------------------------------------------- #
#  WHAT: the layered deep-research reward, implemented per the design worked out #
#        in the tutor session and recorded in NOTES.md.                          #
#  WHY each piece: outcome (EM) = verifiable correctness; groundedness =         #
#        citation-F1 vs the gold supporting set (precision punishes distractor/  #
#        fabricated cites, recall punishes incomplete hops); a small EXTRA toll  #
#        on fabricated cites (worse than a real distractor); a format toll; and  #
#        a RAMPED efficiency toll (flat-from-0 would train tool-avoidance).      #
#  THE ONE THING WORTH UNDERSTANDING: the default is GATED, not additive —       #
#        r = outcome*(beta + (1-beta)*g) - tolls. Grounding does not add a       #
#        separate bonus; it *unlocks* the outcome credit. A right-but-ungrounded #
#        answer keeps only `beta` (0.5) of its credit, so the model can't win by #
#        guessing from memory. Flip cfg.reward_mode="additive" to decouple them. #
# ============================================================================ #
def grounded_outcome(task, traj: Trajectory, judge, cfg) -> RewardInfo:
    """Compute the OUTCOME and GROUNDEDNESS terms and return a RewardInfo with the
    components filled (`.reward` is left 0.0 here; reward_deep_research assembles the
    scalar). Safe to call with judge=None (training path — no judge model)."""
    import citations   # local import: avoids a hard dep for pure-mechanics importers

    pred = citations.strip_citations(traj.final_answer or "")   # score answer, not [Title] markers

    # --- outcome: verifiable correctness (EM 0/1, or soft F1) ---
    em = metrics.exact_match(pred, task.answers)
    f1 = metrics.token_f1(pred, task.answers)
    outcome = float(em) if cfg.reward_kind == "em" else f1

    # --- groundedness: citation-F1 vs the gold supporting set (Branch B) ---
    # gold backend uses traj.supporting_titles; overlap/nli backends generalize to no-gold.
    report = citations.verify_citations(
        traj, backend=cfg.citation_backend,
        align_threshold=cfg.citation_align_threshold)
    g = report.f1

    # --- weak corroborants (mainly for logging / the no-gold branches) ---
    overlap = metrics.answer_recall_in_context(pred, _retrieved_text(traj))
    judge_s = float(judge.score(task, traj)) if judge is not None else 0.0
    hit = traj.retrieval_hit_rate()

    return RewardInfo(
        reward=0.0,
        outcome=outcome, correct=bool(em),
        groundedness=g, judge=judge_s, overlap=overlap, hit_rate=hit,
        cite_f1=report.f1, cite_precision=report.precision, cite_recall=report.recall,
        cite_fabricated=report.n_fabricated, cite_uncited_claims=report.n_uncited_claims,
        format_error=format_error(traj), n_steps=traj.n_turns,
        components={"em": float(em), "f1": f1, "cite_tp": report.cite_tp,
                    "cite_fp": report.cite_fp, "cite_fn": report.cite_fn,
                    "n_gold": report.n_gold,
                    # 2026-08-24: raw citation-attempt count, correct or not — added
                    # because cite_precision/recall=0 is AMBIGUOUS between "cited
                    # nothing" and "cited things, all wrong," and we were only
                    # inferring "stopped citing entirely" indirectly via
                    # cite_fabricated trending to 0 (Probe 2 finding). This makes
                    # that distinction direct instead of inferred.
                    "n_citations": report.n_citations},
    )


def reward_deep_research(task, traj: Trajectory, judge, cfg, step: int = 0) -> RewardInfo:
    """Assemble the full layered reward and return a filled RewardInfo.

    gated (default):  r = outcome*(beta + (1-beta)*g) - w_fab*fab - lam_fmt*fmt - lam_eff(step)*n
    additive:         r = w_outcome*outcome + w_ground*g - w_fab*fab - lam_fmt*fmt - lam_eff(step)*n
    cite_gated:       r = 0 if n_citations==0 else outcome*(cite_gated_floor + (1-cite_gated_floor)*g)
                      - w_fab*fab - lam_fmt*fmt - lam_eff(step)*n

    This is the seam veRL calls per finished trajectory (env._terminal_reward passes the
    global training `step` so the efficiency toll can ramp)."""
    info = grounded_outcome(task, traj, judge, cfg)
    o, g = info.outcome, info.groundedness

    beta = cfg.beta_at(step)  # ramped DOWN over training — see Config.beta_at docstring

    if cfg.reward_mode == "additive":
        base = cfg.w_outcome * o + cfg.w_ground * g
    elif cfg.reward_mode == "cite_gated":
        # 2026-08-24: Probes 1-3 all showed the same failure regardless of incentive
        # shape (beta=0, beta ramped, additive) — citation-avoidance, not bad-quality
        # citation. Root cause of "gated" mode's own exploit: beta gives the SAME
        # partial credit (outcome*beta) to a rollout that never attempted a citation
        # as to one that tried and got it wrong — so not-trying is exactly as safe as
        # trying-badly, and strictly safer once fab_toll risk is considered. This mode
        # closes that specific loophole: a genuine non-attempt (n_citations==0) gets a
        # HARD ZERO, strictly worse than even a bad attempt — while a rollout that DID
        # cite something still gets the same quality-scaled credit as "gated" mode
        # (cite_gated_floor plays beta's role, but is NOT ramped — the ramp existed to
        # avoid crushing EVERYONE's reward before the model could cite; here the crush
        # only ever hits genuine non-attempts, which is exactly what we want to punish
        # from step 1, not ease into). See TRAINING_HISTORY_LOG.md's Probe 4 entry.
        n_cites = info.components.get("n_citations", 0)
        if n_cites > 0:
            base = o * (cfg.cite_gated_floor + (1.0 - cfg.cite_gated_floor) * g)
        else:
            base = 0.0
    else:  # "gated" — grounding UNLOCKS outcome credit (the default; see module note)
        base = o * (beta + (1.0 - beta) * g)

    fab_toll = cfg.w_fab * info.cite_fabricated
    fmt_toll = cfg.lambda_format * info.format_error
    eff_toll = cfg.lambda_eff_at(step) * info.n_steps

    info.reward = base - fab_toll - fmt_toll - eff_toll
    info.components.update({
        "reward_mode": cfg.reward_mode, "base": base, "fab_toll": fab_toll,
        "fmt_toll": fmt_toll, "eff_toll": eff_toll, "lambda_eff": cfg.lambda_eff_at(step),
        "beta": beta, "step": step,
    })
    return info
