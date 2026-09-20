"""Citation extraction + passage-claim verification — the MECHANICS behind a
Proof-of-Use-style groundedness signal (arXiv 2510.10931, "Mitigating Tool-Call
Hacking in Deep Research Agents").

WHY this exists: the failure mode of a search agent is *tool-call hacking* — it
calls `search`, then answers from parametric memory and ignores what it retrieved.
Outcome reward (EM) can't see this (a lucky right answer scores full marks); a soft
judge is fooled by citation-SHAPED text. Proof-of-Use makes credit contingent on
GENUINE evidence use by: (1) requiring the answer to CITE the passages that support
it, (2) VERIFYING each citation actually entails the claim (passage-claim
alignment), and (3) rewarding COVERAGE across distinct sources (multi-hop answers
shouldn't lean on one passage). This module measures exactly those three things.

COACH boundary (CLAUDE.md): this file is pure MEASUREMENT — extract citations,
score passage-claim alignment, and return an honest CitationReport. It does NOT
decide the reward. How the report folds into a scalar groundedness term — the
alignment threshold as a *policy* knob, how much to weight verified-fraction vs
coverage, how hard to penalize a FABRICATED citation (a cite to a passage never
retrieved — the sharpest hacking tell) — is Harpreet's call in reward.py (TODO #1).

Citation format the agent emits (see env.py opening prompt / tools.answer):
    answer[<answer text> [Cited Passage Title] [Another Title]]
i.e. inline `[Title]` markers naming passages the agent READ. Kept simple and
title-based because the agent already retrieves passages by title — so a citation is
cheap to require and cheap to check against what it actually read.

Alignment backends (citation_backend in config):
  * "overlap" — dependency-free token-overlap of the claim against the cited passage
    text. UNGAMEABLE (can't fake tokens that aren't there) but shallow (paraphrase /
    negation blind). The default; good enough to make fabrication + copy-nothing
    answers cost. This is scaffold, not learning logic.
  * "nli" / "embedding" — stronger entailment via a model; left as a HOOK (stub) —
    wire a small NLI or embedding model on the pod if you want paraphrase-robust
    verification (closer to the paper). Same interface, so nothing downstream changes.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from metrics import normalize_answer
from trajectory import Trajectory


# `[Some Title]` markers. Titles can contain spaces/parens (e.g. "Blue Harvest (film)")
# but not a closing bracket; we also skip bare numeric markers like "[1]" so search's
# "[1] Title:" result formatting isn't mistaken for a citation.
_CITE_RE = re.compile(r"\[([^\]]+?)\]")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Citation:
    """One `[Title]` the agent attached to a claim in its answer."""
    title: str            # the cited passage title, as written
    claim: str            # the answer text this citation is meant to support


@dataclass
class CitationCheck:
    """The verification verdict for one citation — the honest per-cite measurement."""
    citation: Citation
    resolved: bool        # does the cited title match a passage the agent READ?
    fabricated: bool      # cited a title the agent NEVER retrieved (hallucinated cite)
    alignment: float      # [0,1] passage-claim alignment (entailment proxy)
    verified: bool        # resolved AND alignment >= threshold


@dataclass
class CitationReport:
    """Everything a Proof-of-Use-style reward needs — measured, not judged. reward.py
    turns these numbers into a groundedness scalar (TODO); this just reports them."""
    checks: list[CitationCheck] = field(default_factory=list)
    n_citations: int = 0
    n_resolved: int = 0            # citations pointing at a passage the agent read
    n_verified: int = 0            # citations whose passage actually supports the claim
    n_fabricated: int = 0          # citations to titles never retrieved — the hacking tell
    distinct_verified_sources: int = 0   # coverage: distinct verified passage titles
    n_uncited_claims: int = 0      # answer sentences with content but no citation at all
    # --- citation-F1 vs the gold supporting set (precision + recall, not just coverage) ---
    #   TP = cited & gold & read   FP = cited but not TP   FN = gold not cited-and-read
    cite_tp: int = 0
    cite_fp: int = 0
    cite_fn: int = 0
    n_gold: int = 0                # |gold supporting titles| for the question (hop count proxy)
    # --- 2026-08-26: the SAME measurement, SPLIT into its two independent halves. ---
    # ADDITIVE ONLY: cite_tp/fp/fn and precision/recall/f1 above are untouched, so the
    # reward is byte-for-byte unchanged. These are diagnostics.
    #
    # WHY. `cite_f1` above is conjunctive — a title counts only if it is cited AND gold
    # AND read. That makes one number answer two different questions, and a zero cannot
    # tell you which half failed. It cost this project real time: `groundedness ~= 0`
    # across all four reward designs in TRAINING_HISTORY_LOG.md was read as "the model
    # cannot cite", when the truth (found 2026-08-26) was "the model never calls read",
    # so no citation was scoreable BY CONSTRUCTION. Four probes tuned the incentive on an
    # action that never happened. Measured live: the GPT teacher cited BOTH gold titles
    # from search snippets and scored cite_f1 = 0.000 — indistinguishable from citing two
    # wrong things.
    #
    #   title_*             : did it pick the RIGHT SOURCES?  (read-agnostic)
    #   read_before_cite_*  : did it VERIFY them first?       (discipline)
    #
    # Keep cite_f1 as the reward if desired; log all three so the next failure is legible.
    title_tp: int = 0              # cited & gold        (read NOT required)
    title_fp: int = 0              # cited but not gold
    title_fn: int = 0              # gold never cited
    n_cited_distinct: int = 0      # distinct titles cited
    n_cited_and_read: int = 0      # of those, how many were actually read

    @property
    def title_precision(self) -> float:
        """Of the distinct titles cited, the fraction that are gold evidence — ignoring
        whether they were read. Answers "did it pick the right sources?"."""
        d = self.title_tp + self.title_fp
        return self.title_tp / d if d else 0.0

    @property
    def title_recall(self) -> float:
        d = self.title_tp + self.title_fn
        return self.title_tp / d if d else 0.0

    @property
    def title_f1(self) -> float:
        p, r = self.title_precision, self.title_recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def read_before_cite_rate(self) -> float:
        """Of the distinct titles cited, the fraction actually read first. Answers "did
        it verify?". 0.0 when nothing was cited (no discipline to measure)."""
        return (self.n_cited_and_read / self.n_cited_distinct
                if self.n_cited_distinct else 0.0)

    @property
    def verified_frac(self) -> float:
        """Of the citations made, what fraction actually check out. 0.0 if no cites."""
        return self.n_verified / self.n_citations if self.n_citations else 0.0

    @property
    def precision(self) -> float:
        """Of the (distinct, read) titles cited, fraction that are gold evidence. Punishes
        citing distractors / fabrications. 0.0 if nothing cited."""
        d = self.cite_tp + self.cite_fp
        return self.cite_tp / d if d else 0.0

    @property
    def recall(self) -> float:
        """Of the gold evidence, fraction the agent cited (and read). Punishes incomplete
        hops. 1.0 if the question has no gold titles recorded."""
        d = self.cite_tp + self.cite_fn
        return self.cite_tp / d if d else 1.0

    @property
    def f1(self) -> float:
        """Citation-F1 = harmonic mean of precision & recall — the single groundedness
        number for the Branch-B (gold) reward. Folds coverage (recall) AND over-citing
        junk/fabrication (precision) into one [0,1] score."""
        p, r = self.precision, self.recall
        return (2 * p * r / (p + r)) if (p + r) else 0.0

    @property
    def any_citation(self) -> bool:
        return self.n_citations > 0


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #
def extract_citations(answer_text: str) -> list[Citation]:
    """Pull `[Title]` citations out of the answer, each paired with the CLAIM it
    supports (the sentence it sits in, markers stripped). Bare numeric markers ("[1]")
    are ignored. Mechanics only."""
    text = (answer_text or "").strip()
    if not text:
        return []
    cites: list[Citation] = []
    for sent in _SENT_SPLIT_RE.split(text) or [text]:
        titles = [m.group(1).strip() for m in _CITE_RE.finditer(sent)]
        titles = [t for t in titles if t and not t.isdigit()]
        if not titles:
            continue
        claim = _CITE_RE.sub("", sent).strip()          # the sentence minus its markers
        for t in titles:
            cites.append(Citation(title=t, claim=claim))
    return cites


def strip_citations(answer_text: str) -> str:
    """The answer with `[Title]` markers removed — what EM/F1 should score (so a
    correct answer isn't penalized by its own citation markers)."""
    return _CITE_RE.sub("", answer_text or "").strip()


def read_evidence_map(traj: Trajectory) -> dict[str, str]:
    """title -> passage text the agent actually READ (full text it saw). Built from
    `read` steps, whose observation is '[title]\\n<text>' (tools._tool_read). This is
    the evidence a citation is verified against — you can only ground a claim in a
    passage you opened, not one you merely glimpsed in a search snippet."""
    ev: dict[str, str] = {}
    for s in traj.steps:
        if s.call is None or s.call.name != "read" or not s.ok:
            continue
        title = s.retrieved_titles[0] if s.retrieved_titles else None
        body = s.observation
        if body.startswith("[") and "]\n" in body:      # strip the "[title]\n" prefix
            body = body.split("]\n", 1)[1]
        if title:
            ev[_norm_title(title)] = body
    return ev


# --------------------------------------------------------------------------- #
# Passage-claim alignment (the entailment proxy)
# --------------------------------------------------------------------------- #
def align(claim: str, passage: str, backend: str = "overlap") -> float:
    """Score in [0,1]: how well `passage` supports `claim`. Dispatches on backend."""
    if backend == "overlap":
        return _align_overlap(claim, passage)
    if backend in ("nli", "embedding"):
        raise NotImplementedError(
            f"citation_backend='{backend}' not wired — add a small "
            f"{'NLI entailment' if backend == 'nli' else 'embedding-similarity'} model "
            f"(pod-side) returning a [0,1] support score. Interface matches _align_overlap "
            f"so nothing downstream changes. See claude-api if you route to a hosted model.")
    raise ValueError(f"unknown citation_backend: {backend}")


def _align_overlap(claim: str, passage: str) -> float:
    """Fraction of the claim's content tokens that appear in the passage. Ungameable
    (can't overlap tokens that aren't there) but paraphrase/negation-blind — hence the
    NLI/embedding hooks above for a stronger check. An empty claim aligns to 0."""
    c = normalize_answer(claim).split()
    if not c:
        return 0.0
    p = set(normalize_answer(passage).split())
    return sum(1 for t in c if t in p) / len(c)


# --------------------------------------------------------------------------- #
# The report — measure all three Proof-of-Use quantities for one trajectory
# --------------------------------------------------------------------------- #
def verify_citations(traj: Trajectory, backend: str = "overlap",
                     align_threshold: float = 0.6,
                     gold_titles: list[str] | None = None) -> CitationReport:
    """Extract the answer's citations, verify each against the passages the agent
    read, and count uncited claims. Returns a CitationReport of pure measurements.

    Two ways to decide a citation is "verified", set by `backend`:
      * "gold"  — the cited title IS one of the question's gold supporting passages
        (set membership vs gold_titles, defaulting to traj.supporting_titles). Clean +
        exact; the STRONG signal for Branch B, where gold labels exist. Consistent with
        RLVR (the reward may use ground truth — the outcome term already uses the gold
        answer). Still requires the agent actually READ the passage (cite-what-you-read).
      * "overlap"/"nli"/"embedding" — passage-claim alignment (see align()). The signal
        that GENERALIZES to no-gold settings (Branch A / a custom SEC-10K corpus /
        deployment), where gold_titles don't exist. `align_threshold` is where "the
        passage supports the claim" is drawn (a MEASUREMENT knob; default 0.6).

    NOTE: this reads traj.final_answer; if that still contains `[Title]` markers, EM/F1
    elsewhere should score strip_citations(final_answer), not the raw text."""
    answer = traj.final_answer or ""
    retrieved = {_norm_title(t) for t in traj.retrieved_title_set}
    evidence = read_evidence_map(traj)
    gold = {_norm_title(t) for t in (gold_titles if gold_titles is not None
                                     else traj.supporting_titles)}

    cites = extract_citations(answer)
    checks: list[CitationCheck] = []
    verified_titles: set[str] = set()
    for c in cites:
        nt = _norm_title(c.title)
        read_here = nt in evidence
        resolved = nt in retrieved            # surfaced at all (search or read)
        fabricated = not resolved             # cited something never retrieved
        if backend == "gold":
            # membership: 1.0 if the cited title is a gold supporting passage, else 0.0
            alignment = 1.0 if nt in gold else 0.0
            verified = read_here and nt in gold
        else:
            alignment = align(c.claim, evidence[nt], backend) if read_here else 0.0
            verified = read_here and alignment >= align_threshold
        if verified:
            verified_titles.add(nt)
        checks.append(CitationCheck(citation=c, resolved=resolved, fabricated=fabricated,
                                    alignment=alignment, verified=verified))

    n_uncited = _count_uncited_claims(answer)

    # --- citation-F1 vs gold (distinct titles; cite-what-you-read) ---
    # TP = distinct titles that are gold AND were read AND cited; FP = every other
    # distinct cited title (non-gold distractor, fabricated, or gold-but-unread);
    # FN = gold titles the agent never cited-and-read.
    read_titles = set(evidence.keys())
    distinct_cited = {_norm_title(c.title) for c in cites}
    tp_titles = distinct_cited & gold & read_titles
    fp = len(distinct_cited - tp_titles)
    fn = len(gold - tp_titles)

    # The split halves (see CitationReport's fields for why). Read-agnostic on purpose:
    # `title_*` deliberately does NOT intersect read_titles — that is the whole point.
    title_tp_set = distinct_cited & gold

    return CitationReport(
        checks=checks,
        n_citations=len(cites),
        n_resolved=sum(1 for ch in checks if ch.resolved),
        n_verified=sum(1 for ch in checks if ch.verified),
        n_fabricated=sum(1 for ch in checks if ch.fabricated),
        distinct_verified_sources=len(verified_titles),
        n_uncited_claims=n_uncited,
        cite_tp=len(tp_titles),
        cite_fp=fp,
        cite_fn=fn,
        n_gold=len(gold),
        title_tp=len(title_tp_set),
        title_fp=len(distinct_cited - title_tp_set),
        title_fn=len(gold - title_tp_set),
        n_cited_distinct=len(distinct_cited),
        n_cited_and_read=len(distinct_cited & read_titles),
    )


def _count_uncited_claims(answer_text: str) -> int:
    """Sentences with real content but no `[Title]` citation — the 'asserted without
    evidence' count a reward may want to penalize."""
    text = (answer_text or "").strip()
    if not text:
        return 0
    n = 0
    for sent in _SENT_SPLIT_RE.split(text) or [text]:
        has_cite = any(not m.group(1).strip().isdigit() for m in _CITE_RE.finditer(sent))
        content = strip_citations(sent)
        if content and not has_cite:
            n += 1
    return n


def _norm_title(t: str) -> str:
    return " ".join((t or "").strip().lower().split())
