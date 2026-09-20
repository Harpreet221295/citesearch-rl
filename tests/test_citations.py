"""Tests for the Proof-of-Use citation-verification scaffold (citations.py) + that
the ReAct parser preserves inline `[Title]` citations inside answer[...]. Mechanics
only — the reward POLICY that consumes these numbers is Harpreet's (reward.py TODO)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import citations
from env import _parse_react_action
from trajectory import Trajectory, Step, ToolCall


def _traj_with_read(answer: str, read_title: str, read_text: str) -> Trajectory:
    """A trajectory that read one passage then answered — the minimal shape
    verify_citations needs (evidence map comes from `read` steps)."""
    return Trajectory(
        task_id="t", query="q", tools=["search", "read", "answer"], gold_answer="x",
        steps=[
            Step(thought="", call=ToolCall("search", {"query": "q"}), observation="search results:\n[1] ...",
                 ok=True, retrieved_titles=[read_title]),
            Step(thought="", call=ToolCall("read", {"title": read_title}),
                 observation=f"[{read_title}]\n{read_text}", ok=True, retrieved_titles=[read_title]),
            Step(thought="", call=ToolCall("answer", {"text": answer}), observation=answer, ok=True),
        ],
        final_answer=answer, done=True,
    )


# --------------------------------------------------------------------------- #
# parser preserves nested citation brackets
# --------------------------------------------------------------------------- #
def test_answer_action_preserves_inline_citations():
    _, name, args, ok = _parse_react_action("Action: answer[American [Jane Doe (director)]]")
    assert ok and name == "answer"
    assert args["text"] == "American [Jane Doe (director)]"


# --------------------------------------------------------------------------- #
# extraction
# --------------------------------------------------------------------------- #
def test_extract_and_strip_citations():
    ans = "The director is American [Jane Doe (director)]."
    cites = citations.extract_citations(ans)
    assert len(cites) == 1
    assert cites[0].title == "Jane Doe (director)"
    assert "[" not in cites[0].claim and "american" in cites[0].claim.lower()
    assert citations.strip_citations(ans) == "The director is American ."  # markers gone


def test_numeric_markers_are_not_citations():
    # search-result formatting like "[1] Title" must not be read as a citation
    assert citations.extract_citations("answer is 42 [1] [2]") == []


# --------------------------------------------------------------------------- #
# verification: verified / fabricated / uncited
# --------------------------------------------------------------------------- #
def test_verified_citation_when_passage_supports_claim():
    traj = _traj_with_read(
        answer="American [Jane Doe (director)]",
        read_title="Jane Doe (director)",
        read_text="Jane Doe is an American film director born in Ohio.")
    rep = citations.verify_citations(traj, backend="overlap", align_threshold=0.5)
    assert rep.n_citations == 1
    assert rep.n_verified == 1
    assert rep.n_fabricated == 0
    assert rep.distinct_verified_sources == 1
    assert rep.verified_frac == 1.0


def test_fabricated_citation_is_flagged():
    # cite a title the agent never retrieved/read → fabricated, not verified
    traj = _traj_with_read(
        answer="American [Nonexistent Source]",
        read_title="Jane Doe (director)",
        read_text="Jane Doe is an American film director.")
    rep = citations.verify_citations(traj)
    assert rep.n_fabricated == 1
    assert rep.n_verified == 0
    assert rep.verified_frac == 0.0


def test_cited_but_unsupported_claim_fails_alignment():
    # cited a real read passage, but the claim's tokens aren't in it → not verified
    traj = _traj_with_read(
        answer="The capital is Paris [Jane Doe (director)]",
        read_title="Jane Doe (director)",
        read_text="Jane Doe is an American film director born in Ohio.")
    rep = citations.verify_citations(traj, align_threshold=0.6)
    assert rep.n_citations == 1
    assert rep.n_resolved == 1        # the passage was read...
    assert rep.n_verified == 0        # ...but it doesn't support "capital is Paris"


def test_uncited_claim_counted():
    traj = _traj_with_read(
        answer="American.",            # no citation at all
        read_title="Jane Doe (director)",
        read_text="Jane Doe is an American film director.")
    rep = citations.verify_citations(traj)
    assert rep.n_citations == 0
    assert rep.n_uncited_claims == 1
    assert not rep.any_citation


def test_gold_backend_verifies_by_membership():
    # cited a gold supporting title that was read → verified under "gold" backend,
    # regardless of word overlap with the claim
    traj = _traj_with_read(
        answer="American [Jane Doe (director)]",
        read_title="Jane Doe (director)",
        read_text="She was born in Ohio.")          # note: no "american" token here
    traj.supporting_titles = ["Jane Doe (director)", "Blue Harvest (film)"]
    rep_overlap = citations.verify_citations(traj, backend="overlap", align_threshold=0.5)
    rep_gold = citations.verify_citations(traj, backend="gold")
    assert rep_overlap.n_verified == 0     # overlap fails: "american" not in passage
    assert rep_gold.n_verified == 1        # gold passes: it's a gold supporting title


def test_gold_backend_rejects_non_gold_title():
    traj = _traj_with_read(
        answer="American [Distractor Passage]",
        read_title="Distractor Passage",
        read_text="Some unrelated text.")
    traj.supporting_titles = ["Jane Doe (director)"]
    rep = citations.verify_citations(traj, backend="gold")
    assert rep.n_verified == 0             # read it, but it's not a gold passage


def _traj_two_reads(answer, reads, supporting):
    """Trajectory that read several passages (reads = [(title, text), ...]) then answered."""
    steps = []
    for title, text in reads:
        steps.append(Step(thought="", call=ToolCall("read", {"title": title}),
                          observation=f"[{title}]\n{text}", ok=True, retrieved_titles=[title]))
    steps.append(Step(thought="", call=ToolCall("answer", {"text": answer}),
                      observation=answer, ok=True))
    return Trajectory(task_id="t", query="q", tools=["search", "read", "answer"],
                      gold_answer="x", supporting_titles=supporting,
                      steps=steps, final_answer=answer, done=True)


def test_citation_f1_perfect_multi_hop():
    # cited BOTH gold passages, nothing else → P=1, R=1, F1=1
    traj = _traj_two_reads(
        "American [Blue Harvest (film)] [Jane Doe (director)]",
        reads=[("Blue Harvest (film)", "directed by Jane Doe"),
               ("Jane Doe (director)", "an American film director")],
        supporting=["Blue Harvest (film)", "Jane Doe (director)"])
    rep = citations.verify_citations(traj, backend="gold")
    assert (rep.cite_tp, rep.cite_fp, rep.cite_fn) == (2, 0, 0)
    assert rep.precision == 1.0 and rep.recall == 1.0 and rep.f1 == 1.0


def test_citation_f1_incomplete_hops_hurts_recall():
    # read both gold, but cited only one → R=0.5, P=1 → F1≈0.667
    traj = _traj_two_reads(
        "American [Jane Doe (director)]",
        reads=[("Blue Harvest (film)", "directed by Jane Doe"),
               ("Jane Doe (director)", "an American film director")],
        supporting=["Blue Harvest (film)", "Jane Doe (director)"])
    rep = citations.verify_citations(traj, backend="gold")
    assert (rep.cite_tp, rep.cite_fp, rep.cite_fn) == (1, 0, 1)
    assert rep.precision == 1.0 and rep.recall == 0.5
    assert abs(rep.f1 - 2/3) < 1e-9


def test_citation_f1_distractor_hurts_precision():
    # cited 1 gold + 1 distractor (both read) → P=0.5, R=0.5 → F1=0.5
    traj = _traj_two_reads(
        "American [Jane Doe (director)] [Some Distractor]",
        reads=[("Jane Doe (director)", "an American film director"),
               ("Some Distractor", "unrelated text")],
        supporting=["Jane Doe (director)", "Blue Harvest (film)"])
    rep = citations.verify_citations(traj, backend="gold")
    assert (rep.cite_tp, rep.cite_fp, rep.cite_fn) == (1, 1, 1)
    assert rep.precision == 0.5 and rep.recall == 0.5 and rep.f1 == 0.5


def test_overlap_alignment_bounds():
    assert citations.align("american director", "an american film director", "overlap") == 1.0
    assert citations.align("capital of france", "an american film director", "overlap") == 0.0
