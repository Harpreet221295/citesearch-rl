"""The split citation metric: does it separate 'right sources' from 'verified them'?

Added 2026-08-26. `cite_f1` is conjunctive (cited AND gold AND read), so a zero cannot
say which half failed — and that ambiguity cost this project four reward-design probes
chasing "the model cannot cite" when the truth was "the model never reads".

The load-bearing case is `test_cited_gold_but_never_read`: the exact situation observed
live with the GPT teacher, where both correct sources were cited from search snippets
and cite_f1 scored 0.000, indistinguishable from citing two wrong things.
"""
from __future__ import annotations

import sys
from pathlib import Path

_LAB = Path(__file__).resolve().parent.parent
if str(_LAB) not in sys.path:
    sys.path.insert(0, str(_LAB))

from citations import verify_citations                          # noqa: E402
from trajectory import Trajectory, Step, ToolCall                # noqa: E402

GOLD = ["Blue Harvest (film)", "Jane Doe (director)"]


def _traj(answer: str, read_titles: list[str], searched: list[str] | None = None):
    """A trajectory that searched `searched` (defaults to GOLD) and read `read_titles`."""
    steps = [Step(thought="", call=ToolCall("search", {"query": "q"}),
                  observation="search results:", ok=True, parse_ok=True,
                  retrieved_titles=list(searched if searched is not None else GOLD))]
    for t in read_titles:
        steps.append(Step(thought="", call=ToolCall("read", {"title": t}),
                          observation=f"[{t}]\nfull passage text about {t}.",
                          ok=True, parse_ok=True, retrieved_titles=[t]))
    steps.append(Step(thought="", call=ToolCall("answer", {"text": answer}),
                      observation=answer, ok=True, parse_ok=True))
    return Trajectory(task_id="t", query="q", tools=["search", "read", "answer"],
                      gold_answer="American", gold_aliases=[], supporting_titles=GOLD,
                      steps=steps, final_answer=answer, done=True)


def _rep(traj):
    return verify_citations(traj, backend="gold", gold_titles=GOLD)


def test_cited_gold_but_never_read():
    """THE case. Both correct sources cited, neither read.

    Old number says 0.0 — looks identical to citing garbage. The split shows the truth:
    source selection is PERFECT, verification discipline is zero.
    """
    r = _rep(_traj("American [Blue Harvest (film)] [Jane Doe (director)]", read_titles=[]))
    assert r.f1 == 0.0                      # conjunctive metric, unchanged
    assert r.title_f1 == 1.0                # picked exactly the right sources
    assert r.read_before_cite_rate == 0.0   # verified none of them
    assert r.n_cited_distinct == 2 and r.n_cited_and_read == 0


def test_cited_and_read_everything():
    r = _rep(_traj("American [Blue Harvest (film)] [Jane Doe (director)]",
                   read_titles=GOLD))
    assert r.f1 == 1.0
    assert r.title_f1 == 1.0
    assert r.read_before_cite_rate == 1.0


def test_read_half_of_what_it_cited():
    """The partial case seen in the teacher smoke run: cited two, read one."""
    r = _rep(_traj("American [Blue Harvest (film)] [Jane Doe (director)]",
                   read_titles=["Blue Harvest (film)"]))
    assert r.f1 == 0.5
    assert r.title_f1 == 1.0                # source selection still perfect
    assert r.read_before_cite_rate == 0.5


def test_wrong_sources_are_not_rescued_by_the_split():
    """The split must not turn a genuinely bad citation into a good-looking one."""
    r = _rep(_traj("American [Some Distractor] [Another Distractor]",
                   read_titles=["Some Distractor", "Another Distractor"],
                   searched=["Some Distractor", "Another Distractor"]))
    assert r.f1 == 0.0
    assert r.title_f1 == 0.0                # cited the wrong things: still zero
    assert r.read_before_cite_rate == 1.0   # it DID read them — diligent, and wrong


def test_partial_source_selection():
    """One gold cited, one missed: precision 1.0, recall 0.5 -> title_f1 = 2/3."""
    r = _rep(_traj("American [Blue Harvest (film)]", read_titles=["Blue Harvest (film)"]))
    assert r.title_precision == 1.0
    assert r.title_recall == 0.5
    assert abs(r.title_f1 - 2 / 3) < 1e-9


def test_no_citations_at_all():
    r = _rep(_traj("American", read_titles=GOLD))
    assert r.title_f1 == 0.0
    assert r.read_before_cite_rate == 0.0   # nothing cited -> no discipline to measure
    assert r.n_cited_distinct == 0


def test_reward_path_is_unchanged():
    """The split is ADDITIVE. cite_tp/fp/fn and f1 must be exactly what they always were,
    or every historical number in TRAINING_HISTORY_LOG.md silently stops comparing."""
    r = _rep(_traj("American [Blue Harvest (film)] [Jane Doe (director)]",
                   read_titles=["Blue Harvest (film)"]))
    assert (r.cite_tp, r.cite_fp, r.cite_fn) == (1, 1, 1)
    assert r.precision == 0.5 and r.recall == 0.5 and r.f1 == 0.5
