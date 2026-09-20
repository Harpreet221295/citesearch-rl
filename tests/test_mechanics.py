"""Mechanics tests for the deep_research_agent foundation — no model, no GPU, no
network. Proves the scaffold (corpus/tools/data/trajectory/metrics/env parser) is
solid BEFORE any of the learning logic (reward TODOs) or the rLLM/veRL wiring.

Run: pytest -q tests/    (from assignments/deep_research_agent/)
"""
import sys
from pathlib import Path

# make the lab modules importable when pytest runs from anywhere
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import corpus
import data
import metrics
import tools
from env import _parse_react_action
from trajectory import Trajectory, Step, ToolCall


# --------------------------------------------------------------------------- #
# corpus / retriever
# --------------------------------------------------------------------------- #
def test_bm25_ranks_relevant_doc_first():
    store = corpus.DocStore.from_passages([
        ("Blue Harvest (film)", "A 2009 drama directed by Jane Doe about a family farm."),
        ("Blue Harvest (album)", "A 2015 studio album by the band Fjord."),
        ("Unrelated", "Cirrus clouds form at high altitude."),
    ])
    hits = store.search("who directed the film Blue Harvest", k=2)
    assert hits, "expected at least one hit"
    assert hits[0][0].title == "Blue Harvest (film)"


def test_get_is_case_insensitive_and_missing_returns_none():
    store = corpus.DocStore.from_passages([("Zurich", "Largest city in Switzerland.")])
    assert store.get("zurich").title == "Zurich"
    assert store.get("nonexistent") is None


# --------------------------------------------------------------------------- #
# tools / executor (never raises; returns retrieved titles)
# --------------------------------------------------------------------------- #
def test_search_read_answer_roundtrip():
    store = corpus.DocStore.from_passages([
        ("Alan Prime", "A physicist who worked at the Federal Polytechnic."),
        ("Federal Polytechnic", "A university headquartered in Zurich, Switzerland."),
    ])
    obs, ok, err, titles = tools.execute(store, "search", {"query": "where did Alan Prime work", "k": 2})
    assert ok and err is None and "Alan Prime" in titles

    obs, ok, err, titles = tools.execute(store, "read", {"title": "Federal Polytechnic"})
    assert ok and "Zurich" in obs and titles == ["Federal Polytechnic"]

    obs, ok, err, titles = tools.execute(store, "answer", {"text": "Zurich"})
    assert ok and obs == "Zurich"


def test_executor_reports_errors_without_raising():
    store = corpus.DocStore.from_passages([("A", "text")])
    _, ok, err, _ = tools.execute(store, "read", {"title": "missing"})
    assert not ok and err == "tool_error"
    _, ok, err, _ = tools.execute(store, "bogus", {"x": 1})
    assert not ok and err == "unknown_tool"
    _, ok, err, _ = tools.execute(store, "search", {})           # missing required arg
    assert not ok and err == "missing_arg"


# --------------------------------------------------------------------------- #
# ReAct action parser (env mechanics)
# --------------------------------------------------------------------------- #
def test_parse_react_action_variants():
    thought, name, args, ok = _parse_react_action(
        "Thought: I should search.\nAction: search[who directed Blue Harvest]")
    assert ok and name == "search" and args == {"query": "who directed Blue Harvest"}
    assert "search" in thought.lower()

    _, name, args, ok = _parse_react_action("Action: read[Blue Harvest (film)]")
    assert ok and name == "read" and args == {"title": "Blue Harvest (film)"}

    _, name, args, ok = _parse_react_action("Action: answer[American]")
    assert ok and name == "answer" and args == {"text": "American"}

    _, name, _, ok = _parse_react_action("I have no idea what to do.")
    assert not ok and name == ""


def test_parse_react_action_ignores_hallucinated_continuation():
    """2026-08-25 regression — found running rft_diagnosis/diagnosis1.py against the
    real Qwen2.5-3B model: with nothing stopping generation at the turn boundary, the
    model sometimes free-runs past its first Action and writes a full hallucinated
    continuation (fake tool results, a fake second turn, even a fake final answer) in
    ONE completion. The old regex (greedy `.*` + `re.DOTALL`) didn't just fail on this
    — it silently ACCEPTED it, stuffing the entire hallucinated rest of the trajectory
    into the FIRST action's argument string. The fix (`re.MULTILINE`, no `DOTALL`)
    must extract ONLY the real first action and ignore everything hallucinated after
    it, on a real example of exactly what the model generated."""
    hallucinated = (
        "Thought: I need to find out who Ms. G.O.A.T. is, and the city the rapper "
        "is from.\n"
        "Action: search[G.O.A.T. rapper city]\n"
        "search results:\n"
        "[1] The Game: The Game is the debut mixtape; born in Queens, raised in "
        "Brooklyn.\n"
        "Thought: The Game is the rapper whose debut mixtape is Ms. G.O.A.T.\n"
        "Action: read[The Game]\n"
        "[The Game]\n"
        "The Game is an American rapper. His debut mixtape is G.O.A.T., released in "
        "1998.\n"
        "Thought: The Game was born and raised in Brooklyn, New York.\n"
        "Action: answer[New York Brooklyn]"
    )
    _, name, args, ok = _parse_react_action(hallucinated)
    assert ok
    assert name == "search"                          # the FIRST real action, not the last
    assert args == {"query": "G.O.A.T. rapper city"}  # not the entire hallucinated blob
    assert "answer" not in args["query"]
    assert "The Game" not in args["query"]


def test_parse_react_action_multiline_citation_answer_still_works():
    """The MULTILINE fix must not regress the legitimate case it has to keep handling:
    multiple `[Title]` citations INLINE on the same Action line."""
    _, name, args, ok = _parse_react_action(
        "Action: answer[American [Blue Harvest (film)] [Jane Doe (director)]]")
    assert ok and name == "answer"
    assert args == {"text": "American [Blue Harvest (film)] [Jane Doe (director)]"}


# --------------------------------------------------------------------------- #
# metrics (EM / F1 / groundedness floor)
# --------------------------------------------------------------------------- #
def test_em_and_f1_normalization():
    assert metrics.exact_match("The USA.", ["usa"])
    assert not metrics.exact_match("Canada", ["usa"])
    assert metrics.token_f1("New York City", ["New York"]) > 0.5
    assert metrics.token_f1("completely wrong", ["New York"]) == 0.0


def test_answer_recall_in_context():
    ctx = "Nimbus Electronics is a company founded in 2004 in Taipei."
    assert metrics.answer_recall_in_context("2004", ctx) == 1.0
    assert metrics.answer_recall_in_context("1999", ctx) == 0.0


# --------------------------------------------------------------------------- #
# data fixture + trajectory metrics
# --------------------------------------------------------------------------- #
def test_fixture_retrieval_surfaces_gold_evidence():
    for task in data.fixture_examples():
        store = task.docstore()
        found = {d.title for d, _ in store.search(task.question, k=3)}
        # BM25 over the question should surface at least one gold supporting passage
        assert set(task.supporting_titles) & found, (
            f"{task.task_id}: no gold title in top-3 (got {found})")


def test_trajectory_retrieval_hit_rate_and_views():
    traj = Trajectory(
        task_id="t", query="q", tools=["search", "read", "answer"],
        gold_answer="Zurich", supporting_titles=["Alan Prime", "Federal Polytechnic"],
        steps=[
            Step(thought="", call=ToolCall("search", {"query": "q"}), observation="...",
                 ok=True, retrieved_titles=["Alan Prime", "Distractor"]),
            Step(thought="", call=ToolCall("read", {"title": "Federal Polytechnic"}),
                 observation="...", ok=True, retrieved_titles=["Federal Polytechnic"]),
            Step(thought="", call=ToolCall("answer", {"text": "Zurich"}), observation="Zurich",
                 ok=True),
        ],
        final_answer="Zurich", done=True,
    )
    assert traj.n_turns == 3
    assert traj.n_searches == 1
    assert traj.retrieval_hit_rate() == 1.0        # both gold titles retrieved


def test_trajectory_json_roundtrip():
    traj = Trajectory(task_id="t", query="q", tools=["search"], gold_answer="a",
                      steps=[Step(thought="th", call=ToolCall("search", {"query": "x"}),
                                  observation="o", ok=True, retrieved_titles=["T"])],
                      final_answer="a", done=True)
    back = Trajectory.from_dict(traj.to_dict())
    assert back.task_id == "t" and back.steps[0].call.name == "search"
    assert back.steps[0].retrieved_titles == ["T"]
