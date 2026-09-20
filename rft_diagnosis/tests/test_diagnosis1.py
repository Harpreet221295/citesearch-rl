"""Offline tests for diagnosis1.py's pure-Python pieces — no GPU, no model. Same
scripted-policy pattern tests/test_evaluate.py already uses (evaluate.rollout_episode
drives DeepResearchEnv with a queued-actions policy_fn) so classify_trajectory and
aggregate_breakdown are trusted BEFORE spending real GPU time sampling a real model."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))     # rft_diagnosis/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # deep_research_agent/

import config
import env as env_mod
import evaluate as ev
from data import DRTask

import diagnosis1 as d1


def _cfg(**kw):
    c = config.Config.sanity_preset()
    c.citation_backend = "gold"
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def _task():
    return DRTask(
        task_id="fx-1", question="nationality of the director of Blue Harvest?",
        gold_answer="American", gold_aliases=["american"],
        supporting_titles=["Blue Harvest (film)", "Jane Doe (director)"],
        passages=[
            ("Blue Harvest (film)", "Blue Harvest is a 2009 drama directed by Jane Doe."),
            ("Jane Doe (director)", "Jane Doe is an American film director born in Ohio."),
            ("Blue Harvest (album)", "A 2015 album by Fjord."),
        ])


def _scripted(actions):
    it = iter(actions)
    return lambda history: next(it, "Thought: give up.\nAction: answer[]")


# --------------------------------------------------------------------------- #
# rich_opening_prompt
# --------------------------------------------------------------------------- #
def test_rich_prompt_has_all_three_worked_examples():
    prompt = d1.rich_opening_prompt(_task(), _cfg())
    assert "Worked example 1" in prompt
    assert "Worked example 2" in prompt
    assert "Worked example 3" in prompt
    # the correcting-read and search-miss-recovery lessons are actually present
    assert "Halcyon Drive" in prompt and "Ridgeline Cycles" in prompt
    assert "Coral Bell Testament" in prompt
    assert "Read" in prompt or "read" in prompt   # tool menu rendered
    assert _task().question in prompt


def test_rich_prompt_falls_back_without_citations():
    cfg = _cfg(require_citations=False)
    task = _task()
    assert d1.rich_opening_prompt(task, cfg) == env_mod._opening_prompt(task, cfg)


def test_patched_rich_prompt_swaps_and_restores():
    original = env_mod._opening_prompt
    with d1.patched_rich_prompt():
        assert env_mod._opening_prompt is d1.rich_opening_prompt
    assert env_mod._opening_prompt is original     # restored even on the happy path


def test_patched_rich_prompt_restores_on_exception():
    original = env_mod._opening_prompt
    try:
        with d1.patched_rich_prompt():
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert env_mod._opening_prompt is original     # restored even if the body raised


# --------------------------------------------------------------------------- #
# classify_trajectory — reuses evaluate.rollout_episode's scripted-policy pattern
# --------------------------------------------------------------------------- #
def test_classify_correct_and_cited():
    cfg = _cfg()
    policy = _scripted([
        "Thought: find director.\nAction: search[director of Blue Harvest]",
        "Thought: read.\nAction: read[Blue Harvest (film)]",
        "Thought: nationality?\nAction: read[Jane Doe (director)]",
        "Thought: done.\nAction: answer[American [Blue Harvest (film)] [Jane Doe (director)]]",
    ])
    traj, _ = ev.rollout_episode(_task(), cfg, policy)
    row = d1.classify_trajectory(_task(), traj, cfg)
    assert row["bucket"] == "correct_and_cited"
    assert row["calls_read"] is True
    assert row["n_reads"] == 2
    assert row["terminated_cleanly"] is True
    assert row["cite_f1"] == 1.0


def test_classify_correct_uncited():
    cfg = _cfg()
    policy = _scripted(["Thought: I just know.\nAction: answer[American]"])
    traj, _ = ev.rollout_episode(_task(), cfg, policy)
    row = d1.classify_trajectory(_task(), traj, cfg)
    assert row["bucket"] == "correct_uncited"
    assert row["correct"] is True
    assert row["n_citations"] == 0
    assert row["calls_read"] is False


def test_classify_wrong_answer():
    cfg = _cfg()
    policy = _scripted(["Thought: guess.\nAction: answer[French]"])
    traj, _ = ev.rollout_episode(_task(), cfg, policy)
    row = d1.classify_trajectory(_task(), traj, cfg)
    assert row["bucket"] == "wrong_answer"
    assert row["correct"] is False


def test_classify_correct_miscited_partial_recall():
    cfg = _cfg()
    # cites only ONE of the two gold passages -> imperfect recall, cite_f1 < 1
    policy = _scripted([
        "Thought: read.\nAction: read[Blue Harvest (film)]",
        "Thought: done.\nAction: answer[American [Blue Harvest (film)]]",
    ])
    traj, _ = ev.rollout_episode(_task(), cfg, policy)
    row = d1.classify_trajectory(_task(), traj, cfg)
    assert row["correct"] is True
    assert row["bucket"] == "correct_miscited"
    assert row["cite_f1"] < 1.0


def test_classify_never_terminates_hits_max_turns():
    cfg = _cfg(max_turns=2)
    policy = _scripted([
        "Thought: search.\nAction: search[something irrelevant]",
        "Thought: search again.\nAction: search[something else irrelevant]",
    ])
    traj, _ = ev.rollout_episode(_task(), cfg, policy)
    row = d1.classify_trajectory(_task(), traj, cfg)
    assert row["terminated_cleanly"] is False
    assert row["bucket"] == "wrong_answer"        # no answer emitted -> empty pred -> not EM


def test_classify_parse_failure_recorded():
    cfg = _cfg()
    policy = _scripted([
        "this has no Thought/Action structure at all",
        "Thought: recover.\nAction: answer[American]",
    ])
    traj, _ = ev.rollout_episode(_task(), cfg, policy)
    row = d1.classify_trajectory(_task(), traj, cfg)
    assert row["n_parse_failures"] >= 1
    assert row["parse_ok"] is False


# --------------------------------------------------------------------------- #
# aggregate_breakdown
# --------------------------------------------------------------------------- #
def test_aggregate_breakdown_rates_and_buckets():
    rows = [
        {"parse_ok": True, "calls_read": True, "terminated_cleanly": True, "correct": True,
         "n_reads": 2, "n_searches": 2, "n_tool_calls": 5, "cite_f1": 1.0, "bucket": "correct_and_cited"},
        {"parse_ok": True, "calls_read": False, "terminated_cleanly": True, "correct": True,
         "n_reads": 0, "n_searches": 0, "n_tool_calls": 1, "cite_f1": 0.0, "bucket": "correct_uncited"},
        {"parse_ok": False, "calls_read": False, "terminated_cleanly": False, "correct": False,
         "n_reads": 0, "n_searches": 1, "n_tool_calls": 2, "cite_f1": 0.0, "bucket": "wrong_answer"},
    ]
    agg = d1.aggregate_breakdown(rows)
    assert agg["n"] == 3
    assert abs(agg["parse_ok_rate"] - 2 / 3) < 1e-9
    assert abs(agg["calls_read_rate"] - 1 / 3) < 1e-9
    assert abs(agg["correct_rate"] - 2 / 3) < 1e-9
    assert agg["bucket_pct"]["correct_and_cited"] == 1 / 3
    assert agg["hop_count_distribution"] == {1: 1, 2: 1, 5: 1}


def test_aggregate_breakdown_empty():
    assert d1.aggregate_breakdown([]) == {"n": 0}


# --------------------------------------------------------------------------- #
# load_pool — offset-stability check. `load_pool` applies its own Python-side
# slice AFTER data.load_tasks, so distinct offsets give distinct, non-overlapping
# slices even in fixture mode (real-HF-data offset-stability needs network, not
# tested offline, but the slicing mechanism itself is exercised here).
# --------------------------------------------------------------------------- #
def test_load_pool_offsets_give_disjoint_slices():
    cfg = _cfg()   # sanity_preset -> use_fixture=True (4-task fixture)
    a = d1.load_pool(cfg, "train", n=2, offset=0)
    b = d1.load_pool(cfg, "train", n=2, offset=2)
    ids_a = [t.task_id for t in a]
    ids_b = [t.task_id for t in b]
    assert len(ids_a) == 2 and len(ids_b) == 2
    assert set(ids_a).isdisjoint(ids_b)          # the whole point of `offset` — no overlap
    assert ids_a == ["fx-1", "fx-2"] and ids_b == ["fx-3", "fx-4"]
