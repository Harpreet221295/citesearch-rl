"""Tests for the framework-agnostic eval harness (evaluate.py): the env+policy rollout,
the metric row, and the anti-hacking probes / gate. No model / GPU / rLLM."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
import evaluate as ev
from data import DRTask


def _cfg(**kw):
    c = config.Config.sanity_preset()
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
    """A stateful policy_fn that returns queued actions in order."""
    it = iter(actions)
    return lambda history: next(it, "Thought: give up.\nAction: answer[]")


def test_oracle_episode_scores_perfect():
    cfg = _cfg(citation_backend="gold")
    policy = _scripted([
        "Thought: find director.\nAction: search[director of Blue Harvest]",
        "Thought: read.\nAction: read[Blue Harvest (film)]",
        "Thought: nationality?\nAction: read[Jane Doe (director)]",
        "Thought: done.\nAction: answer[American [Blue Harvest (film)] [Jane Doe (director)]]",
    ])
    traj, ri = ev.rollout_episode(_task(), cfg, policy)
    assert ri.correct is True
    assert ri.cite_f1 == 1.0
    assert ri.hit_rate == 1.0


def test_memory_guesser_scores_poorly():
    cfg = _cfg(citation_backend="gold")
    # answers immediately from "memory", no tools, no citations
    policy = _scripted(["Thought: I just know.\nAction: answer[American]"])
    traj, ri = ev.rollout_episode(_task(), cfg, policy)
    assert ri.correct is True          # got the answer right...
    assert ri.hit_rate == 0.0          # ...but retrieved nothing
    assert ri.cite_f1 == 0.0           # ...and cited nothing → ungrounded


def test_evaluate_set_aggregates():
    cfg = _cfg(citation_backend="gold")
    policy = _scripted(["Thought: guess.\nAction: answer[American]"])
    row = ev.evaluate_set([_task()], cfg, policy)
    assert row["em"] == 1.0 and row["hit_rate"] == 0.0 and row["n"] == 1


# --------------------------------------------------------------------------- #
# probes + gate
# --------------------------------------------------------------------------- #
def test_probe_memory_guessing_fires():
    base = dict(em=0.3, hit_rate=0.6, cite_f1=0.3, avg_steps=4, fabricated_rate=0.0,
                answer_len=2, f1=0.4, judge=0.0)
    tuned = dict(em=0.6, hit_rate=0.60, cite_f1=0.3, avg_steps=4, fabricated_rate=0.0,
                 answer_len=2, f1=0.6, judge=0.0)   # EM up, hit-rate flat
    fired = ev.probes(base, tuned, judge_used=False)
    assert any("MEMORY-GUESSING" in f for f in fired)


def test_probe_judge_gaming_fires():
    base = dict(em=0.3, hit_rate=0.6, cite_f1=0.3, avg_steps=4, fabricated_rate=0.0,
                answer_len=2, f1=0.4, judge=0.4)
    tuned = dict(em=0.5, hit_rate=0.7, cite_f1=0.3, avg_steps=4, fabricated_rate=0.0,
                 answer_len=2, f1=0.6, judge=0.9)   # judge 0.9 >> cite_f1 0.3
    fired = ev.probes(base, tuned, judge_used=True)
    assert any("JUDGE-GAMING" in f for f in fired)


def test_gate_pass_and_fail():
    base = dict(em=0.30, hit_rate=0.55, cite_f1=0.20, avg_steps=4.0, fabricated_rate=0.0,
                answer_len=2, f1=0.4, judge=0.0)
    good = dict(em=0.52, hit_rate=0.81, cite_f1=0.68, avg_steps=4.3, fabricated_rate=0.0,
                answer_len=2, f1=0.6, judge=0.0)
    passed, reasons = ev.gate(base, good, judge_used=False, margin=0.05)
    assert passed and not reasons

    weak = dict(em=0.32, hit_rate=0.55, cite_f1=0.20, avg_steps=4.0, fabricated_rate=0.0,
                answer_len=2, f1=0.4, judge=0.0)   # EM barely moved
    passed2, reasons2 = ev.gate(base, weak, judge_used=False, margin=0.05)
    assert not passed2 and reasons2
