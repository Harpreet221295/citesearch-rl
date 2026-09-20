"""Tests for the core reward + env seams (full-build): reward_deep_research, the
DeepResearchEnv episode → trajectory → terminal-reward path, and the masking check.
No model / GPU / network. Run: pytest -q tests/"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
import reward as dr_reward
import env as dr_env
from data import DRTask
from trajectory import Trajectory, Step, ToolCall


def _cfg(**kw):
    c = config.Config.sanity_preset()
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def _task():
    return DRTask(
        task_id="fx-1",
        question="nationality of the director of Blue Harvest?",
        gold_answer="American", gold_aliases=["american"],
        supporting_titles=["Blue Harvest (film)", "Jane Doe (director)"],
        passages=[
            ("Blue Harvest (film)", "Blue Harvest is a 2009 drama directed by Jane Doe."),
            ("Jane Doe (director)", "Jane Doe is an American film director born in Ohio."),
            ("Blue Harvest (album)", "A 2015 album by the band Fjord."),
            ("John Roe (producer)", "A British film producer."),
        ],
    )


# --------------------------------------------------------------------------- #
# reward: gated form rewards grounded-correct, penalizes ungrounded/wrong
# --------------------------------------------------------------------------- #
def _traj(answer, reads, supporting, done=True):
    steps = []
    for title, text in reads:
        steps.append(Step(thought="", call=ToolCall("read", {"title": title}),
                          observation=f"[{title}]\n{text}", ok=True, retrieved_titles=[title]))
    steps.append(Step(thought="", call=ToolCall("answer", {"text": answer}),
                      observation=answer, ok=True))
    return Trajectory(task_id="t", query="q", tools=["search", "read", "answer"],
                      gold_answer="American", gold_aliases=["american"],
                      supporting_titles=supporting, steps=steps,
                      final_answer=answer, done=done)


def test_gated_reward_orders_trajectories_correctly():
    cfg = _cfg(reward_mode="gated", beta=0.5, citation_backend="gold",
               lambda_format=0.1, lambda_eff=0.0)
    task = _task()
    gold = ["Blue Harvest (film)", "Jane Doe (director)"]
    reads = [("Blue Harvest (film)", "directed by Jane Doe"),
             ("Jane Doe (director)", "an American film director")]

    # right + fully grounded (both gold cited)
    r_good = dr_reward.reward_deep_research(
        task, _traj("American [Blue Harvest (film)] [Jane Doe (director)]", reads, gold),
        None, cfg).reward
    # right but ungrounded (no citations)
    r_ungrounded = dr_reward.reward_deep_research(
        task, _traj("American", reads, gold), None, cfg).reward
    # wrong answer
    r_wrong = dr_reward.reward_deep_research(
        task, _traj("British [Jane Doe (director)]", reads, gold), None, cfg).reward

    assert r_good > r_ungrounded > r_wrong
    assert abs(r_good - 1.0) < 1e-9            # o=1, g=1 → 1.0, no tolls
    assert abs(r_ungrounded - 0.5) < 1e-9      # o=1, g=0 → beta=0.5


def test_fabricated_citation_penalized():
    cfg = _cfg(reward_mode="gated", beta=0.5, w_fab=0.25, citation_backend="gold",
               lambda_format=0.0, lambda_eff=0.0)
    task = _task()
    gold = ["Blue Harvest (film)", "Jane Doe (director)"]
    reads = [("Jane Doe (director)", "an American film director")]
    # cite a title never retrieved → fabricated → extra toll + precision hit
    r = dr_reward.reward_deep_research(
        task, _traj("American [Jane Doe (director)] [Ghost Source]", reads, gold),
        None, cfg)
    assert r.cite_fabricated == 1
    assert r.reward < 1.0                       # penalized vs a clean grounded answer


def test_ramped_efficiency_toll_off_early_on_later():
    cfg = _cfg(reward_mode="gated", beta=0.5, citation_backend="gold",
               lambda_format=0.0, lambda_eff=0.0, lambda_eff_max=0.05,
               lambda_eff_ramp_start=50, lambda_eff_ramp_end=100)
    task = _task()
    gold = ["Blue Harvest (film)", "Jane Doe (director)"]
    reads = [("Blue Harvest (film)", "directed by Jane Doe"),
             ("Jane Doe (director)", "an American film director")]
    traj = _traj("American [Blue Harvest (film)] [Jane Doe (director)]", reads, gold)
    r_early = dr_reward.reward_deep_research(task, traj, None, cfg, step=0).reward
    r_late = dr_reward.reward_deep_research(task, traj, None, cfg, step=100).reward
    assert r_early > r_late                     # toll ramps in → later reward lower
    assert abs(r_early - 1.0) < 1e-9            # toll off at step 0


# --------------------------------------------------------------------------- #
# env: full episode → trajectory → terminal reward
# --------------------------------------------------------------------------- #
def test_env_episode_end_to_end():
    cfg = _cfg(citation_backend="gold", reward_mode="gated", beta=0.5,
               lambda_format=0.0, lambda_eff=0.0)
    task = _task()
    e = dr_env.DeepResearchEnv.from_dict({"task": task, "cfg": cfg})
    obs, _ = e.reset()
    assert "research agent" in obs.lower()

    # drive a correct, grounded 2-hop episode by feeding ReAct text directly
    e.step("Thought: find the director.\nAction: search[director of Blue Harvest]")
    e.step("Thought: read it.\nAction: read[Blue Harvest (film)]")
    e.step("Thought: nationality?\nAction: read[Jane Doe (director)]")
    obs, r, done, info = e.step(
        "Thought: done.\nAction: answer[American [Blue Harvest (film)] [Jane Doe (director)]]")

    assert done is True
    assert info["correct"] is True
    assert info["cite_f1"] == 1.0
    assert abs(r - 1.0) < 1e-9


def test_env_forced_stop_without_answer_is_scored():
    cfg = _cfg(citation_backend="gold", max_turns=2, lambda_format=0.1)
    task = _task()
    e = dr_env.DeepResearchEnv.from_dict({"task": task, "cfg": cfg})
    e.reset()
    e.step("Thought: hmm.\nAction: search[Blue Harvest]")
    obs, r, done, info = e.step("Thought: still thinking.\nAction: search[Jane Doe]")
    assert done is True and info["correct"] is False     # never answered → wrong


# --------------------------------------------------------------------------- #
# masking check: a fake tokenizer, correct vs leaking mask
# --------------------------------------------------------------------------- #
class _FakeTok:
    """Whitespace tokenizer: id = index into a vocab; decode joins with spaces."""
    def __init__(self):
        self.vocab, self.inv = {}, {}
    def encode(self, text):
        ids = []
        for w in text.split():
            if w not in self.vocab:
                i = len(self.vocab); self.vocab[w] = i; self.inv[i] = w
            ids.append(self.vocab[w])
        return ids
    def decode(self, ids):
        return " ".join(self.inv[int(i)] for i in ids)


def _build_seq(tok, spans):
    """spans = [(text, mask_value)] → (input_ids, loss_mask)."""
    ids, mask = [], []
    for text, mv in spans:
        e = tok.encode(text)
        ids += e; mask += [mv] * len(e)
    return ids, mask


def test_mask_check_passes_on_correct_mask():
    tok = _FakeTok()
    traj = _traj("American [Jane Doe (director)]",
                 reads=[("Jane Doe (director)", "Jane Doe is an American film director born in Ohio")],
                 supporting=["Jane Doe (director)"])
    # correct: prompt & retrieved passage masked 0, model answer masked 1
    ids, mask = _build_seq(tok, [
        ("You are a research agent question here", 0),          # prompt (ignored)
        ("Jane Doe is an American film director born in Ohio", 0),  # retrieved (ignored)
        ("Thought done Action answer American Jane Doe director", 1),  # model (graded)
    ])
    rep = dr_env.assert_verl_masking_matches(tok, ids, mask, traj, min_probe_chars=10, verbose=False)
    assert rep["leaked"] == 0 and rep["checked_passages"] == 1


def test_mask_check_catches_leaked_passage():
    tok = _FakeTok()
    traj = _traj("American [Jane Doe (director)]",
                 reads=[("Jane Doe (director)", "Jane Doe is an American film director born in Ohio")],
                 supporting=["Jane Doe (director)"])
    # BUG: retrieved passage marked 1 (graded) — must raise
    ids, mask = _build_seq(tok, [
        ("You are a research agent question here", 0),
        ("Jane Doe is an American film director born in Ohio", 1),  # LEAK
        ("Thought done Action answer American", 1),
    ])
    import pytest
    with pytest.raises(AssertionError):
        dr_env.assert_verl_masking_matches(tok, ids, mask, traj, min_probe_chars=10, verbose=False)
