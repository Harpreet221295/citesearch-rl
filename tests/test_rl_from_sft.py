"""Offline tests for the 2026-09-07 RL-from-SFT wiring: the preset, the verl override
map, the stop-sequence plumbing knob, and the behaviour counters the env now reports.
No model / GPU / network. Run: pytest -q tests/"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
import env as dr_env
import train_dr
from data import DRTask


def test_rl_from_sft_preset_matches_the_sft_contract():
    c = config.Config.rl_from_sft()
    # training and rollout prompts must match the SFT prompt (SFT_RL_PLAN.md §2)
    assert c.include_worked_example is False
    # RL pool disjoint from SFT; early-stop on sft_dev, never on heldout_eval
    assert c.train_split == "rl_train" and c.eval_split == "sft_dev"
    # merged-SFT base, not the Hub base (reference policy = SFT, see merge_sft.py)
    assert c.model_name.endswith("sft_merged")
    assert c.reward_mode == "cite_gated"
    assert c.verl_use_kl_loss is True
    assert c.rollout_stop_sequences == ("\nThought:", "\nsearch results:", "\n[")
    # JSON round-trip must not choke on the new tuple field
    assert "rollout_stop_sequences" in c.to_json()


def test_probe_is_a_short_no_push_variant():
    p = config.Config.probe_rl_from_sft()
    r = config.Config.rl_from_sft()
    assert p.steps < 30 and p.push_checkpoints is False and p.checkpoint_every in (0, 10)
    assert p.run_name != r.run_name
    for k in ("model_name", "train_split", "eval_split", "reward_mode",
              "rollout_stop_sequences", "include_worked_example"):
        assert getattr(p, k) == getattr(r, k)


def test_historical_presets_are_unchanged_by_the_new_fields():
    for preset in (config.Config.cloud_preset, config.Config.probe_cite_gated,
                   config.Config.sanity_preset):
        c = preset()
        assert c.train_split is None and c.eval_split is None
        assert c.rollout_stop_sequences == ()
        assert c.verl_use_kl_loss is False


def test_verl_overrides_carry_kl_loss_keys():
    c = config.Config.rl_from_sft()
    ov = train_dr.verl_overrides(c, Path("/tmp/t.parquet"), Path("/tmp/v.parquet"))
    assert "actor_rollout_ref.actor.use_kl_loss=True" in ov
    assert "actor_rollout_ref.actor.kl_loss_coef=0.01" in ov
    assert "actor_rollout_ref.actor.kl_loss_type=low_var_kl" in ov
    assert f"data.max_prompt_length={c.max_len}" in ov
    # and the historical preset still says False
    ov0 = train_dr.verl_overrides(config.Config.cloud_preset(), Path("/tmp/t"), Path("/tmp/v"))
    assert "actor_rollout_ref.actor.use_kl_loss=False" in ov0


def test_build_dataset_refuses_heldout_eval_as_a_training_side():
    import pytest
    c = config.Config.rl_from_sft()
    c.eval_split = "heldout_eval"
    with pytest.raises(SystemExit):
        train_dr.build_dataset(c, "eval")


def _task():
    return DRTask(
        task_id="fx-1", question="nationality of the director of Blue Harvest?",
        gold_answer="American", gold_aliases=["american"],
        supporting_titles=["Blue Harvest (film)", "Jane Doe (director)"],
        passages=[("Blue Harvest (film)", "Blue Harvest is a 2009 drama directed by Jane Doe."),
                  ("Jane Doe (director)", "Jane Doe is an American film director born in Ohio."),
                  ("Blue Harvest (album)", "A 2015 album by the band Fjord."),
                  ("John Roe (producer)", "A British film producer.")])


def test_env_reports_behaviour_counters_answered_case():
    cfg = config.Config.sanity_preset()
    e = dr_env.DeepResearchEnv.from_dict({"task": _task(), "cfg": cfg})
    e.reset()
    e.step("Thought: find it\nAction: search[Blue Harvest director]")
    e.step("Thought: read it\nAction: read[Blue Harvest (film)]")
    e.step("Thought: read her\nAction: read[Jane Doe (director)]")
    _, _, done, info = e.step("Thought: done\nAction: answer[American [Blue Harvest (film)] [Jane Doe (director)]]")
    assert done
    assert info["n_reads"] == 2 and info["n_searches"] == 1 and info["n_tool_calls"] == 4
    assert info["answered"] == 1.0 and info["capped"] == 0.0


def test_env_reports_capped_when_budget_runs_out():
    cfg = config.Config.sanity_preset()          # max_turns=4
    e = dr_env.DeepResearchEnv.from_dict({"task": _task(), "cfg": cfg})
    e.reset()
    done, info = False, {}
    for _ in range(cfg.max_turns):
        _, _, done, info = e.step("Thought: keep looking\nAction: search[Blue Harvest]")
    assert done
    assert info["answered"] == 0.0 and info["capped"] == 1.0
    assert info["n_searches"] == cfg.max_turns and info["n_reads"] == 0


def test_process_filter_keeps_wrong_answers_but_not_cite_without_read():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "distill"))
    from build_sft import is_process_clean
    base = {"history": [{"role": "user", "content": "q"}], "api_gave_up": False,
            "terminated_cleanly": True, "n_citations": 1, "read_before_cite_rate": 1.0}
    assert is_process_clean({**base, "correct": True})
    assert is_process_clean({**base, "correct": False})          # wrong answer: KEPT
    assert not is_process_clean({**base, "read_before_cite_rate": 0.5})   # cited unread
    assert not is_process_clean({**base, "terminated_cleanly": False})    # never answered
    assert not is_process_clean({**base, "n_citations": 0})
    assert not is_process_clean({**base, "history": []})


def test_f18_merged_response_budget_is_separate_from_the_per_turn_cap():
    """FINDING F18: verl's data.max_response_length bounds the MERGED multi-turn
    response; the per-turn cap is max_new_tokens. They must not be the same number
    for a multi-turn agent, or trajectories are truncated and their reward zeroed."""
    c = config.Config.rl_from_sft()
    assert c.verl_max_response_length is not None and c.verl_max_response_length >= 2048
    assert c.verl_max_response_length > c.max_new_tokens
    ov = train_dr.verl_overrides(c, Path("/tmp/t"), Path("/tmp/v"))
    assert f"data.max_response_length={c.verl_max_response_length}" in ov
    # historical presets keep the old (broken, but reproducible) behaviour
    c0 = config.Config.cloud_preset()
    assert c0.verl_max_response_length is None
    assert "data.max_response_length=256" in train_dr.verl_overrides(c0, Path("/tmp/t"), Path("/tmp/v"))
