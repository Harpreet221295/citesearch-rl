"""The stage splits must be disjoint and stable, or every downstream number is suspect.

Added 2026-08-26 after a real mistake: the GPT-teacher pilot was reported as running on
"the same 16 questions the student was probed on", and the actual overlap was 0/16 —
because `load_pool(n=16)` and `load_pool(n=2048)[:16]` return different questions. A
comparison presented as controlled was not. These tests make that class of error fail
loudly instead of silently producing a plausible table.

Slow-ish (they load the real datasets), but they guard the thing that invalidates
everything downstream if it breaks.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_LAB = Path(__file__).resolve().parent.parent
if str(_LAB) not in sys.path:
    sys.path.insert(0, str(_LAB))

import config as dr_config                                        # noqa: E402
import splits                                                     # noqa: E402


@pytest.fixture(scope="module")
def cfg():
    from dotenv import load_dotenv
    load_dotenv(_LAB / ".env")
    return dr_config.Config.cloud_preset()


def test_all_splits_are_disjoint(cfg):
    """THE test. A leak between sft_collect and rl_train is invisible in any training
    curve and would quietly inflate every RL number."""
    counts = splits.validate_splits(cfg)          # raises on any overlap
    assert counts["sft_collect"] == 4000
    assert counts["rl_train"] == 14500
    assert counts["sft_dev"] == 500
    assert counts["heldout_eval"] == 500


def test_rl_pool_is_large_and_unseen(cfg):
    """RL needs questions SFT never touched, and lots of them: if the policy memorised
    the answer during SFT, every rollout in a group agrees, advantage is zero and the
    group contributes no gradient (the dead-groups failure already in the training log)."""
    sft = {t.task_id for t in splits.get_split("sft_collect", cfg)}
    rl = {t.task_id for t in splits.get_split("rl_train", cfg)}
    assert not (sft & rl)
    assert len(rl) > 3 * len(sft), "RL pool should dwarf the SFT pool"


def test_heldout_comes_from_a_different_dataset_split(cfg):
    """Not a slice of the train draw — genuinely different rows, so it stays a clean
    yardstick even if the train partition is ever re-cut."""
    held = {t.task_id for t in splits.get_split("heldout_eval", cfg)}
    train_all = set()
    for name in splits.RANGES:
        train_all |= {t.task_id for t in splits.get_split(name, cfg)}
    assert not (held & train_all)


def test_splits_are_stable_across_calls(cfg):
    """Two calls must give identical questions, or collected data cannot be matched back
    to the split it came from."""
    a = [t.task_id for t in splits.get_split("sft_collect", cfg, limit=32)]
    b = [t.task_id for t in splits.get_split("sft_collect", cfg, limit=32)]
    assert a == b


def test_limit_is_a_stable_prefix(cfg):
    """--batch style truncation must be a prefix of the full split, so an incremental
    collection can resume without re-drawing different questions."""
    small = [t.task_id for t in splits.get_split("sft_collect", cfg, limit=16)]
    big = [t.task_id for t in splits.get_split("sft_collect", cfg, limit=64)]
    assert big[:16] == small


def test_the_trap_that_caused_this_module(cfg):
    """Documents WHY splits.py exists, by demonstrating the failure it prevents.

    `load_pool` re-draws with a different `n`, and the shuffle permutation depends on the
    list length — so the 'first 16' of a 16-draw and of a 2048-draw are different
    questions. Anyone tempted to go back to ad-hoc `load_pool(n=...)` slicing should read
    this assertion first.
    """
    sys.path.insert(0, str(_LAB / "rft_diagnosis"))
    from diagnosis1 import load_pool
    a = {t.task_id for t in load_pool(cfg, "train", n=16, offset=0)}
    b = {t.task_id for t in load_pool(cfg, "train", n=2048, offset=0)[:16]}
    assert a != b, ("load_pool became prefix-stable — if that is now true by design, "
                    "this test and splits.py's rationale should be revisited")
