"""THE single source of truth for which questions belong to which stage.

Harpreet, 2026-08-26: *"I'm also hoping you to have some eval set as well when you do SFT,
and during RL have some questions not covered during SFT as well — actually lots of
questions."* Exactly right, and this module exists so that is enforced rather than
remembered.

WHY A MODULE AND NOT A CONVENTION. Asking the same question twice with a different `n`
returns DIFFERENT questions. `data.load_tasks` draws `n` per source, concatenates, then
`random.Random(seed).shuffle(...)` — and a shuffle's permutation depends on the list
length, so the prefix is NOT stable across different `n`. `diagnosis1.load_pool`'s
docstring warns about this, and on 2026-08-26 I walked into it anyway: the GPT-teacher
pilot was reported as running on "the same 16 questions the student was probed on", and
the real overlap was 0/16. A comparison claimed as controlled was nothing of the kind.
Splits defined by prose invite that. Splits defined by code, pinned to a fixed draw size,
and asserted disjoint do not.

THE STAGES, and why each needs its own questions:

    heldout_eval   the frozen yardstick. From the dataset's VALIDATION split with
                   eval_seed — different rows entirely, not a slice of the train draw.
                   Never collected on, never trained on, never used to pick a threshold.
                   Every stage reports against THIS, so numbers stay comparable across
                   Diagnosis 1, SFT, and GRPO (NOTES.md's eval_seed=9999 convention).

    sft_collect    the teacher generates trajectories here. These become training data.

    sft_dev        SFT's own held-out set. Needed because train loss falling while
                   held-out performance stalls IS the overfitting signal, and a small
                   mined dataset overfits fast (the plan doc's data-hygiene section).
                   Disjoint from sft_collect so it measures generalisation, not recall.

    rl_train       GRPO's pool, disjoint from everything above. THIS IS THE ONE THAT
                   MATTERS MOST and the easiest to get wrong. If RL trains on questions
                   the model already memorised in SFT, every rollout in a group agrees,
                   the group has zero advantage, and it contributes no gradient — the
                   `dead_groups_pct` climbing to 84% that TRAINING_HISTORY_LOG.md
                   already records. RL needs questions the policy finds genuinely
                   uncertain. Sized large deliberately: ~105k questions exist upstream
                   (HotpotQA 90,447 + 2Wiki 15,000), so there is no reason to be stingy.

STABILITY. The partition is a fixed slice of ONE draw of exactly `POOL_SIZE` questions.
Changing POOL_SIZE, train_seed, or the dataset mix re-shuffles everything and invalidates
every previously collected split — so those are pinned here, not passed in.
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import data as dr_data

# Pinned. Changing any of these re-shuffles every split — do not tune casually.
POOL_SIZE = 20_000          # one draw from the TRAIN split, partitioned by index below
HELDOUT_SIZE = 500          # from the VALIDATION split — different rows, not a slice

# [start, end) indices into the single POOL_SIZE draw. Disjoint by construction and
# asserted at import time by `validate_splits()`.
RANGES: dict[str, tuple[int, int]] = {
    "sft_collect": (0, 4_000),        # teacher generates here -> SFT training data
    "sft_dev":     (4_000, 4_500),    # SFT's held-out: early-stop / overfitting watch
    "rl_train":    (4_500, 19_000),   # GRPO — 14,500 questions SFT has never seen
    "reserve":     (19_000, 20_000),  # untouched; for a stage we have not planned yet
}


def _train_pool(cfg):
    """The one canonical draw. Always exactly POOL_SIZE, never a different `n` — that is
    the whole point (see the module docstring)."""
    big = replace(cfg, num_train_examples=POOL_SIZE, num_eval_examples=POOL_SIZE)
    return dr_data.load_tasks(big, "train")


MUSIQUE_DEV_SIZE = 300      # 2026-09-08: OOD generalisation set, from MuSiQue's validation split


def get_split(name: str, cfg, limit: int | None = None):
    """Tasks for one stage. `limit` truncates from the front (stable), for cheap runs."""
    if name == "musique_dev":
        # Out-of-distribution: a different dataset entirely (GENERALIZATION_EVAL.md), so
        # disjointness from the HotpotQA/2Wiki pool is by construction. Drawn with
        # eval_seed from MuSiQue's own validation split; sized like heldout_eval's
        # reported slice (300) so the two tables are read the same way.
        tasks = dr_data._load_musique("validation", MUSIQUE_DEV_SIZE, cfg.eval_seed)
        return tasks[:limit] if limit else tasks
    if name == "heldout_eval":
        big = replace(cfg, num_eval_examples=HELDOUT_SIZE)
        tasks = dr_data.load_tasks(big, "eval")[:HELDOUT_SIZE]
        return tasks[:limit] if limit else tasks
    if name not in RANGES:
        raise KeyError(f"unknown split {name!r}; known: "
                       f"{['heldout_eval', 'musique_dev', *RANGES]}")
    a, b = RANGES[name]
    tasks = _train_pool(cfg)[a:b]
    return tasks[:limit] if limit else tasks


def validate_splits(cfg) -> dict:
    """Prove the partition is disjoint instead of trusting the arithmetic.

    Checks the thing that actually matters — that no question appears in two stages —
    against real loaded task_ids, not against the index ranges. An off-by-one in RANGES
    would pass an arithmetic check and silently leak SFT questions into the RL pool,
    which is invisible in any training curve and would quietly inflate every RL number.
    """
    ids = {name: {t.task_id for t in get_split(name, cfg)} for name in
           ["heldout_eval", *RANGES]}
    names = list(ids)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlap = ids[a] & ids[b]
            if overlap:
                raise AssertionError(
                    f"SPLIT LEAK: {a} and {b} share {len(overlap)} questions, "
                    f"e.g. {sorted(overlap)[:3]}")
    for name, s in ids.items():
        n_expected = (HELDOUT_SIZE if name == "heldout_eval"
                      else RANGES[name][1] - RANGES[name][0])
        if len(s) != n_expected:
            raise AssertionError(
                f"{name}: {len(s)} distinct ids but expected {n_expected} — either the "
                f"upstream dataset shrank or ids are colliding")
    return {k: len(v) for k, v in ids.items()}


def summary(cfg) -> str:
    counts = validate_splits(cfg)
    lines = [f"Split definition (POOL_SIZE={POOL_SIZE}, train_seed={cfg.train_seed}, "
             f"eval_seed={cfg.eval_seed})", ""]
    lines.append(f"  {'split':14}{'n':>8}  purpose")
    lines.append(f"  {'-'*14}{'-'*8}  {'-'*58}")
    purpose = {
        "heldout_eval": "FROZEN yardstick — validation split, never trained/collected on",
        "sft_collect":  "teacher generates trajectories here -> SFT training data",
        "sft_dev":      "SFT held-out — early-stop / overfitting watch",
        "rl_train":     "GRPO pool — never seen in SFT (avoids dead groups)",
        "reserve":      "untouched, held back for an unplanned stage",
    }
    for k in ["heldout_eval", *RANGES]:
        lines.append(f"  {k:14}{counts[k]:>8}  {purpose[k]}")
    lines.append("")
    lines.append("  disjointness: VERIFIED against real task_ids, not index arithmetic")
    return "\n".join(lines)


if __name__ == "__main__":
    import config as dr_config
    from dotenv import load_dotenv
    load_dotenv(_HERE / ".env")
    print(summary(dr_config.Config.cloud_preset()))
