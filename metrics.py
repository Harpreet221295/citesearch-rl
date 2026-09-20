"""Answer-scoring + localized eval metrics — pure mechanics (no learning logic).

These are the standard multi-hop QA metrics (SQuAD/HotpotQA style: normalized
Exact-Match and token-F1) plus the deep-research-specific localized signals the
capstone insists on (retrieval hit-rate, step counts, groundedness gap). The
REWARD (reward.py, TODO #1) may CALL exact_match/token_f1 as its outcome term,
but how it combines them with groundedness + efficiency is Harpreet's policy —
these functions are just honest measurements.

`normalize_answer` follows the SQuAD recipe (lowercase, strip articles/punct/extra
space) so "The USA." == "usa" — otherwise EM would be brutally, misleadingly low.
"""
from __future__ import annotations

import re
import string
from collections import Counter
from typing import Iterable


_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)


def normalize_answer(s: str) -> str:
    """SQuAD normalization: lowercase, remove punctuation, articles, extra whitespace."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def exact_match(pred: str, golds: Iterable[str]) -> bool:
    """1 if the normalized prediction equals ANY normalized gold/alias."""
    p = normalize_answer(pred or "")
    return any(p == normalize_answer(g) for g in golds if g is not None)


def token_f1(pred: str, golds: Iterable[str]) -> float:
    """Best token-level F1 of the prediction against any gold/alias (HotpotQA metric)."""
    best = 0.0
    p_toks = normalize_answer(pred or "").split()
    for g in golds:
        if g is None:
            continue
        g_toks = normalize_answer(g).split()
        if not p_toks and not g_toks:
            best = max(best, 1.0)
            continue
        if not p_toks or not g_toks:
            continue
        common = Counter(p_toks) & Counter(g_toks)
        n_same = sum(common.values())
        if n_same == 0:
            continue
        prec = n_same / len(p_toks)
        rec = n_same / len(g_toks)
        best = max(best, 2 * prec * rec / (prec + rec))
    return best


def answer_recall_in_context(pred: str, passages_text: str) -> float:
    """Cheap, ungameable groundedness floor: fraction of the prediction's content
    tokens that actually appear in the retrieved passage text. A verbose answer that
    invents facts scores low even if a judge is fooled — a hacking check that needs
    no model. (reward.py may use this as a groundedness prior; the judge is the
    richer signal.)"""
    p_toks = [t for t in normalize_answer(pred or "").split()]
    if not p_toks:
        return 0.0
    ctx = set(normalize_answer(passages_text or "").split())
    return sum(1 for t in p_toks if t in ctx) / len(p_toks)


def aggregate(rows: list[dict]) -> dict:
    """Mean each numeric field across a list of per-example metric dicts — the eval
    gate's summary row. Missing keys are skipped per-row (not counted as 0)."""
    if not rows:
        return {}
    keys = {k for r in rows for k, v in r.items() if isinstance(v, (int, float, bool))}
    out = {}
    for k in keys:
        vals = [float(r[k]) for r in rows if k in r and isinstance(r[k], (int, float, bool))]
        out[k] = sum(vals) / len(vals) if vals else 0.0
    out["n"] = len(rows)
    return out
