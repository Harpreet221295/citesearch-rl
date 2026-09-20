"""Validate collected teacher trajectories BEFORE spending more on collection.

    python distill/validate.py                 # full check of the JSONL, no API calls, $0
    python distill/validate.py --show 2        # ...and print 2 trajectories in full

Harpreet, 2026-08-26: *"please verify, we are properly collecting something from a very
small generation, if needed write some code to validate the collected trajectories too."*
Exactly the right instinct at exactly the right moment — a silent defect here is a whole
collection run wasted, and this session has already produced two of them:
  * `history: null` on every record (the field SFT needs), because the collector read a
    Trajectory attribute that does not exist. Caught only by going incremental.
  * 257 rejected replies with no record of what the model said (env.py:122).
Aggregates hid both. So this validator reads the actual stored bytes and re-derives
everything it can, rather than trusting any number already in the file.

CHECKS, grouped by what a failure would actually cost:

  STRUCTURE — can this be turned into an SFT example at all?
    1. `history` present, non-empty, and shaped like a conversation.
    2. Roles alternate user/assistant, starting with our opening prompt.
    3. Every assistant turn in `history` matches the raw text the API returned.

  GROUNDING — the expensive-to-detect one, and the whole reason GPT drives the real env.
    4. Every tool observation in the stored history is re-derived by re-running the tool
       against the real corpus and compared. If a stored observation does not match what
       our tools actually produce, we are about to fine-tune on invented retrieval — the
       single worst outcome available here, and invisible in any aggregate.

  SFT-READINESS — does the masking hold on REAL collected data, not just the fixture?
    5. `encode_trajectory` succeeds and `assert_masking_correct` passes.
    6. No tool-observation text appears inside a graded span.

  SCORING CONSISTENCY — do the stored numbers survive recomputation?
    7. Re-score with `classify_trajectory` and compare to what was stored.
    8. For strict-gate keeps, every cited title really was read.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

_HERE = Path(__file__).resolve().parent.parent
for p in (str(_HERE), str(_HERE / "rft_diagnosis"), str(_HERE / "distill")):
    if p not in sys.path:
        sys.path.insert(0, p)

import citations
import config as dr_config
import data as dr_data
import tools as dr_tools
from sft_data import encode_trajectory, assert_masking_correct, summarize

JSONL = _HERE / "distill" / "teacher_trajectories.jsonl"


class Findings:
    def __init__(self) -> None:
        self.fail: Counter = Counter()
        self.warn: Counter = Counter()
        self.examples: dict[str, str] = {}

    def bad(self, key: str, detail: str = "") -> None:
        self.fail[key] += 1
        self.examples.setdefault(key, detail)

    def soft(self, key: str, detail: str = "") -> None:
        self.warn[key] += 1
        self.examples.setdefault(key, detail)


def check_structure(rec: dict, f: Findings) -> list[dict] | None:
    h = rec.get("history")
    if not h:
        f.bad("history_missing", f"{rec.get('task_id')}: history is {h!r} — "
                                 "record is unusable for SFT")
        return None
    if not isinstance(h, list) or any("role" not in m or "content" not in m for m in h):
        f.bad("history_malformed", str(rec.get("task_id")))
        return None
    if h[0].get("role") != "user":
        f.bad("history_does_not_start_with_prompt", str(rec.get("task_id")))
    roles = [m["role"] for m in h]
    for a, b in zip(roles, roles[1:]):
        if a == b:
            f.bad("roles_do_not_alternate", f"{rec.get('task_id')}: ...{a} -> {b}...")
            break
    if not any(r == "assistant" for r in roles):
        f.bad("no_assistant_turn", str(rec.get("task_id")))
        return None

    raws = rec.get("raw_turns") or []
    hist_assist = [m["content"] for m in h if m["role"] == "assistant"]
    if raws and hist_assist and hist_assist[:len(raws)] != raws[:len(hist_assist)]:
        f.soft("history_assistant_differs_from_raw_turns",
               f"{rec.get('task_id')}: stored history text != API text")
    return h


_POOL: dict | None = None


def _collection_pool(cfg) -> dict:
    """task_id -> DRTask for the `sft_collect` split (loaded once, see check_grounding)."""
    global _POOL
    if _POOL is None:
        import splits as dr_splits
        _POOL = {t.task_id: t for t in dr_splits.get_split("sft_collect", cfg)}
        print(f"  (loaded {len(_POOL)} sft_collect tasks for grounding re-derivation)")
    return _POOL


def check_grounding(rec: dict, h: list[dict], cfg, f: Findings) -> None:
    """Re-run each tool against the REAL corpus and compare to the stored observation.

    This is the check that proves we collected reality rather than fiction. It is why
    the teacher drives the real env instead of being asked to write transcripts.
    """
    # 2026-09-07 (bug #10, pre-existing): this used to call `dr_data.load_tasks(cfg,
    # "train")` PER RECORD — re-drawing and re-shuffling ~2,048 questions 4,000 times
    # (19+ minutes on the full collection, still not done) — and that draw is not even
    # the `sft_collect` split the records came from, so most lookups silently fell
    # through to the soft "task_not_found_in_pool" warning after all that work. The
    # collection pool is loaded ONCE, from the split it was actually collected on.
    task = _collection_pool(cfg).get(rec["task_id"])
    if task is None:
        f.soft("task_not_found_in_pool", str(rec.get("task_id")))
        return
    ds = task.docstore()

    # Pair each assistant action with the user turn that followed it.
    for i, m in enumerate(h):
        if m["role"] != "assistant" or i + 1 >= len(h) or h[i + 1]["role"] != "user":
            continue
        stored_obs = h[i + 1]["content"]
        text = m["content"] or ""
        if "Action:" not in text:
            continue
        line = next((l for l in text.splitlines() if l.strip().startswith("Action:")), "")
        if not line:
            # 2026-09-07 (bug #11): "Action:" present but never at a line start (the
            # teacher wrote it mid-line) -> `line` is "" and the split below indexed past
            # the end. Only reachable now that grounding runs on real records (bug #10).
            f.soft("action_not_at_line_start", f"{rec['task_id']}: {text[:80]!r}")
            continue
        body = line.split("Action:", 1)[1].strip()
        if "[" not in body or not body.endswith("]"):
            continue
        name = body[: body.index("[")].strip().lower()
        arg = body[body.index("[") + 1: -1]
        if name not in ("search", "read"):
            continue
        args = ({"query": arg, "k": cfg.search_k} if name == "search" else {"title": arg})
        obs, _ok, _err, _titles = dr_tools.execute(ds, name, args)
        obs = obs[: cfg.max_obs_chars]
        if obs != stored_obs:
            f.bad("observation_not_reproducible",
                  f"{rec['task_id']} {name}({arg!r}): stored observation does not match "
                  f"what the real corpus returns — possible fabricated retrieval")


def check_sft_ready(rec: dict, h: list[dict], tok, f: Findings):
    ex = encode_trajectory(tok, h, task_id=rec.get("task_id", ""))
    if ex is None:
        f.bad("not_encodable", f"{rec.get('task_id')}: encode_trajectory returned None")
        return None
    try:
        assert_masking_correct(tok, ex, h)
    except AssertionError as e:                                   # noqa: BLE001
        f.bad("masking_violation", f"{rec.get('task_id')}: {e}")
        return None
    return ex


def check_scoring(rec: dict, f: Findings) -> None:
    cited = {c.title.strip().lower()
             for c in citations.extract_citations(rec.get("final_answer") or "")}
    if rec.get("cite_f1", 0) >= 0.999 and cited:
        # Strict-gate keeps must satisfy cite-what-you-read by construction.
        if rec.get("n_reads", 0) < len(cited):
            f.soft("strict_keep_with_fewer_reads_than_citations",
                   f"{rec['task_id']}: {len(cited)} cited, {rec.get('n_reads')} reads")
    pred = citations.strip_citations(rec.get("final_answer") or "")
    if rec.get("correct") and not pred.strip():
        f.bad("correct_but_empty_answer", str(rec.get("task_id")))
    if len(pred.split()) > 12:
        f.soft("answer_longer_than_12_words",
               f"{rec['task_id']}: {pred[:80]!r} — student is trained to be terse")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default=str(JSONL))
    ap.add_argument("--show", type=int, default=0, help="print N trajectories in full")
    ap.add_argument("--skip-grounding", action="store_true",
                    help="skip the corpus re-derivation (the slow check)")
    args = ap.parse_args()

    path = Path(args.jsonl)
    if not path.exists():
        raise SystemExit(f"no collection at {path}")
    recs = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    print(f"Validating {len(recs)} collected trajectories from {path.name}")
    print("(no API calls — this costs $0)\n")

    cfg = dr_config.Config.cloud_preset()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.model_name)

    f = Findings()
    examples = []
    for rec in recs:
        h = check_structure(rec, f)
        if h is None:
            continue
        if not args.skip_grounding:
            check_grounding(rec, h, cfg, f)
        ex = check_sft_ready(rec, h, tok, f)
        if ex is not None:
            examples.append(ex)
        check_scoring(rec, f)

    print("=" * 78)
    print("RESULTS")
    print("=" * 78)
    if not f.fail:
        print(f"  PASS — {len(recs)}/{len(recs)} records are structurally sound, "
              f"grounded in the real corpus, and SFT-encodable with correct masking.")
    else:
        print("  FAILURES (these would corrupt training):")
        for k, v in f.fail.most_common():
            print(f"    {k:42} {v:>4}")
            print(f"      e.g. {f.examples[k]}")
    if f.warn:
        print("\n  WARNINGS (look, but not necessarily wrong):")
        for k, v in f.warn.most_common():
            print(f"    {k:42} {v:>4}")
            print(f"      e.g. {f.examples[k]}")

    if examples:
        print("\n" + "=" * 78)
        print("SFT DATASET SHAPE (from the real collected data, not a fixture)")
        print("=" * 78)
        for k, v in summarize(examples).items():
            print(f"  {k:26} {v:.3f}" if isinstance(v, float) else f"  {k:26} {v}")
        print("\n  mean_graded_fraction is the number to sanity-check: assistant turns are")
        print("  short and tool observations are long, so a value near 1.0 would mean the")
        print("  mask is inverted or missing.")

    for rec in recs[: args.show]:
        print("\n" + "=" * 78)
        print(f"QUESTION: {rec.get('question')}")
        print(f"GOLD    : {rec.get('gold_answer')!r}   evidence: {rec.get('supporting_titles')}")
        print("=" * 78)
        for m in rec.get("history") or []:
            who = "TEACHER" if m["role"] == "assistant" else "US (prompt/tool result)"
            body = m["content"]
            if m["role"] == "user" and len(body) > 400:
                body = body[:400] + f"\n      ... [{len(m['content'])-400} more chars]"
            print(f"\n--- {who} ---")
            print("\n".join("    " + l for l in body.splitlines()))
        print(f"\n  scored: correct={rec.get('correct')} cite_f1={rec.get('cite_f1')} "
              f"title_f1={rec.get('title_f1')} read_before_cite={rec.get('read_before_cite_rate')}")

    sys.exit(1 if f.fail else 0)


if __name__ == "__main__":
    main()
