"""Collect teacher trajectories INCREMENTALLY — small batches, resumable, cost-capped.

Questions come from `splits.get_split("sft_collect")` — NOT an ad-hoc `load_pool(n=...)`
slice. That is not a style preference: `load_pool` re-draws and re-shuffles for each `n`,
so two calls with different sizes return different questions. On 2026-08-26 that produced
a teacher-vs-student comparison reported as controlled whose real overlap was 0/16. See
splits.py.

Harpreet's instruction (2026-08-26): *"maybe we should not attempt complete generation in
one go, and proceed step by step, fixing any issues coming along without wasting too much
money."* That instinct immediately paid for itself: the first version of this script
stored `history: null` for every record (it tried to read `traj._history`, which does not
exist), so a single big run would have produced a whole collection unusable for SFT.

    python distill/collect.py --batch 8                     # first 8 of sft_collect
    python distill/collect.py --batch 50                    # next 50 UNSEEN questions
    python distill/collect.py --batch 50 --max-cost 0.50    # ...and stop at 50 cents
    python distill/collect.py --inspect                     # score what is on disk, spend $0

DESIGN, all three points aimed at "do not waste money":

 * APPEND-AS-YOU-GO. Every episode is written to a JSONL the moment it lands, not at the
   end. An interrupted run keeps everything it already paid for.
 * RESUME BY DEFAULT. Questions already in the JSONL are skipped, so re-running continues
   rather than re-buying. `--batch N` means "N questions I have not done yet".
 * HARD COST CAP that can actually fire mid-run (teacher.collect submits in waves). A cap
   checked only after everything is queued is not a cap.

`--inspect` re-scores the JSONL and prints the quality/yield report without calling the
API at all — the between-batches step. Look at it before buying the next batch.
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
import metrics as dr_metrics
import splits as dr_splits
from diagnosis1 import classify_trajectory, aggregate_breakdown
from teacher import Usage, collect, make_client, DEFAULT_MODEL, PRICING

JSONL = _HERE / "distill" / "teacher_trajectories.jsonl"


def load_done(path: Path) -> tuple[list[dict], Counter]:
    """Records already on disk, and how many episodes exist per task_id."""
    if not path.exists():
        return [], Counter()
    rows = []
    for i, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            # A run killed mid-write can leave one truncated line. Skip it loudly rather
            # than crash — the other N-1 records are still paid for and still good.
            print(f"  [collect] skipping malformed line {i} in {path.name}")
    return rows, Counter(r["task_id"] for r in rows)


def report(rows: list[dict], label: str) -> dict:
    """Quality + yield. No API calls — safe to run between batches, costs nothing."""
    if not rows:
        print(f"\n=== {label}: nothing collected yet ===")
        return {}
    b = aggregate_breakdown(rows)
    n = len(rows)
    print(f"\n=== {label} — {n} episodes, {len({r['task_id'] for r in rows})} questions ===")
    for k_ in ("correct_rate", "mean_cite_f1", "mean_title_f1", "mean_read_before_cite",
               "calls_read_rate", "terminated_cleanly_rate", "parse_ok_rate",
               "mean_n_reads", "mean_n_searches"):
        if k_ in b:
            print(f"  {k_:26} {b[k_]:.3f}")
    print(f"  hop_count_distribution     {b.get('hop_count_distribution')}")
    print(f"  buckets                    {dict(Counter(r['bucket'] for r in rows))}")
    gave_up = sum(1 for r in rows if r.get("api_gave_up"))
    if gave_up:
        print(f"  !! api_gave_up             {gave_up}/{n} episodes — EXCLUDE from training")

    corr = [r for r in rows if r["correct"]]
    print(f"\n  --- SFT yield (what the dual gate would keep) ---")
    for bar in (0.999, 0.75, 0.5, 0.34):
        kept = [r for r in corr if r.get("cite_f1", 0) >= bar]
        q = len({r["task_id"] for r in kept})
        print(f"    correct AND cite_f1 >= {bar:<5} : {len(kept):>4} trajectories "
              f"from {q:>3} distinct questions")

    # Diversity, which the plan doc insists on checking BEFORE training: a large but
    # narrow mined set teaches a narrow behaviour and still looks fine on a narrow eval.
    keep = [r for r in corr if r.get("cite_f1", 0) >= 0.999]
    if keep:
        hops = Counter(r["n_tool_calls"] for r in keep)
        per_q = Counter(r["task_id"] for r in keep)
        print(f"\n  --- diversity of the strict-gate set ({len(keep)} trajectories) ---")
        print(f"    hop counts        : {dict(sorted(hops.items()))}")
        print(f"    distinct questions: {len(per_q)}  "
              f"(max {max(per_q.values())} trajectories from any one question)")
        print(f"    answer lengths    : "
              f"{sorted({len((r.get('final_answer') or '').split()) for r in keep})}")
    return b


def score(task, traj, raws, history, cfg, api_gave_up: bool = False) -> dict:
    row = classify_trajectory(task, traj, cfg)
    row["final_answer"] = traj.final_answer
    row["gold_answer"] = task.gold_answer
    row["supporting_titles"] = list(task.supporting_titles)
    row["question"] = task.question
    row["raw_turns"] = raws
    row["history"] = history          # <- what sft_data.encode_trajectory consumes
    # An episode where the API gave up mid-way is not a demonstration of anything. Kept
    # on disk (we paid for it, and it is evidence) but flagged so tiering can drop it.
    row["api_gave_up"] = bool(api_gave_up)
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=None,
                    help="how many NOT-YET-COLLECTED questions to do this run")
    ap.add_argument("--split", default="sft_collect",
                    help="which stage's questions to collect (see splits.py). Collecting "
                         "outside sft_collect contaminates a later stage")
    ap.add_argument("--inspect", action="store_true",
                    help="score what is already on disk and exit — no API calls, $0")
    ap.add_argument("--k", type=int, default=1, help="episodes per question")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--max-cost", type=float, default=1.00,
                    help="hard USD cap for THIS run; stops mid-run when reached")
    ap.add_argument("--jsonl", default=str(JSONL))
    args = ap.parse_args()

    cfg = dr_config.Config.cloud_preset()
    path = Path(args.jsonl)
    done_rows, done_counts = load_done(path)

    if args.inspect:
        report(done_rows, f"ON DISK ({path.name})")
        print(f"\n  {len(done_rows)} episodes over {len(done_counts)} questions. "
              f"No API calls made — this cost $0.")
        return

    if args.split != "sft_collect":
        print(f"  !! collecting on {args.split!r}, NOT sft_collect. Anything collected "
              f"here becomes training data and contaminates that stage.")
    pool = dr_splits.get_split(args.split, cfg)
    todo = [t for t in pool if done_counts[t.task_id] < args.k]
    if args.batch:
        todo = todo[: args.batch]
    if not todo:
        print(f"Nothing to do — all {len(pool)} questions in {args.split} already have "
              f">= k={args.k} episodes in {path.name}. Use --inspect, or raise --k.")
        return

    usage = Usage(model=args.model)
    pin, pout = PRICING.get(args.model, (None, None))
    print(f"Teacher collection (incremental) — model={args.model}"
          + (f"  (${pin}/${pout} per 1M in/out)" if pin else ""))
    print(f"  split           : {args.split} ({len(pool)} questions total)")
    print(f"  already on disk : {len(done_rows)} episodes / {len(done_counts)} questions")
    print(f"  this batch      : {len(todo)} questions x k={args.k} "
          f"= {len(todo)*args.k} episodes")
    print(f"  cost cap        : ${args.max_cost:.2f} (hard, checked mid-run)")
    print(f"  appending to    : {path}")
    print()

    client = make_client()
    fh = path.open("a")
    new_rows: list[dict] = []

    def on_episode(task, traj, raws, history, api_gave_up=False):
        row = score(task, traj, raws, history, cfg, api_gave_up)
        fh.write(json.dumps(row) + "\n")
        fh.flush()                    # survive a kill -9; we already paid for this one
        new_rows.append(row)

    try:
        _res, stopped = collect(client, todo, cfg, usage, model=args.model,
                                temperature=args.temperature, k=args.k,
                                workers=args.workers, max_cost_usd=args.max_cost,
                                on_episode=on_episode)
    finally:
        fh.close()

    report(new_rows, "THIS BATCH")
    report(done_rows + new_rows, "CUMULATIVE")

    print(f"\n--- cost (estimate, not billing) ---")
    print(f"  this batch: {usage.summary()}")
    if new_rows:
        per = usage.cost_usd / len(new_rows)
        print(f"  ~${per:.4f}/episode -> ~${per*len(pool)*args.k:.2f} to finish "
              f"{args.split} ({len(pool)} questions) at k={args.k}")
    if stopped:
        print("\n  NOTE: stopped early on the cost cap. Re-run the same command to "
              "continue — completed questions are skipped automatically.")
    print(f"\n  next: python distill/collect.py --inspect     # free, look before buying more")


if __name__ == "__main__":
    main()
