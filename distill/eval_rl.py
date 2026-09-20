"""The RL gate: SFT vs SFT+RL on the same questions, one engine, LoRA hot-swapped.

    python distill/eval_rl.py --adapter step25=runs/.../_merged/step25/lora_adapter --split sft_dev --n 128
    python distill/eval_rl.py --adapter best=... --split heldout_eval --n 300     # ONCE, at the end

Companion to eval_sft.py (which compares base / base+example / SFT) — reuse, not a
rebuild: the per-episode classifier, the summary, and the anti-hacking probes are
imported from there so every number means the same thing across stages.

WHY A SEPARATE ENGINE BASE. The RL policy is (base + SFT adapter merged) + a fresh RL
LoRA (see distill/merge_sft.py for why). vLLM hosts ONE base model per engine, so the
engine here is built on the MERGED SFT weights: the "sft" arm is that engine with no
adapter, and each --adapter arm is the same engine with an RL LoRA hot-swapped in. The
base-model arms already exist in eval_sft_A_*.json and are not re-run.

THE RL GATE (SFT_RL_PLAN.md §4, written before any RL checkpoint existed):
  correct_rate improves over SFT, cite_f1 does not regress, read_before_cite does not
  regress, and no anti-hacking probe fires. Plus the probe this stage specifically risks
  (START_HERE.md): turns-per-episode collapsing toward 2 — reported as its own line.
`capped_rate` (episodes that exhaust the turn budget) is THE target failure (F12) and is
reported next to the headline so the mechanism of any gain is visible.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

_HERE = Path(__file__).resolve().parent.parent
for p in (str(_HERE), str(_HERE / "rft_diagnosis"), str(_HERE / "distill")):
    if p not in sys.path:
        sys.path.insert(0, p)

import config as dr_config
import splits as dr_splits
from diagnosis1 import _TURN_STOP_SEQUENCES
from eval_sft import score_all, summarize, HEADLINE, KEYMAP, PROBES
from evaluate import build_vllm_engine, close_vllm_engine, run_batched_rollouts

MERGED = _HERE / "sft_merged"


def _slice_stats(rows: list[dict]) -> dict:
    n = max(1, len(rows))
    return {"n": len(rows),
            "correct": sum(r["correct"] for r in rows) / n,
            "cite_f1": sum(r["cite_f1"] for r in rows) / n,
            "correct_and_cited": sum(1 for r in rows if r["bucket"] == "correct_and_cited") / n,
            "capped": sum(1 for r in rows if r.get("_capped")) / n}


def summarize_rl(rows: list[dict], max_turns: int, meta_by_id: dict | None = None) -> dict:
    b = summarize(rows)
    n = len(rows)
    for r in rows:
        r["_capped"] = r["n_tool_calls"] >= max_turns
    b["capped_rate"] = sum(1 for r in rows if r["_capped"]) / n
    b["answered_rate"] = sum(1 for r in rows if r["terminated_cleanly"]) / n
    b["correct_and_cited_rate"] = sum(1 for r in rows if r["bucket"] == "correct_and_cited") / n
    # 2026-09-08 (GENERALIZATION_EVAL.md): per-source and per-hop-count slices, so a gain
    # can be attributed (uniform, or concentrated on the easy 2-hop comparison questions).
    meta_by_id = meta_by_id or {}
    by_src, by_hops = {}, {}
    for r in rows:
        m = meta_by_id.get(r["task_id"], {})
        src = m.get("dataset") or r["task_id"].split("-")[0]
        by_src.setdefault(src, []).append(r)
        hops = m.get("n_hops") or (2 if src in ("hotpotqa", "hotpot") else None)
        if hops:
            by_hops.setdefault(str(hops), []).append(r)
    b["by_dataset"] = {k: _slice_stats(v) for k, v in sorted(by_src.items())}
    b["by_hops"] = {k: _slice_stats(v) for k, v in sorted(by_hops.items())}
    return b


def table(arms: dict[str, dict]) -> None:
    names = list(arms)
    w = 10
    print(f"\n{'':38}" + "".join(f"{n[:w-1]:>{w}}" for n in names))
    print("-" * (38 + w * len(names)))
    rows = HEADLINE + [("correct_and_cited_rate", "correct AND properly cited"),
                       ("capped_rate", "hit the turn cap (F12; lower better)"),
                       ("answered_rate", "produced an answer")]
    for key, label in rows:
        k = KEYMAP.get(key, key)
        print(f"  {label:36}" + "".join(f"{arms[n].get(k, 0.0):>{w}.3f}" for n in names))
    print(f"\n  ANTI-HACKING PROBES")
    print("-" * (38 + w * len(names)))
    for key, label in PROBES + [("mean_n_searches", "searches per episode")]:
        print(f"  {label:36}" + "".join(f"{arms[n].get(key, 0.0):>{w}.2f}" for n in names))
    for n in names:
        print(f"\n  {n:10} buckets {arms[n].get('bucket_pct')}")
        print(f"  {n:10} hops    {arms[n].get('hop_count_distribution')}")
    for key, label in (("by_dataset", "BY SOURCE DATASET"), ("by_hops", "BY HOP COUNT")):
        slices = sorted({s for n in names for s in arms[n].get(key, {})})
        if len(slices) < 2:
            continue
        print(f"\n  {label} (correct / cite_f1 / correct-and-cited / capped)")
        for sl in slices:
            line = f"  {sl:14}"
            for n in names:
                st = arms[n].get(key, {}).get(sl)
                line += (f"  {n[:8]}: n={st['n']:<4} {st['correct']:.3f} {st['cite_f1']:.3f} "
                         f"{st['correct_and_cited']:.3f} {st['capped']:.3f}" if st else f"  {n[:8]}: -")
            print(line)


def gate(sft: dict, rl: dict, name: str) -> bool:
    checks = [
        ("correct_rate improves over SFT (> +0.00)",
         rl["correct_rate"] > sft["correct_rate"]),
        ("cite_f1 does not regress (>= SFT - 0.02)",
         rl["mean_cite_f1"] >= sft["mean_cite_f1"] - 0.02),
        ("read_before_cite does not regress (>= SFT - 0.02)",
         rl["mean_read_before_cite"] >= sft["mean_read_before_cite"] - 0.02),
        ("turns per episode did not collapse (>= 3.0)",
         rl["mean_n_steps"] >= 3.0),
        ("no answer-length inflation (< 1.5x SFT)",
         rl["mean_answer_words"] < 1.5 * max(1.0, sft["mean_answer_words"])),
        ("no citation-count inflation (< 1.5x SFT unless title_f1 also rose)",
         rl["mean_n_citations"] < 1.5 * max(0.5, sft["mean_n_citations"])
         or rl["mean_title_f1"] > sft["mean_title_f1"]),
    ]
    print(f"\n  RL GATE for {name} (SFT_RL_PLAN.md §4 + START_HERE.md's turn-collapse risk)")
    print("-" * 78)
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    passed = all(ok for _, ok in checks)
    print(f"  => {'GATE PASSED' if passed else 'GATE FAILED'}")
    return passed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(MERGED), help="merged-SFT model dir (the engine's base)")
    ap.add_argument("--adapter", action="append", default=[],
                    help="name=path of an RL LoRA adapter; repeatable")
    ap.add_argument("--split", default="sft_dev",
                    choices=["sft_dev", "heldout_eval", "rl_train", "musique_dev"])
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from dotenv import load_dotenv
    load_dotenv(_HERE / ".env")
    from transformers import AutoTokenizer
    from vllm import SamplingParams
    from vllm.lora.request import LoRARequest

    adapters = []
    for spec in args.adapter:
        name, _, path = spec.partition("=")
        assert path, f"--adapter must be name=path, got {spec!r}"
        assert (Path(path) / "adapter_config.json").exists(), f"no adapter at {path}"
        adapters.append((name, path))

    # The prompt the SFT adapter was trained under and RL rolled out under.
    cfg = replace(dr_config.Config.cloud_preset(), model_name=args.base,
                  include_worked_example=False, lora_r=args.lora_rank)
    tasks = dr_splits.get_split(args.split, cfg, limit=args.n)
    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    sp = SamplingParams(temperature=args.temperature,
                        top_p=1.0 if args.temperature == 0 else cfg.top_p,
                        max_tokens=cfg.max_new_tokens, stop=_TURN_STOP_SEQUENCES)

    print(f"RL gate — SFT vs SFT+RL, one engine on the merged SFT weights")
    print(f"  split    : {args.split} ({len(tasks)} questions)")
    print(f"  base     : {args.base}")
    print(f"  adapters : {adapters or '(none — SFT arm only)'}")
    print(f"  prompt   : include_worked_example=False   sampling: T={args.temperature}")

    meta_by_id = {t.task_id: dict(t.meta) for t in tasks}
    llm = build_vllm_engine(cfg, enable_lora=bool(adapters))
    arms, raw = {}, {}
    try:
        print("\n  [sft] merged SFT weights, no adapter ...", flush=True)
        rows = score_all(run_batched_rollouts(tasks, cfg, llm, tok, sp), cfg)
        arms["sft"], raw["sft"] = summarize_rl(rows, cfg.max_turns, meta_by_id), rows
        for i, (name, path) in enumerate(adapters, start=1):
            print(f"  [{name}] + RL adapter {path} ...", flush=True)
            lora = LoRARequest(name, i, path)
            rows = score_all(run_batched_rollouts(tasks, cfg, llm, tok, sp, lora_request=lora), cfg)
            arms[name], raw[name] = summarize_rl(rows, cfg.max_turns, meta_by_id), rows
    finally:
        close_vllm_engine(llm)

    table(arms)
    gates = {name: gate(arms["sft"], arms[name], name) for name in arms if name != "sft"}

    out = Path(args.out or (_HERE / "distill" / f"eval_rl_{args.split}.json"))
    out.write_text(json.dumps({
        "split": args.split, "n": len(tasks), "base": args.base, "adapters": adapters,
        "temperature": args.temperature, "gates": gates,
        "summary": arms, "rows": raw,
    }, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
