"""Push an SFT LoRA adapter + a model card to its OWN Hub repo.

    python distill/push_sft.py --adapter distill/runs/sft_full/final \
        --repo harpreet22happy/deep-research-agent-sft-full \
        --eval distill/eval_sft_full_dev.json [--eval distill/eval_sft_full_heldout.json]

Never touches `harpreet22happy/deep-research-agent-sft` (the 418-example adapter) — that
repo stays as shipped on 2026-08-26. A new adapter gets a new repo, so both stay
addressable and the comparison between them stays reproducible.

The card is generated from the eval JSON(s) `eval_sft.py` wrote, not typed by hand, so
the numbers on the Hub are the numbers that were measured.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_HERE))

ROWS = [("correct_rate", "correct (exact match)"),
        ("mean_title_f1", "picked the right sources"),
        ("mean_read_before_cite", "verified them (read before citing)"),
        ("mean_cite_f1", "citation-F1 (the training reward)"),
        ("calls_read_rate", "called `read` at least once"),
        ("terminated_cleanly_rate", "finished cleanly"),
        ("zero_tool_call_rate", "never used a tool")]


def _table(ev: dict) -> str:
    arms = [("base", ev["base_noex"]), ("base + worked example", ev["base_with_example"]),
            ("**this adapter**", ev["tuned"])]
    out = ["| | " + " | ".join(a for a, _ in arms) + " |", "|---|" + "---|" * len(arms)]
    for k, label in ROWS:
        out.append(f"| {label} | " + " | ".join(f"{s.get(k, 0.0):.3f}" for _, s in arms) + " |")
    cc = [s.get("bucket_pct", {}).get("correct_and_cited", 0.0) for _, s in arms]
    out.append("| **correct AND properly cited** | " + " | ".join(f"**{v*100:.1f}%**" for v in cc) + " |")
    hop = ev["tuned"].get("hop_count_distribution", {})
    n = ev["n"]
    capped = hop.get("8", hop.get(8, 0))
    out.append("")
    out.append(f"Episodes hitting the 8-call cap (the never-commit failure): "
               f"**{capped}/{n} = {100*capped/max(1,n):.1f}%**.")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--eval", action="append", default=[], help="eval_sft.py JSON(s)")
    ap.add_argument("--history", default=None, help="sft_train history.json (for the loss table)")
    ap.add_argument("--note", default="", help="one-paragraph note for the card")
    args = ap.parse_args()

    from dotenv import load_dotenv
    load_dotenv(_HERE / ".env")
    from huggingface_hub import HfApi

    adapter = Path(args.adapter)
    assert (adapter / "adapter_config.json").exists(), f"no adapter at {adapter}"
    if args.repo == "harpreet22happy/deep-research-agent-sft":
        raise SystemExit("refusing: that repo is the shipped 2026-08-26 adapter; use a new repo id.")

    acfg = json.loads((adapter / "adapter_config.json").read_text())
    sections = []
    for ev_path in args.eval:
        ev = json.loads(Path(ev_path).read_text())
        sections.append(f"### `{ev['split']}` — {ev['n']} questions, greedy, "
                        f"{'never trained or tuned on' if ev['split']=='heldout_eval' else 'the early-stop set'}\n\n"
                        + _table(ev))
    hist_md = ""
    if args.history:
        h = json.loads(Path(args.history).read_text())
        hist_md = "\n| epoch | train loss | val loss |\n|---|---|---|\n" + "\n".join(
            f"| {r['epoch']} | {r.get('train_loss', float('nan')):.4f} | {r.get('val_loss', float('nan')):.4f} |"
            for r in h["history"]) + "\n"
        n_ex = h.get("args", {})
    card = f"""---
license: apache-2.0
base_model: Qwen/Qwen2.5-3B-Instruct
library_name: peft
tags: [lora, sft, agent, multi-hop-qa, retrieval, citation, hotpotqa, 2wikimultihopqa]
---

# deep-research-agent — SFT (LoRA) on Qwen2.5-3B-Instruct — FULL teacher set

A LoRA adapter that teaches a 3B model to **verify its sources before citing them** in a
multi-hop research loop (`search` -> `read` -> `answer [Title] [Title]`). Same recipe as
[`harpreet22happy/deep-research-agent-sft`](https://huggingface.co/harpreet22happy/deep-research-agent-sft)
(which trained on the 418 strict-gate trajectories available at the time); this one trains
on **all 1,210** strict-gate (correct AND perfectly cited) trajectories from the complete
4,000-question `sft_collect` split, for 2 epochs instead of 3.

{args.note}

## Results

{chr(10).join(sections) if sections else '_(eval pending)_'}

Three arms: "base + worked example" is the same base model with a full worked trajectory
in its prompt — most of the raw *correctness* gain is available from prompting alone; the
adapter's distinctive contribution is read-before-cite and the correct-and-cited bucket.
Evaluated under the example-free prompt it was trained on.

## Training

- **Data:** 1,210 trajectories generated by `gpt-4.1-mini` driving the *real* environment
  (the teacher chose actions, the corpus answered). Kept only correct AND perfectly cited.
  Dataset: `harpreet22happy/deep-research-agent-trajectories`.
- **Masking:** loss on the model's own turns only; prompt and every tool response masked,
  verified per example against the tokenizer.
- **LoRA** r={acfg.get('r')}, alpha={acfg.get('lora_alpha')}, dropout={acfg.get('lora_dropout')} on
  {', '.join(sorted(acfg.get('target_modules', [])))}. lr 1e-4 cosine, batch 2 x grad-accum 8,
  token-level gradient accumulation.
{hist_md}
## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct", dtype="bfloat16")
model = PeftModel.from_pretrained(base, "{args.repo}")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
```
The prompt/tool format lives in `env._opening_prompt` (`include_worked_example=False`) of
the `Agentic-RL-Alignment-Path` repo, `assignments/deep_research_agent`.
"""
    (adapter / "README.md").write_text(card)
    api = HfApi()
    api.create_repo(repo_id=args.repo, private=True, exist_ok=True)
    api.upload_folder(folder_path=str(adapter), repo_id=args.repo, path_in_repo=".",
                      allow_patterns=["adapter_config.json", "adapter_model.safetensors", "README.md"])
    print(f"pushed {adapter} -> https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
