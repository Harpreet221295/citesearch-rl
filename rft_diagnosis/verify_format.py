"""Format & turn-boundary verification — run this BEFORE trusting any diagnosis number.

WHY THIS EXISTS (2026-08-26, Harpreet's call: "at least first confirm we are using the
right format for the model and turn boundaries in whatever we are trying to do"):

Re-reading `diagnosis1_native_results.json`'s raw rows turned up something the A/B table
in RFT_PLAN_AND_MODEL_DIAGNOSIS.md never surfaced, because the aggregate
`parse_ok_rate: 0.0` got read as "a bit noisy" rather than what it literally is:

    n_parse_failures distribution: {2:4, 3:5, 4:7, 5:8, 6:4, 7:7, 8:24, 9:5}
    rows with >=1 parse failure: 64/64      <- EVERY trajectory
    terminated_cleanly (valid `answer` call): 10/64
    made >=1 real tool call but NEVER answered: 49/64  (19 of them called `read`)

The loop runs `max_turns + 1 = 9` rounds, so a mode of 8 means nearly every round of
nearly every episode failed to parse. That is not a capability signal — it is a harness
signal. Any conclusion drawn from that run's `correct_rate` / `terminated_cleanly_rate`
is confounded until this script's checks pass.

THE LEADING HYPOTHESIS this script is built to confirm or kill:
In Qwen's NATIVE tool-calling convention the model signals "I'm done" by emitting an
ordinary plain-text assistant message — there is no `answer` tool. `diagnosis1_native_
tools.py` invents one and treats a plain-text turn as `error: no valid tool_call found`,
pushing an error message back and looping. If true, the model's CORRECT behaviour is
being scored as a parse failure, which would explain high engagement, `terminated_
cleanly=0.156`, and `correct_rate=0.0` simultaneously — one bug, three symptoms.

This script does NOT assume that. It measures, against the real installed tokenizer and
the real model:

  python verify_format.py template   # offline, no GPU: what the chat template actually
                                     #   renders from the message dicts we build
  python verify_format.py generate   # GPU: real generations, raw text, finish_reason,
                                     #   and a parse-failure CAUSE breakdown
  python verify_format.py all

Design rule followed throughout: print the real artifact (rendered string, raw
completion, finish_reason) rather than a summary of it. The bugs found on 2026-08-25
were all found by reading real transcripts, and all four aggregate-only readings of the
same data missed them.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import config as dr_config
import tools as dr_tools
from diagnosis1 import load_pool
from diagnosis1_native_tools import (
    TOOL_SCHEMAS, _INSTRUCTIONS, _STOP, _parse_tool_call, _TOOL_CALL_RE,
)

SEP = "=" * 78


# ---------------------------------------------------------------------------
# Part A — the chat template, offline. No GPU, no model weights.
# ---------------------------------------------------------------------------

def check_template(cfg) -> None:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.model_name)

    print(SEP)
    print("A1. Tokenizer identity & special tokens")
    print(SEP)
    print(f"  model_name      : {cfg.model_name}")
    print(f"  tokenizer class : {type(tok).__name__}")
    print(f"  eos_token       : {tok.eos_token!r} (id={tok.eos_token_id})")
    print(f"  pad_token       : {tok.pad_token!r} (id={tok.pad_token_id})")
    print(f"  has chat_template: {bool(getattr(tok, 'chat_template', None))}")

    tmpl = getattr(tok, "chat_template", "") or ""
    print()
    print(SEP)
    print("A2. Does the template actually SUPPORT `tools=`, and what keys does it read?")
    print(SEP)
    # Grep the real template source rather than trusting docs — the 2026-08-25 session
    # already got burned once by a web doc that described Qwen3's convention, not Qwen2.5's.
    for probe in ["tools", "tool_call", "tool_calls", "<tool_call>", "<tool_response>",
                  "arguments", "function", "'type'", '"type"', "tojson"]:
        print(f"  {probe!r:20} present in template: {probe in tmpl}")
    Path("/tmp/qwen_chat_template.jinja").write_text(tmpl)
    print("\n  full template written to /tmp/qwen_chat_template.jinja "
          f"({len(tmpl)} chars) — read it directly if anything below looks off.")

    print()
    print(SEP)
    print("A3. THE KEY CHECK — render the exact message dicts run_one_episode() builds")
    print(SEP)
    print("If our assistant-tool_call dict does not round-trip into a literal")
    print("<tool_call>{...}</tool_call> block, the model is being shown a conversation")
    print("in a shape it was never trained on, and every later number is confounded.\n")

    # This is byte-for-byte the shape diagnosis1_native_tools.run_one_episode() appends.
    ours = [
        {"role": "user", "content": _INSTRUCTIONS + "\n\nQuestion: WHO DIRECTED BLUE HARVEST?"},
        {"role": "assistant", "tool_calls": [
            {"function": {"name": "search", "arguments": {"query": "Blue Harvest director", "k": 3}}}]},
        {"role": "tool", "content": "[1] Blue Harvest (film) — a 2007 film directed by Jane Doe."},
    ]
    rendered_ours = tok.apply_chat_template(
        ours, tools=TOOL_SCHEMAS, add_generation_prompt=True, tokenize=False)

    # The canonical OpenAI-style shape most chat templates are written against:
    # `type: "function"` present, and `arguments` as a JSON STRING rather than a dict.
    canonical = [
        ours[0],
        {"role": "assistant", "tool_calls": [
            {"type": "function",
             "function": {"name": "search",
                          "arguments": json.dumps({"query": "Blue Harvest director", "k": 3})}}]},
        ours[2],
    ]
    try:
        rendered_canon = tok.apply_chat_template(
            canonical, tools=TOOL_SCHEMAS, add_generation_prompt=True, tokenize=False)
    except Exception as e:                                    # noqa: BLE001
        rendered_canon = f"<<render failed: {type(e).__name__}: {e}>>"

    print("--- OURS (what diagnosis1_native_tools.py actually sends) " + "-" * 20)
    print(rendered_ours)
    print("--- CANONICAL (type=function, arguments as JSON string) " + "-" * 21)
    print(rendered_canon)
    print("-" * 78)

    same = rendered_ours == rendered_canon
    print(f"\n  identical rendering: {same}")
    if not same:
        print("  !! They differ. Read both blocks above and decide which one matches the")
        print("     <tool_call> shape the template emits for the model. A mismatch here is")
        print("     a REAL bug, not a formatting nit — it is what the model conditions on.")

    print()
    print(SEP)
    print("A4. Turn boundaries — where does an assistant turn actually END?")
    print(SEP)
    gen_prefix = tok.apply_chat_template(
        [ours[0]], tools=TOOL_SCHEMAS, add_generation_prompt=True, tokenize=False)
    print(f"  generation prompt ends with: {gen_prefix[-60:]!r}")
    print(f"  configured stop sequences  : {_STOP}")
    print(f"  eos token                  : {tok.eos_token!r}")
    print()
    print("  What to look for: the model ends a turn with the eos token above. If our")
    print("  stop list does NOT include it and vLLM's own eos handling is the only thing")
    print("  ending generation, that is fine — but a completion that ends at a `stop`")
    print("  string instead means the turn was cut EARLY, and anything the model was")
    print("  about to emit (e.g. a closing </tool_call>) is lost. Part B measures which")
    print("  of these actually happens, per round, on real generations.")

    print()
    print(SEP)
    print("A5. Parser contract — what _parse_tool_call() will and won't accept")
    print(SEP)
    print(f"  regex: {_TOOL_CALL_RE.pattern!r}\n")
    cases = [
        ("well-formed",              '<tool_call>\n{"name": "search", "arguments": {"query": "x"}}\n</tool_call>'),
        ("plain prose (NATIVE finish!)", 'Based on the passages, the answer is Jane Doe.'),
        ("truncated before close",   '<tool_call>\n{"name": "search", "arguments": {"query": "x"}}'),
        ("prose THEN tool call",     'Let me look this up.\n<tool_call>{"name":"search","arguments":{"query":"x"}}</tool_call>'),
        ("arguments as JSON string", '<tool_call>{"name":"search","arguments":"{\\"query\\":\\"x\\"}"}</tool_call>'),
        ("unknown tool name",        '<tool_call>{"name":"finish","arguments":{"text":"Jane Doe"}}</tool_call>'),
    ]
    for label, text in cases:
        name, args, ok = _parse_tool_call(text)
        known = name in dr_tools.TOOLS if name else False
        verdict = "ACCEPTED" if (ok and known) else "PARSE FAILURE"
        print(f"  {label:28} -> {verdict:14} (name={name!r}, ok={ok}, known_tool={known})")
    print()
    print("  Note the 'plain prose' row specifically. Under the native convention that is")
    print("  how the model SIGNALS COMPLETION. If it reads PARSE FAILURE above, the harness")
    print("  is rejecting correct behaviour — see this file's header.")


# ---------------------------------------------------------------------------
# Part B — real generations. Captures raw text + finish_reason per round.
# ---------------------------------------------------------------------------

def _classify_failure(raw: str, finish_reason: str, stop_reason) -> str:
    """Why did this round fail to parse? A cause, not just a count."""
    if _TOOL_CALL_RE.search(raw or ""):
        name, args, ok = _parse_tool_call(raw)
        if not ok:
            return "malformed_json_in_tool_call"
        if name not in dr_tools.TOOLS:
            return f"unknown_tool:{name}"
        return "parsed_ok"
    if "<tool_call>" in (raw or ""):
        # Opened but never closed — almost always a truncation/stop-sequence artifact.
        return ("tool_call_truncated_by_length" if finish_reason == "length"
                else f"tool_call_unclosed(stop={stop_reason!r})")
    if finish_reason == "length":
        return "prose_truncated_by_length"
    return "plain_prose_no_tool_call"


def _banner(ch: str, label: str, width: int = 100) -> str:
    label = f" {label} "
    pad = max(0, width - len(label))
    return ch * (pad // 2) + label + ch * (pad - pad // 2)


def probe_generations(cfg, n_questions: int = 4, max_rounds: int | None = None,
                      n_transcripts: int = 2) -> None:
    """Replay the OLD harness's decision rules against real generations, BATCHED.

    The decisions here are deliberately the old ones (an `answer`-tool-only finish, the
    `<tool_call>`-containing retry message, first-call-only parsing) — that is the whole
    point: this reproduces the diagnosis. What is NOT the old behaviour is the batching.

    Rewritten 2026-08-26 after Harpreet flagged it: the first version generated one
    prompt at a time in a Python loop (4 questions x 9 rounds = 36 calls that should
    have been 9). It used vLLM, so it was not as slow as HF generate(), but batch-1 in a
    loop is exactly the pattern RUNPOD_PLAYBOOK.md pattern #3 exists to prevent, and
    leaving it in the repo invites someone to copy it. Every still-active episode now
    advances together in ONE llm.generate(prompts, ...) call per round, matching
    evaluate.run_batched_rollouts and native_rollout.run_batched_native_rollouts.
    Semantics are unchanged — only the number of engine calls is.

    `n_transcripts` questions also get a full human-readable transcript.
    """
    from transformers import AutoTokenizer
    from vllm import SamplingParams
    from evaluate import build_vllm_engine, close_vllm_engine

    max_rounds = max_rounds if max_rounds is not None else cfg.max_turns + 1
    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    tasks = load_pool(cfg, "train", n=n_questions, offset=0)
    tdir = _HERE / "rft_diagnosis" / "transcripts"
    tdir.mkdir(exist_ok=True)

    print(SEP)
    print(f"B. REAL generations — {n_questions} questions, up to {max_rounds} rounds each")
    print(f"   model={cfg.model_name}  temperature={cfg.temperature}  "
          f"max_new_tokens={cfg.max_new_tokens}")
    print(f"   BATCHED: one generate() per round over every still-active episode")
    print(SEP)
    print("Every round's RAW completion is printed verbatim. Read them — this is the")
    print("artifact that settles the format question, not the counts at the bottom.\n")

    sp = SamplingParams(temperature=cfg.temperature, top_p=cfg.top_p,
                        max_tokens=cfg.max_new_tokens, stop=_STOP)

    class _Ep:
        def __init__(self, i, task):
            self.i, self.task = i, task
            self.docstore = task.docstore()
            self.history = [{"role": "user",
                             "content": _INSTRUCTIONS + f"\n\nQuestion: {task.question}"}]
            self.done = False
            self.T = [] if i < n_transcripts else None
            if self.T is not None:
                self.T.append(_banner("=", "THE QUESTION"))
                self.T.append(f"task_id          : {task.task_id}")
                self.T.append(f"question         : {task.question}")
                self.T.append(f"gold answer      : {task.gold_answer!r}")
                self.T.append(f"supporting titles: {task.supporting_titles}")
                self.T.append("")
                self.T.append("HOW TO READ THIS FILE")
                self.T.append("  'WHAT WE SENT'    = the exact text handed to the model.")
                self.T.append("  'WHAT MODEL WROTE'= its reply, verbatim.")
                self.T.append("  'OUR VERDICT'     = what the OLD parsing code decided.")
                self.T.append("  'WHAT WE DID NEXT'= what we sent back as a result.")
                self.T.append(f"  Settings: temperature={cfg.temperature}, "
                              f"max_new_tokens={cfg.max_new_tokens}, stop={_STOP}")
                self.T.append("")

    causes: dict[str, int] = {}
    finish_reasons: dict[str, int] = {}
    eps = [_Ep(i, t) for i, t in enumerate(tasks)]
    llm = build_vllm_engine(cfg, enable_lora=False)
    try:
        for rnd in range(max_rounds):
            active = [e for e in eps if not e.done]
            if not active:
                break
            prompts = [tok.apply_chat_template(e.history, tools=TOOL_SCHEMAS,
                                               add_generation_prompt=True, tokenize=False)
                       for e in active]
            outs = llm.generate(prompts, sp, use_tqdm=False)
            for e, prompt, o in zip(active, prompts, outs):
                out = o.outputs[0]
                raw, fr, sr = out.text, out.finish_reason, out.stop_reason
                cause = _classify_failure(raw, fr, sr)
                causes[cause] = causes.get(cause, 0) + 1
                finish_reasons[fr] = finish_reasons.get(fr, 0) + 1

                print(f"\n--- [{e.task.task_id}] round {rnd} | finish_reason={fr!r} "
                      f"stop_reason={sr!r} | n_tokens={len(out.token_ids)} | -> {cause}")
                print(f"RAW: {raw!r}")

                name, args, ok = _parse_tool_call(raw)
                accepted = ok and name in dr_tools.TOOLS

                if e.T is not None:
                    e.T.append("")
                    e.T.append(_banner("#", f"ROUND {rnd}"))
                    e.T.append("")
                    e.T.append(_banner("-", "WHAT WE SENT (full prompt, verbatim)"))
                    e.T.append(prompt)
                    e.T.append(_banner("-", "WHAT MODEL WROTE BACK (verbatim)"))
                    e.T.append(raw)
                    e.T.append(_banner("-", "OUR VERDICT"))
                    e.T.append(f"  stopped because  : finish_reason={fr!r}  stop_reason={sr!r}")
                    e.T.append(f"  tokens generated : {len(out.token_ids)} "
                               f"(cap is {cfg.max_new_tokens})")
                    e.T.append(f"  our parser says  : {'ACCEPTED' if accepted else 'REJECTED'}")
                    e.T.append(f"  reason bucket    : {cause}")

                if not accepted:
                    err_msg = ("error: no valid <tool_call> found. Call exactly one "
                               "of search, read, or answer.")
                    if e.T is not None:
                        e.T.append(_banner("-", "WHAT WE DID NEXT"))
                        e.T.append(f"  Sent this back as a new user message: {err_msg!r}")
                        e.T.append("  (the model must now try again; this burns one of "
                                   f"its {max_rounds} rounds)")
                    e.history.append({"role": "assistant", "content": raw})
                    e.history.append({"role": "user", "content": err_msg})
                    continue

                if name == "answer":
                    print(f"    => ANSWER tool called, text={args.get('text')!r}")
                    if e.T is not None:
                        e.T.append(_banner("-", "WHAT WE DID NEXT"))
                        e.T.append("  Model called the `answer` tool -> episode ENDS HERE.")
                        e.T.append(f"  final answer text: {args.get('text')!r}")
                        e.T.append(f"  gold answer      : {e.task.gold_answer!r}")
                    e.done = True
                    continue

                if name == "search" and "k" not in args:
                    args = {**args, "k": cfg.search_k}
                obs, tok_ok, err, titles = dr_tools.execute(e.docstore, name, args)
                obs = obs[: cfg.max_obs_chars]
                print(f"    => {name}({args}) ok={tok_ok} titles={titles}")
                if e.T is not None:
                    e.T.append(_banner("-", "WHAT WE DID NEXT"))
                    e.T.append(f"  Ran the tool: {name}({args})  ok={tok_ok} err={err}")
                    e.T.append(f"  titles returned: {titles}")
                    e.T.append(_banner("-", "TOOL RESULT WE SENT BACK"))
                    e.T.append(obs)
                e.history.append({"role": "assistant", "tool_calls": [
                    {"function": {"name": name, "arguments": args}}]})
                e.history.append({"role": "tool", "content": obs})
    finally:
        close_vllm_engine(llm)

    for e in eps:
        if e.T is not None:
            e.T.append("")
            e.T.append(_banner("=", "END OF EPISODE"))
            path = tdir / f"transcript_{e.task.task_id}.txt"
            path.write_text("\n".join(e.T))
            print(f"\n  >>> human-readable transcript written to {path}")

    print("\n" + SEP)
    print("B. Round-outcome breakdown (the diagnosis)")
    print(SEP)
    total = sum(causes.values())
    for cause, cnt in sorted(causes.items(), key=lambda kv: -kv[1]):
        print(f"  {cause:34} {cnt:4}  ({cnt/total:.1%})")
    print(f"\n  finish_reasons: {finish_reasons}")
    print(f"  total rounds  : {total}")

    out_path = _HERE / "rft_diagnosis" / "verify_format_results.json"
    out_path.write_text(json.dumps(
        {"causes": causes, "finish_reasons": finish_reasons, "total_rounds": total,
         "model": cfg.model_name, "temperature": cfg.temperature,
         "max_new_tokens": cfg.max_new_tokens, "stop": _STOP}, indent=2))
    print(f"\n  wrote {out_path}")



def main() -> None:
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    from dotenv import load_dotenv          # HF_TOKEN: higher Hub rate limits / faster pulls
    load_dotenv(_HERE / ".env")
    cfg = dr_config.Config.cloud_preset()
    if what in ("template", "all"):
        check_template(cfg)
    if what in ("generate", "all"):
        print("\n\n")
        probe_generations(cfg)
    if what not in ("template", "generate", "all"):
        print(__doc__)
        sys.exit(2)


if __name__ == "__main__":
    main()
