"""LLM-as-Judge backend — a soft, model-based **groundedness** score in [0, 1],
separate from the outcome (EM/F1) correctness check.

WHY separate from outcome correctness: outcome correctness (metrics.exact_match)
asks "does the final answer match gold?" — ungameable but binary, and it says
nothing about WHETHER the answer was actually supported by what the agent
retrieved. A model can guess the right answer without reading the evidence
(retrieval hit-rate = 0, EM = 1) — fine for a quiz, fatal for a *research* agent
you want to trust. The judge scores the softer question: **is the final answer
grounded in the retrieved passages?** It is also HACKABLE — a fluent, confident,
citation-shaped answer can fool it. That tension is the whole point: evaluate.py
probes the judge-vs-outcome and judge-vs-token-overlap GAPs to catch judge-gaming
(verbose answers, fake citations). reward_deep_research (TODO #1) decides how much
to trust the judge (w_judge) — start it small, lean on the ungameable signals.

Backends (mirror finqa_agent/judge.py):
  * "mock" — deterministic, offline, INTENTIONALLY shallow/gameable heuristic.
    Local proof-of-life with no second model. Lets the hacking probe actually fire.
  * "vllm" — a real instruct model scores groundedness against a rubric, via vLLM's
    batched offline `LLM.generate()` (RunPod; PREFERRED backend for any real eval —
    project convention is no HF `model.generate()` for eval/inference, see
    RUNPOD_PLAYBOOK.md pattern #3c and evaluate.py's `run_batched_rollouts`).
  * "hf"   — the same rubric via HF `model.generate()` (batched, but still the
    slower path). Kept only for parity/offline-test purposes; prefer "vllm" on the pod.
  * "api"  — hook for a hosted judge (stub; wire your provider — see claude-api).

Judge plumbing is scaffold. reward.py CALLS the judge; how the score folds into the
reward is Harpreet's.
"""
from __future__ import annotations

import re

from metrics import answer_recall_in_context


def _final_answer_text(traj) -> str:
    return (traj.final_answer or "").strip()


def _retrieved_text(traj) -> str:
    """Concatenated text of every passage the agent actually read/surfaced — the
    evidence the answer is supposed to be grounded in. These are ENV tokens (the
    observations), never the model's own thoughts."""
    return "\n".join(s.observation for s in traj.steps
                     if s.call is not None and s.call.name in ("search", "read") and s.ok)


class Judge:
    """score(task, traj) -> float in [0, 1]. Higher = answer better grounded in
    the retrieved passages."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.backend = cfg.judge_backend
        self._hf = None
        self._vllm = None
        self._vllm_tok = None

    def close(self, timeout_s: float = 90.0):
        """Free the resident judge engine's GPU memory. vLLM V1 runs the engine in a
        SEPARATE SUBPROCESS — `del` + gc alone does NOT synchronously free it (the
        Python `LLM` object's `__del__` doesn't call the engine-core shutdown at all,
        verified against installed vllm==0.22.1 source; see evaluate.close_vllm_engine
        for the empirical finding). Explicitly signal shutdown, then POLL GPU memory
        until it's actually back before returning, so a caller loading a DIFFERENT
        engine right after `close()` doesn't race the subprocess's teardown."""
        if self._vllm is not None:
            import gc
            import time
            import torch
            try:
                self._vllm.llm_engine.engine_core.shutdown()
            except Exception:
                pass
            del self._vllm
            self._vllm = None
            gc.collect()
            torch.cuda.empty_cache()
            total = torch.cuda.mem_get_info()[1]
            deadline = time.time() + timeout_s
            while time.time() < deadline:
                free, _ = torch.cuda.mem_get_info()
                if free / total > 0.85:
                    return
                time.sleep(2.0)

    def score(self, task, traj) -> float:
        return self.score_batch([task], [traj])[0]

    def score_batch(self, tasks, trajs) -> list[float]:
        """Batched scoring — one generate() call over many trajectories instead of
        one call each (the finqa throughput lesson: don't leave the judge unbatched
        while rollouts are batched). Same rubric/parsing/[0,1] across backends."""
        if not trajs:
            return []
        if self.backend == "mock":
            return [_mock_score(t, tr) for t, tr in zip(tasks, trajs)]
        if self.backend == "vllm":
            return self._vllm_score_batch(tasks, trajs)
        if self.backend == "hf":
            return self._hf_score_batch(tasks, trajs)
        if self.backend == "api":
            raise NotImplementedError("api judge backend not wired.")
        raise ValueError(f"unknown judge_backend: {self.backend}")

    # --- real judge via vLLM (RunPod; PREFERRED — see module docstring) --
    def _ensure_vllm(self):
        """Lazily builds ONE vLLM engine for the judge model, reused across every
        score_batch call. Caller (evaluate.py) is responsible for freeing the
        POLICY engine before the judge phase starts if both won't fit resident —
        see evaluate.py's `close_vllm_engine`."""
        if self._vllm is None:
            from vllm import LLM
            from transformers import AutoTokenizer
            self._vllm = LLM(
                model=self.cfg.judge_model,
                dtype="bfloat16" if getattr(self.cfg, "bf16", True) else "auto",
                gpu_memory_utilization=getattr(self.cfg, "verl_gpu_mem_util", 0.6),
                trust_remote_code=True,
                seed=getattr(self.cfg, "eval_seed", 0),
            )
            self._vllm_tok = AutoTokenizer.from_pretrained(self.cfg.judge_model)
        return self._vllm

    def _vllm_score_batch(self, tasks, trajs) -> list[float]:
        from vllm import SamplingParams
        llm = self._ensure_vllm()
        prompts = [
            self._vllm_tok.apply_chat_template(
                [{"role": "user", "content": _rubric_prompt(t, tr)}],
                add_generation_prompt=True, tokenize=False)
            for t, tr in zip(tasks, trajs)
        ]
        params = SamplingParams(temperature=0.0, max_tokens=8)
        outs = llm.generate(prompts, params, use_tqdm=False)
        return [_parse_0_10(o.outputs[0].text) / 10.0 for o in outs]

    # --- real HF judge (RunPod) ------------------------------------------
    def _ensure_hf(self):
        if self._hf is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            tok = AutoTokenizer.from_pretrained(self.cfg.judge_model)
            mdl = AutoModelForCausalLM.from_pretrained(
                self.cfg.judge_model,
                dtype=torch.bfloat16 if self.cfg.bf16 else torch.float32,
            ).to(self.cfg.device)
            mdl.eval()
            self._hf = (mdl, tok)
        return self._hf

    def _hf_score(self, task, traj) -> float:
        import torch
        mdl, tok = self._ensure_hf()
        msgs = [{"role": "user", "content": _rubric_prompt(task, traj)}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
        ids = (ids["input_ids"] if hasattr(ids, "keys") else ids).to(mdl.device)
        with torch.no_grad():
            out = mdl.generate(ids, max_new_tokens=8, do_sample=False,
                               pad_token_id=tok.pad_token_id or tok.eos_token_id)
        text = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        return _parse_0_10(text) / 10.0

    def _hf_score_batch(self, tasks, trajs) -> list[float]:
        import torch
        mdl, tok = self._ensure_hf()
        bs = max(1, getattr(self.cfg, "gen_batch_size", 16))
        scores: list[float] = []
        prev = tok.padding_side
        tok.padding_side = "left"
        try:
            for start in range(0, len(trajs), bs):
                t_chunk, tr_chunk = tasks[start:start + bs], trajs[start:start + bs]
                prompts = []
                for task, traj in zip(t_chunk, tr_chunk):
                    msgs = [{"role": "user", "content": _rubric_prompt(task, traj)}]
                    templated = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
                    prompts.append(templated["input_ids"] if hasattr(templated, "keys") else templated)
                pad_id = tok.pad_token_id or tok.eos_token_id
                maxlen = max(len(p) for p in prompts)
                input_ids = torch.full((len(prompts), maxlen), pad_id, dtype=torch.long)
                attn = torch.zeros((len(prompts), maxlen), dtype=torch.long)
                for r, p in enumerate(prompts):
                    input_ids[r, maxlen - len(p):] = torch.tensor(p, dtype=torch.long)
                    attn[r, maxlen - len(p):] = 1
                with torch.no_grad():
                    gen = mdl.generate(input_ids=input_ids.to(mdl.device), attention_mask=attn.to(mdl.device),
                                       max_new_tokens=8, do_sample=False, pad_token_id=pad_id)
                for row in gen[:, maxlen:]:
                    scores.append(_parse_0_10(tok.decode(row.tolist(), skip_special_tokens=True)) / 10.0)
        finally:
            tok.padding_side = prev
        return scores


def _rubric_prompt(task, traj) -> str:
    return (
        "You are grading a research agent's answer for GROUNDEDNESS. Given the "
        "retrieved passages, score 0-10 how well the FINAL ANSWER is SUPPORTED BY "
        "THE PASSAGES (not whether it sounds good). An answer stating facts not in "
        "the passages should score low. Reply with ONLY the number.\n\n"
        f"Question: {task.question}\n"
        f"Retrieved passages:\n{_retrieved_text(traj)[:3000]}\n"
        f"Final answer: {_final_answer_text(traj)}\n\nScore (0-10):"
    )


def _parse_0_10(text: str) -> float:
    m = re.search(r"\d+(?:\.\d+)?", text)
    if not m:
        return 0.0
    return max(0.0, min(10.0, float(m.group())))


# ---------------------------------------------------------------------------
# Mock judge — deterministic, offline, INTENTIONALLY shallow/gameable. It rewards
# surface features of a "grounded-looking" answer (produced an answer, read a
# passage, wrote enough words) WITHOUT verifying the claim is actually supported —
# so a fluent, evidence-shaped-but-wrong trajectory can score high. Exactly the
# failure the eval's judge-vs-overlap gap is meant to expose.
# ---------------------------------------------------------------------------
def _mock_score(task, traj) -> float:
    ans = _final_answer_text(traj)
    read_any = any(s.call is not None and s.call.name in ("search", "read") and s.ok
                   for s in traj.steps)
    s = 0.0
    if traj.done and ans:
        s += 0.4                                    # produced a final answer at all
    if read_any:
        s += 0.3                                    # actually retrieved something
    if len(ans.split()) >= 1:
        s += 0.3                                    # non-empty answer (length, not truth!)
    return min(1.0, s)


def make_judge(cfg) -> Judge:
    return Judge(cfg)
