# Serving a trained checkpoint for inference — vLLM / SGLang

How to take an adapter this lab pushed to HF Hub (via `hub.py` / `train_dr.py
push_checkpoints`, see HANDOFF.md step 6) and stand it up as a real inference
endpoint later — not just score it inside `evaluate.py`'s own batch harness.

**Verified vs unverified, honestly:** the vLLM CLI flags below were confirmed against
the actually-installed `vllm==0.22.1` on this pod (`vllm serve --help=LoRAConfig`,
`--help=Frontend` — see [RLLM_VERL_INSTALL_NOTES.md](RLLM_VERL_INSTALL_NOTES.md) for
why "verify against the real installed version, not docs" is this lab's hard-won
rule). **Not** verified: an actual end-to-end served request against a real trained
adapter (no trained adapter exists yet — training is still blocked on the Workflow
port, HANDOFF.md step 2). Re-verify the flags again before relying on this once a
newer vllm is installed; this stack moves fast.

---

## 1. What's on HF Hub, and where

`hub.py`'s push layout (`RUNPOD_PLAYBOOK.md` pattern #5b — LATEST and BEST are
separate artifacts, never conflate them):

```
<push_to_hub_repo>                          (private HF model repo)
  └─ branch: <run_name>__<DD_MM_YYYY__HH_MM_SS>   (one branch per run, unique)
       ├─ checkpoint/          ← LATEST step's adapter (overwritten each push)
       └─ best/step<N>_em<S>/  ← each kept-best step's adapter (top keep_best_k)
```

Each of those directories is a **standard PEFT LoRA adapter** (`adapter_config.json`
+ `adapter_model.safetensors`) — verl's own `model_merger` reconstructed it from the
raw FSDP checkpoint (see `hub.merge_checkpoint`), so it loads exactly like any HF LoRA
adapter: `PeftModel.from_pretrained`, vLLM's `LoRARequest`, or `vllm serve --lora-modules`.

Find the branch name for a run: `runs/<run_name>/run_meta.json` (`{"run_branch": ...}`),
or check `runs/<run_name>/push_report.json` for the exact best-step scores after a push.

## 2. Pull the adapter down locally

`vllm serve`'s `--lora-modules` wants a **local path** (not a bare repo id) —
download the specific branch + subfolder first:

```python
import hub   # this lab's hub.py
adapter_dir = hub.pull_folder_from_hub(
    repo_id="<you>/<repo>", branch="<run_name>__<timestamp>",
    path_in_repo="best/step200_em0.340",   # or "checkpoint" for latest
    local_dir="local_adapters/deep_research",
)
print(adapter_dir)   # -> local_adapters/deep_research/best/step200_em0.340
```
Or directly with `huggingface_hub` if you don't want the `hub.py` import:
```bash
huggingface-cli download <you>/<repo> --revision "<run_name>__<timestamp>" \
    --include "best/step200_em0.340/*" --local-dir local_adapters/deep_research
```

## 3. Serve it — vLLM (this lab's rollout engine; verified flags exist)

```bash
vllm serve Qwen/Qwen2.5-3B-Instruct \
    --enable-lora \
    --max-lora-rank 16 \
    --lora-modules research=local_adapters/deep_research/best/step200_em0.340 \
    --port 8000
```
- `--enable-lora` / `--lora-modules` / `--max-lora-rank` all confirmed present via
  `vllm serve --help=LoRAConfig` / `--help=Frontend` on the installed `vllm==0.22.1`.
  `--max-lora-rank` must be `>=` `cfg.lora_r` (16 by default — see `config.py`).
- `--lora-modules NAME=PATH` is vLLM's long-standing syntax; newer versions also
  accept a JSON form (`--lora-modules '{"name":"research","path":"...",
  "base_model_name":"..."}'`) per the CLI's generic `--json-arg` convention shown in
  its own `--help` output — check `vllm serve --help=lora-modules` on your installed
  version if the plain form is rejected.
- Request the adapter by name in the `model` field of an OpenAI-style request:
  ```bash
  curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
    "model": "research",
    "messages": [{"role": "user", "content": "..."}],
    "temperature": 0.0, "max_tokens": 256
  }'
  ```
  (omit `model`/use the base model name to hit the BASE model, unaltered, on the
  same server — useful for a quick base-vs-tuned A/B without two servers.)

## 4. Serve it — SGLang (alternative; **not** installed/verified in this repo)

This lab installed **vLLM** as veRL's rollout engine (HANDOFF.md step 1 —
`verl_rollout_engine="vllm"`), so SGLang was never pulled in and this is
undocumented-by-experience here. The documented SGLang pattern (verify against
whatever version you actually install, same discipline as above):
```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-3B-Instruct \
    --lora-paths research=local_adapters/deep_research/best/step200_em0.340 \
    --port 30000
```
SGLang also exposes an OpenAI-compatible `/v1/chat/completions` endpoint, so
everything in §5/§6 below applies the same way regardless of which server you pick.

## 5. The part that's easy to miss: a bare chat endpoint isn't the AGENT

`vllm serve`/SGLang only give you next-token completion over an OpenAI-style API.
This lab's agent is **multi-turn ReAct** (search → read → answer, `env.py`'s
`DeepResearchEnv`) — something still has to run the LOOP: send the opening prompt,
parse the model's `Thought:/Action:` turn, execute the tool, feed the observation
back, repeat until `answer[...]`. A served endpoint alone does none of that.

Two ways to drive it, both already exist in this repo in some form:
- **Reuse this lab's own loop.** `evaluate.py`'s `run_batched_rollouts` already
  implements exactly this stepping logic (build env → reset → loop:
  generate → `env.step()` → repeat) — currently pointed at vLLM's **offline** `LLM`
  class for batch scoring. Swapping the one `llm.generate(...)` call for an HTTP
  call to the served endpoint (`openai.chat.completions.create(model="research", ...)`)
  is a small, mechanical change if you want a live/interactive server instead of an
  offline batch job — the env/reward/tool code underneath doesn't change at all.
- **rLLM's own provision** (confirmed present in the installed `rllm@9beb6e0`):
  `rllm/engine/rollout/openai_engine.py` — an engine that drives rLLM's
  Agent/Workflow machinery against **any** OpenAI-compatible endpoint (exactly what
  `vllm serve`/SGLang expose). Once `DeepResearchEnv` is ported to a `Workflow`
  (HANDOFF.md step 2 — still open), the SAME Workflow class could run under veRL
  for training OR under `OpenAIEngine` against a served vLLM/SGLang endpoint for
  pure inference, with no duplicated agent logic. Worth revisiting once that port
  lands rather than hand-rolling a second loop.

## 6. One more gap: retrieval doesn't generalize past the offline corpus yet

`DeepResearchEnv`'s `search`/`read` tools query a `DocStore` built from
`task.docstore()` — Branch B's bundled **per-question** ~10-paragraph corpus
(`corpus_backend="bundled"`, see README.md "the offline corpus trick"). That's fine
for scoring HotpotQA/2Wiki questions (each ships its own passage set), but a real
"ask it anything" inference server has no such fixed per-question set for an
arbitrary live question. Before serving this agent on open queries, `corpus_backend`
needs to point at a real retriever (`"wiki_index"` — flagged in `config.py` as the
paper-comparable / stretch option, not yet built). Don't let a served endpoint imply
this already works for open-domain questions — it doesn't yet.

## Quick reference — full pull + serve, one shot

```bash
python -c "
import hub
hub.pull_folder_from_hub('<you>/<repo>', '<run_name>__<timestamp>',
                          'best/step200_em0.340', 'local_adapters/deep_research')
"
vllm serve Qwen/Qwen2.5-3B-Instruct --enable-lora --max-lora-rank 16 \
    --lora-modules research=local_adapters/deep_research/best/step200_em0.340
```
