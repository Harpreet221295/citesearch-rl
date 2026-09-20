# Pod lifecycle & persistence — deep_research_agent (this specific pod)

**Date:** 2026-08-24. **For:** the next Claude Code session (or Harpreet) picking this pod
back up after a stop/start. **Purpose:** this pod is being kept alive across multiple
sessions over several days (stop between sessions, start again to resume — NOT
terminate/recreate) rather than torn down each time. This doc is the "what survived, what
didn't, how do I resume" reference so that isn't re-derived from scratch, or worse,
silently assumed and gotten wrong.

If this pod has already been terminated and a fresh one exists instead, most of this doc
is historical — skip to `HANDOFF.md` and `RUNPOD_PLAYBOOK.md`'s general storage-strategy
section instead.

---

## 1. This pod, concretely

- **Pod ID:** `m4j1fsfj84767c` (RunPod, **Secure Cloud** tier — matters, see §4).
- **GPU:** 1× A100 SXM4 80GB. **vCPU:** 32 (AMD EPYC 7742). **RAM:** 251GB.
- **Template:** `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu...`.
- **Storage as provisioned:** 60GB Container Disk + 60GB Volume Disk (`/workspace`).
  **No Network Volume was selected at deploy time** — this matters, see §2.
- **Pricing:** $1.59/hr compute + $0.008/hr container storage (running) + $0.008/hr volume
  storage (running) ≈ **$1.61/hr running**. While **stopped**: compute → $0, container
  storage → $0 (wiped anyway), volume storage **doubles** to ~$0.016/hr (RunPod bills idle
  Volume Disk at $0.20/GB/month vs $0.10/GB/month while running) ≈ **~$0.016/hr stopped**
  (~$0.38/day) for the 60GB volume.

## 2. What survives what — verified against RunPod's own docs, not assumed

Three separate storage classes on this pod, each with a different lifetime (confirmed via
[docs.runpod.io/storage/network-volumes](https://docs.runpod.io/storage/network-volumes)
and [docs.runpod.io/pods/pricing](https://docs.runpod.io/pods/pricing) — a first-pass
assumption that `/workspace` alone was enough for full termination-survival was WRONG and
corrected mid-session; don't repeat that mistake):

| storage | mount | survives STOP? | survives TERMINATE? |
|---|---|---|---|
| Container Disk | `/` (`overlay`) | **NO** — wiped | NO |
| Volume Disk | `/workspace` (`/dev/md1`, XFS) | **YES** | **NO** |
| Network Volume | (not attached to this pod) | YES | YES |

**This pod has no Network Volume.** That means: STOP is safe for pausing between sessions
(the whole point of this doc's plan) — TERMINATE is not, and would lose everything not
already on `/workspace` or pushed to git/HF. A Network Volume can ONLY be attached at pod
*deployment* time, per RunPod's docs ("cannot be attached or detached later without
deleting the Pod") — so if true survive-a-full-delete persistence is ever wanted, that
requires deploying a brand new pod with one selected, not something addable to this pod.

## 3. What actually happened (2026-08-24) — the migration

The repo (and its ~13GB built venv, `.venv-deep-research`) originally lived entirely on
Container Disk (`/root/Agentic-RL-Alignment-Path`) — meaning it would have been wiped on
the very next STOP, not just a terminate. Migrated to `/workspace`:

1. Copied the whole repo (`cp -a`, git history + `.env` included — `.env` is gitignored so
   a plain `git clone` at the new location would NOT have carried it) to
   `/workspace/Agentic-RL-Alignment-Path`, excluding the old venv (can't be moved — venvs
   bake absolute paths into `pyvenv.cfg` and every `bin/*` shebang; verified this
   concretely — `bin/pip`'s shebang was `#!/root/.../​.venv-deep-research/bin/python` — before
   deciding to rebuild rather than hack the paths).
2. Ran `setup_pod.sh` fresh at the new location. Found and fixed two real gaps in the
   script itself while doing this (both `uv` and `liger-kernel` had been installed ad hoc
   during the original session and were never actually captured in the script — a truly
   fresh run died immediately at `uv: command not found` under `set -euo pipefail`). Both
   fixed and committed — see `setup_pod.sh`'s own comments for the detail.
3. `uv`'s package cache (`/root/.cache/uv`, itself on Container Disk — survives only until
   the next stop, but was present for THIS rebuild) let the rebuild reuse cached wheels
   instead of recompiling `flash-attn` from source again — much faster than the original
   install.
4. Verified end-to-end from the new location before trusting it: all 36 tests pass,
   `torch`/`vllm`/`verl`/`rllm`/`liger_kernel` import cleanly, the `LD_LIBRARY_PATH`
   fix for `libcudart.so.13` is intact (baked into `.venv-deep-research/bin/activate`),
   `.env` secrets load, `python train_dr.py --dry-run cloud` resolves identically to the
   old location, and a real `torch.randn(...).cuda()` matmul succeeds.

**Old `/root/Agentic-RL-Alignment-Path` copy:** harmless to ignore, will be wiped on the
next stop regardless. Nothing of value lives there that isn't also in `/workspace` or git.

## 4. How to resume after a stop (the actual playbook)

**SSH reconnection first** (RunPod dashboard → Pod → Connect tab has two options,
confirmed to behave differently across a stop/start):
- `ssh root@<ip> -p <port> -i ~/.ssh/id_ed25519` ("SSH over exposed TCP", direct) — **this
  is what Harpreet's Claude Code session used to connect this pod (2026-08-24).** The
  port comes from `RUNPOD_TCP_PORT_22`, assigned fresh per container start — **the IP
  and/or port are expected to change after a stop/start.** After restarting, re-open the
  Connect tab, copy the new `ssh root@... -p ...` line, and update wherever Claude Code's
  remote connection is configured (on the local machine, not visible from inside the pod)
  before it can reconnect.
- `ssh <podid>-<hash>@ssh.runpod.io -i ~/.ssh/id_ed25519` ("SSH", proxied) — hostname is
  **pod-ID-based**, not IP-based, so this one should keep working unchanged across a
  stop/start. Not what was actually used this session, but worth switching to if Claude
  Code's remote setup supports a hostname target — would remove the manual
  reconnect-details update every single time.

```bash
# Once connected:
cd /workspace/Agentic-RL-Alignment-Path/assignments/deep_research_agent
source .venv-deep-research/bin/activate
# Confirm it actually came back clean before trusting it (cheap, ~10s):
python -m pytest -q tests/
python -c "import torch; print(torch.cuda.is_available())"
```
No reinstall needed — Container Disk resets to the base template image (same Python, same
CUDA toolkit/driver — those come from the image itself, not something we installed), and
`/workspace` comes back exactly as left. **What does NOT survive a stop and would need
redoing** (all cheap, system-level, outside `/workspace`): `tmux` (`apt-get install -y
tmux` after an `apt-get update` — the package index cache is also wiped), anything else
installed via `apt-get` this session.

**If the real cloud training run was mid-flight when stopped:** `trainer.resume_mode` maps
to `cfg.resume` (`train_dr.py::verl_overrides`) — set `cfg.resume=True` and veRL's
`resume_mode=auto` should pick up the latest local checkpoint under
`runs/deep_research_agent_cloud/` (which lives on `/workspace` now, so it survives the
stop). Not yet tested in practice as of this doc — verify the resume actually works
(real checkpoint found, training continues from the right step) before assuming it does,
same "measure don't guess" discipline as everything else in this lab.

## 5. The plan (as of 2026-08-24)

Keep this pod alive across multiple sessions over the next few days via **stop/start**,
not terminate/recreate, to finish this capstone. Rough cost shape: ~$1.61/hr while
actively working or training, ~$0.016/hr while stopped between sessions — cheap enough
that stopping between every session (rather than leaving it running idle) is clearly
worth it. Terminate only once the capstone is genuinely done and nothing more will be
pulled from this pod (checkpoints are pushed to HF regardless, so terminating doesn't
lose the trained model itself — only the pod's local dev environment).
