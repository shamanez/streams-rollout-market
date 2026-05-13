# Iter 06 (cycle FP8) — Megatron MoE teacher-force × {bf16, fp8} rollouts

## Directive

Run `megatron_moe_reference_launch.sh` twice on the spot — once per
rollout precision — producing two router traces. Trainer always
bf16 (only the rollout side varies).

## What I did (after a long detour through docker debugging)

The launcher previously hung at the docker-run step for several
unrelated reasons that I had to fix before the actual teacher-force
could run:

1. **HF snapshot symlinks broke the bind-mount** (carried over from
   iter 02 of the prior cycle). Already fixed via `cp -rL` in the
   prior session.
2. **Multi-rollout `rollouts.json` (list) was being copied to
   `rollout.json` (single-dict path)** — patched the launcher to
   detect list vs dict and copy under the right filename
   (`run_megatron_moe_reference.py` reads `rollouts.json` first,
   `rollout.json` as fallback).
3. **Repeated `cp -rL` of the 57 GB HF model dir filled `/tmp` with
   8 leftover dereferenced copies** (~460 GB) and starved disk I/O,
   making subsequent docker run startup unreliable. Added
   `HF_FLAT_REUSE=$DIR` env var so a one-shot dereferenced copy can
   be reused across N back-to-back launches; pinned a copy at
   `~/hf-flat-qwen3-30b-a3b-bf16/`.
4. **`--ipc=host` triggered docker0 bridge teardown loops** (visible
   in dmesg as `vethXXX entered disabled state` immediately after
   creation). Confirmed via bisection: docker run with `--shm-size=64g`
   alone works; adding `--ipc=host` makes containers stay in
   `Created` state indefinitely. Dropped `--ipc=host` from the
   launcher and added `--network=host` (defensive — bypasses the
   broken docker0 bridge regardless). torchrun's 4 local ranks need
   shared memory which `--shm-size=64g` provides cleanly.
5. **Post-docker cp step only mirrored single-mode output paths.**
   Added handling for the multi-mode `trainers_megatron-bf16.json`
   (plural — the full list) plus an `OUT_SUFFIX=-<tag>_rollouts`
   env var so two consecutive launches don't trample each other.

Once all five fixes were in, both Megatron passes ran cleanly:

```
tmux new-session -d -s megatron5 "
  HF_FLAT_REUSE=~/hf-flat-qwen3-30b-a3b-bf16 ROLLOUT_FILE_HOST=/tmp/rollouts_bf16.json OUT_SUFFIX=-bf16_rollouts bash ~/.../megatron_moe_reference_launch.sh > /tmp/megatron_moe_bf16_v4.log 2>&1
  HF_FLAT_REUSE=~/hf-flat-qwen3-30b-a3b-bf16 ROLLOUT_FILE_HOST=/tmp/rollouts_fp8.json  OUT_SUFFIX=-fp8_rollouts  bash ~/.../megatron_moe_reference_launch.sh > /tmp/megatron_moe_fp8_v4.log  2>&1
  touch /tmp/megatron_both_done.marker
"
```

Total wall: 5 min 12 s for BOTH passes (after HF reuse). Without the
reuse, ~10 min was being spent on the cp -rL alone.

## Acceptance evidence

```text
$ ls -la /tmp/megatron_router-*rollouts.json /tmp/trainers_megatron-bf16-*rollouts.json
-rw-r--r-- 1 ubuntu ubuntu 1276128 May 13 03:52 /tmp/megatron_router-bf16_rollouts.json
-rw-r--r-- 1 ubuntu ubuntu 1234302 May 13 03:54 /tmp/megatron_router-fp8_rollouts.json
-rw-r--r-- 1 ubuntu ubuntu   13374 May 13 03:52 /tmp/trainers_megatron-bf16-bf16_rollouts.json
-rw-r--r-- 1 ubuntu ubuntu   12876 May 13 03:54 /tmp/trainers_megatron-bf16-fp8_rollouts.json

$ python3 -c "import json; r = json.load(open('/tmp/megatron_router-bf16_rollouts.json')); ..."
=== megatron_router-bf16: 19 entries ===
  expert_ids shape per entry: (24, 48, 8)   # [gen_tokens, 48, 8] — already gen-only
=== megatron_router-fp8: 16 entries ===
  expert_ids shape per entry: (24, 48, 8)
```

Megatron writes the router trace **already trimmed to the gen
window** (unlike FSDP which dumps the full prompt+gen). That makes
the pair step easier: no further slicing needed on the Megatron side.

## Notes on what NOT to fix

The lingering "[Gloo] Rank 0 is connected to 0 peer ranks" warnings
in the megatron logs are cosmetic — they come from slime's
`init_process_group` falling back to single-host mode and emit one
line per call but don't affect correctness. Ignored.

The `--network=host` change is a small loss of isolation but the
spot is single-tenant and the container has no inbound network
needs; torchrun uses local-only rdzv. Acceptable.
