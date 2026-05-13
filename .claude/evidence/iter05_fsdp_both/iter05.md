# Iter 05 (cycle FP8) — FSDP MoE teacher-force × {bf16, fp8} rollouts

## Directive

Push both `/tmp/rollouts_bf16.json` and `/tmp/rollouts_fp8.json`
through `scripts/live/run_fsdp_moe_reference.py`. The trainer always
runs in **bf16** (FSDP weights loaded from the bf16 HF dir); only
the rollout side varies.

## What I did

Used `~/rmenv-dev/` (the only venv on this spot — `~/rmenv/` from
the previous bootstrap is gone). Per-precision pass:

```
cp /tmp/rollouts_<prec>.json /tmp/rollouts.json   # ROLLOUTS_PATH is hardcoded
MODEL=Qwen/Qwen3-30B-A3B \                        # FORCE bf16 trainer
FSDP_ROUTER_OUT=/tmp/fsdp_router_<prec>.json \
FSDP_ROUTER_TRAINERS_OUT=/tmp/trainers_fsdp_moe_<prec>.json \
torchrun --nproc-per-node=4 --rdzv-endpoint=127.0.0.1:29512 \
    scripts/live/run_fsdp_moe_reference.py
```

The `MODEL=Qwen/Qwen3-30B-A3B` env var override is the load-bearing
detail — without it, `run_fsdp_moe_reference.py` reads
`rollouts[0]["model"]` which is `Qwen/Qwen3-30B-A3B-FP8` for the
FP8 batch, and we'd compare FP8 rollouts against an FP8 trainer
instead of the bf16 reference.

## Acceptance evidence

```text
=== bf16 : 19 payloads ===
  keys: ['engine', 'engine_fingerprint', 'expert_ids', 'model', 'num_experts',
         'num_layers', 'prompt_idx', 'prompt_text', 'prompt_token_ids',
         'response_token_ids', 'rollout_label', 'seed', 'sharding', 'top_k',
         'world_size']
  expert_ids shape: (529, 48, 8)        # [total_tokens, 48, 8] for prompt_idx=0
                                        # (529 = prompt + gen). Pair script
                                        # trims to the gen window.

=== fp8 : 16 payloads ===
  expert_ids shape: (529, 48, 8)
```

Wall clock on the spot:

| Pass  | Load + wrap | Forward (per rollout) | Total wall |
|-------|------------:|----------------------:|-----------:|
| bf16  | 70.0 s      | ~10 s × 19            | 290.5 s    |
| fp8   | 70.1 s      | ~10 s × 16            | 229.1 s    |

## Note on the missing trainer_logprobs

`run_fsdp_moe_reference.py` emits the MoE router trace only —
`trainer_logprobs` is an empty list in every payload. ESS for these
trajectories would require a parallel run of
`run_fsdp_reference.py` (the dense forward path that emits per-token
logprobs); that is a separate iteration if the matrix's ESS column
needs to be populated. For the **router_flip_rate** acceptance
gate (iters 7+8), `expert_ids` is the load-bearing field and it is
present and correctly shaped here.

## Followup note for iter 6 (Megatron)

The Megatron MoE launcher writes BOTH logprobs and router (separate
files: `/tmp/trainer_megatron-bf16.json` + `/tmp/megatron_router.json`).
For iter 7's pair step, the FSDP side will read `expert_ids` from
the trainers JSON; the Megatron side will read `expert_ids` from the
dedicated router JSON.
