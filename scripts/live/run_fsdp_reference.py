"""Stage 2 (alt): PyTorch FSDP reference forward pass.

Mirrors run_hf_reference.py but wraps the trainer-side model in
torch.distributed.fsdp.FullyShardedDataParallel instead of using
device_map='auto'. Same input contract (/tmp/rollouts.json) and the
same output schema (/tmp/trainers.json), so the existing
build_dense_matrix.py script consumes the result without changes.

The point is to surface a *third* trainer-side numerical path beyond
HF transformers (sdpa, device_map=auto) and vLLM. FSDP shards weights
across ranks and uses all-gather + reshard semantics in the forward
pass, which changes float accumulation order versus the un-sharded
reference. Megatron-LM is the obvious fourth — deferred because
HF→Megatron checkpoint conversion is its own session.

Launch (on the spot host)::

    source ~/rmenv/bin/activate
    export HF_HOME=/home/ubuntu/hf-cache
    cd ~
    torchrun --nproc-per-node=4 run_fsdp_reference.py
"""

from __future__ import annotations

import functools
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ.setdefault("HF_HOME", "/home/ubuntu/hf-cache")


def _load_rollouts() -> list[dict]:
    path = Path("/tmp/rollouts.json")
    if path.exists():
        return json.loads(path.read_text())
    flat = json.loads(Path("/tmp/rollout.json").read_text())
    return [flat]


def main() -> None:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)

    rollouts = _load_rollouts()
    model_id = rollouts[0].get("trainer_reference") or rollouts[0]["model"]

    if rank == 0:
        print(f"[fsdp] world_size={world_size}, model={model_id}", flush=True)
        AutoTokenizer.from_pretrained(model_id)  # warm cache

    # Block other ranks until rank 0 has the tokenizer cached. Required because
    # the AutoModelForCausalLM.from_pretrained call below also reads tokenizer
    # files and the cache is shared across ranks on the same host.
    dist.barrier()

    t0 = time.time()
    # Load to CPU first, then let FSDP shard onto each rank's GPU. This is
    # the textbook FSDP startup pattern and avoids OOM during load.
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    model.eval()
    dist.barrier()
    if rank == 0:
        print(f"[fsdp] HF load complete in {time.time()-t0:.1f}s", flush=True)

    # Auto-wrap by parameter count: any submodule with >= 1e8 parameters
    # becomes its own FSDP unit. For Qwen3-32B that buckets each
    # transformer block neatly without us hardcoding a class name.
    auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=int(1e8)
    )

    t1 = time.time()
    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        auto_wrap_policy=auto_wrap_policy,
        device_id=local_rank,
        use_orig_params=True,
    )
    dist.barrier()
    if rank == 0:
        print(f"[fsdp] FSDP wrap complete in {time.time()-t1:.1f}s", flush=True)

    trainers: list[dict] = []
    t2 = time.time()
    for entry in rollouts:
        prompt_ids = entry["prompt_token_ids"]
        response_ids = entry["response_token_ids"]
        P, R = len(prompt_ids), len(response_ids)
        ids = torch.tensor(
            [prompt_ids + response_ids], dtype=torch.long, device=f"cuda:{local_rank}"
        )
        with torch.no_grad():
            out = model(input_ids=ids)
        # FSDP gathers the final logits on the last rank that holds the
        # output projection. With FULL_SHARD + use_orig_params=True the
        # logits come back fully unsharded; we recompute logprobs locally.
        logits = out.logits.float()
        trainer_logprobs: list[float] = []
        for i in range(R):
            log_probs = torch.log_softmax(logits[0, P - 1 + i, :], dim=-1)
            trainer_logprobs.append(float(log_probs[response_ids[i]].item()))
        if rank == 0:
            trainers.append({
                "model": model_id,
                "trainer_logprobs": trainer_logprobs,
                "engine": "fsdp",
                "attn_impl": "sdpa",
                "dtype": "bfloat16",
                "world_size": world_size,
                "sharding": "FULL_SHARD",
                "prompt_idx": entry.get("prompt_idx", 0),
            })
            print(f"[fsdp]   prompt {entry.get('prompt_idx', 0)}: {R} tokens", flush=True)
    if rank == 0:
        print(f"[fsdp] forwarded {len(trainers)} prompt(s) in {time.time()-t2:.1f}s", flush=True)
        Path("/tmp/trainers.json").write_text(json.dumps(trainers))
        if trainers:
            Path("/tmp/trainer.json").write_text(json.dumps(trainers[0]))
        print(f"[fsdp] wrote /tmp/trainers.json ({len(trainers)} entries)")
        print(f"[fsdp] total {time.time()-t0:.1f}s")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
