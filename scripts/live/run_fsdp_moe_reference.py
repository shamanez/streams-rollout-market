"""Stage 2 (FSDP MoE): teacher-force forward over (prompt + response) with
expert routing trace.

Extends ``run_fsdp_reference.py`` to MoE models (Qwen3-30B-A3B etc.).
Same FULL_SHARD FSDP wrap pattern, but passes
``output_router_logits=True`` to the HF model and extracts the top-k
expert IDs per ``(seq_position, moe_layer)`` so the trace can pair with
``run_vllm_moe_rollout.py``'s output for router-mismatch analysis.

Output ``/tmp/fsdp_router.json`` matches the shape of
``/tmp/vllm_router.json`` (see ``run_vllm_moe_rollout.py``):
``num_tokens = num_prompt_tokens + num_generated_tokens - 1``, positions
``0 .. P+R-2``, expert IDs in ``[0, num_experts)``. That alignment is
load-bearing — ``RouterTrace`` rejects pairs with mismatched
``(num_tokens, num_layers, top_k)``.

Reads ``/tmp/rollout.json`` for the prompt + response token IDs (the
output of ``run_vllm_moe_rollout.py``). Reads ``MODEL`` from env to
pick the MoE checkpoint (defaults to ``Qwen/Qwen3-30B-A3B``).

Launch (on the spot host)::

    source ~/rmenv/bin/activate
    export HF_HOME=/home/ubuntu/hf-cache
    cd ~
    MODEL=Qwen/Qwen3-30B-A3B torchrun --nproc-per-node=4 run_fsdp_moe_reference.py

The torch / transformers imports are deferred inside ``main()`` so this
module is import-safe on a host without those packages — the shape test
(``tests/test_fsdp_moe_reference_shape.py``) imports the module and
exercises only the pure-Python helpers
(``routed_position_indices``, ``format_fsdp_router_payload``,
``_engine_fingerprint``).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path


MODEL_DEFAULT = "Qwen/Qwen3-30B-A3B"
ROLLOUT_PATH = Path("/tmp/rollout.json")
OUT_PATH = Path(os.environ.get("FSDP_ROUTER_OUT", "/tmp/fsdp_router.json"))


def routed_position_indices(prompt_len: int, response_len: int) -> list[int]:
    """Return the position indices that vLLM's ``routed_experts`` covers.

    vLLM emits one ``[moe_layer, top_k]`` entry per position in
    ``range(0, prompt_len + response_len - 1)`` — i.e. every position
    except the last, which would predict a token outside the generated
    sequence. Match that convention exactly so a ``RouterTrace`` built
    from this script's output pairs cleanly with one built from
    ``/tmp/vllm_router.json`` in ``compute_router_mismatch``.
    """
    total = prompt_len + response_len
    if total <= 1:
        return []
    return list(range(total - 1))


def _engine_fingerprint(model_id: str, world_size: int) -> str:
    """Stable fingerprint string for the FSDP-MoE row aggregator.

    Includes the model name (escaped) and the shard count so the dense /
    router dashboards can distinguish a TP=4 FSDP MoE forward from
    other engine configurations.
    """
    safe = model_id.replace("/", "_")
    return f"sha256:fsdp-fullshard-bf16-cuda13-moe-ws{world_size}-{safe}"


def format_fsdp_router_payload(
    *,
    model_id: str,
    prompt_token_ids: list[int],
    response_token_ids: list[int],
    num_layers: int,
    num_experts: int,
    top_k: int,
    expert_ids: list[list[list[int]]],
    seed: int,
    world_size: int,
    rollout_label: str = "fsdp-bf16",
    prompt_text: str = "",
) -> dict:
    """Build the FSDP router payload matching /tmp/vllm_router.json shape."""
    return {
        "model": model_id,
        "engine": "fsdp",
        "engine_fingerprint": _engine_fingerprint(model_id, world_size),
        "prompt_text": prompt_text,
        "prompt_token_ids": prompt_token_ids,
        "response_token_ids": response_token_ids,
        "num_layers": num_layers,
        "num_experts": num_experts,
        "top_k": top_k,
        "expert_ids": expert_ids,
        "seed": seed,
        "rollout_label": rollout_label,
        "world_size": world_size,
        "sharding": "FULL_SHARD",
    }


def _topk_experts_from_router_logits(
    router_logits_per_layer: "list",  # list[torch.Tensor]
    positions: list[int],
    top_k: int,
) -> list[list[list[int]]]:
    """Extract top-k expert IDs at each (position, layer).

    HF's ``output_router_logits=True`` returns one tensor per MoE layer,
    each shape ``[seq_len, num_experts]`` (batch flattened, batch=1
    here). For each layer we read out positions ``positions``, then
    transpose so the outer index is position and inner is layer.
    """
    import torch  # noqa: F401  # torch is required at runtime

    out: list[list[list[int]]] = [[] for _ in positions]
    for layer_logits in router_logits_per_layer:
        topk_indices = torch.topk(layer_logits, k=top_k, dim=-1).indices.tolist()
        for out_idx, pos in enumerate(positions):
            out[out_idx].append([int(x) for x in topk_indices[pos]])
    return out


def main() -> None:
    """Build the FSDP-MoE model, teacher-force, dump /tmp/fsdp_router.json."""
    import torch
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import ShardingStrategy
    from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
    from transformers import AutoModelForCausalLM

    os.environ.setdefault("HF_HOME", "/home/ubuntu/hf-cache")

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)

    rollout = json.loads(ROLLOUT_PATH.read_text())
    model_id = os.environ.get("MODEL") or rollout.get("model") or MODEL_DEFAULT
    prompt_token_ids = rollout["prompt_token_ids"]
    response_token_ids = rollout["response_token_ids"]
    prompt_len = len(prompt_token_ids)
    response_len = len(response_token_ids)
    seed = int(rollout.get("seed", 1234))
    prompt_text = rollout.get("prompt_text", "")

    if rank == 0:
        print(
            f"[fsdp-moe] world_size={world_size} model={model_id} "
            f"prompt_len={prompt_len} response_len={response_len}",
            flush=True,
        )

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        output_router_logits=True,
    )
    model.eval()
    dist.barrier()
    if rank == 0:
        print(f"[fsdp-moe] HF load complete in {time.time()-t0:.1f}s", flush=True)

    cfg = model.config
    num_experts = int(getattr(cfg, "num_experts", 0))
    top_k = int(getattr(cfg, "num_experts_per_tok", 0))
    if num_experts == 0 or top_k == 0:
        raise RuntimeError(
            f"{model_id} does not look like a MoE model "
            f"(num_experts={num_experts}, top_k={top_k})"
        )

    auto_wrap_policy = __import__("functools").partial(
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
        print(f"[fsdp-moe] FSDP wrap complete in {time.time()-t1:.1f}s", flush=True)

    ids = torch.tensor(
        [prompt_token_ids + response_token_ids],
        dtype=torch.long,
        device=f"cuda:{local_rank}",
    )
    t2 = time.time()
    with torch.no_grad():
        out = model(input_ids=ids, output_router_logits=True)
    if out.router_logits is None:
        raise RuntimeError(f"{model_id} returned no router_logits")
    router_logits_per_layer = list(out.router_logits)
    num_layers = len(router_logits_per_layer)
    if rank == 0:
        print(
            f"[fsdp-moe] forward in {time.time()-t2:.1f}s "
            f"(num_layers={num_layers}, num_experts={num_experts}, top_k={top_k})",
            flush=True,
        )

    positions = routed_position_indices(prompt_len, response_len)
    expert_ids = _topk_experts_from_router_logits(
        router_logits_per_layer, positions, top_k
    )

    if rank == 0:
        payload = format_fsdp_router_payload(
            model_id=model_id,
            prompt_token_ids=prompt_token_ids,
            response_token_ids=response_token_ids,
            num_layers=num_layers,
            num_experts=num_experts,
            top_k=top_k,
            expert_ids=expert_ids,
            seed=seed,
            world_size=world_size,
            prompt_text=prompt_text,
        )
        OUT_PATH.write_text(json.dumps(payload))
        print(
            f"[fsdp-moe] wrote {OUT_PATH} "
            f"(num_tokens={len(expert_ids)}, num_layers={num_layers}, "
            f"top_k={top_k}, num_experts={num_experts})",
            flush=True,
        )
        print(f"[fsdp-moe] total {time.time()-t0:.1f}s", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
