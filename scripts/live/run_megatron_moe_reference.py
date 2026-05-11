"""Megatron-LM trainer-side reference forward pass.

Mirrors `run_fsdp_reference.py` but inside the slimerl/slime docker
image, using Megatron's GPTModel + torch-dist checkpoint instead of
HF transformers. The motivation: FSDP forward-only is bit-equivalent
to HF transformers (we measured); Megatron uses different attention
and MLP CUDA kernels and a different MoE dispatch path, so it's the
first trainer engine that actually moves the forward-pass numbers
versus vLLM rollouts.

Run pattern (launched from the host, *inside* docker):

    docker run --rm --gpus all --ipc=host \
        --shm-size=64g \
        --ulimit memlock=-1 --ulimit stack=67108864 \
        -v ~/megatron_conversion/megatron_ckpt:/root/Qwen3-30B-A3B_torch_dist:ro \
        -v /tmp:/host_tmp \
        -v /path/to/run_megatron_reference.py:/root/run_megatron_reference.py \
        slimerl/slime:latest \
        bash -lc 'PYTHONPATH=/root/Megatron-LM/:/root/slime/ \
            torchrun --nproc-per-node 4 /root/run_megatron_reference.py'

Reads /host_tmp/rollouts.json (list) or /host_tmp/rollout.json (single)
on rank 0. Writes /host_tmp/trainers.json (list) and /host_tmp/trainer.json
(first entry) on rank 0 with the per-token trainer-side logprobs.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist

# slime sources the same MODEL_ARGS into argv via the wrapper shell; we
# rely on the caller to pass the Qwen3-30B-A3B MODEL_ARGS through.
from megatron.core import mpu  # noqa: F401
from megatron.core.enums import ModelType
from megatron.training.arguments import parse_args, validate_args
from megatron.training.checkpointing import load_checkpoint
from megatron.training.training import get_model

import slime_plugins.mbridge  # noqa: F401  - mbridge registration side effect
from slime.backends.megatron_utils.arguments import set_default_megatron_args
from slime.backends.megatron_utils.initialize import init
from slime.backends.megatron_utils.model_provider import get_model_provider_func
from slime.utils.logging_utils import configure_logger


HOST_TMP = Path("/host_tmp")  # mounted from host /tmp


def _add_args(parser):
    parser.add_argument("--rollouts-path", default=str(HOST_TMP / "rollouts.json"))
    parser.add_argument(
        "--rollouts-path-fallback", default=str(HOST_TMP / "rollout.json")
    )
    # Write to dedicated megatron filenames so docker-as-root doesn't
    # collide with host-owned /tmp/trainers.json from the vLLM/HF runs.
    parser.add_argument(
        "--out-path", default=str(HOST_TMP / "trainers_megatron-bf16.json")
    )
    parser.add_argument(
        "--out-path-single", default=str(HOST_TMP / "trainer_megatron-bf16.json")
    )
    # slime's set_default_megatron_args reads args.hf_checkpoint internally
    # even when we only load from torch-dist; provide it so the namespace
    # has the attribute. Default to the bind-mount path used by the runner.
    parser.add_argument(
        "--hf-checkpoint", default="/root/Qwen3-30B-A3B",
        help="HF model path; only used by slime's arg defaulting.",
    )
    parser.add_argument(
        "--emit-router", action="store_true",
        help="If set, also write /host_tmp/megatron_router.json with top-k expert IDs per (token, layer).",
    )
    return parser


def _setup_args():
    """Mirror slime/tools/convert_hf_to_torch_dist.py's get_args() flow.

    parse_args → set_default_megatron_args → manual stub fields →
    validate_args (validate_args populates derived fields like
    virtual_pipeline_model_parallel_size).
    """
    args = parse_args(_add_args)
    args = set_default_megatron_args(args)
    args.save_interval = 1
    args.micro_batch_size = 1
    args.global_batch_size = int(os.environ.get("WORLD_SIZE", "1"))
    # Slime's model_provider / mbridge code reads attrs that the convert
    # script registers via its own add_convertion_args parser. Default
    # them here so slime can pass through unchanged.
    if not hasattr(args, "megatron_to_hf_mode"):
        args.megatron_to_hf_mode = "raw"
    if not hasattr(args, "padded_vocab_size"):
        args.padded_vocab_size = None
    validate_args(args)
    return args


def _load_rollouts(args) -> list[dict]:
    p = Path(args.rollouts_path)
    if p.exists():
        return json.loads(p.read_text())
    fallback = Path(args.rollouts_path_fallback)
    if fallback.exists():
        return [json.loads(fallback.read_text())]
    raise FileNotFoundError(
        f"no rollouts at {args.rollouts_path} or {args.rollouts_path_fallback}"
    )


def _install_router_hooks(model_module: torch.nn.Module) -> list[dict]:
    """Find every MoE router and register a forward hook.

    Returns a list of {layer_idx, store: list} dicts; after a forward
    pass each `store` holds the router's output for that layer.
    """
    hooks: list[dict] = []
    handles = []
    for name, mod in model_module.named_modules():
        # Megatron's transformer block names MoE routers as
        # `decoder.layers.{i}.mlp.router`. Pre-MoE layers don't have it.
        if name.endswith(".mlp.router") or name.endswith(".router"):
            store: list = []

            def hook(_mod, _inp, out, _store=store):
                _store.append(out)

            handles.append(mod.register_forward_hook(hook))
            hooks.append({"name": name, "store": store, "handle": handles[-1]})
    return hooks


def _extract_router_topk(
    router_hook_stores: list[dict],
    prompt_len: int,
    response_len: int,
    top_k: int,
) -> list[list[list[int]]]:
    """Pull top-k expert IDs per (response_token, layer) from captured outputs.

    Each router's output is typically a tuple `(scores, indices)` or
    just `indices` of shape [num_tokens, top_k]. We accept either.
    """
    P, R = prompt_len, response_len
    # out[r_idx][layer_idx] = list of top_k expert IDs
    out: list[list[list[int]]] = [[] for _ in range(R)]
    for layer_info in router_hook_stores:
        if not layer_info["store"]:
            continue
        raw = layer_info["store"][-1]
        if isinstance(raw, tuple):
            # (scores, indices) or (probs, indices, ...)
            indices = raw[1]
        else:
            indices = raw
        if not isinstance(indices, torch.Tensor):
            continue
        # Expect shape [num_tokens, top_k]; if it's [batch, seq, top_k] flatten.
        if indices.dim() == 3:
            indices = indices.reshape(-1, indices.shape[-1])
        # Some routers store probs not indices — fall back to topk.
        if indices.shape[-1] != top_k:
            scores = raw[0] if isinstance(raw, tuple) else None
            if scores is not None and isinstance(scores, torch.Tensor):
                indices = torch.topk(scores, k=top_k, dim=-1).indices
        start = P - 1
        for r_idx in range(R):
            t_idx = start + r_idx
            if t_idx < indices.shape[0]:
                out[r_idx].append([int(x) for x in indices[t_idx].tolist()])
    return out


def _teacher_force_one(
    model_module: torch.nn.Module,
    prompt_ids: list[int],
    response_ids: list[int],
    *,
    device: torch.device,
) -> list[float]:
    """Run the model over (prompt + response) and extract per-token logprobs.

    The model is post_process=True on the final pipeline rank, so its
    forward returns logits gathered to local TP rank (parallel_output
    is forced False in get_model_provider_func for inference here).
    """
    P, R = len(prompt_ids), len(response_ids)
    ids = torch.tensor([prompt_ids + response_ids], dtype=torch.long, device=device)
    seq_len = P + R
    position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)
    # Megatron's GPT uses an internal causal mask when attention_mask is
    # None; passing None is fine for inference.
    with torch.no_grad():
        out = model_module(
            input_ids=ids,
            position_ids=position_ids,
            attention_mask=None,
            runtime_gather_output=True,
        )
    # Megatron may return either a tensor (shape [batch, seq, vocab]) or
    # a named-tuple-like with .logits; normalise.
    logits = out if isinstance(out, torch.Tensor) else out.logits
    logits = logits.float()
    # Layout: Megatron typically returns [seq, batch, vocab] for the GPT
    # model with parallel_output=False; some configs return [batch, seq,
    # vocab]. Detect by which dim equals batch=1.
    if logits.dim() == 3 and logits.shape[0] == 1 and logits.shape[1] == seq_len:
        # [batch, seq, vocab]
        seq_logits = logits[0]
    elif logits.dim() == 3 and logits.shape[1] == 1 and logits.shape[0] == seq_len:
        # [seq, batch, vocab]
        seq_logits = logits[:, 0, :]
    else:
        raise RuntimeError(f"unexpected logits shape: {tuple(logits.shape)}")
    trainer_logprobs: list[float] = []
    for i in range(R):
        # Position (P-1+i) predicts response token i.
        log_probs = torch.log_softmax(seq_logits[P - 1 + i], dim=-1)
        trainer_logprobs.append(float(log_probs[response_ids[i]].item()))
    return trainer_logprobs


def main() -> None:
    configure_logger()

    # Mirror the convert script: manually init_process_group BEFORE
    # building args, so MPI/SLURM-style env vars are honoured.
    world_size = int(os.environ.get("WORLD_SIZE") or 1)
    local_rank = int(os.environ.get("LOCAL_RANK") or 0)
    global_rank = int(os.environ.get("RANK") or 0)
    torch.cuda.set_device(local_rank)
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12355")
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            world_size=world_size,
            rank=global_rank,
            device_id=torch.device(f"cuda:{local_rank}"),
        )

    args = _setup_args()
    init(args)

    rank = global_rank
    if rank == 0:
        print(
            f"[megatron] world_size={world_size}, "
            f"TP={args.tensor_model_parallel_size}, "
            f"PP={args.pipeline_model_parallel_size}, "
            f"EP={getattr(args, 'expert_model_parallel_size', 1)}",
            flush=True,
        )

    # Build the model and load the torch-dist checkpoint.
    t0 = time.time()
    provider = get_model_provider_func(args, role="actor")
    model = get_model(provider, ModelType.encoder_or_decoder, wrap_with_ddp=False)
    # model is a list (one entry per pipeline stage); for TP-only it's
    # length 1.
    if rank == 0:
        print(f"[megatron] built model in {time.time()-t0:.1f}s", flush=True)

    t1 = time.time()
    load_checkpoint(model, optimizer=None, opt_param_scheduler=None)
    if rank == 0:
        print(f"[megatron] loaded torch-dist ckpt in {time.time()-t1:.1f}s", flush=True)

    # Use the post-process pipeline stage's module for the LM head.
    model_module = model[0]
    model_module.eval()
    device = torch.device(f"cuda:{local_rank}")

    rollouts = _load_rollouts(args)
    if rank == 0:
        print(f"[megatron] {len(rollouts)} rollout(s) to teacher-force", flush=True)

    # Optionally instrument router hooks for the MoE trace.
    router_hooks = _install_router_hooks(model_module) if args.emit_router else []
    if rank == 0 and args.emit_router:
        print(f"[megatron] router hooks installed on {len(router_hooks)} modules", flush=True)

    trainers: list[dict] = []
    router_records: list[dict] = []
    t2 = time.time()
    for entry in rollouts:
        prompt_ids = entry["prompt_token_ids"]
        response_ids = entry["response_token_ids"]
        # Clear any previous router-hook captures so we only see this run.
        for h in router_hooks:
            h["store"].clear()
        trainer_logprobs = _teacher_force_one(
            model_module, prompt_ids, response_ids, device=device
        )
        if rank == 0 and args.emit_router and router_hooks:
            top_k = getattr(args, "moe_router_topk", 8) or 8
            expert_ids = _extract_router_topk(
                router_hooks, len(prompt_ids), len(response_ids), top_k
            )
            router_records.append({
                "model": entry.get("model", "Qwen/Qwen3-30B-A3B"),
                "prompt_token_ids": prompt_ids,
                "response_token_ids": response_ids,
                "num_layers": len(router_hooks),
                "num_experts": int(getattr(args, "num_experts", 128) or 128),
                "top_k": int(top_k),
                "expert_ids": expert_ids,
                "engine": "megatron",
            })
        if rank == 0:
            trainers.append({
                "model": entry.get("model", "Qwen/Qwen3-30B-A3B"),
                "trainer_logprobs": trainer_logprobs,
                "engine": "megatron",
                "engine_version": "core-r0.13",
                "engine_fingerprint": (
                    f"sha256:megatron-core-r0.13-tp{args.tensor_model_parallel_size}-"
                    f"ep{getattr(args, 'expert_model_parallel_size', 1)}-bf16-torch-dist"
                ),
                "dtype": "bfloat16",
                "world_size": world_size,
                "prompt_idx": entry.get("prompt_idx", 0),
                "trainer_reference": entry.get("trainer_reference") or entry.get("model"),
            })
            print(
                f"[megatron]   prompt {entry.get('prompt_idx', 0)}: "
                f"{len(response_ids)} tokens",
                flush=True,
            )

    if rank == 0:
        print(
            f"[megatron] forwarded {len(trainers)} prompt(s) in {time.time()-t2:.1f}s",
            flush=True,
        )
        Path(args.out_path).write_text(json.dumps(trainers))
        if trainers:
            Path(args.out_path_single).write_text(json.dumps(trainers[0]))
        print(f"[megatron] wrote {args.out_path} ({len(trainers)} entries)")
        if router_records:
            router_out = str(HOST_TMP / "megatron_router.json")
            Path(router_out).write_text(json.dumps(router_records))
            print(
                f"[megatron] wrote {router_out} "
                f"({len(router_records)} record(s), {router_records[0]['num_layers']} layers, "
                f"top_k={router_records[0]['top_k']})"
            )
        print(f"[megatron] total {time.time()-t0:.1f}s")

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
