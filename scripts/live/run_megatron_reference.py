"""Stage 2 (Megatron-LM): teacher-force forward over (prompt + response).

Runs inside the ``slimerl/slime:latest`` container under
``torchrun --nproc-per-node 4``. Reads ``/tmp/rollouts.json`` (preferred)
or ``/tmp/rollout.json`` (legacy single-prompt fallback), loads the
Qwen3-32B torch-dist checkpoint at ``/root/Qwen3-32B_torch_dist/`` at
TP=4, teacher-forces ``(prompt + response)`` through Megatron-LM, gathers
per-response-token logprobs, and writes ``/tmp/trainer_megatron-bf16.json``
in the same shape as ``run_hf_reference.py`` / ``run_fsdp_reference.py``
so ``scripts/live/build_dense_input.py --variant megatron-bf16`` can
ingest it without further glue.

Uses Megatron-LM's stock ``model_provider`` + ``gpt_builder`` directly
rather than slime's wrappers — slime adds bridge / RL-specific args
that aren't relevant for a single teacher-force forward. The
distributed init runs through Megatron's ``initialize_megatron``.

The torch / Megatron imports are deferred inside ``main()`` so this
module is import-safe on a host without Megatron installed — the
shape test (``tests/test_megatron_reference_shape.py``) imports the
module and exercises only the pure-Python helpers
(``response_prediction_positions``, ``format_trainer_payload``,
``_build_megatron_argv``, ``_load_rollouts``).

Launch (host-side, on ``my-vllm-spot-instance``):
    bash scripts/live/megatron_reference_launch.sh

The host launcher bind-mounts ``~/megatron_conversion/qwen3_32b_torch_dist``
into the container as ``/root/Qwen3-32B_torch_dist`` and copies the
host-side ``/tmp/rollout.json`` into the container before invoking this
script under ``torchrun``.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path


ROLLOUT_PATH_PRIMARY = Path("/tmp/rollouts.json")
ROLLOUT_PATH_LEGACY = Path("/tmp/rollout.json")
TRAINER_OUT_PRIMARY = Path("/tmp/trainers_megatron-bf16.json")
TRAINER_OUT_LEGACY = Path("/tmp/trainer_megatron-bf16.json")
CKPT_DIR_DEFAULT = "/root/Qwen3-32B_torch_dist/"
MODEL_ID_DEFAULT = "Qwen/Qwen3-32B"
ENGINE_FINGERPRINT = "sha256:megatron-core-r0.13-bf16-torch-dist"

# Mirrors /root/slime/scripts/models/qwen3-32B.sh inside the slimerl
# image. Duplicated here so a fresh container can run this script
# without sourcing the shell file separately. The torch-dist ckpt was
# saved with these args; they must match exactly to load.
QWEN3_32B_MODEL_ARGS: list[str] = [
    "--swiglu",
    "--num-layers", "64",
    "--hidden-size", "5120",
    "--ffn-hidden-size", "25600",
    "--num-attention-heads", "64",
    "--group-query-attention",
    "--num-query-groups", "8",
    "--use-rotary-position-embeddings",
    "--disable-bias-linear",
    "--normalization", "RMSNorm",
    "--norm-epsilon", "1e-6",
    "--rotary-base", "1000000",
    "--vocab-size", "151936",
    "--kv-channels", "128",
    "--qk-layernorm",
    "--untie-embeddings-and-output-weights",
]


def _load_rollouts() -> list[dict]:
    """Load ``/tmp/rollouts.json`` if present, else wrap ``/tmp/rollout.json``."""
    if ROLLOUT_PATH_PRIMARY.exists():
        return json.loads(ROLLOUT_PATH_PRIMARY.read_text())
    return [json.loads(ROLLOUT_PATH_LEGACY.read_text())]


def response_prediction_positions(prompt_len: int, response_len: int) -> list[int]:
    """Return the logits-tensor positions that predict each response token.

    For an autoregressive teacher-force over ``(prompt + response)`` of
    total length ``P + R``, the model output at position ``P - 1 + i``
    predicts ``response_token_ids[i]``. Mirrors ``run_hf_reference.py``
    and ``run_fsdp_reference.py``.
    """
    return [prompt_len - 1 + i for i in range(response_len)]


def format_trainer_payload(
    *,
    model_id: str,
    trainer_logprobs: list[float],
    prompt_idx: int,
    world_size: int,
    tensor_model_parallel_size: int,
    ckpt_dir: str,
) -> dict:
    """Build the trainer-side payload dict in the run_hf_reference shape."""
    return {
        "model": model_id,
        "trainer_logprobs": trainer_logprobs,
        "engine": "megatron-lm",
        "engine_fingerprint": ENGINE_FINGERPRINT,
        "attn_impl": "transformer_engine",
        "dtype": "bfloat16",
        "world_size": world_size,
        "tensor_model_parallel_size": tensor_model_parallel_size,
        "ckpt_dir": ckpt_dir,
        "prompt_idx": prompt_idx,
    }


def _build_megatron_argv() -> list[str]:
    """Construct sys.argv for Megatron's parse_args() inside main()."""
    ckpt_dir = os.environ.get("MEGATRON_CKPT_DIR", CKPT_DIR_DEFAULT)
    tp = os.environ.get("TENSOR_MODEL_PARALLEL_SIZE", "4")
    seq_len = os.environ.get("SEQ_LENGTH", "4096")
    micro_bs = os.environ.get("MICRO_BATCH_SIZE", "1")
    tokenizer = os.environ.get("HF_TOKENIZER", MODEL_ID_DEFAULT)

    argv = [
        "run_megatron_reference.py",
        "--load", ckpt_dir,
        "--no-load-optim",
        "--no-load-rng",
        "--tensor-model-parallel-size", tp,
        "--pipeline-model-parallel-size", "1",
        "--use-mcore-models",
        "--transformer-impl", "transformer_engine",
        "--bf16",
        "--micro-batch-size", micro_bs,
        "--global-batch-size", micro_bs,
        "--train-iters", "1",
        "--seq-length", seq_len,
        "--max-position-embeddings", seq_len,
        "--position-embedding-type", "rope",
        "--tokenizer-type", "HuggingFaceTokenizer",
        "--tokenizer-model", tokenizer,
        "--exit-on-missing-checkpoint",
        "--no-masked-softmax-fusion",
    ]
    argv.extend(QWEN3_32B_MODEL_ARGS)
    return argv


def main() -> None:
    """Build the Megatron model, load the dist-ckpt, teacher-force, write JSON."""
    import functools

    # Megatron-LM's pretrain_gpt.py imports `model_provider` and
    # `gpt_builders` as top-level modules (they live in /root/Megatron-LM/),
    # so we have to add that path to sys.path before importing them.
    sys.path.insert(0, "/root/Megatron-LM")

    sys.argv = _build_megatron_argv()

    import torch
    import torch.distributed as dist
    from megatron.core import mpu
    from megatron.core.enums import ModelType
    from megatron.core.tensor_parallel import gather_from_tensor_model_parallel_region
    from megatron.training import get_args, get_model
    from megatron.training.checkpointing import load_checkpoint
    from megatron.training.initialize import initialize_megatron
    from gpt_builders import gpt_builder  # type: ignore[import-not-found]
    from model_provider import model_provider  # type: ignore[import-not-found]

    initialize_megatron(args_defaults={})
    args = get_args()

    world_size = mpu.get_data_parallel_world_size() * mpu.get_tensor_model_parallel_world_size()
    rank = dist.get_rank()
    if rank == 0:
        print(
            f"[megatron-ref] world_size={world_size} "
            f"TP={args.tensor_model_parallel_size} ckpt={args.load}",
            flush=True,
        )

    rollouts = _load_rollouts()
    model_id = rollouts[0].get("trainer_reference") or rollouts[0]["model"]

    t0 = time.time()
    provider = functools.partial(model_provider, gpt_builder)
    # wrap_with_ddp=False skips the DDP grad-buffer allocation
    # (~30 GiB on Qwen3-32B); we're inference-only so no optimizer
    # state or gradient bucket is needed. Saves ~half the L40S VRAM.
    model = get_model(provider, ModelType.encoder_or_decoder, wrap_with_ddp=False)
    if rank == 0:
        print(f"[megatron-ref] model built in {time.time()-t0:.1f}s", flush=True)

    t1 = time.time()
    load_checkpoint(model, None, None, strict=False)
    if rank == 0:
        print(f"[megatron-ref] checkpoint loaded in {time.time()-t1:.1f}s", flush=True)

    for m in model:
        m.eval()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    trainers: list[dict] = []
    t2 = time.time()
    for entry in rollouts:
        prompt_ids = entry["prompt_token_ids"]
        response_ids = entry["response_token_ids"]
        prompt_len, response_len = len(prompt_ids), len(response_ids)
        ids = torch.tensor(
            [prompt_ids + response_ids], dtype=torch.long, device=f"cuda:{local_rank}"
        )
        position_ids = torch.arange(
            prompt_len + response_len, dtype=torch.long, device=ids.device
        ).unsqueeze(0)
        with torch.no_grad():
            logits = model[0](
                input_ids=ids,
                position_ids=position_ids,
                attention_mask=None,
            )
            # Default output is TP-sharded across the vocab dim (each rank
            # holds 1/tp_size of the vocab). Gather across TP before
            # log_softmax — the denominator must sum over the full vocab.
            if mpu.get_tensor_model_parallel_world_size() > 1:
                logits = gather_from_tensor_model_parallel_region(logits)
        positions = response_prediction_positions(prompt_len, response_len)
        logits_f32 = logits.float()
        trainer_logprobs: list[float] = []
        for pos, tok in zip(positions, response_ids):
            log_probs = torch.log_softmax(logits_f32[0, pos, :], dim=-1)
            trainer_logprobs.append(float(log_probs[tok].item()))
        payload = format_trainer_payload(
            model_id=model_id,
            trainer_logprobs=trainer_logprobs,
            prompt_idx=entry.get("prompt_idx", 0),
            world_size=world_size,
            tensor_model_parallel_size=args.tensor_model_parallel_size,
            ckpt_dir=args.load,
        )
        if rank == 0:
            trainers.append(payload)
            print(
                f"[megatron-ref]   prompt {payload['prompt_idx']}: "
                f"{response_len} tokens",
                flush=True,
            )
    if rank == 0:
        print(
            f"[megatron-ref] forwarded {len(trainers)} prompt(s) "
            f"in {time.time()-t2:.1f}s",
            flush=True,
        )
        TRAINER_OUT_PRIMARY.write_text(json.dumps(trainers))
        if trainers:
            TRAINER_OUT_LEGACY.write_text(json.dumps(trainers[0]))
        print(
            f"[megatron-ref] wrote {TRAINER_OUT_PRIMARY} "
            f"({len(trainers)} entries)",
            flush=True,
        )
        print(f"[megatron-ref] total {time.time()-t0:.1f}s", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
