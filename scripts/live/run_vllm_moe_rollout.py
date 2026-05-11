"""vLLM MoE rollout that exposes per-(token, layer) routed experts.

Latest vLLM ships a first-class path for surfacing the MoE router's
selected expert IDs via ``AsyncEngineArgs(enable_return_routed_experts=True)``.
Each ``CompletionOutput`` then carries a ``routed_experts`` ndarray with
shape ``[seq_position, moe_layer, top_k]`` and length
``num_prompt_tokens + num_generated_tokens - 1`` (position 0 has no
prediction). Expert IDs are in ``[0, num_experts)``. Reference:
https://docs.vllm.ai/en/latest/examples/rl/routed_experts_e2e/ .

This script generates one response from ``Qwen/Qwen3-30B-A3B`` (or any
MoE model passed via ``MODEL``) and writes ``/tmp/vllm_router.json`` in
a shape that the router-mismatch fixture builder consumes alongside
the trainer-side ``/tmp/megatron_router.json``.

The trace contains *selected* expert IDs only (no gate logits), which
is exactly what ``router_flip_rate`` (top-1 set comparison) needs —
``compute_router_mismatch`` in ``observatory.router_mismatch_lab``
operates on the ``expert_ids[token][layer][top_k]`` shape.

Run on ``my-vllm-spot-instance`` (TP=4 across 4× L40S)::

    MODEL=Qwen/Qwen3-30B-A3B \\
    ROLLOUT_LABEL=vllm-bf16 \\
    TENSOR_PARALLEL_SIZE=4 \\
    python ~/run_vllm_moe_rollout.py

Outputs ``/tmp/vllm_router.json``; scp it back locally and pair it with
``/tmp/megatron_router.json`` via ``scripts/live/build_router_input.py``
(or its env-driven sibling for the vllm→megatron pairing).
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from pathlib import Path

# Lazy vLLM imports — keep this script importable without vLLM installed
# so a tokeniser-only / shape-only test can validate the output schema.


MODEL = os.environ.get("MODEL", "Qwen/Qwen3-30B-A3B")
ROLLOUT_LABEL = os.environ.get("ROLLOUT_LABEL", "vllm-bf16")
TP_SIZE = int(os.environ.get("TENSOR_PARALLEL_SIZE", "4"))
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "4096"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "64"))
ATTN_BACKEND = os.environ.get("ATTENTION_BACKEND", "FLASH_ATTN")
PROMPT_TEXT = os.environ.get(
    "PROMPT_TEXT",
    "List three concrete failure modes a remote rollout worker might exhibit "
    "that an off-policy budget controller could detect.",
)
SEED = int(os.environ.get("SEED", "1234"))
OUT_PATH = Path(os.environ.get("OUT_PATH", "/tmp/vllm_router.json"))


def _engine_fingerprint(model: str, label: str, vllm_version: str) -> str:
    """Stable string we can grep in the dashboard's row aggregator."""
    return f"sha256:vllm-{vllm_version}-{label}-{model.replace('/', '_')}"


def _routed_experts_to_lists(arr) -> list[list[list[int]]]:
    """Coerce the ndarray-shaped routed_experts trace to plain Python lists.

    Shape is ``[seq_position, moe_layer, top_k]``. We tolerate either a
    numpy array (default) or a nested list (test stubs).
    """
    try:
        return [[list(map(int, layer)) for layer in token] for token in arr]
    except TypeError:  # pragma: no cover — defensive on unexpected types
        raise ValueError(f"routed_experts is not a 3-D iterable: {type(arr)!r}")


async def _run_vllm() -> dict:
    from vllm import SamplingParams
    from vllm import __version__ as vllm_version
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM  # type: ignore[import-not-found]

    engine_args = AsyncEngineArgs(
        model=MODEL,
        enable_return_routed_experts=True,
        tensor_parallel_size=TP_SIZE,
        max_model_len=MAX_MODEL_LEN,
        disable_log_stats=True,
        attention_backend=ATTN_BACKEND,
        seed=SEED,
    )
    print(f"[vllm-moe] starting AsyncLLM model={MODEL} TP={TP_SIZE}", flush=True)
    t0 = time.time()
    engine = AsyncLLM.from_engine_args(engine_args)

    sp = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS, seed=SEED)
    request_id = str(uuid.uuid4())
    final = None
    async for out in engine.generate(PROMPT_TEXT, sp, request_id):
        if out.finished:
            final = out
            break
    if final is None:
        raise RuntimeError("AsyncLLM.generate completed without a finished output")
    print(f"[vllm-moe] generate finished in {time.time()-t0:.1f}s", flush=True)

    completion = final.outputs[0]
    routed = completion.routed_experts
    if routed is None:
        raise RuntimeError(
            "completion.routed_experts is None — confirm "
            "enable_return_routed_experts=True and that the model is MoE"
        )
    expert_ids = _routed_experts_to_lists(routed)

    prompt_token_ids = list(map(int, final.prompt_token_ids))
    response_token_ids = list(map(int, completion.token_ids))

    # Sanity: routed_experts spans num_prompt_tokens + num_generated - 1
    # positions (position 0 has no prediction in autoregressive routing).
    expected = len(prompt_token_ids) + len(response_token_ids) - 1
    if len(expert_ids) != expected:
        raise RuntimeError(
            f"routed_experts length {len(expert_ids)} != "
            f"prompt+gen-1 ({expected}); shape contract violated"
        )

    num_tokens = len(expert_ids)
    num_layers = len(expert_ids[0]) if num_tokens else 0
    top_k = len(expert_ids[0][0]) if num_tokens and num_layers else 0
    num_experts = int(max((eid for tok in expert_ids for layer in tok for eid in layer), default=0)) + 1

    return {
        "model": MODEL,
        "engine": "vllm",
        "engine_fingerprint": _engine_fingerprint(MODEL, ROLLOUT_LABEL, vllm_version),
        "prompt_text": PROMPT_TEXT,
        "prompt_token_ids": prompt_token_ids,
        "response_token_ids": response_token_ids,
        "num_layers": num_layers,
        "num_experts": num_experts,
        "top_k": top_k,
        "expert_ids": expert_ids,
        "seed": SEED,
        "rollout_label": ROLLOUT_LABEL,
    }


def main() -> None:
    payload = asyncio.run(_run_vllm())
    OUT_PATH.write_text(json.dumps(payload))
    print(
        f"[vllm-moe] wrote {OUT_PATH} "
        f"(num_tokens={len(payload['expert_ids'])}, "
        f"num_layers={payload['num_layers']}, "
        f"top_k={payload['top_k']}, "
        f"num_experts={payload['num_experts']})",
        flush=True,
    )


if __name__ == "__main__":
    main()
