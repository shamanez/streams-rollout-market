"""MoE router live run for Qwen3-30B-A3B.

Three stages, all via HF transformers (vLLM doesn't surface per-layer
router decisions on its OpenAI-shaped API):

  1. Generate one response from a chosen prompt with the bf16 MoE
     checkpoint (no router output needed here — we just want the
     tokens to teacher-force through both checkpoints).
  2. Teacher-force the (prompt + response) tokens through the bf16
     checkpoint with `output_router_logits=True`. Each MoE layer
     returns router logits over experts; we keep the top-k expert ids
     per (token, layer). This becomes the *trainer-side* RouterTrace.
  3. Teacher-force the same tokens through the FP8 checkpoint with
     `output_router_logits=True`. This becomes the *rollout-side*
     RouterTrace.

The two RouterTrace records have the same num_tokens / num_layers /
top_k / num_experts by construction, so the router_mismatch_lab can
compare them directly.

Output: /tmp/router.json with both traces and identity, plus a
build-step that emits the lab fixture under /tmp/router_input.json.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", "/home/ubuntu/hf-cache")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROLLOUT_MODEL = os.environ.get("MOE_ROLLOUT_MODEL", "Qwen/Qwen3-30B-A3B-FP8")
TRAINER_MODEL = os.environ.get("MOE_TRAINER_MODEL", "Qwen/Qwen3-30B-A3B")
PROMPT_TEXT = os.environ.get(
    "PROMPT_TEXT",
    "List three concrete failure modes a remote rollout worker might exhibit "
    "that an off-policy budget controller could detect.",
)
SEED = int(os.environ.get("SEED", "1234"))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "64"))


def _topk_experts_from_router_logits(
    router_logits_per_layer: list[torch.Tensor],
    prompt_len: int,
    response_len: int,
    top_k: int,
) -> list[list[list[int]]]:
    """Extract top-k expert IDs for the response positions only.

    HF returns one router_logits tensor per MoE layer, with shape
    [num_tokens, num_experts] (flattened across batch). For
    response-token positions we want the predicting positions —
    [P-1 .. P+R-2] for an input of length P+R — so the routing
    decision corresponds to the experts that produced each response
    token.
    """
    out: list[list[list[int]]] = [[] for _ in range(response_len)]
    start = prompt_len - 1
    for r_idx in range(response_len):
        out[r_idx] = []
    for layer_logits in router_logits_per_layer:
        # layer_logits shape: [batch * seq_len, num_experts]; batch=1
        topk = torch.topk(layer_logits, k=top_k, dim=-1).indices.tolist()
        for r_idx in range(response_len):
            t_idx = start + r_idx
            out[r_idx].append([int(x) for x in topk[t_idx]])
    return out


def _trace_for(model_id: str, prompt_ids: list[int], response_ids: list[int]) -> dict:
    """Load model, teacher-force, return router trace + identity."""
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto",
        attn_implementation="sdpa",
        output_router_logits=True,
    )
    model.eval()
    cfg = model.config
    num_layers = sum(
        1
        for layer_decoder in model.model.layers
        if getattr(layer_decoder, "mlp", None) is not None
        and getattr(layer_decoder.mlp, "gate", None) is not None
    )
    num_experts = int(cfg.num_experts)
    top_k = int(cfg.num_experts_per_tok)
    print(f"[moe] {model_id}: layers={num_layers} experts={num_experts} top_k={top_k}", flush=True)
    print(f"[moe] loaded in {time.time()-t0:.1f}s", flush=True)

    P, R = len(prompt_ids), len(response_ids)
    primary_dev = next(model.parameters()).device
    ids = torch.tensor([prompt_ids + response_ids], dtype=torch.long, device=primary_dev)
    with torch.no_grad():
        out = model(input_ids=ids, output_router_logits=True)
    if out.router_logits is None:
        raise RuntimeError(f"{model_id} returned no router_logits")
    expert_ids = _topk_experts_from_router_logits(
        list(out.router_logits), P, R, top_k
    )

    del model
    torch.cuda.empty_cache()

    return {
        "model_id": model_id,
        "num_layers": num_layers,
        "num_experts": num_experts,
        "top_k": top_k,
        "expert_ids": expert_ids,
    }


def main() -> None:
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(TRAINER_MODEL)
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": PROMPT_TEXT}],
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_ids = tok.encode(rendered, add_special_tokens=False)
    print(f"[moe] prompt tokens: {len(prompt_ids)}", flush=True)

    # Stage 1: generate response from the bf16 (full-precision) checkpoint.
    gen_t = time.time()
    gen_model = AutoModelForCausalLM.from_pretrained(
        TRAINER_MODEL,
        torch_dtype="auto",
        device_map="auto",
        attn_implementation="sdpa",
    )
    gen_model.eval()
    primary_dev = next(gen_model.parameters()).device
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=primary_dev)
    torch.manual_seed(SEED)
    with torch.no_grad():
        gen_out = gen_model.generate(
            input_ids=input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
        )
    response_ids = gen_out[0, len(prompt_ids):].tolist()
    response_text = tok.decode(response_ids, skip_special_tokens=True)
    print(f"[moe] generated {len(response_ids)} tokens in {time.time()-gen_t:.1f}s", flush=True)
    print(f"[moe] response: {response_text[:200]}", flush=True)
    del gen_model
    torch.cuda.empty_cache()

    # Stage 2: trainer-side router trace (bf16).
    print("[moe] stage 2: trainer trace (bf16) ...", flush=True)
    trainer_trace = _trace_for(TRAINER_MODEL, prompt_ids, response_ids)

    # Stage 3: rollout-side router trace (fp8).
    print("[moe] stage 3: rollout trace (fp8) ...", flush=True)
    rollout_trace = _trace_for(ROLLOUT_MODEL, prompt_ids, response_ids)

    Path("/tmp/router.json").write_text(json.dumps({
        "rollout_model": ROLLOUT_MODEL,
        "trainer_model": TRAINER_MODEL,
        "prompt_text": PROMPT_TEXT,
        "prompt_token_ids": prompt_ids,
        "response_token_ids": response_ids,
        "response_text": response_text,
        "seed": SEED,
        "rollout_trace": rollout_trace,
        "trainer_trace": trainer_trace,
    }))
    print("[moe] wrote /tmp/router.json", flush=True)
    print(f"[moe] total {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
