"""Stage 1 (alt): sglang rollout for one prompt; save tokens and sampled-token logprobs.

Mirrors run_vllm_rollout.py but uses sglang's offline Engine. Output
schema is the same /tmp/rollout.json so run_hf_reference.py can be
chained downstream unchanged.

sglang.Engine spawns child processes via Python's `spawn` start method
(not fork). The `if __name__ == "__main__"` guard is required so the
spawned children don't recursively re-launch the engine when they
re-import this module.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", "/home/ubuntu/hf-cache")
import sglang as sgl
from transformers import AutoTokenizer

MODEL = os.environ.get("MODEL", "Qwen/Qwen3-32B")
DTYPE = os.environ.get("SGL_DTYPE", "auto")
ROLLOUT_LABEL = os.environ.get("ROLLOUT_LABEL", "sglang-bf16")
TRAINER_REFERENCE = os.environ.get("TRAINER_REFERENCE", MODEL)
PROMPT_TEXT = os.environ.get(
    "PROMPT_TEXT",
    "Explain in two short sentences why off-policy correction matters in distributed RL.",
)
SEED = int(os.environ.get("SEED", "1234"))


def main() -> None:
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": PROMPT_TEXT}],
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_token_ids = tok.encode(prompt, add_special_tokens=False)
    print(f"[sglang] prompt tokens: {len(prompt_token_ids)}", flush=True)

    engine = sgl.Engine(
        model_path=MODEL,
        tp_size=4,
        dtype=DTYPE,
        mem_fraction_static=0.85,
        context_length=4096,
        random_seed=SEED,
        # Spot host has no CUDA toolkit, so sglang's CUDA graph capture and
        # FlashInfer's JIT compile both fail. Triton attention has its own
        # JIT and works without nvcc. Throughput hit is irrelevant for a
        # one-prompt rollout.
        disable_cuda_graph=True,
        attention_backend="triton",
    )
    print(f"[sglang] loaded in {time.time()-t0:.1f}s", flush=True)

    sampling_params = {
        "temperature": 0.7,
        "top_p": 0.95,
        "max_new_tokens": 128,
    }
    outs = engine.generate(
        [prompt],
        sampling_params,
        return_logprob=True,
        top_logprobs_num=1,
    )
    out = outs[0]
    meta = out["meta_info"]
    output_token_logprobs = meta["output_token_logprobs"]
    response_token_ids = [t[1] for t in output_token_logprobs]
    sampled_logprobs = [float(t[0]) for t in output_token_logprobs]
    response_text = out["text"]
    print(f"[sglang] response tokens: {len(response_token_ids)}", flush=True)
    print(f"[sglang] response text: {response_text[:300]}", flush=True)

    Path("/tmp/rollout.json").write_text(json.dumps({
        "model": MODEL,
        "trainer_reference": TRAINER_REFERENCE,
        "rollout_label": ROLLOUT_LABEL,
        "dtype": DTYPE,
        "prompt_text": PROMPT_TEXT,
        "rendered_prompt": prompt,
        "prompt_token_ids": prompt_token_ids,
        "response_token_ids": response_token_ids,
        "rollout_logprobs": sampled_logprobs,
        "response_text": response_text,
        "seed": SEED,
        "engine": "sglang",
    }))
    print("[sglang] wrote /tmp/rollout.json", flush=True)
    print(f"[sglang] total {time.time()-t0:.1f}s")
    engine.shutdown()


if __name__ == "__main__":
    main()
