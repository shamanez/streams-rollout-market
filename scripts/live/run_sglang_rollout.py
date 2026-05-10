"""Stage 1 (alt): sglang rollouts for one or more prompts.

Mirrors run_vllm_rollout.py — same env vars, same /tmp/rollouts.json
schema. sglang.Engine spawns child processes via spawn (not fork), so
the engine creation is wrapped in `if __name__ == "__main__"` to
prevent recursive launches when the children re-import this module.
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
PROMPTS_FILE = os.environ.get("PROMPTS_FILE")
PROMPT_TEXT = os.environ.get(
    "PROMPT_TEXT",
    "Explain in two short sentences why off-policy correction matters in distributed RL.",
)
SEED = int(os.environ.get("SEED", "1234"))


def _load_prompts() -> list[str]:
    if PROMPTS_FILE:
        return json.loads(Path(PROMPTS_FILE).read_text())
    return [PROMPT_TEXT]


def main() -> None:
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    prompts_text = _load_prompts()
    rendered = []
    prompt_token_id_lists = []
    for p in prompts_text:
        rp = tok.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )
        rendered.append(rp)
        prompt_token_id_lists.append(tok.encode(rp, add_special_tokens=False))
    print(f"[sglang] {len(prompts_text)} prompt(s); first tokens: {len(prompt_token_id_lists[0])}", flush=True)

    engine = sgl.Engine(
        model_path=MODEL,
        tp_size=4,
        dtype=DTYPE,
        mem_fraction_static=0.85,
        context_length=4096,
        random_seed=SEED,
        # The spot host has no system CUDA toolkit; sglang's CUDA-graph
        # capture and FlashInfer JIT both need nvcc. Triton attention
        # has its own JIT and works without nvcc as long as the cu13
        # runtime libs are on the path.
        disable_cuda_graph=True,
        attention_backend="triton",
    )
    print(f"[sglang] loaded in {time.time()-t0:.1f}s", flush=True)

    sampling_params = {
        "temperature": 0.7,
        "top_p": 0.95,
        "max_new_tokens": 128,
    }
    t1 = time.time()
    outs = engine.generate(
        rendered,
        sampling_params,
        return_logprob=True,
        top_logprobs_num=1,
    )
    print(f"[sglang] generated in {time.time()-t1:.1f}s", flush=True)

    rollouts = []
    for idx, (text, rp, ptids, out) in enumerate(zip(prompts_text, rendered, prompt_token_id_lists, outs)):
        meta = out["meta_info"]
        output_token_logprobs = meta["output_token_logprobs"]
        response_token_ids = [t[1] for t in output_token_logprobs]
        sampled_logprobs = [float(t[0]) for t in output_token_logprobs]
        response_text = out["text"]
        rollouts.append({
            "model": MODEL,
            "trainer_reference": TRAINER_REFERENCE,
            "rollout_label": ROLLOUT_LABEL,
            "dtype": DTYPE,
            "prompt_idx": idx,
            "prompt_text": text,
            "rendered_prompt": rp,
            "prompt_token_ids": ptids,
            "response_token_ids": response_token_ids,
            "rollout_logprobs": sampled_logprobs,
            "response_text": response_text,
            "seed": SEED,
            "engine": "sglang",
        })

    Path("/tmp/rollouts.json").write_text(json.dumps(rollouts))
    if len(rollouts) == 1:
        Path("/tmp/rollout.json").write_text(json.dumps(rollouts[0]))
    print(f"[sglang] wrote /tmp/rollouts.json ({len(rollouts)} entries)", flush=True)
    print(f"[sglang] total {time.time()-t0:.1f}s")
    engine.shutdown()


if __name__ == "__main__":
    main()
