"""Stage 1: vLLM rollouts for one or more prompts; save tokens and sampled-token logprobs.

PROMPTS_FILE points at a JSON list of strings; if absent, falls back
to a single PROMPT_TEXT. Output is /tmp/rollouts.json — a list of
per-prompt dicts. Single-prompt mode also writes /tmp/rollout.json (a
flat dict matching the older schema) so downstream scripts that read
the single-prompt schema still work.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", "/home/ubuntu/hf-cache")
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

MODEL = os.environ.get("MODEL", "Qwen/Qwen3-32B")
DTYPE = os.environ.get("VLLM_DTYPE", "auto")
ROLLOUT_LABEL = os.environ.get("ROLLOUT_LABEL", "vllm-bf16")
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
    print(f"[vllm] {len(prompts_text)} prompt(s); first tokens: {len(prompt_token_id_lists[0])}", flush=True)

    llm = LLM(
        model=MODEL,
        tensor_parallel_size=4,
        dtype=DTYPE,
        gpu_memory_utilization=0.85,
        max_model_len=4096,
        enforce_eager=False,
        seed=SEED,
    )
    print(f"[vllm] loaded in {time.time()-t0:.1f}s", flush=True)

    sp = SamplingParams(
        temperature=0.7,
        top_p=0.95,
        max_tokens=128,
        logprobs=1,
        seed=SEED,
    )
    t1 = time.time()
    outs = llm.generate(rendered, sp)
    print(f"[vllm] generated in {time.time()-t1:.1f}s", flush=True)

    rollouts = []
    for idx, (text, rp, ptids, completion) in enumerate(zip(prompts_text, rendered, prompt_token_id_lists, outs)):
        out = completion.outputs[0]
        response_token_ids = list(out.token_ids)
        sampled_logprobs = [
            float(out.logprobs[i][response_token_ids[i]].logprob)
            for i in range(len(response_token_ids))
        ]
        response_text = tok.decode(response_token_ids, skip_special_tokens=True)
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
            "engine": "vllm",
        })

    Path("/tmp/rollouts.json").write_text(json.dumps(rollouts))
    if len(rollouts) == 1:
        # Back-compat: single-prompt mode also writes the old flat schema.
        Path("/tmp/rollout.json").write_text(json.dumps(rollouts[0]))
    print(f"[vllm] wrote /tmp/rollouts.json ({len(rollouts)} entries)", flush=True)
    print(f"[vllm] total {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
