"""Stage 1: vLLM rollout for one prompt; save tokens and sampled-token logprobs."""
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
PROMPT_TEXT = os.environ.get(
    "PROMPT_TEXT",
    "Explain in two short sentences why off-policy correction matters in distributed RL.",
)
SEED = int(os.environ.get("SEED", "1234"))

t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL)
prompt = tok.apply_chat_template(
    [{"role": "user", "content": PROMPT_TEXT}],
    tokenize=False,
    add_generation_prompt=True,
)
prompt_token_ids = tok.encode(prompt, add_special_tokens=False)
print(f"[vllm] prompt tokens: {len(prompt_token_ids)}", flush=True)

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
outs = llm.generate([prompt], sp)
out = outs[0].outputs[0]
response_token_ids = list(out.token_ids)
sampled_logprobs = []
for tok_id, lp_dict in zip(out.token_ids, out.logprobs):
    entry = lp_dict[tok_id]
    sampled_logprobs.append(float(entry.logprob))

response_text = tok.decode(response_token_ids, skip_special_tokens=True)
print(f"[vllm] response tokens: {len(response_token_ids)}", flush=True)
print(f"[vllm] response text: {response_text[:300]}", flush=True)

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
    "engine": "vllm",
}))
print("[vllm] wrote /tmp/rollout.json", flush=True)
print(f"[vllm] total {time.time()-t0:.1f}s")
