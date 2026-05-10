"""Build a dense_mismatch_input.json from the real rollout + trainer logprobs."""
import json
import hashlib
from pathlib import Path

rollout = json.loads(Path("/tmp/rollout.json").read_text())
trainer = json.loads(Path("/tmp/trainer.json").read_text())

assert len(rollout["rollout_logprobs"]) == len(trainer["trainer_logprobs"]), "length mismatch"

config_hash = hashlib.sha256(
    json.dumps({"temperature": 0.7, "top_p": 0.95, "max_tokens": 128, "seed": rollout["seed"]}, sort_keys=True).encode()
).hexdigest()[:16]

payload = {
    "run_id": "live-qwen3-32b-bf16-vllm-vs-hf-001",
    "model_id": rollout["model"],
    "checkpoint_digest": "sha256:hf-cache-Qwen3-32B-bf16-snapshot",
    "tokenizer_hash": "tok-qwen3",
    "rollout_engine": {
        "name": "vllm",
        "version": "0.20.1",
        "fingerprint": "sha256:vllm-0.20.1-tp4-bf16-l40s",
    },
    "trainer_engine": {
        "name": "hf-transformers",
        "version": "5.8.0",
        "fingerprint": "sha256:hf-5.8.0-sdpa-bf16-cuda13",
    },
    "precision_class": "bf16",
    "quantization_class": None,
    "prompt_id": "p-rl-explainer-001",
    "rollout_logprobs": rollout["rollout_logprobs"],
    "trainer_logprobs": trainer["trainer_logprobs"],
    "mask": [1] * len(rollout["rollout_logprobs"]),
    "notes": [
        f"Real Qwen3-32B run, 4xL40S, bf16, seed={rollout['seed']}",
        f"Prompt: {rollout['prompt_text']!r}",
        f"Response (first 80 chars): {rollout['response_text'][:80]!r}",
        f"sampling_config_hash: {config_hash}",
    ],
}
out = Path("/tmp/dense_mismatch_input.json")
out.write_text(json.dumps(payload, indent=2))
print(f"wrote {out} ({out.stat().st_size} bytes)")
print(f"num_tokens: {len(payload['rollout_logprobs'])}")
print(f"first 8 deltas: {[round(t-r, 6) for r, t in zip(payload['rollout_logprobs'][:8], payload['trainer_logprobs'][:8])]}")
