"""Build a dense_mismatch_input.json from the FP8 rollout + bf16 HF trainer logprobs."""
import json
import hashlib
from pathlib import Path

rollout = json.loads(Path("/tmp/rollout_fp8.json").read_text())
trainer = json.loads(Path("/tmp/trainer_fp8.json").read_text())
assert len(rollout["rollout_logprobs"]) == len(trainer["trainer_logprobs"])

config_hash = hashlib.sha256(
    json.dumps({"temperature": 0.7, "top_p": 0.95, "max_tokens": 128, "seed": rollout["seed"]}, sort_keys=True).encode()
).hexdigest()[:16]

payload = {
    "run_id": "live-qwen3-32b-fp8-vllm-vs-bf16-hf-001",
    "model_id": rollout["model"],
    "checkpoint_digest": "sha256:hf-cache-Qwen3-32B-FP8-snapshot",
    "tokenizer_hash": "tok-qwen3",
    "rollout_engine": {
        "name": "vllm-fp8",
        "version": "0.20.1",
        "fingerprint": "sha256:vllm-0.20.1-tp4-fp8-l40s",
    },
    "trainer_engine": {
        "name": "hf-transformers",
        "version": "5.8.0",
        "fingerprint": "sha256:hf-5.8.0-sdpa-bf16-cuda13",
    },
    "precision_class": "fp8",
    "quantization_class": "fp8",
    "prompt_id": "p-rl-explainer-001",
    "rollout_logprobs": rollout["rollout_logprobs"],
    "trainer_logprobs": trainer["trainer_logprobs"],
    "mask": [1] * len(rollout["rollout_logprobs"]),
    "notes": [
        f"Real Qwen3-32B-FP8 rollout, 4xL40S, seed={rollout['seed']}",
        f"Trainer reference: {rollout['trainer_reference']} bf16 (full precision)",
        f"sampling_config_hash: {config_hash}",
    ],
}
out = Path("/tmp/dense_mismatch_input_fp8.json")
out.write_text(json.dumps(payload, indent=2))
print(f"wrote {out} ({out.stat().st_size} bytes)")
print(f"first 8 deltas (trainer-rollout): {[round(t-r, 6) for r, t in zip(payload['rollout_logprobs'][:8], payload['trainer_logprobs'][:8])]}")
