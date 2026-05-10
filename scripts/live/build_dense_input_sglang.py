"""Build dense_mismatch_input.json fixtures from sglang rollouts.

Reads /tmp/rollout_sglang_{bf16,fp8}.json and matching trainer files
and emits two fixtures the dense_mismatch_lab CLI can consume.
"""

import hashlib
import json
from pathlib import Path

VARIANTS = [
    {
        "label": "bf16",
        "rollout_path": "/tmp/rollout_sglang_bf16.json",
        "trainer_path": "/tmp/trainer_sglang_bf16.json",
        "out": "/tmp/dense_mismatch_input_sglang_bf16.json",
        "rollout_engine_name": "sglang",
        "rollout_engine_fp": "sha256:sglang-0.5.11-tp4-bf16-l40s-cu13",
        "precision_class": "bf16",
        "quantization_class": None,
        "ckpt_digest": "sha256:hf-cache-Qwen3-32B-bf16-snapshot",
        "run_id": "live-qwen3-32b-bf16-sglang-vs-hf-001",
    },
    {
        "label": "fp8",
        "rollout_path": "/tmp/rollout_sglang_fp8.json",
        "trainer_path": "/tmp/trainer_sglang_fp8.json",
        "out": "/tmp/dense_mismatch_input_sglang_fp8.json",
        "rollout_engine_name": "sglang-fp8",
        "rollout_engine_fp": "sha256:sglang-0.5.11-tp4-fp8-l40s-cu13",
        "precision_class": "fp8",
        "quantization_class": "fp8",
        "ckpt_digest": "sha256:hf-cache-Qwen3-32B-FP8-snapshot",
        "run_id": "live-qwen3-32b-fp8-sglang-vs-bf16-hf-001",
    },
]

for v in VARIANTS:
    rollout = json.loads(Path(v["rollout_path"]).read_text())
    trainer = json.loads(Path(v["trainer_path"]).read_text())
    assert len(rollout["rollout_logprobs"]) == len(trainer["trainer_logprobs"])
    config_hash = hashlib.sha256(
        json.dumps(
            {"temperature": 0.7, "top_p": 0.95, "max_tokens": 128, "seed": rollout["seed"]},
            sort_keys=True,
        ).encode()
    ).hexdigest()[:16]
    payload = {
        "run_id": v["run_id"],
        "model_id": rollout["model"],
        "checkpoint_digest": v["ckpt_digest"],
        "tokenizer_hash": "tok-qwen3",
        "rollout_engine": {
            "name": v["rollout_engine_name"],
            "version": "0.5.11",
            "fingerprint": v["rollout_engine_fp"],
        },
        "trainer_engine": {
            "name": "hf-transformers",
            "version": "5.8.0",
            "fingerprint": "sha256:hf-5.8.0-sdpa-bf16-cuda13",
        },
        "precision_class": v["precision_class"],
        "quantization_class": v["quantization_class"],
        "prompt_id": "p-rl-explainer-001",
        "rollout_logprobs": rollout["rollout_logprobs"],
        "trainer_logprobs": trainer["trainer_logprobs"],
        "mask": [1] * len(rollout["rollout_logprobs"]),
        "notes": [
            f"Real Qwen3-32B {v['label']} rollout via sglang, 4xL40S, seed={rollout['seed']}",
            f"Trainer reference: {rollout['trainer_reference']} bf16",
            f"sampling_config_hash: {config_hash}",
        ],
    }
    Path(v["out"]).write_text(json.dumps(payload, indent=2))
    print(f"wrote {v['out']} ({Path(v['out']).stat().st_size} bytes)")
    deltas = [round(t - r, 6) for r, t in zip(payload['rollout_logprobs'][:8], payload['trainer_logprobs'][:8])]
    print(f"  first 8 deltas: {deltas}")
