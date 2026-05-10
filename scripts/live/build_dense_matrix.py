"""Build dense_mismatch_input.json fixtures for one engine variant
across many prompts.

Reads /tmp/rollouts.{label}.json and /tmp/trainers.{label}.json (pulled
from the spot instance) and emits one fixture per prompt. Each fixture
gets a unique run_id of the form
``live-qwen3-32b-{label}-vs-bf16-hf-p{idx:02d}`` so the dense
dashboard can aggregate them per engine pair.

Pass --label vllm-bf16 / vllm-fp8 / sglang-bf16 / sglang-fp8 etc.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ENGINE_FINGERPRINTS: dict[str, dict] = {
    "vllm-bf16": {
        "rollout_engine_name": "vllm",
        "version": "0.20.1",
        "fingerprint": "sha256:vllm-0.20.1-tp4-bf16-l40s",
        "precision_class": "bf16",
        "quantization_class": None,
        "ckpt_digest": "sha256:hf-cache-Qwen3-32B-bf16-snapshot",
    },
    "vllm-fp8": {
        "rollout_engine_name": "vllm-fp8",
        "version": "0.20.1",
        "fingerprint": "sha256:vllm-0.20.1-tp4-fp8-l40s",
        "precision_class": "fp8",
        "quantization_class": "fp8",
        "ckpt_digest": "sha256:hf-cache-Qwen3-32B-FP8-snapshot",
    },
    "sglang-bf16": {
        "rollout_engine_name": "sglang",
        "version": "0.5.11",
        "fingerprint": "sha256:sglang-0.5.11-tp4-bf16-l40s-cu13",
        "precision_class": "bf16",
        "quantization_class": None,
        "ckpt_digest": "sha256:hf-cache-Qwen3-32B-bf16-snapshot",
    },
    "sglang-fp8": {
        "rollout_engine_name": "sglang-fp8",
        "version": "0.5.11",
        "fingerprint": "sha256:sglang-0.5.11-tp4-fp8-l40s-cu13",
        "precision_class": "fp8",
        "quantization_class": "fp8",
        "ckpt_digest": "sha256:hf-cache-Qwen3-32B-FP8-snapshot",
    },
}


def _config_hash(seed: int) -> str:
    return hashlib.sha256(
        json.dumps(
            {"temperature": 0.7, "top_p": 0.95, "max_tokens": 128, "seed": seed},
            sort_keys=True,
        ).encode()
    ).hexdigest()[:16]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True, choices=list(ENGINE_FINGERPRINTS))
    parser.add_argument("--rollouts", required=True, help="path to rollouts.json")
    parser.add_argument("--trainers", required=True, help="path to trainers.json")
    parser.add_argument("--out-dir", required=True, help="dir to write fixtures into")
    args = parser.parse_args()

    fp = ENGINE_FINGERPRINTS[args.label]
    rollouts = json.loads(Path(args.rollouts).read_text())
    trainers = json.loads(Path(args.trainers).read_text())
    assert len(rollouts) == len(trainers), "rollouts/trainers length mismatch"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for r, t in zip(rollouts, trainers):
        idx = r.get("prompt_idx", 0)
        assert len(r["rollout_logprobs"]) == len(t["trainer_logprobs"])
        run_id = f"live-qwen3-32b-{args.label}-vs-bf16-hf-p{idx:02d}"
        payload = {
            "run_id": run_id,
            "model_id": r["model"],
            "checkpoint_digest": fp["ckpt_digest"],
            "tokenizer_hash": "tok-qwen3",
            "rollout_engine": {
                "name": fp["rollout_engine_name"],
                "version": fp["version"],
                "fingerprint": fp["fingerprint"],
            },
            "trainer_engine": {
                "name": "hf-transformers",
                "version": "5.8.0",
                "fingerprint": "sha256:hf-5.8.0-sdpa-bf16-cuda13",
            },
            "precision_class": fp["precision_class"],
            "quantization_class": fp["quantization_class"],
            "prompt_id": f"p-{idx:02d}",
            "rollout_logprobs": r["rollout_logprobs"],
            "trainer_logprobs": t["trainer_logprobs"],
            "mask": [1] * len(r["rollout_logprobs"]),
            "notes": [
                f"prompt: {r['prompt_text'][:120]!r}",
                f"sampling_config_hash: {_config_hash(r['seed'])}",
            ],
        }
        path = out_dir / f"{run_id}.json"
        path.write_text(json.dumps(payload, indent=2))
        written.append(path)
    print(f"wrote {len(written)} fixtures to {out_dir}")


if __name__ == "__main__":
    main()
