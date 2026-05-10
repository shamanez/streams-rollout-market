"""Build a router_mismatch_input.json from the MoE live run.

Reads /tmp/router.json (produced by run_moe_router.py) and emits the
fixture the router_mismatch_lab CLI consumes.
"""

from __future__ import annotations

import json
from pathlib import Path

router = json.loads(Path("/tmp/router.json").read_text())
trainer = router["trainer_trace"]
rollout = router["rollout_trace"]
assert trainer["num_layers"] == rollout["num_layers"]
assert trainer["num_experts"] == rollout["num_experts"]
assert trainer["top_k"] == rollout["top_k"]
assert len(trainer["expert_ids"]) == len(rollout["expert_ids"])

payload = {
    "run_id": "live-qwen3-30b-a3b-fp8-vs-bf16-001",
    "model_id": trainer["model_id"],
    "rollout_engine": {
        "name": "hf-fp8",
        "version": "5.8.0",
        "fingerprint": "sha256:hf-5.8.0-finegrained-fp8-cu13",
    },
    "trainer_engine": {
        "name": "hf-bf16",
        "version": "5.8.0",
        "fingerprint": "sha256:hf-5.8.0-bf16-cu13",
    },
    "rollout_trace": {
        "num_layers": rollout["num_layers"],
        "num_experts": rollout["num_experts"],
        "top_k": rollout["top_k"],
        "expert_ids": rollout["expert_ids"],
    },
    "trainer_trace": {
        "num_layers": trainer["num_layers"],
        "num_experts": trainer["num_experts"],
        "top_k": trainer["top_k"],
        "expert_ids": trainer["expert_ids"],
    },
    "notes": [
        f"Qwen3-30B-A3B MoE: {trainer['num_layers']} layers, "
        f"{trainer['num_experts']} experts, top_k={trainer['top_k']}",
        f"Rollout side: {router['rollout_model']} (HF finegrained FP8)",
        f"Trainer side: {router['trainer_model']} (HF bf16, sdpa)",
        f"Response tokens: {len(router['response_token_ids'])}",
        f"Prompt: {router['prompt_text'][:120]!r}",
    ],
}
out = Path("/tmp/router_mismatch_input.json")
out.write_text(json.dumps(payload, indent=2))
print(f"wrote {out} ({out.stat().st_size} bytes)")
print(f"num_tokens={len(payload['rollout_trace']['expert_ids'])}, "
      f"num_layers={payload['rollout_trace']['num_layers']}, "
      f"top_k={payload['rollout_trace']['top_k']}")
