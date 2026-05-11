"""Build a router_mismatch_input.json from a paired (rollout, trainer) trace.

Sibling of ``build_router_input.py`` (which reads the legacy single-file
``/tmp/router.json`` produced by ``run_moe_router.py``). This script
pairs two separate trace files — typically the vLLM-side
``/tmp/vllm_router.json`` from ``run_vllm_moe_rollout.py`` with a
trainer-side trace from ``run_fsdp_moe_reference.py`` or
``run_moe_router.py``'s extracted trainer trace.

Usage::

    python scripts/live/build_router_input_pair.py \\
        --rollout-input /tmp/vllm_router.json \\
        --trainer-input /tmp/fsdp_router.json \\
        --variant vllm-bf16-fsdp-bf16 \\
        --out /tmp/router_mismatch_input_vllm_bf16_fsdp.json

The trace files must have identical ``(num_tokens, num_layers, top_k,
num_experts)`` axes — RouterTrace rejects mismatched pairs and
``compute_router_mismatch`` would otherwise compare cells that don't
correspond. The ``--variant`` flag picks the rollout/trainer engine
metadata pinned at build time so the router dashboard can route the
report to the right matrix cell.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EngineMeta:
    name: str
    version: str
    fingerprint: str


@dataclass(frozen=True)
class PairVariant:
    label: str
    run_id_prefix: str
    rollout: EngineMeta
    trainer: EngineMeta


VARIANTS: dict[str, PairVariant] = {
    "vllm-bf16-fsdp-bf16": PairVariant(
        label="vllm-bf16-fsdp-bf16",
        run_id_prefix="live-qwen3-30b-a3b-vllm-bf16-vs-fsdp-bf16",
        rollout=EngineMeta(
            name="vllm",
            version="0.20.1",
            fingerprint="sha256:vllm-0.20.1-tp4-bf16-l40s",
        ),
        trainer=EngineMeta(
            name="fsdp",
            version="torch-2.11",
            fingerprint="sha256:fsdp-tp4-fullshard-bf16-cuda13-moe",
        ),
    ),
    "vllm-fp8-fsdp-bf16": PairVariant(
        label="vllm-fp8-fsdp-bf16",
        run_id_prefix="live-qwen3-30b-a3b-vllm-fp8-vs-fsdp-bf16",
        rollout=EngineMeta(
            name="vllm-fp8",
            version="0.20.1",
            fingerprint="sha256:vllm-0.20.1-tp4-fp8-l40s",
        ),
        trainer=EngineMeta(
            name="fsdp",
            version="torch-2.11",
            fingerprint="sha256:fsdp-tp4-fullshard-bf16-cuda13-moe",
        ),
    ),
    "vllm-bf16-megatron-bf16": PairVariant(
        label="vllm-bf16-megatron-bf16",
        run_id_prefix="live-qwen3-30b-a3b-vllm-bf16-vs-megatron-bf16",
        rollout=EngineMeta(
            name="vllm",
            version="0.20.1",
            fingerprint="sha256:vllm-0.20.1-tp4-bf16-l40s",
        ),
        trainer=EngineMeta(
            name="megatron-lm",
            version="core-r0.13",
            fingerprint="sha256:megatron-core-r0.13-bf16-torch-dist",
        ),
    ),
    "vllm-fp8-megatron-bf16": PairVariant(
        label="vllm-fp8-megatron-bf16",
        run_id_prefix="live-qwen3-30b-a3b-vllm-fp8-vs-megatron-bf16",
        rollout=EngineMeta(
            name="vllm-fp8",
            version="0.20.1",
            fingerprint="sha256:vllm-0.20.1-tp4-fp8-l40s",
        ),
        trainer=EngineMeta(
            name="megatron-lm",
            version="core-r0.13",
            fingerprint="sha256:megatron-core-r0.13-bf16-torch-dist",
        ),
    ),
}


def build_payload(
    *,
    rollout: dict,
    trainer: dict,
    variant: PairVariant,
    run_suffix: str = "",
) -> dict:
    """Pair the two trace files into the router-mismatch input payload."""
    for axis in ("num_layers", "num_experts", "top_k"):
        if rollout[axis] != trainer[axis]:
            raise ValueError(
                f"axis mismatch on {axis}: rollout={rollout[axis]} "
                f"trainer={trainer[axis]}"
            )
    if len(rollout["expert_ids"]) != len(trainer["expert_ids"]):
        raise ValueError(
            f"num_tokens mismatch: rollout={len(rollout['expert_ids'])} "
            f"trainer={len(trainer['expert_ids'])}"
        )
    run_id = f"{variant.run_id_prefix}-{run_suffix}" if run_suffix else variant.run_id_prefix
    return {
        "run_id": run_id,
        "model_id": rollout.get("model") or trainer.get("model_id") or trainer.get("model"),
        "rollout_engine": {
            "name": variant.rollout.name,
            "version": variant.rollout.version,
            "fingerprint": variant.rollout.fingerprint,
        },
        "trainer_engine": {
            "name": variant.trainer.name,
            "version": variant.trainer.version,
            "fingerprint": variant.trainer.fingerprint,
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
            f"Rollout: {rollout.get('model')} via "
            f"{rollout.get('engine', variant.rollout.name)}",
            f"Trainer: {trainer.get('model_id') or trainer.get('model')} via "
            f"{trainer.get('engine', variant.trainer.name)}",
            f"Trace length: {len(rollout['expert_ids'])} positions "
            f"(prompt_len + response_len - 1)",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-input", required=True)
    parser.add_argument("--trainer-input", required=True)
    parser.add_argument("--variant", required=True, choices=sorted(VARIANTS.keys()))
    parser.add_argument("--out", required=True)
    parser.add_argument("--run-suffix", default="", help="optional suffix on the run_id")
    args = parser.parse_args(argv)

    rollout = json.loads(Path(args.rollout_input).read_text())
    trainer = json.loads(Path(args.trainer_input).read_text())
    variant = VARIANTS[args.variant]
    payload = build_payload(
        rollout=rollout, trainer=trainer, variant=variant, run_suffix=args.run_suffix
    )

    out_path = Path(args.out)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out_path} ({out_path.stat().st_size} bytes)")
    print(
        f"num_tokens={len(payload['rollout_trace']['expert_ids'])}, "
        f"num_layers={payload['rollout_trace']['num_layers']}, "
        f"top_k={payload['rollout_trace']['top_k']}, "
        f"num_experts={payload['rollout_trace']['num_experts']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
