"""Build a dense_mismatch_input.json fixture from a (rollout, trainer) pair.

STEER ``cleanup.consolidate_build_dense_input``: the original three
scripts (``build_dense_input.py``, ``build_dense_input_fp8.py``,
``build_dense_input_sglang.py``) collapse into this single env-driven
entry point. Pick a variant with ``--variant``; the script reads
``/tmp/rollout_<variant>.json`` and ``/tmp/trainer_<variant>.json``,
falling back to ``/tmp/rollout.json`` / ``/tmp/trainer.json`` for the
default ``vllm-bf16`` variant to preserve the old call sites.

Usage::

    python scripts/live/build_dense_input.py                       # vllm-bf16
    python scripts/live/build_dense_input.py --variant vllm-fp8
    python scripts/live/build_dense_input.py --variant sglang-bf16
    python scripts/live/build_dense_input.py --variant sglang-fp8
    python scripts/live/build_dense_input.py --variant megatron-bf16

Output lands at ``/tmp/dense_mismatch_input_<variant>.json``; the
``vllm-bf16`` variant additionally writes ``/tmp/dense_mismatch_input.json``
for back-compat with downstream pipelines.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Variant:
    """Static per-variant fixture metadata."""

    label: str
    rollout_default: Path
    trainer_default: Path
    out_default: Path
    legacy_out: Path | None
    run_id: str
    checkpoint_digest: str
    rollout_engine_name: str
    rollout_engine_version: str
    rollout_engine_fingerprint: str
    trainer_engine_name: str
    trainer_engine_version: str
    trainer_engine_fingerprint: str
    precision_class: str
    quantization_class: str | None
    notes_prefix: str


@dataclass(frozen=True)
class TrainerOverride:
    """Optional per-call trainer-engine metadata override.

    The static ``Variant`` table pins a (rollout, trainer) pair, but for
    cross-trainer matrix tiles (e.g. ``vllm-fp8`` rollout paired with an
    ``fsdp`` trainer-side reference instead of the default
    ``hf-transformers``) we want to keep one rollout variant and swap
    the trainer metadata at build time. Pass ``--trainer-label`` to opt
    in; the dashboard ingestion routes the resulting report by
    ``trainer_engine.name``.
    """

    name: str
    version: str
    fingerprint: str


TRAINER_OVERRIDES: dict[str, TrainerOverride] = {
    "hf-transformers": TrainerOverride(
        name="hf-transformers",
        version="5.8.0",
        fingerprint="sha256:hf-5.8.0-sdpa-bf16-cuda13",
    ),
    "fsdp": TrainerOverride(
        name="fsdp",
        version="torch-2.11",
        fingerprint="sha256:fsdp-tp4-fullshard-bf16-cuda13",
    ),
    "megatron-lm": TrainerOverride(
        name="megatron-lm",
        version="core-r0.13",
        fingerprint="sha256:megatron-core-r0.13-bf16-torch-dist",
    ),
}


VARIANTS: dict[str, Variant] = {
    "vllm-bf16": Variant(
        label="vllm-bf16",
        rollout_default=Path("/tmp/rollout_vllm-bf16.json"),
        trainer_default=Path("/tmp/trainer_vllm-bf16.json"),
        out_default=Path("/tmp/dense_mismatch_input_vllm-bf16.json"),
        legacy_out=Path("/tmp/dense_mismatch_input.json"),
        run_id="live-qwen3-32b-bf16-vllm-vs-hf-001",
        checkpoint_digest="sha256:hf-cache-Qwen3-32B-bf16-snapshot",
        rollout_engine_name="vllm",
        rollout_engine_version="0.20.1",
        rollout_engine_fingerprint="sha256:vllm-0.20.1-tp4-bf16-l40s",
        trainer_engine_name="hf-transformers",
        trainer_engine_version="5.8.0",
        trainer_engine_fingerprint="sha256:hf-5.8.0-sdpa-bf16-cuda13",
        precision_class="bf16",
        quantization_class=None,
        notes_prefix="Real Qwen3-32B run, 4xL40S, bf16",
    ),
    "vllm-fp8": Variant(
        label="vllm-fp8",
        rollout_default=Path("/tmp/rollout_vllm-fp8.json"),
        trainer_default=Path("/tmp/trainer_vllm-fp8.json"),
        out_default=Path("/tmp/dense_mismatch_input_vllm-fp8.json"),
        legacy_out=None,
        run_id="live-qwen3-32b-fp8-vllm-vs-bf16-hf-001",
        checkpoint_digest="sha256:hf-cache-Qwen3-32B-FP8-snapshot",
        rollout_engine_name="vllm-fp8",
        rollout_engine_version="0.20.1",
        rollout_engine_fingerprint="sha256:vllm-0.20.1-tp4-fp8-l40s",
        trainer_engine_name="hf-transformers",
        trainer_engine_version="5.8.0",
        trainer_engine_fingerprint="sha256:hf-5.8.0-sdpa-bf16-cuda13",
        precision_class="fp8",
        quantization_class="fp8",
        notes_prefix="Real Qwen3-32B-FP8 rollout, 4xL40S",
    ),
    "sglang-bf16": Variant(
        label="sglang-bf16",
        rollout_default=Path("/tmp/rollout_sglang-bf16.json"),
        trainer_default=Path("/tmp/trainer_sglang-bf16.json"),
        out_default=Path("/tmp/dense_mismatch_input_sglang-bf16.json"),
        legacy_out=None,
        run_id="live-qwen3-32b-bf16-sglang-vs-hf-001",
        checkpoint_digest="sha256:hf-cache-Qwen3-32B-bf16-snapshot",
        rollout_engine_name="sglang",
        rollout_engine_version="0.5.11",
        rollout_engine_fingerprint="sha256:sglang-0.5.11-tp4-bf16-l40s-cu13",
        trainer_engine_name="hf-transformers",
        trainer_engine_version="5.8.0",
        trainer_engine_fingerprint="sha256:hf-5.8.0-sdpa-bf16-cuda13",
        precision_class="bf16",
        quantization_class=None,
        notes_prefix="Real Qwen3-32B bf16 rollout via sglang, 4xL40S",
    ),
    "sglang-fp8": Variant(
        label="sglang-fp8",
        rollout_default=Path("/tmp/rollout_sglang-fp8.json"),
        trainer_default=Path("/tmp/trainer_sglang-fp8.json"),
        out_default=Path("/tmp/dense_mismatch_input_sglang-fp8.json"),
        legacy_out=None,
        run_id="live-qwen3-32b-fp8-sglang-vs-bf16-hf-001",
        checkpoint_digest="sha256:hf-cache-Qwen3-32B-FP8-snapshot",
        rollout_engine_name="sglang-fp8",
        rollout_engine_version="0.5.11",
        rollout_engine_fingerprint="sha256:sglang-0.5.11-tp4-fp8-l40s-cu13",
        trainer_engine_name="hf-transformers",
        trainer_engine_version="5.8.0",
        trainer_engine_fingerprint="sha256:hf-5.8.0-sdpa-bf16-cuda13",
        precision_class="fp8",
        quantization_class="fp8",
        notes_prefix="Real Qwen3-32B fp8 rollout via sglang, 4xL40S",
    ),
    "megatron-bf16": Variant(
        label="megatron-bf16",
        rollout_default=Path("/tmp/rollout_megatron-bf16.json"),
        trainer_default=Path("/tmp/trainer_megatron-bf16.json"),
        out_default=Path("/tmp/dense_mismatch_input_megatron-bf16.json"),
        legacy_out=None,
        run_id="live-qwen3-30b-a3b-bf16-vllm-vs-megatron-001",
        checkpoint_digest="sha256:hf-cache-Qwen3-30B-A3B-bf16-snapshot",
        rollout_engine_name="vllm",
        rollout_engine_version="0.20.1",
        rollout_engine_fingerprint="sha256:vllm-0.20.1-tp4-bf16-l40s",
        trainer_engine_name="megatron-lm",
        trainer_engine_version="core-r0.13",
        trainer_engine_fingerprint="sha256:megatron-core-r0.13-bf16-torch-dist",
        precision_class="bf16",
        quantization_class=None,
        notes_prefix="Qwen3-30B-A3B rollout (vLLM) vs Megatron-LM reference, 4xL40S",
    ),
}


def _resolve_input_paths(variant: Variant) -> tuple[Path, Path]:
    """Resolve the rollout/trainer input paths, with vllm-bf16 fallback."""
    rollout = variant.rollout_default
    trainer = variant.trainer_default
    if variant.label == "vllm-bf16":
        legacy_rollout = Path("/tmp/rollout.json")
        legacy_trainer = Path("/tmp/trainer.json")
        if not rollout.exists() and legacy_rollout.exists():
            rollout = legacy_rollout
        if not trainer.exists() and legacy_trainer.exists():
            trainer = legacy_trainer
    return rollout, trainer


def build_payload(
    variant: Variant,
    rollout: dict,
    trainer: dict,
    trainer_override: TrainerOverride | None = None,
) -> dict:
    """Build the dense-mismatch-input payload for one variant.

    If ``trainer_override`` is supplied, its name/version/fingerprint
    replace the variant's static trainer-engine metadata. The run_id
    gets a ``-vs-<override.name>`` suffix so the dashboard does not
    collide identifiers when the same rollout is paired with multiple
    trainer references.
    """
    if len(rollout["rollout_logprobs"]) != len(trainer["trainer_logprobs"]):
        raise ValueError(
            f"length mismatch: rollout has {len(rollout['rollout_logprobs'])} "
            f"tokens, trainer has {len(trainer['trainer_logprobs'])}"
        )
    config_hash = hashlib.sha256(
        json.dumps(
            {
                "temperature": 0.7,
                "top_p": 0.95,
                "max_tokens": 128,
                "seed": rollout.get("seed"),
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()[:16]
    trainer_ref = rollout.get("trainer_reference", rollout.get("model", ""))
    trainer_name = trainer_override.name if trainer_override else variant.trainer_engine_name
    trainer_version = (
        trainer_override.version if trainer_override else variant.trainer_engine_version
    )
    trainer_fingerprint = (
        trainer_override.fingerprint
        if trainer_override
        else variant.trainer_engine_fingerprint
    )
    run_id = variant.run_id
    if trainer_override and trainer_override.name != variant.trainer_engine_name:
        run_id = f"{run_id}-vs-{trainer_override.name}"
    notes = [
        f"{variant.notes_prefix}, seed={rollout.get('seed')}",
        f"Trainer reference: {trainer_ref} bf16 (full precision)",
        f"sampling_config_hash: {config_hash}",
    ]
    if trainer_override and trainer_override.name != variant.trainer_engine_name:
        notes.append(f"Trainer-side override: {trainer_override.name}")
    return {
        "run_id": run_id,
        "model_id": rollout["model"],
        "checkpoint_digest": variant.checkpoint_digest,
        "tokenizer_hash": "tok-qwen3",
        "rollout_engine": {
            "name": variant.rollout_engine_name,
            "version": variant.rollout_engine_version,
            "fingerprint": variant.rollout_engine_fingerprint,
        },
        "trainer_engine": {
            "name": trainer_name,
            "version": trainer_version,
            "fingerprint": trainer_fingerprint,
        },
        "precision_class": variant.precision_class,
        "quantization_class": variant.quantization_class,
        "prompt_id": "p-rl-explainer-001",
        "rollout_logprobs": rollout["rollout_logprobs"],
        "trainer_logprobs": trainer["trainer_logprobs"],
        "mask": [1] * len(rollout["rollout_logprobs"]),
        "notes": notes,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        choices=sorted(VARIANTS.keys()),
        default="vllm-bf16",
    )
    parser.add_argument(
        "--rollout-input",
        help="override rollout JSON path (default: /tmp/rollout_<variant>.json)",
    )
    parser.add_argument(
        "--trainer-input",
        help="override trainer JSON path (default: /tmp/trainer_<variant>.json)",
    )
    parser.add_argument(
        "--out",
        help="override output path (default: /tmp/dense_mismatch_input_<variant>.json)",
    )
    parser.add_argument(
        "--trainer-label",
        choices=sorted(TRAINER_OVERRIDES.keys()),
        help=(
            "override the variant's static trainer-engine metadata "
            "(e.g. --trainer-label fsdp pairs a vllm-fp8 rollout with "
            "an FSDP-bf16 trainer reference)"
        ),
    )
    args = parser.parse_args(argv)

    variant = VARIANTS[args.variant]
    rollout_path, trainer_path = _resolve_input_paths(variant)
    if args.rollout_input:
        rollout_path = Path(args.rollout_input)
    if args.trainer_input:
        trainer_path = Path(args.trainer_input)
    out_path = Path(args.out) if args.out else variant.out_default

    rollout = json.loads(rollout_path.read_text())
    trainer = json.loads(trainer_path.read_text())
    trainer_override = (
        TRAINER_OVERRIDES[args.trainer_label] if args.trainer_label else None
    )
    payload = build_payload(variant, rollout, trainer, trainer_override=trainer_override)

    out_path.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out_path} ({out_path.stat().st_size} bytes)")
    # vllm-bf16 keeps the legacy /tmp/dense_mismatch_input.json siblings
    # alive so any downstream pipeline still picks it up.
    if variant.legacy_out is not None and args.out is None:
        variant.legacy_out.write_text(json.dumps(payload, indent=2))
        print(f"wrote {variant.legacy_out} (legacy mirror)")
    deltas = [
        round(t - r, 6)
        for r, t in zip(
            payload["rollout_logprobs"][:8],
            payload["trainer_logprobs"][:8],
        )
    ]
    print(f"first 8 deltas (trainer-rollout): {deltas}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
