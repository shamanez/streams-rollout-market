"""Shape contract for scripts/live/run_megatron_reference.py.

The script's torch / Megatron / slime imports are deferred inside
``main()``; this test loads the script as a module (which is
import-safe) and exercises only the pure-Python helpers
(``response_prediction_positions``, ``format_trainer_payload``,
``_build_megatron_argv``, ``_load_rollouts``).

These four helpers carry the load-bearing invariants:
  - Predicting positions: ``P - 1 + i`` (same as run_hf_reference.py).
  - Output JSON shape matches the run_hf_reference / run_fsdp_reference
    schema so ``build_dense_input.py --variant megatron-bf16`` ingests it.
  - Megatron CLI args include the dense Qwen3-32B MODEL_ARGS plus the
    inference-only knobs (TP=4, --no-load-optim, --bf16).
  - Single-prompt /tmp/rollout.json fallback works.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "live" / "run_megatron_reference.py"


def _load_script_module():
    """Load run_megatron_reference as a module without executing main()."""
    spec = importlib.util.spec_from_file_location(
        "run_megatron_reference", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_response_prediction_positions_matches_hf_reference_indexing() -> None:
    """For a 4-token prompt + 3-token response, positions are [3, 4, 5]."""
    mod = _load_script_module()
    assert mod.response_prediction_positions(4, 3) == [3, 4, 5]
    # Edge case: zero response tokens.
    assert mod.response_prediction_positions(7, 0) == []
    # Edge case: single response token.
    assert mod.response_prediction_positions(1, 1) == [0]


def test_format_trainer_payload_has_run_hf_reference_keys() -> None:
    """The payload must carry the same key shape as run_hf_reference.py."""
    mod = _load_script_module()
    payload = mod.format_trainer_payload(
        model_id="Qwen/Qwen3-32B",
        trainer_logprobs=[-0.1, -0.2, -0.3],
        prompt_idx=0,
        world_size=4,
        tensor_model_parallel_size=4,
        ckpt_dir="/root/Qwen3-32B_torch_dist/",
    )
    for k in ("model", "trainer_logprobs", "engine", "dtype", "prompt_idx"):
        assert k in payload, f"missing key {k} from payload"
    assert payload["engine"] == "megatron-lm"
    assert payload["dtype"] == "bfloat16"
    assert payload["trainer_logprobs"] == [-0.1, -0.2, -0.3]
    assert payload["engine_fingerprint"].startswith("sha256:megatron-")
    assert payload["world_size"] == 4
    assert payload["tensor_model_parallel_size"] == 4


def test_build_megatron_argv_includes_qwen3_32b_model_args(monkeypatch) -> None:
    """sys.argv should carry the Qwen3-32B MODEL_ARGS plus TP knobs."""
    mod = _load_script_module()
    monkeypatch.setenv("TENSOR_MODEL_PARALLEL_SIZE", "4")
    monkeypatch.setenv("MEGATRON_CKPT_DIR", "/root/Qwen3-32B_torch_dist/")
    argv = mod._build_megatron_argv()
    assert argv[0].endswith("run_megatron_reference.py")
    # Inference-only knobs.
    assert "--tensor-model-parallel-size" in argv
    tp_idx = argv.index("--tensor-model-parallel-size")
    assert argv[tp_idx + 1] == "4"
    assert "--bf16" in argv
    assert "--no-load-optim" in argv
    assert "--no-load-rng" in argv
    assert "--exit-on-missing-checkpoint" in argv
    # MODEL_ARGS bits (load-bearing for ckpt compatibility).
    assert "--num-layers" in argv
    layers_idx = argv.index("--num-layers")
    assert argv[layers_idx + 1] == "64"  # dense Qwen3-32B
    assert "--hidden-size" in argv
    hidden_idx = argv.index("--hidden-size")
    assert argv[hidden_idx + 1] == "5120"
    assert "--ffn-hidden-size" in argv
    ffn_idx = argv.index("--ffn-hidden-size")
    assert argv[ffn_idx + 1] == "25600"
    assert "--num-query-groups" in argv  # GQA
    assert "--untie-embeddings-and-output-weights" in argv
    # --load points at the bind-mounted dist-ckpt.
    load_idx = argv.index("--load")
    assert argv[load_idx + 1] == "/root/Qwen3-32B_torch_dist/"


def test_load_rollouts_prefers_rollouts_then_falls_back_to_rollout(
    tmp_path: Path, monkeypatch
) -> None:
    """If /tmp/rollouts.json is missing, fall back to /tmp/rollout.json."""
    mod = _load_script_module()
    primary = tmp_path / "rollouts.json"
    legacy = tmp_path / "rollout.json"
    monkeypatch.setattr(mod, "ROLLOUT_PATH_PRIMARY", primary)
    monkeypatch.setattr(mod, "ROLLOUT_PATH_LEGACY", legacy)

    # 1. Legacy-only: returns a length-1 list wrapping the flat dict.
    legacy.write_text(
        json.dumps(
            {
                "model": "Qwen/Qwen3-32B",
                "prompt_token_ids": [1, 2, 3],
                "response_token_ids": [4, 5],
                "rollout_logprobs": [-0.1, -0.2],
            }
        )
    )
    out = mod._load_rollouts()
    assert isinstance(out, list)
    assert len(out) == 1
    assert out[0]["model"] == "Qwen/Qwen3-32B"

    # 2. Primary present: prefer the list directly.
    primary.write_text(
        json.dumps([
            {"model": "Qwen/Qwen3-32B", "prompt_token_ids": [1], "response_token_ids": [2]},
            {"model": "Qwen/Qwen3-32B", "prompt_token_ids": [3], "response_token_ids": [4]},
        ])
    )
    out = mod._load_rollouts()
    assert len(out) == 2
