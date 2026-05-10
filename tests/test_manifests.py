import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from rollout_market.contracts import PolicyManifest, WorkerManifest

EXAMPLES = Path(__file__).parents[1] / "examples"


def _policy_payload() -> dict:
    return json.loads((EXAMPLES / "minimal_policy_manifest.json").read_text())


def _worker_payload() -> dict:
    return json.loads((EXAMPLES / "minimal_worker_manifest.json").read_text())


def test_policy_manifest_fixture_validates():
    manifest = PolicyManifest.model_validate(_policy_payload())
    assert manifest.policy_version == "v0001"
    assert manifest.allowed_engines == ["vllm", "sglang"]
    assert manifest.patch_lineage == ["v0000"]
    assert manifest.quantization_class is None


def test_policy_manifest_missing_tokenizer_hash_fails():
    payload = _policy_payload()
    payload.pop("tokenizer_hash")
    with pytest.raises(ValidationError):
        PolicyManifest.model_validate(payload)


def test_policy_manifest_empty_tokenizer_hash_fails():
    payload = _policy_payload()
    payload["tokenizer_hash"] = ""
    with pytest.raises(ValidationError):
        PolicyManifest.model_validate(payload)


def test_policy_manifest_missing_checkpoint_digest_fails():
    payload = _policy_payload()
    payload.pop("checkpoint_digest")
    with pytest.raises(ValidationError):
        PolicyManifest.model_validate(payload)


def test_policy_manifest_blank_checkpoint_digest_fails():
    payload = _policy_payload()
    payload["checkpoint_digest"] = "   "
    with pytest.raises(ValidationError):
        PolicyManifest.model_validate(payload)


def test_policy_manifest_defaults_for_optional_fields():
    payload = {
        "policy_version": "v0002",
        "checkpoint_digest": "sha256:deadbeef",
        "tokenizer_hash": "tok-x",
        "model_config_hash": "cfg-x",
    }
    manifest = PolicyManifest.model_validate(payload)
    assert manifest.precision_class == "bf16"
    assert manifest.quantization_class is None
    assert manifest.allowed_engines == []
    assert manifest.patch_lineage == []
    assert manifest.engine_contract_version == "v0"


def test_worker_manifest_fixture_validates():
    manifest = WorkerManifest.model_validate(_worker_payload())
    assert manifest.worker_id == "worker-l40s-0"
    assert manifest.device_count == 4
    assert "tok-qwen3-32b" in manifest.supported_tokenizer_hashes
    assert manifest.returns_top_logprobs is True


def test_worker_manifest_requires_precision_class():
    payload = _worker_payload()
    payload["precision_classes"] = []
    with pytest.raises(ValidationError):
        WorkerManifest.model_validate(payload)


def test_worker_manifest_rejects_zero_context():
    payload = _worker_payload()
    payload["max_context_tokens"] = 0
    with pytest.raises(ValidationError):
        WorkerManifest.model_validate(payload)


def test_worker_manifest_rejects_zero_device_count():
    payload = _worker_payload()
    payload["device_count"] = 0
    with pytest.raises(ValidationError):
        WorkerManifest.model_validate(payload)


def test_worker_manifest_blank_worker_id_fails():
    payload = _worker_payload()
    payload["worker_id"] = ""
    with pytest.raises(ValidationError):
        WorkerManifest.model_validate(payload)


def test_worker_manifest_round_trip_json():
    manifest = WorkerManifest.model_validate(_worker_payload())
    again = WorkerManifest.model_validate_json(manifest.model_dump_json())
    assert again == manifest
