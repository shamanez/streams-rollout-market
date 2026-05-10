import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from rollout_market.contracts import RolloutJob, RolloutLease

EXAMPLES = Path(__file__).parents[1] / "examples"


def _job_payload() -> dict:
    return json.loads((EXAMPLES / "minimal_rollout_job.json").read_text())


def _lease_payload() -> dict:
    return json.loads((EXAMPLES / "minimal_rollout_lease.json").read_text())


def test_rollout_job_fixture_validates():
    job = RolloutJob.model_validate(_job_payload())
    assert job.group_size == 8
    assert job.required_precision_class == "bf16"
    assert job.idempotency_key == "idem-job-0001"
    assert job.sampling_config["seed"] == 42


def test_rollout_job_rejects_zero_group_size():
    payload = _job_payload()
    payload["group_size"] = 0
    with pytest.raises(ValidationError):
        RolloutJob.model_validate(payload)


def test_rollout_job_rejects_blank_idempotency_key():
    payload = _job_payload()
    payload["idempotency_key"] = "   "
    with pytest.raises(ValidationError):
        RolloutJob.model_validate(payload)


def test_rollout_job_rejects_empty_prompt_tokens():
    payload = _job_payload()
    payload["prompt_token_ids"] = []
    with pytest.raises(ValidationError):
        RolloutJob.model_validate(payload)


def test_rollout_job_rejects_deadline_before_created_at():
    payload = _job_payload()
    payload["created_at"] = "2026-05-10T00:00:00+00:00"
    payload["deadline"] = "2026-05-09T00:00:00+00:00"
    with pytest.raises(ValidationError):
        RolloutJob.model_validate(payload)


def test_rollout_job_missing_policy_version_fails():
    payload = _job_payload()
    payload.pop("policy_version")
    with pytest.raises(ValidationError):
        RolloutJob.model_validate(payload)


def test_rollout_lease_fixture_validates():
    lease = RolloutLease.model_validate(_lease_payload())
    assert lease.group_size == 8
    assert lease.policy_version == "v0001"
    assert lease.idempotency_key == "idem-job-0001"


def test_rollout_lease_expiry_must_be_after_issued():
    payload = _lease_payload()
    payload["issued_at"] = "2026-05-10T00:05:00+00:00"
    payload["expires_at"] = "2026-05-10T00:00:00+00:00"
    with pytest.raises(ValidationError):
        RolloutLease.model_validate(payload)


def test_rollout_lease_equal_issued_and_expires_fails():
    payload = _lease_payload()
    payload["issued_at"] = "2026-05-10T00:00:00+00:00"
    payload["expires_at"] = "2026-05-10T00:00:00+00:00"
    with pytest.raises(ValidationError):
        RolloutLease.model_validate(payload)


def test_rollout_lease_is_expired_helper():
    issued = datetime(2026, 5, 10, tzinfo=timezone.utc)
    expires = issued + timedelta(minutes=5)
    lease = RolloutLease.model_validate(
        {
            **_lease_payload(),
            "issued_at": issued.isoformat(),
            "expires_at": expires.isoformat(),
        }
    )
    assert lease.is_expired(issued + timedelta(minutes=1)) is False
    assert lease.is_expired(expires) is True
    assert lease.is_expired(expires + timedelta(seconds=1)) is True


def test_rollout_lease_rejects_zero_group_size():
    payload = _lease_payload()
    payload["group_size"] = 0
    with pytest.raises(ValidationError):
        RolloutLease.model_validate(payload)


def test_rollout_lease_rejects_zero_context():
    payload = _lease_payload()
    payload["max_context_tokens"] = 0
    with pytest.raises(ValidationError):
        RolloutLease.model_validate(payload)


def test_rollout_lease_blank_worker_id_fails():
    payload = _lease_payload()
    payload["worker_id"] = ""
    with pytest.raises(ValidationError):
        RolloutLease.model_validate(payload)


def test_rollout_lease_propagates_idempotency_key_from_job():
    job = RolloutJob.model_validate(_job_payload())
    lease = RolloutLease.model_validate(_lease_payload())
    assert lease.idempotency_key == job.idempotency_key
    assert lease.policy_version == job.policy_version
