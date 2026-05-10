"""Phase 6 launch demo: marketplace simulation tests.

The simulation is the integration smoke test for everything Phase 0-5
shipped. The assertions here are about *aggregate verdicts*: each worker
profile must end up with the recommended_action the contract intends,
and the LiveStore counts must match the audit log.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rollout_market.observatory.marketplace_simulation import (
    SimulationResult,
    WorkerProfile,
    render_html,
    run_simulation,
    write_simulation,
)


@pytest.fixture
def full_run() -> SimulationResult:
    profiles = [
        WorkerProfile.HONEST,
        WorkerProfile.HONEST,
        WorkerProfile.NOISY,
        WorkerProfile.STALE,
        WorkerProfile.TOXIC_TOKENIZER,
        WorkerProfile.TOXIC_PRECISION,
        WorkerProfile.TOXIC_NO_LOGPROBS,
        WorkerProfile.TOXIC_VERSION_DRIFT,
    ]
    return run_simulation(profiles=profiles, num_jobs=24, seed=42)


def test_simulation_consumes_every_job(full_run: SimulationResult):
    assert full_run.jobs_total == 24
    assert full_run.submissions == 24


def test_honest_workers_train(full_run: SimulationResult):
    honest_actions = {
        a: c for (p, a), c in full_run.by_profile_action.items() if p == "honest"
    }
    assert honest_actions == {"train": pytest.approx(honest_actions.get("train", 0))}
    assert honest_actions.get("train", 0) > 0


def test_noisy_workers_train_with_correction(full_run: SimulationResult):
    noisy_actions = {
        a: c for (p, a), c in full_run.by_profile_action.items() if p == "noisy"
    }
    assert "train_with_correction" in noisy_actions
    assert noisy_actions["train_with_correction"] > 0


def test_stale_workers_quarantined(full_run: SimulationResult):
    stale_actions = {
        a: c for (p, a), c in full_run.by_profile_action.items() if p == "stale"
    }
    assert stale_actions == {"quarantine": stale_actions.get("quarantine", 0)}
    assert stale_actions.get("quarantine", 0) > 0


@pytest.mark.parametrize(
    "profile,expected_reason",
    [
        ("toxic_tokenizer", "tokenizer_mismatch"),
        ("toxic_precision", "precision_mismatch"),
        ("toxic_no_logprobs", "missing_logprobs"),
        ("toxic_version_drift", "policy_version_mismatch"),
    ],
)
def test_toxic_profiles_rejected_with_typed_reason(
    full_run: SimulationResult, profile: str, expected_reason: str
):
    actions = {
        a: c for (p, a), c in full_run.by_profile_action.items() if p == profile
    }
    assert actions == {"reject": actions.get("reject", 0)}
    assert actions["reject"] > 0
    assert full_run.rejection_reasons.get(expected_reason, 0) > 0


def test_full_action_surface_is_exercised(full_run: SimulationResult):
    actions = {a for (_, a), _ in full_run.by_profile_action.items()}
    assert {"train", "train_with_correction", "quarantine", "reject"}.issubset(actions)


def test_livestore_state_counts_match_decisions(full_run: SimulationResult):
    by_action: dict[str, int] = {}
    for (_, action), count in full_run.by_profile_action.items():
        by_action[action] = by_action.get(action, 0) + count
    assert full_run.livestore_by_state.get("rejected", 0) == by_action.get("reject", 0)
    assert full_run.livestore_by_state.get("quarantined", 0) == by_action.get(
        "quarantine", 0
    )
    assert full_run.livestore_by_state.get("correctable", 0) == by_action.get(
        "train_with_correction", 0
    )
    accepted_actions = {"train", "replay"}
    accepted = sum(c for a, c in by_action.items() if a in accepted_actions)
    assert full_run.livestore_by_state.get("accepted", 0) == accepted


def test_audit_log_records_lifecycle(full_run: SimulationResult):
    counts = full_run.audit_event_counts
    assert counts.get("worker_registered", 0) > 0
    assert counts.get("job_enqueued", 0) == full_run.jobs_total
    assert counts.get("lease_issued", 0) == full_run.submissions
    assert counts.get("lease_submitted", 0) == full_run.submissions


def test_jobs_distributed_across_multiple_workers(full_run: SimulationResult):
    served_workers = {wid for (wid, _), c in full_run.by_worker_action.items() if c > 0}
    assert len(served_workers) >= 4


def test_render_html_is_self_contained(full_run: SimulationResult):
    page = render_html(full_run)
    assert page.startswith("<!doctype html>")
    assert "</html>" in page.lower()
    assert "Marketplace simulation" in page
    assert "rejection_reasons" not in page  # we render headings, not raw keys
    assert "Rejection reasons" in page


def test_write_simulation_creates_both_files(full_run: SimulationResult, tmp_path: Path):
    out = tmp_path / "runs" / "demo"
    paths = write_simulation(full_run, out)
    assert paths["json"].exists() and paths["html"].exists()
    payload = json.loads(paths["json"].read_text())
    assert payload["jobs_total"] == 24
    assert "rejection_reasons" in payload
    assert "decision_reasons" in payload


def test_minimal_run_with_only_honest_workers():
    result = run_simulation(profiles=[WorkerProfile.HONEST], num_jobs=3, seed=0)
    assert result.submissions == 3
    assert result.livestore_by_state.get("accepted", 0) == 3
    assert result.livestore_by_state.get("rejected", 0) == 0
