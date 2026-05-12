import json
from pathlib import Path

from rollout_market.contracts import SampleGroup
from rollout_market.opbc import compute_budget_report, decide_group


def test_budget_report_accepts_small_mismatch():
    path = Path(__file__).parents[1] / "examples" / "minimal_grpo_group.json"
    group = SampleGroup.model_validate(json.loads(path.read_text()))
    report = compute_budget_report(group, [-0.31, -0.48, -0.22, -0.42, -0.39, -0.33])
    decision = decide_group(report)
    assert report.effective_sample_size > 0.9
    assert decision.recommended_action == "train"


def test_budget_report_threads_router_flip_rate():
    """The optional router_flip_rate kwarg should land on the resulting
    BudgetReport so decide_group's MoE gate can read it."""
    path = Path(__file__).parents[1] / "examples" / "minimal_grpo_group.json"
    group = SampleGroup.model_validate(json.loads(path.read_text()))
    report_with = compute_budget_report(
        group,
        [-0.31, -0.48, -0.22, -0.42, -0.39, -0.33],
        router_flip_rate=0.07,
    )
    assert report_with.router_flip_rate == 0.07
    # Default is None (Dense rollouts skip the gate).
    report_without = compute_budget_report(
        group, [-0.31, -0.48, -0.22, -0.42, -0.39, -0.33]
    )
    assert report_without.router_flip_rate is None


def test_budget_report_quarantines_on_high_flip_rate():
    """End-to-end: a high router_flip_rate on a Dense-ESS-clean group
    still quarantines via the new gate."""
    path = Path(__file__).parents[1] / "examples" / "minimal_grpo_group.json"
    group = SampleGroup.model_validate(json.loads(path.read_text()))
    report = compute_budget_report(
        group,
        [-0.31, -0.48, -0.22, -0.42, -0.39, -0.33],
        router_flip_rate=0.40,  # well above 15% threshold
    )
    decision = decide_group(report)
    assert decision.recommended_action == "quarantine"
