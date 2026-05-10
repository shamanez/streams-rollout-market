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
