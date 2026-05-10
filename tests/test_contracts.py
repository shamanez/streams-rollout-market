import json
from pathlib import Path

from rollout_market.contracts import SampleGroup


def test_minimal_group_validates():
    path = Path(__file__).parents[1] / "examples" / "minimal_grpo_group.json"
    group = SampleGroup.model_validate(json.loads(path.read_text()))
    assert group.policy_version == "v0001"
    assert len(group.trajectories) == 2
