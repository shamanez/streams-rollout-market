from pathlib import Path
import json

from rollout_market.contracts import SampleGroup
from rollout_market.livestore import InMemoryLiveStore
from rollout_market.opbc import compute_budget_report, decide_group


data = json.loads(Path(__file__).with_name("minimal_grpo_group.json").read_text())
group = SampleGroup.model_validate(data)

# In a real trainer, these are recomputed by the train engine for the same valid policy tokens.
train_logprobs = [-0.31, -0.48, -0.22, -0.42, -0.39, -0.33]

report = compute_budget_report(group, train_logprobs)
decision = decide_group(report)

store = InMemoryLiveStore()
if decision.recommended_action in {"train", "train_with_correction"}:
    store.push_group(group)
else:
    store.quarantine_group(group)

print(report.model_dump_json(indent=2))
print(decision.model_dump_json(indent=2))
print(f"livestore_size={len(store)}")
