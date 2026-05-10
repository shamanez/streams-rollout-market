"""streams-rollout-market reference package."""

from .contracts import PolicyManifest, SampleGroup, WorkerHeartbeat, BudgetReport, GroupDecision
from .opbc import compute_budget_report, decide_group

__all__ = [
    "PolicyManifest",
    "SampleGroup",
    "WorkerHeartbeat",
    "BudgetReport",
    "GroupDecision",
    "compute_budget_report",
    "decide_group",
]

from .mismatch_metrics import MismatchSummary, summarize_logprob_mismatch

__all__ += ["MismatchSummary", "summarize_logprob_mismatch"]
