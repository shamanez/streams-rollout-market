"""streams-rollout-market reference package."""

from .contracts import (
    BudgetReport,
    GroupDecision,
    GroupStatus,
    PolicyManifest,
    RejectionReason,
    RolloutJob,
    RolloutLease,
    SampleGroup,
    WorkerHeartbeat,
    WorkerManifest,
)
from .opbc import compute_budget_report, decide_group
from .validators import validate_group_against_lease

__all__ = [
    "PolicyManifest",
    "SampleGroup",
    "WorkerHeartbeat",
    "WorkerManifest",
    "RolloutJob",
    "RolloutLease",
    "BudgetReport",
    "GroupDecision",
    "GroupStatus",
    "RejectionReason",
    "compute_budget_report",
    "decide_group",
    "validate_group_against_lease",
]

from .mismatch_metrics import MismatchSummary, summarize_logprob_mismatch

__all__ += ["MismatchSummary", "summarize_logprob_mismatch"]
