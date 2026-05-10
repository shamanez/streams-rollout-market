"""streams-rollout-market reference package."""

from .contracts import (
    BudgetReport,
    DecisionReason,
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
from .livestore import (
    DuplicateGroupError,
    InMemoryLiveStore,
    StoredGroup,
    UnknownGroupError,
)
from .feedback import FeedbackAggregator, QualityStats, TrainerFeedback
from .opbc import compute_budget_report, decide_group
from .trainer_client import ReplayTier, TrainerBatch, TrainerClient
from .validators import validate_group_against_lease
from .worker import WorkerSDK
from .worker_broker import LeaseRecord, LeaseState, LocalWorkerBroker

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
    "DecisionReason",
    "InMemoryLiveStore",
    "StoredGroup",
    "DuplicateGroupError",
    "UnknownGroupError",
    "TrainerClient",
    "TrainerBatch",
    "ReplayTier",
    "WorkerSDK",
    "LocalWorkerBroker",
    "LeaseRecord",
    "LeaseState",
    "TrainerFeedback",
    "FeedbackAggregator",
    "QualityStats",
    "compute_budget_report",
    "decide_group",
    "validate_group_against_lease",
]

from .mismatch_metrics import MismatchSummary, summarize_logprob_mismatch

__all__ += ["MismatchSummary", "summarize_logprob_mismatch"]
