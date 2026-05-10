"""Trainer feedback ingestion.

After a trainer consumes a SampleGroup, it knows three things the dispatcher
side could not: the *observed* ESS and clipped fraction (computed under the
current policy, not the rollout policy), whether the trainer ultimately
included the group in a step, and current-policy logprob statistics on the
same tokens. The TrainerFeedback record carries those numbers back; the
FeedbackAggregator rolls them up by worker_id and engine so a dispatcher
can later prefer engines that have produced trainable rollouts.

This module is firewall-clean: feedback is *reported by* the trainer; this
package never computes a loss or runs an optimizer step.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable

from pydantic import BaseModel, Field, field_validator, model_validator


class TrainerFeedback(BaseModel):
    """One trainer-side report on one consumed SampleGroup.

    Attribution fields (worker_id, engine_name, engine_version,
    policy_version) duplicate fields already on the original SampleGroup.
    Carrying them here means the aggregator does not need to hold a
    reference to the LiveStore — feedback can be ingested out-of-process
    or after the original group has rolled out of the store.
    """

    group_id: str
    worker_id: str
    engine_name: str
    engine_version: str
    policy_version: str

    num_policy_tokens: int
    ess_observed: float
    clipped_fraction_observed: float
    accepted_by_trainer: bool
    trainer_rejection_reason: str | None = None

    current_policy_logprob_mean: float
    current_policy_logprob_std: float = 0.0

    reported_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator(
        "group_id",
        "worker_id",
        "engine_name",
        "engine_version",
        "policy_version",
    )
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("must be a non-empty string")
        return value

    @model_validator(mode="after")
    def _validate(self) -> "TrainerFeedback":
        if self.num_policy_tokens < 0:
            raise ValueError("num_policy_tokens must be non-negative")
        if not 0.0 <= self.ess_observed <= 1.0:
            raise ValueError("ess_observed must be in [0, 1]")
        if not 0.0 <= self.clipped_fraction_observed <= 1.0:
            raise ValueError("clipped_fraction_observed must be in [0, 1]")
        if self.current_policy_logprob_std < 0.0:
            raise ValueError("current_policy_logprob_std must be non-negative")
        if self.accepted_by_trainer and self.trainer_rejection_reason:
            raise ValueError(
                "trainer_rejection_reason is only valid when accepted_by_trainer is False"
            )
        return self


@dataclass
class QualityStats:
    """Running totals for one attribution key (worker_id or engine_name).

    Means are computed on demand to avoid drift across many ingestions; the
    sums and counts are the canonical state.
    """

    count: int = 0
    accepted_count: int = 0
    sum_ess: float = 0.0
    sum_clipped: float = 0.0
    sum_logprob_mean: float = 0.0
    sum_num_policy_tokens: int = 0
    last_reported_at: datetime | None = None
    rejection_reasons: dict[str, int] = field(default_factory=dict)

    @property
    def acceptance_rate(self) -> float:
        return self.accepted_count / self.count if self.count else 0.0

    @property
    def mean_ess(self) -> float:
        return self.sum_ess / self.count if self.count else 0.0

    @property
    def mean_clipped_fraction(self) -> float:
        return self.sum_clipped / self.count if self.count else 0.0

    @property
    def mean_current_policy_logprob(self) -> float:
        return self.sum_logprob_mean / self.count if self.count else 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "count": self.count,
            "accepted_count": self.accepted_count,
            "acceptance_rate": self.acceptance_rate,
            "mean_ess": self.mean_ess,
            "mean_clipped_fraction": self.mean_clipped_fraction,
            "mean_current_policy_logprob": self.mean_current_policy_logprob,
            "sum_num_policy_tokens": self.sum_num_policy_tokens,
            "last_reported_at": (
                self.last_reported_at.isoformat() if self.last_reported_at else None
            ),
            "rejection_reasons": dict(self.rejection_reasons),
        }


def _bump(stats: QualityStats, fb: TrainerFeedback) -> None:
    stats.count += 1
    stats.sum_ess += fb.ess_observed
    stats.sum_clipped += fb.clipped_fraction_observed
    stats.sum_logprob_mean += fb.current_policy_logprob_mean
    stats.sum_num_policy_tokens += fb.num_policy_tokens
    stats.last_reported_at = fb.reported_at
    if fb.accepted_by_trainer:
        stats.accepted_count += 1
    elif fb.trainer_rejection_reason:
        key = fb.trainer_rejection_reason
        stats.rejection_reasons[key] = stats.rejection_reasons.get(key, 0) + 1


class FeedbackAggregator:
    """Rolls up TrainerFeedback into per-worker and per-engine quality stats.

    The aggregator is intentionally a pure data structure: it does not call
    back into the trainer, the dispatcher, or the broker. A dispatcher that
    wants to bias toward higher-quality workers can read aggregator state
    on its own schedule.
    """

    def __init__(self) -> None:
        self._workers: dict[str, QualityStats] = defaultdict(QualityStats)
        self._engines: dict[str, QualityStats] = defaultdict(QualityStats)
        self._policies: dict[str, QualityStats] = defaultdict(QualityStats)
        self._seen: set[str] = set()

    def ingest(self, feedback: TrainerFeedback) -> None:
        """Apply one feedback record to the per-worker and per-engine aggregates.

        Idempotent: a second feedback with the same group_id is ignored —
        the trainer is not allowed to inflate stats by re-reporting.
        """
        if feedback.group_id in self._seen:
            return
        self._seen.add(feedback.group_id)
        _bump(self._workers[feedback.worker_id], feedback)
        _bump(self._engines[feedback.engine_name], feedback)
        _bump(self._policies[feedback.policy_version], feedback)

    def ingest_all(self, feedbacks: Iterable[TrainerFeedback]) -> None:
        for fb in feedbacks:
            self.ingest(fb)

    def stats_for_worker(self, worker_id: str) -> QualityStats:
        return self._workers.get(worker_id, QualityStats())

    def stats_for_engine(self, engine_name: str) -> QualityStats:
        return self._engines.get(engine_name, QualityStats())

    def stats_for_policy(self, policy_version: str) -> QualityStats:
        return self._policies.get(policy_version, QualityStats())

    def all_worker_stats(self) -> dict[str, QualityStats]:
        return dict(self._workers)

    def all_engine_stats(self) -> dict[str, QualityStats]:
        return dict(self._engines)

    def all_policy_stats(self) -> dict[str, QualityStats]:
        return dict(self._policies)

    def __len__(self) -> int:
        return len(self._seen)
