from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class GroupStatus(str, Enum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    CORRECTABLE = "correctable"
    QUARANTINED = "quarantined"
    REJECTED = "rejected"


class PolicyManifest(BaseModel):
    """Immutable description of a policy snapshot a worker must serve.

    Carries every field a verifier needs to refuse a rollout that drifted from
    the snapshot the trainer expects: tokenizer, weights, precision, quant,
    engine contract, and patch lineage.
    """

    policy_version: str
    checkpoint_digest: str
    tokenizer_hash: str
    model_config_hash: str
    precision_class: str = "bf16"
    quantization_class: str | None = None
    engine_contract_version: str = "v0"
    allowed_engines: list[str] = Field(default_factory=list)
    parent_version: str | None = None
    patch_digest: str | None = None
    patch_lineage: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("policy_version", "checkpoint_digest", "tokenizer_hash", "model_config_hash")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("must be a non-empty string")
        return value


class WorkerManifest(BaseModel):
    """Static capability declaration a worker publishes once on registration.

    Distinct from WorkerHeartbeat: the manifest changes only when the worker
    image, hardware, or supported contracts change, so dispatchers can use it
    as a stable filter when matching policies to workers.
    """

    worker_id: str
    engine_name: str
    engine_version: str
    engine_contract_version: str = "v0"
    device_type: str
    device_count: int = 1
    precision_classes: list[str]
    quantization_classes: list[str] = Field(default_factory=list)
    supported_tokenizer_hashes: list[str] = Field(default_factory=list)
    max_context_tokens: int
    max_concurrent_groups: int = 1
    returns_token_ids: bool = True
    returns_sampled_logprobs: bool = True
    returns_top_logprobs: bool = False
    returns_router_logits: bool = False
    region: str | None = None
    price_hint_per_hour: float | None = None
    announced_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("worker_id", "engine_name", "engine_version", "device_type")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("must be a non-empty string")
        return value

    @field_validator("precision_classes")
    @classmethod
    def _at_least_one_precision(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("worker must declare at least one precision_class")
        return value

    @model_validator(mode="after")
    def _positive_limits(self) -> "WorkerManifest":
        if self.max_context_tokens <= 0:
            raise ValueError("max_context_tokens must be positive")
        if self.device_count <= 0:
            raise ValueError("device_count must be positive")
        if self.max_concurrent_groups <= 0:
            raise ValueError("max_concurrent_groups must be positive")
        return self


class RolloutJob(BaseModel):
    """Pinned rollout request issued before any trajectory begins.

    Carries every field the dispatcher needs to bind a single worker to a
    single policy snapshot for one group of samples: tokenized prompt, policy
    pin, sampling-config hash, group size, engine constraints, and an
    idempotency key so retries do not produce duplicate work.
    """

    job_id: str
    task_id: str
    prompt_id: str
    prompt_token_ids: list[int]
    policy_version: str
    checkpoint_digest: str
    tokenizer_hash: str
    sampling_config_hash: str
    sampling_config: dict[str, Any] = Field(default_factory=dict)
    group_size: int
    required_precision_class: str
    required_quantization_class: str | None = None
    allowed_engines: list[str] = Field(default_factory=list)
    max_context_tokens: int
    idempotency_key: str
    deadline: datetime | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator(
        "job_id",
        "task_id",
        "prompt_id",
        "policy_version",
        "checkpoint_digest",
        "tokenizer_hash",
        "sampling_config_hash",
        "required_precision_class",
        "idempotency_key",
    )
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("must be a non-empty string")
        return value

    @model_validator(mode="after")
    def _validate(self) -> "RolloutJob":
        if self.group_size < 1:
            raise ValueError("group_size must be >= 1")
        if self.max_context_tokens <= 0:
            raise ValueError("max_context_tokens must be positive")
        if not self.prompt_token_ids:
            raise ValueError("prompt_token_ids must be non-empty")
        if self.deadline is not None and self.deadline <= self.created_at:
            raise ValueError("deadline must be strictly after created_at")
        return self


class RolloutLease(BaseModel):
    """Dispatcher-issued ticket binding one worker to one job for a window.

    A lease is the server's promise that exactly one worker may produce the
    group described by `job_id`. It pins the policy snapshot the worker must
    serve and carries the idempotency key from the job so duplicate
    submissions can be deduped without consulting the job table.
    """

    lease_id: str
    job_id: str
    assignment_id: str
    worker_id: str
    policy_version: str
    required_precision_class: str
    required_quantization_class: str | None = None
    max_context_tokens: int
    group_size: int
    idempotency_key: str
    issued_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime

    @field_validator(
        "lease_id",
        "job_id",
        "assignment_id",
        "worker_id",
        "policy_version",
        "required_precision_class",
        "idempotency_key",
    )
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("must be a non-empty string")
        return value

    @model_validator(mode="after")
    def _validate(self) -> "RolloutLease":
        if self.group_size < 1:
            raise ValueError("group_size must be >= 1")
        if self.max_context_tokens <= 0:
            raise ValueError("max_context_tokens must be positive")
        if self.expires_at <= self.issued_at:
            raise ValueError("expires_at must be strictly after issued_at")
        return self

    def is_expired(self, now: datetime | None = None) -> bool:
        """Return True if the lease has expired at the given (UTC) instant."""
        ref = now if now is not None else datetime.now(timezone.utc)
        return ref >= self.expires_at


class WorkerHeartbeat(BaseModel):
    worker_id: str
    engine_name: str
    engine_version: str
    device_type: str
    precision_classes: list[str]
    max_context_tokens: int
    price_hint_per_hour: float | None = None
    current_policy_version: str | None = None
    healthy: bool = True
    last_seen_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Trajectory(BaseModel):
    response_token_ids: list[int]
    rollout_logprobs: list[float]
    action_mask: list[int]
    tool_output_mask: list[int]
    reward: float
    turn_boundaries: list[int] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_lengths(self) -> "Trajectory":
        n = len(self.response_token_ids)
        for name in ("rollout_logprobs", "action_mask", "tool_output_mask"):
            if len(getattr(self, name)) != n:
                raise ValueError(f"{name} length must match response_token_ids")
        return self


class SampleGroup(BaseModel):
    group_id: str
    task_id: str
    prompt_id: str
    worker_id: str
    assignment_id: str
    policy_version: str
    checkpoint_digest: str
    tokenizer_hash: str
    sampling_config_hash: str
    engine_name: str
    engine_version: str
    device_type: str
    precision_class: str
    quantization_class: str | None = None
    prompt_token_ids: list[int]
    trajectories: list[Trajectory]
    policy_lag_steps: int = 0
    wall_clock_seconds: float | None = None
    sample_hash: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def validate_group(self) -> "SampleGroup":
        if not self.trajectories:
            raise ValueError("SampleGroup must include at least one trajectory")
        if self.policy_lag_steps < 0:
            raise ValueError("policy_lag_steps must be non-negative")
        return self


class BudgetReport(BaseModel):
    group_id: str
    mean_weight: float
    std_weight: float
    second_moment: float
    effective_sample_size: float
    clipped_fraction: float
    veto_fraction: float
    max_abs_log_ratio: float
    policy_lag_steps: int
    notes: list[str] = Field(default_factory=list)


class GroupDecision(BaseModel):
    group_id: str
    status: GroupStatus
    reason: str
    budget_report: BudgetReport | None = None
    recommended_action: Literal["train", "train_with_correction", "quarantine", "reject"]
