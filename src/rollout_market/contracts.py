from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class GroupStatus(str, Enum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    CORRECTABLE = "correctable"
    QUARANTINED = "quarantined"
    REJECTED = "rejected"


class PolicyManifest(BaseModel):
    policy_version: str
    checkpoint_digest: str
    tokenizer_hash: str
    model_config_hash: str
    precision_class: str = "bf16"
    engine_contract_version: str = "v0"
    parent_version: str | None = None
    patch_digest: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


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
