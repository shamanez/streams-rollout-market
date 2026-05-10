from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from .contracts import PolicyManifest, WorkerHeartbeat


@dataclass(frozen=True)
class Assignment:
    assignment_id: str
    worker_id: str
    policy_version: str
    prompt_id: str
    group_size: int


def choose_worker(workers: list[WorkerHeartbeat], manifest: PolicyManifest, max_context_tokens: int) -> WorkerHeartbeat:
    candidates = [
        w for w in workers
        if w.healthy
        and manifest.precision_class in w.precision_classes
        and w.max_context_tokens >= max_context_tokens
    ]
    if not candidates:
        raise RuntimeError("no healthy worker satisfies the policy and context requirements")
    return sorted(candidates, key=lambda w: (w.price_hint_per_hour is None, w.price_hint_per_hour or 0.0))[0]


def make_assignment(worker: WorkerHeartbeat, manifest: PolicyManifest, prompt_id: str, group_size: int) -> Assignment:
    return Assignment(
        assignment_id=str(uuid4()),
        worker_id=worker.worker_id,
        policy_version=manifest.policy_version,
        prompt_id=prompt_id,
        group_size=group_size,
    )
