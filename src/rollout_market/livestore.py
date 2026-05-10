from __future__ import annotations

from collections import deque

from .contracts import SampleGroup


class InMemoryLiveStore:
    """Small reference store for tests and local demos.

    Production deployments should use Postgres, Redis, Kafka, object storage,
    or Arrow Flight, but the contract should stay the same.
    """

    def __init__(self) -> None:
        self._groups: deque[SampleGroup] = deque()
        self._quarantine: dict[str, SampleGroup] = {}

    def push_group(self, group: SampleGroup) -> None:
        self._groups.append(group)

    def get_batch(self, max_groups: int) -> list[SampleGroup]:
        batch: list[SampleGroup] = []
        while self._groups and len(batch) < max_groups:
            batch.append(self._groups.popleft())
        return batch

    def quarantine_group(self, group: SampleGroup) -> None:
        self._quarantine[group.group_id] = group

    def __len__(self) -> int:
        return len(self._groups)
