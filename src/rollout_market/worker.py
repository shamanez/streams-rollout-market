from __future__ import annotations

from .contracts import SampleGroup, WorkerHeartbeat


class WorkerRuntime:
    """Skeleton for a remote rollout worker.

    A real worker should:
    1. Pull a PolicyManifest.
    2. Load or patch the inference engine.
    3. ACK readiness for the pinned policy version.
    4. Run the environment loop.
    5. Submit SampleGroup objects with token IDs and rollout logprobs.
    """

    def __init__(self, heartbeat: WorkerHeartbeat) -> None:
        self.heartbeat = heartbeat

    def submit(self, group: SampleGroup) -> SampleGroup:
        # Placeholder hook for signing, compression, and upload.
        return group
