from __future__ import annotations

import hashlib
import json

from .contracts import SampleGroup


def canonical_group_hash(group: SampleGroup) -> str:
    payload = group.model_dump(mode="json", exclude={"sample_hash"})
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def verify_group_hash(group: SampleGroup) -> bool:
    if group.sample_hash is None:
        return False
    return group.sample_hash == canonical_group_hash(group)
