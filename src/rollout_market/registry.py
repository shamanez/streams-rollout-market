from __future__ import annotations

import json
from pathlib import Path

from .contracts import PolicyManifest


class FilePolicyRegistry:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.latest_file = self.root / "latest.json"

    def publish(self, manifest: PolicyManifest) -> None:
        path = self.root / f"{manifest.policy_version}.json"
        path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        self.latest_file.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

    def latest(self) -> PolicyManifest:
        if not self.latest_file.exists():
            raise FileNotFoundError("no policy manifest has been published")
        return PolicyManifest.model_validate_json(self.latest_file.read_text(encoding="utf-8"))
