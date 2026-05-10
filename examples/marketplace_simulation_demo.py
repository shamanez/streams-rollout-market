"""Phase 6 launch demo: local marketplace simulation.

Runs a small concurrent simulation with a mix of honest, noisy, stale,
and four flavours of toxic workers; aggregates the outcome by worker
profile and decision reason; writes the dashboard JSON+HTML under
runs/<UTC-timestamp>/.

No model is actually run — every group is synthesised. The point of the
demo is to show the contract layer (validators + OPBC + LiveStore +
broker audit) reaching the right verdicts on each profile.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from rollout_market.observatory.marketplace_simulation import (
    WorkerProfile,
    render_html,
    run_simulation,
    write_simulation,
)


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def main() -> None:
    profiles = [
        WorkerProfile.HONEST,
        WorkerProfile.HONEST,
        WorkerProfile.HONEST,
        WorkerProfile.NOISY,
        WorkerProfile.STALE,
        WorkerProfile.TOXIC_TOKENIZER,
        WorkerProfile.TOXIC_PRECISION,
        WorkerProfile.TOXIC_NO_LOGPROBS,
        WorkerProfile.TOXIC_VERSION_DRIFT,
    ]
    result = run_simulation(profiles=profiles, num_jobs=27, seed=1234)

    print(json.dumps(result.as_dict(), indent=2))

    out_dir = Path("runs") / _utc_timestamp()
    paths = write_simulation(result, out_dir)
    print(str(paths["json"]))
    print(str(paths["html"]))
    assert "<!doctype html>" in render_html(result)


if __name__ == "__main__":
    main()
