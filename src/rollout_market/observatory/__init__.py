"""Mismatch Observatory experiments.

Tools that probe inference endpoints and compute mismatch telemetry. The
endpoint probe in this package is read-only with respect to providers — it
records contract coverage, not training claims.
"""

from .endpoint_probe import (
    EndpointContractReport,
    inspect_chat_completion_response,
    probe_endpoint,
    write_report,
)

__all__ = [
    "EndpointContractReport",
    "inspect_chat_completion_response",
    "probe_endpoint",
    "write_report",
]
