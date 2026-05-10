"""Mismatch Observatory experiments.

Tools that probe inference endpoints and compute mismatch telemetry. The
endpoint probe in this package is read-only with respect to providers — it
records contract coverage, not training claims.
"""

from .dense_mismatch_lab import (
    DenseMismatchReport,
    EngineFingerprint,
    compute_dense_mismatch,
    load_input_fixture,
    render_html,
    write_dense_mismatch,
)
from .dense_dashboard import (
    DenseDashboard,
    DenseRow,
    EnginePairAggregate,
)
from .router_dashboard import (
    RouterDashboard,
    RouterEnginePairAggregate,
    RouterRow,
)
from .endpoint_dashboard import (
    EndpointDashboard,
    EndpointRow,
    build_dashboard,
    load_reports_glob,
    write_dashboard,
)
from .endpoint_probe import (
    EndpointContractReport,
    inspect_chat_completion_response,
    probe_endpoint,
    write_report,
)
from .router_mismatch_lab import (
    RouterMismatchReport,
    RouterTrace,
    compute_router_mismatch,
    write_router_mismatch,
)
from .agent_trajectory_lab import (
    AgentStep,
    AgentTrajectory,
    ToolCallRecord,
    TrajectoryDivergenceReport,
    compute_trajectory_divergence,
    load_trajectory,
    load_trajectories_glob,
    write_trajectory_diff,
)
from .agent_dashboard import (
    AgentDashboard,
    AgentEnginePairAggregate,
    AgentRow,
)

__all__ = [
    "EndpointContractReport",
    "inspect_chat_completion_response",
    "probe_endpoint",
    "write_report",
    "DenseMismatchReport",
    "EngineFingerprint",
    "compute_dense_mismatch",
    "load_input_fixture",
    "render_html",
    "write_dense_mismatch",
    "RouterTrace",
    "RouterMismatchReport",
    "compute_router_mismatch",
    "write_router_mismatch",
    "EndpointDashboard",
    "EndpointRow",
    "build_dashboard",
    "load_reports_glob",
    "write_dashboard",
    "DenseDashboard",
    "DenseRow",
    "EnginePairAggregate",
    "RouterDashboard",
    "RouterRow",
    "RouterEnginePairAggregate",
    "AgentStep",
    "AgentTrajectory",
    "ToolCallRecord",
    "TrajectoryDivergenceReport",
    "compute_trajectory_divergence",
    "load_trajectory",
    "load_trajectories_glob",
    "write_trajectory_diff",
    "AgentDashboard",
    "AgentEnginePairAggregate",
    "AgentRow",
]
