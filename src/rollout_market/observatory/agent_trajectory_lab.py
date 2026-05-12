"""Agent-trajectory observatory.

Long-horizon, tool-using rollouts surface a different kind of mismatch than
dense token-level logprob deltas. A 1.4× single-token importance-weight
ratio is invisible at the trajectory level until it nudges the agent into
a different *tool call* on step k, at which point the rest of the
trajectory rolls off in a different direction. This module captures that.

Schema:
  AgentStep         — one step in an agent run (assistant turn or tool result)
  AgentTrajectory   — full run for one (engine, task) pair
  ToolCallRecord    — one tool invocation made within a step
  TrajectoryDivergenceReport — pairwise comparison of two AgentTrajectories

The lab is firewall-clean: no PPO/GRPO/loss/optimizer code. It only
records what an agent did and computes set-level / sequence-level
agreement between two such recordings.
"""

from __future__ import annotations

import hashlib
import html as _html
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from ._device import DeviceFingerprint
from .dense_mismatch_lab import EngineFingerprint


def _hash(payload: str | dict | list) -> str:
    if not isinstance(payload, str):
        payload = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class ToolCallRecord(BaseModel):
    """One tool invocation captured within an assistant step.

    arguments_hash and result_hash let two trajectories compare tool
    invocations cheaply (same tool, same args -> same hash) without
    having to dump the full args/result text into every metric.
    """

    name: str
    arguments_json: str
    arguments_hash: str
    result_text: str
    result_hash: str

    @classmethod
    def from_call(
        cls, *, name: str, arguments: dict | str, result: str
    ) -> "ToolCallRecord":
        args_str = arguments if isinstance(arguments, str) else json.dumps(
            arguments, sort_keys=True
        )
        return cls(
            name=name,
            arguments_json=args_str,
            arguments_hash=_hash(args_str),
            result_text=result,
            result_hash=_hash(result),
        )


class AgentStep(BaseModel):
    """One step in an agent trajectory.

    role="assistant" carries the model's decision (content + optional
    tool_calls). role="tool" carries the result of one tool invocation
    fed back into the conversation.
    """

    step_idx: int
    role: Literal["assistant", "tool"]
    content: str = ""
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    finish_reason: str | None = None

    # Token-level data is optional because chat-completions APIs vary
    # in what they expose. Empty lists are fine.
    response_token_count: int = 0
    response_logprobs_hash: str | None = None

    @model_validator(mode="after")
    def _validate(self) -> "AgentStep":
        if self.role == "tool" and self.tool_calls:
            raise ValueError("tool-role steps cannot themselves emit tool_calls")
        return self


class AgentTrajectory(BaseModel):
    """Full agent run for one (model, engine, task) triple.

    Two trajectories with the same task_id and the same model_id but
    different rollout_engine fingerprints are the comparison unit the
    divergence metrics operate on.
    """

    schema_version: Literal["agent_trajectory.v0"] = "agent_trajectory.v0"
    run_id: str
    task_id: str
    task_text: str
    model_id: str
    rollout_engine: EngineFingerprint
    available_tools: list[str]
    steps: list[AgentStep]
    success: bool | None = None
    terminal_reason: Literal["answered", "max_steps", "error", "no_op"] = "answered"
    total_assistant_tokens: int = 0
    final_answer: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def assistant_steps(self) -> list[AgentStep]:
        return [s for s in self.steps if s.role == "assistant"]

    def tool_call_signature(self) -> list[tuple[str, str]]:
        """List of (tool_name, arguments_hash) pairs in invocation order."""
        sig: list[tuple[str, str]] = []
        for step in self.steps:
            for call in step.tool_calls:
                sig.append((call.name, call.arguments_hash))
        return sig


class TrajectoryDivergenceReport(BaseModel):
    """Pairwise comparison of one rollout-side trajectory against a trainer-side one.

    rollout = the engine under test (e.g. vllm-fp8, sglang-bf16).
    trainer = the reference trajectory (typically vllm-bf16 or hf-bf16).
    """

    schema_version: Literal["agent_divergence.v0"] = "agent_divergence.v0"
    run_id: str
    task_id: str
    task_text: str
    model_id: str
    rollout_engine: EngineFingerprint
    trainer_engine: EngineFingerprint

    rollout_step_count: int
    trainer_step_count: int
    rollout_assistant_steps: int
    trainer_assistant_steps: int

    first_divergence_step: int | None
    first_divergence_kind: (
        Literal["tool_choice", "tool_args", "content", "termination"] | None
    )

    tool_choice_disagreement_rate: float
    tool_call_jaccard: float
    final_answer_match: bool
    final_answer_match_soft: bool = False
    rollout_terminal_reason: str
    trainer_terminal_reason: str

    same_step_count: bool
    rollout_tool_signature: list[list[str]]
    trainer_tool_signature: list[list[str]]
    device: DeviceFingerprint | None = None
    notes: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


def _normalize(text: str) -> str:
    return " ".join(text.strip().split()).lower()


def _content_matches(a: str, b: str) -> bool:
    if not a and not b:
        return True
    return _normalize(a) == _normalize(b)


# Numeric tokens: signed/unsigned integers + decimals + commas (e.g.
# "8,294,400", "-2.5", "120"). The trailing comma/period inside the
# bracket lets us match "8,294,400" as one group; we strip non-digit
# bookends afterwards.
_NUM_RE = re.compile(r"[-+]?\d[\d,.\-]*\d|\d")


def _extract_numbers(text: str) -> set[str]:
    """Return canonicalised numeric tokens found in ``text``.

    Strips commas + trailing zeros so 1,200,000 and 1200000 and 1200000.0
    all map to the same canonical key. Returns an empty set if nothing
    parses.
    """
    out: set[str] = set()
    for m in _NUM_RE.findall(text):
        s = m.replace(",", "").strip("-.")
        if not s:
            continue
        try:
            f = float(s)
        except ValueError:
            continue
        if f == int(f):
            out.add(str(int(f)))
        else:
            # Normalize "1.20" -> "1.2"
            out.add(f"{f:g}")
    return out


def _soft_content_match(a: str, b: str) -> bool:
    """Looser equality for multi-turn agent final answers.

    Hermes Agent (and Qwen3 in general) produces answers like
    "The total is **8,294,400**." vs "The total number is 8294400."
    that carry the *same load-bearing number* with different surface
    text. The strict ``_content_matches`` returns False; this softer
    check returns True when:

      - both texts are empty (trivial agreement), OR
      - the strict matcher returns True, OR
      - the set of numeric tokens in each text is non-empty and
        equal (the surface phrasing differs but the numbers don't),
        OR
      - one normalised text is a proper substring of the other (the
        longer answer contains the shorter as a prefix/middle).

    Designed to be load-bearing for the marketplace question ("did
    the agents land on the same answer?"), NOT to replace the strict
    metric — both are reported in ``TrajectoryDivergenceReport`` so
    dashboards can show either.
    """
    if not a and not b:
        return True
    if _content_matches(a, b):
        return True
    na = _extract_numbers(a)
    nb = _extract_numbers(b)
    if na and na == nb:
        return True
    ra = _normalize(a)
    rb = _normalize(b)
    if ra and rb and (ra in rb or rb in ra):
        # Cheap substring match — protects against trailing prose or
        # introductory boilerplate around an otherwise-identical core.
        return True
    return False


def compute_trajectory_divergence(
    rollout: AgentTrajectory,
    trainer: AgentTrajectory,
    *,
    notes: list[str] | None = None,
    device: DeviceFingerprint | None = None,
) -> TrajectoryDivergenceReport:
    """Compute the divergence report between two trajectories on the same task.

    Both trajectories must share task_id and model_id; the rollout side
    is the engine under test and the trainer side is the reference. The
    metric walks the assistant-step sequence and reports the first step
    at which the two diverge plus aggregate set-level agreement on the
    tool-call signature.
    """
    if rollout.task_id != trainer.task_id:
        raise ValueError(
            f"task_id mismatch: rollout={rollout.task_id!r} trainer={trainer.task_id!r}"
        )
    if rollout.model_id != trainer.model_id:
        raise ValueError(
            f"model_id mismatch: rollout={rollout.model_id!r} trainer={trainer.model_id!r}"
        )

    r_assist = rollout.assistant_steps()
    t_assist = trainer.assistant_steps()
    pair_count = min(len(r_assist), len(t_assist))

    first_div: int | None = None
    first_kind: (
        Literal["tool_choice", "tool_args", "content", "termination"] | None
    ) = None
    disagreement_steps = 0
    eligible_steps = 0

    for i in range(pair_count):
        r_step = r_assist[i]
        t_step = t_assist[i]
        r_calls = r_step.tool_calls
        t_calls = t_step.tool_calls

        # Compare tool calls first — they're the load-bearing decision.
        if r_calls or t_calls:
            eligible_steps += 1
            r_names = [c.name for c in r_calls]
            t_names = [c.name for c in t_calls]
            r_args = [c.arguments_hash for c in r_calls]
            t_args = [c.arguments_hash for c in t_calls]
            if r_names != t_names:
                disagreement_steps += 1
                if first_div is None:
                    first_div = i
                    first_kind = "tool_choice"
                continue
            if r_args != t_args:
                if first_div is None:
                    first_div = i
                    first_kind = "tool_args"
                continue
            # Same tools, same args — agreement at this step.
        else:
            # Pure content step. Compare normalized content.
            if not _content_matches(r_step.content, t_step.content):
                if first_div is None:
                    first_div = i
                    first_kind = "content"

    if first_div is None and len(r_assist) != len(t_assist):
        first_div = pair_count
        first_kind = "termination"

    # Tool-call set Jaccard over (tool_name, arguments_hash) tuples.
    r_sig_set = set(rollout.tool_call_signature())
    t_sig_set = set(trainer.tool_call_signature())
    union = r_sig_set | t_sig_set
    jaccard = (len(r_sig_set & t_sig_set) / len(union)) if union else 1.0

    final_match = _content_matches(rollout.final_answer, trainer.final_answer)
    final_match_soft = _soft_content_match(rollout.final_answer, trainer.final_answer)
    if rollout.final_answer == "" and trainer.final_answer == "":
        # Neither answered — call it a match only if both ran out of steps
        # for the same reason; otherwise treat as a non-match because
        # neither produced a usable answer.
        same_terminal = (
            rollout.terminal_reason == trainer.terminal_reason
            and rollout.terminal_reason == "answered"
        )
        final_match = same_terminal
        final_match_soft = same_terminal

    return TrajectoryDivergenceReport(
        run_id=f"{rollout.run_id}-vs-{trainer.run_id}",
        task_id=rollout.task_id,
        task_text=rollout.task_text,
        model_id=rollout.model_id,
        rollout_engine=rollout.rollout_engine,
        trainer_engine=trainer.rollout_engine,
        rollout_step_count=len(rollout.steps),
        trainer_step_count=len(trainer.steps),
        rollout_assistant_steps=len(r_assist),
        trainer_assistant_steps=len(t_assist),
        first_divergence_step=first_div,
        first_divergence_kind=first_kind,
        tool_choice_disagreement_rate=(
            disagreement_steps / eligible_steps if eligible_steps else 0.0
        ),
        tool_call_jaccard=jaccard,
        final_answer_match=final_match,
        final_answer_match_soft=final_match_soft,
        rollout_terminal_reason=rollout.terminal_reason,
        trainer_terminal_reason=trainer.terminal_reason,
        same_step_count=len(r_assist) == len(t_assist),
        rollout_tool_signature=[list(t) for t in rollout.tool_call_signature()],
        trainer_tool_signature=[list(t) for t in trainer.tool_call_signature()],
        device=device,
        notes=list(notes or []),
    )


# --- HTML rendering ---------------------------------------------------------


def _step_summary(step: AgentStep) -> str:
    if step.tool_calls:
        names = ", ".join(c.name for c in step.tool_calls)
        return f"<code>tool_call</code> → {_html.escape(names)}"
    if step.role == "tool":
        text = step.content.strip().splitlines()[0] if step.content else ""
        return f"<code>tool_result</code>: {_html.escape(text[:80])}"
    text = step.content.strip().splitlines()[0] if step.content else ""
    return f"<code>assistant</code>: {_html.escape(text[:80])}"


def render_trajectory_html(report: TrajectoryDivergenceReport, rollout: AgentTrajectory, trainer: AgentTrajectory) -> str:
    """Render a side-by-side trajectory diff for one comparison."""

    r_assist = rollout.assistant_steps()
    t_assist = trainer.assistant_steps()
    n = max(len(r_assist), len(t_assist))

    rows = []
    for i in range(n):
        r_step = r_assist[i] if i < len(r_assist) else None
        t_step = t_assist[i] if i < len(t_assist) else None
        is_div = report.first_divergence_step is not None and i == report.first_divergence_step
        is_after_div = report.first_divergence_step is not None and i > report.first_divergence_step
        klass = "div" if is_div else ("after" if is_after_div else "match")
        r_html = _step_summary(r_step) if r_step else "<em>—</em>"
        t_html = _step_summary(t_step) if t_step else "<em>—</em>"
        rows.append(
            f"<tr class='{klass}'><td>{i}</td>"
            f"<td>{r_html}</td>"
            f"<td>{t_html}</td></tr>"
        )

    summary_rows = "".join(
        f"<tr><th>{_html.escape(k)}</th><td>{_html.escape(str(v))}</td></tr>"
        for k, v in [
            ("task_id", report.task_id),
            ("model_id", report.model_id),
            (
                "rollout",
                f"{report.rollout_engine.name} {report.rollout_engine.version}",
            ),
            (
                "trainer",
                f"{report.trainer_engine.name} {report.trainer_engine.version}",
            ),
            ("rollout assistant steps", report.rollout_assistant_steps),
            ("trainer assistant steps", report.trainer_assistant_steps),
            ("first divergence step", report.first_divergence_step),
            ("first divergence kind", report.first_divergence_kind),
            (
                "tool choice disagreement rate",
                f"{report.tool_choice_disagreement_rate:.4f}",
            ),
            ("tool call jaccard", f"{report.tool_call_jaccard:.4f}"),
            ("final answer match", report.final_answer_match),
            ("rollout terminal", report.rollout_terminal_reason),
            ("trainer terminal", report.trainer_terminal_reason),
        ]
    )

    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        f"<title>Agent trajectory diff — {_html.escape(report.task_id)}</title>"
        "<style>"
        "body{font-family:system-ui,sans-serif;max-width:1080px;margin:2rem auto;color:#111}"
        "table{border-collapse:collapse;width:100%;margin-bottom:1rem}"
        "th,td{border:1px solid #ddd;padding:.4rem .6rem;text-align:left;vertical-align:top}"
        "th{background:#f5f5f5}"
        "tr.match td{background:#f4fbf4}"
        "tr.div td{background:#fff0f0;font-weight:600}"
        "tr.after td{background:#fff8e8}"
        "code{font-family:ui-monospace,monospace;font-size:.85em}"
        "h1{font-size:1.2rem}h2{font-size:1rem;margin-top:1.5rem}"
        ".legend span{display:inline-block;padding:.1rem .5rem;margin-right:.5rem;border-radius:3px}"
        ".legend .m{background:#f4fbf4}.legend .d{background:#fff0f0}.legend .a{background:#fff8e8}"
        "</style></head><body>"
        f"<h1>Agent trajectory diff — {_html.escape(report.task_id)}</h1>"
        f"<p><strong>Task:</strong> {_html.escape(report.task_text)}</p>"
        "<h2>Summary</h2>"
        f"<table>{summary_rows}</table>"
        "<h2>Side-by-side steps</h2>"
        "<p class='legend'><span class='m'>match</span><span class='d'>first divergence</span><span class='a'>after divergence</span></p>"
        "<table><thead><tr><th>step</th><th>rollout</th><th>trainer</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
        "</body></html>"
    )


def write_trajectory_diff(
    report: TrajectoryDivergenceReport,
    rollout: AgentTrajectory,
    trainer: AgentTrajectory,
    out_dir: Path | str,
) -> dict[str, Path]:
    """Persist JSON + HTML side by side under out_dir."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    json_path = out_path / "agent_divergence_report.json"
    html_path = out_path / "agent_divergence_report.html"
    json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    html_path.write_text(
        render_trajectory_html(report, rollout, trainer), encoding="utf-8"
    )
    return {"json": json_path, "html": html_path}


def load_trajectory(path: Path | str) -> AgentTrajectory:
    return AgentTrajectory.model_validate_json(Path(path).read_text(encoding="utf-8"))


def load_trajectories_glob(pattern: str) -> list[AgentTrajectory]:
    import glob as _glob
    return [load_trajectory(p) for p in sorted(_glob.glob(pattern, recursive=True))]
