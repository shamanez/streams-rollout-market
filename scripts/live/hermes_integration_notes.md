# Hermes Agent integration notes (Iter 3 discovery)

## What was assumed in Iter 1

`scripts/live/hermes_agent_runner.py`'s `adapt_hermes_record` was
designed against an OpenAI chat-completions JSONL shape — one record
per message, with `role ∈ {assistant, tool}`, `tool_calls` as a list
of OpenAI tool-call dicts, `tool_call_id` on tool-role records.

This was an educated guess. The STEER directive says the runner
"shells out to the NousResearch/hermes-agent CLI (run one task at a
time with OPENAI_BASE_URL+OPENAI_API_KEY env), parses the resulting
JSONL trajectory" — implying a per-task CLI invocation that emits a
chat-style JSONL.

## What Hermes Agent actually emits

The real entry point for batch trajectory generation is
`/tmp/hermes-agent/batch_runner.py` (cloned from
`https://github.com/NousResearch/hermes-agent`). Its output JSONL is
**one record per prompt** (not per message), in ShareGPT shape:

```json
{
  "prompt_index": 0,
  "conversations": [
    {"from": "system", "value": "<system prompt + <tools>...</tools>>"},
    {"from": "human",  "value": "<user task text>"},
    {"from": "gpt",    "value": "<think>...</think>\n<tool_call>\n{\"name\":\"calculator\",\"arguments\":{...}}\n</tool_call>"},
    {"from": "tool",   "value": "<tool_response>...</tool_response>"},
    {"from": "gpt",    "value": "Final answer text."}
  ],
  "metadata": {...},
  "completed": true,
  "api_calls": 4,
  "toolsets_used": [...],
  "tool_stats": {...}
}
```

Key shape differences from the Iter 1 assumption:

1. **Per-prompt, not per-message.** One JSON object per task, with
   the conversation history as a nested `conversations` array.
2. **ShareGPT `from/value` fields**, not OpenAI `role/content`. The
   `from` field uses `system|human|gpt|tool`, not
   `system|user|assistant|tool`.
3. **Tool calls live inside the `value` string** as inline XML
   (`<tool_call>...</tool_call>`), not as a structured `tool_calls`
   array on a separate field. Arguments are JSON inside the XML body.
4. **Tool results live inside the `value` string** as inline XML
   (`<tool_response>...</tool_response>`), without a `tool_call_id`
   linking back to the originating call.
5. **`<think>` reasoning is also inline** in the `value` string,
   prepended to the assistant turn.

## What Iter 1 still earns

The `AgentTrajectory` schema is unchanged — it's still the right
target type. `adapt_hermes_record` and `adapt_hermes_jsonl` correctly
handle a hypothetical chat-completions adapter; they just don't match
hermes-agent's actual shape. They are still useful for any
OpenAI-chat-completions agent harness (including our existing
`scripts/live/agent_runner.py`).

The `tests/test_hermes_agent_runner.py` 18 tests pass against the
adapter's *documented* contract. They do NOT validate the adapter
against real hermes-agent output — that would require a new fixture
captured from a real `batch_runner.py` run.

## What Iter 3 needs to actually ship

To produce real trajectory JSONs under
`runs/live/agent/hermes/qwen3-32b/{bf16,bf16_seedB,fp8}/<task_id>.json`
that meet STEER.md's acceptance gate, the following work is required:

1. **Install hermes-agent.** The README's `curl … | bash` script
   bootstraps uv, Python 3.11, Node, ripgrep, ffmpeg into `~/.hermes`
   (~500MB). This is a heavy autonomous install that the operator
   should approve explicitly — installing arbitrary remote bash
   scripts is not a routine autonomous action.

2. **Configure hermes-agent to talk to our spot vLLM serve.** The
   Hermes provider model is configured via `hermes model` / `hermes
   config set`; pointing at an OpenAI-compatible endpoint requires
   setting `OPENAI_BASE_URL` and `OPENAI_API_KEY` (or via the config
   YAML).

3. **Author a dataset JSONL** in hermes-agent's expected format
   (one prompt per line) from `scripts/live/agent_tasks_hermes.json`.
   Use `batch_runner.py --dataset_file=tasks.jsonl --batch_size=1
   --run_name=qwen3-32b-bf16` per side.

4. **Adapt the ShareGPT output to AgentTrajectory.** Replace the
   chat-completions adapter with one that:
   - Reads one trajectory record per JSONL line.
   - Splits each `from=gpt` `value` into `<think>` content + zero or
     more inline `<tool_call>` blocks (regex or a small XML-tag
     scanner).
   - Pairs each `<tool_call>` with the *next* `<tool_response>` block
     (positional pairing; hermes does not emit `tool_call_id`).
   - Maps `from` values: `system|human` → drop; `gpt` → assistant;
     `tool` → tool.
   - Sets `terminal_reason` from the trajectory record's `completed`
     and `partial` fields.

5. **Validate the runner against a real captured trace** before
   running 36 trajectories on the spot. One smoke-test trajectory
   from a small model (e.g. Qwen3-8B locally if available) is
   cheaper than 36 botched ones on the L40S.

## Recommended path forward (operator decision)

There are three credible options:

**A. Pivot Iter 1's adapter to the ShareGPT shape.** Update
`scripts/live/hermes_agent_runner.py` to parse `from/value` records
and inline XML, drop the chat-completions tests, replace them with
fixture tests against the real shape. ~1 iteration of code-only work,
then proceed with Iter 3 as planned. The 18 chat-completions tests
become obsolete.

**B. Relax the STEER directive to "hermes-compatible tool-call
parser".** Keep using our existing `scripts/live/agent_runner.py`
(OpenAI chat-completions client) but have vLLM serve with
`--tool-call-parser hermes`. The "real Hermes Agent" credibility
narrative weakens — we use Hermes's tool-call format but not its
agent runtime. This is what the plan called Path B and rejected.
Resurfacing it because Path A turned out to be heavier than
budgeted.

**C. Defer Iter 3-6 entirely**, mark all four `hermes_agent.*`
remote-run entries as out-of-scope-for-now in `PROGRESS.md`, and
focus on a different research follow-up from
`docs/future_research.md`. The 8-tile matrix already answers the
load-bearing question; adding agent trajectories was a "third
corroboration" that can wait.

The autonomous loop cannot pick among these without operator input —
each represents a different scope contract.

## Captured commands (for option A or operator-driven Iter 3)

```bash
# install (one-time on Macbook):
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
source ~/.bashrc
hermes setup            # interactive; configure OPENAI_BASE_URL + OPENAI_API_KEY
hermes doctor           # verify install

# serve vLLM Qwen3-32B bf16 on spot (existing script):
ssh my-vllm-spot-instance 'MODEL=Qwen/Qwen3-32B DTYPE=bfloat16 bash ~/serve_qwen_for_agent.sh'

# tunnel from Mac:
ssh -L 8000:localhost:8000 my-vllm-spot-instance -N &

# author dataset.jsonl from scripts/live/agent_tasks_hermes.json:
python3 -c '
import json
tasks = json.load(open("scripts/live/agent_tasks_hermes.json"))
with open("/tmp/hermes_tasks.jsonl", "w") as f:
    for t in tasks:
        f.write(json.dumps({"prompt": t["task_text"]}) + "\n")
'

# run batch:
cd /tmp/hermes-agent
OPENAI_BASE_URL=http://localhost:8000/v1 \
OPENAI_API_KEY=dummy \
HERMES_MODEL=Qwen/Qwen3-32B \
python batch_runner.py --dataset_file=/tmp/hermes_tasks.jsonl \
  --batch_size=1 \
  --run_name=qwen3-32b-bf16 \
  --max_iterations=8

# output: trajectories under ./trajectory_outputs/qwen3-32b-bf16/*.jsonl
# adapter (option A): scripts/live/hermes_agent_runner.py (after rewrite)
#   reads one trajectory record per line, emits one AgentTrajectory JSON per task
```

## Open questions for the operator

1. Approve installing hermes-agent into `~/.hermes` (~500MB; uv +
   Python 3.11 + Node + ffmpeg)? If no, options B or C only.
2. Approve rewriting Iter 1's adapter to ShareGPT shape (obsoletes
   the existing 18 chat-completions tests)? Or keep them as a
   "OpenAI-chat-completions adapter" library for our toy harness
   and add a *separate* `adapt_sharegpt_trajectory` for the real
   thing?
3. Comfortable with ~60-90 min of spot wall-clock for the three
   serve cycles (Iter 3) plus an equivalent block for Iter 5?
