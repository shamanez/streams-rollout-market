---
name: autonomous-loop
description: Standalone autonomous engine. Type `/autonomous-loop` once and it runs iteration after iteration end-to-end (pick task → implement → test → evaluate → commit → next) inside a single turn until a stop condition fires. No `/loop` wrapper required. Phase 0-6 is complete; the loop now picks tasks from .claude/feature-results.json (when STEER.md is active), PROGRESS.md "Next work", or docs/future_research.md, in that order.
user-invocable: true
allowed-tools: Read, Write, Edit, Bash, Glob, Grep
---

# Autonomous Development Loop

Continuous, hands-free work on streams-rollout-market. **The original
Phase 0-6 plan is complete** — the loop now operates in
*evidence-extension* mode, picking the next entry in
`.claude/feature-results.json` (when STEER.md is active) or a research
follow-up otherwise.

## How invocation works

This skill drives itself. **The single command `/autonomous-loop` is
the entire user interface.** When invoked:

1. Run one iteration of the Loop cycle below.
2. At the end of the iteration, check the Stop conditions.
3. If none fire, **immediately begin the next iteration in the same
   turn** — do not wait for user input, do not call ScheduleWakeup, do
   not stop the assistant turn.
4. Only stop the turn when a Stop condition is satisfied.

The assistant turn is a single long-running session that ticks through
iterations back-to-back. Across iterations, write status lines so the
operator watching `watch -n 5 'tail -30 PROGRESS.md'` and
`watch -n 5 'git log --oneline -8'` sees progress.

If the turn does eventually exit (manual interrupt, context
compaction, AGENT_STOP), the operator resumes by simply typing
`/autonomous-loop` again — state lives in `PROGRESS.md`,
`.claude/feature-results.json`, and git, so the next invocation picks
up exactly where this one left off.

## Loop cycle

Each iteration performs ONE unit of work.

### 1. Context check
- Read `PROGRESS.md` — the source of truth for state.
- Check `STEER.md` — if it exists, follow operator redirect instead.
- Check `AGENT_STOP` — if it exists, stop immediately after updating
  PROGRESS.md.

If `STEER.md` exists, also read `.claude/feature-results.json`. Pick the
first entry whose `passes` is `false` and use its `directive` field
verbatim as the input to `/implement-feature`. The loop does NOT mark an
entry true directly — it writes evidence to `.claude/evidence/` and then
submits a write to `feature-results.json` that the PreToolUse hook
`.claude/scripts/verify-result-write.sh` approves only if the evidence
path exists and is referenced from `evidence_log.jsonl`. Honor every
"Global directives" bullet in STEER.md (graphs over tables, engine
filter, device axis, verl glossary shape, one feature-results entry per
commit).
- Run `bash .claude/scripts/check-usage.sh` to log session metrics.
- If session > 70% or weekly > 85%: switch to token-saving mode
  (`/codex-review` for review, haiku for test-runner, skip
  adversarial passes).

### 2. Task selection
The plan in `docs/git_plan_v2.md` is complete. Pick the next unit of
work from, in order of preference:

1. **PROGRESS.md "Next work"** — operator-curated research follow-ups
   ranked by effort-to-evidence ratio.
2. **`docs/future_research.md`** — the long-form open-questions list.
3. **`STEER.md`** — operator redirect overrides everything else.

If a follow-up looks too large for one loop iteration (e.g. a full
n=30 matrix run), break it into a setup PR + a run PR + a writeup PR
and only attempt one per iteration.

### 3. Implementation (git-flow)
- Create a feature branch from main:
  ```bash
  git checkout main && git pull origin main
  git checkout -b research/<short-description>
  ```
- For code changes: follow the `/implement-feature` workflow
  (contract-first, AGENTS.md compliant). Honor the architecture
  firewall — no trainer code outside `tests/` or `examples/`.
- For live runs on the spot instance: use the env-driven scripts in
  `scripts/live/` (`MODEL`, `VLLM_DTYPE`, `ROLLOUT_LABEL`,
  `TRAINER_REFERENCE`) and write outputs under `runs/<UTC-ts>-<uuid>/`.
- Run `pytest -q && ruff check .` (must pass).
- Run `bash .claude/scripts/verify-gate.sh` (must pass).

### 4. Evaluation
- Prefer `/codex-review --effort low` for routine reviews.
- Use `/evaluate` for deep checks or when Codex is unavailable.
- On NEEDS_WORK or FAIL: fix the specific issues, re-test, re-eval
  (max 2 retries). After that, record the blocker in PROGRESS.md
  and move on.

### 5. Record, commit, push
- Update PROGRESS.md:
  - Move the completed item from "Next work" to "Completed
    (operational follow-ups...)" (or to a new "Research follow-ups
    shipped" subsection if it was from `future_research.md`).
  - Refresh test count and date.
  - Re-rank remaining "Next work" items if the result changed
    priorities (e.g. a finding moved the needle on a sibling
    question).
- Commit with a descriptive prefix:
  - `live(...)` for spot-instance evidence runs.
  - `feat(...)` for new code.
  - `docs(...)` for write-ups and dashboard changes.
  - `fix(...)` for bug fixes.
- Push and open a PR with the acceptance evidence in the body
  (numbers from the run, dashboard delta, etc).

### 6. Return to main and continue
- `git checkout main && git pull origin main`.
- Re-check the Stop conditions below.
- **If no stop condition fires, go back to Step 1 (Context check) in
  the SAME turn.** Do not pause, do not wait for input, do not call
  ScheduleWakeup. The skill is a long-running in-turn loop.
- Log a one-line status to stdout so the operator can see progress
  (`[autonomous-loop] iteration N complete: <one-line summary>`).

## Stop conditions

Check after every iteration. **If any of these fire, halt the turn
cleanly** (update PROGRESS.md, commit if needed, then end the
assistant turn so the operator can re-invoke).

- `AGENT_STOP` file present at repo root.
- `STEER.md` is gone AND `PROGRESS.md` "Next work" is empty AND
  `docs/future_research.md` has no open items.
- When `STEER.md` is active: every entry in
  `.claude/feature-results.json` has `passes: true`. Delete `STEER.md`
  on the way out per its own stop clause; the next `/autonomous-loop`
  invocation will resume in PROGRESS.md mode.
- Two consecutive iterations produce no committable change — signal
  that the next unit is genuinely blocked on operator input. Record
  the blocker in PROGRESS.md and stop.
- Weekly usage > 95% (from `check-usage.sh`) — stop and let the
  operator decide whether to switch models or wait for reset.

## Error recovery

- **Test failures**: fix code, re-test, max 3 attempts per issue.
- **Import errors**: `pip install -e ".[dev]"`, retry.
- **Spot instance unavailable**: AWS may reclaim it; skip to a code-
  only follow-up and note the blocker in PROGRESS.md.
- **Free-tier rate limits**: add delays, never burst, never enter
  payment info (see CLAUDE.md).
- **Design ambiguity**: check `docs/git_plan_v2.md` for original
  intent, then `docs/future_research.md` for current framing. Make a
  reasonable choice, note it in PROGRESS.md.

## Monitoring (from another terminal)

```bash
watch -n 5 'tail -30 PROGRESS.md'                                  # progress
watch -n 5 'git log --oneline -8'                                  # commits
watch -n 10 'tail -5 .claude/evidence/evidence_log.jsonl'          # test evidence
```

## Operator controls

- **Halt**: `bash .claude/scripts/kill-switch.sh` (or `touch AGENT_STOP`).
- **Redirect**: `bash .claude/scripts/steer.sh "Run n=30 dense matrix next"`.
- **Resume after halt**: `rm AGENT_STOP`.
