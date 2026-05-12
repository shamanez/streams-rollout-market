---
name: autonomous-loop
description: Standalone autonomous engine. Type `/autonomous-loop` once and it runs iteration after iteration end-to-end (pick task → implement → test → evaluate → commit → next) inside a single turn until a stop condition fires. The queue lives in `.claude/feature-results.json` (seeded by `/seed-from-plan` or written manually); STEER.md carries the global directives for the current initiative.
user-invocable: true
allowed-tools: Read, Write, Edit, Bash, Glob, Grep
---

# Autonomous Development Loop

Continuous, hands-free work on streams-rollout-market. The loop is
generic — it doesn't know about any specific past initiative; it just
walks whatever queue is currently in `.claude/feature-results.json`.

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
- Read `PROGRESS.md` — the source of truth for handoff state.
- Check `STEER.md` — operator directive + global acceptance gates for
  the current initiative. If absent, the loop has nothing to drive;
  see Stop conditions.
- Check `AGENT_STOP` — if it exists, stop immediately after updating
  PROGRESS.md.
- Read `.claude/feature-results.json`. The queue is a flat dict of
  entries. Pick the first entry (file order) whose `passes` is
  `false` and use its `directive` field verbatim as the spec for this
  iteration.
- Honor every "Global directives" bullet in STEER.md.
- Run `bash .claude/scripts/check-usage.sh` to log session metrics.
- If session > 70% or weekly > 85%: switch to token-saving mode
  (`/codex-review` for review, haiku for test-runner, skip
  adversarial passes).

### 2. Implementation (git-flow)
- Create a feature branch from main:
  ```bash
  git checkout main && git pull origin main
  git checkout -b feature/<entry-slug>
  ```
- Follow the `/implement-feature` workflow (contract-first, AGENTS.md
  compliant). Honor the architecture firewall — no trainer code
  outside `tests/` or `examples/`.
- If the entry's acceptance criteria require a live run on the spot
  instance, use the env-driven scripts in `scripts/live/` and write
  outputs under `runs/<UTC-ts>-<uuid>/`.
- Run `pytest -q && ruff check .` (must pass).
- Run `bash .claude/scripts/verify-gate.sh` (must pass).

### 3. Evaluation
- Prefer `/codex-review --effort low` for routine reviews.
- Use `/evaluate` for deep checks or when Codex is unavailable.
- On NEEDS_WORK or FAIL: fix the specific issues, re-test, re-eval
  (max 2 retries). After that, record the blocker in PROGRESS.md
  and leave the entry `passes:false` so the loop moves on.

### 4. Record, commit, push
- Flip the entry's `passes:true` and set `evidence` to a path under
  `.claude/evidence/` that exists and is referenced from
  `evidence_log.jsonl`. The `verify-result-write.sh` PreToolUse hook
  enforces this — a missing/unreferenced path makes the write fail.
- Update PROGRESS.md:
  * Move the entry's slug into "Completed this cycle".
  * Refresh test count and date.
  * If the entry's outcome reordered the remaining queue, re-sort
    `.claude/feature-results.json` accordingly.
- Commit with a descriptive prefix:
  * `live(...)` for spot-instance evidence runs.
  * `feat(...)` for new code.
  * `docs(...)` for write-ups and dashboard changes.
  * `fix(...)` for bug fixes.
- (Optional) push and open a PR with the acceptance evidence in the
  body (numbers from the run, dashboard delta, etc).

### 5. Return to main and continue
- `git checkout main && git pull origin main`.
- Re-check the Stop conditions below.
- **If no stop condition fires, go back to Step 1 (Context check) in
  the SAME turn.** Do not pause, do not wait for input, do not call
  ScheduleWakeup. The skill is a long-running in-turn loop.
- Log a one-line status to stdout so the operator can see progress
  (`[autonomous-loop] iteration N complete: <one-line summary>`).

## Stop conditions

**These are the ONLY reasons to halt.** Anything else — context
getting long, "next iteration looks like substantial new code", a
spot run taking 20 minutes, an iteration spanning setup + run +
writeup — is NOT a stop reason. Keep iterating.

Check after every iteration. **If any of these fire, halt the turn
cleanly** (update PROGRESS.md, commit if needed, then end the
assistant turn so the operator can re-invoke).

- `AGENT_STOP` file present at repo root.
- `STEER.md` is gone AND every entry in `.claude/feature-results.json`
  has `passes:true` (or the file is `{}`). On clean exit, delete
  `STEER.md` per its own stop clause; the next `/autonomous-loop`
  invocation will find no work and halt immediately.
- `.claude/feature-results.json` is `{}` and `STEER.md` is absent —
  no queue, nothing to drive.
- Two consecutive iterations produce no committable change — signal
  that the next unit is genuinely blocked on operator input. Record
  the blocker in PROGRESS.md and stop.
- Weekly usage > 95% (from `check-usage.sh`) — stop and let the
  operator decide whether to switch models or wait for reset.

### What is NOT a stop condition (do NOT halt for these)

- "The next iteration looks big" — break it into setup + run +
  writeup commits and keep going. Each is one iteration.
- "I've done several iterations and the conversation is long" — the
  runtime compacts context automatically; trust it and continue.
- "The next step needs a long spot run (10+ min)" — kick it off with
  `run_in_background: true`, do other useful work or wait for the
  notification, then continue.
- "I'm not sure how to do the next step exactly" — pick a reasonable
  approach, write a setup commit with TODOs, and iterate. The loop
  prefers committed progress over perfect planning.
- "It feels like a clean stopping point" — feelings are not a stop
  condition.

If you find yourself about to halt without one of the hard
conditions above firing, **do not halt** — start the next iteration.
The operator can always `touch AGENT_STOP` if they want to stop.

## Error recovery

- **Test failures**: fix code, re-test, max 3 attempts per issue.
- **Import errors**: `pip install -e ".[dev]"`, retry.
- **Spot instance unavailable**: AWS may reclaim it; skip to a code-
  only entry and note the blocker in PROGRESS.md.
- **Free-tier rate limits**: add delays, never burst, never enter
  payment info (see CLAUDE.md).
- **Design ambiguity**: re-read the originating plan referenced by
  `STEER.md`. Make a reasonable choice, note it in PROGRESS.md.

## Monitoring (from another terminal)

```bash
watch -n 5 'tail -30 PROGRESS.md'                                  # progress
watch -n 5 'git log --oneline -8'                                  # commits
watch -n 10 'tail -5 .claude/evidence/evidence_log.jsonl'          # test evidence
watch -n 5 'jq ".[] | select(.passes==false)" .claude/feature-results.json | head -20'  # next-up
```

## Operator controls

- **Seed a new initiative**: write a plan markdown, then
  `/seed-from-plan <path/to/plan.md>` — this writes `STEER.md` and
  `.claude/feature-results.json` for you.
- **Halt**: `bash .claude/scripts/kill-switch.sh` (or `touch AGENT_STOP`).
- **Redirect mid-flight**: `bash .claude/scripts/steer.sh "<directive>"`
  rewrites `STEER.md`; the next iteration picks it up.
- **Resume after halt**: `rm AGENT_STOP`.
