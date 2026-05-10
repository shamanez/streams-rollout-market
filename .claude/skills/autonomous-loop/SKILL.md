---
name: autonomous-loop
description: Self-pacing engine that drives the Phase 0-6 development plan autonomously. Picks next task, implements, evaluates, commits, and moves on. Use with /loop for continuous hands-free development.
user-invocable: true
allowed-tools: Read, Write, Edit, Bash, Glob, Grep
---

# Autonomous Development Loop

Self-pacing engine for continuous, hands-free development of the
streams-rollout-market project through its Phase 0-6 plan.

## Loop cycle

Each iteration of this loop performs ONE task from the development plan:

### 1. Context check
- Read `PROGRESS.md` to find the current phase and next task
- Check for `STEER.md` -- if it exists, follow operator redirect instead of the plan
- Check for `AGENT_STOP` -- if it exists, stop immediately after updating PROGRESS.md
- Run `bash .claude/scripts/check-usage.sh` to log session metrics
- If nearing session limits, switch to token-saving mode:
  - Use `/codex-review` instead of Claude evaluator
  - Use haiku test-runner instead of sonnet
  - Skip adversarial reviews, do simple pass/fail checks only

### 2. Task selection
- Open `streams_rollout_market_git_agent_plan_v2.md`
- Find the next incomplete task from PROGRESS.md's "Next tasks" section
- If no tasks remain in the current phase, advance to the next phase

### 3. Implementation (git-flow)
- Create a feature branch from main:
  ```bash
  git checkout main && git pull origin main
  git checkout -b feature/phase-X.Y-<short-description>
  ```
- Use the `/implement-feature` skill workflow:
  - Read AGENTS.md (mandatory)
  - Read relevant source files
  - Implement contract-first: models -> logic -> tests -> docs
  - Run `pytest -q && ruff check .` (must pass)
  - Run `bash .claude/scripts/verify-gate.sh` (must pass)

### 4. Evaluation
- Prefer `/codex-review --effort low` for routine reviews (saves Claude tokens)
- Use the `/evaluate` skill (Claude evaluator agent) for deep checks or when Codex is unavailable
- If evaluation returns NEEDS_WORK or FAIL:
  - Fix the specific issues identified
  - Re-run tests
  - Re-evaluate (max 2 retries per task)
- If evaluation still fails after retries:
  - Record the issue in PROGRESS.md
  - Move to the next task

### 5. Record, commit, and push
- Update PROGRESS.md:
  - Move completed task to "Completed" section
  - Set next task(s) in "Next tasks"
  - Update test status and date
- Commit with task ID in message:
  ```
  Phase X.Y: <task title>
  ```
- Push the feature branch and create a PR to main:
  ```bash
  git push -u origin feature/phase-X.Y-<short-description>
  gh pr create --title "Phase X.Y: <task title>" --body "<acceptance criteria>"
  ```
- After PR is created, merge it (if auto-merge is enabled) or move on

### 6. Return to main and continue
- `git checkout main && git pull origin main`
- If more tasks remain: continue to next iteration
- If AGENT_STOP appeared: stop after handoff
- If STEER.md appeared: follow the new direction
- If all Phase 0-6 tasks complete: run final evaluation and stop

## Error recovery

- **Test failures**: Fix code, re-test, max 3 attempts per issue
- **Import errors**: Run `pip install -e ".[dev]"`, retry
- **Design ambiguity**: Check the plan document, make a reasonable choice, note it in PROGRESS.md
- **Blocked task**: Record blocker, skip to next task

## Monitoring (from another terminal)

```bash
watch -n 5 'tail -20 PROGRESS.md'           # agent's progress
watch -n 5 'git log --oneline -8'            # committed work
watch -n 10 'cat .claude/evidence/evidence_log.jsonl | tail -5'  # test evidence
```

## Operator controls

- **Halt**: `bash .claude/scripts/kill-switch.sh` (or `touch AGENT_STOP`)
- **Redirect**: `bash .claude/scripts/steer.sh "Switch to Phase 2.1"`
- **Resume after halt**: `rm AGENT_STOP`
