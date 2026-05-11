---
name: implement-feature
description: Implement a feature for streams-rollout-market. Pass a task ID from the (now-complete) Phase 0-6 plan in docs/git_plan_v2.md, or a description of a research follow-up from PROGRESS.md / docs/future_research.md. Follows contract-first workflow with AGENTS.md compliance.
user-invocable: true
allowed-tools: Read, Write, Edit, Bash, Glob, Grep
---

# Implement Feature

Implement a feature task. The Phase 0-6 plan in `docs/git_plan_v2.md`
is complete, so new work usually comes from PROGRESS.md's "Next work"
list or `docs/future_research.md`.

## Input

`$ARGUMENTS` -- task ID (e.g., "1.1", "2.3") or a feature description.

## Workflow

1. **Read AGENTS.md** -- MANDATORY before any code changes
2. **Create a feature branch** from main:
   ```bash
   git checkout main && git pull origin main
   git checkout -b feature/phase-X.Y-<short-description>
   ```
3. **Look up the task** in `docs/git_plan_v2.md`
   - Find the task matching `$ARGUMENTS`
   - Note: Context, Goal, Non-goals, Required changes, Acceptance criteria
4. **Read relevant source files** listed in the task's context
5. **Implement** using contract-first order:
   a. Pydantic models / contracts first
   b. Feature logic second
   c. Tests third
   d. Doc updates if contracts changed
5. **Verify** -- run the build-and-test suite:
   ```bash
   pytest -q --tb=short && ruff check . && ruff format --check .
   ```
6. **Verify evidence** -- confirm test evidence exists:
   ```bash
   bash .claude/scripts/verify-gate.sh
   ```

   If the task was selected from `.claude/feature-results.json` (i.e.
   `STEER.md` is active), the final action of the iteration is to flip
   that entry's `passes` to `true` with `evidence` set to the concrete
   path produced during this iteration (a rendered HTML, a test log, a
   screenshot under `.claude/evidence/`). The PreToolUse hook
   `.claude/scripts/verify-result-write.sh` will deny the write if the
   evidence path does not exist or is not referenced from
   `.claude/evidence/evidence_log.jsonl`. After a successful flip,
   commit and move to the next `false` entry on the next iteration.
7. **Update PROGRESS.md**:
   - Move the task from "Next tasks" to "Completed"
   - Add the next task(s) to "Next tasks"
   - Update test status
8. **Commit** with a descriptive message:
   ```
   Phase X.Y: <task title>

   <what was done and why>
   ```
9. **Push branch and create PR**:
   ```bash
   git push -u origin feature/phase-X.Y-<short-description>
   ```
   Create a PR to main with the task description as the PR body.

## Conventions

- Pydantic `BaseModel` for data contracts
- `dataclass(frozen=True)` for internal config
- Type hints + docstrings on all public functions
- ruff line-length = 100
- Tests: `test_<module>.py`, import from `rollout_market`
- Every new contract: at least one fixture + one validation test

## Failure handling

- If tests fail: fix the code (not the tests, unless the test itself is wrong)
- If blocked: record in PROGRESS.md, skip to next task
- If AGENTS.md would be violated: STOP and do not proceed
