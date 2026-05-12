---
name: implement-feature
description: Implement one unit of work for streams-rollout-market. The unit is either a slug from .claude/feature-results.json (when the autonomous loop drives it) or a free-form directive. Contract-first workflow with AGENTS.md compliance.
user-invocable: true
allowed-tools: Read, Write, Edit, Bash, Glob, Grep
---

# Implement Feature

Implement one unit of work. The unit of work is either:

* a **slug** from `.claude/feature-results.json` (the queue seeded by
  `/seed-from-plan` or written manually). The first entry whose
  `passes` is `false` is the target.
* a **free-form directive** the operator passed in as `$ARGUMENTS`.

## Input

`$ARGUMENTS` -- a slug from `.claude/feature-results.json` OR a free-form
description of what to build. If empty, pick the first `passes:false`
entry from the queue.

## Workflow

1. **Read AGENTS.md** -- MANDATORY before any code changes.
2. **Resolve the directive**:
   * If `$ARGUMENTS` matches a slug in `.claude/feature-results.json`,
     load that entry's `directive` verbatim.
   * Otherwise treat `$ARGUMENTS` (or the first `passes:false` entry's
     directive) as the spec.
3. **Create a feature branch** from main:
   ```bash
   git checkout main && git pull origin main
   git checkout -b feature/<slug-or-short-description>
   ```
4. **Read relevant source files** referenced by the directive's
   acceptance criteria.
5. **Implement** using contract-first order:
   a. Pydantic models / contracts first
   b. Feature logic second
   c. Tests third
   d. Doc updates if contracts changed
6. **Verify** -- run the build-and-test suite:
   ```bash
   pytest -q --tb=short && ruff check . && ruff format --check .
   ```
7. **Verify evidence**:
   ```bash
   bash .claude/scripts/verify-gate.sh
   ```
8. **Flip the entry** (only if it came from `.claude/feature-results.json`):
   set `passes:true` and `evidence` to a path under `.claude/evidence/`
   that exists AND is referenced from `evidence_log.jsonl`. The hook
   `verify-result-write.sh` enforces this; a missing/unreferenced path
   makes the write fail.
9. **Update PROGRESS.md**:
   * Move the entry into "Completed this cycle".
   * Refresh test count and date.
10. **Commit** with a descriptive message:
    ```
    <type>(<slug>): <one-line summary>

    <what was done and why>
    ```
11. **Push branch and create PR** (optional, depending on workflow):
    ```bash
    git push -u origin feature/<slug-or-short-description>
    ```

## Conventions

* Pydantic `BaseModel` for data contracts
* `dataclass(frozen=True)` for internal config
* Type hints + docstrings on all public functions
* ruff line-length = 100
* Tests: `test_<module>.py`, import from `rollout_market`
* Every new contract: at least one fixture + one validation test

## Failure handling

* Tests fail: fix the code (not the tests, unless the test itself is wrong).
* Blocked: record in `PROGRESS.md`, leave the entry `passes:false`,
  move on to the next entry on the next iteration.
* AGENTS.md would be violated: STOP and do not proceed.
