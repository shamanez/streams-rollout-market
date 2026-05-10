---
name: build-and-test
description: Run the full build, test, and lint suite for streams-rollout-market. Use after making changes to verify everything passes.
user-invocable: true
allowed-tools: Bash, Read
---

# Build and Test

Run the complete verification suite for the project.

## Steps

1. Install dev dependencies (only if needed):
   ```bash
   pip install -e ".[dev]"
   ```

2. Run the test suite:
   ```bash
   pytest -q --tb=short
   ```

3. Run the linter:
   ```bash
   ruff check .
   ```

4. Check formatting:
   ```bash
   ruff format --check .
   ```

5. Run the smoke test:
   ```bash
   python examples/local_worker_demo.py
   ```

## Success criteria

All 4 steps must exit with code 0. If any step fails:
- Report the specific failure with file:line detail
- Classify the failure type (test bug, code bug, lint issue, format issue, import error)
- Do NOT attempt to fix anything -- just report

## Output format

```
## Build & Test Results
- pytest: PASS/FAIL (N passed, N failed)
- ruff check: PASS/FAIL (N issues)
- ruff format: PASS/FAIL (N issues)
- smoke test: PASS/FAIL
- Overall: PASS/FAIL
```
