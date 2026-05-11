---
name: autonomous-loop
description: Self-pacing engine that drives the streams-rollout-market repo autonomously. Phase 0-6 plan is complete; the loop now picks the next research follow-up from docs/future_research.md (or the explicit "Next work" list in PROGRESS.md), implements/runs it, evaluates, commits, and moves on. Use with /loop for hands-free continuation.
user-invocable: true
allowed-tools: Read, Write, Edit, Bash, Glob, Grep
---

# Autonomous Development Loop

Self-pacing engine for continuous, hands-free work on
streams-rollout-market. **The original Phase 0-6 plan is complete** —
the loop now operates in *evidence-extension* mode, picking
research follow-ups rather than plan tasks.

## Loop cycle

Each iteration performs ONE unit of work.

### 1. Context check
- Read `PROGRESS.md` — the source of truth for state.
- Check `STEER.md` — if it exists, follow operator redirect instead.
- Check `AGENT_STOP` — if it exists, stop immediately after updating
  PROGRESS.md.
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
- If more items remain in "Next work": next iteration.
- If "Next work" is empty: read `docs/future_research.md` and refill
  it (re-ranking is itself a valid loop output — commit it as
  `docs(progress): refresh research backlog`).
- If `AGENT_STOP`: stop after PROGRESS.md handoff.
- If `STEER.md`: follow the new direction.

## Stop conditions
- `AGENT_STOP` file present.
- `STEER.md` instructs a different mode.
- All open questions in `docs/future_research.md` are resolved
  (unlikely in practice — the list is meant to keep growing).
- Two consecutive iterations produce no committable change (signal
  that the next unit is genuinely blocked on operator input).

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
