# PROGRESS.md — Agent Handoff State

## Last updated
2026-05-12 — idle.

## Status
Idle. No active queue.

## How to start a new initiative

1. **Plan**: enter Plan Mode (`Shift+Tab`), describe the initiative
   in plain English. Claude drafts a plan and exits plan mode. Hit
   Accept.
2. **Seed**: `/seed-from-plan` (no args) — reads the plan from
   conversation context, writes `STEER.md` (the plan verbatim) and
   `.claude/feature-results.json` (one `passes:false` entry per
   iteration), and commits.
3. **Drive**: `/autonomous-loop` walks the queue end-to-end.

No template required. The plan can be in whatever shape Plan Mode
produced — the skill auto-detects iteration boundaries.

## Reproduction (durable)

Full single-host pipeline (laptop ↔ spot) documented in `CLAUDE.md`
under "End-to-end reproduction".
