---
name: session-handoff
description: Prepare for session end by recording final test status, updating PROGRESS.md, and committing all work. Use before ending a session to ensure continuity.
user-invocable: true
allowed-tools: Bash, Read, Write, Edit
---

# Session Handoff

Prepare for clean session-to-session continuity.

## Steps

1. **Run final tests** to record current status:
   ```bash
   pytest -q --tb=no 2>&1 | tail -5
   ```

2. **Update PROGRESS.md** with:
   - `Last updated`: today's date
   - `Current phase`: what phase/task you're on
   - `Completed`: add any tasks finished this session
   - `Next tasks`: what should be done next (reference Phase 0-6 plan)
   - `Known issues`: any blockers or problems discovered
   - `Test status`: results from step 1

3. **Stage and commit**:
   ```bash
   git add -u
   git add PROGRESS.md
   git commit -m "Session handoff: <summary of what was done>

   Next: <what the next session should work on>"
   ```

4. **Report handoff state**:
   ```
   ## Session Handoff Complete
   - Completed this session: <list>
   - Tests: N passed, N failed
   - Next session should: <task>
   - Blockers: <list or "none">
   ```
