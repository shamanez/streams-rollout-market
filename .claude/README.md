# `.claude/` — autonomous infrastructure map

This directory implements the four canonical primitives from
[anthropics/cwc-long-running-agents](https://github.com/anthropics/cwc-long-running-agents)
plus this project's local extensions (firewall, AWS guard, evidence
tracking). The thesis:

> An agent you can leave running for hours needs more than a good prompt.
> It needs enforcement that "done" means observed evidence, an independent
> check on its own claims, and a way for the next session to pick up
> cleanly where this one left off.

## Current state (2026-05-12)

- **`feature-results.json`**: 23/23 entries `passes: true`. The
  STEER cycle that drove the dashboard to 8 green/amber matrix tiles
  is complete; `STEER.md` self-deleted per its own stop clause.
- **`evidence/`**: one subdirectory per flipped entry (rollout +
  trainer JSONs + dashboard renders + codex verdicts).
- **Next work** lives in `../PROGRESS.md` "Next work" and
  `../docs/future_research.md`. Re-arm STEER with
  `bash scripts/steer.sh "<directive>"` to redirect the loop.

## Primitive → file map

| cwc primitive               | file(s) in this repo                                       |
|-----------------------------|------------------------------------------------------------|
| **Default-FAIL contract**   | `feature-results.json` + `scripts/verify-result-write.sh`  |
| **Fresh-context evaluator** | `agents/evaluator.md`                                      |
| **Agent-maintained handoff**| `../PROGRESS.md` + `scripts/commit-on-stop.sh`             |
| **Kill switch**             | `scripts/kill-switch.sh` + `../AGENT_STOP` (sentinel)      |
| **Steer**                   | `scripts/steer.sh` + `../STEER.md`                         |

Local extensions:

| extension                   | file(s)                                                    |
|-----------------------------|------------------------------------------------------------|
| Architecture firewall       | `scripts/firewall-check.sh`                                |
| AWS guard                   | `scripts/aws-guard.sh`                                     |
| Evidence tracking           | `scripts/track-evidence.sh` + `evidence/evidence_log.jsonl`|
| Session start banner        | `scripts/session-start.sh`                                 |

## Default-FAIL contract — how it works

`feature-results.json` is a flat dict, one entry per success criterion.
Every entry starts as:

```json
"<key>": { "passes": false, "evidence": null, "directive": "<task spec>" }
```

The autonomous loop reads the first `false` entry, runs the implementer
agent against its `directive`, captures evidence under `.claude/evidence/`,
then writes a new version of the JSON file flipping that entry to
`{ "passes": true, "evidence": "<path>" }`.

The PreToolUse hook `scripts/verify-result-write.sh` intercepts the write
and exits non-zero unless:

1. The evidence path exists on disk.
2. The evidence path (or its basename) appears in
   `evidence/evidence_log.jsonl` (the rolling log of `pytest`/`ruff` and
   other `track-evidence.sh`-captured events).

Agents cannot mark work complete without observable proof.

## Skills

| skill                  | what it does                                              |
|------------------------|-----------------------------------------------------------|
| `autonomous-loop`      | Self-pacing loop; reads STEER.md → feature-results.json   |
| `implement-feature`    | Contract-first feature implementation; flips one entry    |
| `build-and-test`       | `pytest -q && ruff check . && ruff format --check .`      |
| `evaluate`             | Fresh-context evaluator subagent                          |
| `codex-review`         | Delegate review to Codex CLI (token-saving)               |
| `session-handoff`      | Update PROGRESS.md, commit, close out                     |

## Subagents

| agent                  | role                                                       |
|------------------------|------------------------------------------------------------|
| `evaluator.md`         | Read-only grader — no Write/Edit (cwc canonical)           |
| `implementer.md`       | Opus, Write/Edit/Bash; contract-first                      |
| `code-reviewer.md`     | Reviews diffs for AGENTS.md compliance                     |
| `test-runner.md`       | Haiku; runs build-and-test                                 |

## Operator controls

Halt:
```bash
touch AGENT_STOP        # or
bash .claude/scripts/kill-switch.sh
```

Resume:
```bash
rm AGENT_STOP
```

Redirect mid-run:
```bash
bash .claude/scripts/steer.sh "Run n=30 dense matrix next"
# or write STEER.md directly
```

## Hooks (declared in `settings.json`)

| event           | matcher       | script                              |
|-----------------|---------------|-------------------------------------|
| `SessionStart`  | —             | `scripts/session-start.sh`          |
| `PreToolUse`    | `Write\|Edit` | `scripts/firewall-check.sh`         |
| `PreToolUse`    | `Write\|Edit` | `scripts/verify-result-write.sh`    |
| `PreToolUse`    | `Bash`        | `scripts/aws-guard.sh`              |
| `PreToolUse`    | (all)         | `scripts/kill-switch-check.sh`      |
| `PostToolUse`   | `Bash`        | `scripts/track-evidence.sh`         |
| `Stop`          | —             | `scripts/commit-on-stop.sh`         |

## Quality-loop trace (monitoring from another terminal)

```bash
watch -n 5  'tail -30 PROGRESS.md'                             # handoff
watch -n 5  'git log --oneline -8'                             # commits
watch -n 10 'tail -5 .claude/evidence/evidence_log.jsonl'      # evidence
watch -n 5  'jq ".[] | select(.passes==false) | input_line_number" \
              .claude/feature-results.json | head -3'          # next-up
```

## Provenance

This layout follows the cwc-long-running-agents reference architecture:

> "These are example ingredients, not a turnkey harness. … Each primitive
> is standalone with no interdependencies — copy only what fits your
> harness."

Each primitive in the table above is cherry-picked and adapted for the
`streams-rollout-market` repo; the cwc README is the canonical reference
for the shape of each one.
