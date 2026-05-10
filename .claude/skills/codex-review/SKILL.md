---
name: codex-review
description: Delegate code review to Codex to save Claude tokens. Uses /codex:review for standard reviews or /codex:adversarial-review for design-challenge reviews. Best for bounded, read-only review tasks.
user-invocable: true
allowed-tools: Bash, Read
---

# Codex Review (Token-Saving)

Delegate code review to Codex instead of using Claude's evaluator/code-reviewer agents.
This saves Claude tokens for implementation work where Opus reasoning matters most.

## When to use this vs Claude agents

- **Use Codex**: routine reviews, lint-style checks, convention compliance, PR reviews
- **Use Claude evaluator**: when you need deep AGENTS.md firewall analysis or plan alignment checks
- **Never use Codex for**: open-ended implementation or rescue tasks (burns tokens fast)

## Usage

Standard review of recent changes:
```
/codex-review
```

With arguments:
```
/codex-review --base main          # compare against main branch
/codex-review --adversarial        # challenge design decisions
/codex-review --effort low         # minimize token usage
```

## Steps

1. **Assess scope** first:
   ```bash
   git diff --stat HEAD~1
   ```
   If more than 10 files changed, use `--effort low` to conserve tokens.

2. **Run Codex review**:
   - Default: `/codex:review --wait --effort medium`
   - If `$ARGUMENTS` contains `--adversarial`: `/codex:adversarial-review --wait`
   - If `$ARGUMENTS` contains `--base`: pass through to Codex

3. **Report results** verbatim from Codex (do not summarize or paraphrase).

4. **If Codex fails or times out**: fall back to the Claude `code-reviewer` agent.

## Token conservation tips

- Use `--effort low` or `--effort minimal` for quick sanity checks
- Use `--scope working-tree` to limit review to uncommitted changes only
- Avoid `--effort high` or `--effort xhigh` unless reviewing critical code
- Never use `/codex:rescue` for implementation — that's where token burn happens
