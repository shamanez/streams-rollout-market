---
name: seed-from-plan
description: Take a plan (either drafted in Plan Mode and just Accepted, or saved as a markdown file) and seed `STEER.md` + `.claude/feature-results.json` so `/autonomous-loop` can drive it. Loose format — works with whatever shape the plan came out of Plan Mode in. Pass an optional file path; otherwise read the most recent plan from conversation context.
user-invocable: true
allowed-tools: Read, Write, Edit, Bash, Glob, Grep
---

# Seed-from-Plan

After Plan Mode is Accepted, the plan only lives in conversation
context until you save it. This skill closes that gap: it takes the
plan (from context OR a file), writes `STEER.md` and
`.claude/feature-results.json`, commits, and leaves the queue ready
for `/autonomous-loop`.

## Input

* `$ARGUMENTS` empty → use the most recent plan in this conversation
  (typically the body of the last `ExitPlanMode` call you Accepted).
* `$ARGUMENTS = path/to/plan.md` → read from a file.

Either way, the plan is just markdown. There is no required template.

## What's "the plan"

Anything that includes a list of distinct iterations the loop can
walk. The skill auto-detects iteration boundaries from markdown
headings or numbered lists. Helpful but not mandatory shape:

```markdown
# <title>            (or any text — not required)

<motivation, acceptance gates, global rules — any structure>

## Iteration 1 — <slug>: <title>      (or "### 1. <slug>: <title>"
                                        or "1. <slug>: <title>"
                                        or "- [ ] <slug>: <title>")
<body — what to build, files to touch, acceptance criteria>

## Iteration 2 — <slug>: <title>
...
```

You pick the iteration count and slug naming. The skill respects what
you wrote.

### Slug extraction

For each iteration block, derive a slug:
* Prefer an explicit `<slug>:` token in the heading
  (e.g. `### 1. dashboard.add_real_tools: …`).
* If absent, derive one from the heading (lowercase, replace
  whitespace with `_`, strip non-`[a-z0-9_.-]`).
* If two iterations produce the same slug, append `_2`, `_3`, ….
* Final slug must match `^[a-z0-9_.-]+$`.

## Workflow

1. **Resolve the plan source**:
   * If `$ARGUMENTS` is a file path that exists → read it.
   * Otherwise → use the most recent plan you drafted in this
     conversation (typically the body of the last `ExitPlanMode`).
2. **Refuse to overwrite an active queue** (idempotency guard): if
   `STEER.md` exists OR `.claude/feature-results.json` contains any
   entry with `passes:false`, halt and report. Offer the operator:
   * confirm explicit overwrite (then proceed), or
   * clear first with
     `echo '{}' > .claude/feature-results.json && rm -f STEER.md`.
3. **Check the architecture firewall**: scan the plan body for
   trainer-code patterns (PPO, GRPO, FSDP, Megatron, etc. outside
   tests/examples), AWS infra mutations, paid-tier API mentions. If
   any are present, refuse to seed and report which iteration is the
   problem.
4. **Extract iterations**: walk the plan markdown, split into blocks
   (one per iteration heading you detect), pull out slug + body.
   Validate: ≥1 iteration, each has a body. Reject on slug
   collisions (after suffix attempt) or zero iterations.
5. **Write `STEER.md`** as the plan itself, with a short header
   prepended:

   ```markdown
   # STEER.md — Operator Redirect

   Seeded at <UTC timestamp> from <plan source: file path or
   "Plan Mode (this conversation)">.

   Stop clause: when every entry in `.claude/feature-results.json`
   has `passes:true`, delete this STEER.md as the final commit.

   ---

   <verbatim plan markdown>
   ```

   Keeping the plan body verbatim means the operator can read STEER.md
   later and see exactly what they Accepted.

6. **Write `.claude/feature-results.json`** as a flat JSON object —
   one entry per extracted iteration, file order matching plan order:

   ```json
   {
     "<slug-1>": {
       "passes": false,
       "evidence": null,
       "directive": "<full iteration body, including any sub-headings>"
     },
     ...
   }
   ```

   The `directive` is what the implementer reads at the start of each
   iteration; include enough context to act on it.

7. **Commit** with a clear message:
   ```
   chore(queue): seed from plan (<N> iterations)

   STEER.md armed. Run /autonomous-loop to drive.
   ```

8. **Report** to the operator:
   ```
   Seeded queue (<N> iterations):
     1. <slug-1>: <one-line title>
     2. <slug-2>: <one-line title>
     ...

   Next: type /autonomous-loop
   ```

## Constraints

* **Do not silently rewrite the plan.** STEER.md must hold the plan
  verbatim. Any structural reshaping (e.g. inferring slugs) must be
  visible in the iteration block / report so the operator can sanity
  check.
* **Idempotency**: do not seed on top of an active queue without
  explicit overwrite confirmation.
* **Architecture firewall**: refuse to seed plans that would have an
  iteration violate AGENTS.md (no trainer code, no AWS infra
  mutations, no paid-tier API entries).
* **Atomicity**: if `STEER.md` writes but `feature-results.json` fails
  (or vice versa), undo the partial write so the queue is either
  fully armed or fully unarmed — never half.
