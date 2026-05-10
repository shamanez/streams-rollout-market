---
name: evaluate
description: Run the evaluator agent to independently grade recent work. Use after completing implementation to get an unbiased quality assessment.
user-invocable: true
allowed-tools: Read, Bash
---

# Evaluate Recent Work

Invoke the evaluator sub-agent for independent quality grading.

## Steps

1. **Invoke the evaluator agent** to grade recent work:
   - The evaluator reads PROGRESS.md, git log, changed files
   - It checks AGENTS.md compliance, conventions, test coverage
   - It returns structured PASS/NEEDS_WORK/FAIL grades

2. **Review the evaluation results**

3. **If any FAIL grades**:
   - Flag them in PROGRESS.md under "Known issues"
   - Report the specific issues that need fixing

4. **If all PASS**:
   - Confirm the work is ready
   - Report the clean evaluation

## Output format

```
## Evaluation Results
- Evaluator verdict: <PASS | NEEDS_WORK>
- Tasks graded: N
- Passed: N
- Needs work: N
- Critical issues: <list or "none">

### Details
<evaluator's full report>
```
