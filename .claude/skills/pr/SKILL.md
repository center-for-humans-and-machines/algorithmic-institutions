---
name: pr
description: Create a pull request with a concise, meaningful summary. Use when the user asks to create a PR, open a pull request, or push changes for review.
disable-model-invocation: true
argument-hint: [base-branch]
---

# Create a Pull Request

## Approach

PR messages should highlight only meaningful structural changes -- omit incidental or mechanical side-effects that are obvious from the diff. The test plan should focus exclusively on what genuinely needs human judgement (e.g. correctness, usefulness), not mechanical checks that are self-evident. Keep everything concise: if a change doesn't need explaining, don't mention it.

## Format

```
## Summary
<2-4 bullet points covering only meaningful changes>

## Test plan
<Checkbox list of ALL verification steps. The agent must run every test it
can (unit tests, linting, pre-commit) and check them off. Items that
require manual testing remain unchecked.>
- [x] `pre-commit run --all-files` -- passed
- [x] `pytest` -- N passed
- [ ] Manual: <short description of what to verify>

## Human verification
<This section is REQUIRED when the test plan has unchecked manual items OR
when the PR contains consequential design decisions. It serves two purposes:>

**Manual testing instructions** (when applicable):
<For EVERY unchecked item in the test plan, provide a step-by-step
walkthrough: the exact command(s) to run, what output to expect, and what
to look out for (e.g. "you should see X in the terminal", "check that Y
does NOT appear", "the exit code should be 0"). The reviewer should be
able to verify the feature by copy-pasting commands without reading the
code first.>

**Design decisions** (when applicable):
<Only include when changes have long-term consequences for the codebase
trajectory or implications for research outcomes. For each item:>
- What was changed and why (with code references, e.g. `src/foo.py:42`)
- What alternative was considered and why it was not chosen
- What the downstream consequences are if this turns out to be wrong
```

**When to include "Human verification"**: Always include when there are unchecked manual test items. Also include when the PR contains decisions that could affect research validity, architectural choices that are hard to reverse, or trade-offs where reasonable people might disagree. Omit only for routine changes where all tests are automated and pass.

**IMPORTANT**: Do NOT add "Generated with Claude Code" or any AI attribution footer to the PR message.

**IMPORTANT**: Never check off a test item (`[x]`) unless you have actually run it yourself in this session and verified it passed. If you did not run it, leave it unchecked (`[ ]`). Claiming tests passed without running them is worse than leaving them unchecked.

## Images

PRs that touch training pipelines, evaluation, or simulation should **always** include relevant plots. Check `plots/` for new or changed files on the branch (`git diff --name-only main..HEAD -- plots/`) and embed them in the summary. Visual results are a key part of communicating what the PR achieved.

When embedding images in the PR body, **always** use `raw.githubusercontent.com` URLs with the **commit SHA** (from `git rev-parse HEAD`) — branch names break after the branch is deleted post-merge.

```
<!-- Wrong: relative path -->
![Plot](plots/example.png)

<!-- Wrong: branch name (breaks after merge + branch deletion) -->
![Plot](https://raw.githubusercontent.com/center-for-humans-and-machines/algorithmic-institutions/<branch>/plots/example.png)

<!-- Correct: commit SHA permalink -->
![Plot](https://raw.githubusercontent.com/center-for-humans-and-machines/algorithmic-institutions/<commit-sha>/plots/example.png)
```

## Instructions

1. Examine all commits on the current branch vs the base branch (default: `main`, or `$ARGUMENTS` if provided)
2. Draft a short PR title (under 70 chars) and body following the format above
3. Present the draft and ask the user to review before creating the PR
4. Once approved, push and create the PR using `gh pr create`
