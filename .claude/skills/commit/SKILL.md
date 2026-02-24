---
name: commit
description: Create a git commit with careful staging verification. Use when the user asks to commit changes.
disable-model-invocation: true
argument-hint: [message]
---

# Create a Commit

**IMPORTANT**: Before every commit, verify that only the relevant files are staged. Never blindly commit whatever happens to be staged — always review the staged file list and confirm it matches the user's intent.

## Pre-commit checklist

1. Run `git status` to see staged, unstaged, and untracked files
2. Review the staged files — are they all relevant to this commit? If unrelated files are staged, unstage them first
3. If no files are staged, ask the user which files to include
4. Show the user the list of files that will be committed and get confirmation before proceeding

## Commit message

- If `$ARGUMENTS` is provided, use it as the commit message
- Otherwise, draft a concise message based on the staged changes
- Follow the project's existing commit message style (see `git log --oneline -5`)
- Add `Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>` trailer

## After committing

- Run `git status` to confirm the result
- Report what was committed
