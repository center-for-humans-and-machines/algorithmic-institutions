# AI-Assisted Development Workflow

This project uses [Claude Code](https://claude.com/claude-code) with
specialized agents to accelerate development. The workflow is designed
so that a human collaborator stays in control of all decisions while
agents handle implementation work.

## How it works

1. **Issues drive everything.** Every piece of work starts as a GitHub
   issue. Labels control what happens next.

2. **Architect agent** reads a well-specified issue, explores the
   codebase, and writes an implementation plan to `doc/plans/`. The
   plan is committed to a `plan/*` branch and linked in an issue
   comment. The issue is labeled `human-plan-review`.

3. **Human reviews the plan.** Approve, request changes, or reject.
   Once approved, relabel the issue to `engineer-agent-ready` or
   `data-analyst-agent-ready`.

4. **Engineer / Data analyst agent** picks up the approved plan,
   implements it in an isolated worktree, runs tests and linting,
   creates a PR with a checkbox test plan, comments on the issue, and
   labels it `human-review`.

5. **Reviewer agent** checks out the PR branch, runs automated tests
   (`pytest`, `pre-commit run --all-files`), and updates the PR test
   checkboxes. It also reviews the diff for code duplication and dead
   code, leaving findings as PR comments. It never modifies source code.

6. **Human reviews the PR.** Automated tests are already checked off
   by the reviewer agent; unchecked items need manual verification
   (detailed instructions are in the PR body). Merge when satisfied.

## Label lifecycle

```
architect-agent-ready  →  human-plan-review  →  engineer-agent-ready
                                                 data-analyst-agent-ready
                                              →  human-review  →  (merged/closed)
```

- `architect-agent-ready` — issue is well-specified, ready for
  planning
- `human-plan-review` — architect wrote a plan, awaiting human
  approval
- `engineer-agent-ready` / `data-analyst-agent-ready` — plan approved,
  ready for implementation
- `human-review` — PR is open, awaiting human review
- `human-specification-required` — issue needs clarification before
  any agent work

## Agents

| Agent | Role | Config |
|-------|------|--------|
| **Architect** | Explores codebase, writes plans | `.claude/agents/architect.md` |
| **Engineer** | Implements features, writes tests | `.claude/agents/engineer.md` |
| **Data Analyst** | Implements analysis pipelines, visualisations | `.claude/agents/data-analyst.md` |
| **Literature Researcher** | Searches and synthesizes academic literature | `.claude/agents/literature-researcher.md` |
| **Reviewer** | Runs tests, checks PR checkboxes, reviews for code quality | `.claude/agents/reviewer.md` |

Agents run in isolated git worktrees so they cannot interfere with
each other or with your working tree. Permissions are configured in
`.claude/settings.json`.
