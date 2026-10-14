---
name: policy-finder-summarizer
description: Bring the policy-finder observation app's Summary up to date (new_instance.sh runs it headless after every sweep). Reads the rule config and sweep of every instance version not yet summarized, places each rule in the Summary's categories (adding one only when a rule needs it), records it in summary.json beside the instances, and republishes the shared artifact. Use when the user asks to update, refresh or redo the policy-finder summary.
argument-hint: "[artifact url]"
---

Update the Summary of the policy-finder observation app.

$ARGUMENTS

The Summary's categories (`summary.json`) and its reference pools
(`references.json`) are kept on Raven, in `/u/certuer/policy-finder-observatory/`
(one fixed folder for everyone, reached through the `raven` ssh alias), with
working copies beside the instances in
`../policy-finder-worktrees/`. `summarize.py` keeps the two in step: `status`
pulls from Raven, and every `record` or `category` pulls, writes and pushes
back. None of it is ever committed to `policy-finder-base`: every agent starts
from that branch and can read all of it. The live app reads the working
copies on every load; the shared artifact is a snapshot that this skill
re-exports. Numbers (pool, se, members) and the inputs a rule reads are not
stored: the app reads them from each sweep and rule.

If `summarize.py` warns that Raven is unreachable, it works on the local
copies: say so in the report, and run `summarize.py push` once `ssh raven`
is up again.

1. **What's new.** Run

   ```bash
   uv run --no-sync python scripts/policy_finder/observation_app/summarize.py status
   ```

   It prints the categories with their meanings, then only the versions to
   summarize: a rule never recorded, or changed since (by its hash), with the
   paths of its rule and sweep. If nothing is listed, skip to step 4: the
   summary already covers every rule.

2. **Read each listed version.** Each version of an instance is summarized
   on its own (`<name>`, then `<name>-v2`, ...): its rule config (`params`,
   `sweep_config`, `code`) and, if validated, its final best
   (`validate/sweep.json` in its sweep dir). Do not change any instance's
   files.

3. **Place and record it.** Decide, from the rule's code:
   - its **bar**, who it punishes: one `bar` category
   - its **mechanisms**: every `mechanism` category it uses
   - its **idea**: one plain sentence a colleague can read without the code,
     like the existing ones in `summary.json`

   Reuse the existing categories. Add a new one only when a rule does
   something none of them describes, and give it a one-line meaning:

   ```bash
   uv run --no-sync python scripts/policy_finder/observation_app/summarize.py \
       category "<name>" --kind bar|mechanism --meaning "<one line>"
   ```

   Then record the version:

   ```bash
   uv run --no-sync python scripts/policy_finder/observation_app/summarize.py \
       record <version> --bar "<category>" --mechanism "<category>" ... --idea "<one line>"
   ```

4. **Republish the artifact**, when this session has the Artifact tool.
   A headless run (the launcher's, after a sweep) has none: skip this step
   and say so; the session that launched the instance republishes it.
   Export the page as a fragment into the session's scratchpad (or
   `$CLAUDE_JOB_DIR/tmp`), then publish it with the Artifact tool to the
   shared artifact's URL: the argument if given, else
   https://claude.ai/artifact/QUAxcas7ui7Dv3ihz87NHr. Read that artifact
   first (`action: "read"`) when this conversation has not, as the tool
   requires.

   ```bash
   uv run --no-sync --with streamlit python \
       scripts/policy_finder/observation_app/export_html.py --fragment --out <tmp>/policy-finder-instances.html
   ```

5. **Report** in a few lines: which versions were added (or none), any new
   category and why, and how the category counts changed. The live app needs
   only a reload.
