# Policy finder: orchestration and reporting

Handover sub-report 4. Scope: how instances are launched, sandboxed, swept on
Raven and reported. The numerics of `manager/rule.py`, the sweep, the GP fit
and the validation are covered elsewhere. All refs are to
`origin/policy-finder-base` (794a25c5, 2026-10-15). `yannik/handover` contains
it, and its copies of these files match.

## TL;DR

- One instance is one git branch `policy-finder/<name>` (not `pf-*`). It is
  cut from local branch `policy-finder-base` and checked out as a worktree in
  `../policy-finder-worktrees/<name>`. A headless `claude -p` runs a Sonnet
  relay (`policy-finder-host`), which spawns an Opus subagent
  (`policy-finder`). The subagent may write only its rule, its notes and a
  scripts folder.
- A launcher on the host machine does everything after the agent: check,
  commit, sweep, GP fit, validation and summarizer. It drives Raven with
  `ssh`/`rsync`/`sbatch` through an SSH ControlMaster. Each branch gets its
  own remote dir, `~/ai-isolated/policy-finder--<name>`. There are no slurm
  arrays: each sweep part (2 for 256 points) and the validation are separate
  `sbatch` jobs.
- The snapshot page comes from `observation_app/export_html.py`. It reads the
  worktrees and five files kept outside git (`summary.json`,
  `references.json`, `overview.html`, `<version>.report.md`, Claude session
  logs).
- **None of the 13 instance branches is in this clone or on GitHub.** The
  fetch of 2026-10-23 lists 117 remote branches and no `policy-finder/*`.
  They live only in cemrtkn's clone at `/Users/ertuerkan/Desktop/` (paths
  embedded in the snapshot), with sim outputs under
  `/u/certuer/ai-isolated/` on Raven.
- Under the command-server model the pipeline fails before any cluster step.
  The first failure is `new_instance.sh:112`, "no branch policy-finder-base",
  because this clone has only `origin/policy-finder-base`. The first cluster
  failure is `run_sweep.sh:55`, `ssh -O check raven`. Every Raven call
  (sync, submit, poll, fetch, summary store) needs an adapter. The macOS
  sandbox settings also need a Linux rewrite before an agent can safely run
  in this container.

## Lifecycle

```
 /policy-finder skill (interactive session, SKILL.md)   picks name pf-YYYYMMDD-n
        |  runs in background:
        v
 new_instance.sh <name> [--iterate-on] --headless        (host, outside sandbox)
   |- git worktree add -b policy-finder/<name>  ../policy-finder-worktrees/<name>  policy-finder-base
   |   (--iterate-on: git merge policy-finder-base into the existing branch instead, version k+1)
   |- git lfs pull --include experiments/...csv, plots/simulation/25_LEVIN_run1_*/**
   |- instance_settings.py -> <wt>/.claude/policy_finder.json + settings.local.json (gitignored)
   |- claude -p "Design your rule." --agent policy-finder-host --strict-mcp-config --allowedTools=Bash
   |      host (sonnet, relay only) --Agent--> policy-finder (opus, no CLAUDE.md, sandbox + guard)
   |      writes: configs/managers/rule_based/<stem>.yml, notes/policy_finder/<stem>.md,
   |              scripts/policy_finder/<name>/**      report -> ../policy-finder-worktrees/<stem>.report.md
   |- check_instance.sh <name> --commit         git diff since start_commit, notes headings, validate-rule
   |- run_sweep.sh <name>
   |      generate_sim_config.py (2 parts) -> simulate_cluster.sh --sync-only (rsync -> raven:~/ai-isolated/<branch>)
   |      simulate_cluster.sh --no-sync part.yml  (ssh: python run.py -> sbatch run_simulation.sh)  x N parts
   |      poll ssh raven squeue /60 s -> rsync parts back -> merge_sweeps.py -> fit_surrogate.py -> git commit
   |      run_validate.sh <name>: generate --validate -> sync -> sbatch -> poll -> rsync validate/ -> git commit
   |- claude -p /policy-finder-summarizer (main checkout)  summarize.py status/record <-> raven:/u/certuer/policy-finder-observatory
 interactive session: export_html.py --fragment -> Artifact publish (shared page)
```

Stem: `<name>` for v1, `<name>-v<k>` for later versions
(`instance_settings.py:48-50`). A lineage is one instance and its versions.
All versions share the branch, the worktree and `scripts/policy_finder/<name>/`.

## 1. Instance lifecycle scripts

**`scripts/policy_finder/new_instance.sh`**
`<name> [--max-params N] [--min-params N] [--sobol-points N] [--no-start | --headless [--prompt T]]`
or `--iterate-on <name> ...`

- `:104-110` derive the paths:
  - `MAIN` is the main checkout.
  - `BASE=${PF_BASE:-policy-finder-base}`.
  - `BRANCH=policy-finder/$NAME`.
  - `WT_ROOT=${PF_WORKTREE_ROOT:-$(dirname MAIN)/policy-finder-worktrees}`.
  - `PYTHON=$MAIN/.venv/bin/python`.
  - `UV_PYTHON_DIR=~/.local/share/uv/python`.
- `:112-114` require the local branch `$BASE` and the `.venv` Python.
- `:118-151` handle `--iterate-on`:
  1. The worktree must be clean.
  2. The previous stem must have `..._sweep/validate/sweep.json`.
  3. Then `VERSION=k+1`.
  4. It runs `git merge $BASE` into the instance branch, so a new version
     gets the latest agent and code.
  5. With no param flags, params run from 1 with no upper bound.
     `--sobol-points` defaults to the previous version's.
- `:153-163` new instance: `git worktree add -b policy-finder/<name> <wt> $BASE`, `VERSION=1`.
- `:164` `git lfs pull` of the two inputs the agent reads.
- `:166-169` `instance_settings.py` with `--start-commit $(git rev-parse HEAD)`.
- `:177-194` headless mode:
  1. `claude -p "$PROMPT" --agent policy-finder-host --strict-mcp-config --allowedTools=Bash < /dev/null > $WT_ROOT/$STEM.report.md`
  2. `check_instance.sh --commit`
  3. `run_sweep.sh`
  4. In `MAIN`: `claude -p "/policy-finder-summarizer" --strict-mcp-config --add-dir $WT_ROOT --allowedTools=Bash,Read,Artifact > $WT_ROOT/$STEM.summary.md`.
     A failure here is non-fatal.
- `:196` interactive mode: `exec claude --agent policy-finder-host --strict-mcp-config`.

**`scripts/policy_finder/instance_settings.py`** writes two gitignored files
(`.gitignore:34-35`) into `<wt>/.claude/`:

- `policy_finder.json` (`:137-146`): `name`, `version`, `rule_name`,
  `start_commit`, `min_params`, `max_params` (null means unbounded),
  `sobol_points` and `python`.
- `settings.local.json` (`:76-129`):
  - `autoMemoryEnabled: false`.
  - allow only `Edit(/<rule>)`, `Edit(/<notes>)` and `Edit(/<scripts>/**)`.
  - deny `Read(/.git/**)`, `Read(//<main>/**)`, `Read(//~/.claude/**)`,
    `WebFetch` and `WebSearch`.
  - sandbox: enabled, `autoAllowBashIfSandboxed`, no unsandboxed escape;
    `denyRead [$HOME, /private/tmp, /Volumes, <wt>/.git]`;
    `allowRead [<wt>, uv python, <main>/.venv, /private/tmp/claude-<uid>/<wt-dashed>]`;
    `denyWrite` = every sibling of the write paths at every level, plus
    `<main>/.git` (`complement()`, `:63-73`).
  - a `PreToolUse` hook on `*`: `python3 <wt>/.claude/hooks/policy_finder_guard.py`.

**`scripts/policy_finder/check_instance.sh <name> [--commit]`**
(local only, no cluster):

- `:53-61` read the current version from the instance file.
- `:73-93` fail if any path changed outside these, either since
  `start_commit` (`git diff $START $BRANCH`) or in the worktree
  (uncommitted or untracked):
  - the rule, the notes and `scripts/policy_finder/<name>/`;
  - the version's own sweep configs and outputs.
- `:101-109` the notes need `## Explorations`, `## Key findings`,
  `## Hypothesis` and `### Justification`, in that order.
- `:112-117` run `PYTHONPATH=$WT/src $PY -m aimanager validate-rule <rule> --min-params --max-params --sobol-points`.
- `:121-131` `--commit` commits the write paths on the branch. The agent has
  no git.

**`scripts/policy_finder/run_sweep.sh <name>`** (host):

- `:28-30` constants: `N_EPISODES=500`, `MAX_JOB_EPISODES=125000` (one 16 GB
  job), `POLL_SECONDS=60`.
- `:53-56` `REMOTE=$(raven_remote_dir $WT ~/algorithmic-institutions)`, then
  `ssh -O check raven`.
- `:59-68` `N_PARTS=ceil(sobol*500/125000)`, which gives 2 parts for 256
  points and 3 for 512. Then `generate_sim_config.py --n-parts` writes
  `configs/simulation/policy_finder/<stem>_sweep_p<k>of<N>.yml`.
- `:71-79` sync once, then one `simulate_cluster.sh --no-sync <part>.yml` per
  part. The job id is parsed from `Submitted batch job N`.
- `:83-88` loop on `ssh raven 'squeue -h -u $USER -o %i'` until no job is left.
- `:92-102` `rsync -az raven:$REMOTE/plots/simulation/policy_finder/<part>/ $WT_ROOT/<stem>.sweep_parts/<part>/`.
  A part with no `sweep.json` counts as failed.
- `:105-115` `merge_sweeps.py <parts> --out <wt>/plots/.../<stem>_sweep/sweep.json`,
  then move each part's parquet to `per_round_p<k>ofN.parquet`.
- `:118` `fit_surrogate.py <sweep dir>` writes `landscapes.json`.
- `:121-123` `git add` the sweep dir and sim configs, then
  `git commit "policy-finder <stem>: sweep"`.
- `:127` call `run_validate.sh`.

**`scripts/policy_finder/run_validate.sh <name>`**:

- `:49-54` needs `landscapes.json`, then the same remote-dir and ssh check.
- `:57-58` `$MAIN/.../generate_sim_config.py --validate <sweep>/sweep.json --root $WT`
  (GP peak plus top 5, 2000 episodes, seed 1000).
- `:61-71` sync, submit one job. `:68-69` is a fallback for old instances that
  finds the job by name with `squeue -n`.
- `:75-77` poll.
- `:80-84` rsync `<stem>_sweep/validate/`.
- `:87-89` commit.
- `:96-111` print the final best against the lineage's best earlier version.

**`scripts/policy_finder/merge_sweeps.py`**:

- `:21` and `:30-44` check that the dirs are parts 1..N of a single sweep.
- `:47-57` `aimanager.simulation.sweep.merge_sweeps` builds the global
  `best`, then `copy_best_plots` copies the plots of the part that holds it.

### How Raven is reached (manual SSH path only)

- `.raven_remote_dir` holds `~/ai-isolated/{branch}`. It was added in 6aabf5c3
  so that a sync of base, with `rsync --delete`, never replaces the shared
  checkout.
- `scripts/raven_remote_dir.sh:18-36` resolves the remote dir in this order:
  1. `$AI_REMOTE_DIR`;
  2. the template with `/` written as `--` (`:32`), for example
     `~/ai-isolated/policy-finder--pf-20261008-3`;
  3. the shared checkout. A detached HEAD fails.
- `scripts/simulate_cluster.sh`:
  - `:59` `ssh -O check raven`.
  - `:76-78` for an isolated dir: `ssh mkdir -p` and `cp -n ~/algorithmic-institutions/.env`.
  - `:81-91` `rsync -azP --delete` filtered by `.gitignore`, excluding
    `.git .venv .log plots temp artifacts/manager`.
  - `:100-115` `ssh raven "bash -l -c 'cd <remote> && source ~/algorithmic-institutions/.venv/bin/activate && export AIMANAGER_VENV=... PYTHONPATH=<remote>/src SBATCH_EXPORT=ALL && python src/aimanager/simulation/run.py <config>'"`.
- `src/aimanager/simulation/run.py:44-74` fills `scripts/run_simulation.sh`
  into `.log/<config>/<uuid8>/run.sh`. It requests 2 CPUs, 16 GB, one A100,
  1 h and `module load cuda/11.4`, then runs
  `python -m aimanager simulate <config>`. It submits with `sbatch` (`:68`).
- `scripts/fetch_cluster.sh:71-73` is a plain `rsync raven:<remote>/<path>`.
  It is not used by `run_*.sh`, which rsync directly.
- The `/simulate` and `/fetch-cluster` skills are thin wrappers around those
  scripts.

## 2. The agent and its sandbox

- **`.claude/agents/policy-finder-host.md`** uses `model: sonnet`. Its only
  tools are `Agent(policy-finder)` and `SendMessage`. It relays the prompt
  verbatim and returns the subagent's report verbatim, which becomes
  `<stem>.report.md`.
- **`.claude/agents/policy-finder.md`**:
  - `:4-7` front matter: `model: opus`, `omitClaudeMd: true` (it never sees
    the repo CLAUDE.md or Levin's rules), tools
    `Read Grep Glob Write Edit NotebookEdit Bash TodoWrite`. Disallowed:
    Web*, Agent, Skill, Workflow, SendMessage, Artifact.
  - `:8-13` its own copy of the guard hook.
  - What it reads (`:59-72`):
    - `experiments/2group_8agent_50ep.csv`
    - `plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/`
    - `reports/`, `notes/` and `src/aimanager/`
    - from v2 on, earlier versions' rules, notes, `sweep.json`,
      `per_round_*.parquet`, `landscapes.json` and `validate/` (`:74-105`).
  - Write paths (`:50-57`): the rule, the notes and the scripts folder.
  - It never runs the simulation (`:33-38`).
- **`.claude/hooks/policy_finder_guard.py`** fails closed (`:141`). It enforces:
  - writes only to the rule, notes and scripts paths (`:76-86`);
  - Read, Grep and Glob only inside the worktree, never `.git` (`:89-109`);
  - no `git` or `gh` word in any Bash command (`:37`, `:112-117`);
  - no Web* or `mcp__*` tool (`:126`).
- **Three layers enforce this.** Permission rules, the Claude Code Bash
  sandbox (filesystem plus no network) and the guard. In headless mode Bash is
  pre-allowed (`--allowedTools=Bash`), so the sandbox alone confines the
  agent's Python.
- **Information hygiene:**
  - #235 deleted the old rule-based managers from base.
  - `summary.json`, `references.json` and `overview.html` stay outside every
    worktree.
  - The skill forbids passing an instance anything about other instances
    (`SKILL.md:48-51`).
- **`.claude/settings.json`** (`respectGitignore: false`) exists only on
  `yannik/handover` and is unrelated. This checkout's `settings.local.json`
  is a stray personal allowlist. Instance settings are generated per worktree.
- **`.claude/skills/policy-finder/SKILL.md`**:
  - `:12-24` maps the arguments.
  - `:25-31` naming: `pf-<YYYYMMDD>-<n>`, the first n with no branch or
    worktree.
  - `:33-41` checks `ssh -O check raven` first, then launches
    `new_instance.sh ... --headless` in the background.
  - `:57-73` the report reads only `validate/sweep.json` for the final best.
  - `:74-77` the interactive session republishes the page.
- **`.claude/skills/policy-finder-summarizer/SKILL.md`**:
  1. `summarize.py status`.
  2. Read each new version's rule and validation.
  3. `summarize.py category ... --kind rule|policy-category` and
     `summarize.py record <version> --rule ... --policy-category ... --idea ...`.
  4. If the Artifact tool is present, export a fragment and publish it to
     `https://claude.ai/artifact/QUAxcas7ui7Dv3ihz87NHr` (`:73-86`).

## 3. Reporting layer (`scripts/policy_finder/observation_app/`)

- **`summarize.py`**:
  - `:62-68` `WT_ROOT/summary.json`, with
    `OBSERVATORY=${PF_OBSERVATORY:-raven:/u/certuer/policy-finder-observatory}`.
    That folder sits in cemrtkn's Raven account.
  - `:89-109` `pull` and `push` rsync `summary.json` and `references.json`
    over ssh. If `ssh -O check` fails, they warn and work locally (`:77-86`).
  - `:147-170` discovers instances and versions from `WT_ROOT/*/.claude/policy_finder.json`.
  - `:223-226` `record` and `category` run under an `fcntl` lock.
- **`dashboard.py`** is the live Streamlit app. Its inputs, all read from
  `WT_ROOT`:
  - instances: dirs with `.claude/policy_finder.json` (`:100-105`);
  - status: `ps` for a running `new_instance.sh` (`:108-115`), the
    `<version>.report.md` size and `git ls-files` (`:118-135`);
  - per version: notes, rule YAML and the scripts folder;
  - `<sweep>/sweep.json`, `landscapes.json` and `validate/sweep.json` (`:597-613`);
  - the activity chart: Write/Edit calls parsed from
    `~/.claude/projects/<wt path dashed>/**/*.jsonl` (`:157`, `:201-250`);
  - `references.json` (`:429`, `:885-893`), `summary.json` (`:896-907`) and
    `overview.html` (`:78`);
  - `zero`'s pool from `plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet` (`:46`, `:450-457`);
  - the evolution tree orders nodes by each rule's first commit,
    `git log --diff-filter=A` (`:1025-1035`, `:1043-1080`). Its root is the
    `references.json` row marked `root` (AH vs AH).
  - Run it with
    `uv run --no-sync --with streamlit streamlit run scripts/policy_finder/observation_app/dashboard.py`.
- **`export_html.py`** imports `dashboard` as `d` and runs the same functions
  (`build()`, `:309-340`). Overview is `counts_html` plus `overview.html`.
  Summary is the evolution tree, the agent table, the rules included and the
  inputs chart. Agent browser shows, per version, the Notes, Rule, Scripts,
  Report and Sweep tabs (landscapes are Vega-Lite specs, best-point JPEGs are
  base64). The JS libraries come from jsdelivr and cdnjs (`:500-505`).
  - Command:
    `uv run --no-sync --with streamlit python scripts/policy_finder/observation_app/export_html.py [--fragment] [--out F]`.
  - The default output is `WT_ROOT/dashboard.html` (`:640`).
  - The 15.6 MB snapshot (stamp "2026-10-15 15:17") embeds:
    - all 32 rule YAMLs, the notes and reports (506 Markdown blocks);
    - 62 JPEGs (2 per validated version);
    - every sweep point (in the Vega specs).
  - It is itself a partial recovery copy (no parquet).
- **CSS:** `dashboard.css`, `overview.css` and `summary.css` hold the shared
  styles. The export adds its own `STYLE`.
- **No generator in the repo** for `references.json` (rows: AH-vs-AH 56.9,
  Levin 69.8, never-punish 59.5, "rerun at the validation's conditions, 2000
  episodes, seed 1000") or for `overview.html`. Both were hand-made.

## 4. Where the results are

| Item | Location (inferred from code and the snapshot's embedded paths) |
|---|---|
| Branches `policy-finder/pf-20261007-{1..4}`, `pf-20261008-{1..8}`, `pf-20261013-1` (13 lineages; `-1` has 7 versions, `-3` 8, `-8` 7, the rest 1) | cemrtkn's main clone `/Users/ertuerkan/Desktop/algorithmic-institutions/.git`. Not on origin, not in this clone. |
| Worktrees, `*.report.md`, `*.summary.md`, `*.sweep_parts/`, `dashboard.html`, `overview.html`, working copies of `summary.json`/`references.json` | `/Users/ertuerkan/Desktop/policy-finder-worktrees/` |
| Kept `summary.json`, `references.json` | `raven:/u/certuer/policy-finder-observatory/` |
| Raw sim outputs and slurm logs | `raven:/u/certuer/ai-isolated/policy-finder--pf-*/{plots/simulation/policy_finder,.log}/` |
| Agent session logs (activity chart) | cemrtkn's `~/.claude/projects/-Users-ertuerkan-Desktop-policy-finder-worktrees-pf-*/` |
| Per-round parquet on the branches | LFS objects in cemrtkn's `.git/lfs` (pushing needs `git lfs push`) |

To reproduce the snapshot:

1. cemrtkn pushes the 13 branches with their LFS objects, and sends the
   `policy-finder-worktrees/` side files and `summary.json`/`references.json`.
2. For each lineage, run `git worktree add $PF_WORKTREE_ROOT/<name> policy-finder/<name>`.
3. Recreate each `<wt>/.claude/policy_finder.json` from the latest version's
   values. It is gitignored, and without it `instances()` lists nothing.
4. Copy the side files into `$PF_WORKTREE_ROOT`.
5. Install `streamlit` and `altair`, then run `export_html.py`. The activity
   charts stay empty without the session logs.

## 5. History (`git log origin/policy-finder-base`, all by cemrtkn)

| Date | Issue / PR | What landed |
|---|---|---|
| 10-07 | #235 | Rule-based managers, sims and reports deleted from base; rule names scrubbed (6396b40e..977e8ce1) |
| 10-07 | #236 / PR #237 | Guard hook, agent definitions, launcher, acceptance check, `.git` write deny, notes, headless mode, skill (449242cf..6d83a056) |
| 10-07 | #227 / PR #240 | Rule params with definition and type plus `sweep_config`, `validate-rule` with Sobol design, `generate_sim_config.py`, `sweep: true`, parts plus `merge_sweeps.py`; 6aabf5c3 one remote dir per branch |
| 10-07/08 | #241 | Sweep run from the launcher (`run_sweep.sh`), committed on the instance branch, observation app (44edbc99), `--num-params`/`--sobol-points`, rules see both groups (1ee0be61) |
| 10-13 | PR #242 | Summary data kept on Raven (9e79dd60), Overview/Summary views, `export_html.py`, auto-summarizer after each sweep |
| 10-13 | #238 / PR #243 | GP fit (`fit_surrogate.py`, `landscapes.json`), validation on fresh seeds, app shows only validated bests |
| 10-14 | #244 (merge 75ede4e6) | Versions: per-version settings, guard, check, sweep; per-round parquet on the branch; `--iterate-on`; lineage comparison |
| 10-15 | (no issue) | Summarizer policy categories, evolution tree, landscape trimming |

## 6. Portability to the command-server proxy

These facts were verified in this container:

- Only `origin/policy-finder-base` exists.
- There is no `.venv`, no `git lfs`, no `rsync`, no `bwrap` and no `socat`.
- `/opt` is not writable. `$HOME` is `/home/claude`. The OS is Linux.
- `claude` 2.1.309 is installed, has credentials and accepts every flag used.
- `python3` 3.10 has torch 2.5.1 CPU, sklearn, and `aimanager` via
  `PYTHONPATH`. It has no streamlit.

| Step | Mechanism today | Works under proxy? | What's needed |
|---|---|---|---|
| Skill pre-check | `ssh -O check raven` (`SKILL.md:33`) | No | Probe `http://cc_command_server:8765` instead |
| Resolve base | `git rev-parse policy-finder-base` (`new_instance.sh:112`) | **No: first failure** | `git branch policy-finder-base origin/policy-finder-base`, or `PF_BASE=origin/policy-finder-base` |
| Interpreter | `$MAIN/.venv/bin/python` (`:109,114`) | No | A `PF_PYTHON` override (`/usr/bin/python3`); written into `policy_finder.json` and used by check, sweep and validate |
| Worktree root | `$(dirname MAIN)/policy-finder-worktrees`, i.e. `/opt/...` (`:107`) | No (read-only) | `PF_WORKTREE_ROOT=/home/claude/pf-worktrees`, but see isolation below |
| Branch plus worktree | `git worktree add -b` (`:159`) | Yes | Branches in the shared clone; the 280 uncommitted changes don't matter |
| Agent inputs | `git lfs pull --include` (`:164`) | No | Install git-lfs in `Dockerfile`. Without it the worktree checks out pointers, and committing parquet stores raw blobs |
| Sandbox | macOS paths `/private/tmp`, `/Volumes` (`instance_settings.py:80,111`); isolation relies on everything private living under `$HOME` | **Unsafe** | Linux rewrite: bubblewrap and socat in the image; deny reads of `/opt/algorithmic-institutions`, the worktree root's siblings and side files, `/var/cc_output`, `/var/cluster_transfer`; session tmp `/tmp/claude-<uid>/...`. Network deny must block `cc_command_server:8765`, or an agent could sync or submit. Verify that the sandbox engages inside Docker before any run, since Bash is pre-allowed |
| Agent session | `claude -p --agent ... --strict-mcp-config` | Likely | Test nested `claude -p` (auth, `CLAUDE_CODE_*` env). Sonnet host plus Opus subagent cost |
| Check and commit | local git plus `validate-rule` | Yes, with a working `python` | Same `PF_PYTHON` fix. 18 `test_policy_finder_check.py` tests fail here only because there is no `.venv`, and `:112` replaces `PYTHONPATH` |
| Sim configs | `generate_sim_config.py` (local) | Yes | Rule referenced by repo-relative path, so it must reach Raven |
| Sync | `rsync --delete` to `~/ai-isolated/<branch>` (`simulate_cluster.sh:81-91`) | No | `POST /sync_to_cluster` syncs only `/opt/algorithmic-institutions` to one fixed dir. Stage `<stem>.yml` and `<stem>_sweep*.yml` into the main checkout (untracked), sync, then remove. Sims then run handover `src`, which differs from base only in `cli.py` and a comment |
| Submit | `ssh ... run.py`, then `sbatch` per part, 2 or 3 parts + 1 validation | No | `POST /submit_slurm_job` with `launcher: python`, `program_call: -m aimanager simulate configs/simulation/policy_finder/<part>.yml`, `time 01:00:00`, `n_cpu 2`, `mem 16G`, `n_gpu 1`. Job id from `/var/cc_output/<run_id>.log` |
| Concurrency | unbounded (8 instances on 10-08, 16 jobs at once) | Cap 4, 409 on the 5th | A submit queue: at most 2 sweeps in flight, retry on 409 |
| Wait | `ssh raven squeue` every 60 s | No | `GET /jobs?cluster=raven` or `waiting_for_jobs` |
| Fetch | `rsync raven:<remote>/plots/...` | No | `POST /run_in_container` (`cp -r /opt/algorithmic-institutions/plots/simulation/policy_finder/<part> /opt/cluster_transfer/pf/`), then `POST /sync_from_cluster` and read `/var/cluster_transfer/pf/`. Check parquet size against the ~1 GB limit |
| Merge, fit, commit | local Python and git | Yes | none |
| Summary store | ssh/rsync to `/u/certuer/...` | Degrades to local | Point `PF_OBSERVATORY` at a store the user owns, or keep it local |
| Headless summarizer | `claude -p /policy-finder-summarizer` | Likely | Same nested-claude test |
| Dashboard / export | `uv run --with streamlit` | No streamlit | Install `streamlit` in the image. The export needs only the import; the live app needs a published port |
| Republish | Artifact tool, interactive | Yes | Publish a new artifact; the old URL is cemrtkn's |

**Adapter sketch.** Add `scripts/policy_finder/cluster.py`, selected by
`PF_CLUSTER=ssh|proxy`, with four functions:

- `sync(paths_to_stage)`: copy into the main checkout, then `/sync_to_cluster`.
- `submit(config) -> jobid`.
- `wait(jobids)`.
- `fetch(remote_rel, local)`: `run_in_container` cp, then `/sync_from_cluster`, then cp.

`run_sweep.sh` and `run_validate.sh` would call it in place of
`simulate_cluster.sh`, `ssh` and `rsync`. The proxy needs no per-branch remote
dir, because names are unique per stem in `configs/` and `plots/`. That still
breaks the isolation of 6aabf5c3, so the staged files must never be committed
from the main checkout. Add `PF_PYTHON`, a Linux branch in
`instance_settings.py`, and image changes (git-lfs, bubblewrap, socat,
streamlit).

## Open questions

- Will cemrtkn push the `policy-finder/*` branches with their LFS objects,
  and the `policy-finder-worktrees/` side files? Is the snapshot HTML the
  only copy otherwise?
- Can `ykeller` read `/u/certuer/ai-isolated/*` and
  `/u/certuer/policy-finder-observatory/`? Who owns the summary store from now on?
- Does `/u/ykeller/algorithmic-institutions/artifacts/` hold the three AH
  `.pt` models and the baseline `.joblib` named in
  `configs/simulation/manager_testing/25_LEVIN_run1_ah_zero_pairings_batched.yml:12-19`?
  `artifacts` is excluded from the sync.
- Does `/sync_to_cluster` delete remote files that are not excluded?
  If it does, staged configs from parallel instances could race.
- Does the Claude Code sandbox run inside this Docker container (user
  namespaces), and what does it do if bubblewrap is missing while
  `allowUnsandboxedCommands: false`?
- How large is a 128-point part's `per_round.parquet`, compared with the
  ~1 GB transfer guideline? Does a part fit the 1 h wall time on the
  proxy's slurm template?
- Which configs produced `references.json` (2000 episodes, seed 1000)? They
  are not in the repo.
- On `yannik/handover`, `reports/` moved to `docs/reports/`. The agent prompt
  (`policy-finder.md:29-31,66`) still says `reports/`, which is correct on
  base but wrong if base adopts the new layout.
