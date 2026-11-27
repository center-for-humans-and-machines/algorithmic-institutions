"""Run one version of a policy-finder instance, from its agent to its final run.

new_instance.sh prepares the version (branch, worktree, instance file) and
then runs `pipeline.py run <name>`. Each step is recorded in
`<worktrees>/<stem>.state.json` as it finishes, so `run` resumes a version
where it stopped (after a crash, a restart of the container or a kill)
without redoing finished steps or resubmitting jobs it already submitted.

Steps (STEPS):
  export        the agent's copy of the instance (export_instance.py)
  agent         the agent in its container (agent_runner.py); its report in
                <stem>.report.md, its run (cost, tokens, models) in the state,
                its transcripts copied to <stem>.transcripts/<run id>/ (the
                spawner's run homes are not kept for good); the instance's
                `harness` (claude, the default, or opencode) picks the agent
  collect       the rule, notes and scripts back into the worktree; a file
                written anywhere else in the write folders fails the version
                (version 1 of a single-idea instance has no agent: these
                three steps write the seed rule of seeds/ and its notes)
  check         check_instance.sh --commit: the acceptance check, and the
                commit of the agent's work on policy-finder/<name>
  sweep_config  the sweep's sim configs (generate_sim_config.py), committed
  sweep_snap    the commit published as a Raven snapshot (cluster_proxy.py)
  image_check   validate-rule in the Raven image (torch 1.11) on the snapshot
  sweep_jobs    one sim job per part, then waiting for them
  sweep_fetch   each part's output; parts without a whole sweep.json are
                rerun together, once (a sim that died with its runner, the
                local backend's Slurm job, is rerun without counting)
  fit           merge the parts, fit the Gaussian process, commit the sweep
  final_config  the final run's config: the GP peak and the 5 best measured
                points (generate_sim_config.py --final), committed
  final_snap    that commit as a snapshot
  final_jobs    the final run's job, then waiting for it
  final_fetch   its output (rerun once without sweep.json), marked with
                --score-final, committed: the version's result

Paths come from PF_RUNS (the root the container spawner's `pf_exports` root
is the `exports` folder of): worktrees and side files in PF_RUNS/worktrees
(PF_WORKTREE_ROOT overrides), exports in PF_RUNS/exports, agent slot locks in
PF_RUNS/locks (PF_LOCKS overrides).

Usage:
    python scripts/policy_finder/pipeline.py run <name> [--prompt TEXT]
    python scripts/policy_finder/pipeline.py status <name>
"""

import argparse
import fcntl
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import agent_runner  # noqa: E402
import cluster_proxy  # noqa: E402
import export_instance  # noqa: E402
import usage  # noqa: E402

MAIN = HERE.parents[1]
SEEDS = HERE / "seeds"
STEPS = (
    "export", "agent", "collect", "check",
    "sweep_config", "sweep_snap", "image_check", "sweep_jobs", "sweep_fetch", "fit",
    "final_config", "final_snap", "final_jobs", "final_fetch",
)
#: Episodes per sweep point, and per final-run point (smoke runs lower them
#: with PF_SWEEP_EPISODES and PF_FINAL_EPISODES); a setting whose points play
#: several opponents counts them per pairing, one per opponent and seat
#: (SETTING_EPISODES: 8 pairings x 64 = 512 per sweep point, x 1000 = 8000
#: per final point; exp2_diff 2 x 250 = 500 and 2 x 1000 = 2000, exp2's)
N_EPISODES = 500
FINAL_EPISODES = 2000
SETTING_EPISODES = {"exp2_multi": (64, 1000), "exp2_diff": (250, 1000)}
#: A setting's opponents (generate_sim_config.py SETTINGS), in the worktree
OPPONENTS = "configs/simulation/opponents/{}.yml"
#: One sim job holds this many episodes (16 GB): 256 points x 500 = 2 parts
MAX_JOB_EPISODES = 125000
OUT_DIR = "plots/simulation/policy_finder"
CONFIG_DIR = "configs/simulation/policy_finder"
#: Agent runs and sim parts tried this many times
AGENT_ATTEMPTS = 2
#: Harness opencode: attempts, and the wait before the next one (seconds,
#: times the attempts so far). A failed run counts, rate limits included:
#: the Claude plan's limit handling (usage.py) is for harness claude only
OPENCODE_ATTEMPTS = 3
OPENCODE_RETRY_WAIT = int(os.environ.get("PF_AGENT_RETRY_WAIT", "120"))
#: validate-rule on the Raven login node: at most this many at once, tries
#: when it does not run (not a verdict), seconds between them (times the try)
IMAGE_CHECK_SLOTS = 2
IMAGE_CHECK_TRIES = 4
IMAGE_CHECK_RETRY = int(os.environ.get("PF_IMAGE_CHECK_RETRY", "120"))
JOB_ATTEMPTS = 2
#: Jobs lost with their runner (the local backend's sims die with their Slurm
#: job): rounds of reruns within one pipeline run, the pause before each, and
#: the most a config may lose over all runs. Within one run a lost job is not
#: a kill (the kill would have taken the pipeline too) but a worker that dies
#: without a status, which would recur: the version fails instead of looping.
LOST_ROUNDS = 2
LOST_PAUSE = int(os.environ.get("PF_LOST_PAUSE", "30"))
LOST_LIMIT = 6


class Failed(Exception):
    pass


def roots():
    pf_runs = os.environ.get("PF_RUNS")
    wt_root = os.environ.get("PF_WORKTREE_ROOT") or (pf_runs and f"{pf_runs}/worktrees")
    if not pf_runs or not wt_root:
        sys.exit("set PF_RUNS (the folder whose exports/ the spawner mounts)")
    return Path(pf_runs), Path(wt_root)


class Version:
    """One version of an instance, and its state file."""

    def __init__(self, name):
        self.pf_runs, self.wt_root = roots()
        self.exports = self.pf_runs / "exports"
        # PF_LOCKS: node-local slot locks when all pipelines share one node
        # (a DAIS job: flock on GPFS does not exclude reliably)
        self.locks = Path(os.environ.get("PF_LOCKS") or self.pf_runs / "locks")
        self.name = name
        self.wt = self.wt_root / name
        conf_path = self.wt / ".claude/policy_finder.json"
        if not conf_path.exists():
            raise Failed(f"no instance {name}: {conf_path} missing")
        self.conf = json.loads(conf_path.read_text())
        self.stem = self.conf["rule_name"]
        # the game (manager/rule.py SETTINGS); absent: exp2
        self.setting = self.conf.get("setting", "exp2")
        self.version = self.conf.get("version", 1)
        # version 1 of a single-idea instance is the given seed rule
        self.seeded = self.conf.get("mode") == "single_idea" and self.version == 1
        # the agent's harness (agent_runner.py): claude, or opencode with `model`
        self.harness = self.conf.get("harness", "claude")
        self.rule = f"configs/managers/rule_based/{self.stem}.yml"
        self.sweep = f"{OUT_DIR}/{self.stem}_sweep"
        self.final = f"{self.sweep}/final"
        self.state_path = self.wt_root / f"{self.stem}.state.json"
        self.log_path = self.wt_root / f"{self.stem}.pipeline.log"
        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text())
        else:
            self.state = {
                "name": name, "stem": self.stem, "version": self.version,
                "status": "running", "done": [], "times": {}, "data": {},
            }

    # -- bookkeeping ---------------------------------------------------------

    def save(self):
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=2) + "\n")
        os.replace(tmp, self.state_path)

    def log(self, message):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {self.stem}: {message}"
        print(line, flush=True)
        with open(self.log_path, "a") as f:
            f.write(line + "\n")

    @property
    def data(self):
        return self.state["data"]

    def sh(self, *args, cwd=None, env=None, check=True):
        result = subprocess.run(
            [str(a) for a in args], cwd=cwd or self.wt, capture_output=True, text=True,
            env={**os.environ, **(env or {})},
        )
        if check and result.returncode != 0:
            raise Failed(
                f"{' '.join(map(str, args[:3]))} failed:\n{result.stdout[-1500:]}"
                f"\n{result.stderr[-3000:]}"
            )
        return result

    def py(self, script, *args, check=True):
        env = {"PYTHONPATH": f"{self.wt}/src"}
        return self.sh(sys.executable, MAIN / "scripts/policy_finder" / script, *args,
                       env=env, check=check)

    def git(self, *args, check=True):
        return self.sh("git", "-C", self.wt, *args, check=check)

    def commit(self, paths, message):
        self.git("add", "--", *paths)
        if self.git("diff", "--cached", "--quiet", check=False).returncode == 0:
            return self.head()
        self.git("commit", "-q", "-m", f"policy-finder {self.stem}: {message}")
        return self.head()

    def head(self):
        return self.git("rev-parse", "HEAD").stdout.strip()

    # -- the run -------------------------------------------------------------

    def run(self, prompt=None):
        # one pipeline per version: a second one waits, then sees its state
        # a POSIX lock: GPFS honours it across nodes (flock it may not). It
        # is held by this process until it exits; nothing else here opens
        # the file (closing any descriptor of it would release the lock)
        self.lock_path = self.wt_root / f"{self.stem}.lock"
        self._lock = open(self.lock_path, "w")
        try:
            fcntl.lockf(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.log("another pipeline runs this version: waiting for it")
            fcntl.lockf(self._lock, fcntl.LOCK_EX)
            self.state = json.loads(self.state_path.read_text())
        if self.state["status"] in ("scored", "failed"):
            self.log(f"already {self.state['status']}")
            return self.state
        return self._run(prompt)

    def _run(self, prompt=None):
        if self.state["status"] == "scored":
            self.log("already scored")
            return self.state
        self.state["status"] = "running"
        self.state.pop("error", None)
        self.data.setdefault("prompt", prompt or (
            "Design your rule." if self.version == 1
            else f"Design version {self.version} of your rule."
        ))
        self.data.setdefault("episodes", self.default_episodes())
        self.save()
        try:
            for step in STEPS:
                if step in self.state["done"]:
                    continue
                self.log(f"step {step}")
                started = time.time()
                getattr(self, f"step_{step}")()
                self.state["done"].append(step)
                self.state["times"][step] = round(time.time() - started)
                self.save()
            self.state["status"] = "scored"
            self.log(f"scored: {json.dumps(self.data.get('score'))}")
            self.release_snapshots()
        except Failed as e:
            self.state["status"] = "failed"
            self.state["error"] = str(e)[-6000:]
            self.log(f"FAILED: {e}")
        except agent_runner.BudgetError as e:
            # the OpenRouter cap, not the version: the campaign stops here
            self.state["status"] = "stopped"
            self.state["error"] = str(e)[-6000:]
            self.log(f"STOPPED: {e}")
        except Exception:
            # an unexpected error (the network, say): resumable, not a verdict
            self.state["status"] = "error"
            self.state["error"] = traceback.format_exc()[-6000:]
            self.log(f"ERROR: {traceback.format_exc()}")
        self.save()
        return self.state

    def release_snapshots(self):
        """A scored version's snapshots: its results are fetched and
        committed, so the backend may remove them (cluster_proxy.cleanup; a
        failed version's stay, for debugging). Never fails the version."""
        for key in ("sweep_sha", "final_sha"):
            sha = self.data.get(key)
            try:
                if sha and cluster_proxy.cleanup(sha):
                    self.log(f"snapshot {sha[:12]} removed")
            except Exception as e:  # noqa: BLE001 - only disk space is at stake
                self.log(f"snapshot {sha[:12]} not removed: {e}")

    def step_export(self):
        if self.seeded:
            notes = self.wt / f"notes/policy_finder/{self.stem}.md"
            for path in (self.wt / self.rule, notes):
                path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(SEEDS / "single_idea.yml", self.wt / self.rule)
            notes.write_text((SEEDS / "single_idea.md").read_text().replace(
                "{stem}", self.stem))
            self.log("seed rule written, no agent")
            return
        try:
            export_instance.build(self.wt, MAIN, self.exports)
        except export_instance.ExportError as e:
            raise Failed(str(e)) from None

    def step_agent(self):
        if self.seeded:
            return
        write_dirs = export_instance.write_dirs(self.name, self.stem)
        attempts = self.data.setdefault("agent_runs", [])
        limit = AGENT_ATTEMPTS if self.harness == "claude" else OPENCODE_ATTEMPTS
        while len(attempts) < limit:
            if self.harness != "claude" and attempts and not self.data.get("agent_current"):
                # bounded back-off after a failed attempt (a rate limit, say)
                time.sleep(OPENCODE_RETRY_WAIT * len(attempts))
            with agent_runner.slot(self.locks):
                run_id = self.data.get("agent_current")
                if run_id:
                    self.log(f"attaching to agent run {run_id}")
                else:
                    if (attempts or self.data.get("limited_runs")
                            or self.data.get("lost_runs")):
                        # a fresh copy for the retry: the failed run may have written
                        export_instance.build(self.wt, MAIN, self.exports)
                    if self.harness == "claude":
                        # the plan's usage: wait while a window is over its ceiling
                        self.log(f"usage: {usage.describe(usage.gate(log=self.log))}")
                    elif not agent_runner.MAX_USAGE and not self.conf.get("base_url"):
                        self.log("WARNING: PF_OPENROUTER_MAX_USAGE is not set: no spend"
                                 " cap before this agent starts")
                    run_id = agent_runner.start(self.stem, write_dirs, self.data["prompt"],
                                                harness=self.harness)
                    self.data["agent_current"] = run_id
                    self.save()
                    self.log(f"agent run {run_id} started")
                try:
                    record = agent_runner.finish(run_id, log=self.log, harness=self.harness)
                except agent_runner.BudgetError:
                    # not started: a resume starts a new run once there is budget
                    self.data.pop("agent_current", None)
                    self.save()
                    raise
            report = record.pop("report")
            record["transcripts"] = self.keep_transcripts(run_id)
            if record["status"] == "lost":
                # the run ended with the job that ran it (agent_apptainer.py):
                # not one of the agent's attempts; a fresh run takes its place
                lost = self.data.setdefault("lost_runs", [])
                lost.append(record)
                self.data.pop("agent_current", None)
                self.save()
                self.log(f"agent run {run_id} lost (its job ended); starting a new one")
                if len(lost) > limit:
                    raise Failed(f"{len(lost)} agent runs lost: {record.get('error')}")
                continue
            rule_written = (self.exports / self.stem / self.rule).exists()
            ok = record["status"] == "succeeded" and report.strip() and rule_written
            home = agent_runner.run_home(run_id, self.harness)
            hits = ([] if ok or self.harness != "claude"
                    else usage.limited({**record, "report": report}, home))
            if hits:
                # the account's limit, not the agent: not one of its attempts
                self.data.setdefault("limited_runs", []).append(record)
                self.data.pop("agent_current", None)
                self.save()
                usage.wait_after_limit(hits, log=self.log)
                continue
            attempts.append(record)
            self.data.pop("agent_current", None)
            self.save()
            if self.harness == "opencode":
                cost = (f"tokens {record['result'].get('tokens')}, OpenCode's catalogue"
                        f" cost $ {record['result'].get('cost_catalog_usd')}, OpenRouter"
                        f" key usage +$ {record['result'].get('key_usage_delta_usd')}"
                        " (every run on the key in that time)")
                for warning in record["result"].get("warnings") or []:
                    self.log(f"WARNING: {warning}")
                if record.get("error"):
                    self.log(f"agent run {run_id} failed: {record['error']}")
            else:
                cost = (f"API-equivalent cost $ {record['result'].get('total_cost_usd')}"
                        " (plan, not billed)")
            self.log(
                f"agent run {run_id}: {record['status']}, models {record['models']},"
                f" {cost}, rule {'written' if rule_written else 'missing'}"
            )
            if "network_audit" in record:
                found = record["network_audit"]["findings"]
                self.log(f"network audit of {run_id}: {len(found)} finding(s)"
                         + "".join(f"\n  {f['kind']}: {f.get('matches') or f.get('text')}"
                                   for f in found[:10]))
            if ok:
                (self.wt_root / f"{self.stem}.report.md").write_text(report + "\n")
                return
        raise Failed(f"the agent failed {limit} times: {attempts[-1]}")

    def keep_transcripts(self, run_id):
        """Copy an agent run's session transcripts next to the state file.

        Every attempt is kept, also a failed one: they are the record of what
        the agent ran. Returns the number of files copied (0, with a log line,
        when the run home is not mounted here).
        """
        source, paths = agent_runner.transcripts(run_id, self.harness)
        target = self.wt_root / f"{self.stem}.transcripts" / run_id
        count, copied = 0, []
        try:
            for path in paths:
                dest = target / os.path.relpath(path, source)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, dest)
                copied.append(dest)
                count += 1
        except OSError as e:
            self.log(f"transcripts of {run_id} not copied: {e}")
        # the key, should the runner not have redacted it (killed, say)
        _, hits = agent_runner.scrub(run_id, self.harness, copied)
        if hits:
            self.log(f"the key was in {len(hits)} transcript file(s) of {run_id}: redacted")
        if not count:
            self.log(f"no transcripts of {run_id} under {source}")
        return count

    def step_collect(self):
        if self.seeded:
            return
        strays = export_instance.collect(self.wt, self.exports)
        if strays:
            raise Failed("the agent wrote outside its paths: " + ", ".join(strays))
        export_instance.remove(self.exports, self.stem)

    def step_check(self):
        result = self.sh(
            "bash", MAIN / "scripts/policy_finder/check_instance.sh", self.name, "--commit",
            cwd=MAIN, check=False,
            env={"PF_RUNS": str(self.pf_runs), "PF_WORKTREE_ROOT": str(self.wt_root)},
        )
        self.data["check"] = (result.stdout + result.stderr)[-3000:]
        if result.returncode != 0:
            raise Failed(f"check_instance.sh failed:\n{result.stdout}{result.stderr}")

    # -- sweep ---------------------------------------------------------------

    def default_episodes(self):
        sweep, final = SETTING_EPISODES.get(self.setting, (N_EPISODES, FINAL_EPISODES))
        return {
            "sweep": int(os.environ.get("PF_SWEEP_EPISODES", sweep)),
            "final": int(os.environ.get("PF_FINAL_EPISODES", final)),
        }

    def pairings(self):
        """Pairings per design point: one per opponent of the setting and seat,
        else one (generate_sim_config.pairings_per_point)."""
        path = self.wt / OPPONENTS.format(self.setting)
        if not path.exists():
            return 1
        return 2 * len(yaml.safe_load(path.read_text()))

    def design_points(self):
        """The most points the sweep's design holds: the instance's
        `sobol_points`, or one for a rule without a ranged param (the seed of
        a single-idea instance), whose design is a single point."""
        rule = self.wt / self.rule
        if rule.exists():
            sweep = (yaml.safe_load(rule.read_text()) or {}).get("sweep_config") or {}
            if not any(isinstance(spec, list) for spec in sweep.values()):
                return 1
        return self.conf.get("sobol_points", 256)

    def parts(self):
        # the backend's split (cluster_proxy.sweep_parts) of the real design
        # (step_sweep_config counts it), kept in the state so that a resumed
        # version keeps its parts
        if "n_parts" not in self.data:
            per_point = self.pairings() * self.data["episodes"]["sweep"]
            points = self.data.get("design_points") or self.design_points()
            self.data["n_parts"], self.data["parts_aligned"] = cluster_proxy.sweep_parts(
                points, per_point, self.setting, MAX_JOB_EPISODES)
        n = self.data["n_parts"]
        if n == 1:
            return [f"{self.stem}_sweep"]
        return [f"{self.stem}_sweep_p{k}of{n}" for k in range(1, n + 1)]

    def _generate_sweep(self, *extra):
        upper = self.conf["max_params"]
        self.py(
            "generate_sim_config.py", "--root", self.wt, "--config", self.wt / self.rule,
            "--sobol-points", self.conf.get("sobol_points", 256),
            "--n-episodes", self.data["episodes"]["sweep"],
            "--min-params", self.conf["min_params"],
            "--max-params", "none" if upper is None else upper,
            "--setting", self.setting, *extra,
        )

    def count_design(self):
        """The sweep's distinct points (the Sobol design merges equal ones, so
        a rule with few int values has fewer than sobol_points), from the
        sweep config generate_sim_config.py writes for it whole."""
        whole = self.wt_root / f"{self.stem}.design.yml"
        try:
            self._generate_sweep("--n-parts", 1, "--out", whole)
            managers = yaml.safe_load(whole.read_text())["managers"]
        finally:
            whole.unlink(missing_ok=True)
        pattern = re.compile(rf"{re.escape(self.stem)}_s\d+$")
        return sum(1 for name in managers if pattern.match(name))

    def step_sweep_config(self):
        if "n_parts" not in self.data:
            self.data["design_points"] = self.count_design()
        self._generate_sweep("--n-parts", len(self.parts()))
        configs = [f"{CONFIG_DIR}/{p}.yml" for p in self.parts()]
        self.data["sweep_commit"] = self.commit(configs, "sweep configs")

    def step_sweep_snap(self):
        self.data["sweep_sha"] = cluster_proxy.snapshot(self.data["sweep_commit"])
        self.log(f"snapshot {self.data['sweep_sha'][:12]}")

    def step_image_check(self):
        """validate-rule in the Raven image. Only its own "Invalid:" is a
        verdict on the rule; anything else (the login node out of threads or
        processes, a container that did not start) is retried, then left as a
        resumable error. At most IMAGE_CHECK_SLOTS run on the login node."""
        upper = self.conf["max_params"]
        for attempt in range(1, IMAGE_CHECK_TRIES + 1):
            with agent_runner.slot(self.locks / "image_check", slots=IMAGE_CHECK_SLOTS):
                ok, log = cluster_proxy.validate_rule(
                    self.data["sweep_sha"], self.rule, self.conf["min_params"], upper,
                    self.conf.get("sobol_points", 256), self.setting,
                )
            if ok:
                return
            if re.search(r"^Invalid: ", log, flags=re.M):
                raise Failed(f"{self.rule} fails validate-rule in the Raven image:\n{log[-3000:]}")
            tail = log.strip().splitlines()[-1:] or ["(no output)"]
            self.log(f"image check {attempt}/{IMAGE_CHECK_TRIES} did not run: {tail[0][:200]}")
            if attempt < IMAGE_CHECK_TRIES:
                time.sleep(IMAGE_CHECK_RETRY * attempt)
        raise cluster_proxy.ClusterError(
            f"validate-rule did not run in the Raven image ({IMAGE_CHECK_TRIES} tries):"
            f"\n{log[-2000:]}")

    def _submit(self, key, sha, config, why="submitted"):
        jobs = self.data[key]
        jobs.setdefault(config, []).append(
            cluster_proxy.submit(config, sha, run_group=self.stem))
        self.save()
        self.log(f"{why} {config}: job {jobs[config][-1]}")

    def _drop_lost(self, key, config):
        """A job that died with its runner (the local backend's sims die with
        their Slurm job) is moved aside: it is rerun, not counted as one of
        the config's JOB_ATTEMPTS, up to LOST_LIMIT times."""
        jobs = self.data[key]
        if jobs.get(config) and cluster_proxy.lost(jobs[config][-1]):
            job = jobs[config].pop()
            lost = self.data.setdefault(f"{key}_lost", {}).setdefault(config, [])
            lost.append(job)
            self.save()
            if len(lost) > LOST_LIMIT:
                raise Failed(f"{config}: {len(lost)} jobs died without a status {lost}:"
                             " see their worker.log (the pool's PF_POOL_LOCKS,"
                             " PF_POOL_CPUS)")
            self.log(f"job {job} of {config} died with its runner: rerun, not an attempt")
            return True
        return False

    def _lost_round(self, configs):
        """Before rerunning jobs lost within this run: fail after LOST_ROUNDS
        such rounds, else pause."""
        self._lost_rounds = getattr(self, "_lost_rounds", 0) + 1
        if self._lost_rounds > LOST_ROUNDS:
            raise Failed(
                f"jobs keep dying without a status while the pipeline runs"
                f" ({self._lost_rounds - 1} rounds; last {configs[:3]}...): the"
                " workers fail before they can record it; see their worker.log"
                " (the pool's PF_POOL_LOCKS, PF_POOL_CPUS)")
        self.log(f"{len(configs)} jobs died while the pipeline ran: rerun in"
                 f" {LOST_PAUSE} s (round {self._lost_rounds} of {LOST_ROUNDS})")
        time.sleep(LOST_PAUSE)

    def _jobs(self, key, sha, configs):
        """Submit each config not yet submitted (ids kept in data[key]), wait;
        jobs lost with their runner are resubmitted together, until none is."""
        jobs = self.data.setdefault(key, {})
        while True:
            for config in configs:
                self._drop_lost(key, config)  # lost before this run (a resume)
                if not jobs.get(config):
                    self._submit(key, sha, config)
            pending = [jobs[c][-1] for c in configs]
            self.log(f"waiting for jobs {pending}")
            cluster_proxy.wait(pending)
            lost = [c for c, job in zip(configs, pending) if cluster_proxy.lost(job)]
            if not lost:
                return
            self._lost_round(lost)

    def _complete(self, local, config):
        """Has the fetched output a whole sweep.json: parseable, with every
        point of its config? (A sim killed while writing leaves a truncated
        one: treated as missing.)"""
        try:
            sweep = json.loads((Path(local) / "sweep.json").read_text())
            names = {p["name"] for p in sweep["points"]}
        except (OSError, ValueError, KeyError, TypeError):
            return False
        path = self.wt / config
        if not path.exists():
            return bool(names)
        pattern = re.compile(rf"{re.escape(self.stem)}_(s\d+|gp)$")
        wanted = {m for m in yaml.safe_load(path.read_text())["managers"] if pattern.match(m)}
        return bool(names) and wanted <= names

    def _fetch_all(self, key, sha, items):
        """Fetch each sim's output, (config, remote, local) in `items`; rerun
        every one without a whole sweep.json, all together, once each (a job
        lost with its runner is rerun without counting, within the limits of
        _drop_lost and _lost_round).

        A fetch that could not run (Raven unreachable) raises ClusterError,
        resumable, and costs no job attempt.
        """
        jobs = self.data[key]
        todo, first = list(items), True
        while True:
            missing = []
            for config, remote, local in todo:
                try:
                    size = cluster_proxy.fetch(sha, remote, local,
                                               key=f"{self.stem}/{Path(remote).name}")
                    self.log(f"fetched {remote}: {size / 1e6:.1f} MB")
                except cluster_proxy.NoOutput as e:
                    self.log(f"{e}")
                if not self._complete(local, config):
                    missing.append((config, remote, local))
            if not missing:
                return
            lost = [c for c, _, _ in missing if jobs[c] and cluster_proxy.lost(jobs[c][-1])]
            if lost and not first:  # lost while this run waited for them
                self._lost_round(lost)
            first = False
            for config, _, local in missing:
                if not self._drop_lost(key, config) and len(jobs[config]) >= JOB_ATTEMPTS:
                    raise Failed(f"{config} wrote no sweep.json in {JOB_ATTEMPTS} jobs"
                                 f" {jobs[config]}")
                shutil.rmtree(local, ignore_errors=True)  # a partial or truncated copy
                self._submit(key, sha, config, why="resubmitted")
            cluster_proxy.wait([jobs[c][-1] for c, _, _ in missing])
            todo = missing

    def part_dir(self, part):
        if len(self.parts()) == 1:
            return self.wt / self.sweep
        return self.wt_root / f"{self.stem}.sweep_parts" / part

    def step_sweep_jobs(self):
        configs = [f"{CONFIG_DIR}/{p}.yml" for p in self.parts()]
        self._jobs("sweep_jobs", self.data["sweep_sha"], configs)

    def step_sweep_fetch(self):
        self._fetch_all("sweep_jobs", self.data["sweep_sha"], [
            (f"{CONFIG_DIR}/{part}.yml", f"{OUT_DIR}/{part}", self.part_dir(part))
            for part in self.parts()])

    def step_fit(self):
        parts = self.parts()
        if len(parts) > 1:
            self.py("merge_sweeps.py", *[self.part_dir(p) for p in parts],
                    "--out", self.wt / self.sweep / "sweep.json")
            # each part's per-round data beside the sweep (not in git: *.parquet)
            for part in parts:
                source = self.part_dir(part) / "per_round.parquet"
                if source.exists():
                    shutil.move(source, self.wt / self.sweep /
                                f"per_round_{part.rsplit('_sweep_', 1)[1]}.parquet")
            shutil.rmtree(self.wt_root / f"{self.stem}.sweep_parts", ignore_errors=True)
        self.py("fit_surrogate.py", self.wt / self.sweep)
        self.data["fit_commit"] = self.commit([self.sweep], "sweep")

    # -- final run -----------------------------------------------------------

    def step_final_config(self):
        self.py("generate_sim_config.py", "--root", self.wt,
                "--final", self.wt / self.sweep / "sweep.json",
                "--n-episodes", self.data["episodes"]["final"], "--setting", self.setting)
        config = f"{CONFIG_DIR}/{self.stem}_sweep_final.yml"
        self.data["final_commit"] = self.commit([config], "final run config")

    def step_final_snap(self):
        self.data["final_sha"] = cluster_proxy.snapshot(self.data["final_commit"])
        self.log(f"snapshot {self.data['final_sha'][:12]}")

    def step_final_jobs(self):
        self._jobs("final_jobs", self.data["final_sha"],
                   [f"{CONFIG_DIR}/{self.stem}_sweep_final.yml"])

    def step_final_fetch(self):
        self._fetch_all("final_jobs", self.data["final_sha"], [
            (f"{CONFIG_DIR}/{self.stem}_sweep_final.yml", self.final, self.wt / self.final)])
        self.py("generate_sim_config.py", "--score-final", self.wt / self.final / "sweep.json")
        final = json.loads((self.wt / self.final / "sweep.json").read_text())
        points = {p["name"]: p for p in final["points"]}
        gp, top = points[final["best_name"]], points[final["best_of_n_name"]]
        self.data["score"] = {
            "pool": gp["pool"], "pool_se": gp["pool_se"], "members": gp["members"],
            "params": gp["params"],
            "best_of_n": top["pool"], "best_of_n_se": top["pool_se"],
            "best_of_n_name": top["name"], "best_of_n_params": top["params"],
            "n_points": final["n_points"],
        }
        for key in ("own", "other", "gain", "gain_se"):  # pool_diff_vs_opponents; a baseline
            if key in gp:
                self.data["score"][key] = gp[key]
        if "pools" in gp:  # against each opponent (pool_vs_opponents)
            self.data["score"]["pools"] = {
                k: {"pool": v["pool"], "pool_se": v["pool_se"]} for k, v in gp["pools"].items()
            }
        self.commit([self.final], "final run")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("name")
    r.add_argument("--prompt")
    s = sub.add_parser("status")
    s.add_argument("name")
    args = ap.parse_args()
    try:
        version = Version(args.name)
    except Failed as e:
        sys.exit(f"FAIL: {e}")
    if args.cmd == "status":
        print(json.dumps(version.state, indent=2))
        return
    state = version.run(args.prompt)
    sys.exit(0 if state["status"] == "scored" else 1)


if __name__ == "__main__":
    main()
