"""Run one version of a policy-finder instance, from its agent to its final run.

new_instance.sh prepares the version (branch, worktree, instance file) and
then runs `pipeline.py run <name>`. Each step is recorded in
`<worktrees>/<stem>.state.json` as it finishes, so `run` resumes a version
where it stopped (after a crash, a restart of the container or a kill)
without redoing finished steps or resubmitting jobs it already submitted.

Steps (STEPS):
  export        the agent's copy of the instance (export_instance.py)
  agent         the agent in its container (agent_runner.py); its report in
                <stem>.report.md, its run (cost, tokens, models) in the state
  collect       the rule, notes and scripts back into the worktree; a file
                written anywhere else in the write folders fails the version
  check         check_instance.sh --commit: the acceptance check, and the
                commit of the agent's work on policy-finder/<name>
  sweep_config  the sweep's sim configs (generate_sim_config.py), committed
  sweep_snap    the commit published as a Raven snapshot (cluster_proxy.py)
  image_check   validate-rule in the Raven image (torch 1.11) on the snapshot
  sweep_jobs    one sim job per part, then waiting for them
  sweep_fetch   each part's output; a part without sweep.json is rerun once
  fit           merge the parts, fit the Gaussian process, commit the sweep
  final_config  the final run's config: the GP peak and the 5 best measured
                points (generate_sim_config.py --final), committed
  final_snap    that commit as a snapshot
  final_jobs    the final run's job, then waiting for it
  final_fetch   its output (rerun once without sweep.json), marked with
                --score-final, committed: the version's result

Only the cluster path through the command server (the Claude proxy) is
implemented here; the SSH path stays run_sweep.sh and run_final.sh.

Paths come from PF_RUNS (the root the container spawner's `pf_exports` root
is the `exports` folder of): worktrees and side files in PF_RUNS/worktrees
(PF_WORKTREE_ROOT overrides), exports in PF_RUNS/exports, agent slot locks in
PF_RUNS/locks.

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
STEPS = (
    "export", "agent", "collect", "check",
    "sweep_config", "sweep_snap", "image_check", "sweep_jobs", "sweep_fetch", "fit",
    "final_config", "final_snap", "final_jobs", "final_fetch",
)
#: Episodes per sweep point, and per final-run point (smoke runs lower them
#: with PF_SWEEP_EPISODES and PF_FINAL_EPISODES); a setting whose points play
#: several opponents counts them per pairing, one per opponent and seat
#: (SETTING_EPISODES: 8 pairings x 64 = 512 per sweep point, x 1000 = 8000
#: per final point)
N_EPISODES = 500
FINAL_EPISODES = 2000
SETTING_EPISODES = {"exp2_multi": (64, 1000)}
#: A setting's opponents (generate_sim_config.py SETTINGS), in the worktree
OPPONENTS = "configs/simulation/opponents/{}.yml"
#: One sim job holds this many episodes (16 GB): 256 points x 500 = 2 parts
MAX_JOB_EPISODES = 125000
OUT_DIR = "plots/simulation/policy_finder"
CONFIG_DIR = "configs/simulation/policy_finder"
#: Agent runs and sim parts tried this many times
AGENT_ATTEMPTS = 2
#: validate-rule on the Raven login node: at most this many at once, tries
#: when it does not run (not a verdict), seconds between them (times the try)
IMAGE_CHECK_SLOTS = 2
IMAGE_CHECK_TRIES = 4
IMAGE_CHECK_RETRY = int(os.environ.get("PF_IMAGE_CHECK_RETRY", "120"))
JOB_ATTEMPTS = 2


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
        self.locks = self.pf_runs / "locks"
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
        self.lock_path = self.wt_root / f"{self.stem}.lock"
        self._lock = open(self.lock_path, "w")
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.log("another pipeline runs this version: waiting for it")
            fcntl.flock(self._lock, fcntl.LOCK_EX)
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
        except Failed as e:
            self.state["status"] = "failed"
            self.state["error"] = str(e)[-6000:]
            self.log(f"FAILED: {e}")
        except Exception:
            # an unexpected error (the network, say): resumable, not a verdict
            self.state["status"] = "error"
            self.state["error"] = traceback.format_exc()[-6000:]
            self.log(f"ERROR: {traceback.format_exc()}")
        self.save()
        return self.state

    def step_export(self):
        try:
            export_instance.build(self.wt, MAIN, self.exports)
        except export_instance.ExportError as e:
            raise Failed(str(e)) from None

    def step_agent(self):
        write_dirs = export_instance.write_dirs(self.name, self.stem)
        attempts = self.data.setdefault("agent_runs", [])
        while len(attempts) < AGENT_ATTEMPTS:
            with agent_runner.slot(self.locks):
                run_id = self.data.get("agent_current")
                if run_id:
                    self.log(f"attaching to agent run {run_id}")
                else:
                    if attempts or self.data.get("limited_runs"):
                        # a fresh copy for the retry: the failed run may have written
                        export_instance.build(self.wt, MAIN, self.exports)
                    # the plan's usage: wait while a window is over its ceiling
                    self.log(f"usage: {usage.describe(usage.gate(log=self.log))}")
                    run_id = agent_runner.start(self.stem, write_dirs, self.data["prompt"])
                    self.data["agent_current"] = run_id
                    self.save()
                    self.log(f"agent run {run_id} started")
                record = agent_runner.finish(run_id, log=self.log)
            report = record.pop("report")
            rule_written = (self.exports / self.stem / self.rule).exists()
            ok = record["status"] == "succeeded" and report.strip() and rule_written
            home = os.path.join(agent_runner.SPAWNER_STATE, "runs", run_id, "home")
            hits = [] if ok else usage.limited({**record, "report": report}, home)
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
            self.log(
                f"agent run {run_id}: {record['status']}, models {record['models']},"
                f" API-equivalent cost $ {record['result'].get('total_cost_usd')} (plan, not billed),"
                f" rule {'written' if rule_written else 'missing'}"
            )
            if ok:
                (self.wt_root / f"{self.stem}.report.md").write_text(report + "\n")
                return
        raise Failed(f"the agent failed {AGENT_ATTEMPTS} times: {attempts[-1]}")

    def step_collect(self):
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

    def parts(self):
        sobol = self.conf.get("sobol_points", 256)
        per_point = self.pairings() * self.data["episodes"]["sweep"]
        n = -(-sobol * per_point // MAX_JOB_EPISODES)
        if n == 1:
            return [f"{self.stem}_sweep"]
        return [f"{self.stem}_sweep_p{k}of{n}" for k in range(1, n + 1)]

    def step_sweep_config(self):
        upper = self.conf["max_params"]
        self.py(
            "generate_sim_config.py", "--root", self.wt, "--config", self.wt / self.rule,
            "--sobol-points", self.conf.get("sobol_points", 256),
            "--n-episodes", self.data["episodes"]["sweep"], "--n-parts", len(self.parts()),
            "--min-params", self.conf["min_params"],
            "--max-params", "none" if upper is None else upper,
            "--setting", self.setting,
        )
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

    def _jobs(self, key, sha, configs):
        """Submit each config not yet submitted (ids kept in data[key]), wait."""
        jobs = self.data.setdefault(key, {})
        for config in configs:
            if config not in jobs:
                jobs[config] = [cluster_proxy.submit(config, sha, run_group=self.stem)]
                self.save()
                self.log(f"submitted {config}: job {jobs[config][-1]}")
        pending = [j[-1] for j in jobs.values()]
        self.log(f"waiting for jobs {pending}")
        cluster_proxy.wait(pending)

    def _fetch(self, key, sha, config, remote, local):
        """Fetch one sim's output; rerun its job once if it has no sweep.json.

        Only a job that left no output (or no sweep.json) is rerun; a fetch
        that could not run (Raven unreachable) raises ClusterError, resumable,
        and costs no job attempt.
        """
        jobs = self.data[key]
        while True:
            try:
                size = cluster_proxy.fetch(sha, remote, local, key=f"{self.stem}/{Path(remote).name}")
                self.log(f"fetched {remote}: {size / 1e6:.1f} MB")
            except cluster_proxy.NoOutput as e:
                self.log(f"{e}")
            if (Path(local) / "sweep.json").exists():
                return
            if len(jobs[config]) >= JOB_ATTEMPTS:
                raise Failed(f"{config} wrote no sweep.json in {JOB_ATTEMPTS} jobs {jobs[config]}")
            jobs[config].append(cluster_proxy.submit(config, sha, run_group=self.stem))
            self.save()
            self.log(f"resubmitted {config}: job {jobs[config][-1]}")
            cluster_proxy.wait([jobs[config][-1]])

    def part_dir(self, part):
        if len(self.parts()) == 1:
            return self.wt / self.sweep
        return self.wt_root / f"{self.stem}.sweep_parts" / part

    def step_sweep_jobs(self):
        configs = [f"{CONFIG_DIR}/{p}.yml" for p in self.parts()]
        self._jobs("sweep_jobs", self.data["sweep_sha"], configs)

    def step_sweep_fetch(self):
        for part in self.parts():
            self._fetch("sweep_jobs", self.data["sweep_sha"], f"{CONFIG_DIR}/{part}.yml",
                        f"{OUT_DIR}/{part}", self.part_dir(part))

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
        self._fetch("final_jobs", self.data["final_sha"],
                    f"{CONFIG_DIR}/{self.stem}_sweep_final.yml", self.final,
                    self.wt / self.final)
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
