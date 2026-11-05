"""Run a policy-finder campaign: fresh lineages, then generations iterating the best.

A campaign is a YAML file (scripts/policy_finder/campaigns/*.yml):

    prefix: pf-20261104     # lineages <prefix>-1 ... <prefix>-<n_fresh>
    n_fresh: 12             # generation 1: fresh lineages
    generations: 6          # then this many generations of --iterate-on
    top_k: 4                # ... on the lineages with the highest lineage best
    min_params: 1           # generation 1's param bounds; later versions
    max_params: 5           #   may declare any number (new_instance.sh)
    sobol_points: 256
    parallel: 12            # versions running at once (agents are capped
                            #   separately by agent_runner's slots)
    error_retries: 3        # resumes of a version whose pipeline hit an error
    retry_wait: 600         # seconds before such a resume
    summarize: null         # a shell command run after each generation
                            #   (non-fatal; its output goes to the log)

The lineage best is the highest GP-peak final (`score.pool`) of any of the
lineage's scored versions. Each later generation iterates the top_k lineages
by lineage best, chosen when the generation starts. A version whose pipeline
failed (its check, its sweep, ...) counts as no improvement; its slot is not
given to another lineage. The driver reads every score, but an agent gets
nothing from it: new_instance.sh and pipeline.py give the agent its own
lineage only.

State is `$PF_RUNS/campaign/<campaign>.json`: the members of each generation,
written before the generation runs. `run` resumes a killed campaign: scored and
failed versions stay as they are, a version that stopped resumes where it
stopped. Before resuming a version after an error the campaign waits until
Raven answers; an error while Raven was down is not counted. A version that
still has an error after error_retries stops the campaign (that is the
environment, not the agent): fix it and `run` again.

Usage:
    python scripts/policy_finder/campaign.py run <campaign.yml>
    python scripts/policy_finder/campaign.py summary <campaign.yml>
"""

import argparse
import json
import math
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cluster_proxy  # noqa: E402

HERE = Path(__file__).resolve().parent
DEFAULTS = {
    "n_fresh": 12, "generations": 6, "top_k": 4, "min_params": 1, "max_params": 5,
    "sobol_points": 256, "parallel": 12, "error_retries": 3, "retry_wait": 600,
    "setting": "exp2",
    "summarize": None,
}


class CampaignError(Exception):
    pass


def stem_of(name, version):
    return name if version == 1 else f"{name}-v{version}"


class Campaign:
    def __init__(self, path, pf_runs=None):
        self.path = Path(path)
        self.settings = {**DEFAULTS, **yaml.safe_load(self.path.read_text())}
        if not self.settings.get("prefix"):
            raise CampaignError(f"{path}: no prefix")
        self.pf_runs = Path(pf_runs or os.environ.get("PF_RUNS") or "")
        if not str(self.pf_runs) or str(self.pf_runs) == ".":
            raise CampaignError("set PF_RUNS")
        self.wt_root = Path(os.environ.get("PF_WORKTREE_ROOT") or self.pf_runs / "worktrees")
        self.dir = self.pf_runs / "campaign"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.dir / f"{self.path.stem}.json"
        self.log_path = self.dir / f"{self.path.stem}.log"
        self.state = (json.loads(self.state_path.read_text()) if self.state_path.exists()
                      else {"generations": []})
        self._lock = threading.Lock()

    # -- bookkeeping ---------------------------------------------------------

    def save(self):
        with self._lock:
            tmp = self.state_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.state, indent=2) + "\n")
            os.replace(tmp, self.state_path)

    def log(self, message):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        with self._lock:
            print(line, flush=True)
            with open(self.log_path, "a") as f:
                f.write(line + "\n")

    def lineages(self):
        prefix = self.settings["prefix"]
        return [f"{prefix}-{i}" for i in range(1, self.settings["n_fresh"] + 1)]

    def conf(self, name):
        path = self.wt_root / name / ".claude/policy_finder.json"
        return json.loads(path.read_text()) if path.exists() else None

    def version_state(self, name, version):
        path = self.wt_root / f"{stem_of(name, version)}.state.json"
        return json.loads(path.read_text()) if path.exists() else None

    def lineage_best(self, name, upto=None):
        """The highest GP-peak final of the lineage's scored versions (<= upto)."""
        best = -math.inf
        conf = self.conf(name)
        last = conf.get("version", 1) if conf else 0
        for v in range(1, (upto or last) + 1):
            state = self.version_state(name, v) or {}
            score = state.get("data", {}).get("score")
            if state.get("status") == "scored" and score:
                best = max(best, score["pool"])
        return best

    def select(self):
        """The top_k lineages by lineage best (ties: the lower lineage number)."""
        names = self.lineages()
        ranked = sorted(names, key=lambda n: (-self.lineage_best(n), names.index(n)))
        return ranked[: self.settings["top_k"]]

    # -- running one version -------------------------------------------------

    def execute(self, args, out):
        """Run a command; its output appended to `out`. Returns the exit code."""
        with open(out, "a") as f:
            f.write(f"\n$ {' '.join(map(str, args))}\n")
            f.flush()
            return subprocess.run([str(a) for a in args], stdout=f, stderr=subprocess.STDOUT,
                                  env={**os.environ, "PF_RUNS": str(self.pf_runs),
                                       "PF_WORKTREE_ROOT": str(self.wt_root)}).returncode

    def start_command(self, name, version):
        """The command that starts (or resumes) `version` of lineage `name`."""
        s = self.settings
        conf = self.conf(name)
        current = conf.get("version", 1) if conf else 0
        if current == version:
            return [sys.executable, HERE / "pipeline.py", "run", name]
        if current == 0 and version == 1:
            if (self.wt_root / name).exists():
                raise CampaignError(f"{self.wt_root / name} exists without an instance file")
            command = [HERE / "new_instance.sh", name, "--min-params", s["min_params"],
                       "--max-params", s["max_params"], "--sobol-points", s["sobol_points"]]
            if s["setting"] != "exp2":
                command += ["--setting", s["setting"]]
            return command
        if current == version - 1 and current >= 1:
            command = [HERE / "new_instance.sh", "--iterate-on", name]
            previous = self.version_state(name, current) or {}
            if previous.get("status") == "failed":
                command.append("--after-failure")
            return command
        raise CampaignError(f"{name} is at version {current}, cannot run version {version}")

    def run_version(self, name, version):
        """Run one version until it is scored or failed; return its status."""
        stem = stem_of(name, version)
        out = self.wt_root / f"{stem}.campaign.out"
        self.wt_root.mkdir(parents=True, exist_ok=True)
        retries, ran = 0, False
        while True:
            state = self.version_state(name, version) or {}
            status = state.get("status")
            if status in ("scored", "failed"):
                self.log(f"{stem}: {status}" + (
                    f" {state['data']['score']['pool']:.2f}" if status == "scored" else
                    f" at {self._failed_step(state)}"))
                return status
            if state and status == "error":
                if ran:  # an error this run hit: give the environment time
                    time.sleep(self.settings["retry_wait"])
                if self.wait_for_cluster(stem):
                    # Raven was down: the error says nothing about the version
                    self.log(f"{stem}: error while Raven was down, resuming")
                else:
                    if retries >= self.settings["error_retries"]:
                        raise CampaignError(f"{stem}: error after {retries} retries: "
                                            f"{state.get('error', '')[-800:]}")
                    retries += 1
                    self.log(f"{stem}: error, resuming "
                             f"({retries}/{self.settings['error_retries']})")
            elif state and status == "running":
                self.log(f"{stem}: resuming")
            else:
                self.log(f"{stem}: starting")
            code = self.execute(self.start_command(name, version), out)
            ran = True
            after = self.version_state(name, version)
            if after is None or after == state and after.get("status") != "error":
                # the command failed before the pipeline wrote anything new
                raise CampaignError(f"{stem}: start failed (exit {code}), see {out}")

    @staticmethod
    def cluster_up():
        try:
            cluster_proxy.queue()
            return True
        except Exception:  # noqa: BLE001 - any failure to reach Raven
            return False

    def wait_for_cluster(self, stem):
        """Until Raven answers squeue; True if it did not at first."""
        waited = False
        while not self.cluster_up():
            if not waited:
                self.log(f"{stem}: Raven does not answer, waiting for it")
            waited = True
            time.sleep(max(self.settings["retry_wait"], 60))
        return waited

    @staticmethod
    def _failed_step(state):
        done = state.get("done", [])
        steps = ("export", "agent", "collect", "check", "sweep_config", "sweep_snap",
                 "image_check", "sweep_jobs", "sweep_fetch", "fit", "final_config",
                 "final_snap", "final_jobs", "final_fetch")
        return next((s for s in steps if s not in done), "?")

    # -- the campaign --------------------------------------------------------

    def members(self, index):
        """Generation `index`'s members, chosen (and saved) when it starts."""
        generations = self.state["generations"]
        if index < len(generations):
            return generations[index]["members"]
        if index == 0:
            names, version = self.lineages(), {n: 1 for n in self.lineages()}
        else:
            names = self.select()
            # a lineage's newest version that no generation holds yet (run
            # ahead of its selection, from the same history) is adopted
            # rather than followed by another
            held = {(m["name"], m["version"]) for g in generations for m in g["members"]}
            version = {}
            for n in names:
                current = self.conf(n)["version"]
                version[n] = current if (n, current) not in held else current + 1
            self.log(f"generation {index + 1}: lineage bests "
                     + ", ".join(f"{n} {self.lineage_best(n):.2f}" for n in names))
        members = [{"name": n, "version": version[n]} for n in names]
        generations.append({"index": index + 1, "members": members,
                            "started": time.strftime("%Y-%m-%d %H:%M:%S")})
        self.save()
        return members

    def run(self):
        total = 1 + self.settings["generations"]
        for index in range(total):
            members = self.members(index)
            self.log(f"generation {index + 1}/{total}: "
                     + ", ".join(stem_of(m["name"], m["version"]) for m in members))
            with ThreadPoolExecutor(self.settings["parallel"]) as pool:
                futures = [pool.submit(self.run_version, m["name"], m["version"])
                           for m in members]
                errors = []
                for future in futures:
                    try:
                        future.result()
                    except Exception as e:  # noqa: BLE001 - all members finish first
                        errors.append(e)
            self.write_summary()
            if errors:
                raise CampaignError("; ".join(map(str, errors)))
            self.state["generations"][index]["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
            self.save()
            self.summarize()
        self.log("campaign done")

    def summarize(self):
        command = self.settings.get("summarize")
        if not command:
            return
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        self.log(f"summarize exited {result.returncode}: {result.stdout[-500:]}"
                 f"{result.stderr[-500:]}")

    # -- the summary ---------------------------------------------------------

    def rows(self):
        rows = []
        for generation in self.state["generations"]:
            for m in generation["members"]:
                state = self.version_state(m["name"], m["version"]) or {}
                data = state.get("data", {})
                score = data.get("score") or {}
                cost = sum((r.get("result") or {}).get("total_cost_usd") or 0
                           for r in data.get("agent_runs", []))
                rows.append({
                    "generation": generation["index"], "name": m["name"],
                    "version": m["version"], "stem": stem_of(m["name"], m["version"]),
                    "status": state.get("status", "pending"),
                    "failed_step": (self._failed_step(state)
                                    if state.get("status") == "failed" else None),
                    "pool": score.get("pool"), "pool_se": score.get("pool_se"),
                    "best_of_n": score.get("best_of_n"),
                    "best_of_n_se": score.get("best_of_n_se"),
                    "n_points": score.get("n_points"),
                    "agent_cost_usd": round(cost, 2),
                    "times": state.get("times", {}),
                })
        return rows

    def write_summary(self):
        rows = self.rows()
        (self.dir / f"{self.path.stem}.rows.json").write_text(json.dumps(rows, indent=2) + "\n")

        def num(x, se=None):
            if x is None:
                return ""
            return f"{x:.2f}" + (f" ± {se:.2f}" if se is not None else "")

        lines = [f"# Campaign {self.path.stem}", "",
                 "| gen | version | status | GP peak | best of n | agent, API-equivalent $ |",
                 "|---|---|---|---|---|---|"]
        for r in rows:
            status = r["status"] + (f" ({r['failed_step']})" if r["failed_step"] else "")
            lines.append(f"| {r['generation']} | {r['stem']} | {status} "
                         f"| {num(r['pool'], r['pool_se'])} "
                         f"| {num(r['best_of_n'], r['best_of_n_se'])} | {r['agent_cost_usd']} |")
        lines += ["", "| lineage | best GP peak |", "|---|---|"]
        for n in sorted(self.lineages(), key=lambda n: -self.lineage_best(n)):
            best = self.lineage_best(n)
            lines.append(f"| {n} | {'' if best == -math.inf else f'{best:.2f}'} |")
        text = "\n".join(lines) + "\n"
        (self.dir / f"{self.path.stem}.summary.md").write_text(text)
        return text


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["run", "summary"])
    ap.add_argument("campaign")
    args = ap.parse_args()
    try:
        campaign = Campaign(args.campaign)
        if args.cmd == "summary":
            print(campaign.write_summary(), end="")
            return
        campaign.run()
    except CampaignError as e:
        sys.exit(f"FAIL: {e}")


if __name__ == "__main__":
    main()
