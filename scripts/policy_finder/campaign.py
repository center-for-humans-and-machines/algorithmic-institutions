"""Run a policy-finder campaign: fresh lineages, then generations iterating the best.

A campaign is a YAML file (scripts/policy_finder/campaigns/*.yml):

    prefix: pf-20261104     # lineages <prefix>-1 ... <prefix>-<n_fresh>;
                            #   prefix and model may name an environment
                            #   variable, {env:NAME} (ENV_KEYS)
    n_fresh: 12             # generation 1: fresh lineages
    generations: 6          # then this many generations of --iterate-on
    top_k: 4                # ... on the lineages with the highest lineage best
    min_params: 1           # generation 1's param bounds; later versions
    max_params: 5           #   may declare any number (new_instance.sh),
    iterate_max_params: null  # or 0 to this many, when set
    sobol_points: 256
    parallel: 12            # versions running at once (agents are capped
                            #   separately by agent_runner's slots)
    error_retries: 3        # resumes of a version whose pipeline hit an error
    retry_wait: 600         # seconds before such a resume
    summarize: null         # a shell command run after each generation
                            #   (non-fatal; its output goes to the log)
    mode: free              # single_idea: generation 1 is the seed rule of
                            #   seeds/ in every lineage (no agent), and each
                            #   later version changes one idea of its
                            #   lineage's best version (new_instance.sh)
    openrouter_max_usage: null  # harness opencode on OpenRouter: the key's
                            #   total usage (USD) at which agents no longer
                            #   start and the campaign stops (else
                            #   PF_OPENROUTER_MAX_USAGE)
    harness: claude         # opencode: OpenCode on an OpenRouter model
    model: null             #   (harness opencode: an OpenRouter id), routed
    providers: null         #   to these upstream providers if given, in
                            #   order, no fallback (agent_runner.py)
    base_url: null          # harness opencode on a server of our own instead
                            #   of OpenRouter (model: its served name); on
                            #   DAIS "{env:PF_LLM_BASE_URL}", which the job
                            #   sets (infra/dais/campaign.slurm.sh)

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
import re
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
    "setting": "exp2", "mode": "free", "iterate_max_params": None,
    "summarize": None, "harness": "claude", "model": None, "providers": None,
    "base_url": None,
}


class CampaignError(Exception):
    pass


#: Settings that may name an environment variable, `{env:NAME}`, expanded when
#: the campaign is loaded: one file then serves every model, the job naming
#: the served model (PF_LLM_MODEL) and a short tag for the prefix
#: (PF_MODEL_TAG; infra/dais/campaign.slurm.sh). `base_url` is not among
#: them: OpenCode reads it from the agent's environment itself.
ENV_KEYS = ("prefix", "model")
ENV_REF = re.compile(r"\{env:([A-Za-z_][A-Za-z0-9_]*)\}")
#: A prefix as new_instance.sh accepts instance names
PREFIX = re.compile(r"[a-z0-9][a-z0-9_-]*")


def expand_env(value, key, path):
    """`value` with every `{env:NAME}` replaced by that variable's value."""
    if not isinstance(value, str):
        return value

    def sub(match):
        found = os.environ.get(match.group(1))
        if not found:
            raise CampaignError(f"{path}: {key} names {match.group(1)}, which is not set")
        return found
    return ENV_REF.sub(sub, value)


def check_harness(s, path):
    """The harness settings: a known harness, a model for opencode, providers
    as a list of names, a cap as a number."""
    if s["harness"] not in ("claude", "opencode"):
        raise CampaignError(f"{path}: harness is claude or opencode, not {s['harness']!r}")
    if s["harness"] == "claude":
        for key in ("model", "providers"):
            if s[key] is not None:
                raise CampaignError(f"{path}: {key} is for harness opencode")
        return
    if not isinstance(s["model"], str) or not s["model"].strip():
        raise CampaignError(f"{path}: harness opencode needs model (an OpenRouter id)")
    providers = s["providers"]
    if providers is not None and (
            not isinstance(providers, list)
            or not all(isinstance(p, str) and p and "," not in p for p in providers)):
        raise CampaignError(f"{path}: providers is a list of provider names, "
                            f"not {providers!r}")
    cap = s.get("openrouter_max_usage")
    if cap is not None and (isinstance(cap, bool) or not isinstance(cap, (int, float))):
        raise CampaignError(f"{path}: openrouter_max_usage is a number (USD)")


def stem_of(name, version):
    return name if version == 1 else f"{name}-v{version}"


class Campaign:
    def __init__(self, path, pf_runs=None):
        self.path = Path(path)
        self.settings = {**DEFAULTS, **yaml.safe_load(self.path.read_text())}
        if not self.settings.get("prefix"):
            raise CampaignError(f"{path}: no prefix")
        for key in ENV_KEYS:
            self.settings[key] = expand_env(self.settings[key], key, path)
        if not PREFIX.fullmatch(str(self.settings["prefix"])):
            raise CampaignError(f"{path}: prefix {self.settings['prefix']!r} is not"
                                " [a-z0-9][a-z0-9_-]*")
        check_harness(self.settings, path)
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

    @staticmethod
    def rank_score(score):
        """What ranks a final: its paired gain over the setting's baseline
        when it has one (generate_sim_config.py SETTINGS `baseline`; each
        version's final plays its own games), else its score."""
        return score["gain"] if score.get("gain") is not None else score["pool"]

    def lineage_best(self, name, upto=None):
        """The highest GP-peak final of the lineage's scored versions (<= upto),
        by rank_score."""
        best = -math.inf
        conf = self.conf(name)
        last = conf.get("version", 1) if conf else 0
        for v in range(1, (upto or last) + 1):
            state = self.version_state(name, v) or {}
            score = state.get("data", {}).get("score")
            if state.get("status") == "scored" and score:
                best = max(best, self.rank_score(score))
        return best

    def select(self):
        """The top_k lineages by lineage best (ties: the lower lineage number)."""
        names = self.lineages()
        ranked = sorted(names, key=lambda n: (-self.lineage_best(n), names.index(n)))
        return ranked[: self.settings["top_k"]]

    # -- running one version -------------------------------------------------

    def env(self):
        """The pipeline's environment from the campaign file."""
        env = {"PF_RUNS": str(self.pf_runs), "PF_WORKTREE_ROOT": str(self.wt_root)}
        if self.settings.get("openrouter_max_usage") is not None:
            env["PF_OPENROUTER_MAX_USAGE"] = str(self.settings["openrouter_max_usage"])
        return env

    def execute(self, args, out):
        """Run a command; its output appended to `out`. Returns the exit code."""
        with open(out, "a") as f:
            f.write(f"\n$ {' '.join(map(str, args))}\n")
            f.flush()
            return subprocess.run([str(a) for a in args], stdout=f, stderr=subprocess.STDOUT,
                                  env={**os.environ, **self.env()}).returncode

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
            if s["mode"] != "free":
                command += ["--mode", s["mode"]]
            if s["harness"] != "claude":
                # the agent's harness and model (agent_runner.py); later versions keep them
                command += ["--harness", s["harness"], "--model", s["model"]]
                if s["providers"]:
                    command += ["--providers", ",".join(s["providers"])]
                if s["base_url"]:
                    command += ["--base-url", s["base_url"]]
            return command
        if current == version - 1 and current >= 1:
            command = [HERE / "new_instance.sh", "--iterate-on", name]
            previous = self.version_state(name, current) or {}
            if previous.get("status") == "failed":
                command.append("--after-failure")
            if s["iterate_max_params"] is not None:
                command += ["--min-params", 0, "--max-params", s["iterate_max_params"]]
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
            if status == "stopped" and ran:
                # the agent's budget (agent_runner.BudgetError) in this run: not
                # the version's; a later `campaign run` resumes it
                raise CampaignError(f"{stem}: stopped: {state.get('error', '')[-800:]}")
            if status in ("scored", "failed"):
                self.log(f"{stem}: {status}" + (
                    f" {state['data']['score']['pool']:.2f}"
                    + (f" (gain {state['data']['score']['gain']:+.2f})"
                       if state['data']['score'].get('gain') is not None else "")
                    if status == "scored" else
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
            elif state and status in ("running", "stopped"):
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
                runs = data.get("agent_runs", [])
                cost = sum((r.get("result") or {}).get("total_cost_usd") or 0 for r in runs)
                # harness opencode: OpenCode's catalogue cost and the tokens
                catalogue = sum((r.get("result") or {}).get("cost_catalog_usd") or 0
                                for r in runs)
                tokens = {k: sum(((r.get("result") or {}).get("tokens") or {}).get(k) or 0
                                 for r in runs)
                          for k in ("input", "cache_read", "output")}
                rows.append({
                    "generation": generation["index"], "name": m["name"],
                    "version": m["version"], "stem": stem_of(m["name"], m["version"]),
                    "status": state.get("status", "pending"),
                    "failed_step": (self._failed_step(state)
                                    if state.get("status") == "failed" else None),
                    "pool": score.get("pool"), "pool_se": score.get("pool_se"),
                    "gain": score.get("gain"), "gain_se": score.get("gain_se"),
                    "best_of_n": score.get("best_of_n"),
                    "best_of_n_se": score.get("best_of_n_se"),
                    "n_points": score.get("n_points"),
                    "pools": score.get("pools"),
                    "agent_cost_usd": round(cost, 2),
                    "agent_catalogue_usd": round(catalogue, 3),
                    "agent_tokens": tokens,
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

        # against each opponent, when the setting has several (exp2_multi)
        rivals = sorted({k for r in rows for k in (r.get("pools") or {})})
        extra = "".join(f" {k} |" for k in rivals)
        gains = any(r.get("gain") is not None for r in rows)
        extra = (" gain over baseline |" if gains else "") + extra
        opencode = self.settings["harness"] == "opencode"
        # Claude: the plan's API-equivalent cost; OpenCode: its own price
        # catalogue (not the endpoint's bill) and the tokens
        agent_head = (" agent: OpenCode catalogue $, tokens in / cached / out (k) |"
                      if opencode else " agent, API-equivalent $ |")
        lines = [f"# Campaign {self.path.stem}", "",
                 f"| gen | version | status | GP peak | best of n |{extra}{agent_head}",
                 "|---|---|---|---|---|" + "---|" * (len(rivals) + gains) + "---|"]
        for r in rows:
            status = r["status"] + (f" ({r['failed_step']})" if r["failed_step"] else "")
            pools = r.get("pools") or {}
            cells = "".join(
                f" {num(pools[k]['pool'], pools[k]['pool_se']) if k in pools else ''} |"
                for k in rivals
            )
            if gains:
                cells = f" {num(r.get('gain'), r.get('gain_se'))} |" + cells
            if opencode:
                t = r.get("agent_tokens") or {}
                agent = (f"{r.get('agent_catalogue_usd', 0)}, "
                         + " / ".join(f"{(t.get(k) or 0) / 1000:.0f}"
                                      for k in ("input", "cache_read", "output")))
            else:
                agent = r["agent_cost_usd"]
            lines.append(f"| {r['generation']} | {r['stem']} | {status} "
                         f"| {num(r['pool'], r['pool_se'])} "
                         f"| {num(r['best_of_n'], r['best_of_n_se'])} |{cells}"
                         f" {agent} |")
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
