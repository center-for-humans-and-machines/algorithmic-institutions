"""Check that this Claude proxy can run the policy finder, before a first run.

Run inside the proxy (`docker compose ... run claude`), from the repo root:

    python scripts/policy_finder/preflight.py            # everything
    python scripts/policy_finder/preflight.py --local    # skip Raven
    python scripts/policy_finder/preflight.py --fix      # also create PF_RUNS folders

Each check prints PASS, WARN or FAIL with what to do. Exit code 1 on any FAIL.

Local: RAVEN_USER and PF_RUNS, the base branch, git identity, the data files
the agent's export copies (export_instance.AGENT_DATA) and the models the
sweep sims load. Services: the container spawner and the command server.
Raven (through the command server): squeue over SSH, then one login-node
command in algoinst.sif on the live checkout: the Raven user, torch and PyG,
the writable folders and every model the sweep sims load.

What it cannot see from here: the host (infra/claude_proxy/host_check.sh),
the Claude token inside the spawner (a smoke instance shows it) and the slurm
template on Raven (infra/raven/raven_setup.sh --check).
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import cluster_proxy  # noqa: E402
import export_instance  # noqa: E402

ROOT = HERE.parents[1]
SPAWNER = os.environ.get("PF_SPAWNER", "http://container_spawner:8877")
#: The sim configs the sweep is built from (generate_sim_config.py), whose
#: `artifacts/...` paths every sweep job loads on Raven
SIM_SOURCES = (
    "configs/simulation/manager_testing/25_LEVIN_run1_ah_zero_pairings_batched.yml",
    "configs/simulation/manager_testing/29_EXP1_reference.yml",
    "configs/simulation/opponents/exp2_multi.yml",
    "configs/simulation/opponents/exp2_diff.yml",
)
FETCH_HINT = "python scripts/fetch_large_files.py --via command-server {}"

failures = 0


def report(status, what, hint=""):
    global failures
    if status == "FAIL":
        failures += 1
    line = f"[{status}] {what}"
    if hint and status != "PASS":
        line += f"\n         -> {hint}"
    print(line, flush=True)


def git(*args):
    out = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True)
    return out.returncode, out.stdout.strip()


def model_paths():
    paths = set()
    for source in SIM_SOURCES:
        f = ROOT / source
        if f.exists():
            paths.update(re.findall(r"(artifacts/[^\s'\"]+)", f.read_text()))
    return sorted(paths)


def check_local(fix):
    user = os.environ.get("RAVEN_USER", "")
    if user:
        report("PASS", f"RAVEN_USER={user}")
    else:
        report("FAIL", "RAVEN_USER is not set", "set it in infra/claude_proxy/container/CC_env/.env and restart the proxy")

    pf_runs = os.environ.get("PF_RUNS", "")
    if not pf_runs:
        report("FAIL", "PF_RUNS is not set", "compose sets it; start the proxy from infra/claude_proxy/container/CC_env/docker-compose.yml")
    elif not os.path.isdir(pf_runs):
        report("FAIL", f"PF_RUNS={pf_runs} is not a folder here", "create it on the host (host_check.sh --fix)")
    else:
        # exports/ is the spawner's mount root; worktrees/ and locks/ are made on demand
        exports = os.path.join(pf_runs, "exports")
        if not os.path.isdir(exports) and fix:
            os.makedirs(exports, exist_ok=True)
        for path in (pf_runs, exports):
            if os.path.isdir(path) and os.access(path, os.W_OK):
                report("PASS", f"{path} writable")
            else:
                report("FAIL", f"{path} missing or not writable",
                       "run with --fix, or check the owner on the host (the proxy runs as HOST_UID)")

    base = os.environ.get("PF_BASE", "main")
    code, sha = git("rev-parse", "--verify", "--quiet", base)
    if code == 0:
        report("PASS", f"base branch {base} ({sha[:8]})")
    else:
        report("FAIL", f"base branch {base} does not exist", "git fetch origin && git branch main origin/main, or set PF_BASE")
    _, name = git("config", "user.name")
    _, email = git("config", "user.email")
    if name and email:
        report("PASS", f"git identity {name} <{email}>")
    else:
        report("WARN" if name else "FAIL", "git user.email is not set: the pipeline commits on instance branches",
               "set GIT_USER_EMAIL in infra/claude_proxy/container/CC_env/.env and rebuild the proxy "
               "(or git config --global user.email ...)")

    for pattern in export_instance.AGENT_DATA:
        if glob.glob(str(ROOT / pattern)):
            report("PASS", f"agent data {pattern}")
        else:
            report("FAIL", f"agent data {pattern} missing", FETCH_HINT.format(repr(pattern)))

    missing = [p for p in model_paths() if not (ROOT / p).exists()]
    if missing:
        report("WARN", f"{len(missing)} sweep model file(s) missing locally (needed on Raven, sent from here)",
               FETCH_HINT.format(" ".join(repr(p) for p in missing)))
    else:
        report("PASS", "sweep model files present locally")


def check_services():
    try:
        with urllib.request.urlopen(SPAWNER + "/health", timeout=10) as r:
            health = json.loads(r.read().decode())
        if health.get("ok") and health.get("project") == "algoinst":
            report("PASS", "container spawner (project algoinst)")
        else:
            report("FAIL", f"container spawner answers {health}", "CC_PROJECT=algoinst on the container_spawner service")
    except (urllib.error.URLError, OSError) as e:
        report("FAIL", f"container spawner unreachable ({e})", "docker compose ... up -d container_spawner")

    try:
        response = cluster_proxy._request("GET", f"/jobs?cluster={cluster_proxy.CLUSTER}")
        code, log = cluster_proxy.wait_run(response["run_id"], timeout=600)
    except Exception as e:  # noqa: BLE001 - any failure here is one message
        report("FAIL", f"command server unreachable: {str(e)[:300]}",
               "docker compose ... up -d cc_command_server")
        return False
    if code == 0 and "JOBID" in log:
        report("PASS", "command server reaches Raven (squeue)")
    elif "slurm" in log.lower():
        # SSH got through; Slurm itself answered with an error (often transient)
        report("WARN", f"SSH to Raven works, but squeue failed: {log.strip().splitlines()[-1][:200]}",
               "retry later; if it persists, check Raven's status page")
    else:
        report("FAIL", f"command server cannot run commands on Raven ({code}): {log[-300:]}",
               "docker compose ... exec cc_command_server bash -ic 'ssh -fN raven' (password + OTP)")
        return False
    return True


def check_raven():
    user = os.environ.get("RAVEN_USER", "")
    tests = ["id -un",
             "python -c 'import torch, torch_geometric; print(\"torch\", torch.__version__, \"pyg\", torch_geometric.__version__)'",
             "for d in /tmp /opt/cluster_transfer /opt/outputs; do test -w $d || echo NOTWRITABLE $d; done"]
    models = model_paths()
    if models:
        tests.append("for f in " + " ".join(models) + "; do test -e $f || echo MISSING $f; done")
    command = f"cd {cluster_proxy.REPO} && " + " ; ".join(tests)
    try:
        code, log = cluster_proxy.run(command, timeout=900)
    except Exception as e:  # noqa: BLE001
        report("FAIL", f"login-node command in algoinst.sif: {str(e)[:300]}",
               "on Raven: bash infra/raven/raven_setup.sh --check")
        return
    if code != 0 and "torch" not in log:
        report("FAIL", f"login-node command in algoinst.sif exited {code}: {log[-600:]}",
               "on Raven: bash infra/raven/raven_setup.sh --check (image, folders, live checkout); "
               "on the host: raven_user in CC_command_server/env.yml")
        return
    lines = log.splitlines()
    if user and not any(line.strip() == user for line in lines):
        report("FAIL", f"Raven commands do not run as {user}",
               "the User in cc_ssh_keys/config, and raven_user in CC_command_server/env.yml")
    else:
        report("PASS", f"Raven commands run as {user}")
    version = next((line for line in lines if line.startswith("torch ")), None)
    if version:
        report("PASS", f"algoinst.sif: {version}")
    else:
        report("FAIL", "algoinst.sif did not import torch and torch_geometric", "on Raven: bash infra/raven/raven_setup.sh --check")
    unwritable = [line.split()[1] for line in lines if line.startswith("NOTWRITABLE ")]
    if unwritable:
        report("FAIL", f"not writable in the image: {', '.join(unwritable)}",
               "on Raven: bash infra/raven/raven_setup.sh (creates /ptmp/$USER/cc/...)")
    elif version:
        report("PASS", "/tmp, /opt/cluster_transfer, /opt/outputs writable in the image")
    missing = [line.split()[1] for line in lines if line.startswith("MISSING ")]
    if missing:
        report("FAIL", f"{len(missing)} model file(s) missing in the Raven live checkout: {', '.join(missing)[:400]}",
               'fetch them here, then POST /sync_to_cluster {"cluster": "raven", "includes": ["artifacts"]} (docs/handover/handover.md)')
    elif models:
        report("PASS", f"{len(models)} sweep model files in the Raven live checkout")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--local", action="store_true", help="skip the services and Raven")
    ap.add_argument("--fix", action="store_true", help="create missing PF_RUNS folders")
    args = ap.parse_args()
    print("== local")
    check_local(args.fix)
    if not args.local:
        print("== services")
        if check_services():
            print("== Raven")
            check_raven()
    print(f"\n{'OK' if not failures else f'{failures} check(s) failed'}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
