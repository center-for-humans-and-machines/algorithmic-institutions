"""Check that this Claude proxy can run the policy finder, before a first run.

Run inside the proxy (`docker compose ... run claude`), from the repo root:

    python scripts/policy_finder/preflight.py            # everything
    python scripts/policy_finder/preflight.py --local    # skip Raven
    python scripts/policy_finder/preflight.py --fix      # also create PF_RUNS folders
    python scripts/policy_finder/preflight.py --dais     # a DAIS job (no proxy, no Raven)

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

--dais checks what infra/dais/campaign.slurm.sh needs instead of the
services and Raven: the images with their expected checksums (PF_SIM_SIF,
PF_AGENT_SIF against SIM_SIF_SHA256, AGENT_SIF_SHA256), the weights at the
pinned revision (HF_HOME, MODEL, MODEL_REV), the driver's imports, every
setting's agent data in the checkout and the sweep models in PF_ARTIFACTS,
and write access to the run folders.
"""

import argparse
import glob
import hashlib
import importlib
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
#: Files the agent definitions promise beside the settings' agent_data globs
#: (.claude/agents/policy-finder.md, policy-finder-exp1.md: the reference
#: sims' per-round data, not in git)
PROMISED_DATA = (
    "plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet",
    "plots/simulation/29_EXP1_reference/per_round.parquet",
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


def check_local(fix, dais=False):
    user = os.environ.get("RAVEN_USER", "")
    if dais:
        pass  # no Raven
    elif user:
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

    if dais:
        return
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


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def check_dais():
    for var, sha_var in (("PF_SIM_SIF", "SIM_SIF_SHA256"), ("PF_AGENT_SIF", "AGENT_SIF_SHA256")):
        path, want = os.environ.get(var, ""), os.environ.get(sha_var, "")
        if not path or not os.path.isfile(path):
            report("FAIL", f"{var} is not a file: {path!r}", "infra/dais/README.md, Agents on DAIS")
        elif not want:
            report("FAIL", f"{sha_var} is not set", "infra/dais/common.sh sets it")
        elif sha256(path) != want:
            report("FAIL", f"{path} is not the expected image", f"its sha256 should be {want}")
        else:
            report("PASS", f"{var} {path} (sha256 {want[:12]})")

    hf, model, rev = (os.environ.get(v, "") for v in ("HF_HOME", "MODEL", "MODEL_REV"))
    snap = Path(hf, "hub", "models--" + model.replace("/", "--"), "snapshots", rev)
    index = snap / "model.safetensors.index.json"
    if not (hf and model and rev) or not index.is_file():
        report("FAIL", f"no weights of {model!r} at revision {rev!r} under {hf!r}",
               "download them with HF_HOME set (docs/reports/dais_feasibility.md)")
    else:
        files = set(json.loads(index.read_text())["weight_map"].values())
        missing = [f for f in files if not (snap / f).exists()]
        if missing:
            report("FAIL", f"{len(missing)} weight file(s) of {model} missing: {missing[:3]}")
        else:
            report("PASS", f"weights {model}@{rev[:8]}: {len(files)} files")

    bad = []
    for name in ("torch", "numpy", "scipy", "sklearn", "pandas", "yaml"):
        try:
            importlib.import_module(name)
        except Exception as e:  # noqa: BLE001
            bad.append(f"{name} ({e})")
    if bad:
        report("FAIL", f"the driver cannot import {', '.join(bad)}", "infra/dais/README.md: the driver's python")
    else:
        report("PASS", f"driver python {sys.executable} imports torch, numpy, scipy, sklearn, pandas, yaml")

    # files, not folders: a reference folder with only its tracked plots
    # passed once although its per_round.parquet was missing
    missing = []
    for pattern in sorted({p for s in export_instance.SETTINGS.values()
                           for p in s["agent_data"]} | set(PROMISED_DATA)):
        files = [f for f in glob.glob(str(ROOT / pattern)) if os.path.isfile(f)]
        if not files:
            missing.append(pattern)
        for f in files:
            with open(f, "rb") as handle:
                if handle.read(40).startswith(b"version https://git-lfs"):
                    missing.append(f"{os.path.relpath(f, ROOT)} (a git-lfs pointer)")
    if missing:
        report("FAIL", f"agent data missing in the checkout: {missing}",
               "copy them from a checkout that has them (docs/large_files.md), or "
               "recreate the reference sims (infra/dais/reference_sims.slurm.sh)")
    else:
        report("PASS", "every setting's agent data in the checkout, files present")
    artifacts = Path(os.environ.get("PF_ARTIFACTS", ROOT / "artifacts"))
    missing = [p for p in model_paths()
               if not (artifacts / os.path.relpath(p, "artifacts")).exists()]
    if missing:
        report("FAIL", f"{len(missing)} sweep model file(s) missing in {artifacts}: {missing[:3]}")
    else:
        report("PASS", f"sweep model files in {artifacts}")

    pf_runs = Path(os.environ.get("PF_RUNS", ""))
    for sub in ("", "exports", "agent_runs", "local_cluster", "worktrees"):
        path = pf_runs / sub
        try:
            path.mkdir(parents=True, exist_ok=True)
            probe = path / ".preflight"
            probe.write_text("ok\n")
            probe.unlink()
            report("PASS", f"{path} writable")
        except OSError as e:
            report("FAIL", f"{path} not writable: {e}")


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
    ap.add_argument("--dais", action="store_true", help="a DAIS job: images, weights, "
                    "driver, data and run folders instead of the services and Raven")
    args = ap.parse_args()
    print("== local")
    check_local(args.fix or args.dais, dais=args.dais)
    if args.dais:
        print("== DAIS")
        check_dais()
    elif not args.local:
        print("== services")
        if check_services():
            print("== Raven")
            check_raven()
    print(f"\n{'OK' if not failures else f'{failures} check(s) failed'}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
