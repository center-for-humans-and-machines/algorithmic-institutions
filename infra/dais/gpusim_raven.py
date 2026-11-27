"""The Raven side of the GPU sim image comparison
(docs/reports/dais_gpu_sim_equivalence.md), through the policy finder's own
Raven path only (scripts/policy_finder/cluster_proxy.py: snapshot, run,
submit, wait, fetch). Runs where cluster_proxy runs (a container on the
command server's network, with its log and transfer folders mounted).

    python infra/dais/gpusim_raven.py REF OUT_DIR [--configs a.yml ...]
        snapshots REF; on the login node, in the Raven image, checksums the
        model files and captures and replays the models' forward pass
        (gpusim_forward.py); submits one sim job per config; waits; fetches
        each job's output dir and the forward files into OUT_DIR. Resumable:
        OUT_DIR/state.json keeps the snapshot, the job ids and what is
        fetched. --no-fetch stops after the wait.
"""

import argparse
import json
import sys
from pathlib import Path

import cluster_proxy as cp

CONFIGS = [
    f"infra/dais/gpusim_configs/{ref}_j{j}.yml"
    for ref in ("final1", "v7slice")
    for j in range(4)
]
MODELS = [
    "artifacts/artificial_humans/group_switching_contribution_50ep_vnode_stimulus"
    "_skip_herding_copula/model/"
    "architecture_node+edge+rnn__dataset_50ep__epochs_575.pt",
    "artifacts/artificial_humans/raven_script_22/model/rnn_False__dataset_full.pt",
    "artifacts/artificial_humans/switch_joint_exodus/model/"
    "architecture_mlp+rnn+edge__dataset_50ep_doubled.pt",
    "artifacts/baselines/punishment_multinomial_timeout_severity_copula.joblib",
]
FORWARD = "plots/gpusim/forward"


BEGIN, END = "__GPUSIM_BEGIN__", "__GPUSIM_END__"


def _via_log(sha, config, out):
    """A job's per-episode scores and sweep.json through the run log: reduced
    on the login node into the job's own snapshot output, then printed
    (gzip, base64). No /sync_from_cluster, so no shared transfer tree."""
    import base64
    import gzip

    stem = Path(config).stem
    sim = f"plots/simulation/gpusim/{stem}"
    code, log = cp.run(
        f"cd {cp.REPO} && OMP_NUM_THREADS=2 python -W ignore"
        f" infra/dais/gpusim_equiv.py episodes {config} {sim}/episodes.csv"
        f' && for f in episodes.csv sweep.json; do echo {BEGIN[:-2]}""__ $f;'
        f' gzip -c {sim}/$f | base64 -w0; echo; echo {END[:-2]}""__; done',
        sha,
        timeout=1800,
    )
    if code != 0:
        sys.exit(f"{stem}: reduce on the login node failed ({code}): {log[-1500:]}")
    lines = log.splitlines()
    target = out / "raven_gpu"
    target.mkdir(exist_ok=True)
    for i, line in enumerate(lines):
        if line.startswith(BEGIN + " "):
            name = line.split()[1]
            data = gzip.decompress(base64.b64decode(lines[i + 1].strip()))
            dest = target / (
                f"{stem}.csv" if name == "episodes.csv" else f"{stem}.json"
            )
            dest.write_bytes(data)
            print("got", dest, len(data), "bytes", flush=True)
    if not (target / f"{stem}.csv").exists():
        sys.exit(f"{stem}: no episodes in the log")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("out_dir")
    ap.add_argument("--configs", nargs="*", default=CONFIGS)
    ap.add_argument(
        "--via-log",
        action="store_true",
        help="bring back per-episode scores through the run log, not a fetch",
    )
    ap.add_argument(
        "--no-fetch", action="store_true", help="stop once the jobs have left the queue"
    )
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    state_file = out / "state.json"
    state = json.loads(state_file.read_text()) if state_file.exists() else {}

    def save():
        state_file.write_text(json.dumps(state, indent=2) + "\n")

    if "sha" not in state:
        state["sha"] = cp.snapshot(a.ref)
        save()
    sha = state["sha"]
    print("snapshot", sha, flush=True)

    jobs = state.setdefault("jobs", {})
    for config in a.configs:
        if config not in jobs:
            jobs[config] = cp.submit(config, sha, run_group="gpusim")
            save()
            print("submitted", config, jobs[config], flush=True)
    if "forward" not in state:
        py = "CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 python -W ignore"
        cfg = "infra/dais/gpusim_configs/final1_j0.yml"
        code, log = cp.run(
            f"cd {cp.REPO} && sha256sum {' '.join(MODELS)} && mkdir -p {FORWARD}"
            f" && {py} infra/dais/gpusim_forward.py capture {cfg}"
            f" {FORWARD}/capture_old.pkl --episodes 64"
            f" && {py} infra/dais/gpusim_forward.py replay {cfg}"
            f" {FORWARD}/capture_old.pkl {FORWARD}/replay_old_cpu_raven.npz",
            sha,
        )
        print(log[-3000:], flush=True)
        if code != 0:
            sys.exit(f"forward pass on the login node failed ({code})")
        state["forward"] = log[-3000:]
        save()

    cp.wait(list(jobs.values()), poll=60)
    print("queue empty", flush=True)
    if a.no_fetch:
        return

    fetched = state.setdefault("fetched", [])
    for config in a.configs:
        stem = Path(config).stem
        if stem in fetched:
            continue
        if a.via_log:
            _via_log(sha, config, out)
        else:
            n = cp.fetch(
                sha, f"plots/simulation/gpusim/{stem}", out / stem, f"gpusim_{stem}"
            )
            print("fetched", stem, n, "bytes", flush=True)
        fetched.append(stem)
        save()
    if "forward" not in fetched and not a.via_log:
        cp.fetch(sha, FORWARD, out / "forward", "gpusim_forward")
        fetched.append("forward")
        save()
    print("done", flush=True)


if __name__ == "__main__":
    main()
