"""The DAIS model server's launch lines (infra/dais/common.sh serve_model) on
one node and across nodes, without a GPU: apptainer, srun, taskset and module
are stubs that print their command line. The job scripts run under DAIS's
bash 4.4; an older bash (macOS's 3.2) skips these tests."""

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).parent / "fixtures" / "serve_model"


def bash_version():
    out = subprocess.run(["bash", "-c", "echo ${BASH_VERSINFO[0]} ${BASH_VERSINFO[1]}"],
                         capture_output=True, text=True).stdout.split()
    return tuple(int(x) for x in out)


pytestmark = pytest.mark.skipif(bash_version() < (4, 4),
                                reason="the DAIS job scripts need bash >= 4.4")

STUBS = r"""
module() { :; }
apptainer() { printf '%q ' apptainer "$@"; echo; }
srun() { printf '%q ' srun "$@"; echo; }
taskset() { printf 'taskset %q %q ' "$1" "$2"; shift 2; "$@"; }
export -f apptainer srun taskset
"""


def serve(tmp_path, env, body):
    """Run `body` after sourcing common.sh in a job-like environment; its
    stdout, stderr and exit code, with the port and the folder normalised."""
    script = (STUBS + 'source "$REPO/infra/dais/common.sh"\ninit_run > /dev/null\n'
              + body)
    base = {"PATH": "/usr/bin:/bin", "USER": "u", "REPO": str(ROOT),
            "SLURM_SUBMIT_DIR": str(ROOT),
            "PF_SCRATCH": f"{tmp_path}/scratch", "HF_HOME": "/models",
            "SLURM_JOB_NAME": "t", "SLURM_JOB_ID": "1", "CUDA_VISIBLE_DEVICES": "0,1"}
    r = subprocess.run(["bash", "-c", "set -euo pipefail\n" + script],
                       env={**base, **env}, capture_output=True, text=True)

    def norm(text):
        text = re.sub(r"--(master-)?port [0-9]+", r"--\1port PORT", text)
        return text.replace(str(tmp_path), "OUT")
    return norm(r.stdout), norm(r.stderr), r.returncode


@pytest.mark.parametrize("profile", ["default", "deepseek-v4.1-flash-1017"])
def test_one_node_is_the_old_command_line(tmp_path, profile):
    env = {} if profile == "default" else {"MODEL_PROFILE": profile}
    out, err, rc = serve(tmp_path, env, 'SERVER_CPUS=0-3 serve_model 0.90 2 '
                         '"$PF_RUN/vllm.log" --extra-arg\nwait\n'
                         'cat "$PF_RUN"/vllm*.log\necho "pids ${#SERVER_PIDS[@]}"')
    assert rc == 0, err
    log, pids = out.rsplit("pids ", 1)
    assert log == (FIXTURES / f"{profile}.txt").read_text()
    assert pids.strip() == "1"


TWO = {"MODEL_PROFILE": "kimi-k3.1-pro", "SERVE_DRY_RUN": "1",
       "SERVER_NODES_OVERRIDE": "nodeA nodeB", "SERVER_HEAD_IP": "10.0.0.1",
       "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7"}


def test_two_nodes_start_a_head_and_a_headless_worker(tmp_path):
    out, err, rc = serve(tmp_path, TWO, 'serve_model 0.90 8 "$PF_RUN/vllm.log" --x')
    assert rc == 0, err
    log = "OUT/scratch/runs/t.1/vllm.node1.log"
    head, worker = out.split(f"worker 1 on nodeB, log {log}:")
    assert head.startswith("head on nodeA")
    common = ("--tensor-parallel-size 8", "--pipeline-parallel-size 2", "--nnodes 2",
              "--master-addr 10.0.0.1", "--master-port PORT",
              "--distributed-executor-backend mp", "--bind /dev/infiniband",
              "NCCL_SOCKET_IFNAME=ib0", "GLOO_SOCKET_IFNAME=ib0", "--x",
              "vllm serve moonshotai/Kimi-K3.1-Pro --revision 80aff651")
    for part in common:
        assert part in head and part in worker, part
    assert "--node-rank 0" in head and "--headless" not in head
    assert "--port PORT" in head and "VLLM_HOST_IP=10.0.0.1" in head
    assert worker.strip().startswith(
        "srun --nodes=1 --ntasks=1 --nodelist=nodeB --overlap")
    assert "/bin/bash -c" in worker
    assert worker.rstrip().endswith("--node-rank 1 --headless")
    assert "--gres=gpu:8" in worker and "home.node1" in worker
    assert "--host" not in worker and "VLLM_API_KEY" not in worker


def test_too_few_nodes_fail_clearly(tmp_path):
    env = {**TWO, "SERVER_NODES_OVERRIDE": "nodeA"}
    out, err, rc = serve(tmp_path, env, 'serve_model 0.90 8 "$PF_RUN/vllm.log"')
    assert rc != 0 and "head on" not in out
    assert "spans 2 nodes" in err and "this job has 1 (nodeA)" in err and "-N 2" in err


def test_one_node_needs_no_node_list(tmp_path):
    env = {**TWO, "SERVER_NODES": "1", "SERVER_NODES_OVERRIDE": ""}
    out, err, rc = serve(tmp_path, env, 'serve_model 0.90 8 "$PF_RUN/vllm.log"')
    assert rc == 0, err
    assert "worker" not in out and "--nnodes" not in out and "infiniband" not in out
