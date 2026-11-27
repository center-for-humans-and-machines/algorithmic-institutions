#!/bin/bash -l
# Check the agent container on a DAIS node, without a model server: OpenCode
# starts from the read-only SIF under --containall with a home on scratch,
# reads the local-server config, and tries nothing on the network at start
# (it runs under noinet.py here, so any attempt shows in its log); the agent's
# shell (agent_shell.sh + noinet.py, as $SHELL) refuses internet sockets but
# runs what the agent legitimately does (pandas, subprocesses and pipes,
# multiprocessing, validate-rule). Prints PASS/FAIL lines; exits 1 on a FAIL.
#
#   sbatch infra/dais/agent_check.slurm.sh
#   # with the model: vLLM on the job's GPUs, then one OpenCode session in the
#   # container that runs the network checks through its own shell tool, and
#   # the sampling settings vLLM received (its request log)
#   sbatch --export=WITH_MODEL=1 -p gpudev --qos=debug --gres=gpu:2 \
#     --cpus-per-task=24 --mem=400000 --time=00:15:00 infra/dais/agent_check.slurm.sh
#
#SBATCH -J pf_agentchk
#SBATCH -o /dais/fs/scratch/%u/policy-finder-agents/logs/%x.%j.out
#SBATCH --account=mpib_inst
#SBATCH --partition=small
#SBATCH --qos=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000
#SBATCH --time=00:20:00

set -uo pipefail
export PF_SCRATCH="${PF_SCRATCH:-/dais/fs/scratch/$USER/policy-finder-agents}"
export PF_REPO="${PF_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$PF_REPO/infra/dais/common.sh"
init_run
REPO=/opt/algorithmic-institutions
fails=0
check() {  # check <name> <expect: ok|fail> <command...>
  local name="$1" expect="$2" rc=0
  shift 2
  "$@" > "$PF_RUN/$name.log" 2>&1 || rc=$?
  if { [ "$expect" = ok ] && [ "$rc" = 0 ]; } || { [ "$expect" = fail ] && [ "$rc" != 0 ]; }; then
    echo "PASS $name (exit $rc, expected $expect)"
  else
    echo "FAIL $name (exit $rc, expected $expect)"; fails=$((fails + 1))
  fi
  tail -3 "$PF_RUN/$name.log" | sed 's/^/     | /'
}

echo "node $(hostname), $(lscpu | sed -n 's/^Model name: *//p')"
sha=$(sha256sum "$AGENT_SIF" | cut -d' ' -f1)
[ "$sha" = "$AGENT_SIF_SHA256" ] && echo "PASS image sha256 $sha" \
  || { echo "FAIL image sha256 $sha"; fails=$((fails + 1)); }
apptainer inspect --environment "$AGENT_SIF" | grep -E "PATH|XDG|OPENCODE|PYTHON" | sed 's/^/env | /'

# an export stand-in: the checkout read-only, its .opencode from export_instance
mkdir -p "$PF_RUN/oc/agents" "$PF_RUN/tmp" "$PF_RUN/notes"
ro_home=()
for d in .opencode .config .claude .agents; do  # agent_apptainer.HOME_READONLY
  mkdir -p "$PF_RUN/home/$d"; ro_home+=(--bind "$PF_RUN/home/$d:/home/claude/$d:ro")
done
"${PF_PYTHON:-$PF_SCRATCH/venv/bin/python}" - "$PF_RUN/oc" <<'EOF'
import json, sys
sys.path.insert(0, "scripts/policy_finder")
import export_instance as ex
out = sys.argv[1]
conf = ex.opencode_config({"model": "Qwen/Qwen3-Coder-Next",
                           "base_url": "{env:PF_LLM_BASE_URL}"})
json.dump(conf, open(f"{out}/opencode.json", "w"), indent=2)
open(f"{out}/.gitignore", "w").write(ex.OPENCODE_GITIGNORE)
open(f"{out}/../home/.opencode/.gitignore", "w").write(ex.OPENCODE_GITIGNORE)
open(f"{out}/agents/policy-finder.md", "w").write(
    "---\ndescription: check\nmode: primary\n---\nSay ok.\n")
EOF
box() {  # the container as agent_apptainer.py starts it
  APPTAINERENV_PF_LLM_API_KEY="${LLM_KEY:-dummy}" apptainer exec --containall --cleanenv --no-mount cwd,tmp,bind-paths \
    --home "$PF_RUN/home:/home/claude" \
    --bind "$PF_REPO:$REPO:ro" --bind "$PF_RUN/oc:$REPO/.opencode:ro" \
    --bind "$PF_RUN/notes:$REPO/notes/policy_finder:rw" \
    --bind "$PF_RUN/tmp:/tmp:rw" "${ro_home[@]}" \
    --bind "$PF_REPO/infra/claude_proxy/container/CC_env/opencode/opencode.json:/opt/opencode/config/opencode/opencode.json:ro" \
    --bind "$PF_HERE/agent_shell.sh:/opt/pf-agent/shell:ro" \
    --bind "$PF_HERE/noinet.py:/opt/pf-agent/noinet.py:ro" \
    --env SHELL=/opt/pf-agent/shell --env OMP_NUM_THREADS=2 \
    --env PF_LLM_BASE_URL="${LLM_URL:-http://127.0.0.1:9/v1}" \
    --env PYTHONDONTWRITEBYTECODE=1 \
    --pwd "$REPO" "$AGENT_SIF" "$@"
}
sh_() { box /opt/pf-agent/shell -c "$1"; }

check opencode_version ok box opencode --version
check opencode_config ok box opencode debug config
grep -E '"(model|baseURL|enabled_providers|temperature|top_k|webfetch|websearch)"' -A1 \
  "$PF_RUN/opencode_config.log" | head -30 | sed 's/^/     cfg | /'
check opencode_agents ok box opencode agent list
# OpenCode itself under the filter: whatever it fetches at start fails and is logged
# (`debug wait` does not end by itself: exit 124 from timeout is the pass)
check opencode_start_offline ok box bash -c '/usr/bin/python3 /opt/pf-agent/noinet.py \
  timeout 30 opencode debug wait; [ $? = 124 ]'
box sh -c 'ls ~/.local/share/opencode/log/ && cat ~/.local/share/opencode/log/*.log' \
  > "$PF_RUN/opencode_logs.txt" 2>&1
grep -iE "fetch|http|npm|install|ECONN|EPERM|error" "$PF_RUN/opencode_logs.txt" | head -20 \
  | sed 's/^/     log | /'

# the agent's shell: no internet, everything else
# unfiltered, the container reaches the internet by address (names do not
# resolve in it: no resolv.conf under --no-mount bind-paths)
check shell_plain_curl_ip_works ok box bash -c 'curl -sS -m 10 -o /dev/null http://1.1.1.1'
check shell_curl fail sh_ 'curl -sS -m 10 -o /dev/null https://huggingface.co'
check shell_curl_ip fail sh_ 'curl -sS -m 10 -o /dev/null http://1.1.1.1'
check shell_curl_loopback fail sh_ 'curl -sS -m 5 -o /dev/null http://127.0.0.1:9/'
check shell_py_inet fail sh_ 'python3 -c "import socket; socket.socket(socket.AF_INET)"'
check shell_py_inet6 fail sh_ 'python3 -c "import socket; socket.socket(socket.AF_INET6)"'
check shell_pip fail sh_ 'pip download --no-deps -d /tmp/x requests'
check shell_py_unix ok sh_ 'python3 -c "import socket; socket.socket(socket.AF_UNIX); socket.socketpair()"'
check shell_pipes ok sh_ 'seq 1 100000 | sort -n | tail -1 | grep -qx 100000 && (echo a; echo b) | wc -l'
check shell_subprocess ok sh_ 'python3 -c "import subprocess; print(subprocess.run([\"ls\", \"-1\"], capture_output=True, text=True).stdout[:50])"'
check shell_multiprocessing ok sh_ 'python -c "
import multiprocessing as mp
def f(x): return x * x
if __name__ == \"__main__\":
    with mp.Pool(2) as p: print(sum(p.map(f, range(1000))))
    q = mp.Manager().Queue(); q.put(1); print(q.get())
"'
check shell_pandas ok sh_ 'python -c "
import pandas as pd
df = pd.read_csv(\"experiments/2group_8agent_50ep.csv\")
df.describe().to_csv(\"notes/policy_finder/check_describe.csv\")
print(df.shape)
"'
check shell_validate_rule ok sh_ 'python -m aimanager validate-rule scripts/policy_finder/seeds/single_idea.yml --min-params 0 --max-params 0 --sobol-points 16 --setting exp2'
check shell_write_readonly fail sh_ 'touch src/x'
check shell_write_notes ok sh_ 'touch notes/policy_finder/x && rm notes/policy_finder/x'
# what the unfiltered side runs or loads must not be writable from the filter
check shell_overwrite_wrapper fail sh_ 'echo x > /opt/pf-agent/shell'
check shell_overwrite_noinet fail sh_ 'echo x > /opt/pf-agent/noinet.py'
check shell_overwrite_config fail sh_ 'echo {} > .opencode/opencode.json'
check shell_overwrite_global_config fail sh_ 'echo {} > /opt/opencode/config/opencode/opencode.json'
check shell_home_opencode fail sh_ 'mkdir -p ~/.opencode/plugin && touch ~/.opencode/plugin/x.ts'
check shell_home_config fail sh_ 'mkdir -p ~/.config/opencode && touch ~/.config/opencode/opencode.json'
check shell_path_readonly ok sh_ 'echo "PATH=$PATH"; for d in ${PATH//:/ }; do [ -d "$d" ] || continue; if touch "$d/.w" 2>/dev/null; then rm -f "$d/.w"; echo "writable: $d"; exit 1; fi; done'
check shell_bashrc_ignored ok sh_ 'echo "touch /tmp/bashrc_ran" > ~/.bashrc; /opt/pf-agent/shell -c true; rm -f ~/.bashrc; [ ! -e /tmp/bashrc_ran ]'
# the host's user services are out of sight under --containall
check run_user_absent ok sh_ '[ ! -e /run/user ] || [ -z "$(ls -A /run/user)" ]'
check systemd_run_user fail sh_ 'systemd-run --user true'
check busctl_user fail sh_ 'busctl --user list'
check tmux_ls fail sh_ 'tmux ls'
check screen_ls fail sh_ 'screen -ls | grep -q "[0-9]\."'
check shell_process_tree ok sh_ 'ps -o pid,ppid,args --forest; grep -E "Seccomp|NoNewPrivs" /proc/self/status'

# agent_apptainer.py's PF_AGENT_STEP=1: a `srun --overlap` step caps memory,
# and taskset inside it pins to CPUs of our choosing (the agent's group)
if [ -n "${SLURM_JOB_ID:-}" ]; then
  C=$(python3 "$PF_HERE/cpu_slots.py" slots 1 1 1)  # the second core's CPUs
  step() { srun --overlap --ntasks=1 --cpus-per-task=2 --mem=1024M --cpu-bind=none \
    --kill-on-bad-exit=1 --job-name=agent-check taskset -c "$C" "$@"; }
  check step_taskset ok step python3 -c "import os; a = sorted(os.sched_getaffinity(0)); print(a); assert a == sorted(map(int, '$C'.split(',')))"
  check step_oom fail step python3 -c "b = b'x' * (3 * 2**30); print(len(b))"
  check step_agent ok step apptainer exec --containall --cleanenv "$AGENT_SIF" opencode --version
fi

if [ "${WITH_MODEL:-0}" = 1 ]; then
  nvidia-smi --query-gpu=index,name --format=csv,noheader | sed 's/^/GPU /'
  assert_vllm_image || exit 1
  NGPU=$(tr ',' '\n' <<< "$CUDA_VISIBLE_DEVICES" | wc -l)
  SERVER_CPUS=$(python3 "$PF_HERE/cpu_slots.py" reserved 4) \
    serve_model 0.90 "$NGPU" "$PF_RUN/vllm.log" --enable-log-requests --moe-backend triton
  if wait_ready 720; then
    LLM_URL="$SERVER_URL"; LLM_KEY=$(cat "$SERVER_KEY_FILE")
    prompt='Use your bash tool to run exactly these four commands, one tool call each, and then reply with one line per command saying whether it succeeded: (1) curl -sS -m 10 -o /dev/null http://1.1.1.1 (2) python3 -c "import socket; socket.socket(socket.AF_INET)" (3) ps -o pid,ppid,args --forest (4) grep -E "Seccomp|NoNewPrivs" /proc/self/status'
    t0=$SECONDS
    check model_session ok box timeout 600 opencode run --agent policy-finder \
      --format json "$prompt"
    echo "     session took $(( SECONDS - t0 )) s"
    "${PF_PYTHON:-$PF_SCRATCH/venv/bin/python}" "$PF_HERE/agent_events.py" \
      "$PF_RUN/model_session.log"
    grep -o "SamplingParams([^)]*)" "$PF_RUN/vllm.log" | sort | uniq -c | head -5 \
      | sed 's/^/     vllm | /'
    # the same through the backend the pipeline uses (agent_apptainer.py and
    # opencode_run.py, the runner): a stand-in export, one run, its record
    exp="$PF_RUN/pf_runs/exports/pfchk"
    mkdir -p "$exp"
    (cd "$PF_REPO" && tar --exclude=.git -cf - src docs experiments configs) | tar -xf - -C "$exp"
    cp -r "$PF_RUN/oc" "$exp/.opencode"
    mkdir -p "$exp/notes/policy_finder"
    check backend_run ok env PF_RUNS="$PF_RUN/pf_runs" PF_AGENT_BACKEND=apptainer \
      PF_AGENT_SIF="$AGENT_SIF" PF_AGENT_POOL_CPUS="$(python3 "$PF_HERE/cpu_slots.py" slots 5 1 2 | paste -sd, -)" \
      PF_AGENT_SHELL="$PF_HERE/agent_shell.sh" PF_AGENT_NOINET="$PF_HERE/noinet.py" \
      PF_LLM_BASE_URL="$SERVER_URL" PF_LLM_API_KEY_FILE="$SERVER_KEY_FILE" PF_AGENT_TIMEOUT=900 \
      "${PF_PYTHON:-$PF_SCRATCH/venv/bin/python}" - <<'EOF'
import json, sys
sys.path.insert(0, "scripts/policy_finder")
import agent_runner as ar
run = ar.start("pfchk", ["notes/policy_finder"], "Use your bash tool once to run "
               "`curl -sS -m 5 http://1.1.1.1`, then write the file "
               "notes/policy_finder/check.md with one line saying what happened, "
               "and reply with that line.", harness="opencode")
rec = ar.finish(run, harness="opencode")
r = rec["result"]
print("status", rec["status"], "error", rec["error"], "models", rec["models"])
print("failures", r.get("failures"), "seconds", r.get("seconds"), "tokens", r.get("tokens"))
print("audit", json.dumps(rec["network_audit"])[:400])
print("stray", r.get("stray_processes"), "ended_on", r.get("ended_on"),
      "report_chars", r.get("report_chars"), "compactions", r.get("compactions"))
print("report", rec["report"][:300])
source, paths = ar.transcripts(run, "opencode")
key = open(__import__("os").environ["PF_LLM_API_KEY_FILE"]).read().strip()
leaks = [p for p in paths if key in open(p, errors="replace").read()]
print("transcripts", len(paths), "key in", leaks)
sys.exit(0 if rec["status"] == "succeeded" and not leaks
         and not (r.get("stray_processes") or {}).get("enabled") else 1)
EOF
    sed 's/^/     run  | /' "$PF_RUN/backend_run.log" | tail -9
    ls -la "$exp/notes/policy_finder" | sed 's/^/     out  | /'
  else
    tail -30 "$PF_RUN/vllm.log"; fails=$((fails + 1))
  fi
fi

echo "$fails failure(s)"
[ "$fails" = 0 ]
