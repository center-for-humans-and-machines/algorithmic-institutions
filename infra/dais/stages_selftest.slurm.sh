#!/bin/bash -l
# Exercise campaign.slurm.sh's stages loop with stubbed commands (no GPU, no
# model, no agents): a fake common.sh (server = a sleep process), curl,
# nvidia-smi, taskset, cpu_slots.py, preflight.py, campaign.py,
# campaign_check.py and reevaluate.py; the real kill_job_children.py. In a
# Slurm job, because the server watch kills every process of the job's
# cgroup but its own. Scenarios:
#   A  two healthy stages (one with a re-evaluation, one with finished/10)
#   B  a failing first stage: the job stops there, exit 1
#   C  the server dies during a stage: the watch kills the campaign, exit 3
#   D  a re-evaluation already done is skipped; a partial one moved aside
#   E  a kill test whose campaign ends before any agent ran fails at once
#   F  a kill test done in an earlier job (its marker) is skipped
#
#   sbatch infra/dais/stages_selftest.slurm.sh
#
#SBATCH -J pf_stagestest
#SBATCH -o /dais/fs/scratch/%u/policy-finder-agents/logs/%x.%j.out
#SBATCH --account=mpib_inst
#SBATCH --partition=small
#SBATCH --qos=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4000
#SBATCH --time=00:15:00

set -uo pipefail
SRC="${SLURM_SUBMIT_DIR:-$PWD}"
T=$(mktemp -d -p "${TMPDIR:-/tmp}" pfstages.XXXXXX)
trap 'rm -rf "$T"' EXIT
fails=0

make_tree() {  # a fresh fake checkout in $T/repo
  rm -rf "${T:?}/repo" "${T:?}/bin" "${T:?}/scratch"
  mkdir -p "$T/repo/infra/dais" "$T/repo/scripts/policy_finder" "$T/bin"
  cp "$SRC/infra/dais/campaign.slurm.sh" "$SRC/infra/dais/kill_job_children.py" \
    "$T/repo/infra/dais/"
  git -C "$T/repo" init -q
  git -C "$T/repo" -c user.name=t -c user.email=t@t commit -q --allow-empty -m init
  cat > "$T/repo/infra/dais/common.sh" <<'EOF'
PF_HERE="$PF_REPO/infra/dais"
SIM_SIF=/dev/null SIM_SIF_SHA256=x AGENT_SIF=/dev/null AGENT_SIF_SHA256=x MODEL=m MODEL_REV=r VLLM_SIF=/dev/null MAX_MODEL_LEN=131072
init_run() { PF_RUN="$PF_SCRATCH/run"; mkdir -p "$PF_RUN/tmp"; BG_PIDS=(); }
cleanup() { local p; for p in "${BG_PIDS[@]}"; do kill "$p" 2>/dev/null || true; done; }
assert_vllm_image() { :; }
serve_model() {
  SERVER_URL=http://127.0.0.1:9/v1 SERVER_KEY_FILE="$PF_RUN/key" SERVER_AUTH_HEADER="$PF_RUN/auth"
  echo k > "$SERVER_KEY_FILE"; echo "Authorization: Bearer k" > "$SERVER_AUTH_HEADER"
  echo "INFO a log line" > "$3"
  sleep 100000 & SERVER_PID=$!; echo "$SERVER_PID" > "$PF_RUN/server.pid"
  BG_PIDS+=("$SERVER_PID")
}
wait_ready() { echo "ready after 0 s"; }
EOF
  cat > "$T/bin/curl" <<'EOF'
#!/bin/bash
# /health: 200 while the fake server lives; anything else: nothing
for a in "$@"; do
  if [[ "$a" == */health ]]; then
    if kill -0 "$(cat "$STUB_RUN/server.pid")" 2>/dev/null; then echo -n 200; else echo -n 000; fi
    exit 0
  fi
done
exit 0
EOF
  printf '#!/bin/bash\necho stub-gpu\n' > "$T/bin/nvidia-smi"
  printf '#!/bin/bash\nshift 2; exec "$@"\n' > "$T/bin/taskset"
  cat > "$T/repo/infra/dais/cpu_slots.py" <<'EOF'
import sys
first, n = int(sys.argv[2]), int(sys.argv[4]) if len(sys.argv) > 4 else 4
for c in range(first, first + n):
    print(f"{c},{c + 100}")
EOF
  echo 'import sys; sys.exit(0)' > "$T/repo/scripts/policy_finder/preflight.py"
  cat > "$T/repo/scripts/policy_finder/campaign.py" <<'EOF'
import os, sys, time
name = os.path.basename(sys.argv[2])
time.sleep(float(os.environ.get("STUB_SECONDS", "1")))
sys.exit(1 if os.environ.get("STUB_FAIL", "-") in name else 0)
EOF
  echo 'import sys; print("check", sys.argv[1:]); sys.exit(0)' \
    > "$T/repo/infra/dais/campaign_check.py"
  cat > "$T/repo/infra/dais/reevaluate.py" <<'EOF'
import os, sys
os.makedirs(sys.argv[2], exist_ok=True)
open(os.path.join(sys.argv[2], "summary.json"), "w").write('{"improvement": 1}\n')
print("reevaluated", sys.argv[1:])
EOF
  chmod +x "$T/bin"/*
}

run_job() {  # run_job <name> <stages text> [ENV=VALUE ...]: exit code; log in $T/<name>.log
  local name="$1" stages="$2"
  shift 2
  printf '%s\n' "$stages" > "$T/repo/stages.txt"
  ( cd "$T/repo" && env PATH="$T/bin:$PATH" SLURM_SUBMIT_DIR="$T/repo" PF_REPO="$T/repo" \
      PF_SCRATCH="$T/scratch" PF_RUNS="$T/scratch/pf_runs" PF_PYTHON=python3 PF_BASE=x \
      CUDA_VISIBLE_DEVICES=0,1 STUB_RUN="$T/scratch/run" WATCH_INTERVAL=1 WATCH_FAILS=2 "$@" \
      bash "$T/repo/infra/dais/campaign.slurm.sh" stages stages.txt ) > "$T/$name.log" 2>&1
}

expect() {  # expect <name> <exit code> <got> <grep pattern> [<absent pattern>]
  if [ "$3" = "$2" ] && grep -qE "$4" "$T/$1.log" \
      && { [ -z "${5:-}" ] || ! grep -qE "$5" "$T/$1.log"; }; then
    echo "PASS $1 (exit $3)"
  else
    echo "FAIL $1 (exit $3, expected $2)"; fails=$((fails + 1)); sed 's/^/     | /' "$T/$1.log" | tail -25
  fi
}

make_tree
run_job A "run scored 20 20 a.yml
# a comment
run finished/10+nolimits default default b.yml:2 c.yml"
expect A 0 $? "re-evaluation exit 0" "stage .* failed"
grep -q -- "'--min-first-scored', '10', '--no-expired', '--no-limit-stops'" "$T/A.log" \
  && echo "PASS A passes finished/10 to the check" \
  || { echo "FAIL A: no --min-first-scored"; fails=$((fails + 1)); }

make_tree  # a setting with a reference list re-evaluates with it
mkdir -p "$T/repo/infra/dais/references"
echo "zero: null" > "$T/repo/infra/dais/references/exp2.yml"
printf 'prefix: r\nsetting: exp2\n' > "$T/repo/r.yml"
run_job R "run scored 20 20 r.yml:2"
grep -q "re-evaluation in 2 slices .*--references .*references/exp2.yml" "$T/R.log" \
  && echo "PASS R the setting's references are passed" \
  || { echo "FAIL R: no --references"; fails=$((fails + 1)); sed 's/^/     | /' "$T/R.log" | tail -15; }

make_tree
run_job B "run scored 20 20 a.yml
run scored 20 20 b.yml" STUB_FAIL=a
expect B 1 $? "== stage run failed: stopping" "campaign b: start"

make_tree
( sleep 4; kill "$(cat "$T/scratch/run/server.pid")" ) &
run_job C "run scored 20 20 a.yml
run scored 20 20 b.yml" STUB_SECONDS=60
expect C 3 $? "the model server went down" "campaign b: start"
grep -q "^kill .*campaign.py" "$T/C.log" && echo "PASS C killed the campaign" \
  || { echo "FAIL C: campaign not killed"; fails=$((fails + 1)); }

make_tree
mkdir -p "$T/scratch/pf_runs/reevaluate/a" "$T/scratch/pf_runs/reevaluate/b"
echo '{}' > "$T/scratch/pf_runs/reevaluate/a/summary.json"
run_job D "run scored 20 20 a.yml:2 b.yml:2"
expect D 0 $? "campaign a: re-evaluation already done"
ls -d "$T/scratch/pf_runs/reevaluate/b.partial."* >/dev/null 2>&1 \
  && [ -f "$T/scratch/pf_runs/reevaluate/b/summary.json" ] \
  && echo "PASS D partial moved aside, new result at the expected path" \
  || { echo "FAIL D"; fails=$((fails + 1)); }

make_tree
run_job E "killtest 20 20 a.yml
run scored 20 20 a.yml" STUB_SECONDS=1 KILL_AFTER=1
expect E 1 $? "the campaign ended before an agent had run" "campaign a: start"

make_tree
mkdir -p "$T/scratch/pf_runs/campaign"
echo "job 1" > "$T/scratch/pf_runs/campaign/a.killtest-done"
run_job F "killtest 20 20 a.yml
run scored 20 20 a.yml"
expect F 0 $? "done in an earlier job"

echo "$fails failure(s)"
[ "$fails" = 0 ]
