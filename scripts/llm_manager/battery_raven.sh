#!/usr/bin/env bash
#
# Submit the standard battery with the language model as one of its arms:
# N vLLM servers on the job's own node, then `run_battery.py --llm` against
# them. The baselines run in the same invocation, on the same stack, from the
# same seed schedule -- which is the whole reason the battery takes them
# together rather than quoting them from another table.
#
# Run this ON Raven, from the checkout you want measured.
#
#   scripts/llm_manager/battery_raven.sh                      # 8B, 4 servers
#   CONFIG=configs/llm_manager/qwen3_32b.yaml TP=2 \
#     SEEDS=42,43,44,45,46 scripts/llm_manager/battery_raven.sh
#
# MANY SMALL ROLLOUTS, NOT ONE BIG ONE. `EPISODES` is the width of a single
# batched rollout and `SEEDS` is how many of them are pooled, so the total is
# EPISODES x |SEEDS| independent episodes. The serving measurement found the
# limit is the KV cache rather than concurrency: while the whole population's
# traces fit in GPU memory a round re-prefills only the block that changed,
# and past that every round re-prefills its entire trace. The crossover sits
# sharply between 200 and 400 episodes at 8B on four A100s, where a decision
# is 2.3x cheaper below it -- so the default width is 200 and the budget is
# bought with seeds. `harness.run_arm` already pools them into one frame of
# independent episodes, and every arm faces the same schedule.
#
# The width is a property of THIS prompt length, THIS round count and THIS
# GPU count, not a constant. At 32B there is no cheap regime at all -- the
# weights leave too little KV cache -- so there the width buys nothing and
# only the total matters.
#
# Knobs, all optional:
#   CONFIG     manager + serving config  (configs/llm_manager/qwen3_8b.yaml)
#   MODEL      HF id, else from CONFIG   (Qwen/Qwen3-8B)
#   TP         GPUs per server           (1)
#   GPUS       cards to request          (4)
#   TIME       wall clock limit          (04:00:00)
#   EPISODES   width of one rollout      (200)
#   SEEDS      comma-separated seeds     (42..71, i.e. 30 x 200 = 6,000)
#   OUT        results directory         (plots/data_analysis/llm_manager_battery/$TAG)
#   SMOKE      tiny run first, 0 to skip (1)
#   WORK       scratch + model cache     (/ptmp/$USER/llm-manager-2026-09-22)
#   TAG        names the output dir      (derived from MODEL)

set -euo pipefail

CONFIG="${CONFIG:-configs/llm_manager/qwen3_8b.yaml}"
MODEL="${MODEL:-Qwen/Qwen3-8B}"
TP="${TP:-1}"
GPUS="${GPUS:-4}"
TIME="${TIME:-04:00:00}"
EPISODES="${EPISODES:-200}"
SEEDS="${SEEDS:-$(seq -s, 42 71)}"
SMOKE="${SMOKE:-1}"
# The weights are already cached here, 77 GB of them; re-downloading them is
# the one avoidable cost in this job.
WORK="${WORK:-/ptmp/$USER/llm-manager-2026-09-22}"
TAG="${TAG:-$(basename "$MODEL" | tr '[:upper:]' '[:lower:]')}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PY:-$HOME/algorithmic-institutions/.venv/bin/python}"
OUT="${OUT:-$REPO/plots/data_analysis/llm_manager_battery/$TAG}"

mkdir -p "$OUT" "$REPO/temp"

# THE TRAP: without this the package resolves to the shared checkout and the
# run measures code other than this one. It is also what makes the evaluation
# score 22 rows rather than silently scoring 21.
export PYTHONPATH="$REPO/src"

N_SEEDS=$(awk -F, '{print NF}' <<<"$SEEDS")
echo "[battery] repo     $REPO"
echo "[battery] config   $CONFIG"
echo "[battery] model    $MODEL  tp=$TP  gpus=$GPUS"
echo "[battery] budget   $EPISODES episodes x $N_SEEDS seeds = $((EPISODES * N_SEEDS)) per arm"
echo "[battery] out      $OUT"

# The smoke run and the real run share one job, so the smoke run costs the
# seconds it takes rather than a second four-minute server startup. If it
# fails the script stops and the real run never starts.
RUN="set -euo pipefail"
if [ "$SMOKE" != "0" ]; then
    RUN="$RUN
echo '[battery] smoke run: 8 episodes x 1 seed, every path once'
'$PY' '$REPO/scripts/llm_manager/run_battery.py' \\
    --llm '$REPO/$CONFIG' --out '$OUT/smoke' \\
    --episodes 8 --chunk 8 --seeds 42 --no-stub
echo '[battery] smoke run OK'"
fi
RUN="$RUN
echo '[battery] the run'
'$PY' '$REPO/scripts/llm_manager/run_battery.py' \\
    --llm '$REPO/$CONFIG' --out '$OUT' \\
    --episodes '$EPISODES' --chunk '$EPISODES' --seeds '$SEEDS'"

sbatch \
    --chdir="$REPO" \
    --gres="gpu:a100:$GPUS" \
    --time="$TIME" \
    --job-name="llmbat-$TAG" \
    --output="$REPO/temp/battery-$TAG-%j.log" \
    --error="$REPO/temp/battery-$TAG-%j.log" \
    --export="ALL,PYTHONPATH=$REPO/src,LLM_MODEL=$MODEL,LLM_TP_SIZE=$TP,LLM_WORK=$WORK,LLM_HF_HOME=$WORK/hf_cache,LLM_CONTAINER=/ptmp/$USER/containers/vllm.sif" \
    "$REPO/scripts/llm_manager/serve_vllm.slurm.sh" \
    bash -c "$RUN"
