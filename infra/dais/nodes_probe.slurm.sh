#!/bin/bash -l
# The model server across nodes, small: Qwen3-Coder-Next (the default model)
# as a head and one headless worker, pipeline parallel over the two, started
# by the same serve_model as a campaign (common.sh, SERVER_NODES=2); then
# /health, one chat completion and the tool calls of probe_endpoint.py. First
# each node's network: its interfaces, InfiniBand ports and /dev/infiniband.
#
# On a job of one node (gpudev and gpu1 take one node per job) head and
# worker share the node: the head on the job's first GPU, the worker, through
# srun --overlap as on a second node, on its second; they meet at the node's
# address on SERVER_NET_IF, and NCCL is kept off NVLink and shared memory
# (NCCL_P2P_DISABLE, NCCL_SHM_DISABLE), so that its data goes through the
# network transport, the InfiniBand HCA:
#   sbatch -p gpudev --qos=debug --gres=gpu:b200:2 -c 24 --mem=400000 -t 00:15:00 \
#     -o <log> --export=PF_SCRATCH=<scratch> infra/dais/nodes_probe.slurm.sh
# On a job of two nodes (partition gpu only) the worker runs on the second:
#   sbatch -p gpu --qos=g0008 -N 2 --ntasks=2 --ntasks-per-node=1 --gres=gpu:b200:8 --exclusive --mem=0 -c 96 \
#     -t 00:30:00 -o <log> --export=PF_SCRATCH=<scratch>,SERVER_GPUS=4 \
#     infra/dais/nodes_probe.slurm.sh
# NCCL logs at INFO (NCCL_DEBUG); the evidence lines are printed at the end.
#
#SBATCH -J pf_nodes
#SBATCH --account=mpib_inst
#SBATCH --ntasks=1
#SBATCH --time=00:15:00

set -euo pipefail
export SERVER_NODES=2 MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
source "${SLURM_SUBMIT_DIR:-.}/infra/dais/common.sh"
init_run
mapfile -t nodes < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
read -r -a gpus <<< "${CUDA_VISIBLE_DEVICES//,/ }"
echo "job ${SLURM_JOB_ID:-local}: nodes ${nodes[*]}, GPUs ${CUDA_VISIBLE_DEVICES}"

# the network of a node, as the server would use it
FABRIC='echo "== $(hostname -s)"; ip -4 -br addr | grep -v "^lo"
ls /dev/infiniband 2>&1 | paste -sd" " | sed "s/^/\/dev\/infiniband: /"
for d in /sys/class/infiniband/*; do
  [ -e "$d" ] || continue
  echo "$(basename "$d"): $(cat "$d/ports/1/state" "$d/ports/1/rate" "$d/ports/1/link_layer" 2>/dev/null | paste -sd" ")"
done'
bash -c "$FABRIC"
for node in "${nodes[@]:1}"; do
  srun --nodes=1 --ntasks=1 --nodelist="$node" --overlap --gres=gpu:0 bash -c "$FABRIC" || true
done
nvidia-smi topo -m | head -20 || true

assert_vllm_image
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
MODEL_VLLM_ENV="$MODEL_VLLM_ENV NCCL_DEBUG_SUBSYS=INIT,NET"
if (( ${#nodes[@]} == 1 )); then
  (( ${#gpus[@]} >= 2 )) || { echo "FATAL: one node needs 2 GPUs" >&2; exit 2; }
  echo "one node: head on GPU ${gpus[0]}, worker on GPU ${gpus[1]}, NCCL over the network transport"
  export SERVER_NODES_OVERRIDE="${nodes[0]} ${nodes[0]}"
  export SERVER_WORKER_DEVICES="${gpus[1]}" SERVER_WORKER_GPUS="${#gpus[@]}" SERVER_WORKER_CPUS=8
  MODEL_VLLM_ENV="$MODEL_VLLM_ENV NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1"
  head_devices="${gpus[0]}" tp=1
else
  tp="${SERVER_GPUS:-1}"
  head_devices=$(tr ',' '\n' <<< "$CUDA_VISIBLE_DEVICES" | head -n "$tp" | paste -sd, -)
fi
t0=$SECONDS
CUDA_VISIBLE_DEVICES="$head_devices" SERVER_CPUS="${SERVER_CPUS:-}" \
  serve_model 0.90 "$tp" "$PF_RUN/vllm.log" --enforce-eager
rc=0
if wait_ready "${READY_TIMEOUT:-780}"; then
  health=$(curl -s -m 20 -o /dev/null -w '%{http_code}' -H "@$SERVER_AUTH_HEADER" \
    "${SERVER_URL%/v1}/health")
  echo "/health: $health"
  [ "$health" = 200 ] || rc=1
  in_vllm_image python3 /opt/dais/probe_endpoint.py --base "$SERVER_URL" \
    --key-file "$SERVER_KEY_FILE" --model "$MODEL" --trials 1 --long-tokens 2000 || rc=1
else
  rc=1
fi
echo "after $(( SECONDS - t0 )) s"
echo "== evidence: head ($PF_RUN/vllm.log)"
grep -hE "DP group leader|world_size|NET/IB|NET/Socket|via NET|Using network|NCCL_SOCKET_IFNAME|ncclCommInitRank.*Init COMPLETE|Error|error" \
  "$PF_RUN/vllm.log" | cut -c1-220 | sort | uniq -c | sort -rn | head -25 || true
for log in "$PF_RUN"/vllm.node*.log; do
  [ -f "$log" ] || continue
  echo "== evidence: worker ($log)"
  grep -hE "worker on|headless|head node address|NET/IB|NET/Socket|via NET|Using network|Init COMPLETE|Error|error|FATAL" \
    "$log" | cut -c1-220 | sort | uniq -c | sort -rn | head -25 || true
done
echo "PROBE $([ "$rc" = 0 ] && echo OK || echo FAILED)"
exit "$rc"
