#!/usr/bin/env bash
# One-time Raven setup for the algoinst Claude proxy (policy finder).
#
# Run as yourself on a Raven login node. Safe to rerun: it only adds what is
# missing. --check changes nothing and only reports.
#
#   bash raven_setup.sh            # set up, then check
#   bash raven_setup.sh --check    # check only
#
# The command server reaches Raven as $USER and expects, for that user:
#   /ptmp/$USER/cc/{tmp,container_outputs,cluster_transfer,cluster_sync,
#                   container_cache,apptainer-home}       bound into every job
#   read access to the shared image (CC_command_server's SIF_OWNER tree):
#   /raven/ptmp/ykeller/cc/containers/apptainer/algoinst.sif
#   ~/slurm_tools and ~/.venv with slurm_tools installed  (`slurm_job`)
#   ~/algorithmic-institutions/artifacts                  models, read-only
#                                                         in every job
#   ~/cc-snapshots, ~/cc-snapshot-outputs                 one per commit
#   ~/private/cc_secrets                                  WANDB_API_KEY=...
#                                                         (optional: jobs warn
#                                                         and run without it)
#
# The live checkout's contents come from the proxy: after this script,
# POST /sync_to_cluster {"cluster": "raven", "includes": ["artifacts"]}
# (docs/handover/handover.md). Nothing here needs git credentials or S3.
#
# Overrides: ALGOINST_SHARED_SIF (the image to check; jobs use the one
# CC_command_server's SIF_OWNER names), SLURM_TOOLS_REPO,
# PYTHON (a Python >= 3.9 for ~/.venv).

set -uo pipefail

CHECK=0
[[ "${1:-}" == "--check" ]] && CHECK=1
U="$(id -un)"
SHARED_SIF="${ALGOINST_SHARED_SIF:-/raven/ptmp/ykeller/cc/containers/apptainer/algoinst.sif}"
SLURM_TOOLS_REPO="${SLURM_TOOLS_REPO:-https://github.com/yannikkellerde/slurm_tools.git}"
PTMP="/ptmp/$U/cc"
SIF="$SHARED_SIF"
TEMPLATE="$HOME/slurm_tools/slurm_tools/run_slurm_raven_apptainer_algoinst.sh"
FAILS=0

pass() { echo "[PASS] $1"; }
warn() { echo "[WARN] $1"; [[ -n "${2:-}" ]] && echo "         -> $2"; return 0; }
fail() { echo "[FAIL] $1"; [[ -n "${2:-}" ]] && echo "         -> $2"; FAILS=$((FAILS + 1)); }
doing() { [[ "$CHECK" == 0 ]]; }

# `module` is a shell function that non-interactive shells may not have
if ! command -v module >/dev/null 2>&1; then
    for f in /etc/profile.d/modules.sh /usr/share/lmod/lmod/init/bash; do
        # shellcheck disable=SC1090
        [[ -f "$f" ]] && source "$f" && break
    done
fi

echo "== folders"
for d in "$PTMP/tmp" "$PTMP/container_outputs" "$PTMP/cluster_transfer" "$PTMP/cluster_sync" \
         "$PTMP/container_cache/wandb" "$PTMP/apptainer-home" \
         "$HOME/cc-snapshots" "$HOME/cc-snapshot-outputs" "$HOME/algorithmic-institutions/artifacts"; do
    if [[ -d "$d" ]]; then
        pass "$d"
    elif doing && mkdir -p "$d"; then
        pass "$d (created)"
    else
        fail "$d missing" "rerun without --check"
    fi
done

echo "== image (shared, used in place by the command server)"
if [[ -r "$SIF" ]]; then
    pass "$SIF readable"
else
    fail "$SIF is not readable by $U" \
        "its owner grants read on the file and execute on every parent folder"
fi
if [[ -r "$SIF" ]] && module load apptainer >/dev/null 2>&1; then
    if out="$(apptainer exec --cleanenv "$SIF" python -c \
        'import torch, torch_geometric; print(torch.__version__, torch_geometric.__version__)' 2>&1)"; then
        pass "image runs: torch, PyG $(echo "$out" | tail -n1)"
    else
        fail "image does not import torch and torch_geometric" "$(echo "$out" | tail -n3 | tr '\n' ' ')"
    fi
elif [[ -r "$SIF" ]]; then
    warn "module load apptainer failed; image not tested" "run on a login node with the module system"
fi

echo "== slurm_tools (job template and slurm_job)"
if [[ -d "$HOME/slurm_tools/.git" ]]; then
    pass "$HOME/slurm_tools ($(git -C "$HOME/slurm_tools" log -1 --format=%h 2>/dev/null))"
elif doing && git clone -q "$SLURM_TOOLS_REPO" "$HOME/slurm_tools"; then
    pass "$HOME/slurm_tools (cloned)"
else
    fail "$HOME/slurm_tools missing" "rerun without --check, or git clone $SLURM_TOOLS_REPO ~/slurm_tools"
fi
if [[ -f "$TEMPLATE" ]]; then
    if [[ "$U" != ykeller ]] && grep -q "ykeller" "$TEMPLATE" "$HOME/slurm_tools/slurm_tools/do_slurm_job.py"; then
        fail "slurm_tools still has ykeller's paths" \
            "git -C ~/slurm_tools pull (GitHub has the user-neutral version since 2026-11-26)"
    else
        pass "template $TEMPLATE"
    fi
else
    fail "template $TEMPLATE missing" "a full slurm_tools clone at ~/slurm_tools"
fi
if [[ -x "$HOME/.venv/bin/slurm_job" ]]; then
    pass "$HOME/.venv/bin/slurm_job"
elif doing; then
    PY="${PYTHON:-}"
    if [[ -z "$PY" ]]; then
        for cand in python3.12 python3.11 python3.10 python3.9 python3; do
            if command -v "$cand" >/dev/null && "$cand" -c 'import sys; sys.exit(sys.version_info < (3, 9))'; then
                PY="$cand"; break
            fi
        done
    fi
    if [[ -z "$PY" ]]; then
        fail "no Python >= 3.9 for ~/.venv" "module load a Python module (module avail python), or set PYTHON=..., and rerun"
    elif { [[ -d "$HOME/.venv" ]] || "$PY" -m venv "$HOME/.venv"; } \
        && "$HOME/.venv/bin/pip" install -q -e "$HOME/slurm_tools"; then
        pass "$HOME/.venv with slurm_tools ($("$HOME/.venv/bin/python" --version 2>&1))"
    else
        fail "could not install slurm_tools into ~/.venv" "see the pip output above"
    fi
else
    fail "$HOME/.venv/bin/slurm_job missing" "rerun without --check"
fi

echo "== secrets"
SECRETS="$HOME/private/cc_secrets"
if [[ ! -f "$SECRETS" ]] && doing; then
    mkdir -p "$HOME/private" && chmod 700 "$HOME/private"
    printf 'WANDB_API_KEY=\n' > "$SECRETS" && chmod 600 "$SECRETS"
    echo "        created $SECRETS"
fi
if [[ -f "$SECRETS" ]]; then
    [[ "$(stat -c %a "$SECRETS")" == 600 ]] || warn "$SECRETS is readable by others" "chmod 600 $SECRETS"
    if grep -qE '^WANDB_API_KEY=.+' "$SECRETS"; then
        pass "$SECRETS has WANDB_API_KEY"
    else
        warn "$SECRETS has no WANDB_API_KEY" "add WANDB_API_KEY=<your key>; only runs with wandb_mode online need it"
    fi
else
    warn "$SECRETS missing (jobs run without W&B credentials)" "rerun without --check"
fi

echo "== live checkout models"
n="$(find "$HOME/algorithmic-institutions/artifacts" -type f 2>/dev/null | wc -l)"
if [[ "$n" -gt 0 ]]; then
    pass "$n files in ~/algorithmic-institutions/artifacts"
else
    warn "~/algorithmic-institutions/artifacts is empty" \
        "from the proxy: fetch the models, then POST /sync_to_cluster with includes [\"artifacts\"] (docs/handover/handover.md)"
fi

echo
if [[ "$FAILS" == 0 ]]; then echo "OK"; else echo "$FAILS check(s) failed"; fi
[[ "$FAILS" == 0 ]]
