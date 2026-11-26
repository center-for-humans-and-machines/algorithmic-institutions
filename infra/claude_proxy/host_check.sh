#!/usr/bin/env bash
# Check the host before starting the algoinst Claude proxy (docker compose).
#
# Usage: infra/claude_proxy/host_check.sh [--fix]
#
# Reads container/CC_env/.env and checks every host path docker-compose.yml
# mounts (compose refuses to start on a missing one), the secrets layout
# (secrets.example/README.md), Docker and Compose, and the uid the proxy runs
# as. --fix creates the folders that hold no secrets (state dirs, PF_RUNS).
# Exit code 1 on any FAIL. Inside the running proxy,
# scripts/policy_finder/preflight.py checks the services and Raven.
#
# Runs on Linux and on macOS (bash 3.2, BSD tools); on macOS it also checks
# what docs/handover/handover.md "On a Mac" covers.

set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
ENV_FILE="$HERE/container/CC_env/.env"
SPAWNER_CONFIG="$HERE/container/spawner_config/algoinst.yml"
FIX=0
[[ "${1:-}" == "--fix" ]] && FIX=1
FAILS=0
OS="$(uname -s)"

# ver_ge A B: version A >= version B (no sort -V on macOS)
ver_ge() {
    awk -v a="$1" -v b="$2" 'BEGIN {
        n = split(a, x, "."); m = split(b, y, ".")
        for (i = 1; i <= (n > m ? n : m); i++) {
            if (x[i] + 0 > y[i] + 0) exit 0
            if (x[i] + 0 < y[i] + 0) exit 1
        }
        exit 0 }'
}
# owner_uid PATH (GNU stat, else BSD stat)
owner_uid() { stat -c %u "$1" 2>/dev/null || stat -f %u "$1"; }

pass() { echo "[PASS] $1"; }
warn() { echo "[WARN] $1"; [[ -n "${2:-}" ]] && echo "         -> $2"; return 0; }
fail() { echo "[FAIL] $1"; [[ -n "${2:-}" ]] && echo "         -> $2"; FAILS=$((FAILS + 1)); }

# need <d|f|s> <path> <hint> [fixable]   (directory, file, socket)
need() {
    local kind="$1" path="$2" hint="$3" fixable="${4:-0}"
    if [[ "$kind" == d && -d "$path" ]] || [[ "$kind" == f && -f "$path" ]] \
        || [[ "$kind" == s && -S "$path" ]]; then
        pass "$path"
    elif [[ "$kind" == d && "$fixable" == 1 && "$FIX" == 1 ]]; then
        mkdir -p "$path" && pass "$path (created)"
    else
        fail "$path missing" "$hint"
    fi
}

echo "== tools"
if command -v docker >/dev/null; then
    pass "docker $(docker version --format '{{.Client.Version}}' 2>/dev/null)"
    if docker info >/dev/null 2>&1; then
        pass "Docker daemon reachable"
    else
        fail "cannot talk to the Docker daemon" "start Docker (Docker Desktop on a Mac); on Linux add your user to the docker group"
    fi
    v="$(docker compose version --short 2>/dev/null | sed 's/^v//')"
    if [[ -z "$v" ]]; then
        fail "docker compose v2 missing" "install the Docker Compose plugin"
    elif ver_ge "$v" 2.24.0; then
        pass "docker compose $v"
    else
        fail "docker compose $v is older than 2.24 (optional env_file)" "update the Docker Compose plugin"
    fi
else
    fail "docker not found" "install Docker Engine and the Compose plugin"
fi
need s /var/run/docker.sock "the spawner mounts the Docker socket"

echo "== settings ($ENV_FILE)"
if [[ -f "$ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
    pass ".env"
else
    fail ".env missing" "cp $HERE/container/CC_env/.env.example $ENV_FILE and fill it in"
fi
if [[ -n "${RAVEN_USER:-}" ]]; then pass "RAVEN_USER=$RAVEN_USER"; else fail "RAVEN_USER not set" "set it in $ENV_FILE"; fi
if [[ -n "${GIT_USER_EMAIL:-}" ]]; then pass "GIT_USER_EMAIL=$GIT_USER_EMAIL"; else warn "GIT_USER_EMAIL not set" "the pipeline commits on instance branches; set it in $ENV_FILE"; fi

HOST_UID="${HOST_UID:-1000}"
HOST_GID="${HOST_GID:-1000}"
if [[ "$HOST_UID:$HOST_GID" == "$(id -u):$(id -g)" ]]; then
    pass "HOST_UID:HOST_GID=$HOST_UID:$HOST_GID matches $(id -un)"
else
    fail "HOST_UID:HOST_GID=$HOST_UID:$HOST_GID but you are $(id -u):$(id -g)" "set HOST_UID and HOST_GID in $ENV_FILE"
fi
if grep -q "user: \"$HOST_UID:$HOST_GID\"" "$SPAWNER_CONFIG"; then
    pass "spawner profile runs agents as $HOST_UID:$HOST_GID"
else
    fail "spawner_config/algoinst.yml does not run agents as $HOST_UID:$HOST_GID" "edit user: in $SPAWNER_CONFIG"
fi

CLAUDE_TOOLS="${CLAUDE_TOOLS:-$HOME/claude_tools}"
SECRETS_DIR="${SECRETS_DIR:-$HOME/secrets}"
CC_STATE_DIR="${CC_STATE_DIR:-$HOME/tmp}"
PF_RUNS="${PF_RUNS:-$HOME/pf_runs}"
for var in CLAUDE_TOOLS SECRETS_DIR CC_STATE_DIR PF_RUNS; do
    [[ "${!var}" == /* ]] || fail "$var=${!var} is not an absolute path" "use an absolute path in $ENV_FILE"
done

echo "== Claude login (the proxy's own)"
need d "$HOME/.claude" "run claude once on this host and log in"
need f "$HOME/.claude.json" "run claude once on this host and log in"

echo "== tool repos ($CLAUDE_TOOLS)"
need f "$CLAUDE_TOOLS/CC_command_server/Dockerfile" \
    "git clone https://github.com/yannikkellerde/CC_command_server.git $CLAUDE_TOOLS/CC_command_server"
need f "$CLAUDE_TOOLS/container_spawner/Dockerfile" \
    "git clone https://github.com/yannikkellerde/container_spawner.git $CLAUDE_TOOLS/container_spawner"
need d "$CLAUDE_TOOLS/container_spawner/container_spawner" "a full clone of container_spawner"
need d "$CLAUDE_TOOLS/container_spawner/proxy" "a full clone of container_spawner"
# The command server reads its Raven account from env.yml beside its code
# (CC_command_server/env.yml.example); without it, every path is ykeller's.
CC_ENV_YML="$CLAUDE_TOOLS/CC_command_server/env.yml"
if [[ ! -f "$CLAUDE_TOOLS/CC_command_server/env_config.py" ]]; then
    fail "CC_command_server is too old (no env.yml support)" "git -C $CLAUDE_TOOLS/CC_command_server pull"
elif [[ -f "$CC_ENV_YML" ]]; then
    cc_user="$(sed -nE 's/^raven_user:[[:space:]]*["'"'"']?([^"'"'"'[:space:]#]+).*/\1/p' "$CC_ENV_YML")"
    if [[ -n "${RAVEN_USER:-}" && "$cc_user" == "$RAVEN_USER" ]]; then
        pass "$CC_ENV_YML: raven_user $cc_user"
    else
        fail "$CC_ENV_YML has raven_user '$cc_user', .env has RAVEN_USER '${RAVEN_USER:-}'" "make them the same"
    fi
elif [[ "${RAVEN_USER:-}" == ykeller ]]; then
    pass "no $CC_ENV_YML: the command server's defaults are ykeller's"
else
    fail "$CC_ENV_YML missing: the command server would use ykeller's Raven paths" \
        "cp $CLAUDE_TOOLS/CC_command_server/env.yml.example $CC_ENV_YML; set dais_user and raven_user to ${RAVEN_USER:-your MPCDF login}"
fi

echo "== secrets ($SECRETS_DIR, layout: infra/claude_proxy/secrets.example/README.md)"
if [[ -s "$SECRETS_DIR/claude_setup_token" ]]; then
    pass "$SECRETS_DIR/claude_setup_token"
elif [[ -e "$SECRETS_DIR/claude_setup_token" ]]; then
    fail "$SECRETS_DIR/claude_setup_token is empty" "claude setup-token, paste the token"
else
    fail "$SECRETS_DIR/claude_setup_token missing" "claude setup-token, paste the token as the only line"
fi
need d "$SECRETS_DIR/cc_ssh_keys" "an SSH key for Raven and config (secrets.example/cc_ssh_keys/config.example)"
if [[ -e "$SECRETS_DIR/cc_ssh_keys/config" ]]; then
    if [[ -r "$SECRETS_DIR/cc_ssh_keys/config" ]]; then
        grep -qiE '^[[:space:]]*(Host|Match)[[:space:]].*raven' "$SECRETS_DIR/cc_ssh_keys/config" \
            && pass "cc_ssh_keys/config covers raven" \
            || fail "cc_ssh_keys/config has no Host or Match line for raven" "see secrets.example/cc_ssh_keys/config.example"
        grep -q "<raven user>" "$SECRETS_DIR/cc_ssh_keys/config" \
            && fail "cc_ssh_keys/config still has <raven user>" "replace it with your MPCDF login"
        # Docker Desktop (macOS) maps file owners itself; this applies to Linux
        [[ "$OS" == Darwin || "$(owner_uid "$SECRETS_DIR/cc_ssh_keys/config")" == 0 ]] \
            || warn "cc_ssh_keys is not owned by root: if ssh in the command server says 'Bad owner or permissions', fix it with" \
                "sudo chown -R root:root $SECRETS_DIR/cc_ssh_keys; sudo chmod 700 $SECRETS_DIR/cc_ssh_keys; sudo chmod 600 $SECRETS_DIR/cc_ssh_keys/*"
    else
        pass "cc_ssh_keys/config (not readable by you: root-owned, as intended)"
    fi
elif [[ -d "$SECRETS_DIR/cc_ssh_keys" && ! -r "$SECRETS_DIR/cc_ssh_keys" ]]; then
    pass "cc_ssh_keys (root-owned; contents not checked)"
else
    fail "$SECRETS_DIR/cc_ssh_keys/config missing" "see secrets.example/cc_ssh_keys/config.example"
fi
need f "$SECRETS_DIR/S3/.s3cfg" "s3cmd config with CHM S3 credentials"
need d "$SECRETS_DIR/algoinst_deploy_key" "GitHub key for git push from the proxy; an empty folder works" 1
[[ -f "$SECRETS_DIR/.cc_env" ]] && pass "$SECRETS_DIR/.cc_env (optional)"

echo "== state ($CC_STATE_DIR, $PF_RUNS)"
for d in cc_logs cluster_transfer container_spawner; do
    need d "$CC_STATE_DIR/$d" "rerun with --fix" 1
done
need d "$PF_RUNS" "rerun with --fix" 1
need d "$PF_RUNS/exports" "rerun with --fix" 1

echo "== repo ($REPO)"
branch="$(git -C "$REPO" branch --show-current 2>/dev/null)"
[[ "$branch" == main ]] && pass "on main" || warn "checkout is on '$branch', not main" "the policy finder bases instances on main (PF_BASE overrides)"
if [[ -f "$REPO/experiments/2group_8agent_50ep.csv" ]]; then
    pass "agent data present"
else
    warn "data files not fetched yet" "inside the proxy, scripts/policy_finder/preflight.py lists them with the fetch command"
fi

echo "== images"
if command -v docker >/dev/null && docker info >/dev/null 2>&1; then
    for img in algoinst_claude:latest cc-cmd-server container-spawner; do
        if docker image inspect "$img" >/dev/null 2>&1; then
            pass "image $img"
        else
            warn "image $img not built yet" "start_command.md builds it (the spawner copies algoinst_claude:latest for the agents)"
        fi
    done
fi

if [[ "$OS" == Darwin ]]; then
    echo "== macOS (docs/handover/handover.md, On a Mac)"
    # The Mac's Claude Code keeps its login in the Keychain, which the proxy
    # container cannot read: it needs a token in .cc_env.
    if grep -qE '^CLAUDE_CODE_OAUTH_TOKEN=.+' "$SECRETS_DIR/.cc_env" 2>/dev/null; then
        pass "CLAUDE_CODE_OAUTH_TOKEN in $SECRETS_DIR/.cc_env"
    else
        fail "no CLAUDE_CODE_OAUTH_TOKEN in $SECRETS_DIR/.cc_env: the proxy would start logged out" \
            "claude setup-token, then add CLAUDE_CODE_OAUTH_TOKEN=<token> to $SECRETS_DIR/.cc_env"
    fi
    # Docker Desktop shares /Users, /Volumes, /private and /tmp by default
    for var in REPO CLAUDE_TOOLS SECRETS_DIR CC_STATE_DIR PF_RUNS; do
        case "${!var}" in
            /Users/*|/Volumes/*|/private/*|/tmp/*) ;;
            *) warn "$var=${!var} is outside Docker Desktop's default file sharing" \
                   "add it under Docker Desktop > Settings > Resources > File sharing" ;;
        esac
    done
    mem="$(docker info --format '{{.MemTotal}}' 2>/dev/null)"
    if [[ -n "$mem" && "$mem" -lt 17000000000 ]]; then
        warn "Docker Desktop has $((mem / 1000000000)) GB: up to 4 agents take 4 GB each" \
            "raise Docker Desktop's memory, or add PF_AGENT_SLOTS=2 to $SECRETS_DIR/.cc_env"
    fi
    if [[ "$(uname -m)" == arm64 ]]; then
        arch="$(docker image inspect -f '{{.Architecture}}' algoinst_claude:latest 2>/dev/null)"
        if [[ "$arch" == amd64 ]]; then
            pass "algoinst_claude:latest is amd64 (PyG has no Linux ARM wheels)"
        elif [[ -n "$arch" ]]; then
            fail "algoinst_claude:latest is $arch: PyG has no Linux ARM wheels" \
                "rebuild with -f container/CC_env/docker-compose.amd64.yml (start_command.md)"
        else
            warn "Apple Silicon: build the proxy for amd64" \
                "add -f container/CC_env/docker-compose.amd64.yml to the compose commands (start_command.md)"
        fi
    fi
fi

echo
if [[ "$FAILS" == 0 ]]; then echo "OK"; else echo "$FAILS check(s) failed"; fi
[[ "$FAILS" == 0 ]]
