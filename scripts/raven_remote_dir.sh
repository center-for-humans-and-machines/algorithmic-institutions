#!/usr/bin/env bash
#
# Which Raven dir the cluster scripts sync to, run in and fetch from.
# Sourced by simulate_cluster.sh, remote_test.sh and fetch_cluster.sh.
#
#   raven_remote_dir <local project dir> <shared remote dir>
#
# prints, in order of precedence:
#   1. $AI_REMOTE_DIR, when set
#   2. the template in <local project dir>/.raven_remote_dir, with {branch}
#      the current branch and `/` in it as `--` (a branch that carries the
#      file, like policy-finder-base and its branches, never touches the
#      shared checkout: one isolated dir per branch)
#   3. the shared remote dir
#
# Fails on a detached HEAD in case 2: set AI_REMOTE_DIR instead.

raven_remote_dir() {
    local root="$1" shared="$2"
    if [[ -n "${AI_REMOTE_DIR:-}" ]]; then
        echo "${AI_REMOTE_DIR}"
        return
    fi
    local file="${root}/.raven_remote_dir"
    if [[ -f "${file}" ]]; then
        local branch template
        if ! branch="$(git -C "${root}" symbolic-ref --quiet --short HEAD)"; then
            echo "raven_remote_dir: detached HEAD in ${root}; set AI_REMOTE_DIR" >&2
            return 1
        fi
        template="$(head -n 1 "${file}")"
        echo "${template//\{branch\}/${branch//\//--}}"
        return
    fi
    echo "${shared}"
}
