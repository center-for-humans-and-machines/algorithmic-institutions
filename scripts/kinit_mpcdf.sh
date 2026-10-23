#!/usr/bin/env bash
# Renew the MPCDF Kerberos ticket without an interactive prompt.
#
# Long cluster runs outlive a Kerberos ticket. This renews it from a password
# held in a .env file, without an interactive prompt.
#
# WHAT THIS DOES NOT DO. A valid ticket does not by itself open an SSH
# connection to Raven. Measured 2026-09-22: gssapi-with-mic authenticates
# against the MPCDF gateway with "partial success" and the gateway then
# requires a password or keyboard-interactive second factor, which no stored
# credential can answer and which a public key is not offered in place of.
# Access is carried by the SSH master connection, which persists 24 h and can
# only be opened by a human running `ssh raven`. When that master dies, every
# ssh, rsync and squeue fails regardless of the ticket. `--check` reports both
# so the two failures are never confused.
#
#   scripts/kinit_mpcdf.sh            # renew only if the ticket is missing/expired
#   scripts/kinit_mpcdf.sh --force    # renew regardless
#   scripts/kinit_mpcdf.sh --check    # report state, change nothing, exit 1 if invalid
#
# Configuration, all optional:
#   MPCDF_ENV_FILE   path to the .env holding MPCDF_PW (else the search list below)
#   MPCDF_PRINCIPAL  default levinb@IPP-GARCHING.MPG.DE
#   MPCDF_PW         if already exported, used directly and no file is read
#
# The password is piped to kinit on stdin. It is never passed as an argument,
# so it does not appear in `ps`, and it is never written to a temporary file.
#
# Better options than a plaintext password, if they become available:
#   * a Kerberos keytab, which is the purpose-built mechanism and stores no
#     password at all -- ask MPCDF whether user keytabs are issued;
#   * `kinit --keychain`, which stores the password in the macOS Keychain and
#     puts it behind the login password rather than in a readable file.

set -euo pipefail

PRINCIPAL="${MPCDF_PRINCIPAL:-levinb@IPP-GARCHING.MPG.DE}"

# Measured against the MPCDF KDC on 2026-09-22: it caps a ticket at 10 hours
# however much you ask for, but it grants a renewable window of 30 days. So a
# ticket obtained with --renewable-life can be extended every 10 hours with
# `kinit -R` and no password at all, for a month. Asking for the window is
# what makes the password a monthly event rather than a daily one.
LIFETIME="${MPCDF_LIFETIME:-10h}"
RENEWABLE="${MPCDF_RENEWABLE:-30d}"

ENV_CANDIDATES=(
    "${MPCDF_ENV_FILE:-}"
    "$HOME/.config/mpcdf/env"
    "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/.env"
    "$HOME/repros/llm-strategic-tuning/.env"
    "$HOME/.env"
)

log()  { printf '%s\n' "$*" >&2; }
die()  { printf 'error: %s\n' "$*" >&2; exit 1; }

ticket_valid() {
    # Heimdal klist -t tests for a valid ticket and sets the exit status.
    # Fall back to parsing if -t is unavailable.
    if klist -t >/dev/null 2>&1; then
        return 0
    elif klist 2>/dev/null | grep -q '>>>Expired<<<'; then
        return 1
    elif klist >/dev/null 2>&1; then
        return 0
    fi
    return 1
}

# A valid ticket is necessary but not sufficient. Measured 2026-09-22: the
# MPCDF gateways enforce multi-factor, so gssapi-with-mic authenticates with
# "partial success" and the gate then demands a password or keyboard-
# interactive response. No stored credential can answer that, and a public key
# is not offered as an alternative. What actually carries access is the SSH
# master connection, which persists 24 h and can only be opened interactively.
# So this reports both, and says which one needs a human.
master_alive() {
    ssh -O check raven >/dev/null 2>&1
}

report_master() {
    if master_alive; then
        log "ssh master to raven is up"
        return 0
    fi
    log "ssh master to raven is DOWN. A ticket alone will not reconnect,"
    log "because the gateway enforces a second factor. Open one yourself:"
    log "    ssh raven          # in a terminal, persists 24 h"
    return 1
}

report() {
    if ticket_valid; then
        local exp
        # klist row: "Sep 22 11:28:30 2026  Sep 22 21:28:07 2026  krbtgt/..."
        # fields 1-4 are the issue time, 5-8 the expiry.
        exp=$(klist 2>/dev/null | awk '/krbtgt/ {print $5, $6, $7, $8; exit}')
        log "ticket valid for ${PRINCIPAL}${exp:+, expires $exp}"
        return 0
    fi
    log "no valid ticket for ${PRINCIPAL}"
    return 1
}

# Read one key out of a .env without sourcing it. Sourcing would execute
# whatever else the file contains; this only ever reads the one assignment.
read_pw_from_file() {
    local file="$1" line
    line=$(grep -m1 -E '^[[:space:]]*(export[[:space:]]+)?MPCDF_PW=' "$file" 2>/dev/null) || return 1
    line="${line#*MPCDF_PW=}"
    # strip one layer of matching quotes and any trailing carriage return
    line="${line%$'\r'}"
    case "$line" in
        \"*\") line="${line#\"}"; line="${line%\"}" ;;
        \'*\') line="${line#\'}"; line="${line%\'}" ;;
    esac
    [ -n "$line" ] || return 1
    printf '%s' "$line"
}

warn_if_readable() {
    local file="$1" mode
    mode=$(stat -f '%OLp' "$file" 2>/dev/null || stat -c '%a' "$file" 2>/dev/null) || return 0
    # A password file should give group and other nothing at all. Anything
    # other than 00 in the last two digits is readable by someone else,
    # including plain read (4), which an earlier version of this check missed.
    case "$mode" in
        *00) return 0 ;;
    esac
    log "warning: $file is mode $mode and holds a password. Fix with:"
    log "    chmod 600 \"$file\""
}

MODE="auto"
case "${1:-}" in
    --force) MODE="force" ;;
    --check) MODE="check" ;;
    "")      ;;
    *)       die "unknown argument: $1 (expected --force, --check, or nothing)" ;;
esac

if [ "$MODE" = "check" ]; then
    t=0; report || t=1
    m=0; report_master || m=1
    exit $(( t || m ))
fi

if [ "$MODE" = "auto" ] && ticket_valid; then
    report
    exit 0
fi

command -v kinit >/dev/null 2>&1 || die "kinit not found on PATH"

# A renewable ticket can be extended without any password at all. Cheapest
# path, and it fails harmlessly when the ticket is already fully expired.
if kinit -R >/dev/null 2>&1 && ticket_valid; then
    log "renewed the existing ticket without a password"
    report
    exit 0
fi

PW=""
SOURCE=""
if [ -n "${MPCDF_PW:-}" ]; then
    PW="$MPCDF_PW"
    SOURCE="the exported MPCDF_PW"
else
    for f in "${ENV_CANDIDATES[@]}"; do
        [ -n "$f" ] && [ -f "$f" ] || continue
        if PW=$(read_pw_from_file "$f"); then
            SOURCE="$f"
            warn_if_readable "$f"
            break
        fi
    done
fi

if [ -z "$PW" ]; then
    die "no MPCDF_PW found. Set MPCDF_ENV_FILE, or add MPCDF_PW= to one of:
  ${ENV_CANDIDATES[*]}"
fi

log "requesting a ticket for ${PRINCIPAL} (password from ${SOURCE})"

# --password-file=STDIN keeps the password off the command line and out of ps.
if printf '%s\n' "$PW" | kinit --password-file=STDIN \
        --lifetime="$LIFETIME" --renewable-life="$RENEWABLE" "$PRINCIPAL"; then
    PW=""
    report
else
    PW=""
    die "kinit failed for ${PRINCIPAL}"
fi
