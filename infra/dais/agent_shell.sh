#!/bin/bash
# The shell of a policy-finder agent on DAIS (scripts/policy_finder/agent_apptainer.py):
# bound read-only into the agent's container at /opt/pf-agent/shell with
# noinet.py beside it, and set as $SHELL, which OpenCode's bash tool runs
# every command with (`$SHELL -c <command>`). Each command, and everything it
# starts, runs under noinet.py's seccomp filter: no socket but AF_UNIX, so no
# internet and no TCP to the model server either. OpenCode itself is not under
# the filter and keeps its connection to the model.
#
# Until the filter is installed, nothing a command could have written runs:
# this file and noinet.py are read-only binds; every program is named by an
# absolute path in the read-only image (not looked up in PATH); python3 runs
# isolated (-I: no user site-packages, so no .pth file in the writable home,
# no PYTHON* variables, not the script's folder) and without site (-S), and
# noinet.py needs only the standard library; bash reads no start-up file
# (--noprofile --norc, no BASH_ENV or ENV).
#
# What stays writable for a command is the run's home, /tmp and the write
# folders. OpenCode (unfiltered) also uses the home (~/.local/share/opencode,
# ~/.cache/opencode) and /tmp (where its runtime unpacks native libraries); a
# process OpenCode starts after a command ran (the runner's `opencode export`)
# could load what a command left there. That is outside this barrier, which is
# against ordinary network use by the agent's commands, not a sandbox.
exec /usr/bin/env -u BASH_ENV -u ENV /usr/bin/python3 -I -S /opt/pf-agent/noinet.py \
  /bin/bash --noprofile --norc "$@"
