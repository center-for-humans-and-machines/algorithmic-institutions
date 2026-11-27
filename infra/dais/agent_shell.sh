#!/bin/bash
# The shell of a policy-finder agent on DAIS (scripts/policy_finder/agent_apptainer.py):
# bound read-only into the agent's container at /opt/pf-agent/shell with
# noinet.py beside it, and set as $SHELL, which OpenCode's bash tool runs
# every command with (`$SHELL -c <command>`). Each command, and everything it
# starts, runs under noinet.py's seccomp filter: no socket but AF_UNIX, so no
# internet and no TCP to the model server either. OpenCode itself is not under
# the filter and keeps its connection to the model.
#
# The filter holds only while nothing the unfiltered side runs is writable by
# a filtered command: this file and noinet.py are read-only binds, every
# program is named by an absolute path in the read-only image (not looked up
# in PATH), and bash reads no start-up file (--noprofile --norc, no BASH_ENV
# or ENV).
exec /usr/bin/env -u BASH_ENV -u ENV /usr/bin/python3 /opt/pf-agent/noinet.py \
  /bin/bash --noprofile --norc "$@"
