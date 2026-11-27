"""Kill every process of this Slurm job step but the given ones and their
descendants, as the end of a job would (SIGKILL): the resume test of
campaign.slurm.sh, which keeps the model server running and then restarts the
campaign from its state files.

    python3 kill_job_children.py [--dry-run] <keep pid> [<keep pid> ...]

This process and its ancestors are kept too. Lists what it killed (with
--dry-run: what it would kill, and what it keeps).
"""

import os
import signal
import sys


def ppids():
    out = {}
    for name in os.listdir("/proc"):
        if name.isdigit():
            try:
                with open(f"/proc/{name}/stat") as f:
                    out[int(name)] = int(f.read().rsplit(")", 1)[1].split()[1])
            except (OSError, IndexError, ValueError):
                pass
    return out


def cmdline(pid):
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return f.read().replace(b"\0", b" ").decode(errors="replace")[:150]
    except OSError:
        return "?"


def main():
    args = sys.argv[1:]
    dry = "--dry-run" in args
    keep = {int(p) for p in args if p != "--dry-run"}
    with open("/proc/self/cgroup") as f:
        cgroup = next(line.split("::", 1)[1].strip() for line in f if line.startswith("0::"))
    with open(f"/sys/fs/cgroup{cgroup}/cgroup.procs") as f:
        procs = {int(p) for p in f.read().split()}
    parent = ppids()
    mine, pid = set(), os.getpid()
    while pid in parent:  # this process and its ancestors (not their children)
        mine.add(pid)
        pid = parent[pid]

    def kept(p):
        seen = set()
        while p in parent and p not in seen:
            if p in keep:
                return True
            seen.add(p)
            p = parent[p]
        return p in keep

    for p in sorted(procs):
        if dry:
            what = "keep" if p in mine or kept(p) else "kill"
            print(f"{what} {p} {cmdline(p)}", flush=True)
            continue
        if p not in mine and not kept(p):
            print(f"kill {p} {cmdline(p)}", flush=True)
            try:
                os.kill(p, signal.SIGKILL)
            except ProcessLookupError:
                pass


if __name__ == "__main__":
    main()
