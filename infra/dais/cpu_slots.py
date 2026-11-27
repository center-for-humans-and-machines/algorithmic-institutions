"""This job's CPUs grouped by physical core, for pinning with `taskset -c`.

    python3 cpu_slots.py reserved R      the CPUs of the first R cores, as one list
    python3 cpu_slots.py slots R P [N]   after the first R cores, one line per slot
                                         of P cores (at most N slots)

A core's CPUs are its hyperthread siblings that this process may run on.
"""

import os
import sys


def siblings(cpu):
    path = f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list"
    out = set()
    with open(path) as f:
        for part in f.read().strip().split(","):
            lo, _, hi = part.partition("-")
            out.update(range(int(lo), int(hi or lo) + 1))
    return out


def cores():
    allowed = os.sched_getaffinity(0)
    seen, out = set(), []
    for cpu in sorted(allowed):
        if cpu not in seen:
            group = sorted(siblings(cpu) & allowed)
            seen.update(group)
            out.append(group)
    return out


def fmt(groups):
    return ",".join(str(c) for g in groups for c in g)


def main():
    cmd, reserve = sys.argv[1], int(sys.argv[2])
    cs = cores()
    if cmd == "reserved":
        print(fmt(cs[:reserve]))
        return
    per = int(sys.argv[3])
    limit = int(sys.argv[4]) if len(sys.argv) > 4 else None
    rest = cs[reserve:]
    slots = [rest[i : i + per] for i in range(0, len(rest) - per + 1, per)]
    for slot in slots[:limit]:
        print(fmt(slot))


if __name__ == "__main__":
    main()
