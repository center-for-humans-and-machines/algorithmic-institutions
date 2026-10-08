"""Join the sweep.json of a sweep's parts into one (#227).

A sweep generated with `--n-parts N` runs as N sims, each writing its own
sweep.json under plots/simulation/policy_finder/<rule>_sweep_p<k>of<N>. This
writes the sweep.json of the whole design, `best` over all points, by
default to plots/simulation/policy_finder/<rule>_sweep/sweep.json, and copies
next to it the plots of that best point, drawn by the part that holds it.

Usage:
    python scripts/policy_finder/merge_sweeps.py <part sim dir> ... [--out <json>]
"""

import argparse
import json
import re
import sys
from pathlib import Path

from aimanager.simulation.sweep import SWEEP_FILE, copy_best_plots, merge_sweeps

PART = re.compile(r"^(?P<sweep>.+)_p(?P<k>\d+)of(?P<n>\d+)$")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("sim_dirs", nargs="+", help="the parts' sim output dirs")
    ap.add_argument("--out", help="the merged sweep.json")
    args = ap.parse_args()

    dirs = [Path(d) for d in args.sim_dirs]
    parts = [PART.match(d.name) for d in dirs]
    if all(parts):
        sweeps = {(m["sweep"], int(m["n"])) for m in parts}
        (sweep, n), *other = sweeps
        if other:
            sys.exit(f"the dirs are parts of different sweeps: {sorted(sweeps)}")
        found = sorted(int(m["k"]) for m in parts)
        if found != list(range(1, n + 1)):
            sys.exit(f"{sweep}: needs parts 1..{n}, got {found}")
        out = Path(args.out or dirs[0].parent / sweep / SWEEP_FILE)
    elif args.out:
        out = Path(args.out)
    else:
        sys.exit("the dirs are not named <sweep>_p<k>of<N>: pass --out")

    try:
        merged = merge_sweeps([json.loads((d / SWEEP_FILE).read_text()) for d in dirs])
    except ValueError as e:
        sys.exit(str(e))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(merged, indent=2))
    for path in copy_best_plots(dirs, merged["best_name"], out.parent):
        print(f"Copied: {path}")
    print(
        f"Wrote {out}: {len(merged['points'])} points from {len(dirs)} parts,"
        f" best {merged['best_name']} {merged['best']}"
    )


if __name__ == "__main__":
    main()
