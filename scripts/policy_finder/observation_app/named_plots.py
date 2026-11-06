"""Redraw an exp2_multi campaign's final-best plots with the opponents' names.

A sweep's plots (simulation/sweep.py, BEST_PLOTS) label the opponents by their
opaque keys (opponent_1..4), since the agents read them. This redraws them for
the observation page only, from each final run's per_round.parquet, with the
keys replaced by the names in configs/simulation/opponents/exp2_multi.yml. The
plots go to <worktree root>/named_plots/<version>/, outside every worktree, and
the Sweep tab shows them in place of the originals (dashboard.sweep_blocks).
They show the final best (sweep.json's best_name), the point the tab's tiles
report.

    PF_SETTING=exp2_multi PF_WORKTREE_ROOT=<root> \\
        python scripts/policy_finder/observation_app/named_plots.py [--jobs 2]
"""
import argparse
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dashboard as d  # noqa: E402

#: The two-group plots of d.BEST_PLOTS that name the opponents
PLOTS = ("comparison_pairing_side.jpg", "group_size_evolution_global.jpg")


def named(text: str) -> str:
    return re.sub(r"opponent_\d+", lambda m: d.OPPONENTS.get(m.group(0), m.group(0)), text)


def redraw(final: Path, out: Path) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import pandas as pd
    from aimanager.simulation.sim_plots import plot_group_size, plot_pairing_side, prepare

    sweep = json.loads((final / "sweep.json").read_text())
    best = sweep["best_name"]
    point = next(p for p in sweep["points"] if p["name"] == best) if isinstance(
        sweep["points"], list) else None
    params = sweep["best"] if point is None else point["params"]
    title = f"{best}: " + ", ".join(f"{k}={v:g}" for k, v in params.items())
    others = [d.OPPONENTS.get(o, o) for o in sweep["opponents"]]
    pairings = [{"name": f"{best}_vs_{o}", "group_0": best, "group_1": o} for o in others]
    pairings += [{"name": f"{o}_vs_{best}", "group_0": o, "group_1": best} for o in others]
    keys = sweep["opponents"]
    wanted = [f"ah group_switching managed by {a}_vs_{b}"
              for o in keys for a, b in ((best, o), (o, best))]
    # only the best point's runs: the whole final run is 9 million rows
    df = pd.read_parquet(final / "per_round.parquet", filters=[("run", "in", wanted)])
    if df.empty:
        raise ValueError(f"{final}: no runs named like {wanted[0]}")
    runs = {r: named(r) for r in df["run"].unique()}
    df = prepare(df.assign(run=df["run"].map(runs)).reset_index(drop=True))
    out.mkdir(parents=True, exist_ok=True)
    plot_pairing_side(df, pairings, str(out), title)
    plot_group_size(df, str(out), title)
    return str(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--force", action="store_true", help="redraw existing plots too")
    args = ap.parse_args()
    if not d.OPPONENTS:
        sys.exit("needs PF_SETTING=exp2_multi")
    todo = []
    for wt in sorted(p for p in d.WT_ROOT.iterdir() if (p / ".git").exists()):
        for final in sorted(wt.glob(d.SWEEP_DIR.format(name="*") + "/" + d.FINAL)):
            if not (final / "per_round.parquet").exists():
                continue
            name = final.parent.name.removesuffix("_sweep")
            out = d.NAMED_PLOTS / name
            if args.force or not all((out / p).exists() for p in PLOTS):
                todo.append((final, out))
    print(len(todo), "to draw")
    with ProcessPoolExecutor(args.jobs) as ex:
        for done in ex.map(redraw, *zip(*todo)) if todo else []:
            print("drew", done, flush=True)


if __name__ == "__main__":
    main()
