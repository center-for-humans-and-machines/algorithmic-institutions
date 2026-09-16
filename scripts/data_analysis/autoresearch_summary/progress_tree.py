"""Render the campaign progress tree (summary doc section 5) as a PNG.

Node = experiment PR at x = chronological order, y = resulting stack mean
(lower is better). Edges run child -> parent; the two deep lineages get
distinct edge styles. Successes are colored by method category with a
distinct marker shape per category (identity is never color-alone);
failures are faint grey. The dotted step traces the best recorded stack
mean so far.

Static preview of the interactive report view. Usage:
    python scripts/data_analysis/autoresearch_summary/progress_tree.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

DATA = Path(__file__).parent / "data"
OUT = Path("plots/data_analysis/autoresearch_summary/progress_tree.png")

ROOT_MEAN = 1.7595567320354153  # top-ranked stack, 23_stack_sweep_updated

# Palette reference instance (dataviz skill), light mode. Only categories
# with at least one success carry a hue; hue + marker shape together so
# identity survives CVD (node not available to run the validator).
CATEGORY_STYLE = {
    "correlated-sampling": ("#2a78d6", "o"),
    "nonlinear-emission": ("#eb6834", "^"),
    "feature-engineering": ("#1baf7a", "s"),
    "structured-head": ("#4a3aa7", "D"),
    "architecture": ("#e87ba4", "v"),
}
FALLBACK_SUCCESS = ("#52514e", "P")  # a success outside the four (none today)
FAIL_GREY = "#c9c8c3"
INK = "#0b0b0b"
INK_2 = "#52514e"
SURFACE = "#fcfcfb"


def load():
    experiments = json.loads((DATA / "experiments.json").read_text())
    categories = {
        c["pr"]: c["category"]
        for c in json.loads((DATA / "categories.json").read_text())
    }
    return experiments, categories


def resolve_means(experiments):
    """y position per PR; a no-eval run sits at its parent's level."""
    by_pr = {e["pr"]: e for e in experiments}

    def mean_of(pr):
        e = by_pr[pr]
        m = e["metrics"].get("mean")
        if m is not None:
            return m
        return mean_of(e["parent_pr"]) if e["parent_pr"] else ROOT_MEAN

    return {e["pr"]: mean_of(e["pr"]) for e in experiments}


def lineage_root(experiments):
    """The main-rooted ancestor of every PR (itself if main-rooted)."""
    by_pr = {e["pr"]: e for e in experiments}

    def root(pr):
        parent = by_pr[pr]["parent_pr"]
        return pr if parent is None else root(parent)

    return {e["pr"]: root(e["pr"]) for e in experiments}


def main():
    experiments, categories = load()
    experiments.sort(key=lambda e: e["pr"])
    means = resolve_means(experiments)
    roots = lineage_root(experiments)
    xs = {e["pr"]: i + 1 for i, e in enumerate(experiments)}

    # The two deep trees: the gnn-stack punisher trunks and the gmlp trunk.
    gnn_tree = {pr for pr, r in roots.items() if r in (160, 161)}
    gmlp_tree = {pr for pr, r in roots.items() if r == 167}
    TREE_COLOR = {"gnn": "#6b6a66", "gmlp": "#b08968"}

    fig, ax = plt.subplots(figsize=(14, 7.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    # Edges (child -> parent). Solid = the success spine (traceable from
    # main to the last PR of each tree); dashed = a failed branch off it;
    # faint dots = one-off shots straight from main.
    for e in experiments:
        pr = e["pr"]
        px = xs[e["parent_pr"]] if e["parent_pr"] else 0
        py = means[e["parent_pr"]] if e["parent_pr"] else ROOT_MEAN
        in_tree = pr in gnn_tree or pr in gmlp_tree
        tree = TREE_COLOR["gnn"] if pr in gnn_tree else TREE_COLOR["gmlp"]
        if in_tree and e["verdict"] == "SUCCESS":
            style = dict(ls="-", lw=2.2, color=tree)
        elif in_tree:
            style = dict(ls="--", lw=1.1, color=tree, alpha=0.55)
        elif e["verdict"] == "SUCCESS":  # main one-off that succeeded
            style = dict(ls="-", lw=1.0, color="#c9c8c3")
        else:  # main one-off that failed
            style = dict(ls=":", lw=0.9, color="#d6d5d0")
        ax.plot([px, xs[pr]], [py, means[pr]], zorder=1, **style)

    # Frontier: best mean reached so far along the two lineages' spines
    # (successes only — a failed run's stack is never adopted).
    fx, fy, best = [0], [ROOT_MEAN], ROOT_MEAN
    for e in experiments:
        m = e["metrics"].get("mean")
        if (e["pr"] in gnn_tree or e["pr"] in gmlp_tree) \
                and e["verdict"] == "SUCCESS" and m is not None and m < best:
            best = m
        fx.append(xs[e["pr"]])
        fy.append(best)
    ax.step(fx, fy, where="post", color="#8f8e89", lw=1.0, ls=(0, (2, 3)),
            zorder=2)

    # Root.
    ax.scatter([0], [ROOT_MEAN], marker="D", s=70, color=INK, zorder=4)
    ax.annotate("main", (0, ROOT_MEAN), textcoords="offset points",
                xytext=(-4, 10), fontsize=9, color=INK, ha="right")

    # Nodes + PR labels.
    for e in experiments:
        pr, x, y = e["pr"], xs[e["pr"]], means[e["pr"]]
        no_eval = e["metrics"].get("mean") is None
        if e["verdict"] == "SUCCESS":
            color, marker = CATEGORY_STYLE.get(
                categories[pr], FALLBACK_SUCCESS
            )
            ax.scatter([x], [y], marker=marker, s=130, color=color,
                       edgecolors=SURFACE, linewidths=1.5, zorder=5)
        else:
            face = SURFACE if no_eval else FAIL_GREY
            ax.scatter([x], [y], marker="o", s=55, facecolors=face,
                       edgecolors=FAIL_GREY, linewidths=1.4, zorder=3)
        dy = 9 if e["verdict"] == "SUCCESS" else -13
        ax.annotate(str(pr), (x, y), textcoords="offset points",
                    xytext=(0, dy), fontsize=7.5, color=INK_2, ha="center")

    ax.set_xlabel("experiment order", color=INK_2)
    ax.set_ylabel("stack mean score (lower is better)", color=INK_2)
    ax.set_title(
        "Autoresearch campaign: every experiment PR, its parent, and the "
        "stack mean it left behind", color=INK, fontsize=12, pad=14,
    )
    ax.grid(axis="y", color="#eceae6", lw=0.7)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#d6d5d0")
    ax.tick_params(colors=INK_2, labelsize=8.5)

    legend = [
        *(
            Line2D([], [], marker=m, ls="", color=c, markersize=8, label=cat)
            for cat, (c, m) in CATEGORY_STYLE.items()
        ),
        Line2D([], [], marker="o", ls="", markerfacecolor=FAIL_GREY,
               markeredgecolor=FAIL_GREY, markersize=7, label="failed attempt"),
        Line2D([], [], marker="o", ls="", markerfacecolor=SURFACE,
               markeredgecolor=FAIL_GREY, markersize=7,
               label="failed, no evaluation"),
        Line2D([], [], ls="-", lw=2.2, color=TREE_COLOR["gnn"],
               label="gnn-stack tree (solid = success spine)"),
        Line2D([], [], ls="-", lw=2.2, color=TREE_COLOR["gmlp"],
               label="gaussian-MLP tree (solid = success spine)"),
        Line2D([], [], ls="--", lw=1.1, color="#8f8e89",
               label="failed branch off a tree"),
        Line2D([], [], ls=(0, (2, 3)), color="#8f8e89",
               label="best lineage mean so far"),
    ]
    ax.legend(handles=legend, loc="lower left", frameon=False, fontsize=8.5,
              labelcolor=INK_2, ncol=2)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
