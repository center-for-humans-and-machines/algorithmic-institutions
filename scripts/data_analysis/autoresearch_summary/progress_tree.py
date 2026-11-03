"""Layout helpers and node styling for the report's progress-tree layer.

Node = experiment PR at x = chronological order, y = resulting stack mean
(lower is better); a run with no evaluation sits at its parent's level.
The rendering itself lives in build_report.tree_svg; this module holds
the pieces it computes positions and styles from.
"""

import json
from pathlib import Path

DATA = Path(__file__).parent / "data"

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
FALLBACK_SUCCESS = ("#52514e", "P")  # a success outside the five (none today)
FAIL_GREY = "#c9c8c3"
INK = "#0b0b0b"
INK_2 = "#52514e"
SURFACE = "#fcfcfb"


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
