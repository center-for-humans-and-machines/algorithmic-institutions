"""Calibrate `copula_rho` for the GNN switch predictor's herding copula.

`--raw-stats` measures the raw within-(episode, decision-round, group)
co-switching of the human train split -- the feasibility gate that decides
whether a shared-latent sampler has anything to capture. The conditional
pairwise-likelihood MLE against the GNN's own predicted marginals, and the
pre-flight, arrive with plan steps 5 and 9.

Method details, conventions, and the estimator history:
notes/autoresearch_log/switch-herding-copula.md (appendix).

Runs locally (pandas/numpy/scipy, no PyG):
    uv run python scripts/baselines/switch_copula_rho.py --raw-stats
"""

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("DISABLE_PANDERA_IMPORT_WARNING", "True")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

TRAIN = ROOT / "experiments/baseline/2group_8agent_50ep_bline_train.csv"
MASK = "switch_valid"
TARGET = "does_switch"
SWITCH_EVERY = 4  # decision cadence -> sets switch_mask / switch_valid
EXPERIMENTS = ("ah_group_switching",)
N_QUAD = 32  # Gauss-Legendre nodes for Phi_2
RHO_MAX = 0.95  # quadrature stays accurate well inside |rho| = 1
P_TINY = 1e-300  # log floor for a cell probability
MIN_PAIRS = 1500  # gate: power
MIN_RHO = 0.10  # gate: signal

# planner's read-only feasibility numbers (note 5), printed for eyeball check
EXPECTED = dict(
    eligible=1515,
    rate=0.29372937,
    per_round={3: 0.4290, 7: 0.3026, 11: 0.2601, 15: 0.2292, 19: 0.2434},
    cells=374,
    cells_ge2=330,
    hist={1: 44, 2: 46, 3: 51, 4: 95, 5: 47, 6: 41, 7: 35, 8: 15},
    pairs=3009,
    n11=341,
    n11_exp=215.6,
    odds_ratio=2.748,
    phi=0.2125,
    rho=0.3554,
)

from scipy.special import ndtr, ndtri  # noqa: E402

GL_X, GL_W = np.polynomial.legendre.leggauss(N_QUAD)


def f(x):
    """Unrounded float for the log."""
    return repr(float(x))


def line(label, value, expected=None, note=""):
    """One aligned `label  value  EXPECTED e` row."""
    exp = "" if expected is None else f"EXPECTED {expected}"
    print(f"{label:<30} {str(value):<23} {exp}{note}".rstrip())


# --------------------------------------------------------------------------- #
# data: the repo's own parsing, keeping the (episode, round, group) index
# --------------------------------------------------------------------------- #
def load_rows(path=TRAIN, switch_every=SWITCH_EVERY):
    """Every `switch_valid` decision row of `path`, with its cell index.

    Parsed by `parse_agent_rounds` -- the same labelling the switch models
    train on. The cell is (episode, decision round, PRE-switch group), which
    is the simulation's `state["agent_group"]` at that round (log note D7).
    """
    from aimanager.generic.data import parse_agent_rounds

    df = pd.read_csv(path)
    df = df[df["experiment_name"].isin(EXPERIMENTS)]
    df = df[~df["global_group_id"].str.contains("(flipped)", regex=False)]
    n_games = df["global_group_id"].nunique()
    parsed = parse_agent_rounds(df.copy(), switch_every=switch_every)
    rows = parsed[parsed[MASK]].copy()
    keys = ["group_idx", "round_number", "agent_group"]
    cell = rows.groupby(keys, sort=True).ngroup().to_numpy()
    return dict(
        y=rows[TARGET].to_numpy().astype(np.int64),
        cell=cell.astype(np.int64),
        episode=rows["group_idx"].to_numpy().astype(np.int64),
        round=rows["round_number"].to_numpy().astype(np.int64),
        group=rows["agent_group"].to_numpy().astype(np.int64),
        player=rows["player_idx"].to_numpy().astype(np.int64),
        n_games=int(n_games),
    )


# --------------------------------------------------------------------------- #
# cells and within-cell pairs
# --------------------------------------------------------------------------- #
def blocks(cell):
    """Sort rows by cell; return (order, starts, sizes) with each cell a
    contiguous block (what np.add.reduceat needs)."""
    order = np.argsort(cell, kind="stable")
    k = cell[order]
    starts = np.flatnonzero(np.r_[True, k[1:] != k[:-1]])
    sizes = np.diff(np.r_[starts, len(k)])
    return order, starts, sizes


def pair_index(cell):
    """Row indices (i, j), i < j, of every within-cell pair."""
    order, starts, sizes = blocks(cell)
    ii, jj = [], []
    for s, n in zip(starts, sizes):
        if n < 2:
            continue
        idx = order[s : s + n]
        a, b = np.triu_indices(n, k=1)
        ii.append(idx[a])
        jj.append(idx[b])
    if not ii:
        return np.array([], np.int64), np.array([], np.int64)
    return np.concatenate(ii), np.concatenate(jj)


# --------------------------------------------------------------------------- #
# bivariate normal CDF (Drezner-Wesolowsky, Gauss-Legendre, vectorised) --
# shared with the pairwise MLE of plan step 5
# --------------------------------------------------------------------------- #
def bvn_cdf(h, k, rho):
    """Phi_2(h, k; rho) elementwise over the arrays h, k for a scalar rho."""
    base = ndtr(h) * ndtr(k)
    if rho == 0.0:
        return base
    r = 0.5 * rho * (GL_X + 1.0)  # [Q] nodes on (0, rho)
    om = 1.0 - r**2
    hh, kk = h[:, None], k[:, None]
    dens = np.exp(-(hh**2 - 2.0 * r * hh * kk + kk**2) / (2.0 * om)) / np.sqrt(om)
    return base + (0.5 * rho / (2.0 * np.pi)) * (dens @ GL_W)


def check_bvn(n=400, seed=42):
    """Max abs deviation of bvn_cdf from scipy's mvn CDF on random points."""
    try:
        from scipy.stats import multivariate_normal as mvn
    except ImportError:  # pragma: no cover
        return None
    rng = np.random.default_rng(seed)
    h = rng.uniform(-4.0, 4.0, n)
    k = rng.uniform(-4.0, 4.0, n)
    worst = 0.0
    for rho in (0.05, 0.2, 0.5, 0.8, 0.9):
        mine = bvn_cdf(h, k, rho)
        cov = [[1.0, rho], [rho, 1.0]]
        ref = np.array(
            [mvn.cdf([hi, ki], mean=[0.0, 0.0], cov=cov) for hi, ki in zip(h, k)]
        )
        worst = max(worst, float(np.abs(mine - ref).max()))
    return worst


# --------------------------------------------------------------------------- #
# the pooled 2x2 pair table and its unconditional tetrachoric rho
# --------------------------------------------------------------------------- #
def pair_table(y, ii, jj):
    """Symmetrised 2x2 co-switching table over the within-cell pairs.

    A within-cell pair is unordered, so each pair enters the table in both
    orderings and the table is halved: n10 = n01 = (n10_raw + n01_raw) / 2.
    That is what makes the margins exchangeable -- the convention the
    reference numbers in note 5 use, and the one the exchangeable copula
    assumes. Counts sum to len(ii).
    """
    a = y[ii].astype(bool)
    b = y[jj].astype(bool)
    n11 = float((a & b).sum())
    n00 = float((~a & ~b).sum())
    off = float((a & ~b).sum() + (~a & b).sum()) / 2.0
    return n11, off, off, n00


def table_stats(n11, n10, n01, n00):
    """(pair-slot switch rate, expected n11 under independence, OR, phi)."""
    n = n11 + n10 + n01 + n00
    p_bar = (n11 + n10) / n  # both margins, symmetrised
    exp11 = n * p_bar**2
    odds = (n11 * n00) / (n10 * n01)
    phi = (n11 * n00 - n10 * n01) / np.sqrt(
        (n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00)
    )
    return float(p_bar), float(exp11), float(odds), float(phi)


def tetrachoric_rho(n11, n10, n01, n00):
    """Latent correlation of a 2x2 table by bivariate-normal MLE.

    Thresholds are the marginal MLEs, h = Phi^-1(1 - p_bar) on both axes;
    rho then maximises the multinomial log-likelihood of the four cells.
    Returns (rho_hat, nll(rho_hat), nll(0)).
    """
    from scipy.optimize import minimize_scalar

    n = n11 + n10 + n01 + n00
    h = np.array([ndtri(1.0 - (n11 + n10) / n)])
    k = np.array([ndtri(1.0 - (n11 + n01) / n)])

    def nll(rho):
        p00 = bvn_cdf(h, k, float(np.clip(rho, -RHO_MAX, RHO_MAX)))[0]
        p10 = ndtr(k)[0] - p00  # X > h, Y <= k
        p01 = ndtr(h)[0] - p00
        p11 = 1.0 - ndtr(h)[0] - ndtr(k)[0] + p00
        cells = np.array([p11, p10, p01, p00])
        counts = np.array([n11, n10, n01, n00])
        return -float((counts * np.log(np.maximum(cells, P_TINY))).sum())

    res = minimize_scalar(
        nll, bounds=(-RHO_MAX, RHO_MAX), method="bounded", options=dict(xatol=1e-9)
    )
    return float(res.x), float(res.fun), nll(0.0)


# --------------------------------------------------------------------------- #
# --raw-stats
# --------------------------------------------------------------------------- #
def raw_stats(rows):
    """Print the feasibility numbers and return the gate verdict string."""
    y, cell = rows["y"], rows["cell"]
    order, starts, sizes = blocks(cell)
    ii, jj = pair_index(cell)
    n_pairs = len(ii)

    print("\n=== eligible switch decisions ===")
    line(f"n eligible ({MASK})", len(y), EXPECTED["eligible"])
    line("overall switch rate", f(y.mean()), EXPECTED["rate"])
    line("per decision round", "rate", note="EXPECTED  n eligible")
    for r in np.unique(rows["round"]):
        m = rows["round"] == r
        line(
            f"  round {r}",
            f(y[m].mean()),
            f"{EXPECTED['per_round'][int(r)]:<8}",
            f"  {int(m.sum())}",
        )

    print("\n=== (episode, decision round, group) cells ===")
    line("n cells", len(sizes), EXPECTED["cells"])
    line("n cells with >= 2 eligible", int((sizes >= 2).sum()), EXPECTED["cells_ge2"])
    hist = np.bincount(sizes)
    line("cell size histogram", "count", note="EXPECTED")
    for s in range(1, len(hist)):
        if hist[s]:
            line(f"  size {s}", int(hist[s]), EXPECTED["hist"].get(s, "-"))
    line("n within-cell pairs", n_pairs, EXPECTED["pairs"])

    n11, n10, n01, n00 = pair_table(y, ii, jj)
    p_bar, exp11, odds, phi = table_stats(n11, n10, n01, n00)
    print("\n=== co-switching over within-cell pairs (symmetrised 2x2) ===")
    line("n11 (both switch)", int(n11), EXPECTED["n11"])
    line("n10 = n01 (one switches)", f(n10))
    line("n00 (neither switches)", int(n00))
    line(
        "pair-slot switch rate",
        f(p_bar),
        note=f"(row rate {f(y.mean())}; big cells switch less)",
    )
    line("expected n11 (independence)", f(exp11), EXPECTED["n11_exp"])
    line("odds ratio", f(odds), EXPECTED["odds_ratio"])
    line("phi coefficient", f(phi), EXPECTED["phi"])

    rho, nll_hat, nll0 = tetrachoric_rho(n11, n10, n01, n00)
    err = check_bvn()
    print("\n=== unconditional tetrachoric (latent-scale) rho ===")
    line("rho (2x2 bivar-normal MLE)", f(rho), EXPECTED["rho"])
    line("  nll(rho_hat)", f(nll_hat))
    line("  nll(0)", f(nll0))
    line("  2*(nll(0) - nll(rho_hat))", f(2.0 * (nll0 - nll_hat)))
    print(
        f"  Phi_2 nodes={N_QUAD}, max abs err vs scipy mvn="
        f"{'unchecked' if err is None else f(err)}"
    )
    print(
        "  UNCONDITIONAL: shared observable features are still in here. The "
        "conditional\n  rho against the GNN marginals (plan step 5) will be "
        "smaller; this is the ceiling."
    )

    ok = n_pairs >= MIN_PAIRS and rho > MIN_RHO
    if ok:
        return f"PASS (pairs {n_pairs} >= {MIN_PAIRS}, rho {f(rho)} > {MIN_RHO})"
    why = []
    if n_pairs < MIN_PAIRS:
        why.append(f"within-cell pairs {n_pairs} < {MIN_PAIRS}")
    if rho <= MIN_RHO:
        why.append(f"tetrachoric rho {f(rho)} <= {MIN_RHO}")
    return "ESCALATE: " + "; ".join(why)


# --------------------------------------------------------------------------- #
def main():
    t0 = time.time()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--raw-stats",
        action="store_true",
        help="feasibility gate: raw within-cell co-switching of the human "
        "train split, no model marginals involved",
    )
    args = ap.parse_args()
    if not args.raw_stats:
        ap.error("--raw-stats is the only mode implemented so far")

    rows = load_rows()
    print(f"data      {TRAIN.relative_to(ROOT)}")
    print(
        f"  games={rows['n_games']} experiments={list(EXPERIMENTS)} "
        f"switch_every={SWITCH_EVERY} mask={MASK} target={TARGET}"
    )
    print("  cell key = (episode, decision round, pre-switch agent_group)")

    verdict = raw_stats(rows)
    print(f"\nGATE: {verdict}")
    print(f"total runtime {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
