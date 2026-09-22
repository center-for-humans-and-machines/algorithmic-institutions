"""Turn a sweep's per-episode parquet shards into per-design-point numbers.

Two objectives are carried side by side and never averaged together, because
they have just been measured to disagree: in the paired competing setting the
correctly-targeted rule minus never-punishing is +8.12 [4.01, 12.22] on total
contribution and +1.07 [-5.79, 7.93] on the common pool. Punishment buys 8.12
contribution, worth 13.0 of pool at the 1.6 multiplier, and costs 12.72 in
spend -- so on the pool it is break-even by construction and the optimal
`P_max` may genuinely be near zero, while on contribution it is not.

Everything is a seat TOTAL per round: the focal manager holds one group
against a live rival and members move, so a policy that raises per-member
contributions while losing the members is not ahead.
"""

import glob
import os

import numpy as np
import pandas as pd

from aimanager.manager.paired_rollout import RPA_LABELS

#: The two objectives, reported separately throughout.
OBJECTIVES = ("focal_contribution", "focal_pool")

#: Per-episode columns that are already per-round means and so pool by a
#: plain average over episodes.
MEAN_COLS = (
    "focal_members",
    "focal_n_valid",
    "focal_contribution",
    "focal_punishment",
    "focal_pool",
    "rival_members",
    "rival_contribution",
    "rival_punishment",
    "rival_pool",
)

#: Ratios that must be formed from pooled numerators and denominators, not
#: by averaging per-episode ratios (episodes differ in how many member-rounds
#: and how many switch decisions they contain).
RATIOS = {
    "mean_p": ("focal_p_num", "focal_p_den"),
    "mean_c_valid": ("focal_c_num", "focal_c_den"),
    "c_leavers": ("lv_c", "lv_n"),
    "c_stayers": ("st_c", "st_n"),
    "p_leavers": ("lv_p", "lv_n"),
    "p_stayers": ("st_p", "st_n"),
    **{f"rpa_{lab}": (f"rpa_p_{lab}", f"rpa_n_{lab}") for lab in RPA_LABELS},
}


def load_sweep(run_dir):
    """Every episode of a sweep run, one row each."""
    shards = sorted(glob.glob(os.path.join(run_dir, "episodes_shard*.parquet")))
    assert shards, f"no episode shards in {run_dir}"
    return pd.concat([pd.read_parquet(p) for p in shards], ignore_index=True)


def _ratio(g, num, den):
    d = g[den].sum()
    return g[num].sum() / d if d > 0 else np.nan


def per_point(df, seeds=None, by=("name",)):
    """Pooled per-design-point summary, plus the standard error of each
    objective over the episodes that went into it."""
    if seeds is not None:
        df = df[df["seed"].isin(list(seeds))]
    by = list(by)
    out = df.groupby(by, sort=False)[list(MEAN_COLS)].mean()
    out["n_episodes"] = df.groupby(by, sort=False).size()
    for obj in OBJECTIVES:
        sd = df.groupby(by, sort=False)[obj].std(ddof=1)
        out[f"se_{obj}"] = sd / np.sqrt(out["n_episodes"])
    # sum the numerators and denominators once, then divide: averaging
    # per-episode ratios would weight episodes by nothing in particular
    parts = sorted({c for pair in RATIOS.values() for c in pair})
    tot = df.groupby(by, sort=False)[parts].sum().reindex(out.index)
    for k, (num, den) in RATIOS.items():
        out[k] = np.where(tot[den] > 0, tot[num] / tot[den].replace(0, np.nan), np.nan)
    out["c_gap"] = out["c_leavers"] - out["c_stayers"]
    out["p_gap"] = out["p_leavers"] - out["p_stayers"]
    out["leave_rate"] = tot["lv_n"] / (tot["lv_n"] + tot["st_n"]).replace(0, np.nan)
    return out.reset_index()


def with_design(summary, design):
    return design.merge(summary, on="name", how="inner")


def paired_bootstrap(df, name_a, name_b, column, n=10000, seed=42):
    """Difference of two design points' means, resampling episodes.

    The episodes of two points in the same rollout are not paired -- a
    manager that punishes differently makes the players act differently and
    consumes a different number of draws -- so this is an unpaired bootstrap
    of the difference, which is the conservative reading.
    """
    rng = np.random.default_rng(seed)
    a = df.loc[df["name"] == name_a, column].to_numpy()
    b = df.loc[df["name"] == name_b, column].to_numpy()
    da = rng.choice(a, size=(n, len(a)), replace=True).mean(1)
    db = rng.choice(b, size=(n, len(b)), replace=True).mean(1)
    d = da - db
    return (
        float(a.mean() - b.mean()),
        float(np.quantile(d, 0.025)),
        float(np.quantile(d, 0.975)),
    )
