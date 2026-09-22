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
    shards = sorted(glob.glob(os.path.join(run_dir, "episodes_*.parquet")))
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


def _midranks(marginal):
    """Tie-corrected ranks for a level with `marginal` observations."""
    cum = np.cumsum(marginal)
    return cum - marginal + (marginal + 1) / 2.0


def spearman_from_counts(counts):
    """Exact tie-corrected Spearman rho from a contribution x punishment
    contingency table.

    **The targeting statistic.** It is invariant to any monotone rescaling of
    the punishment, so a rule that punishes the same people three times as
    hard scores the same -- which a difference of bin means does not. Negative
    means the rule punishes low contributors (correct targeting), positive
    means it punishes high ones, and a rule that never punishes has no
    variance and scores nan rather than 0.
    """
    counts = np.asarray(counts, dtype=float)
    n = counts.sum()
    if n == 0:
        return np.nan
    rc = _midranks(counts.sum(1))[:, None]
    rp = _midranks(counts.sum(0))[None, :]
    mc = (counts * rc).sum() / n
    mp = (counts * rp).sum() / n
    cov = (counts * (rc - mc) * (rp - mp)).sum() / n
    vc = (counts * (rc - mc) ** 2).sum() / n
    vp = (counts * (rp - mp) ** 2).sum() / n
    if vc <= 0 or vp <= 0:
        return np.nan
    return float(cov / np.sqrt(vc * vp))


def load_shape(run_dir):
    """Every sweep part's contingency tables, stacked."""
    parts = sorted(glob.glob(os.path.join(run_dir, "shape_*.parquet")))
    assert parts, f"no shape tables in {run_dir}"
    return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)


def shape_stats(shape, seeds=None):
    """Aim, level and rate per design point -- the three kept apart.

    `rho` is the rank correlation between contribution and punishment: the
    aim, invariant to force. `mean_p_valid` is the force. `punish_rate` and
    `mean_p_given_positive` split that force into how often and how hard, so
    a reader can see whether two rules differ in aim or only in intensity.
    """
    if seeds is not None:
        shape = shape[shape["seed"].isin(list(seeds))]
    rows = []
    for name, g in shape.groupby("name", sort=False):
        tab = (
            g.pivot_table(
                index="contribution",
                columns="punishment",
                values="count",
                aggfunc="sum",
                fill_value=0,
            )
            .reindex(index=range(21), columns=range(31), fill_value=0)
            .to_numpy()
        )
        n = tab.sum()
        n_pos = tab[:, 1:].sum()
        levels = np.arange(31)[None, :]
        total_p = (tab * levels).sum()
        rows.append(
            {
                "name": name,
                "rho": spearman_from_counts(tab),
                "n_decisions": int(n),
                "mean_p_valid": total_p / n if n else np.nan,
                "punish_rate": n_pos / n if n else np.nan,
                "mean_p_given_positive": total_p / n_pos if n_pos else np.nan,
                # How far outside the contribution model's own evidence the
                # rule is asking it to extrapolate. Only 4.49% of that model's
                # training rows follow a punishment above 10 and 1.50% follow
                # one above 20 (manager review S2), so a rule whose spend
                # lives up there is a claim about the model, not about people.
                "spend_share_gt10": (tab[:, 11:] * levels[:, 11:]).sum()
                / (total_p or np.nan),
                "spend_share_gt20": (tab[:, 21:] * levels[:, 21:]).sum()
                / (total_p or np.nan),
                "decision_share_gt10": tab[:, 11:].sum() / n if n else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _bin_means_and_se(num, den, min_bin_n):
    """Pooled bin means and their standard errors over episodes.

    A bin mean is a ratio of two sums over episodes, so its error is the
    ratio estimator's: `sum_e (p_e - m * n_e)^2 / (sum_e n_e)^2`. Episodes
    are the independent unit; agent-rounds inside one are not, which is why
    the error is taken across episodes and not across cells.
    """
    e = len(num)
    tot_n, tot_p = den.sum(0), num.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = np.where(
            tot_n >= min_bin_n, tot_p / np.where(tot_n > 0, tot_n, np.nan), np.nan
        )
        resid = num - np.nan_to_num(m)[None, :] * den
        var = (
            (resid**2).sum(0)
            * (e / max(e - 1, 1))
            / np.where(tot_n > 0, tot_n, np.nan) ** 2
        )
    return m, np.sqrt(var)


def targeting_triple(episodes, seeds=None, min_bin_n=20):
    """Magnitude and noise gate for a policy shape, per design point.

    A rank correlation alone is not enough to call a rule targeted, in
    either direction:

      * **It hides force.** The human managers and the clone differ by 1.088
        on a difference of bin means, which reads as a difference in aim;
        on rank correlation they are identical, both strictly monotone
        decreasing across all six bins, and rescaling the clone to the human
        mean recovers most of the gap. 42% of that apparent difference in
        aim was purely force.
      * **It hides flatness.** Rank discards magnitude entirely, so a profile
        falling 5.00 to 4.99 scores the same as one falling 4.76 to 0.27. A
        sibling's exploration buffer scored -0.540, which sounds like a real
        contingency, on a relationship whose spread relative to its own mean
        was 0.002 -- a flat policy plus sampling noise, ranked.

    So a rule counts as targeting only if it is strong in rank AND
    non-negligible in `magnitude` (the range of its bin means over their
    mean) AND above the `noise_gate` (that range over its own standard
    error). This matters more here than in a hand-picked arm: a search over
    thousands of candidates will find parameter settings that score well on
    rank while punishing almost nothing.
    """
    if seeds is not None:
        episodes = episodes[episodes["seed"].isin(list(seeds))]
    num_cols = [f"rpa_p_{lab}" for lab in RPA_LABELS]
    den_cols = [f"rpa_n_{lab}" for lab in RPA_LABELS]
    rows = []
    for name, g in episodes.groupby("name", sort=False):
        rows.append(
            {
                "name": name,
                **targeting_triple_from_arrays(
                    g[num_cols].to_numpy(float),
                    g[den_cols].to_numpy(float),
                    min_bin_n,
                ),
            }
        )
    return pd.DataFrame(rows)


def targeting_triple_from_arrays(num, den, min_bin_n=20):
    """`(blocks, bins)` punishment sums and counts -> magnitude and gate."""
    m, se = _bin_means_and_se(num, den, min_bin_n)
    ok = ~np.isnan(m)
    if ok.sum() < 2:
        return {"n_bins": int(ok.sum())}
    hi, lo = int(np.nanargmax(m)), int(np.nanargmin(m))
    rng = float(m[hi] - m[lo])
    se_rng = float(np.sqrt(se[hi] ** 2 + se[lo] ** 2))
    mean_m = float(np.nanmean(m))
    if se_rng > 0:
        gate = rng / se_rng
    elif rng > 0:
        # a deterministic rule -- a hard threshold's extreme bins are the same
        # number in every episode -- has no sampling error in them at all, so
        # the gate is unbounded rather than undefined
        gate = np.inf
    else:
        gate = np.nan
    return {
        "n_bins": int(ok.sum()),
        "bin_mean_range": rng,
        "magnitude": rng / mean_m if mean_m else np.nan,
        "noise_gate": gate,
    }


def shape_curve(shape, name, seeds=None):
    """Mean punishment at every contribution level, for one design point."""
    g = shape[shape["name"] == name]
    if seeds is not None:
        g = g[g["seed"].isin(list(seeds))]
    tot = g.groupby("contribution")["count"].sum()
    num = g.assign(x=g["count"] * g["punishment"]).groupby("contribution")["x"].sum()
    return (num / tot).reindex(range(21))


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
