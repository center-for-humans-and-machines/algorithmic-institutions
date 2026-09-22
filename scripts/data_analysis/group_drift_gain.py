"""Is the simulation's missing late divergence a *gain* problem?

With two groups there is one number per (episode, round): the gap between the
two group mean contributions, ``gap = mean(group 0) - mean(group 1)``. If a gap
feeds itself forward, the lag-one coefficient of ``gap[t+1]`` on ``gap[t]`` is
near or above one; if it decays, it is well below. This script estimates that
coefficient on the human games and on the committed simulation arms, pooled and
by thirds of the game, with and without the transitions that cross a membership
change, clustered by game.

Headline specification (choices justified in the log file):

* **raw gap**, not normalised -- a lag-one slope is already scale free, and a
  time-varying normaliser would fold the normaliser's own trend into it;
* **unweighted**, one row per realised (episode, round) transition -- the
  realised group mean is what the agents are shown, not a noisy proxy for a
  latent, so its sampling noise is part of the stimulus and not measurement
  error to be corrected away;
* **no intercept** -- the group labels are arbitrary (the human CSVs carry each
  game twice with the labels swapped and the suite keeps one copy), so the gap
  is symmetric about zero by construction;
* **no episode fixed effects** -- a persistent per-episode group offset is the
  object of interest, and within-episode demeaning would absorb exactly that;
* **lag one**, with an AR(2) fit reported beside it as a form check.

Because the round-to-round growth of between-group spread is governed by the
pair (beta, innovation sd), sigma and the implied stationary sd of the gap,
``sigma / sqrt(1 - beta^2)``, are reported next to every coefficient.

PR #195's five reseed replicas are not committed on this branch; point
``--seed-dir`` at a directory holding their ``per_round.parquet`` as
``seed_1.parquet`` .. ``seed_5.parquet`` and ``shipped.parquet``, checked out
read-only from ``origin/auto/seed-spread-noise-floor``.

Usage (local, CPU, no model forward pass):

    .venv/bin/python scripts/data_analysis/group_drift_gain.py \
        --seed-dir <dir with PR #195's six per_round.parquet>

Outputs under plots/data_analysis/evaluation/group_drift_gain/.
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("DISABLE_PANDERA_IMPORT_WARNING", "True")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.api as sm  # noqa: E402

from aimanager.evaluation_suite.convert import load_human, load_sim  # noqa: E402

OUT = ROOT / "plots/data_analysis/evaluation/group_drift_gain"
SIM = ROOT / "plots/simulation"
HUMAN_CSV = ROOT / "experiments/2group_8agent_50ep.csv"
SKIP = "23_2g8a_contr_stimulus_skip_self_gnncopar1_contr_gnn_switch"

# The three arms the question is posed over, plus the two pre-serving-fix
# parents as a guard that nothing here is an artefact of PR #196's fix.
ARMS = {
    "sim_noise_off": SIM / "23_2g8a_sim_timeout_rho0/per_round.parquet",
    "sim_noise_on": SIM / (SKIP + "_simtimeout/per_round.parquet"),
    "sim_noise_off_prefix": SIM / "23_2g8a_sim_timeout_rho0_base/per_round.parquet",
    "sim_noise_on_prefix": SIM / (SKIP + "_timeout/per_round.parquet"),
}
# PR #195's five reseeds of the accepted contributor + the shipped sixth draw.
# Same architecture, same config, same data, same simulation seed: the spread
# over these six is the run-to-run noise floor of any quantity computed here.
SEED_NAMES = [f"seed_{s}" for s in range(1, 6)] + ["shipped"]

BLOCKS = [("1-8", 0, 8), ("9-16", 8, 16), ("17-24", 16, 24)]
CELL = ["episode_id", "round_number", "group_id"]
BOOT = 2000


# --------------------------------------------------------------------------- #
# the gap series
# --------------------------------------------------------------------------- #
def gap_series(canon):
    """One row per (episode, round) where both groups are non-empty.

    gap    mean(group 0) - mean(group 1), over the suite's own valid rows
    n0/n1  the two group sizes at that round (1..8, summing to <= 8)
    sd_ind sd of individual contributions in that episode-round
    within pooled within-group variance of individual contributions
    roster frozenset of members per group, used to flag composition changes
    """
    valid = canon.dropna(subset=["contribution"])
    g = valid.groupby(CELL)["contribution"].agg(["mean", "size", "var"])
    # The roster is membership, taken over ALL rows: a human timeout NaNs the
    # contribution but leaves the player in the group, and counting it as a
    # membership change would put a spurious ~9% of human transitions into the
    # switch bucket and none of the simulation's (the sim records no NaN).
    g["roster"] = canon.groupby(CELL)["participant_code"].agg(frozenset)
    wide = g.unstack("group_id")
    both = wide["mean"].notna().all(axis=1)
    wide = wide[both]
    out = pd.DataFrame(
        {
            "gap": wide[("mean", 0)] - wide[("mean", 1)],
            "n0": wide[("size", 0)].astype(int),
            "n1": wide[("size", 1)].astype(int),
            "roster0": wide[("roster", 0)],
            "roster1": wide[("roster", 1)],
        }
    )
    # pooled within-group variance, the sampling-noise ingredient
    v = wide[[("var", 0), ("var", 1)]].to_numpy()
    n = wide[[("size", 0), ("size", 1)]].to_numpy()
    num = np.nansum(np.where(n > 1, v * (n - 1), 0.0), axis=1)
    den = np.clip(np.where(n > 1, n - 1, 0).sum(axis=1), 1, None)
    out["within_var"] = num / den
    out["ss_gap"] = out["within_var"] * (1.0 / out["n0"] + 1.0 / out["n1"])
    ind = valid.groupby(["episode_id", "round_number"])["contribution"].std()
    out["sd_ind_round"] = ind.reindex(out.index).to_numpy()
    return out.reset_index()


def transitions(gaps):
    """Consecutive (t, t+1) pairs within an episode, with lag-two where it
    exists, block labels, and a flag for a realised composition change."""
    g = gaps.sort_values(["episode_id", "round_number"]).copy()
    by = g.groupby("episode_id")
    for col, sh in [("gap_next", -1), ("gap_lag", 1)]:
        g[col] = by["gap"].shift(sh)
    g["round_next"] = by["round_number"].shift(-1)
    g["round_lag"] = by["round_number"].shift(1)
    g["roster0_next"] = by["roster0"].shift(-1)
    g["roster1_next"] = by["roster1"].shift(-1)
    g["ss_next"] = by["ss_gap"].shift(-1)
    g["sd_ind_next"] = by["sd_ind_round"].shift(-1)
    g = g[g["round_next"] == g["round_number"] + 1].copy()
    g["contiguous_lag"] = g["round_lag"] == g["round_number"] - 1
    g["switch_round"] = (g["roster0"] != g["roster0_next"]) | (
        g["roster1"] != g["roster1_next"]
    )
    g["decision_round"] = (g["round_number"] + 1) % 4 == 0
    g["block"] = "other"
    for name, lo, hi in BLOCKS:
        g.loc[(g["round_number"] >= lo) & (g["round_number"] < hi), "block"] = name
    g["w_prec"] = 1.0 / (1.0 / g["n0"] + 1.0 / g["n1"])
    g["min_n"] = g[["n0", "n1"]].min(axis=1)
    return g


# --------------------------------------------------------------------------- #
# the regressions
# --------------------------------------------------------------------------- #
def ar1(d, weights=None, intercept=False, ycol="gap_next", xcols=("gap",)):
    """Lag-one (or lag-p) OLS clustered by episode."""
    d = d.dropna(subset=[ycol, *xcols])
    if len(d) < 10:
        return None
    X = d[list(xcols)].to_numpy(float)
    if intercept:
        X = sm.add_constant(X, has_constant="add")
    y = d[ycol].to_numpy(float)
    kw = {} if weights is None else {"weights": d[weights].to_numpy(float)}
    model = sm.WLS(y, X, **kw) if weights else sm.OLS(y, X)
    fit = model.fit(cov_type="cluster", cov_kwds={"groups": d["episode_id"].to_numpy()})
    k = 1 if intercept else 0
    beta = float(fit.params[k])
    se = float(fit.bse[k])
    resid = y - fit.fittedvalues
    # Between-group spread grows as sigma / sqrt(1 - beta^2), so beta alone is
    # half the story and the innovation sd is reported beside it everywhere.
    sigma = float(np.sqrt(np.average(resid**2, weights=kw.get("weights"))))
    return dict(
        beta=beta,
        se=se,
        lo=beta - 1.96 * se,
        hi=beta + 1.96 * se,
        n=int(len(d)),
        games=int(d["episode_id"].nunique()),
        sigma=sigma,
        stationary_sd=(
            float(sigma / np.sqrt(1 - beta**2)) if abs(beta) < 1 else float("inf")
        ),
        sd_gap=float(d["gap"].std()),
        params=[float(v) for v in fit.params],
    )


def boot_draws(d, draws=BOOT, seed=0):
    """Episode-cluster bootstrap of the no-intercept lag-one slope."""
    d = d.dropna(subset=["gap_next", "gap"])
    rng = np.random.default_rng(seed)
    eps = d["episode_id"].unique()
    parts = {
        e: d.loc[d["episode_id"] == e, ["gap", "gap_next"]].to_numpy() for e in eps
    }
    out = np.empty(draws)
    for i in range(draws):
        pick = rng.choice(len(eps), len(eps), replace=True)
        m = np.vstack([parts[eps[j]] for j in pick])
        out[i] = (m[:, 0] @ m[:, 1]) / (m[:, 0] @ m[:, 0])
    return out


def boot_ci(d, draws=BOOT, seed=0):
    b = boot_draws(d, draws, seed)
    return float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))


def diff_ci(human_tr, sim_tr):
    """Games resampled whole and independently in both arms; the quantiles of
    the difference are the sampling uncertainty on human minus simulation. It
    does not contain the simulation's run-to-run (retraining) variation, which
    is what the reseed floor supplies separately."""
    rows = []
    for sample, hs, ss in [
        ("all", human_tr, sim_tr),
        (
            "stable",
            human_tr[~human_tr["switch_round"]],
            sim_tr[~sim_tr["switch_round"]],
        ),
        ("switch", human_tr[human_tr["switch_round"]], sim_tr[sim_tr["switch_round"]]),
    ]:
        d = boot_draws(hs, seed=1) - boot_draws(ss, seed=2)
        rows.append(
            dict(
                sample=sample,
                beta_human=ar1(hs)["beta"],
                beta_sim=ar1(ss)["beta"],
                diff=ar1(hs)["beta"] - ar1(ss)["beta"],
                diff_lo=float(np.quantile(d, 0.025)),
                diff_hi=float(np.quantile(d, 0.975)),
                p_two_sided=float(2 * min((d < 0).mean(), (d > 0).mean())),
            )
        )
    return rows


def eiv_beta(d):
    """Method-of-moments attenuation correction. The regressor's sampling
    variance given the rosters is ss_gap = within_var * (1/n0 + 1/n1); dividing
    the slope by 1 - mean(ss_gap)/var(gap) gives the slope that would hold if
    group means were measured without sampling noise. Reported as a robustness
    check only -- see the module docstring for why it is not the headline."""
    d = d.dropna(subset=["gap_next", "gap"])
    lam = 1.0 - d["ss_gap"].mean() / d["gap"].var()
    base = ar1(d)
    return base["beta"] / lam, float(lam)


# --------------------------------------------------------------------------- #
# per-arm table
# --------------------------------------------------------------------------- #
def arm_rows(arm, tr, with_boot=True):
    rows = []
    for sample, sub in [
        ("all", tr),
        ("no_switch_round", tr[~tr["switch_round"]]),
        ("switch_round", tr[tr["switch_round"]]),
    ]:
        for block in ["pooled"] + [b[0] for b in BLOCKS]:
            d = sub if block == "pooled" else sub[sub["block"] == block]
            fit = ar1(d)
            if fit is None:
                continue
            r = dict(arm=arm, sample=sample, block=block)
            r.update({k: v for k, v in fit.items() if k != "params"})
            if with_boot and block == "pooled" and sample != "switch_round":
                r["boot_lo"], r["boot_hi"] = boot_ci(d)
            rows.append(r)
    return rows


def quantities(tr, sd_ind):
    """Every headline number as one flat dict, so the same function can be run
    over the five reseed replicas and give a run-to-run spread for each."""
    q = {"sd_individual": sd_ind}
    for sample, sub in [
        ("all", tr),
        ("stable", tr[~tr["switch_round"]]),
        ("switch", tr[tr["switch_round"]]),
    ]:
        for block in ["pooled"] + [b[0] for b in BLOCKS]:
            d = sub if block == "pooled" else sub[sub["block"] == block]
            fit = ar1(d)
            if fit is None:
                continue
            q[f"beta_{sample}_{block}"] = fit["beta"]
            if block == "pooled":
                q[f"sigma_{sample}"] = fit["sigma"]
                # sigma is in contribution units and simulated individuals are
                # less variable than real ones, so the ratio to the arm's own
                # individual sd says whether the gap innovation is short for a
                # group-level reason or only because individuals are.
                q[f"sigma_over_sd_ind_{sample}"] = fit["sigma"] / sd_ind
                q[f"stationary_sd_{sample}"] = fit["stationary_sd"]
    for k in (2, 3, 4):
        f = ar1(tr[tr["min_n"] >= k])
        if f is not None:
            q[f"beta_all_pooled_min_n{k}"] = f["beta"]
        f = ar1(tr[(tr["min_n"] >= k) & ~tr["switch_round"]])
        if f is not None:
            q[f"beta_stable_pooled_min_n{k}"] = f["beta"]
    f = ar1(tr[tr["contiguous_lag"]], xcols=("gap", "gap_lag"))
    if f is not None:
        q["beta_ar2_sum"] = sum(f["params"])
    for name, lo, hi in BLOCKS:
        m = (tr["round_number"] >= lo) & (tr["round_number"] < hi)
        q[f"sd_gap_{name}"] = float(tr.loc[m, "gap"].std())
    return q


def cycle_params(tr):
    """The four numbers that govern the gap process: three of every four
    transitions leave the rosters alone, the fourth reshuffles them."""
    st = ar1(tr[~tr["switch_round"]])
    sw = ar1(tr[tr["switch_round"]])
    return dict(b_st=st["beta"], s_st=st["sigma"], b_sw=sw["beta"], s_sw=sw["sigma"])


def cycle_sd(b_st, s_st, b_sw, s_sw, k=3):
    """Steady-state sd of the gap under k stable steps then one switch step.

    v -> b^2 v + s^2 each step; solving the four-step cycle for its fixed point
    and averaging the variance over the cycle gives the between-group spread
    the process settles at. This is the quantity the original finding is stated
    in (sd of the group mean, rising 4.19 -> 5.54 -> 6.09 in the humans), so it
    is where a gain difference and an innovation difference can be compared on
    one scale."""
    g = b_st**2
    acc = s_st**2 * sum(g**i for i in range(k))  # variance after k stable steps
    denom = 1 - b_sw**2 * g**k
    if denom <= 0:
        return float("inf")
    v0 = (b_sw**2 * acc + s_sw**2) / denom  # variance just after a reshuffle
    vs = [v0]
    for _ in range(k):
        vs.append(g * vs[-1] + s_st**2)
    return float(np.sqrt(np.mean(vs)))


def counterfactuals(human_tr, sim_tr):
    """Swap each of the four parameters from the simulation to the human one at
    a time and read off how much of the spread shortfall it closes."""
    h, s = cycle_params(human_tr), cycle_params(sim_tr)
    base_h, base_s = cycle_sd(**h), cycle_sd(**s)
    rows = [
        dict(swap="simulation as is", **s, cycle_sd=base_s, closed=0.0),
    ]
    for k in ("b_st", "b_sw", "s_st", "s_sw"):
        p = dict(s)
        p[k] = h[k]
        v = cycle_sd(**p)
        rows.append(
            dict(swap=f"{k} <- human", **p, cycle_sd=v,
                 closed=(v - base_s) / (base_h - base_s))  # fmt: skip
        )
    p = dict(s)
    p["b_st"], p["b_sw"] = h["b_st"], h["b_sw"]
    v = cycle_sd(**p)
    rows.append(dict(swap="both betas <- human", **p, cycle_sd=v,
                     closed=(v - base_s) / (base_h - base_s)))  # fmt: skip
    p = dict(s)
    p["s_st"], p["s_sw"] = h["s_st"], h["s_sw"]
    v = cycle_sd(**p)
    rows.append(dict(swap="both sigmas <- human", **p, cycle_sd=v,
                     closed=(v - base_s) / (base_h - base_s)))  # fmt: skip
    rows.append(dict(swap="human", **h, cycle_sd=base_h, closed=1.0))
    return rows


def alt_rows(arm, tr):
    """Alternative specifications, pooled and on the full sample."""
    out = []

    def add(name, fit, **extra):
        if fit is None:
            return
        r = dict(arm=arm, spec=name, beta=fit["beta"], se=fit["se"], lo=fit["lo"],
                 hi=fit["hi"], n=fit["n"], sigma=fit["sigma"])  # fmt: skip
        r.update(extra)
        out.append(r)

    add("headline (raw gap, unweighted, no intercept)", ar1(tr))
    f = ar1(tr, intercept=True)
    add("with intercept", f, note=f"const {f['params'][0]:+.4f}")
    add("precision weighted w=n0*n1/(n0+n1)", ar1(tr, weights="w_prec"))
    add("both groups >= 2 members", ar1(tr[tr["min_n"] >= 2]))
    add("both groups >= 3 members", ar1(tr[tr["min_n"] >= 3]))
    tr2 = tr[tr["contiguous_lag"]]
    f2 = ar1(tr2, xcols=("gap", "gap_lag"))
    if f2 is not None:
        b1, b2 = f2["params"]
        add("AR(2), lag-one coefficient", f2, note=f"lag2 {b2:+.4f}, sum {b1 + b2:.4f}")
    z = tr.dropna(subset=["sd_ind_round", "sd_ind_next"]).copy()
    z["gap"] = z["gap"] / z["sd_ind_round"]
    z["gap_next"] = z["gap_next"] / z["sd_ind_next"]
    add("gap normalised by the round's individual sd", ar1(z))
    b, lam = eiv_beta(tr)
    out.append(dict(arm=arm, spec="errors-in-variables corrected", beta=b,
                    se=float("nan"), lo=float("nan"), hi=float("nan"),
                    n=int(tr["gap_next"].notna().sum()), sigma=float("nan"),
                    note=f"attenuation factor {lam:.4f}"))  # fmt: skip
    return out


def imputation_sensitivity(human, rate, seed=42, draws=20):
    """The suite drops a human timeout (NaN) but scores a simulated one at the
    env's imputed 9, on ~2.2% of simulated agent-rounds. Bound what that does
    to the coefficient by pushing the same share of human contributions to 9
    and re-fitting. A mean-ward imputation is mean-reverting, so this is the
    direction that would flatter the human arm."""
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(draws):
        h = human.copy()
        hit = rng.random(len(h)) < rate
        h.loc[hit & h["contribution"].notna(), "contribution"] = 9.0
        tr = transitions(gap_series(h))
        for sample, sub in [("all", tr), ("no_switch_round", tr[~tr["switch_round"]])]:
            f = ar1(sub)
            rows.append(dict(draw=d, sample=sample, beta=f["beta"]))
    df = pd.DataFrame(rows)
    return (
        df.groupby("sample")["beta"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
        .assign(rate=rate, draws=draws)
    )


def spread_rows(arm, canon, gaps):
    """sd of the gap and, as the control on the whole pipeline, sd of the group
    mean itself -- the quantity PR #189 states the finding in (human 4.19 ->
    5.54 -> 6.09 over the three blocks)."""
    valid = canon.dropna(subset=["contribution"])
    gm = valid.groupby(CELL)["contribution"].mean().reset_index()
    rows = []
    for name, lo, hi in BLOCKS:
        m = (gaps["round_number"] >= lo) & (gaps["round_number"] < hi)
        mg = (gm["round_number"] >= lo) & (gm["round_number"] < hi)
        rows.append(
            dict(
                arm=arm,
                block=name,
                sd_group_mean=float(gm.loc[mg, "contribution"].std()),
                sd_gap=float(gaps.loc[m, "gap"].std()),
                n_rounds=int(m.sum()),
            )
        )
    return rows


# --------------------------------------------------------------------------- #
def load_arm(path):
    runs = load_sim(path)
    assert len(runs) == 1, f"{path}: expected one run, got {sorted(runs)}"
    return next(iter(runs.values()))


def md(df, path, floatfmt="{:.4f}"):
    """Same table style as copula_closed_loop_variance.py."""
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = [
            floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v)
            for v in r
        ]
        lines.append("| " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-dir", type=Path, required=True)
    args = ap.parse_args()
    seed_arms = {n: args.seed_dir / f"{n}.parquet" for n in SEED_NAMES}

    OUT.mkdir(parents=True, exist_ok=True)
    canon = {"human": load_human(HUMAN_CSV)}
    for arm, p in {**ARMS, **seed_arms}.items():
        canon[arm] = load_arm(p)

    gaps = {a: gap_series(c) for a, c in canon.items()}
    trs = {a: transitions(g) for a, g in gaps.items()}

    main_arms = ["human", "sim_noise_off", "sim_noise_on",
                 "sim_noise_off_prefix", "sim_noise_on_prefix"]  # fmt: skip
    coef = pd.DataFrame(sum((arm_rows(a, trs[a]) for a in main_arms), []))
    coef.to_csv(OUT / "coefficients.csv", index=False)
    md(coef, OUT / "coefficients.md")

    alts = pd.DataFrame(
        sum(
            (alt_rows(a, trs[a]) for a in ["human", "sim_noise_off", "sim_noise_on"]),
            [],
        )
    )
    alts.to_csv(OUT / "alternatives.csv", index=False)
    md(alts, OUT / "alternatives.md")

    spread = pd.DataFrame(
        sum((spread_rows(a, canon[a], gaps[a]) for a in main_arms), [])
    )
    spread.to_csv(OUT / "gap_spread.csv", index=False)
    md(spread, OUT / "gap_spread.md")

    floor_arms = list(seed_arms)
    sd_ind = {a: float(c["contribution"].std()) for a, c in canon.items()}
    q = pd.DataFrame({a: quantities(trs[a], sd_ind[a]) for a in floor_arms + main_arms})
    q.index.name = "quantity"
    floor = q[floor_arms].copy()
    out = pd.DataFrame(
        {
            "seed_mean": floor.mean(axis=1),
            "seed_sd": floor.std(axis=1, ddof=1),
            "seed_min": floor.min(axis=1),
            "seed_max": floor.max(axis=1),
        }
    )
    for a in main_arms:
        out[a] = q[a]
    out["human_minus_noise_off"] = out["human"] - out["sim_noise_off"]
    out["in_seed_sd"] = out["human_minus_noise_off"].abs() / out["seed_sd"]
    out = out.reset_index()
    q.reset_index().to_csv(OUT / "noise_floor_raw.csv", index=False)
    md(q.reset_index(), OUT / "noise_floor_raw.md")
    out.to_csv(OUT / "noise_floor.csv", index=False)
    md(out, OUT / "noise_floor.md")

    cover = pd.DataFrame(
        [
            dict(
                arm=a,
                episodes=int(canon[a]["episode_id"].nunique()),
                agent_rounds=int(len(canon[a])),
                contrib_nan=int(canon[a]["contribution"].isna().sum()),
                rounds_both_groups=int(len(gaps[a])),
                rounds_possible=int(
                    canon[a].groupby(["episode_id", "round_number"]).ngroups
                ),
                transitions=int(len(trs[a])),
                switch_transitions=int(trs[a]["switch_round"].sum()),
                switch_outside_decision=int(
                    (trs[a]["switch_round"] & ~trs[a]["decision_round"]).sum()
                ),
                mean_gap=float(gaps[a]["gap"].mean()),
                mean_min_group=float(trs[a]["min_n"].mean()),
            )
            for a in ["human", *main_arms[1:], *floor_arms]
        ]
    )
    cover.to_csv(OUT / "coverage.csv", index=False)
    md(cover, OUT / "coverage.md")

    dif = pd.DataFrame(
        [
            dict(sim_arm=a, **r)
            for a in ["sim_noise_off", "sim_noise_on"]
            for r in diff_ci(trs["human"], trs[a])
        ]
    )
    dif.to_csv(OUT / "human_minus_sim.csv", index=False)
    md(dif, OUT / "human_minus_sim.md")
    print(dif.to_string(index=False))

    cf = pd.DataFrame(
        [
            dict(sim_arm=a, **r)
            for a in ["sim_noise_off", "sim_noise_on"]
            for r in counterfactuals(trs["human"], trs[a])
        ]
    )
    cf.to_csv(OUT / "counterfactuals.csv", index=False)
    md(cf, OUT / "counterfactuals.md")
    print(cf.to_string(index=False))

    size = pd.DataFrame(
        [
            dict(
                arm=a,
                sample=s,
                min_n=k,
                **{
                    kk: vv
                    for kk, vv in ar1(sub[sub["min_n"] >= k]).items()
                    if kk in ("beta", "se", "lo", "hi", "n")
                },
            )  # fmt: skip
            for a in ["human", "sim_noise_off", "sim_noise_on"]
            for s, sub in [
                ("all", trs[a]),
                ("no_switch_round", trs[a][~trs[a]["switch_round"]]),
            ]
            for k in (1, 2, 3, 4)
        ]
    )
    size.to_csv(OUT / "group_size_cuts.csv", index=False)
    md(size, OUT / "group_size_cuts.md")

    sizes = pd.DataFrame(
        [
            dict(
                arm=a,
                **{
                    f"min_n={k}": v
                    for k, v in trs[a]["min_n"]
                    .value_counts(normalize=True)
                    .sort_index()
                    .items()
                },
            )
            for a in ["human", "sim_noise_off", "sim_noise_on", *floor_arms]
        ]
    )
    sizes.to_csv(OUT / "group_size_distribution.csv", index=False)
    md(sizes, OUT / "group_size_distribution.md")

    rate = 1.0 - canon["sim_noise_off"]["contribution"].notna().mean()
    rate = max(rate, 0.0224)  # the simulation's measured timeout rate (PR #196)
    imp = imputation_sensitivity(canon["human"], rate)
    imp.to_csv(OUT / "imputation_sensitivity.csv", index=False)
    md(imp, OUT / "imputation_sensitivity.md")
    print(imp.to_string(index=False))

    figure(coef, out, gaps, main_arms[:3])
    print(coef.to_string(index=False))
    print(out.to_string(index=False))


COLORS = {"human": "#222222", "sim_noise_off": "#c44e52", "sim_noise_on": "#4c72b0"}


def figure(coef, floor, gaps, arms):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    blocks = ["pooled"] + [b[0] for b in BLOCKS]
    x = np.arange(len(blocks))
    f = floor.set_index("quantity")
    for panel, sample, title in [
        (0, "all", "All transitions"),
        (1, "no_switch_round", "Between membership changes"),
    ]:
        key = "all" if sample == "all" else "stable"
        sd = np.array([f.loc[f"beta_{key}_{b}", "seed_sd"] for b in blocks])
        mu = np.array([f.loc[f"beta_{key}_{b}", "seed_mean"] for b in blocks])
        ax[panel].bar(x, 2 * sd, bottom=mu - sd, width=0.66, color="0.87", zorder=0,
                      label="+-1 sd, 6 reseeds")  # fmt: skip
        for i, a in enumerate(arms):
            d = (
                coef[(coef["arm"] == a) & (coef["sample"] == sample)]
                .set_index("block")
                .reindex(blocks)
            )
            ax[panel].errorbar(
                x + (i - 1) * 0.17,
                d["beta"],
                yerr=[d["beta"] - d["lo"], d["hi"] - d["beta"]],
                fmt="o",
                capsize=3,
                color=COLORS[a],
                label=a,
            )
        ax[panel].axhline(1.0, color="0.5", lw=0.8, ls="--")
        ax[panel].set_xticks(x)
        ax[panel].set_xticklabels(blocks)
        ax[panel].set_ylim(0.55, 1.05)
        ax[panel].set_ylabel("lag-one gap coefficient")
        ax[panel].set_title(title)
        ax[panel].legend(fontsize=7, loc="lower right")
    for a in arms:
        s = gaps[a].groupby("round_number")["gap"].std()
        ax[2].plot(s.index, s.to_numpy(), color=COLORS[a], label=a)
    ax[2].set_xlabel("round")
    ax[2].set_ylabel("sd of the group gap")
    ax[2].set_title("Between-group spread over the game")
    ax[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "group_drift_gain.jpg", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
