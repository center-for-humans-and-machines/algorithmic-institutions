"""Stage 2 of copula_missing_state.py: the residual analysis (no PyG).

Rows: every valid contribution of the 50 canonical human games, with the skip
trunk's teacher-forced marginal. Cells: (game, round, group_id). Three residual
scales, all "given the model":

  level   r = c - E[c]                     (PR #140's 0.073 lives here)
  latent  z = Phi^-1(F(c-1) + p(c)/2)      mid-point normal score of the
                                           model's own marginal, i.e. the
                                           copula's latent given the level
  MLE     rho of the exchangeable Gaussian copula by the pairwise rectangle
          likelihood (the calibration script's estimator, imported unchanged)

Within-group co-movement is the correlation across cross-member pairs inside a
cell, three estimators: pair-weighted moment (the calibration script's PIT
diagnostic), plain pooled Pearson over the stacked pair list, and the MLE.

A candidate missing-state variable X (group-level, computable from history up
to t-1 plus the membership at t) is "partialled out" by OLS of the residual on
X (intercept + X, pooled over rows); the co-movement of the partialled residual
is recomputed, and the share explained is 1 - rho_after / rho_before. On the
MLE scale the fitted X b shifts both rectangle bounds of the latent (a location
shift of the copula's latent by the candidate's effect). Bootstrap over games
(resampling whole games, cells re-keyed per draw so a game drawn twice never
pairs with itself).
"""

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import ndtri

import punishment_copula_rho as pc

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "plots/data_analysis/evaluation/copula_missing_state"
TABLE = OUT_DIR / "residual_table.parquet"
FULL = ROOT / "experiments/2group_8agent_50ep.csv"
N_LEVELS = 21
SEED = 38381
C_DEF = 9.0  # the training default for prev_contribution (round-0 prior)
SWITCH_EVERY = 4
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]

# candidate -> (family, legal as a trunk feature?, definition)
CANDIDATES = {
    "grp_pun_last": ("a", True, "group's mean punishment received at t-1"),
    "grp_pun_last3": ("a", True, "group's mean punishment over t-3..t-1"),
    "grp_share_pun_last": ("a", True, "share of the group's members punished at t-1"),
    "rounds_since_change": (
        "b",
        True,
        "rounds since the group's membership last changed",
    ),
    "left_at_last": (
        "b",
        True,
        "a member left the group at its last membership change",
    ),
    "joined_at_last": (
        "b",
        True,
        "a member joined the group at its last membership change",
    ),
    "size_now": ("b", True, "the group's current size"),
    "stable_block": ("b", True, "membership unchanged at the last switch round (t>=4)"),
    "other_mean_last": ("c", True, "the other group's mean contribution at t-1"),
    "other_size_now": ("c", True, "the other group's current size"),
    "gap_last": ("c", True, "own-group minus other-group mean contribution at t-1"),
    "own_mean_last": ("d", True, "own group's mean contribution at t-1"),
    "own_mean_last3": ("d", True, "own group's mean contribution over t-3..t-1"),
    "own_trend": ("d", True, "own group's mean at t-1 minus at t-3"),
    "own_sd_last": ("d", True, "within-group SD of contributions at t-1"),
    "early_type": ("e", True, "own group's mean contribution in rounds 0-2 (t>=3)"),
    "cum_mean": ("e", True, "own group's running mean contribution over rounds < t"),
    "round": ("t", True, "round number (the RNN's implicit clock)"),
    "fe_game": ("f", False, "game fixed effect (one manager pair per game)"),
    "fe_game_group": (
        "f",
        False,
        "(game, group_id) fixed effect = one manager, static",
    ),
    "fe_session": ("f", False, "session fixed effect"),
}
FE = {k for k, v in CANDIDATES.items() if v[0] == "f"}
LEGAL = [k for k, v in CANDIDATES.items() if v[1] and v[0] != "t"]


def f(x):
    return repr(float(x))


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def load_frame():
    raw = pd.read_csv(FULL)
    raw = raw[raw["experiment_name"] == "ah_group_switching"]
    keep = raw["episode_id"] == raw.groupby("pair_id")["episode_id"].transform("min")
    raw = raw[keep].copy()
    raw["c_valid"] = raw["player_no_input"] == 0
    raw["p_valid"] = raw["manager_no_input"] == 0
    raw["game"] = raw["pair_id"].astype(int)
    raw = raw.rename(columns={"round_number": "round", "group_id": "group"})
    raw["group"] = raw["group"].astype(int)
    pred = pd.read_parquet(TABLE)
    key = ["global_group_id", "episode_id", "player_id", "round"]
    df = raw.merge(pred.drop(columns=["group", "pair_id"]), on=key, how="left")
    assert len(df) == len(raw) == 9600
    assert df.loc[df["c_valid"], "y"].notna().all() and df["y"].notna().sum() == len(
        pred
    )
    assert (df.loc[df["c_valid"], "y"] == df.loc[df["c_valid"], "contribution"]).all()
    return df.sort_values(["game", "round", "group", "player_id"]).reset_index(
        drop=True
    )


def add_candidates(df):
    """Group-level state from history < t (membership at t is known at t)."""
    c = df["contribution"].where(df["c_valid"])
    p = df["punishment"].where(df["p_valid"])
    g = df.groupby(["game", "round", "group"])
    cell = pd.DataFrame(
        {
            "n": g.size(),
            "mean_c": c.groupby([df.game, df["round"], df.group]).mean(),
            "sd_c": c.groupby([df.game, df["round"], df.group]).std(ddof=0),
            "mean_p": p.groupby([df.game, df["round"], df.group]).mean(),
            "share_p": (p > 0)
            .where(p.notna())
            .groupby([df.game, df["round"], df.group])
            .mean(),
            "members": g["player_id"].agg(frozenset),
        }
    )
    # complete grid (game, round, group) so empty groups exist as rows
    games = sorted(df.game.unique())
    idx = pd.MultiIndex.from_product(
        [games, range(24), [0, 1]], names=["game", "round", "group"]
    )
    cell = cell.reindex(idx)
    cell["n"] = cell["n"].fillna(0).astype(int)
    cell["members"] = cell["members"].apply(
        lambda s: s if isinstance(s, frozenset) else frozenset()
    )
    cell = cell.reset_index()
    cell = cell.sort_values(["game", "group", "round"]).reset_index(drop=True)
    by = cell.groupby(["game", "group"])
    lag = lambda col, k: by[col].shift(k)  # noqa: E731

    cell["grp_pun_last"] = lag("mean_p", 1)
    cell["grp_pun_last3"] = pd.concat(
        [lag("mean_p", k) for k in (1, 2, 3)], axis=1
    ).mean(axis=1)
    cell["grp_share_pun_last"] = lag("share_p", 1)
    cell["own_mean_last"] = lag("mean_c", 1)
    cell["own_mean_last3"] = pd.concat(
        [lag("mean_c", k) for k in (1, 2, 3)], axis=1
    ).mean(axis=1)
    cell["own_trend"] = lag("mean_c", 1) - lag("mean_c", 3)
    cell["own_sd_last"] = lag("sd_c", 1)
    cum = by["mean_c"].transform(lambda s: s.shift(1).expanding().mean())
    cell["cum_mean"] = cum
    early = by["mean_c"].transform(lambda s: s.iloc[:3].mean())
    cell["early_type"] = np.where(cell["round"] >= 3, early, cum)
    cell["size_now"] = cell["n"]

    # membership changes: compare the member set with the previous round's
    prev_members = by["members"].shift(1)
    changed = [
        (r > 0) and (m != pm)
        for r, m, pm in zip(cell["round"], cell["members"], prev_members)
    ]
    cell["changed"] = changed
    left = [
        bool(pm - m) if isinstance(pm, frozenset) else False
        for m, pm in zip(cell["members"], prev_members)
    ]
    joined = [
        bool(m - pm) if isinstance(pm, frozenset) else False
        for m, pm in zip(cell["members"], prev_members)
    ]
    cell["left_now"], cell["joined_now"] = left, joined
    cell["last_change_round"] = by["round"].transform(
        lambda s: s.where(cell.loc[s.index, "changed"]).ffill()
    )
    cell["rounds_since_change"] = cell["round"] - cell["last_change_round"].fillna(0)
    cell["left_at_last"] = (
        by["left_now"]
        .transform(
            lambda s: s.where(cell.loc[s.index, "changed"]).ffill().fillna(False)
        )
        .astype(float)
    )
    cell["joined_at_last"] = (
        by["joined_now"]
        .transform(
            lambda s: s.where(cell.loc[s.index, "changed"]).ffill().fillna(False)
        )
        .astype(float)
    )
    # membership changed at the most recent switch round (4k) <= t
    block_start = (cell["round"] // SWITCH_EVERY) * SWITCH_EVERY
    cell["comp_changed_block"] = (
        cell["last_change_round"].fillna(-1) >= block_start
    ) & (cell["round"] >= SWITCH_EVERY)

    # the other group's state
    other = cell[["game", "round", "group", "mean_c", "n"]].copy()
    other["group"] = 1 - other["group"]
    other = other.rename(columns={"mean_c": "other_mean_c", "n": "other_size_now"})
    cell = cell.merge(other, on=["game", "round", "group"], how="left")
    cell = cell.sort_values(["game", "group", "round"]).reset_index(drop=True)
    cell["other_mean_last"] = cell.groupby(["game", "group"])["other_mean_c"].shift(1)
    cell["gap_last"] = cell["own_mean_last"] - cell["other_mean_last"]

    # round-0 priors and empty-group fills (what the player could know)
    for col in ("grp_pun_last", "grp_pun_last3", "grp_share_pun_last"):
        cell[col] = cell[col].fillna(0.0)
    for col in (
        "own_mean_last",
        "own_mean_last3",
        "cum_mean",
        "early_type",
        "other_mean_last",
    ):
        cell[col] = cell[col].fillna(C_DEF)
    cell["gap_last"] = cell["gap_last"].fillna(0.0)
    cell["own_trend"] = cell["own_trend"].fillna(0.0)
    cell["own_sd_last"] = cell["own_sd_last"].fillna(0.0)

    cell["stable_block"] = (
        (~cell["comp_changed_block"]) & (cell["round"] >= SWITCH_EVERY)
    ).astype(float)
    keep = ["game", "round", "group", "comp_changed_block", "changed"] + [
        k for k in CANDIDATES if k not in FE and k != "round"
    ]
    out = df.merge(cell[keep], on=["game", "round", "group"], how="left")
    assert out[[k for k in CANDIDATES if k not in FE]].notna().all().all()
    out["fe_game"] = out["game"]
    out["fe_game_group"] = out["game"] * 2 + out["group"]
    out["fe_session"] = out["session"].astype("category").cat.codes
    return out


# --------------------------------------------------------------------------- #
# residuals and the three co-movement estimators
# --------------------------------------------------------------------------- #
def residual_scales(v):
    P = v[[f"p{k}" for k in range(N_LEVELS)]].to_numpy(np.float64)
    y = v["y"].to_numpy(np.int64)
    f_lo, p_y = pc.pit_parts(P, y)
    u_mid = np.clip(f_lo + 0.5 * p_y, pc.U_EPS, 1 - pc.U_EPS)
    z_lo, z_hi = pc.cdf_bounds(P, y)
    return dict(
        c=v["y"].to_numpy(np.float64),
        r=v["y"].to_numpy(np.float64) - v["e"].to_numpy(np.float64),
        z=ndtri(u_mid),
        z_lo=z_lo,
        z_hi=z_hi,
        P=P,
        y=y,
    )


def cells_of(v):
    return (v["game"].to_numpy() * 24 + v["round"].to_numpy()) * 2 + v[
        "group"
    ].to_numpy()


def pair_moment(x, ii, jj, rows=None):
    """Pair-weighted moment estimator (pc.rho_pairs): mean over pairs of the
    centred product over the row variance; `rows` = the row multiset."""
    xr = x if rows is None else x[rows]
    m, var = xr.mean(), xr.var(ddof=1)
    return float(((x[ii] - m) * (x[jj] - m)).mean() / var)


def pair_pearson(x, ii, jj):
    """Plain Pearson over the stacked, symmetric pair list."""
    a = np.concatenate([x[ii], x[jj]])
    b = np.concatenate([x[jj], x[ii]])
    return float(np.corrcoef(a, b)[0, 1])


def mle(z_lo, z_hi, ii, jj, shift=None, grid=pc.RHO_GRID):
    if shift is not None:
        z_lo, z_hi = z_lo - shift, z_hi - shift
    H, K, sgn = pc.rect_points(z_lo, z_hi, ii, jj)
    return pc.rho_mle(H, K, sgn, grid=grid)[0]


def design(df, names):
    cols = []
    for n in names:
        x = df[n].to_numpy(np.float64)
        cols.append(x)
    X = np.column_stack(cols) if cols else np.empty((len(df), 0))
    return np.column_stack([np.ones(len(df)), X])


def partial_out(target, df, names, rows=None):
    """target - fitted(intercept + X) with the fit on the row multiset `rows`;
    fixed effects (names in FE) are demeaned by key instead. Returns
    (partialled target over ALL rows, the fitted shift, coefficients)."""
    fe = [n for n in names if n in FE]
    ols = [n for n in names if n not in FE]
    shift = np.zeros(len(target))
    t = target.copy()
    for n in fe:
        key = df[n].to_numpy()
        means = pd.Series(t).groupby(key).transform("mean").to_numpy()
        shift += means
        t = t - means
    beta = None
    if ols:
        X = design(df, ols)
        Xr, tr = (X, t) if rows is None else (X[rows], t[rows])
        beta = np.linalg.lstsq(Xr, tr, rcond=None)[0]
        fit = X @ beta
        shift += fit
        t = t - fit
    return t, shift, beta


# --------------------------------------------------------------------------- #
# bootstrap over games
# --------------------------------------------------------------------------- #
class GameBoot:
    def __init__(self, df, cell, seed=SEED):
        self.game = df["game"].to_numpy()
        self.games = np.unique(self.game)
        self.ii, self.jj = pc.pair_index(cell)
        pg = self.game[self.ii]
        assert np.array_equal(pg, self.game[self.jj])
        self.pairs_of = {int(g): np.flatnonzero(pg == g) for g in self.games}
        self.rows_of = {int(g): np.flatnonzero(self.game == g) for g in self.games}
        self.rng = np.random.default_rng(seed)

    def draw(self):
        d = self.rng.choice(self.games, size=len(self.games), replace=True)
        rows = np.concatenate([self.rows_of[int(g)] for g in d])
        pairs = np.concatenate([self.pairs_of[int(g)] for g in d])
        return rows, self.ii[pairs], self.jj[pairs]


def stat_all(sc, df, names, rows, ii, jj, with_mle, grid):
    """(moment on z, pearson on r, pearson on z, mle) after partialling
    `names` (fit on `rows`), over the pair list (ii, jj)."""
    r_p, _, _ = partial_out(sc["r"], df, names, rows)
    z_p, shift, _ = partial_out(sc["z"], df, names, rows)
    out = dict(
        pit_moment=pair_moment(z_p, ii, jj, rows),
        lvl_pearson=pair_pearson(r_p, ii, jj),
        pit_pearson=pair_pearson(z_p, ii, jj),
    )
    if with_mle:
        out["mle"] = mle(sc["z_lo"], sc["z_hi"], ii, jj, shift if names else None, grid)
    return out


def ci(a):
    a = np.asarray(a, float)
    return float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5))


# --------------------------------------------------------------------------- #
def analyse(args):
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta = json.loads((OUT_DIR / "predict_meta.json").read_text())
    df = add_candidates(load_frame())
    v = df[df["c_valid"]].reset_index(drop=True)
    sc = residual_scales(v)
    cell = cells_of(v)
    boot = GameBoot(v, cell)
    ii, jj = boot.ii, boot.jj
    n_boot, n_boot_mle = args.n_boot, args.n_boot_mle
    log = []

    def say(s=""):
        print(s)
        log.append(s)

    say(f"rows={len(v)} games={v.game.nunique()} pairs={len(ii)}")

    # ------------------------------------------------------------ baseline
    say("\n=== 1. baseline: within-group co-movement, cross-member pairs ===")
    base = {}
    base["raw contribution, plain Pearson"] = pair_pearson(sc["c"], ii, jj)
    base["raw contribution, pair moment"] = pair_moment(sc["c"], ii, jj)
    base["level residual, plain Pearson"] = pair_pearson(sc["r"], ii, jj)
    base["level residual, pair moment"] = pair_moment(sc["r"], ii, jj)
    base["latent (mid-PIT), plain Pearson"] = pair_pearson(sc["z"], ii, jj)
    base["latent (mid-PIT), pair moment"] = pair_moment(sc["z"], ii, jj)
    Zr, _ = pc.latents(sc["P"], sc["y"], pc.N_PIT, SEED)
    base["latent (randomised PIT), pair moment"] = float(
        np.mean([pair_moment(Zr[:, k], ii, jj) for k in range(Zr.shape[1])])
    )
    rho50 = mle(sc["z_lo"], sc["z_hi"], ii, jj)
    base["latent, pairwise MLE (50 games)"] = rho50
    tr = v["in_train_split"].to_numpy()
    ii_t, jj_t = pc.pair_index(np.where(tr, cell, -1 - np.arange(len(cell))))
    base["latent, pairwise MLE (40 train games)"] = mle(
        sc["z_lo"], sc["z_hi"], ii_t, jj_t
    )
    base["level residual, plain Pearson (40 train games)"] = pair_pearson(
        sc["r"], ii_t, jj_t
    )
    base["raw contribution, plain Pearson (40 train games)"] = pair_pearson(
        sc["c"], ii_t, jj_t
    )
    assert (
        abs(base["latent, pairwise MLE (40 train games)"] - meta["rho_mle_train40"])
        < 1e-9
    )
    # bootstrap the four headline numbers
    bs = {k: [] for k in ("raw", "lvl", "pit", "mle")}
    for b in range(n_boot):
        rows, bi, bj = boot.draw()
        bs["raw"].append(pair_pearson(sc["c"], bi, bj))
        bs["lvl"].append(pair_pearson(sc["r"], bi, bj))
        bs["pit"].append(pair_moment(sc["z"], bi, bj, rows))
        if b < n_boot_mle:
            bs["mle"].append(mle(sc["z_lo"], sc["z_hi"], bi, bj))
    base_ci = {
        "raw contribution, plain Pearson": ci(bs["raw"]),
        "level residual, plain Pearson": ci(bs["lvl"]),
        "latent (mid-PIT), pair moment": ci(bs["pit"]),
        "latent, pairwise MLE (50 games)": ci(bs["mle"]),
    }
    base_tab = pd.DataFrame(
        [
            {
                "quantity": k,
                "value": val,
                "ci_lo": base_ci.get(k, (np.nan, np.nan))[0],
                "ci_hi": base_ci.get(k, (np.nan, np.nan))[1],
            }
            for k, val in base.items()
        ]
    )
    say(base_tab.to_string(index=False))
    base_tab.to_csv(OUT_DIR / "baseline.csv", index=False)

    # splits
    say("\n--- baseline by split (point estimates) ---")
    rnd = v["round"].to_numpy()
    splits = {
        "all": np.ones(len(v), bool),
        "round 0 excluded": rnd > 0,
        "rounds 0-7": rnd <= 7,
        "rounds 8-15": (rnd >= 8) & (rnd <= 15),
        "rounds 16-23": rnd >= 16,
        "membership changed this block (t>=4)": v["comp_changed_block"].to_numpy(),
        "membership unchanged this block (t>=4)": (~v["comp_changed_block"].to_numpy())
        & (rnd >= 4),
        "membership changed this round": v["changed"].to_numpy(),
    }
    srows = []
    for name, sel in splits.items():
        m = sel[ii] & sel[jj]
        srows.append(
            dict(
                split=name,
                n_rows=int(sel.sum()),
                n_pairs=int(m.sum()),
                raw_pearson=pair_pearson(sc["c"], ii[m], jj[m]),
                lvl_pearson=pair_pearson(sc["r"], ii[m], jj[m]),
                pit_moment=pair_moment(sc["z"], ii[m], jj[m], np.flatnonzero(sel)),
                mle=mle(sc["z_lo"], sc["z_hi"], ii[m], jj[m]),
            )
        )
    split_tab = pd.DataFrame(srows)
    say(split_tab.to_string(index=False))
    split_tab.to_csv(OUT_DIR / "baseline_splits.csv", index=False)

    # ------------------------------------------------------------ candidates
    say("\n=== 2. candidates: regression and share of co-movement explained ===")
    game = v["game"].to_numpy()
    full = stat_all(sc, v, [], None, ii, jj, True, pc.RHO_GRID)
    rows_c = []
    boot_draws = [boot.draw() for _ in range(n_boot)]
    for name, (fam, legal, desc) in CANDIDATES.items():
        # pooled OLS with game-clustered SEs, level and latent scale
        reg = {}
        if name not in FE:
            X = sm.add_constant(v[[name]].to_numpy(np.float64))
            for tgt, lab in ((sc["r"], "lvl"), (sc["z"], "pit")):
                fit = sm.OLS(tgt, X).fit(cov_type="cluster", cov_kwds={"groups": game})
                reg[f"beta_{lab}"] = float(fit.params[1])
                reg[f"t_{lab}"] = float(fit.tvalues[1])
                reg[f"r2_{lab}"] = float(fit.rsquared)
            # cell-mean version: the shared component regressed on the candidate
            cm = pd.DataFrame(
                {
                    "z": sc["z"],
                    "x": v[name].to_numpy(np.float64),
                    "cell": cell,
                    "game": game,
                }
            )
            cm = cm.groupby("cell").agg(
                z=("z", "mean"),
                x=("x", "first"),
                n=("z", "size"),
                game=("game", "first"),
            )
            cm = cm[cm["n"] >= 2]
            fit = sm.WLS(
                cm["z"].to_numpy(),
                sm.add_constant(cm["x"].to_numpy()),
                weights=cm["n"].to_numpy(),
            ).fit(cov_type="cluster", cov_kwds={"groups": cm["game"].to_numpy()})
            reg["beta_cellmean"] = float(fit.params[1])
            reg["t_cellmean"] = float(fit.tvalues[1])
            reg["r2_cellmean"] = float(fit.rsquared)
        after = stat_all(sc, v, [name], None, ii, jj, True, pc.RHO_GRID)
        shares = {k: 1 - after[k] / full[k] for k in after}
        # bootstrap the shares
        bsh = {k: [] for k in after}
        for b, (rows, bi, bj) in enumerate(boot_draws):
            do_mle = b < n_boot_mle
            grid = np.unique(
                np.clip(
                    np.round(
                        np.arange(full["mle"] - 0.15, full["mle"] + 0.1501, 0.05), 6
                    ),
                    0,
                    pc.RHO_MAX,
                )
            )
            f0 = stat_all(sc, v, [], rows, bi, bj, do_mle, grid)
            f1 = stat_all(sc, v, [name], rows, bi, bj, do_mle, grid)
            for k in f1:
                bsh[k].append(1 - f1[k] / f0[k] if f0[k] != 0 else np.nan)
        row = dict(candidate=name, family=fam, legal=legal, description=desc, **reg)
        for k in after:
            lo, hi = ci(bsh[k])
            row[f"rho_after_{k}"] = after[k]
            row[f"share_{k}"] = shares[k]
            row[f"share_{k}_lo"] = lo
            row[f"share_{k}_hi"] = hi
        rows_c.append(row)
        say(
            f"{name:<22} share: pit_moment {shares['pit_moment']:+.3f} "
            f"[{row['share_pit_moment_lo']:+.3f},{row['share_pit_moment_hi']:+.3f}]  "
            f"lvl {shares['lvl_pearson']:+.3f} [{row['share_lvl_pearson_lo']:+.3f},{row['share_lvl_pearson_hi']:+.3f}]  "
            f"mle {shares['mle']:+.3f} [{row['share_mle_lo']:+.3f},{row['share_mle_hi']:+.3f}]"
            + (
                f"  | t_cellmean {reg['t_cellmean']:+.2f} r2 {reg['r2_cellmean']:.3f}"
                if reg
                else ""
            )
        )
    cand_tab = pd.DataFrame(rows_c)
    cand_tab.to_csv(OUT_DIR / "candidates.csv", index=False)
    pd.Series({f"rho_full_{k}": val for k, val in full.items()}).to_json(
        OUT_DIR / "rho_full.json", indent=2
    )

    # ------------------------------------------------------------ forward selection
    say(
        "\n=== 3. forward selection over the legal candidates (criterion: pit_moment) ==="
    )
    chosen, cur = [], full["pit_moment"]
    sel_rows = []
    remaining = list(LEGAL)
    while remaining:
        best, best_val = None, cur
        for name in remaining:
            val = stat_all(sc, v, chosen + [name], None, ii, jj, False, None)[
                "pit_moment"
            ]
            if val < best_val:
                best, best_val = name, val
        if best is None:
            break
        drops = []
        for rows, bi, bj in boot_draws:
            a = stat_all(sc, v, chosen, rows, bi, bj, False, None)["pit_moment"]
            b_ = stat_all(sc, v, chosen + [best], rows, bi, bj, False, None)[
                "pit_moment"
            ]
            drops.append(a - b_)
        lo, hi = ci(drops)
        sel_rows.append(
            dict(
                step=len(chosen) + 1,
                added=best,
                rho_before=cur,
                rho_after=best_val,
                drop=cur - best_val,
                drop_lo=lo,
                drop_hi=hi,
                kept=lo > 0,
            )
        )
        say(
            f"step {len(chosen)+1}: +{best:<20} rho {cur:.4f} -> {best_val:.4f}  drop {cur-best_val:+.4f} [{lo:+.4f},{hi:+.4f}]  {'KEEP' if lo > 0 else 'STOP (CI includes 0)'}"
        )
        if lo <= 0:
            break
        chosen.append(best)
        remaining.remove(best)
        cur = best_val
    sel_tab = pd.DataFrame(sel_rows)
    sel_tab.to_csv(OUT_DIR / "forward_selection.csv", index=False)

    say(f"\njoint set: {chosen}")
    joint = stat_all(sc, v, chosen, None, ii, jj, True, pc.RHO_GRID) if chosen else full
    bj_ = {k: [] for k in joint}
    for b, (rows, bi, bjj) in enumerate(boot_draws):
        do_mle = b < n_boot_mle
        grid = np.unique(
            np.clip(
                np.round(np.arange(full["mle"] - 0.15, full["mle"] + 0.1501, 0.05), 6),
                0,
                pc.RHO_MAX,
            )
        )
        f0 = stat_all(sc, v, [], rows, bi, bjj, do_mle, grid)
        f1 = stat_all(sc, v, chosen, rows, bi, bjj, do_mle, grid)
        for k in f1:
            bj_[k].append(1 - f1[k] / f0[k])
    joint_rows = []
    for k in joint:
        lo, hi = ci(bj_[k])
        joint_rows.append(
            dict(
                scale=k,
                rho_before=full[k],
                rho_after=joint[k],
                share=1 - joint[k] / full[k],
                share_lo=lo,
                share_hi=hi,
            )
        )
    joint_tab = pd.DataFrame(joint_rows)
    say(joint_tab.to_string(index=False))
    joint_tab.to_csv(OUT_DIR / "joint_model.csv", index=False)
    if chosen:
        X = sm.add_constant(v[chosen].to_numpy(np.float64))
        fit = sm.OLS(sc["z"], X).fit(cov_type="cluster", cov_kwds={"groups": game})
        coef = pd.DataFrame(
            {"term": ["const"] + chosen, "beta_pit": fit.params, "t": fit.tvalues}
        )
        say(coef.to_string(index=False))
        coef.to_csv(OUT_DIR / "joint_model_coefficients.csv", index=False)
    # the same for the full legal set and for the FE decomposition
    say("\n--- reference sets ---")
    ref_rows = []
    for label, names in (
        ("all legal candidates", LEGAL),
        ("game FE", ["fe_game"]),
        ("(game, group) FE", ["fe_game_group"]),
        ("joint set + (game, group) FE", chosen + ["fe_game_group"]),
    ):
        st = stat_all(sc, v, names, None, ii, jj, True, pc.RHO_GRID)
        ref_rows.append(
            dict(
                set=label,
                **{f"rho_{k}": st[k] for k in st},
                **{f"share_{k}": 1 - st[k] / full[k] for k in st},
            )
        )
    ref_tab = pd.DataFrame(ref_rows)
    say(ref_tab.to_string(index=False))
    ref_tab.to_csv(OUT_DIR / "reference_sets.csv", index=False)

    # ------------------------------------------------------------ persistence
    say("\n=== 4. persistence of the leftover within-group residual ===")
    z_after, shift_after, _ = partial_out(sc["z"], v, chosen)
    pers_rows = []
    for label, z in (("before partialling", sc["z"]), ("after joint set", z_after)):
        cm = pd.DataFrame(
            {"z": z, "game": game, "round": rnd, "group": v["group"].to_numpy()}
        )
        m = (
            cm.groupby(["game", "group", "round"])
            .agg(z=("z", "mean"), n=("z", "size"))
            .reset_index()
        )
        m = m[m["n"] >= 2].sort_values(["game", "group", "round"])
        # pooled lag-k autocorrelation of the cell-mean residual within (game, group)
        wide = m.pivot_table(index=["game", "group"], columns="round", values="z")
        ac = {}
        for k in (1, 2, 3, 4, 8):
            a = wide.iloc[:, :-k].to_numpy().ravel() if k else None
            b_ = wide.iloc[:, k:].to_numpy().ravel()
            ok = np.isfinite(a) & np.isfinite(b_)
            ac[k] = float(np.corrcoef(a[ok], b_[ok])[0, 1])
        # static vs round-local: one-way ICC of the cell mean across rounds within (game, group)
        series_mean = m.groupby(["game", "group"])["z"].transform("mean")
        var_static = float(m.groupby(["game", "group"])["z"].mean().var(ddof=1))
        var_local = float((m["z"] - series_mean).var(ddof=1))
        # cross-player lag-1 pairs (the calibration script's phi numerator)
        keyr = {}
        for pos, (g, gr, r) in enumerate(zip(game, v["group"].to_numpy(), rnd)):
            keyr.setdefault((g, gr, r), []).append(pos)
        pid = v["player_id"].to_numpy()

        def cross_lag(k):
            li, lj = [], []
            for (g, gr, r), a in keyr.items():
                b_ = keyr.get((g, gr, r + k))
                if not b_:
                    continue
                for i in a:
                    for j in b_:
                        if pid[i] != pid[j]:
                            li.append(i)
                            lj.append(j)
            return np.array(li), np.array(lj)

        li, lj = cross_lag(1)
        rho0 = pair_moment(z, ii, jj)
        rho1 = pair_moment(z, li, lj)
        # the lag profile of cross-player pair correlation: rho_k = static +
        # local * phi^k, so the plateau is the static (game, group) component
        # and the decay is the round-local one -- unbiased, unlike a fixed
        # effect demeaned in-sample
        lagprof = {}
        for k in (1, 2, 3, 4, 6, 8, 12):
            a_, b_ = cross_lag(k)
            lagprof[k] = pair_moment(z, a_, b_)
        rho1_mle = mle(
            sc["z_lo"],
            sc["z_hi"],
            li,
            lj,
            shift_after if label.startswith("after") else None,
        )
        rho0_mle = mle(
            sc["z_lo"],
            sc["z_hi"],
            ii,
            jj,
            shift_after if label.startswith("after") else None,
        )
        pers_rows.append(
            dict(
                stage=label,
                **{f"acf_lag{k}_cellmean": val for k, val in ac.items()},
                **{f"rho_cross_lag{k}": val for k, val in lagprof.items()},
                var_static_cellmean=var_static,
                var_local_cellmean=var_local,
                static_share=var_static / (var_static + var_local),
                rho_within=rho0,
                rho_lag1_cross=rho1,
                phi_moment=rho1 / rho0,
                rho_within_mle=rho0_mle,
                rho_lag1_cross_mle=rho1_mle,
                phi_mle=rho1_mle / rho0_mle if rho0_mle > 0 else np.nan,
            )
        )
    pers_tab = pd.DataFrame(pers_rows)
    say(pers_tab.T.to_string())
    pers_tab.to_csv(OUT_DIR / "persistence.csv", index=False)

    # ------------------------------------------------------------ figures
    figures(cand_tab, split_tab, pers_tab, full)
    (OUT_DIR / "analysis_log.txt").write_text("\n".join(log) + "\n")
    print(f"analysis wall {time.time() - t0:.1f}s")


def figures(cand_tab, split_tab, pers_tab, full):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {"font.size": 9, "axes.spines.top": False, "axes.spines.right": False}
    )
    # 1. share explained per candidate, three scales
    t = cand_tab.sort_values("share_pit_moment", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 0.32 * len(t) + 1.2))
    yy = np.arange(len(t))
    for k, (scale, col, lab) in enumerate(
        (
            ("pit_moment", PALETTE[0], "latent, pair moment"),
            ("lvl_pearson", PALETTE[1], "level, Pearson"),
            ("mle", PALETTE[2], "latent, MLE"),
        )
    ):
        off = (k - 1) * 0.27
        ax.errorbar(
            t[f"share_{scale}"],
            yy + off,
            xerr=[
                np.clip(t[f"share_{scale}"] - t[f"share_{scale}_lo"], 0, None),
                np.clip(t[f"share_{scale}_hi"] - t[f"share_{scale}"], 0, None),
            ],
            fmt="o",
            ms=4,
            color=col,
            ecolor=col,
            elinewidth=1,
            capsize=0,
            label=lab,
        )
    ax.axvline(0, color="#999", lw=1)
    ax.set_yticks(yy)
    ax.set_yticklabels(
        [
            f"{c}{'' if l else ' (FE, illegal)'}"
            for c, l in zip(t["candidate"], t["legal"])
        ]
    )
    ax.set_xlabel(
        "share of residual co-movement explained, 1 - rho_after / rho_before (95% game bootstrap)"
    )
    ax.set_title(
        "Candidate missing state: share of the copula's residual dependence explained",
        loc="left",
    )
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "candidate_shares.png", dpi=150)
    plt.close(fig)

    # 2. baseline by split
    fig, ax = plt.subplots(figsize=(7, 3.2))
    x = np.arange(len(split_tab))
    w = 0.38
    ax.bar(
        x - w / 2,
        split_tab["raw_pearson"],
        w,
        color=PALETTE[0],
        label="raw contribution",
    )
    ax.bar(
        x + w / 2,
        split_tab["lvl_pearson"],
        w,
        color=PALETTE[1],
        label="residual given the trunk (level)",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(split_tab["split"], rotation=30, ha="right")
    ax.set_ylabel("cross-member pair correlation")
    ax.set_title("Within-group co-movement: raw vs residual, by split", loc="left")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "baseline_splits.png", dpi=150)
    plt.close(fig)

    # 3. persistence: lag profile of the cell-mean residual
    fig, ax = plt.subplots(figsize=(5, 3))
    lags = [0, 1, 2, 3, 4, 6, 8, 12]
    for k, (_, row) in enumerate(pers_tab.iterrows()):
        ys = [row["rho_within"]] + [row[f"rho_cross_lag{l}"] for l in lags[1:]]
        ax.plot(lags, ys, "o-", color=PALETTE[k], label=row["stage"], lw=2, ms=5)
    ax.axhline(0, color="#999", lw=1)
    ax.set_xlabel("lag (rounds); 0 = within-round")
    ax.set_ylabel("cross-player pair correlation of the latent")
    ax.set_title("Persistence of the shared residual within (game, group)", loc="left")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "persistence.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# persistence, bootstrapped: the static component read off lags >= 2
# --------------------------------------------------------------------------- #
def persistence_boot(args):
    """A static (game, group) latent produces the SAME cross-player
    correlation at every lag, a round-local shock none beyond its echo. So
    the pooled cross-player correlation over all round pairs at lag >= 2 is
    an unbiased read of the static component (the in-sample fixed effect is
    not: demeaning a 24-round series removes part of a local shock too).
    Bootstrapped over games; before and after the joint set."""
    df = add_candidates(load_frame())
    v = df[df["c_valid"]].reset_index(drop=True)
    sc = residual_scales(v)
    cell = cells_of(v)
    ii, jj = pc.pair_index(cell)
    game = v["game"].to_numpy()
    coef_path = OUT_DIR / "joint_model_coefficients.csv"
    chosen = pd.read_csv(coef_path)["term"].tolist()[1:] if coef_path.exists() else []
    z_after, shift_after, _ = partial_out(sc["z"], v, chosen)
    grp, rnd, pid = (
        v["group"].to_numpy(),
        v["round"].to_numpy(),
        v["player_id"].to_numpy(),
    )
    keyr = {}
    for pos, k in enumerate(zip(game, grp, rnd)):
        keyr.setdefault(k, []).append(pos)

    def cross(lags):
        li, lj = [], []
        for (g, gr, r), a in keyr.items():
            for k in lags:
                b_ = keyr.get((g, gr, r + k))
                if not b_:
                    continue
                for i in a:
                    for j in b_:
                        if pid[i] != pid[j]:
                            li.append(i)
                            lj.append(j)
        return np.array(li), np.array(lj)

    sets = {
        "lag0": (ii, jj),
        "lag1": cross([1]),
        "lag2": cross([2]),
        "lag3": cross([3]),
        "lag>=2": cross(range(2, 24)),
        "lag>=4": cross(range(4, 24)),
    }
    games = np.unique(game)
    per_game = {
        k: {int(g): np.flatnonzero(game[a] == g) for g in games}
        for k, (a, _) in sets.items()
    }
    rng = np.random.default_rng(SEED)
    rows = []
    for label, z, shift in (
        ("before partialling", sc["z"], None),
        ("after joint set", z_after, shift_after),
    ):
        point = {k: pair_moment(z, a, b) for k, (a, b) in sets.items()}
        point_mle = {
            k: mle(sc["z_lo"], sc["z_hi"], a, b, shift) for k, (a, b) in sets.items()
        }
        bs = {k: [] for k in sets}
        bs["static_share"] = []
        for _ in range(args.n_boot):
            d = rng.choice(games, size=len(games), replace=True)
            rows_b = np.concatenate([np.flatnonzero(game == g) for g in d])
            est = {}
            for k, (a, b) in sets.items():
                pos = np.concatenate([per_game[k][int(g)] for g in d])
                est[k] = pair_moment(z, a[pos], b[pos], rows_b)
                bs[k].append(est[k])
            bs["static_share"].append(est["lag>=2"] / est["lag0"])
        row = dict(stage=label)
        for k in sets:
            lo, hi = ci(bs[k])
            row[f"{k}_moment"], row[f"{k}_lo"], row[f"{k}_hi"] = point[k], lo, hi
            row[f"{k}_mle"], row[f"{k}_pairs"] = point_mle[k], len(sets[k][0])
        lo, hi = ci(bs["static_share"])
        row["static_share"], row["static_share_lo"], row["static_share_hi"] = (
            point["lag>=2"] / point["lag0"],
            lo,
            hi,
        )
        row["phi_lag1"] = point["lag1"] / point["lag0"]
        rows.append(row)
    tab = pd.DataFrame(rows)
    print(tab.T.to_string())
    tab.to_csv(OUT_DIR / "persistence_boot.csv", index=False)
