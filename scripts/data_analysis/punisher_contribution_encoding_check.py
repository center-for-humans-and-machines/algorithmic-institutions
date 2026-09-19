"""Does the human punishment distribution need more than one shape in c_t?

The parent branch (`notes/autoresearch_log/punisher-ceiling-fix.md`) fixed the
punisher at the contribution ceiling and left the rest of the contribution
response untouched: the OLS weight of punishment on the current contribution
is about -0.14 in every simulated stack against the human -0.242. Contribution
enters both punishers as a single numeric feature, so its effect on the
31-level punishment distribution is one shape stretched across the whole range
(exactly one weight per class on the multinomial). This script asks the human
data whether a richer encoding of c_t buys anything, BEFORE any cluster time.

Three numbers per encoding, each reproducing an existing convention exactly so
the comparison is like for like:

  * `cv_logloss_train` -- the 31-class multinomial by 4-fold CV on the locked
    train split with the grid's seed and folds (`mask: punishment_valid`,
    C = 1.0): what `run_baseline_cv.py` scores, comparable to the parent's
    1.3446. `test_logloss` is the locked test split, reported not selected on.

  * `slope_artifact` -- fit on the WHOLE locked train split (what
    `inspect_best_model.py --save-best` saves) and replayed on the mechanism
    check's rows (single copy, 50 games, `punishment_valid &
    contribution_valid`), then OLS of the predicted expected punishment on c_t
    and c_{t-1}: exactly `punisher_mechanism_check.py`'s `OLS c_t`, so it is
    comparable to the parent artifact's -0.1255 and to the human -0.242. This
    is the number that predicts what the retrained artifact will do.

  * `slope_oof50` -- the same OLS from out-of-fold predictions over all 50
    games (5 folds by episode, mechanism mask): the estimate of the slope the
    model class supports, free of the locked split's particular draw.

  Plus, on the mechanism rows, `P(p>0 | c_t = 20)` and `E[p | p>0]` at 20, so
  the ceiling behaviour the parent won can be checked to survive.

Encodings compared (all sharing `prev_contribution, prev_punishment,
round_number, is_first`):

  numeric        c_t as a number (the grandparent's encoding)
  numeric+max    c_t as a number plus I(c_t = 20)   -- the PARENT's artifact
  onehot         21 dummies, one per possible contribution -- subsumes I(c=20)
  onehot+numeric the 21 dummies plus the numeric term (L2 then shrinks the
                 deviations toward the linear trend rather than toward zero)
  spline{k}      a natural cubic regression spline in c_t with k df
  spline{k}+max  the same plus I(c_t = 20) (a spline cannot bend at a point)

The one-hot is NOT the evaluation's binning: RPA is defined on {0}, 1-5, 6-10,
11-15, 16-19, {20}, and a per-value one-hot introduces no boundary at all.

    python scripts/data_analysis/punisher_contribution_encoding_check.py \
        [--out CSV]
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("DISABLE_PANDERA_IMPORT_WARNING", "True")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import log_loss  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "baselines"))

from handcrafted_grid import (  # noqa: E402
    build_feature_pool,
    load_config,
    load_episodes,
    prepare_data,
)

from aimanager.generic.data import create_torch_data  # noqa: E402

FULL = ROOT / "experiments/2group_8agent_50ep.csv"
TRAIN_CFG = ROOT / "configs/training/baselines/punishment/multinomial_ceiling.yml"
TEST_FILE = "experiments/baseline/2group_8agent_50ep_bline_test.csv"
EXPERIMENTS = ["ah_group_switching"]
SWITCH_EVERY = 4
N_LEVELS = 31
N_CONTRIB = 21
BASE = ["prev_contribution", "prev_punishment", "round_number", "is_first"]
SPLINE_DF = (4, 6)


# --------------------------------------------------------------------------- #
# contribution bases
# --------------------------------------------------------------------------- #
def _spline_basis(c, df):
    """Natural cubic regression spline basis in c_t, knots at the quantiles
    patsy picks (a fixed function of c, not fitted, so it is identical in
    every fold and on every row set)."""
    from patsy import dmatrix

    B = dmatrix(
        f"cr(x, df={df}) - 1",
        {"x": np.asarray(c, float)},
        return_type="dataframe",
    )
    return B.to_numpy(), [f"cr{df}_{i}" for i in range(B.shape[1])]


def contribution_basis(name, c):
    """(matrix [N, k], column names) for the contribution part of the design."""
    c = np.asarray(c, float)
    if name == "numeric":
        return c[:, None], ["contribution"]
    if name == "numeric+max":
        return np.column_stack([c, c == N_CONTRIB - 1]).astype(float), [
            "contribution",
            "contribution_max",
        ]
    if name == "onehot":
        return np.eye(N_CONTRIB)[np.rint(c).astype(int)], [
            f"contribution_is_{i:02d}" for i in range(N_CONTRIB)
        ]
    if name == "onehot+numeric":
        M, n = contribution_basis("onehot", c)
        return np.column_stack([M, c]), n + ["contribution"]
    if name.startswith("spline"):
        df = int(name.replace("spline", "").replace("+max", ""))
        M, n = _spline_basis(c, df)
        if name.endswith("+max"):
            return np.column_stack([M, (c == N_CONTRIB - 1).astype(float)]), n + [
                "contribution_max"
            ]
        return M, n
    raise ValueError(name)


ENCODINGS = (
    ["numeric", "numeric+max", "onehot", "onehot+numeric"]
    + [f"spline{k}" for k in SPLINE_DF]
    + [f"spline{k}+max" for k in SPLINE_DF]
)


# --------------------------------------------------------------------------- #
# fitting helpers
# --------------------------------------------------------------------------- #
def _fit(X, y, C=1.0):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc = StandardScaler().fit(X)
        m = LogisticRegression(C=C, max_iter=2000).fit(sc.transform(X), y)
    return m, sc


def _proba(m, sc, X):
    P = np.full((len(X), N_LEVELS), 1e-12)
    P[:, m.classes_] = m.predict_proba(sc.transform(X))
    return P / P.sum(1, keepdims=True)


def fold_logloss(X, y, fold):
    """Per-fold log loss (mean, se) -- the grid's own metric."""
    ll = []
    for k in sorted(set(fold.tolist())):
        tr, te = fold != k, fold == k
        m, sc = _fit(X[tr], y[tr])
        ll.append(
            log_loss(
                y[te], _proba(m, sc, X[te]), labels=list(range(N_LEVELS))
            )
        )
    return float(np.mean(ll)), float(np.std(ll) / np.sqrt(len(ll)))


def oof_proba(X, y, fold):
    P = np.zeros((len(y), N_LEVELS))
    for k in sorted(set(fold.tolist())):
        tr, te = fold != k, fold == k
        m, sc = _fit(X[tr], y[tr])
        P[te] = _proba(m, sc, X[te])
    return P


# --------------------------------------------------------------------------- #
# row sets
# --------------------------------------------------------------------------- #
def split_rows(cfg, data_file=None):
    """Base design, c_t, y and folds for a baseline split, exactly as
    prepare_data builds them (mask punishment_valid)."""
    if data_file is not None:
        cfg = {**cfg, "data": {**cfg["data"], "data_file": data_file}}
    prep = prepare_data(cfg, ROOT)
    base = np.column_stack([prep["X"][:, prep["col_of"][f]] for f in BASE])
    return base, prep["X"][:, prep["col_of"]["contribution"]], prep["y_cat"], prep[
        "fold_row"
    ]


def mechanism_rows(default_values):
    """The mechanism check's rows: single copy, 50 games, punishment_valid &
    contribution_valid, tensors built with the model's stored defaults."""
    df = pd.read_csv(FULL)
    df = df[df["experiment_name"].isin(EXPERIMENTS)]
    df = df[~df["global_group_id"].str.contains("(flipped)", regex=False)]
    data, _, _ = create_torch_data(
        df, default_values=default_values, switch_every=SWITCH_EVERY
    )
    pool = build_feature_pool(data, SWITCH_EVERY)
    m = (data["punishment_valid"] & data["contribution_valid"]).numpy()
    prev_ok = (
        data["prev_contribution_valid"].numpy() & (data["round_number"].numpy() > 0)
    )[m]
    ep = np.broadcast_to(np.arange(m.shape[0])[:, None, None], m.shape)[m]
    return (
        np.column_stack([pool[f][m] for f in BASE]),
        pool["contribution"][m],
        pool["prev_contribution"][m],
        data["punishment"].numpy()[m].astype(int),
        ep,
        prev_ok,
    )


def episode_folds(ep, n_folds=5, seed=42):
    rng = np.random.default_rng(seed)
    eps = np.unique(ep)
    fold_of = dict(zip(eps, rng.permutation(len(eps)) % n_folds))
    return np.array([fold_of[e] for e in ep])


def ols_slope(P, c, c_prev, ok):
    """punisher_mechanism_check's OLS: expected punishment on c_t, c_{t-1}."""
    ep = P @ np.arange(N_LEVELS, dtype=float)
    X = np.column_stack([np.ones(int(ok.sum())), c[ok], c_prev[ok]])
    b, *_ = np.linalg.lstsq(X, ep[ok], rcond=None)
    return float(b[1]), float(b[2])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--decompose",
        action="store_true",
        help="also split the artifact's slope deficit into mask and split",
    )
    args = ap.parse_args()

    cfg = load_config(TRAIN_CFG)
    # the defaults the artifact stores come from its own train file
    df_tr = load_episodes(cfg, ROOT)
    _, dv_tr, _ = create_torch_data(df_tr, switch_every=SWITCH_EVERY)

    tb, tc, ty, tfold = split_rows(cfg)
    eb, ec, ey, _ = split_rows(cfg, TEST_FILE)
    mb, mc, mcp, my, mep, ok = mechanism_rows(dv_tr)
    mfold = episode_folds(mep)
    print(
        f"train split {len(ty)} rows / {len(set(tfold.tolist()))} grid folds; "
        f"locked test {len(ey)} rows; mechanism {len(my)} rows / "
        f"{len(np.unique(mep))} episodes ({int(ok.sum())} with c_t-1)"
    )
    hs, hsp = ols_slope(np.eye(N_LEVELS)[my], mc, mcp, ok)
    hpos = (my > 0)[mc == 20].mean()
    hsev = my[(my > 0) & (mc == 20)].mean()
    print(
        f"human on the mechanism rows: OLS c_t {hs:+.3f} c_t-1 {hsp:+.3f}; "
        f"P(p>0|c=20) {hpos:.3f}; E[p|p>0] at 20 {hsev:.2f}\n"
    )

    rows = {}
    for name in ENCODINGS:
        Bt, cols = contribution_basis(name, tc)
        Xt = np.column_stack([Bt, tb])
        cv, se = fold_logloss(Xt, ty, tfold)

        m, sc = _fit(Xt, ty)  # the artifact's own fit: all of the train split
        Xe = np.column_stack([contribution_basis(name, ec)[0], eb])
        test_ll = log_loss(ey, _proba(m, sc, Xe), labels=list(range(N_LEVELS)))

        Xm = np.column_stack([contribution_basis(name, mc)[0], mb])
        P = _proba(m, sc, Xm)
        s_art, sp_art = ols_slope(P, mc, mcp, ok)
        pos, exp = 1.0 - P[:, 0], P @ np.arange(N_LEVELS, dtype=float)
        at20 = mc == 20
        nll = -np.log(np.clip(P[np.arange(len(my)), my], 1e-12, None)).mean()

        s_oof, _ = ols_slope(oof_proba(Xm, my, mfold), mc, mcp, ok)

        rows[name] = {
            "n_contribution_cols": len(cols),
            "cv_logloss_train": cv,
            "cv_logloss_se": se,
            "test_logloss": test_ll,
            "mech_nll": nll,
            "slope_artifact": s_art,
            "slope_artifact_c_t-1": sp_art,
            "slope_oof50": s_oof,
            "P(p>0|c=20)": pos[at20].mean(),
            "E[p|p>0]_at_20": exp[at20].sum() / pos[at20].sum(),
        }
        print(
            f"[{name:<16}] k={len(cols):>2}  CV {cv:.4f} (se {se:.4f})  "
            f"test {test_ll:.4f}  slope_art {s_art:+.4f}  "
            f"slope_oof50 {s_oof:+.4f}  P(p>0|20) {pos[at20].mean():.3f}  "
            f"E[p|p>0]@20 {exp[at20].sum() / pos[at20].sum():.2f}"
        )

    T = pd.DataFrame(rows).T
    T.loc["human", ["slope_artifact", "slope_artifact_c_t-1", "slope_oof50",
                    "P(p>0|c=20)", "E[p|p>0]_at_20"]] = [hs, hsp, hs, hpos, hsev]
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        T.to_csv(args.out)
        print(f"\nwrote {args.out}")

    if args.decompose:
        decompose(dv_tr, hs, args.out)


# --------------------------------------------------------------------------- #
# where the artifact's slope deficit actually comes from
# --------------------------------------------------------------------------- #
def decompose(dv_tr, human_slope, out=None, encoding="numeric+max", n_draws=20):
    """The parent artifact's -0.125 against the human -0.242, split into the
    two things that are NOT the encoding: the training mask and the locked
    split's particular draw.

    Rows are all the same features, the same estimator and the same replay on
    the mechanism rows; only the rows the model is FITTED on change.

      train/punishment_valid   the artifact exactly (7,345 rows, 178 of them
                               with an imputed contribution of 9 and a real
                               punishment -- the manager punished a player
                               whose contribution is missing)
      train/both_valid         the same 40 episodes with those rows dropped
      all50/punishment_valid   all 50 episodes, the artifact's mask
      all50/both_valid         all 50 episodes, the mechanism mask
      random40/both_valid      n_draws random 40-episode subsets: the sampling
                               distribution the locked split is one draw from
    """
    print("\n=== where the slope deficit comes from (encoding fixed) ===")
    df = pd.read_csv(FULL)
    df = df[df["experiment_name"].isin(EXPERIMENTS)]
    df = df[~df["global_group_id"].str.contains("(flipped)", regex=False)]
    data, _, _ = create_torch_data(
        df, default_values=dv_tr, switch_every=SWITCH_EVERY
    )
    pool = build_feature_pool(data, SWITCH_EVERY)
    pv = data["punishment_valid"].numpy()
    both = pv & data["contribution_valid"].numpy()
    y_all = data["punishment"].numpy().astype(int)
    ep_all = np.broadcast_to(np.arange(pv.shape[0])[:, None, None], pv.shape)

    tr_groups = set(
        pd.read_csv(ROOT / "experiments/baseline/2group_8agent_50ep_bline_train.csv")[
            "global_group_id"
        ].unique()
    )
    order = list(dict.fromkeys(df["global_group_id"].tolist()))
    is_tr = np.array([g in tr_groups for g in order])

    # the replay row set: the mechanism rows
    mc, mcp = pool["contribution"][both], pool["prev_contribution"][both]
    ok = (
        data["prev_contribution_valid"].numpy() & (data["round_number"].numpy() > 0)
    )[both]
    Xm = np.column_stack(
        [contribution_basis(encoding, mc)[0]]
        + [pool[f][both][:, None] for f in BASE]
    )

    def slope_of(fit_mask, label=None):
        c = pool["contribution"][fit_mask]
        X = np.column_stack(
            [contribution_basis(encoding, c)[0]]
            + [pool[f][fit_mask][:, None] for f in BASE]
        )
        m, sc = _fit(X, y_all[fit_mask])
        s, _ = ols_slope(_proba(m, sc, Xm), mc, mcp, ok)
        if label:
            print(f"  {label:<26} n={int(fit_mask.sum()):>5}  OLS c_t {s:+.4f}")
        return s

    ep_tr = is_tr[ep_all]
    rows = {
        "train/punishment_valid": slope_of(pv & ep_tr, "train/punishment_valid"),
        "train/both_valid": slope_of(both & ep_tr, "train/both_valid"),
        "all50/punishment_valid": slope_of(pv, "all50/punishment_valid"),
        "all50/both_valid": slope_of(both, "all50/both_valid"),
    }
    rng = np.random.default_rng(0)
    draws = []
    for _ in range(n_draws):
        keep = np.zeros(pv.shape[0], bool)
        keep[rng.choice(pv.shape[0], size=int(is_tr.sum()), replace=False)] = True
        draws.append(slope_of(both & keep[ep_all]))
    print(
        f"  {'random40/both_valid':<26} x{n_draws}  OLS c_t "
        f"{np.mean(draws):+.4f} +- {np.std(draws, ddof=1):.4f} "
        f"(min {np.min(draws):+.4f}, max {np.max(draws):+.4f})"
    )
    print(f"  {'human (observed)':<26}        OLS c_t {human_slope:+.4f}")
    rows["random40/both_valid_mean"] = float(np.mean(draws))
    rows["random40/both_valid_sd"] = float(np.std(draws, ddof=1))
    rows["human"] = human_slope
    if out:
        p = Path(out).with_name(Path(out).stem + "_decomposition.csv")
        pd.Series(rows, name="OLS_c_t").to_frame().to_csv(p)
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
