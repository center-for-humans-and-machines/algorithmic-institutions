"""Does the human punish-or-not decision carry a step at the contribution
ceiling beyond a linear term in c_t?

The simulated punishers (linear multinomial and GNN, both with `contribution`
entering numerically) punish full contributors three to four times too often
(P(p>0 | c_t = 20) 0.10-0.16 vs human 0.038) and too lightly (E[p | p>0]
3.9-5.1 vs 7.0), which is why RCC did not move in the re-baseline
(`notes/autoresearch_log/punisher-current-contribution.md`). Before adding an
indicator feature, this script fits the human decision on the same rows the
mechanism check uses (single copy, 50 games, `punishment_valid &
contribution_valid`) with the linear punisher's own features, with and
without I(c_t = 20) and I(c_t = 0), and reports:

  * the logit of P(p > 0): coefficient, cluster-robust (episode) SE and p of
    each indicator, the LR test against the linear-only model, and the
    per-episode-fold CV log loss of the binary decision;
  * the fitted P(p>0 | c_t = 20) and P(p>0 | c_t = 0) of each model against
    the observed rates (does the linear model interpolate the ceiling from
    the 15-19 band?);
  * the severity given punished: OLS of p on the same regressors over the
    p > 0 rows, with the indicator (is the full contributor, when punished,
    punished harder?);
  * the 31-class multinomial the artifact actually is, by 4-fold CV on the
    locked train split with the grid's seed and folds: log loss with and
    without the indicators.

    python scripts/data_analysis/punisher_ceiling_check.py [--out CSV]
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("DISABLE_PANDERA_IMPORT_WARNING", "True")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.api as sm  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import log_loss  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "baselines"))

from handcrafted_grid import (  # noqa: E402
    build_feature_pool,
    load_config,
    prepare_data,
)

from aimanager.generic.data import create_torch_data  # noqa: E402

FULL = ROOT / "experiments/2group_8agent_50ep.csv"
TRAIN_CFG = ROOT / "configs/training/baselines/punishment/multinomial_current_contr.yml"
EXPERIMENTS = ["ah_group_switching"]
SWITCH_EVERY = 4
N_LEVELS = 31
BASE = [
    "contribution",
    "prev_contribution",
    "prev_punishment",
    "round_number",
    "is_first",
]
MODELS = {
    "linear": BASE,
    "+max": BASE + ["contribution_max"],
    "+max+zero": BASE + ["contribution_max", "contribution_zero"],
}


def human_rows():
    df = pd.read_csv(FULL)
    df = df[df["experiment_name"].isin(EXPERIMENTS)]
    df = df[~df["global_group_id"].str.contains("(flipped)", regex=False)]
    data, _, _ = create_torch_data(df, switch_every=SWITCH_EVERY)
    pool = build_feature_pool(data, SWITCH_EVERY)
    m = (data["punishment_valid"] & data["contribution_valid"]).numpy()
    ep = np.broadcast_to(np.arange(m.shape[0])[:, None, None], m.shape)[m]
    X = pd.DataFrame({k: pool[k][m] for k in MODELS["+max+zero"]})
    y = data["punishment"].numpy()[m].astype(int)
    return X, y, ep


def logit(X, y, ep, feats):
    Z = sm.add_constant(X[feats].astype(float))
    return sm.Logit((y > 0).astype(float), Z).fit(
        disp=0, cov_type="cluster", cov_kwds={"groups": ep}
    )


def cv_binary(X, y, ep, feats, n_folds=5, seed=42):
    rng = np.random.default_rng(seed)
    eps = np.unique(ep)
    fold_of = dict(zip(eps, rng.permutation(len(eps)) % n_folds))
    fold = np.array([fold_of[e] for e in ep])
    ll = []
    for k in range(n_folds):
        tr, te = fold != k, fold == k
        sc = StandardScaler().fit(X.loc[tr, feats])
        m = LogisticRegression(C=1e6, max_iter=2000).fit(
            sc.transform(X.loc[tr, feats]), y[tr] > 0
        )
        p = m.predict_proba(sc.transform(X.loc[te, feats]))[:, 1]
        ll.append(log_loss(y[te] > 0, p, labels=[False, True]))
    return float(np.mean(ll))


def multinomial_cv(feats):
    """4-fold CV log loss of the 31-class multinomial on the locked train
    split, the grid's folds (cv.seed) and C = 1.0 -- what run_baseline_cv
    computes for the artifact's feature set."""
    cfg = load_config(TRAIN_CFG)
    prep = prepare_data(cfg, ROOT)
    cols = [prep["col_of"][f] for f in feats]
    X, y, fr = prep["X"][:, cols], prep["y_cat"], prep["fold_row"]
    ll = []
    for k in sorted(set(fr.tolist())):
        tr, te = fr != k, fr == k
        sc = StandardScaler().fit(X[tr])
        m = LogisticRegression(C=1.0, max_iter=1000).fit(sc.transform(X[tr]), y[tr])
        p = np.full((te.sum(), N_LEVELS), 1e-12)
        p[:, m.classes_] = m.predict_proba(sc.transform(X[te]))
        ll.append(log_loss(y[te], p / p.sum(1, keepdims=True), labels=range(N_LEVELS)))
    return float(np.mean(ll)), float(np.std(ll) / np.sqrt(len(ll)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    X, y, ep = human_rows()
    c = X["contribution"].to_numpy()
    pos = y > 0
    print(f"rows={len(y)} episodes={len(np.unique(ep))}")
    print(
        f"observed P(p>0): c=20 {pos[c == 20].mean():.3f} (n {int((c == 20).sum())}), "
        f"c=0 {pos[c == 0].mean():.3f} (n {int((c == 0).sum())}), "
        f"15-19 {pos[(c >= 15) & (c < 20)].mean():.3f}, "
        f"1-4 {pos[(c >= 1) & (c <= 4)].mean():.3f}"
    )
    print(
        f"observed E[p|p>0]: c=20 {y[pos & (c == 20)].mean():.2f} "
        f"(n {int((pos & (c == 20)).sum())}), 15-19 "
        f"{y[pos & (c >= 15) & (c < 20)].mean():.2f}, c=0 {y[pos & (c == 0)].mean():.2f}"
    )

    rows = {}
    fits = {name: logit(X, y, ep, feats) for name, feats in MODELS.items()}
    base_llf = fits["linear"].llf
    print("\n=== logit P(p>0), cluster-robust SE by episode ===")
    for name, fit in fits.items():
        feats = MODELS[name]
        r = {"cv_binary_logloss": cv_binary(X, y, ep, feats)}
        for ind in ("contribution_max", "contribution_zero"):
            if ind in feats:
                r[f"{ind}_coef"] = fit.params[ind]
                r[f"{ind}_se"] = fit.bse[ind]
                r[f"{ind}_p"] = fit.pvalues[ind]
        if name != "linear":
            lr = 2 * (fit.llf - base_llf)
            df = len(feats) - len(BASE)
            r["lr_vs_linear"], r["lr_p"] = lr, stats.chi2.sf(lr, df)
        ph = fit.predict(sm.add_constant(X[feats].astype(float)))
        r["fit_P(p>0|c=20)"] = ph[c == 20].mean()
        r["fit_P(p>0|c=0)"] = ph[c == 0].mean()
        r["fit_P(p>0|15-19)"] = ph[(c >= 15) & (c < 20)].mean()
        r["llf"] = fit.llf
        rows[name] = r
        print(f"\n[{name}] llf={fit.llf:.1f}")
        print(
            fit.summary2()
            .tables[1][["Coef.", "Std.Err.", "z", "P>|z|"]]
            .to_string(float_format=lambda v: f"{v:.3f}")
        )
        print(
            f"  fitted P(p>0|c=20)={r['fit_P(p>0|c=20)']:.3f}  "
            f"P(p>0|c=0)={r['fit_P(p>0|c=0)']:.3f}  "
            f"P(p>0|15-19)={r['fit_P(p>0|15-19)']:.3f}  "
            f"CV binary log loss={r['cv_binary_logloss']:.4f}"
            + (
                f"  LR vs linear={r['lr_vs_linear']:.1f} (p={r['lr_p']:.2e})"
                if name != "linear"
                else ""
            )
        )

    print("\n=== severity | p>0: OLS of p on the regressors (cluster SE) ===")
    for name, feats in MODELS.items():
        Z = sm.add_constant(X.loc[pos, feats].astype(float))
        fit = sm.OLS(y[pos].astype(float), Z).fit(
            cov_type="cluster", cov_kwds={"groups": ep[pos]}
        )
        line = f"[{name}] contribution {fit.params['contribution']:+.3f}"
        for ind in ("contribution_max", "contribution_zero"):
            if ind in feats:
                line += (
                    f"  {ind} {fit.params[ind]:+.2f} (se {fit.bse[ind]:.2f}, "
                    f"p {fit.pvalues[ind]:.3f})"
                )
                rows[name][f"sev_{ind}_coef"] = fit.params[ind]
                rows[name][f"sev_{ind}_p"] = fit.pvalues[ind]
        print(line)

    print("\n=== 31-class multinomial, 4-fold CV on the train split (grid folds) ===")
    for name, feats in MODELS.items():
        mean, se = multinomial_cv(feats)
        rows[name]["cv_multinomial_logloss"] = mean
        print(f"[{name}] {mean:.4f} (se {se:.4f})")

    T = pd.DataFrame(rows).T
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        T.to_csv(args.out)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
