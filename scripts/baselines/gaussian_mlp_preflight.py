"""Behavioral pre-flight for the gaussian_mlp contribution candidate vs the
linear gaussian incumbent (autoresearch step 10, PR #148 pattern).

Teacher-forced on the REAL human rows (no simulation): both bundles are scored
on the same prepared feature matrix -- each on ITS OWN feature list + scaler --
over the train split and the locked test split (`_train` -> `_test`). What it
answers, per the log's Declaration:

  * fit          -- continuous Gaussian NLL and the 21-way binned CE that the
                    sim actually samples from (gaussian_regressor.binned_logloss)
  * sigma(x)     -- summary and breakout by state; the hypothesis is that the
                    MLP's spread is SHARP at sticky states (prev 0 / 20) and
                    wide mid-range, where a linear log-sigma head is near-flat
  * repeat spike -- the implied P(round(y) == prev_contribution) under
                    N(mu(x), sigma(x)) with the binned convention, against the
                    empirical exact-repeat rate (~0.44)
  * where the CE margin comes from, and what the continuous-NLL tail rows cost
    once discretized
  * conformity   -- the two partial derivatives that govern how an
                    own-vs-group deviation evolves. Writing mu ~ a*own_prev +
                    b*group_prev, a = d mu / d prev_contribution at a fixed
                    group level and b = d mu / d prev_contribution_mean_group
                    at a fixed own level.

                    Read them as follows. The deviation (own - group)
                    CONTRACTS iff a < 1: that, not a negative a, is
                    conditional cooperation -- a < 0 would mean a player who
                    contributed more last round contributes less next round
                    regardless of the group, which is contrarian, not
                    conformist, and no sane fit produces it. The per-round
                    contraction is 1 - a.

                    a + b is the quantity that governs CG. Averaging mu over
                    a group's members sends own_prev -> the group mean, so a
                    group's mean contribution evolves as an AR(1) with
                    coefficient a + b: below 1 the group means are mean
                    reverting and stay together, at 1 they random-walk apart.
                    A candidate whose a + b is closer to 1 than the incumbent's
                    lets group means drift further before reverting, which is
                    mechanically PR #151's CG-explosion mode.

Deterministic (closed form via scipy.stats.norm, no sampling).

Runs locally (CPU torch, no PyG):
    .venv/bin/python scripts/baselines/gaussian_mlp_preflight.py \
        [--config configs/training/baselines/contribution/gaussian_mlp.yml] \
        [--candidate PATH] [--incumbent PATH]
"""

import argparse
import copy
import os
import sys
from pathlib import Path

os.environ.setdefault("DISABLE_PANDERA_IMPORT_WARNING", "True")
import numpy as np  # noqa: E402
from scipy.stats import norm  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "baselines"))

from handcrafted_grid import load_config, prepare_data  # noqa: E402
from gaussian_regressor import binned_logloss  # noqa: E402

DEFAULT_CFG = ROOT / "configs/training/baselines/contribution/gaussian_mlp.yml"
CANDIDATE = ROOT / "artifacts/baselines/contribution_gaussian_mlp_best.joblib"
INCUMBENT = ROOT / "artifacts/baselines/contribution_gaussian_best.joblib"

K = 21  # contribution levels 0..20
TAIL_NLL = 10.0  # per-row continuous NLL above this = tail row
# a "repeated last round" indicator needs the agent's t-2 contribution; these
# are the only pool names that could carry it (see the printed NOTE).
LAG2_FEATS = ("prev2_contribution", "prev_prev_contribution", "contribution_lag2")
# the conformity read needs both halves of the own-vs-group deviation
DEV_FEATS = ("prev_contribution", "prev_contribution_mean_group")
CONFORM_H = 1.0  # forward step, RAW contribution units
CONFORM_H_C = 0.5  # +-step of the central-difference estimate


def bin_probs(mu, sigma, k=K):
    """[N, k] level probabilities of N(mu, sigma) under binned_logloss's
    convention: level bins +-0.5, tails folded into levels 0 and k-1."""
    mu = np.asarray(mu, float).reshape(-1)
    sigma = np.broadcast_to(np.asarray(sigma, float).reshape(-1), mu.shape)
    ks = np.arange(k)
    P = norm.cdf((ks + 0.5 - mu[:, None]) / sigma[:, None]) - norm.cdf(
        (ks - 0.5 - mu[:, None]) / sigma[:, None]
    )
    P[:, 0] = norm.cdf((0.5 - mu) / sigma)
    P[:, -1] = 1.0 - norm.cdf((k - 1.5 - mu) / sigma)
    P = np.clip(P, 1e-12, None)
    return P / P.sum(1, keepdims=True)


def _rel(path):
    p = Path(path)
    return str(p.relative_to(ROOT)) if p.is_absolute() and ROOT in p.parents else path


def levels(y, k=K):
    return np.clip(np.rint(np.asarray(y, float).reshape(-1)), 0, k - 1).astype(int)


def score(bundle, prep):
    """Teacher-forced predictions of one bundle on one prepared split."""
    cols = [prep["col_of"][f] for f in bundle["features"]]
    X = bundle["scaler"].transform(prep["X"][:, cols])
    m = bundle["estimator"]
    mu, sigma = m.predict(X), np.asarray(m.predict_std(X), float).reshape(-1)
    y = prep["y_cont"]
    P = bin_probs(mu, sigma)
    idx = np.arange(len(y))
    return dict(
        mu=mu,
        sigma=sigma,
        nll_row=0.5 * (np.log(2 * np.pi * sigma**2) + (y - mu) ** 2 / sigma**2),
        ll_row=-np.log(P[idx, levels(y)]),
        p_repeat=P[idx, levels(prep["prev"])],
        ce=lambda s: binned_logloss(mu[s], y[s], sigma[s], K),
    )


def table(header, rows, indent="  "):
    cells = [[str(c) for c in header]] + [[str(c) for c in r] for r in rows]
    w = [max(len(r[i]) for r in cells) for i in range(len(header))]
    for j, r in enumerate(cells):
        line = r[0].ljust(w[0]) + "".join(f"  {r[i]:>{w[i]}}" for i in range(1, len(r)))
        print(indent + line)
        if j == 0:
            print(indent + "-" * len(line))
    print()


def groups_of(prep):
    """[(label, row mask)] the breakouts are conditioned on."""
    lvl = levels(prep["prev"])
    return [
        ("prev = 0", lvl == 0),
        ("prev 1-19", (lvl > 0) & (lvl < K - 1)),
        ("prev = 20", lvl == K - 1),
    ]


def conformity_slope(bundle, prep, h, central=False, feature="prev_contribution"):
    """d mu / d `feature`, every other feature held fixed.

    `feature="prev_contribution"` gives a (own persistence, the group level
    held fixed -- i.e. the derivative along the own-vs-group deviation);
    `feature="prev_contribution_mean_group"` gives b (the conformity pull, the
    agent's own level held fixed). The step is applied in RAW feature space and
    then re-standardized through the bundle's own scaler (equivalent to scaling
    h by scaler.scale_ for that column). Nothing is refit, nothing is sampled."""
    cols = [prep["col_of"][f] for f in bundle["features"]]
    j = bundle["features"].index(feature)
    raw = np.asarray(prep["X"][:, cols], float)
    est = bundle["estimator"]

    def mu_at(shift):
        Z = raw.copy()
        Z[:, j] += shift
        return np.asarray(est.predict(bundle["scaler"].transform(Z)), float)

    if central:
        return (mu_at(h) - mu_at(-h)) / (2.0 * h)
    return (mu_at(h) - mu_at(0.0)) / h


def conformity_schemes(bundle, prep):
    """{scheme label: per-row slopes} for one bundle, or {} when the bundle's
    feature list cannot express the own-vs-group deviation."""
    if not set(DEV_FEATS) <= set(bundle["features"]):
        return {}
    return {
        f"forward h=+{CONFORM_H:g}": conformity_slope(bundle, prep, CONFORM_H),
        f"central h=+-{CONFORM_H_C:g}": conformity_slope(
            bundle, prep, CONFORM_H_C, central=True
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_CFG))
    ap.add_argument("--candidate", default=str(CANDIDATE))
    ap.add_argument("--incumbent", default=str(INCUMBENT))
    args = ap.parse_args()

    import joblib

    cfg = load_config(Path(args.config))
    models = [
        ("candidate", joblib.load(args.candidate)),
        ("incumbent", joblib.load(args.incumbent)),
    ]

    splits = {}
    for split in ("train", "test"):
        c = copy.deepcopy(cfg)
        if split == "test":
            c["data"]["data_file"] = cfg["data"]["data_file"].replace("_train", "_test")
        prep = prepare_data(c, ROOT)
        prep["prev"] = prep["X"][:, prep["col_of"]["prev_contribution"]]
        prep["file"] = c["data"]["data_file"]
        splits[split] = (prep, {n: score(b, prep) for n, b in models})

    print("gaussian_mlp pre-flight -- teacher-forced on human rows, no simulation")
    print(f"  config    {_rel(args.config)}")
    for (name, b), path in zip(models, (args.candidate, args.incumbent)):
        extra = f" hidden={b['hidden']}" if "hidden" in b else ""
        print(
            f"  {name} {b['model']}  {len(b['features'])} features{extra}  "
            f"{_rel(path)}"
        )
    for split, (prep, _) in splits.items():
        print(f"  {split:<9} {prep['file']}  ({len(prep['y_cont'])} rows)")
    print()

    print("[1] fit (each model on its own feature set); delta = cand - inc")
    rows = []
    for split, (prep, sc) in splits.items():
        all_rows = np.ones(len(prep["y_cont"]), bool)
        for metric, get in (
            ("continuous NLL", lambda s: float(s["nll_row"].mean())),
            ("binned CE (21-way)", lambda s: s["ce"](all_rows)),
        ):
            c, i = get(sc["candidate"]), get(sc["incumbent"])
            rows.append([split, metric, f"{c:.4f}", f"{i:.4f}", f"{c - i:+.4f}"])
    table(["split", "metric", "candidate", "incumbent", "delta"], rows)

    print("[2] sigma(x) summary")
    rows = [
        [
            split,
            name,
            f"{s['sigma'].mean():.4f}",
            f"{s['sigma'].std():.4f}",
            f"{s['sigma'].min():.4f}",
            f"{s['sigma'].max():.4f}",
        ]
        for split, (_, sc) in splits.items()
        for name, s in sc.items()
    ]
    table(["split", "model", "mean", "sd", "min", "max"], rows)

    print("[3] mean sigma(x) by prev_contribution state")
    rows = [
        [
            split,
            label,
            int(m.sum()),
            f"{sc['candidate']['sigma'][m].mean():.4f}",
            f"{sc['incumbent']['sigma'][m].mean():.4f}",
        ]
        for split, (prep, sc) in splits.items()
        for label, m in groups_of(prep)
    ]
    table(["split", "state", "n", "candidate", "incumbent"], rows)
    lag2 = sorted(set(splits["train"][0]["col_of"]) & set(LAG2_FEATS))
    if not lag2:
        print(
            "  NOTE: prepare_data's pool has prev_contribution but no t-2 own\n"
            f"  contribution (checked {', '.join(LAG2_FEATS)}),\n"
            "  so a 'repeated last round' indicator cannot be built from X --\n"
            "  breakouts are by prev_contribution state only.\n"
        )

    print("[4] implied exact-repeat mass  P(round(y) == prev_contribution)")
    rows = [
        [
            split,
            len(prep["y_cont"]),
            f"{sc['candidate']['p_repeat'].mean():.4f}",
            f"{sc['incumbent']['p_repeat'].mean():.4f}",
            f"{np.mean(levels(prep['y_cont']) == levels(prep['prev'])):.4f}",
        ]
        for split, (prep, sc) in splits.items()
    ]
    table(["split", "n", "candidate", "incumbent", "empirical"], rows)

    print("[5] implied vs empirical repeat mass by prev_contribution state")
    rows = [
        [
            split,
            label,
            int(m.sum()),
            f"{sc['candidate']['p_repeat'][m].mean():.4f}",
            f"{sc['incumbent']['p_repeat'][m].mean():.4f}",
            f"{np.mean(levels(prep['y_cont'])[m] == levels(prep['prev'])[m]):.4f}",
        ]
        for split, (prep, sc) in splits.items()
        for label, m in groups_of(prep)
    ]
    table(["split", "state", "n", "candidate", "incumbent", "empirical"], rows)

    print("[6] 21-way binned CE by prev_contribution state; delta = cand - inc")
    rows = []
    for split, (prep, sc) in splits.items():
        for label, m in groups_of(prep):
            c, i = sc["candidate"]["ce"](m), sc["incumbent"]["ce"](m)
            rows.append(
                [split, label, int(m.sum()), f"{c:.4f}", f"{i:.4f}", f"{c - i:+.4f}"]
            )
    table(["split", "state", "n", "candidate", "incumbent", "delta"], rows)

    print(f"[7] tail risk: rows with that model's continuous NLL > {TAIL_NLL:g}")
    rows = []
    for split, (prep, sc) in splits.items():
        for name, s in sc.items():
            m = s["nll_row"] > TAIL_NLL
            if not m.any():
                rows.append([split, name, 0, "-", "-", "-"])
                continue
            rows.append(
                [
                    split,
                    name,
                    int(m.sum()),
                    f"{s['nll_row'][m].mean():.2f}",
                    f"{sc['candidate']['ll_row'][m].mean():.4f}",
                    f"{sc['incumbent']['ll_row'][m].mean():.4f}",
                ]
            )
    table(
        [
            "split",
            "tail rows of",
            "n",
            "own cont NLL",
            "cand binned LL",
            "inc binned LL",
        ],
        rows,
    )

    prep_te = splits["test"][0]
    print("[8] conformity read -- d mu / d (own-vs-group deviation), TEST rows")
    print("  route: step applied in RAW feature space (prev_contribution += h,")
    print("  prev_contribution_mean_group held fixed), then re-standardized")
    print("  through each bundle's own scaler. No refit, no sampling.")
    slopes = {n: conformity_schemes(b, prep_te) for n, b in models}
    for name, b in models:
        if not slopes[name]:
            missing = [f for f in DEV_FEATS if f not in b["features"]]
            print(f"  {name}: SKIPPED -- feature list lacks {', '.join(missing)}")
    print()
    rows = [
        [
            name,
            scheme,
            f"{s.mean():+.4f}",
            f"{np.median(s):+.4f}",
            f"{np.mean(s < 1.0):.4f}",
        ]
        for name, _ in models
        for scheme, s in slopes[name].items()
    ]
    table(["model", "scheme", "mean a", "median a", "frac a<1"], rows)

    # b = conformity pull, and a + b = the AR(1) coefficient of a group's mean
    # contribution -- the quantity CG actually responds to.
    print()
    print("  a = d mu / d own_prev (group fixed); b = d mu / d group (own fixed)")
    print("  deviation contracts iff a < 1 (per-round contraction 1 - a);")
    print("  group means evolve as AR(1) with coefficient a + b -> CG risk.")
    ab = {}
    rows = []
    for name, b in models:
        if not slopes[name]:
            continue
        a_s = slopes[name][f"forward h=+{CONFORM_H:g}"]
        b_s = conformity_slope(
            b, prep_te, CONFORM_H, feature="prev_contribution_mean_group"
        )
        ab[name] = (a_s.mean(), b_s.mean())
        rows.append(
            [
                name,
                f"{a_s.mean():+.4f}",
                f"{b_s.mean():+.4f}",
                f"{1.0 - a_s.mean():+.4f}",
                f"{a_s.mean() + b_s.mean():+.4f}",
            ]
        )
    table(["model", "a (own)", "b (group)", "contraction 1-a", "a+b (CG)"], rows)

    fwd = f"forward h=+{CONFORM_H:g}"
    print(f"[9] conformity read by prev_contribution state (TEST, {fwd})")
    rows = []
    for label, m in groups_of(prep_te):
        row = [label, int(m.sum())]
        for name, _ in models:
            s = slopes[name].get(fwd)
            row += (
                ["-", "-"]
                if s is None
                else [f"{s[m].mean():+.4f}", f"{np.mean(s[m] < 1.0):.4f}"]
            )
        rows.append(row)
    table(
        ["state", "n", "cand a", "cand frac a<1", "inc a", "inc frac a<1"],
        rows,
    )

    cand = slopes["candidate"].get(fwd)
    if cand is None:
        print("CONFORMITY: not measurable -- candidate lacks the deviation feats")
        return
    a_c, b_c = ab["candidate"]
    if a_c >= 1.0:
        print(f"CONFORMITY: FAIL -- deviations do not contract (a = {a_c:+.4f} >= 1)")
    else:
        print(
            f"CONFORMITY: pull toward group (a = {a_c:+.4f} < 1, "
            f"contraction {1.0 - a_c:+.4f}/round, b = {b_c:+.4f})"
        )
    if "incumbent" in ab:
        a_i, b_i = ab["incumbent"]
        d = (a_c + b_c) - (a_i + b_i)
        verdict = "CG RISK" if d > 0 else "CG ok"
        print(
            f"{verdict}: group-mean AR(1) a+b = {a_c + b_c:+.4f} vs incumbent "
            f"{a_i + b_i:+.4f} (delta {d:+.4f}); closer to 1 = group means "
            "drift further before reverting"
        )


if __name__ == "__main__":
    main()
