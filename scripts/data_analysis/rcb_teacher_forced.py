"""Step 0 of the RCB experiment: the trunk's TEACHER-FORCED punishment response.

Measures, on the human data and with the model never seeing its own draws, the
parent trunk's conditional response to punishment -- the quantity the RCB row
scores -- so the experiment can decide between its two pre-declared changes
(notes/autoresearch_log/contribution-punishment-response.md, "What the change
will be"):

  (A) within-contribution-band OLS slopes of the contribution change on the
      punishment received, over the RCB population (punishment > 0,
      contribution < 20, next-round contribution valid), in the bands
      0-4 / 5-9 / 10-14 / 15-19;
  (B) the RCB bin means over the punishment-rate bins
      (0,0.25] / (0.25,0.5] / (0.5,1] / >1, rate = punishment / (20 -
      contribution), plus the human-bin-frequency-weighted mean absolute
      discrepancy against the human bin means;
  (C) (A) and (B) recomputed on the OBSERVED human next-round contribution over
      exactly the same rows -- a self-check that the population matches the
      canonical evaluation frame's.

The model's predicted change for a stimulus row at round t is

    E[c_{t+1}] - c_t,   E[c_{t+1}] = sum_k k * P[t+1, k],

where P[t+1] is the teacher-forced predicted marginal AT ROUND t+1 -- the
model's prediction of c_{t+1} given prev_contribution = c_t and
prev_punishment = p_t. The stimulus (c_t, p_t) therefore pairs with the NEXT
row's marginal, not its own; the alignment is asserted (see `check_alignment`).

Model: the BARE trunk
artifacts/artificial_humans/group_switching_contribution_50ep_group_vnode/
model/architecture_node+edge+rnn__dataset_50ep__epochs_575.pt
and not the copula-stamped copy. The copula affects sampling only; this is a
teacher-forced conditional measurement, so the two artifacts are weight-
identical for this purpose.

Measurement only: this script trains nothing and writes no artifact.

Imports graph.py, so this runs on Raven only:
    .venv/bin/python scripts/data_analysis/rcb_teacher_forced.py [--model PT]
"""

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("DISABLE_PANDERA_IMPORT_WARNING", "True")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch as th  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "artificial_humans"))
sys.path.insert(0, str(ROOT / "scripts" / "baselines"))

# the copula calibration's loaders, imported unmodified: that module's top
# level is imports, constants and defs only, so importing it runs nothing.
# It also installs the torch_geometric.nn.meta alias the legacy pickles need.
import contribution_copula_rho as cc  # noqa: E402

from aimanager.generic.graph import GraphNetwork  # noqa: E402

DEFAULT_MODEL = (
    "artifacts/artificial_humans/group_switching_contribution_50ep_group_vnode/"
    "model/architecture_node+edge+rnn__dataset_50ep__epochs_575.pt"
)

MASK = cc.MASK  # "contribution_valid"
PMASK = "punishment_valid"

# contribution bands of the hypothesis table
BAND_EDGES = [-0.5, 4.5, 9.5, 14.5, 19.5]
BAND_LABELS = ["0-4", "5-9", "10-14", "15-19"]

# the RCB rate bins, copied from the (frozen, read-only) metric definition
RATE_EDGES = [0.0, 0.25, 0.5, 1.0, float("inf")]
RATE_LABELS = ["(0,0.25]", "(0.25,0.5]", "(0.5,1]", ">1"]

# the human reference numbers quoted in the declaration, for the (C) self-check
HUMAN_SLOPES = {"0-4": 0.1397, "5-9": 0.1038, "10-14": -0.0767, "15-19": -0.1615}
HUMAN_BIN_MEANS = np.array([0.891761, 1.340774, 1.678647, 2.014440])
HUMAN_BIN_COUNTS = np.array([1238.0, 672.0, 473.0, 277.0])


def f(x):
    """Unrounded float for the log."""
    return repr(float(x))


# --------------------------------------------------------------------------- #
# stimulus rows: observed (c_t, p_t, c_{t+1}) and the predicted E[c_{t+1}]
# --------------------------------------------------------------------------- #
def dense_predictions(rows):
    """Scatter the flat teacher-forced rows back into dense [G, A, T] arrays:
    E[c] = sum_k k * P[.., k], and the full marginal P, NaN / 0 where the row
    was masked out."""
    n_ep, n_agents, n_rounds = rows["shape"]
    n_lev = rows["P"].shape[1]
    levels = np.arange(n_lev, dtype=np.float64)
    e = np.full((n_ep, n_agents, n_rounds), np.nan, dtype=np.float64)
    p = np.zeros((n_ep, n_agents, n_rounds, n_lev), dtype=np.float64)
    e[rows["episode"], rows["agent"], rows["round"]] = rows["P"] @ levels
    p[rows["episode"], rows["agent"], rows["round"]] = rows["P"]
    return e, p


def stimulus_frame(model, data, idx, label):
    """One row per valid stimulus round t of the selected episodes.

    Observed arrays come from the SAME tensors `teacher_forced_rows` indexes,
    so observed and predicted are indexed identically. Returns the frame and
    the dense arrays the alignment check needs.
    """
    rows = cc.teacher_forced_rows(model, data, idx)
    sub = {k: v[th.as_tensor(idx)] for k, v in data.items()}
    n_ep, n_agents, n_rounds = rows["shape"]

    e_pred, p_pred = dense_predictions(rows)  # E[c_t], P[c_t] given hist < t
    contr = sub["contribution"].numpy().astype(np.float64)
    punish = sub["punishment"].numpy().astype(np.float64)
    c_ok = sub[MASK].numpy().astype(bool)
    p_ok = sub[PMASK].numpy().astype(bool)
    grp = sub["agent_group"].numpy().astype(np.int64)

    # the stimulus round t runs to T-2; t+1 is the response round
    t = slice(0, n_rounds - 1)
    t1 = slice(1, n_rounds)
    valid = c_ok[:, :, t] & c_ok[:, :, t1]  # dc needs both contributions
    g, a, r = np.nonzero(valid)

    df = pd.DataFrame(
        {
            "episode": g,
            "agent": a,
            "round": r,
            "group": grp[:, :, t][valid],
            "c_t": contr[:, :, t][valid],
            "p_t": punish[:, :, t][valid],
            "p_valid": p_ok[:, :, t][valid],
            "c_next": contr[:, :, t1][valid],
            "e_next": e_pred[:, :, t1][valid],  # aligned: predicts c_{t+1}
            "e_self": e_pred[:, :, t][valid],  # off-by-one: predicts c_t
        }
    )
    assert df["e_next"].notna().all() and df["e_self"].notna().all()
    df["dc_human"] = df["c_next"] - df["c_t"]
    df["dc_model"] = df["e_next"] - df["c_t"]
    df["dc_misaligned"] = df["e_self"] - df["c_t"]
    print(
        f"{label}: episodes={n_ep} agents={n_agents} rounds={n_rounds} "
        f"teacher-forced rows={len(rows['y'])} stimulus rows={len(df)}"
    )
    dense = dict(e=e_pred, p=p_pred, c=contr, c_ok=c_ok)
    return df, dense


def check_alignment(model, data, df, dense, label):
    """Pin down that `e_next` really is the model's prediction of c_{t+1}.

    Four handles, structural first:

      1. the tensor shift identity: prev_contribution[.., t] is
         contribution[.., t - 1] (and the same for punishment), so index t of
         the model's input really is the lagged pair of round t;
      2. the model conditions on NOTHING from the current round -- every
         contribution / punishment feature in x_encoding is a `prev_` one --
         so P[.., t] can only be a prediction of contribution[.., t];
      3. the lag profile: e_pred[.., t] is a deterministic function of its
         conditioning input, so its correlation with contribution[.., t + k]
         must PEAK at k = -1 (the input) and be lower at k = 0 (the target it
         predicts only imperfectly) and k = +1. An off-by-one indexing would
         move the peak;
      4. the requested magnitude check: over the RCB population the OLS slope
         of E[c_{t+1}] on c_t is strongly positive -- own-contribution
         stickiness, human dc-on-c slope ~ -0.20, i.e. ~ +0.80 -- while the
         misaligned E[c_t] on c_t is not the same quantity.

    The mean predictive NLL at each lag is printed as context (it is NOT an
    alignment test: the marginal is concentrated on the lagged contribution,
    so lag -1 can score well by stickiness alone).
    """
    print(f"\n--- alignment check ({label}) ---")

    # (1) structural: the shift the encoder relies on
    for name in ("contribution", "punishment"):
        cur = data[name]
        prev = data[f"prev_{name}"]
        assert th.equal(prev[:, :, 1:], cur[:, :, :-1]), f"prev_{name} misshifted"
    print("  prev_contribution / prev_punishment are the t-1 tensors        OK")

    # (2) structural: no current-round conditioning
    names = [e["name"] for e in model.x_encoding]
    illegal = [n for n in names if n in ("contribution", "punishment")]
    assert not illegal, f"model conditions on the current round: {illegal}"
    print(f"  x_encoding carries no current-round feature ({names})  OK")

    # (3) the lag profile of the predicted mean
    e, c, ok = dense["e"], dense["c"], dense["c_ok"]
    n_rounds = c.shape[2]
    prof = {}
    core = slice(2, n_rounds - 2)
    for k in (-2, -1, 0, 1, 2):
        sl = slice(2 + k, n_rounds - 2 + k)
        m = ok[:, :, core] & ok[:, :, sl] & np.isfinite(e[:, :, core])
        prof[k] = float(np.corrcoef(e[:, :, core][m], c[:, :, sl][m])[0, 1])
    print("  corr(E[c_t], c_{t+k}) by lag k:")
    for k in sorted(prof):
        tags = {-1: "  <- conditioning input", 0: "  <- target"}
        tag = tags.get(k, "")
        print(f"    k={k:+d}  {f(prof[k])}{tag}")
    peak = max(prof, key=prof.get)
    assert peak == -1, f"correlation peaks at lag {peak}, not the input lag -1"
    assert prof[-1] > prof[0] > prof[1], "lag profile is not monotone away from -1"

    # NLL context only
    p = dense["p"]
    lvl = c.astype(np.int64).clip(0, p.shape[3] - 1)
    print("  mean -log P[.., t][c_{t+k}] by lag k (context, not a test):")
    for k in (-1, 0, 1):
        sl = slice(2 + k, n_rounds - 2 + k)
        m = ok[:, :, core] & ok[:, :, sl]
        pr = np.take_along_axis(
            p[:, :, core], lvl[:, :, sl][..., None], axis=3
        )[..., 0]
        print(f"    k={k:+d}  {f(-np.log(np.clip(pr[m], 1e-12, None)).mean())}")

    # (4) the slope
    pop = rcb_population(df)
    slope_pred = ols_slope(pop["c_t"].to_numpy(), pop["e_next"].to_numpy())
    slope_mis = ols_slope(pop["c_t"].to_numpy(), pop["e_self"].to_numpy())
    slope_obs = ols_slope(pop["c_t"].to_numpy(), pop["c_next"].to_numpy())
    corr = float(np.corrcoef(pop["c_t"], pop["e_next"])[0, 1])
    print(f"  OLS slope E[c_t+1] ~ c_t (RCB pop)   {f(slope_pred)}  <- aligned")
    print(f"  OLS slope E[c_t]   ~ c_t (RCB pop)   {f(slope_mis)}  <- off by one")
    print(f"  OLS slope c_t+1    ~ c_t (RCB pop)   {f(slope_obs)}  <- human")
    print(f"  corr(c_t, E[c_t+1])                  {f(corr)}")
    assert slope_pred > 0.5, f"E[c_t+1] not sticky in c_t: {slope_pred}"
    assert corr > 0.5, f"E[c_t+1] not correlated with c_t: {corr}"


# --------------------------------------------------------------------------- #
# the two tables
# --------------------------------------------------------------------------- #
def rcb_population(df):
    """The RCB population, built exactly as the frozen metric builds it:
    punished (punishment > 0, and not a manager timeout -> the canonical
    frame's punishment is NaN there), non-full contributors (< 20), with a
    valid next-round contribution (already imposed when the frame was built).
    """
    pop = df[df["p_valid"] & (df["p_t"] > 0) & (df["c_t"] < 20)].copy()
    pop["band"] = pd.cut(pop["c_t"], BAND_EDGES, labels=BAND_LABELS)
    rate = pop["p_t"] / (20.0 - pop["c_t"])
    pop["rate_bin"] = pd.cut(rate, RATE_EDGES, labels=RATE_LABELS)
    return pop


def ols_slope(x, y):
    """Slope of the least-squares line y ~ a + b x."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    xc = x - x.mean()
    denom = float((xc * xc).sum())
    if denom == 0.0:
        return float("nan")
    return float((xc * (y - y.mean())).sum() / denom)


def band_slopes(pop, col):
    """(A): OLS slope of `col` on the punishment received, within band."""
    out = []
    for band in BAND_LABELS:
        sel = pop[pop["band"] == band]
        out.append(
            {
                "band": band,
                "n": int(len(sel)),
                "slope": ols_slope(sel["p_t"], sel[col]),
                "mean_dc": float(sel[col].mean()) if len(sel) else float("nan"),
                "mean_p": float(sel["p_t"].mean()) if len(sel) else float("nan"),
            }
        )
    return pd.DataFrame(out)


def bin_means(pop, col):
    """(B): the RCB statistic -- mean `col` per punishment-rate bin, plus the
    human-count-weighted mean absolute discrepancy against the human means."""
    g = pop.groupby("rate_bin", observed=False)[col]
    means = g.mean().reindex(RATE_LABELS).to_numpy(dtype=np.float64)
    counts = g.size().reindex(RATE_LABELS).to_numpy(dtype=np.float64)
    disc = np.abs(means - HUMAN_BIN_MEANS)
    weighted = float((HUMAN_BIN_COUNTS * disc).sum() / HUMAN_BIN_COUNTS.sum())
    tab = pd.DataFrame(
        {
            "rate_bin": RATE_LABELS,
            "n": counts,
            "mean_dc": means,
            "human_mean": HUMAN_BIN_MEANS,
            "abs_disc": disc,
            "human_weight": HUMAN_BIN_COUNTS / HUMAN_BIN_COUNTS.sum(),
        }
    )
    return tab, weighted


def show(tab):
    print(tab.to_string(index=False, float_format=lambda v: f"{v: .10f}"))


def report(pop, label):
    print(f"\n================ {label} (RCB population n={len(pop)}) ============")
    for col, name in (
        ("dc_model", "(A/B) MODEL, teacher-forced E[c_t+1] - c_t"),
        ("dc_human", "(C) HUMAN, observed c_t+1 - c_t"),
        ("dc_misaligned", "[diagnostic] MISALIGNED E[c_t] - c_t"),
    ):
        print(f"\n--- {name} ---")
        slopes = band_slopes(pop, col)
        slopes["human_slope"] = [HUMAN_SLOPES[b] for b in slopes["band"]]
        slopes["ratio_to_human"] = slopes["slope"] / slopes["human_slope"]
        print("(A) within-contribution-band slope of dc on punishment:")
        show(slopes)
        tab, weighted = bin_means(pop, col)
        print("(B) RCB bin means:")
        show(tab)
        print(f"    human-weighted mean abs discrepancy = {f(weighted)}")


def selfcheck(pop, label):
    """(C) must reproduce the declaration's human numbers to ~2 decimals."""
    ok = True
    slopes = band_slopes(pop, "dc_human").set_index("band")["slope"]
    for band, ref in HUMAN_SLOPES.items():
        got = float(slopes[band])
        if abs(got - ref) > 5e-3:
            print(f"!! SELF-CHECK FAIL {label} slope {band}: {f(got)} vs {ref}")
            ok = False
    tab, weighted = bin_means(pop, "dc_human")
    if weighted > 5e-3:
        print(f"!! SELF-CHECK FAIL {label} bin means: weighted gap {f(weighted)}")
        ok = False
    n_ref = int(HUMAN_BIN_COUNTS.sum())
    if len(pop) != n_ref:
        print(f"!! SELF-CHECK NOTE {label}: n={len(pop)} vs declaration {n_ref}")
    print(f"\nSELF-CHECK {label}: {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    t0 = time.time()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=str(ROOT / DEFAULT_MODEL))
    args = ap.parse_args()
    in_path = Path(args.model).resolve()

    model = GraphNetwork.load(str(in_path), device="cpu")
    model.eval()
    assert model.y_name == "contribution", f"not a contributor: {model.y_name}"
    print(f"model     {cc.rel(in_path)}")
    print(f"  y_name={model.y_name} y_levels={model.y_levels}")
    print(f"  x_encoding={[e['name'] for e in model.x_encoding]}")
    print(f"  copula_rho={model.copula_rho} copula_phi={model.copula_phi}")

    data, pair_id, key_to_idx, defaults = cc.load_full()
    tr_idx = cc.select_split(key_to_idx, cc.TRAIN, cc.N_TRAIN_EP)
    te_idx = cc.select_split(key_to_idx, cc.TEST, cc.N_TEST_EP)
    assert not set(tr_idx.tolist()) & set(te_idx.tolist()), "train/test overlap"
    assert not set(pair_id[tr_idx].tolist()) & set(
        pair_id[te_idx].tolist()
    ), "a flip copy of a train game sits in the holdout"
    both_idx = np.array(sorted(set(tr_idx.tolist()) | set(te_idx.tolist())))
    print(f"data      {cc.rel(cc.FULL)} (episodes={len(key_to_idx)})")
    print(f"  train {len(tr_idx)} ep, test {len(te_idx)} ep, union {len(both_idx)} ep")
    print(f"  contribution default={f(defaults['contribution'])}")

    frames, denses = {}, {}
    for label, idx in (
        ("train split (40 ep)", tr_idx),
        ("held-out split (10 ep)", te_idx),
        ("train+test, single copy (50 ep)", both_idx),
    ):
        frames[label], denses[label] = stimulus_frame(model, data, idx, label)

    union = "train+test, single copy (50 ep)"
    check_alignment(model, data, frames[union], denses[union], union)

    ok = True
    for label, df in frames.items():
        pop = rcb_population(df)
        report(pop, label)
        if label == union:
            ok = selfcheck(pop, label)

    print(f"\nwall {time.time() - t0:.1f}s")
    if not ok:
        print("!! (C) does not reproduce the declaration -- comparison INVALID")
        sys.exit(2)


if __name__ == "__main__":
    main()
