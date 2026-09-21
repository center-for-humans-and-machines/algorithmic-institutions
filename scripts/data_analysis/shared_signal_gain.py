"""Does the contribution model under-respond to a SHARED, group-level signal?

Every teacher-forced verification so far measured a player's response to their
OWN punishment. This script measures the other kind of input: how strongly a
player's next contribution follows **their own group's previous level**, the
quantity that decides whether a group moves as a body.

The estimand is ``b`` in

    c[i, t+1] = a * c[i, t] + b * m[-i, g(i), t] + k + e,

where ``m[-i, g, t]`` is the **leave-one-out** mean of the valid contributions
of the other members of i's group at round t. Leave-one-out, because including
i's own c[i, t] in the group mean would measure i's own stickiness; ``a`` is in
the regression, because the own lag and the group's level are correlated and a
specification without it splits the credit arbitrarily.

Four arms, all on the same estimand:

* **human** -- ``experiments/2group_8agent_50ep.csv`` through the evaluation
  suite's canonical single-copy frame (``convert.load_human``);
* **model, teacher-forced** (route a) -- the same regression with y replaced by
  the model's conditional expectation ``E[c[i, t+1] | human history]``. This is
  like-for-like by iterated projections: the regressors are history-measurable,
  so the population projection of the realised c is the projection of E[c|H];
* **model, interventional** (route b) -- shift a group's recent level by fixed
  amounts and read how far the predicted expectation moves (``probe``). The
  same weight through a different instrument, and it also separates the own
  channel from the shared one by shifting only one of them at a time;
* **simulation, realised** -- the same regression on a finished simulation's
  own trajectories. By the same iterated-projection argument this *is* the
  teacher-forced coefficient at the simulation's own states, so it needs no
  forward pass; it says whether the following weakens in the closed loop.

Noise floor: PR #195's five reseed replicas of the same contributor plus the
shipped sixth draw. Their simulations give the run-to-run spread of the
realised coefficient (``--seed-dir``); their bare artifacts give the spread of
the teacher-forced one (``--seed-model-dir``). A human-to-model difference
smaller than that spread is not a finding.

Stages
------
    local (macOS, no PyG):
        python scripts/data_analysis/shared_signal_gain.py analyse \\
            --seed-dir <dir with PR #195's six per_round.parquet>

    Raven (needs torch_geometric):
        python scripts/data_analysis/shared_signal_gain.py tf \\
            [--seed-model-dir <dir with seed_{1..5}.pt>]
        python scripts/data_analysis/shared_signal_gain.py probe \\
            [--seed-model-dir <dir>] [--one-round]

``tf`` and ``probe`` write into the output directory; ``analyse`` picks up
whatever of them is present. Nothing is trained and no model is written.

Outputs under plots/data_analysis/evaluation/shared_signal_gain/.
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("DISABLE_PANDERA_IMPORT_WARNING", "True")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OUT = ROOT / "plots/data_analysis/evaluation/shared_signal_gain"
SIM = ROOT / "plots/simulation"
HUMAN_CSV = ROOT / "experiments/2group_8agent_50ep.csv"
SKIP = "23_2g8a_contr_stimulus_skip_self_gnncopar1_contr_gnn_switch"

TRUNK = (
    "artifacts/artificial_humans"
    "/group_switching_contribution_50ep_vnode_stimulus_skip"
    "/model/architecture_node+edge+rnn__dataset_50ep__epochs_575.pt"
)

# The simulation arms, identical to PR #202's: the serving-fixed frontier stack
# with the contribution copula off (the honest arm) and on, plus the two
# pre-serving-fix parents as a guard that nothing is an artefact of that fix.
ARMS = {
    "sim_noise_off": SIM / "23_2g8a_sim_timeout_rho0/per_round.parquet",
    "sim_noise_on": SIM / (SKIP + "_simtimeout/per_round.parquet"),
    "sim_noise_off_prefix": SIM / "23_2g8a_sim_timeout_rho0_base/per_round.parquet",
    "sim_noise_on_prefix": SIM / (SKIP + "_timeout/per_round.parquet"),
}
SEED_NAMES = [f"seed_{s}" for s in range(1, 6)] + ["shipped"]
MODELS = ["trunk"] + [f"seed_{s}" for s in range(1, 6)]

BLOCKS = [("1-8", 0, 8), ("9-16", 8, 16), ("17-24", 16, 24)]
CELL = ["episode_id", "round_number", "group_id"]
FWD = ["episode_id", "round_number", "group_next"]
ROUND = ["episode_id", "round_number"]
PLAYER = ["episode_id", "participant_code"]
BOOT = 2000
DELTAS = [-6, -4, -2, 2, 4, 6]
ONE_ROUND_DELTAS = [-4, 4]
ONE_ROUND_STIMULI = [2, 6, 10, 14, 18]
K = 21  # contribution levels


# --------------------------------------------------------------------------- #
# the panel
# --------------------------------------------------------------------------- #
def _loo(d, keys, num, den):
    """Leave-one-out mean of `num` over the cell `keys`, excluding the row."""
    g = d.groupby(keys, dropna=False)
    n = g[den].transform("sum") - d[den]
    s = g[num].transform("sum") - d[num]
    return s.where(n > 0) / n.where(n > 0), n


def panel(frame, extra_y=()):
    """One row per (episode, player, round t) that has a valid contribution at
    t and at t+1, with the regressors and the sample flags.

    c_own          c[i, t], the player's own lagged contribution
    peer_mean      leave-one-out mean of the VALID contributions of the other
                   members of i's group AT ROUND t -- the level i just watched
    peer_mean_fwd  the same mean over the members of i's group at round t+1,
                   i.e. the group i is in when it acts; identical to peer_mean
                   on every transition that does not move anyone
    other_mean     mean of the valid contributions of the other group at t
    n_peers        how many peers peer_mean is over (0..7)
    y              c[i, t+1], the response
    stable         neither group's member set changes between t and t+1
    """
    d = frame.sort_values(PLAYER + ["round_number"]).copy()
    d["_c"] = d["contribution"].fillna(0.0)
    d["_ok"] = d["contribution"].notna().astype(float)
    d["_p"] = d["punishment"].fillna(0.0)
    d["_pok"] = d["punishment"].notna().astype(float)

    g = d.groupby(CELL)
    r = d.groupby(ROUND)
    for src, dst in [("_c", "c"), ("_ok", "ok")]:
        d[f"g_{dst}"] = g[src].transform("sum")
        d[f"r_{dst}"] = r[src].transform("sum")
    d["peer_mean"], n_peer = _loo(d, CELL, "_c", "_ok")
    d["peer_p_mean"], _ = _loo(d, CELL, "_p", "_pok")
    d["n_peers"] = n_peer.astype(int)
    oth_n = d["r_ok"] - d["g_ok"]
    oth_s = d["r_c"] - d["g_c"]
    d["n_other"] = oth_n.astype(int)
    d["other_mean"] = (oth_s.where(oth_n > 0) / oth_n.where(oth_n > 0)).astype(float)

    # rosters are MEMBERSHIP, over all rows: a human timeout NaNs the
    # contribution but leaves the player in the group, and counting that as a
    # membership change would put ~9% of the human transitions in the wrong
    # bucket and none of the simulation's (PR #202 section 2)
    rost = g["participant_code"].agg(frozenset).rename("roster")
    wide = rost.unstack("group_id")
    nxt = wide.groupby("episode_id").shift(-1)
    chg = pd.Series(False, index=wide.index)
    for col in wide.columns:
        chg |= wide[col].ne(nxt[col]) & nxt[col].notna()
    d = d.merge(chg.rename("roster_change").reset_index(), on=ROUND, how="left")

    by = d.groupby(PLAYER)
    d["y"] = by["contribution"].shift(-1)
    d["round_next"] = by["round_number"].shift(-1)
    d["group_next"] = by["group_id"].shift(-1)
    for name in extra_y:
        d[f"{name}_next"] = by[name].shift(-1)
    d["peer_mean_fwd"], n_fwd = _loo(d, FWD, "_c", "_ok")
    d["n_peers_fwd"] = n_fwd.fillna(0).astype(int)

    d = d[d["round_next"] == d["round_number"] + 1].copy()
    d["c_own"] = d["contribution"]
    d["p_own"] = d["punishment"]
    d["ego_moves"] = d["group_next"].ne(d["group_id"])
    d["stable"] = ~d["roster_change"].fillna(False)
    d["block"] = "other"
    for name, lo, hi in BLOCKS:
        d.loc[(d["round_number"] >= lo) & (d["round_number"] < hi), "block"] = name
    keep = [
        "episode_id", "participant_code", "round_number", "group_id", "block",
        "c_own", "p_own", "peer_mean", "peer_mean_fwd", "peer_p_mean",
        "other_mean", "n_peers", "n_peers_fwd", "n_other", "y", "stable",
        "ego_moves",
    ]  # fmt: skip
    keep += [f"{n}_next" for n in extra_y]
    return d.dropna(subset=["y", "c_own"])[keep].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# the regression
# --------------------------------------------------------------------------- #
def fit(d, xcols=("c_own", "peer_mean"), ycol="y", absorb=None):
    """OLS with standard errors clustered by game; `absorb` demeans within a
    fixed effect (and drops the intercept)."""
    d = d.dropna(subset=[ycol, *xcols])
    intercept = absorb is None
    if absorb:
        d = d.copy()
        cols = [ycol, *xcols]
        d[cols] = d[cols] - d.groupby(absorb)[cols].transform("mean")
    X = d[list(xcols)].to_numpy(float)
    if intercept:
        X = np.column_stack([np.ones(len(X)), X])
    y = d[ycol].to_numpy(float)
    cl = d["episode_id"].to_numpy()
    if len(y) < 20:
        return None
    xtx_inv = np.linalg.pinv(X.T @ X)
    beta = xtx_inv @ (X.T @ y)
    resid = y - X @ beta
    meat = np.zeros((X.shape[1], X.shape[1]))
    for e in np.unique(cl):
        m = cl == e
        s = X[m].T @ resid[m]
        meat += np.outer(s, s)
    ng = len(np.unique(cl))
    n, k = X.shape
    scale = ng / max(ng - 1, 1) * (n - 1) / max(n - k, 1)
    se = np.sqrt(np.clip(np.diag(xtx_inv @ meat @ xtx_inv * scale), 0, None))
    off = 1 if intercept else 0
    out = dict(n=int(n), games=int(ng), sigma=float(np.sqrt(np.mean(resid**2))))
    if intercept:
        out["const"] = float(beta[0])
    for i, nm in enumerate(xcols):
        out[nm] = float(beta[off + i])
        out[f"{nm}_se"] = float(se[off + i])
        out[f"{nm}_lo"] = float(beta[off + i] - 1.96 * se[off + i])
        out[f"{nm}_hi"] = float(beta[off + i] + 1.96 * se[off + i])
    return out


def boot_coefs(d, ycols, xcols=("c_own", "peer_mean"), draws=BOOT, seed=0):
    """Game-cluster bootstrap of the slopes, with the SAME resampled games used
    for every y in `ycols`. Paired by construction, which is what the human
    versus teacher-forced comparison needs: those two share the games, the rows
    and the regressors and differ only in the response."""
    d = d.dropna(subset=[*ycols, *xcols])
    X = np.column_stack([np.ones(len(d)), d[list(xcols)].to_numpy(float)])
    ys = [d[c].to_numpy(float) for c in ycols]
    eps = d["episode_id"].to_numpy()
    uniq = np.unique(eps)
    xtx, xty = {}, {}
    for e in uniq:
        m = eps == e
        xtx[e] = X[m].T @ X[m]
        xty[e] = [X[m].T @ y[m] for y in ys]
    rng = np.random.default_rng(seed)
    k = X.shape[1]
    out = np.empty((draws, len(ycols), k - 1))
    for i in range(draws):
        pick = uniq[rng.integers(0, len(uniq), len(uniq))]
        a = np.zeros((k, k))
        bs = [np.zeros(k) for _ in ycols]
        for e in pick:
            a += xtx[e]
            for j in range(len(ycols)):
                bs[j] += xty[e][j]
        ai = np.linalg.pinv(a)
        for j in range(len(ycols)):
            out[i, j] = (ai @ bs[j])[1:]
    return out


def _quant(diff):
    return dict(
        diff_lo=float(np.quantile(diff, 0.025)),
        diff_hi=float(np.quantile(diff, 0.975)),
        p_two_sided=float(2 * min((diff < 0).mean(), (diff > 0).mean())),
    )


def paired_diff(d, y_a, y_b, xcols=("c_own", "peer_mean"), seed=1, label=""):
    """Bootstrap of the difference in the peer-mean slope, same games in both."""
    fa, fb = fit(d, xcols, y_a), fit(d, xcols, y_b)
    if fa is None or fb is None:
        return None
    b = boot_coefs(d, [y_a, y_b], xcols, seed=seed)
    j = list(xcols).index("peer_mean")
    return dict(
        sample=label,
        human=fa["peer_mean"],
        model=fb["peer_mean"],
        diff=fa["peer_mean"] - fb["peer_mean"],
        n_rows=fa["n"],
        **_quant(b[:, 0, j] - b[:, 1, j]),
    )


def unpaired_diff(da, db, xcols=("c_own", "peer_mean"), seed=1):
    """Games resampled whole and independently in two different arms."""
    ba = boot_coefs(da, ["y"], xcols, seed=seed)[:, 0, 1]
    bb = boot_coefs(db, ["y"], xcols, seed=seed + 1)[:, 0, 1]
    return _quant(ba - bb)


# --------------------------------------------------------------------------- #
# the tables
# --------------------------------------------------------------------------- #
SAMPLES = [
    ("all", lambda d: d),
    ("stable", lambda d: d[d["stable"]]),
    ("change", lambda d: d[~d["stable"]]),
]
BLOCK_NAMES = ["pooled"] + [b[0] for b in BLOCKS]


def _cells(d):
    for sample, sel in SAMPLES:
        sub = sel(d)
        for block in BLOCK_NAMES:
            x = sub if block == "pooled" else sub[sub["block"] == block]
            yield sample, block, x


def coef_rows(arm, d, ycol="y"):
    rows = []
    for sample, block, x in _cells(d):
        f = fit(x, ycol=ycol)
        if f is not None:
            rows.append(dict(arm=arm, sample=sample, block=block, **f))
    return rows


SPECS = [
    ("headline: own lag + LOO peer mean", ("c_own", "peer_mean"), None),
    ("without the own lag", ("peer_mean",), None),
    ("+ the other group's mean", ("c_own", "peer_mean", "other_mean"), None),
    (
        "+ own and peer punishment",
        ("c_own", "peer_mean", "p_own", "peer_p_mean"),
        None,
    ),  # fmt: skip
    ("within player (episode x player FE)", ("c_own", "peer_mean"), PLAYER),
    ("within round (round FE)", ("c_own", "peer_mean"), ["round_number"]),
    ("within episode", ("c_own", "peer_mean"), ["episode_id"]),
]


def _spec_row(arm, spec, f, key="peer_mean", **extra):
    return dict(
        arm=arm, spec=spec, peer=f[key], peer_se=f[f"{key}_se"],
        peer_lo=f[f"{key}_lo"], peer_hi=f[f"{key}_hi"],
        c_own=f.get("c_own", float("nan")), n=f["n"], **extra,
    )  # fmt: skip


def spec_rows(arm, d, ycol="y"):
    rows = []
    for name, xcols, absorb in SPECS:
        f = fit(d, xcols, ycol, absorb)
        if f is not None:
            rows.append(_spec_row(arm, name, f))
    f = fit(d, ("c_own", "peer_mean_fwd"), ycol)
    if f is not None:
        rows.append(_spec_row(arm, "peer set = the group at t+1", f, "peer_mean_fwd"))
    for k in (2, 3, 4):
        f = fit(d[d["n_peers"] >= k], ycol=ycol)
        if f is not None:
            rows.append(_spec_row(arm, f"at least {k} peers", f))
    f = fit(d[d["stable"] & (d["n_peers"] >= 2)], ycol=ycol)
    if f is not None:
        rows.append(_spec_row(arm, "stable and at least 2 peers", f))
    # the conformity form: the change on the distance to the group
    z = d.dropna(subset=["peer_mean"]).copy()
    z["_dc"] = z[ycol] - z["c_own"]
    z["_gap"] = z["peer_mean"] - z["c_own"]
    f = fit(z, ("_gap",), "_dc")
    if f is not None:
        rows.append(_spec_row(arm, "dc on (peer mean - own), pull form", f, "_gap"))
    f = fit(d, ycol=ycol)
    lam, corrected = eiv(d, f)
    rows.append(
        dict(arm=arm, spec="errors-in-variables corrected", peer=corrected,
             peer_se=float("nan"), peer_lo=float("nan"), peer_hi=float("nan"),
             c_own=float("nan"), n=f["n"], attenuation=lam)  # fmt: skip
    )
    return rows


def eiv(d, f):
    """Method-of-moments attenuation of the peer-mean slope. The leave-one-out
    mean of n peers carries sampling variance within/n given the roster;
    dividing by one minus its share of the regressor's variance AFTER the own
    lag is projected out gives the slope a noiseless group level would get.
    Reported as a robustness check only: the realised peer mean is what the
    player is shown, not a noisy proxy for a latent (PR #202 section 2)."""
    z = d.dropna(subset=["peer_mean", "c_own"])
    within = z.groupby(CELL)["c_own"].transform("var")
    ss = float((within / z["n_peers"].clip(lower=1)).replace([np.inf], np.nan).mean())
    x = z["peer_mean"].to_numpy(float)
    c = z["c_own"].to_numpy(float)
    slope = np.cov(x, c)[0, 1] / np.var(c)
    lam = 1.0 - ss / np.var(x - slope * c)
    return float(lam), float(f["peer_mean"] / lam) if lam > 0 else float("nan")


def quantities(d, ycol="y"):
    """Every headline number as one flat dict, so the same function runs over
    the reseed replicas and gives a run-to-run spread for each."""
    q = {}
    for sample, block, x in _cells(d):
        f = fit(x, ycol=ycol)
        if f is None:
            continue
        q[f"b_{sample}_{block}"] = f["peer_mean"]
        q[f"a_{sample}_{block}"] = f["c_own"]
        if block == "pooled":
            q[f"sum_{sample}"] = f["c_own"] + f["peer_mean"]
            q[f"share_{sample}"] = f["peer_mean"] / (f["c_own"] + f["peer_mean"])
    for name, xcols, absorb, key in [
        ("b_with_other", ("c_own", "peer_mean", "other_mean"), None, "peer_mean"),
        ("d_other", ("c_own", "peer_mean", "other_mean"), None, "other_mean"),
        ("b_within_player", ("c_own", "peer_mean"), PLAYER, "peer_mean"),
        ("a_within_player", ("c_own", "peer_mean"), PLAYER, "c_own"),
        ("b_within_round", ("c_own", "peer_mean"), ["round_number"], "peer_mean"),
        ("b_no_own_lag", ("peer_mean",), None, "peer_mean"),
        ("b_fwd_group", ("c_own", "peer_mean_fwd"), None, "peer_mean_fwd"),
        ("b_with_punish", ("c_own", "peer_mean", "p_own", "peer_p_mean"), None,
         "peer_mean"),  # fmt: skip
    ]:
        f = fit(d, xcols, ycol, absorb)
        if f is not None:
            q[name] = f[key]
    for k in (2, 3):
        f = fit(d[d["n_peers"] >= k], ycol=ycol)
        if f is not None:
            q[f"b_min{k}_peers"] = f["peer_mean"]
    q["mean_peer_mean"] = float(d["peer_mean"].mean())
    q["sd_peer_mean"] = float(d["peer_mean"].std())
    q["mean_n_peers"] = float(d["n_peers"].mean())
    return q


# --------------------------------------------------------------------------- #
# io
# --------------------------------------------------------------------------- #
def load_arm(path):
    from aimanager.evaluation_suite.convert import load_sim

    runs = load_sim(path)
    assert len(runs) == 1, f"{path}: expected one run, got {sorted(runs)}"
    return next(iter(runs.values()))


def md(df, path, floatfmt="{:.4f}"):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = [
            floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v)
            for v in r
        ]
        lines.append("| " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n")


def dump(df, name):
    df.to_csv(OUT / f"{name}.csv", index=False)
    md(df, OUT / f"{name}.md")
    return df


# --------------------------------------------------------------------------- #
# Raven: teacher forcing and the interventional probe
# --------------------------------------------------------------------------- #
def _gnn_setup(seed_model_dir):
    sys.path.insert(0, str(ROOT / "scripts" / "artificial_humans"))
    sys.path.insert(0, str(ROOT / "scripts" / "baselines"))
    import contribution_copula_rho as cc

    from aimanager.generic.graph import GraphNetwork

    data, _, key_to_idx, defaults = cc.load_full()
    tr = cc.select_split(key_to_idx, cc.TRAIN, cc.N_TRAIN_EP)
    te = cc.select_split(key_to_idx, cc.TEST, cc.N_TEST_EP)
    idx = np.array(sorted(set(tr.tolist()) | set(te.tolist())))
    import torch as th

    sub = {k: v[th.as_tensor(idx)] for k, v in data.items()}
    paths = {"trunk": ROOT / TRUNK}
    if seed_model_dir:
        for s in range(1, 6):
            p = Path(seed_model_dir).expanduser() / f"seed_{s}.pt"
            if p.exists():
                paths[f"seed_{s}"] = p
    models = {}
    for name, p in paths.items():
        m = GraphNetwork.load(str(p), device="cpu")
        m.eval()
        assert m.copula_rho == 0.0, f"{name}: teacher-force the BARE trunk"
        names = [e["name"] for e in m.x_encoding]
        assert "contribution" not in names, "conditions on the current round"
        models[name] = m
    print(f"models: {sorted(models)}; x_encoding {names}")
    return cc, sub, defaults, models


def _expect(model, d, edge_index):
    import torch as th

    with th.no_grad():
        _, proba = model.predict_independent(
            d, sample=False, reset_rnn=True, edge_index=edge_index
        )
    lev = np.arange(proba.shape[-1], dtype=np.float64)
    return proba.double().numpy() @ lev


def run_tf(args):
    """Teacher-force every model over the 50 canonical human games and write a
    tidy frame with the same schema as the evaluation suite's canonical one,
    plus `e` = E[c | history]. The regressions themselves run in `analyse`, on
    the same code path the human and simulated arms go through."""
    OUT.mkdir(parents=True, exist_ok=True)
    cc, sub, _, models = _gnn_setup(args.seed_model_dir)
    n_ep, n_ag, n_rd = sub["contribution"].shape
    ok = sub[cc.MASK].numpy().astype(bool)
    pok = sub["punishment_valid"].numpy().astype(bool)
    contr = sub["contribution"].numpy().astype(float)
    punish = sub["punishment"].numpy().astype(float)
    grp = sub["agent_group"].numpy().astype(int)
    g, a, t = np.meshgrid(
        np.arange(n_ep), np.arange(n_ag), np.arange(n_rd), indexing="ij"
    )
    base = pd.DataFrame(
        {
            "episode_id": g.ravel(),
            "participant_code": a.ravel().astype(str),
            "round_number": t.ravel(),
            "group_id": grp.ravel(),
            "contribution": np.where(ok, contr, np.nan).ravel(),
            "punishment": np.where(pok, punish, np.nan).ravel(),
        }
    )
    print(f"tensor frame: {n_ep} episodes x {n_ag} agents x {n_rd} rounds")
    print(f"  valid contributions {int(ok.sum())} of {ok.size}")
    first = next(iter(models.values()))
    edge_index = first.create_fully_connected(n_ag, n_batch=n_ep)
    for name, model in models.items():
        e = _expect(model, sub, edge_index)
        df = base.copy()
        df["e"] = np.where(ok, e, np.nan).ravel()
        df.to_parquet(OUT / f"tf_human_{name}.parquet", index=False)
        print(f"  {name}: mean E {np.nanmean(df['e']):.6f}")


def _masks(data, target):
    """[G, A, T] masks for the target's own cells, for the cells of the peers
    it shares a group with, and for the other group's cells. Membership is read
    at the round being shifted. The three partition the tensor, so the three
    responses must add up to the everything-shifted one."""
    grp = data["agent_group"].numpy().astype(int)
    same = grp == grp[:, target : target + 1, :]
    own = np.zeros_like(same)
    own[:, target, :] = True
    return {"own": own, "peers": same & ~own, "other": ~same}


def _apply_shift(data, mask, delta, default):
    import torch as th

    from aimanager.generic.data import shift as roll

    c = data["contribution"].clone()
    valid = data["contribution_valid"].numpy().astype(bool)
    sel = th.as_tensor(mask & valid)
    c[sel] = (c[sel] + delta).clamp(0, K - 1)
    d = dict(data)
    d["contribution"] = c
    d["prev_contribution"] = roll(c, default)
    return d


def _features(data, target):
    """The three regressors of the regression route as [G, T] arrays, read off
    the tensors the model is actually fed: the target's own previous
    contribution, the leave-one-out mean of the previous contributions of the
    peers it shares a group with at that round, and the other group's mean.
    `n_peers` is the count the second is over."""
    c = data["contribution"].numpy().astype(float)
    ok = data["contribution_valid"].numpy().astype(bool)
    grp = data["agent_group"].numpy().astype(int)
    same = (grp == grp[:, target : target + 1, :]).copy()
    other = ~same
    same[:, target, :] = False
    pad_c = np.full_like(c[:, :, :1], np.nan)
    prev_c = np.concatenate([pad_c, c[:, :, :-1]], axis=2)
    pad_ok = np.zeros_like(ok[:, :, :1])
    prev_ok = np.concatenate([pad_ok, ok[:, :, :-1]], axis=2).astype(bool)
    out = {}
    for key, mask in (("peers", same), ("other", other)):
        w = mask & prev_ok
        num = np.where(w, np.nan_to_num(prev_c), 0.0).sum(axis=1)
        den = w.sum(axis=1)
        out[key] = np.where(den > 0, num / np.maximum(den, 1), np.nan)
        if key == "peers":
            out["n_peers"] = den
    out["own"] = np.where(prev_ok[:, target, :], prev_c[:, target, :], np.nan)
    return out


CONDS = ("own", "peers", "other")


def _probe_rows(model, name, sub, ok, stable, default, edge_index, one_round):
    n_ep, n_ag, n_rd = sub["contribution"].shape
    e0 = _expect(model, sub, edge_index)
    base = {a: _features(sub, a) for a in range(n_ag)}
    npeer = np.stack([base[a]["n_peers"] for a in range(n_ag)], axis=1)
    rows = []
    deltas = ONE_ROUND_DELTAS if one_round else DELTAS
    for delta in deltas:
        for stim in ONE_ROUND_STIMULI if one_round else [None]:
            shape = (n_ep, n_ag, n_rd)
            d_e = {c: np.full(shape, np.nan) for c in CONDS}
            shift_f = {c: np.full(shape, np.nan) for c in CONDS}
            for a in range(n_ag):
                masks = _masks(sub, a)
                for cond in CONDS:
                    mask = masks[cond]
                    if stim is not None:
                        keep = np.zeros(n_rd, bool)
                        keep[stim] = True
                        mask = mask & keep
                    d = _apply_shift(sub, mask, delta, default)
                    e = _expect(model, d, edge_index)
                    d_e[cond][:, a, :] = e[:, a, :] - e0[:, a, :]
                    f = _features(d, a)
                    shift_f[cond][:, a, :] = f[cond] - base[a][cond]
            if stim is None:
                d_all = _apply_shift(sub, np.ones_like(ok), delta, default)
                e_all = _expect(model, d_all, edge_index) - e0
            else:
                e_all = np.full(shape, np.nan)
            sets = {
                "all": ok & (npeer > 0),
                "stable": ok & (npeer > 0) & stable,
                "min2_peers": ok & (npeer >= 2),
            }
            if stim is not None:
                keep = np.zeros(n_rd, bool)
                keep[stim + 1] = True
                sets = {k: v & keep for k, v in sets.items()}
            for sname, sel in sets.items():
                for c in CONDS:
                    sel = sel & np.isfinite(shift_f[c])
                if not sel.any():
                    continue
                row = dict(
                    model=name, delta=delta, stimulus_round=stim, set=sname,
                    n=int(sel.sum()), d_e_all=float(e_all[sel].mean()),
                )  # fmt: skip
                for c in CONDS:
                    row[f"d_e_{c}"] = float(d_e[c][sel].mean())
                    row[f"shift_{c}"] = float(shift_f[c][sel].mean())
                rows.append(row)
            print(f"  {name} delta {delta:+d} stim {stim} done", flush=True)
    return rows


def run_probe(args):
    """Shift a group's recent level and read how far the predicted expectation
    moves. Three conditions per target agent -- the target's own history, its
    same-group peers', the other group's -- so the shared channel is separated
    from the own channel, plus the everything-shifted total PR #191 reported,
    which the three must add up to."""
    OUT.mkdir(parents=True, exist_ok=True)
    cc, sub, defaults, models = _gnn_setup(args.seed_model_dir)
    n_ep, n_ag, n_rd = sub["contribution"].shape
    ok = sub[cc.MASK].numpy().astype(bool)
    grp = sub["agent_group"].numpy().astype(int)
    # the roster is stable at round t when NOBODY moved between t-1 and t; the
    # eight players are a fixed population partitioned into two groups, so that
    # is the same event the regression's roster_change flags
    moved = (grp[:, :, 1:] != grp[:, :, :-1]).any(axis=1, keepdims=True)
    stable = np.zeros((n_ep, n_ag, n_rd), bool)
    stable[:, :, 1:] = ~np.broadcast_to(moved, (n_ep, n_ag, n_rd - 1))
    first = next(iter(models.values()))
    edge_index = first.create_fully_connected(n_ag, n_batch=n_ep)
    rows = []
    for name, model in models.items():
        rows += _probe_rows(
            model, name, sub, ok, stable, defaults["contribution"], edge_index,
            args.one_round,
        )  # fmt: skip
    df = pd.DataFrame(rows)
    for c in CONDS:
        df[f"gain_{c}"] = df[f"d_e_{c}"] / df[f"shift_{c}"]
    # PR #191's everything-shifted gain, normalised the same way: the realised
    # shift of the own feature, which is what the grid clipping leaves
    df["gain_all"] = df["d_e_all"] / df["shift_own"]
    df["gain_all_raw"] = df["d_e_all"] / df["delta"]
    df["additivity"] = df["d_e_own"] + df["d_e_peers"] + df["d_e_other"] - df["d_e_all"]
    dump(df, "probe_one_round" if args.one_round else "probe")
    print(df.to_string(index=False))


# --------------------------------------------------------------------------- #
# local analysis
# --------------------------------------------------------------------------- #
def coverage_row(arm, frame, d):
    return dict(
        arm=arm,
        episodes=int(frame["episode_id"].nunique()),
        agent_rounds=int(len(frame)),
        contrib_nan=int(frame["contribution"].isna().sum()),
        transitions=int(len(d)),
        with_peers=int(d["peer_mean"].notna().sum()),
        solo=int((d["n_peers"] == 0).sum()),
        change_transitions=int((~d["stable"]).sum()),
        ego_moves=int(d["ego_moves"].sum()),
        mean_n_peers=float(d["n_peers"].mean()),
        sd_peer_mean=float(d["peer_mean"].std()),
    )


def _floor_block(qdf, cols, route):
    f = qdf[cols]
    return pd.DataFrame(
        {
            "route": route,
            "seed_mean": f.mean(axis=1),
            "seed_sd": f.std(axis=1, ddof=1),
            "seed_min": f.min(axis=1),
            "seed_max": f.max(axis=1),
        }
    )


def run_analyse(args):
    from aimanager.evaluation_suite.convert import load_human

    OUT.mkdir(parents=True, exist_ok=True)
    frames = {"human": load_human(HUMAN_CSV)}
    for arm, p in ARMS.items():
        if p.exists():
            frames[arm] = load_arm(p)
    if args.seed_dir:
        for n in SEED_NAMES:
            p = Path(args.seed_dir).expanduser() / f"{n}.parquet"
            if p.exists():
                frames[f"floor_{n}"] = load_arm(p)
    pans = {a: panel(f) for a, f in frames.items()}

    tf = {}
    for name in MODELS:
        p = OUT / f"tf_human_{name}.parquet"
        if p.exists():
            tf[name] = panel(pd.read_parquet(p), extra_y=("e",))
            tf[name]["e_model"] = tf[name]["e_next"]

    main = [a for a in ["human", *ARMS] if a in pans]
    coef = dump(
        pd.DataFrame(sum((coef_rows(a, pans[a]) for a in main), [])), "coefficients"
    )
    if tf:
        rows = []
        for name, t in tf.items():
            rows += coef_rows(f"tf_{name}", t, "e_model")
        rows += coef_rows("tf_frame_human_y", tf["trunk"], "y")
        coef = pd.concat([coef, dump(pd.DataFrame(rows), "coefficients_tf")])

    alt = [a for a in ("human", "sim_noise_off", "sim_noise_on") if a in pans]
    alt_rows = sum((spec_rows(a, pans[a]) for a in alt), [])
    if "trunk" in tf:
        alt_rows += spec_rows("tf_trunk", tf["trunk"], "e_model")
        alt_rows += spec_rows("tf_frame_human_y", tf["trunk"], "y")
    dump(pd.DataFrame(alt_rows), "alternatives")
    dump(
        pd.DataFrame([coverage_row(a, frames[a], pans[a]) for a in frames]), "coverage"
    )

    q = {a: quantities(pans[a]) for a in main}
    floor_arms = [f"floor_{n}" for n in SEED_NAMES if f"floor_{n}" in pans]
    q.update({a: quantities(pans[a]) for a in floor_arms})
    q.update({f"tf_{n}": quantities(t, "e_model") for n, t in tf.items()})
    if "trunk" in tf:
        q["tf_frame_human_y"] = quantities(tf["trunk"], "y")
    qdf = pd.DataFrame(q)
    qdf.index.name = "quantity"
    dump(qdf.reset_index(), "quantities")

    blocks = []
    if floor_arms:
        blocks.append((_floor_block(qdf, floor_arms, "realised"), "sim_noise_off"))
    tf_cols = [f"tf_{n}" for n in MODELS if f"tf_{n}" in qdf.columns]
    if len(tf_cols) > 1:
        blocks.append((_floor_block(qdf, tf_cols, "teacher forced"), "tf_trunk"))
    if blocks:
        parts = []
        for fl, ref in blocks:
            fl = fl.reset_index().rename(columns={"index": "quantity"})
            fl["human"] = fl["quantity"].map(qdf["human"])
            fl["model"] = fl["quantity"].map(qdf[ref])
            fl["model_arm"] = ref
            fl["human_minus_model"] = fl["human"] - fl["model"]
            fl["in_seed_sd"] = fl["human_minus_model"].abs() / fl["seed_sd"]
            parts.append(fl)
        dump(pd.concat(parts, ignore_index=True), "noise_floor")

    diffs = []
    if "trunk" in tf:
        cells = [(s, sel(tf["trunk"])) for s, sel in SAMPLES]
        cells += [(b, tf["trunk"][tf["trunk"]["block"] == b]) for b, _, _ in BLOCKS]
        cells += [
            (
                "stable, 1-8",
                tf["trunk"][tf["trunk"]["stable"] & (tf["trunk"]["block"] == "1-8")],
            ),  # noqa: E501
            (
                "stable, 9-16",
                tf["trunk"][tf["trunk"]["stable"] & (tf["trunk"]["block"] == "9-16")],
            ),  # noqa: E501
            (
                "stable, 17-24",
                tf["trunk"][tf["trunk"]["stable"] & (tf["trunk"]["block"] == "17-24")],
            ),  # noqa: E501
        ]
        for label, cell in cells:
            r = paired_diff(cell, "y", "e_model", label=label)
            if r:
                diffs.append(dict(comparison="human vs teacher-forced trunk", **r))
    for arm in [a for a in ARMS if a in pans]:
        for sample, sel in SAMPLES:
            ha, sa = sel(pans["human"]), sel(pans[arm])
            fa, fb = fit(ha), fit(sa)
            if fa is None or fb is None:
                continue
            diffs.append(
                dict(comparison=f"human vs {arm}", sample=sample,
                     human=fa["peer_mean"], model=fb["peer_mean"],
                     diff=fa["peer_mean"] - fb["peer_mean"], n_rows=fa["n"],
                     **unpaired_diff(ha, sa))  # fmt: skip
            )
    dump(pd.DataFrame(diffs), "human_minus_model")

    for name in ("probe", "probe_one_round"):
        p = OUT / f"{name}.csv"
        if p.exists():
            print(f"--- {name} ---")
            print(pd.read_csv(p).to_string(index=False))
    figure(pans, tf, qdf, floor_arms, coef)
    print(qdf.to_string())


COLORS = {
    "human": "#222222",
    "tf_trunk": "#8172b3",
    "sim_noise_off": "#c44e52",
    "sim_noise_on": "#4c72b0",
}


def figure(pans, tf, qdf, floor_arms, coef):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    x = np.arange(len(BLOCK_NAMES))
    series = [("human", "human")]
    if "trunk" in tf:
        series.append(("tf_trunk", "model, teacher-forced"))
    series += [(a, a) for a in ("sim_noise_off", "sim_noise_on") if a in pans]
    bars = {"human", "tf_trunk"}
    for i_ax, sample in [(0, "all"), (1, "stable")]:
        keys = [f"b_{sample}_{b}" for b in BLOCK_NAMES]
        if floor_arms:
            f = qdf.loc[keys, floor_arms]
            sd = f.std(axis=1, ddof=1).to_numpy()
            mu = f.mean(axis=1).to_numpy()
            ax[i_ax].bar(x, 2 * sd, bottom=mu - sd, width=0.7, color="0.88", zorder=0,
                         label="+-1 sd, 6 reseed sims")  # fmt: skip
        for i, (a, lab) in enumerate(series):
            if a not in qdf.columns:
                continue
            vals = np.array([qdf[a].get(k, np.nan) for k in keys], float)
            pos = x + (i - 1) * 0.08
            err = None
            if a in bars:
                c = coef[(coef["arm"] == a) & (coef["sample"] == sample)]
                c = c.set_index("block").reindex(BLOCK_NAMES)
                err = [
                    (vals - c["peer_mean_lo"]).to_numpy(),
                    (c["peer_mean_hi"] - vals).to_numpy(),
                ]
            ax[i_ax].errorbar(pos, vals, yerr=err, fmt="o-", capsize=3, lw=1.2,
                              color=COLORS.get(a, "0.4"), label=lab)  # fmt: skip
        ax[i_ax].set_xticks(x)
        ax[i_ax].set_xticklabels(BLOCK_NAMES)
        ax[i_ax].set_ylabel("coefficient on the LOO group mean")
        ax[i_ax].set_title(f"Shared-signal weight ({sample} transitions)")
        ax[i_ax].legend(fontsize=7)
    p = OUT / "probe.csv"
    if p.exists():
        pr = pd.read_csv(p)
        pr = pr[(pr["set"] == "all") & (pr["model"] == "trunk")]
        for col, mark, lab in [
            ("gain_own", "s-", "own history"),
            ("gain_peers", "o-", "same-group peers (shared)"),
            ("gain_other", "^-", "the other group"),
        ]:
            ax[2].plot(pr["delta"], pr[col], mark, label=lab)
        ax[2].axhline(0, color="0.6", lw=0.8)
        ax[2].set_xlabel("shift applied to that channel's recent level")
        ax[2].set_ylabel("gain of E[c] per unit realised shift")
        ax[2].set_title("Interventional probe (trunk)")
        ax[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "shared_signal_gain.jpg", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="stage", required=True)
    a = sp.add_parser("analyse")
    a.add_argument("--seed-dir", type=Path, default=None)
    a.set_defaults(fn=run_analyse)
    t = sp.add_parser("tf")
    t.add_argument("--seed-model-dir", type=Path, default=None)
    t.set_defaults(fn=run_tf)
    p = sp.add_parser("probe")
    p.add_argument("--seed-model-dir", type=Path, default=None)
    p.add_argument("--one-round", action="store_true")
    p.set_defaults(fn=run_probe)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
