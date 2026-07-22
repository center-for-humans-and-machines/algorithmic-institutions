"""Feature engineering + data preparation for the hand-crafted linear AH
baselines (issue #119).

Builds the hand-crafted feature pool from the `create_torch_data` tensors and
flattens the TRAIN split into a single [N, n_features] matrix with per-row CV
folds. Consumed by:
  * scripts/baselines/run_baseline_cv.py     -- the CV grid driver
  * scripts/baselines/inspect_best_model.py  -- coefficient inspection

Feature semantics are specified in notes/baseline_feature_defs.md. Only
create_torch_data + get_cross_validations are used from src/; every derived
feature is computed here from the raw [G, A, T] tensors. payoff = 20 -
contribution - punishment + common_good (per-capita, reports/basics.md).

Runs locally (CPU torch, no PyG).
"""

import os

os.environ.setdefault("DISABLE_PANDERA_IMPORT_WARNING", "True")
import numpy as np
import yaml

ENDOWMENT = 20.0  # per-round private endowment (reports/basics.md)


# --------------------------------------------------------------------------- #
# config / data loading
# --------------------------------------------------------------------------- #
def load_config(path):
    with open(path) as fh:
        return yaml.safe_load(fh)


def load_episodes(cfg, root):
    """Load the experiment rows, dropping the pair-flip copies when
    `data.exclude_flipped` is set (train on the real episodes, not the doubled)."""
    import pandas as pd

    df = pd.read_csv(root / cfg["data"]["data_file"])
    df = df[df["experiment_name"].isin(cfg["data"]["experiment_names"])]
    if cfg["data"].get("exclude_flipped", False):
        df = df[~df["global_group_id"].str.contains("(flipped)", regex=False)]
    return df


# --------------------------------------------------------------------------- #
# feature helpers  (all operate on [G, A, T] numpy arrays)
# --------------------------------------------------------------------------- #
def _payoff(c, p, cg):
    return ENDOWMENT - c - p + cg


def _group_prev_means(measure, group):
    """LOO own-group mean (`peers`) and other-group mean (`other`) of a t-1
    measure, grouped by t-1 membership. Used for the lag_* switch twins.
    Empty other group -> 0."""
    m = measure.astype(float)
    gp = group.astype(int)
    G, _, T = m.shape
    peers = np.zeros_like(m)
    other = np.zeros_like(m)
    for g in range(G):
        for t in range(T):
            grp, x = gp[g, :, t], m[g, :, t]
            for s in (0, 1):
                sel = grp == s
                if not sel.any():
                    continue
                n = sel.sum()
                peers[g, sel, t] = (x[sel].sum() - x[sel]) / (n - 1) if n > 1 else 0.0
                oth = grp == (1 - s)
                if oth.any():
                    other[g, oth, t] = x[sel].mean()
    return peers, other


def _obs_group_means(measure, prev_group, cur_group, loo):
    """Observational Group/Other means: t-1 values over the t-1 roster of the
    agent's CURRENT group ids. loo=True drops the agent's own value when they
    were in that roster (per-member measures); loo=False takes the full roster
    mean (group-level measures like common_good). Empty t-1 roster -> 0."""
    m = measure.astype(float)
    pg = prev_group.astype(int)
    cg = cur_group.astype(int)
    G, A, T = m.shape
    grp = np.zeros_like(m)
    oth = np.zeros_like(m)
    for g in range(G):
        for t in range(T):
            roster, x, cur = pg[g, :, t], m[g, :, t], cg[g, :, t]
            for a in range(A):
                for out, gid in ((grp, cur[a]), (oth, 1 - cur[a])):
                    sel = roster == gid
                    n = sel.sum()
                    if n == 0:
                        val = 0.0
                    elif out is grp and loo and roster[a] == gid:
                        val = (x[sel].sum() - x[a]) / (n - 1) if n > 1 else 0.0
                    else:
                        val = x[sel].mean()
                    out[g, a, t] = val
    return grp, oth


def _obs_group_size(prev_group, cur_group, prev_recorded):
    """t-1 head-count of the agent's CURRENT group id / the opponent id."""
    pg = prev_group.astype(int)
    cg = cur_group.astype(int)
    rec = prev_recorded.astype(bool)
    G, A, T = pg.shape
    own = np.zeros((G, A, T), float)
    oth = np.zeros((G, A, T), float)
    for g in range(G):
        for t in range(T):
            roster, r, cur = pg[g, :, t], rec[g, :, t], cg[g, :, t]
            for a in range(A):
                own[g, a, t] = np.sum(r & (roster == cur[a]))
                oth[g, a, t] = np.sum(r & (roster == 1 - cur[a]))
    return own, oth


def _group_full_mean(measure, group):
    """Full own-group mean (INCLUDES self). Used for lag_payoff_mean_group."""
    m = measure.astype(float)
    gp = group.astype(int)
    G, _, T = m.shape
    out = np.zeros_like(m)
    for g in range(G):
        for t in range(T):
            grp, x = gp[g, :, t], m[g, :, t]
            for s in (0, 1):
                sel = grp == s
                if sel.any():
                    out[g, sel, t] = x[sel].mean()
    return out


def _since_switch_window(per_round_mean, does_switch, include_reset=True):
    """Running mean of a per-round (t-1) series over the agent's tenure,
    resetting on the agent's switches. include_reset=False (value windows):
    the arrival round outputs 0 and is not accumulated; include_reset=True
    (size windows): the arrival round keeps its value."""
    x = per_round_mean
    ds = does_switch.astype(bool)
    G, A, T = x.shape
    out = np.zeros_like(x)
    for g in range(G):
        for a in range(A):
            s, c = 0.0, 0
            for t in range(T):
                if ds[g, a, t]:  # arrived in a new group -> new tenure
                    s, c = 0.0, 0
                    if not include_reset:
                        out[g, a, t] = 0.0
                        continue
                s += x[g, a, t]
                c += 1
                out[g, a, t] = s / c
    return out


def _group_size(group, recorded):
    """Per-(g, a, t) count of recorded agents sharing a's group membership."""
    gp = group.astype(int)
    rec = recorded.astype(bool)
    G, A, T = gp.shape
    out = np.zeros((G, A, T), float)
    for g in range(G):
        for t in range(T):
            grp, r = gp[g, :, t], rec[g, :, t]
            for a in range(A):
                out[g, a, t] = np.sum(r & (grp == grp[a]))
    return out


def _rounds_since_switch(does_switch):
    """Rounds since the agent last actually switched groups (tenure; 0 on the
    switch round). NB: 'since last actual switch', not 'since last option'."""
    ds = does_switch.astype(bool)
    G, A, T = ds.shape
    out = np.zeros((G, A, T), float)
    for g in range(G):
        for a in range(A):
            cnt = 0
            for t in range(T):
                cnt = 0 if (t == 0 or ds[g, a, t]) else cnt + 1  # round 0 = 0
                out[g, a, t] = cnt
    return out


def _switched_last_choice(does_switch, switch_every, strict_prev=False):
    """'Did I switch last time I had the choice?' -- the agent's does_switch at
    its most recent decision round, forward-filled.

    strict_prev=False (default): most recent decision r <= t. At a decision round
      t this is does_switch[t] itself -- valid for the contribution target (group
      membership is known before contributing), but it LEAKS the switch target.
    strict_prev=True: most recent decision STRICTLY before t. At a decision round
      t this is does_switch[t - switch_every] (the previous opportunity) -- the
      leakage-safe version for predicting does_switch[t]."""
    ds = does_switch.astype(int)
    _, _, T = ds.shape
    out = np.zeros_like(ds, dtype=float)
    for t in range(T):
        ref = (t - 1) if strict_prev else t
        last_dec = (ref // switch_every) * switch_every  # largest multiple <= ref
        if last_dec > 0:  # <= 0 => no earlier decision
            out[:, :, t] = ds[:, :, last_dec]
    return out


# --------------------------------------------------------------------------- #
# feature pool
# --------------------------------------------------------------------------- #
def build_feature_pool(d, switch_every):
    """Return {feature_name: [G, A, T] float array} for the full feature pool.

    `d` is a create_torch_data data dict of tensors; `switch_every` is the
    decision cadence. Semantics per notes/baseline_feature_defs.md."""
    npd = {
        k: d[k].numpy()
        for k in (
            "prev_contribution",
            "prev_punishment",
            "prev_common_good",
            "round_number",
            "does_switch",
            "prev_agent_group",
            "agent_group",
            "recorded",
            "prev_recorded",
        )
    }
    f = {}

    # -- direct from tensors --
    f["prev_contribution"] = npd["prev_contribution"].astype(float)
    f["prev_punishment"] = npd["prev_punishment"].astype(float)
    f["round_number"] = npd["round_number"].astype(float)
    own_cg = npd["prev_common_good"].astype(float)  # own EXPERIENCED t-1 cg
    f["prev_payoff"] = _payoff(f["prev_contribution"], f["prev_punishment"], own_cg)

    # -- Group / Other (observational: t-1 values over the t-1 roster of the
    #    agent's CURRENT group ids; prev_common_good = the current group's
    #    t-1 cg, own experienced cg stays in prev_payoff only) --
    ga = npd["agent_group"]
    gp = npd["prev_agent_group"]
    grp, oth = {}, {}
    for m, loo in (
        ("contribution", True),
        ("punishment", True),
        ("common_good", False),
    ):
        grp[m], oth[m] = _obs_group_means(npd[f"prev_{m}"], gp, ga, loo=loo)
        # round 0: every t-1 value is the same default -> both sides read it
        grp[m][:, :, 0] = npd[f"prev_{m}"][:, :, 0]
        oth[m][:, :, 0] = npd[f"prev_{m}"][:, :, 0]
    f["prev_contribution_mean_group"] = grp["contribution"]
    f["prev_punishment_mean_group"] = grp["punishment"]
    f["prev_common_good"] = grp["common_good"]
    f["prev_contribution_mean_other"] = oth["contribution"]
    f["prev_punishment_mean_other"] = oth["punishment"]
    f["prev_common_good_other"] = oth["common_good"]
    f["prev_payoff_mean_group"] = _payoff(
        grp["contribution"], grp["punishment"], grp["common_good"]
    )
    f["prev_payoff_mean_other"] = _payoff(
        oth["contribution"], oth["punishment"], oth["common_good"]
    )

    # -- Gap/Delta (Group - Other) --
    f["prev_contribution_mean_gap"] = grp["contribution"] - oth["contribution"]
    f["prev_punishment_mean_gap"] = grp["punishment"] - oth["punishment"]
    f["prev_common_good_gap"] = grp["common_good"] - oth["common_good"]
    f["prev_payoff_mean_gap"] = (
        f["prev_payoff_mean_group"] - f["prev_payoff_mean_other"]
    )

    # -- observational sizes (t-1 head-count of the current / opponent id) --
    own_cur = _group_size(ga, npd["recorded"])
    total_cur = npd["recorded"].astype(float).sum(axis=1, keepdims=True)
    own_prev, oth_prev = _obs_group_size(gp, ga, npd["prev_recorded"])
    # round 0 has no real previous round (prev_recorded all False -> sizes 0);
    # default the prev sizes to the round-0 current sizes (balanced 4/4 start).
    own_prev[:, :, 0] = own_cur[:, :, 0]
    oth_prev[:, :, 0] = (total_cur - own_cur)[:, :, 0]

    # -- since-switch windows (0 at arrival; payoff windowed as its own
    #    series so it is 0 there too; size windows keep the arrival value) --
    ds = npd["does_switch"]
    f["win_contribution_mean_peers"] = _since_switch_window(
        grp["contribution"], ds, include_reset=False
    )
    f["win_punishment_mean_peers"] = _since_switch_window(
        grp["punishment"], ds, include_reset=False
    )
    f["win_common_good_peers"] = _since_switch_window(
        grp["common_good"], ds, include_reset=False
    )
    f["win_payoff_mean_peers"] = _since_switch_window(
        f["prev_payoff_mean_group"], ds, include_reset=False
    )
    f["win_group_size"] = _since_switch_window(own_prev, ds)
    f["win_contribution_mean_other"] = _since_switch_window(
        oth["contribution"], ds, include_reset=False
    )
    f["win_punishment_mean_other"] = _since_switch_window(
        oth["punishment"], ds, include_reset=False
    )
    f["win_common_good_other"] = _since_switch_window(
        oth["common_good"], ds, include_reset=False
    )
    f["win_payoff_mean_other"] = _since_switch_window(
        f["prev_payoff_mean_other"], ds, include_reset=False
    )
    f["win_group_size_other"] = _since_switch_window(oth_prev, ds)

    # -- lag_* twins (SWITCH target): t-1 values over t-1 membership; the
    #    standard means aggregate over agent_group[t], which IS the target --
    lpeers, lother = {}, {}
    for m in ("contribution", "punishment", "common_good"):
        lpeers[m], lother[m] = _group_prev_means(npd[f"prev_{m}"], gp)
        lother[m][:, :, 0] = lpeers[m][:, :, 0]  # symmetric round-0 default
    for m in ("contribution", "punishment"):
        f[f"lag_{m}_mean_peers"] = lpeers[m]
    for m in ("contribution", "punishment", "common_good"):
        f[f"lag_{m}_mean_other"] = lother[m]
        f[f"lag_{m}_mean_gap"] = lpeers[m] - lother[m]
    f["lag_payoff_mean_other"] = _payoff(
        lother["contribution"], lother["punishment"], lother["common_good"]
    )
    f["lag_payoff_mean_gap"] = (
        _payoff(lpeers["contribution"], lpeers["punishment"], lpeers["common_good"])
        - f["lag_payoff_mean_other"]
    )
    f["lag_payoff_mean_group"] = _payoff(
        _group_full_mean(npd["prev_contribution"], gp),
        _group_full_mean(npd["prev_punishment"], gp),
        own_cg,  # t-1 group-mates shared the agent's own experienced cg
    )

    # -- structural sizes / counters --
    f["group_size"] = own_cur
    f["prev_group_size"] = own_prev
    f["prev_group_size_other"] = oth_prev
    f["prev_group_size_delta"] = own_prev - oth_prev
    f["rounds_since_switch"] = _rounds_since_switch(ds)
    f["switched_last_choice"] = _switched_last_choice(ds, switch_every)

    # switch-target variant: only ever reads does_switch[<t]
    f["prev_switched_last_choice"] = _switched_last_choice(
        ds, switch_every, strict_prev=True
    )

    def _shift1(a):  # value as of t-1 (round 0 keeps its own value; masked anyway)
        out = np.roll(a, 1, axis=2)
        out[:, :, 0] = a[:, :, 0]
        return out

    for name in (
        "win_contribution_mean_peers",
        "win_punishment_mean_peers",
        "win_common_good_peers",
        "win_payoff_mean_peers",
        "win_group_size",
        "win_contribution_mean_other",
        "win_punishment_mean_other",
        "win_common_good_other",
        "win_payoff_mean_other",
        "win_group_size_other",
    ):
        f[f"prev_{name}"] = _shift1(f[name])  # window as of t-1 (reset uses <t)

    return f


# --------------------------------------------------------------------------- #
# data preparation: build the pool once, flatten to rows, assign folds
# --------------------------------------------------------------------------- #
def prepare_data(cfg, root):
    """Build the feature pool once on the (train) data, flatten valid rows to a
    single [N, n_features] matrix, and tag each row with its pair-level CV fold.

    Returns dict(X, col_of, y_cat, y_cont, fold_row)."""
    import random

    import torch as th

    from aimanager.generic.data import create_torch_data, get_cross_validations

    seed = cfg["cv"]["seed"]
    th.random.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    df = load_episodes(cfg, root)
    switch_every = cfg["data"].get("switch_every")
    data, default_values, pair_id = create_torch_data(df, switch_every=switch_every)
    G = data["contribution"].shape[0]

    pool = build_feature_pool(data, switch_every)
    feats = sorted(pool)
    col_of = {f: i for i, f in enumerate(feats)}

    mask = data[cfg["data"]["mask"]].numpy().astype(bool)
    sel = mask.reshape(-1)
    X = np.stack([pool[f].reshape(-1)[sel] for f in feats], axis=1)
    tgt = data[cfg["data"]["target"]].numpy().reshape(-1)[sel]

    # per-episode CV fold: always decided here from the cv args (seed + n_folds),
    # grouped by pair_id. The locked test set is a separate file, so every fold
    # produced here is a train fold. Changing cv.seed / cv.n_folds re-partitions.
    data["_eid"] = th.arange(G)
    fold_of_ep = np.full(G, -1, int)
    for i, _, te in get_cross_validations(
        data, cfg["cv"]["n_folds"], 1.0, group_key=pair_id
    ):
        if i is None:  # get_cross_validations emits a trailing (None, .., None)
            continue
        for e in te["_eid"].tolist():
            fold_of_ep[e] = i
    fold_row = np.broadcast_to(fold_of_ep[:, None, None], mask.shape).reshape(-1)[sel]

    return dict(
        X=X,
        col_of=col_of,
        y_cat=tgt.astype(int),
        y_cont=tgt.astype(float),
        fold_row=fold_row,
        default_values=default_values,
        switch_every=switch_every,
    )
