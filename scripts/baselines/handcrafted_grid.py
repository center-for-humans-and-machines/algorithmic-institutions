"""Feature engineering + data preparation for the hand-crafted linear AH
baselines (issue #119).

Builds the hand-crafted feature pool from the `create_torch_data` tensors and
flattens the TRAIN split into a single [N, n_features] matrix with per-row CV
folds. Consumed by:
  * scripts/baselines/run_baseline_cv.py     -- the CV grid driver
  * scripts/baselines/inspect_best_model.py  -- coefficient inspection

Feature semantics are specified in notes/baseline_feature_defs.md: a current
family (no prefix) for the switch target -- anchored at the pre-switch round
since #123 -- and a prev family for the contribution target. Only
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


def _obs_group_means(measure, roster_group, cur_group, loo):
    """Group/Other means of `measure` over the `roster_group` membership, keyed
    to the agent's CURRENT group ids. Prev family: t-1 values over the t-1
    roster (roster_group=prev). Current family: round-t values over the round-t
    roster (roster_group=cur). loo=True drops the agent's own value when they
    are in that roster (per-member measures); loo=False takes the full roster
    mean (group-level measures like common_good). Empty roster -> 0."""
    m = measure.astype(float)
    pg = roster_group.astype(int)
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


def _obs_group_size(roster_group, cur_group, recorded):
    """Head-count of the agent's CURRENT group id / the opponent id over the
    `roster_group` membership (prev roster for the prev family, current roster
    for the current sizes)."""
    pg = roster_group.astype(int)
    cg = cur_group.astype(int)
    rec = recorded.astype(bool)
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


def _since_switch_window(per_round_mean, arrival, include_reset=True):
    """Running mean of a per-round series over the agent's tenure, resetting
    at the agent's arrival rounds. include_reset=True (current family + size
    windows): the arrival round's value starts the new window; include_reset=
    False (prev value windows): the arrival round outputs 0 and is not
    accumulated (its prev-observed value belongs to the left group)."""
    x = per_round_mean
    ds = arrival.astype(bool)
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


def _rounds_since_switch(arrival):
    """Rounds since the agent last actually switched groups (tenure; 0 on the
    arrival round). NB: 'since last actual switch', not 'since last option'."""
    ds = arrival.astype(bool)
    G, A, T = ds.shape
    out = np.zeros((G, A, T), float)
    for g in range(G):
        for a in range(A):
            cnt = 0
            for t in range(T):
                cnt = 0 if (t == 0 or ds[g, a, t]) else cnt + 1  # round 0 = 0
                out[g, a, t] = cnt
    return out


def _switched_last_choice(arrival, switch_every):
    """'Did I switch at my most recent decision?' -- the arrival marker at the
    most recent arrival round <= t, forward-filled. Reads only decisions
    resolved by row t, so it is legal for both targets."""
    ds = arrival.astype(int)
    _, _, T = ds.shape
    out = np.zeros_like(ds, dtype=float)
    for t in range(T):
        last_arr = (t // switch_every) * switch_every  # largest multiple <= t
        if last_arr > 0:  # <= 0 => no arrival yet
            out[:, :, t] = ds[:, :, last_arr]
    return out


# --------------------------------------------------------------------------- #
# feature pool
# --------------------------------------------------------------------------- #
# Current-family features that read round-t contributions/punishments/common
# good -- the round the contribution target is drawn in. ILLEGAL for the
# contribution target (hard error at config validation); membership-derived
# current features (sizes, tenure counters) are legal for both targets.
CURRENT_VALUED = frozenset(
    [
        "contribution",
        "punishment",
        "payoff",
        "common_good",
        "common_good_other",
        "common_good_gap",
        "contribution_mean_group",
        "contribution_mean_other",
        "contribution_mean_gap",
        "punishment_mean_group",
        "punishment_mean_other",
        "punishment_mean_gap",
        "payoff_mean_group",
        "payoff_mean_other",
        "payoff_mean_gap",
        "win_contribution_mean_group",
        "win_punishment_mean_group",
        "win_common_good",
        "win_payoff_mean_group",
        "win_contribution_mean_other",
        "win_punishment_mean_other",
        "win_common_good_other",
        "win_payoff_mean_other",
    ]
)


def validate_feature_legality(cfg):
    """Hard error if a contribution-target config selects a current-valued
    feature (issue #123 leak rule)."""
    if cfg["data"]["target"] != "contribution":
        return
    used = {
        feat
        for blk in cfg.get("blocks", {}).values()
        for s in blk["sets"]
        for feat in s
    }
    illegal = sorted(used & CURRENT_VALUED)
    if illegal:
        raise ValueError(
            "current-valued features are illegal for the contribution target "
            f"(they read the target's round): {illegal}"
        )


def build_feature_pool(d, switch_every):
    """Return {feature_name: [G, A, T] float array} for the full feature pool.

    `d` is a create_torch_data data dict of tensors; `switch_every` is the
    decision cadence. Semantics per notes/baseline_feature_defs.md: current
    family (no prefix) for the switch target at decision rows, prev family for
    the contribution target."""
    npd = {
        k: d[k].numpy()
        for k in (
            "contribution",
            "punishment",
            "common_good",
            "prev_contribution",
            "prev_punishment",
            "prev_common_good",
            "round_number",
            "prev_agent_group",
            "agent_group",
            "recorded",
            "prev_recorded",
        )
    }
    f = {}
    ga = npd["agent_group"]
    gp = npd["prev_agent_group"]

    # arrival marker: the agent's membership changed vs the previous round.
    # NOT the does_switch label (which sits at the decision row s = arrival-1);
    # window resets and tenure counters key on the physical arrival.
    arrival = np.zeros(ga.shape, dtype=bool)
    arrival[:, :, 1:] = ga[:, :, 1:] != ga[:, :, :-1]

    # ---------------- current family (switch target) ---------------- #
    c = npd["contribution"].astype(float)
    p = npd["punishment"].astype(float)
    cg = npd["common_good"].astype(float)  # own group's per-capita cg
    f["contribution"] = c
    f["punishment"] = p
    f["common_good"] = cg
    f["payoff"] = _payoff(c, p, cg)

    cgrp, coth = {}, {}
    for m, x, loo in (("contribution", c, True), ("punishment", p, True)):
        cgrp[m], coth[m] = _obs_group_means(x, ga, ga, loo=loo)
    cg_oth = _obs_group_means(cg, ga, ga, loo=False)[1]
    f["contribution_mean_group"] = cgrp["contribution"]
    f["punishment_mean_group"] = cgrp["punishment"]
    f["contribution_mean_other"] = coth["contribution"]
    f["punishment_mean_other"] = coth["punishment"]
    f["common_good_other"] = cg_oth
    f["payoff_mean_group"] = _payoff(cgrp["contribution"], cgrp["punishment"], cg)
    f["payoff_mean_other"] = _payoff(coth["contribution"], coth["punishment"], cg_oth)
    f["contribution_mean_gap"] = cgrp["contribution"] - coth["contribution"]
    f["punishment_mean_gap"] = cgrp["punishment"] - coth["punishment"]
    f["common_good_gap"] = cg - cg_oth
    f["payoff_mean_gap"] = f["payoff_mean_group"] - f["payoff_mean_other"]

    own_cur, oth_cur = _obs_group_size(ga, ga, npd["recorded"])
    f["group_size"] = own_cur
    f["group_size_other"] = oth_cur
    f["group_size_delta"] = own_cur - oth_cur

    # current windows: tenure mean INCLUDING the current round; the arrival
    # round starts the new window with the joined group's own outcome.
    for name, series in (
        ("win_contribution_mean_group", cgrp["contribution"]),
        ("win_punishment_mean_group", cgrp["punishment"]),
        ("win_common_good", cg),
        ("win_payoff_mean_group", f["payoff_mean_group"]),
        ("win_group_size", own_cur),
        ("win_contribution_mean_other", coth["contribution"]),
        ("win_punishment_mean_other", coth["punishment"]),
        ("win_common_good_other", cg_oth),
        ("win_payoff_mean_other", f["payoff_mean_other"]),
        ("win_group_size_other", oth_cur),
    ):
        f[name] = _since_switch_window(series, arrival)

    # ---------------- prev family (contribution target) ---------------- #
    f["prev_contribution"] = npd["prev_contribution"].astype(float)
    f["prev_punishment"] = npd["prev_punishment"].astype(float)
    own_cg = npd["prev_common_good"].astype(float)  # own EXPERIENCED t-1 cg
    f["prev_payoff"] = _payoff(f["prev_contribution"], f["prev_punishment"], own_cg)

    # observational: t-1 values over the t-1 roster of the agent's CURRENT
    # group ids; prev_common_good = the current group's t-1 cg, own
    # experienced cg stays in prev_payoff only.
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
    f["prev_contribution_mean_gap"] = grp["contribution"] - oth["contribution"]
    f["prev_punishment_mean_gap"] = grp["punishment"] - oth["punishment"]
    f["prev_common_good_gap"] = grp["common_good"] - oth["common_good"]
    f["prev_payoff_mean_gap"] = (
        f["prev_payoff_mean_group"] - f["prev_payoff_mean_other"]
    )

    # observational sizes (t-1 head-count of the current / opponent id)
    own_prev, oth_prev = _obs_group_size(gp, ga, npd["prev_recorded"])
    # round 0 has no real previous round (prev_recorded all False -> sizes 0);
    # default the prev sizes to the round-0 current sizes (balanced 4/4 start).
    own_prev[:, :, 0] = own_cur[:, :, 0]
    oth_prev[:, :, 0] = oth_cur[:, :, 0]
    f["prev_group_size"] = own_prev
    f["prev_group_size_other"] = oth_prev
    f["prev_group_size_delta"] = own_prev - oth_prev

    # prev windows: the win_* windows as of t-1 -- tenure mean of the
    # prev-observed series (0 at arrival; payoff windowed as its own series
    # so it is 0 there too; size windows keep the arrival value).
    for name, series, inc in (
        ("prev_win_contribution_mean_group", grp["contribution"], False),
        ("prev_win_punishment_mean_group", grp["punishment"], False),
        ("prev_win_common_good", grp["common_good"], False),
        ("prev_win_payoff_mean_group", f["prev_payoff_mean_group"], False),
        ("prev_win_group_size", own_prev, True),
        ("prev_win_contribution_mean_other", oth["contribution"], False),
        ("prev_win_punishment_mean_other", oth["punishment"], False),
        ("prev_win_common_good_other", oth["common_good"], False),
        ("prev_win_payoff_mean_other", f["prev_payoff_mean_other"], False),
        ("prev_win_group_size_other", oth_prev, True),
    ):
        f[name] = _since_switch_window(series, arrival, include_reset=inc)

    # ---------------- structural (shared) ---------------- #
    f["round_number"] = npd["round_number"].astype(float)
    f["rounds_since_switch"] = _rounds_since_switch(arrival)
    f["switched_last_choice"] = _switched_last_choice(arrival, switch_every)

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

    validate_feature_legality(cfg)

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
