"""Feature engineering + data preparation for the hand-crafted linear AH
baselines (issue #119).

Builds the hand-crafted feature pool from the `create_torch_data` tensors and
flattens the TRAIN split into a single [N, n_features] matrix with per-row CV
folds. Consumed by:
  * scripts/baselines/run_baseline_cv.py     -- the CV grid driver
  * scripts/baselines/inspect_best_model.py  -- coefficient inspection

Design (see doc/plans/119-handcrafted-linear-baselines.md):
  * No src/ changes -- only create_torch_data + get_cross_validations are used;
    every derived feature is computed here from the raw [G, A, T] tensors.
  * All behavioural features are previous-round (t-1). Group-mean / gap / window
    features use PREVIOUS-round group membership (prev_agent_group); structural
    `group_size` uses current membership, `prev_group_size*` use previous.
  * payoff = 20 - contribution - punishment + common_good (reports/basics.md;
    common_good is already the per-capita share).

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
    """Leave-one-out own-group mean (`peers`) and other-group mean (`other`) of
    `measure`, grouped by `group` membership. Both inputs are t-1 tensors, so the
    result is the previous-round peers' mean.

    A genuinely empty other group (everyone merged into one sub-group mid-game)
    is treated as a 0-sized, all-zero group -> `other = 0`, so the gap (peers -
    other) reflects the real 'other group emptied out' asymmetry. The round-0
    symmetric default (no real previous round) is handled in build_feature_pool."""
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


def _group_full_mean(measure, group):
    """Full own-group mean of `measure` (INCLUDES self): each member gets their
    group's t-1 mean. Used for group-level payoff -- payoff is a group quantity
    (shared common_good), so a leave-one-out peer mean is not meaningful."""
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


def _since_switch_window(per_round_mean, does_switch):
    """Running mean of a per-round (t-1) group-mean feature over the agent's
    current tenure, resetting the accumulator whenever the agent switches
    (does_switch). Leakage-safe: it averages only t-1-and-earlier group means."""
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
    decision cadence (for switched_last_choice). prev_common_good_mean_peers is
    computed internally (for peer payoff) but not exposed -- group-level cg
    equals the own-group prev_common_good."""
    npd = {
        k: d[k].numpy()
        for k in (
            "prev_contribution", "prev_punishment", "prev_common_good",
            "round_number", "does_switch",
            "prev_agent_group", "agent_group", "recorded", "prev_recorded",
        )
    }
    f = {}

    # -- direct from tensors --
    f["prev_contribution"] = npd["prev_contribution"].astype(float)
    f["prev_punishment"] = npd["prev_punishment"].astype(float)
    f["prev_common_good"] = npd["prev_common_good"].astype(float)
    f["round_number"] = npd["round_number"].astype(float)

    # -- group means (prev-round membership) --
    gp = npd["prev_agent_group"]
    peers, other = {}, {}
    for m in ("contribution", "punishment", "common_good"):
        peers[m], other[m] = _group_prev_means(npd[f"prev_{m}"], gp)
        # Round 0 has no real previous round (prev_agent_group defaults everyone
        # to group 0 -> empty other). Use the symmetric start default: other =
        # peers, so every gap is 0 (both groups look identical at kick-off).
        other[m][:, :, 0] = peers[m][:, :, 0]
    for m in ("contribution", "punishment"):
        f[f"prev_{m}_mean_peers"] = peers[m]
    for m in ("contribution", "punishment", "common_good"):
        f[f"prev_{m}_mean_other"] = other[m]
        f[f"prev_{m}_mean_gap"] = peers[m] - other[m]

    # -- payoff (self / peers / other / gap / full group) --
    f["prev_payoff"] = _payoff(
        f["prev_contribution"], f["prev_punishment"], f["prev_common_good"]
    )
    f["prev_payoff_mean_peers"] = _payoff(
        peers["contribution"], peers["punishment"], peers["common_good"]
    )
    f["prev_payoff_mean_other"] = _payoff(
        other["contribution"], other["punishment"], other["common_good"]
    )
    f["prev_payoff_mean_gap"] = f["prev_payoff_mean_peers"] - f["prev_payoff_mean_other"]
    # full own-group mean payoff (incl self): payoff is group-level (shared cg),
    # so the whole group's mean is meaningful where a peer LOO mean is not.
    f["prev_payoff_mean_group"] = _payoff(
        _group_full_mean(npd["prev_contribution"], gp),
        _group_full_mean(npd["prev_punishment"], gp),
        f["prev_common_good"],
    )

    # -- since-switch windows --
    ds = npd["does_switch"]
    for m in ("contribution", "punishment", "common_good"):
        f[f"win_{m}_mean_peers"] = _since_switch_window(peers[m], ds)
        f[f"win_{m}_mean_other"] = _since_switch_window(other[m], ds)
    f["win_payoff_mean_peers"] = _payoff(
        f["win_contribution_mean_peers"],
        f["win_punishment_mean_peers"],
        f["win_common_good_mean_peers"],
    )
    f["win_payoff_mean_other"] = _payoff(
        f["win_contribution_mean_other"],
        f["win_punishment_mean_other"],
        f["win_common_good_mean_other"],
    )

    # -- structural sizes / counters --
    own_cur = _group_size(npd["agent_group"], npd["recorded"])
    total_cur = npd["recorded"].astype(float).sum(axis=1, keepdims=True)
    own_prev = _group_size(gp, npd["prev_recorded"])
    prev_total = npd["prev_recorded"].astype(float).sum(axis=1, keepdims=True)
    oth_prev = prev_total - own_prev
    # round 0 has no real previous round (prev_recorded all False -> sizes 0);
    # default the prev sizes to the round-0 current sizes (the balanced 4/4 start).
    own_prev[:, :, 0] = own_cur[:, :, 0]
    oth_prev[:, :, 0] = (total_cur - own_cur)[:, :, 0]
    f["group_size"] = own_cur
    f["prev_group_size"] = own_prev
    f["prev_group_size_other"] = oth_prev
    f["prev_group_size_delta"] = own_prev - oth_prev
    f["rounds_since_switch"] = _rounds_since_switch(ds)
    f["switched_last_choice"] = _switched_last_choice(ds, switch_every)

    # Leakage-safe (strictly t-1) variants for the SWITCH target: the current-round
    # decision / window-reset above encode does_switch[t] (the target), so provide
    # variants that only ever use does_switch[<t].
    f["prev_switched_last_choice"] = _switched_last_choice(
        ds, switch_every, strict_prev=True)  # switch at the PREVIOUS decision

    def _shift1(a):  # value as of t-1 (round 0 keeps its own value; masked anyway)
        out = np.roll(a, 1, axis=2)
        out[:, :, 0] = a[:, :, 0]
        return out

    for name in ("win_contribution_mean_peers", "win_punishment_mean_peers",
                 "win_common_good_mean_peers", "win_payoff_mean_peers",
                 "win_contribution_mean_other", "win_punishment_mean_other",
                 "win_common_good_mean_other", "win_payoff_mean_other"):
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
    data, _, pair_id = create_torch_data(df, switch_every=switch_every)
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

    return dict(X=X, col_of=col_of, y_cat=tgt.astype(int),
                y_cont=tgt.astype(float), fold_row=fold_row)
