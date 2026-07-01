"""Shared harness for the hand-crafted linear AH baselines (issue #119).

Builds the 30-feature hand-crafted pool from the `create_torch_data` tensors and
(TODO, next checkpoint) runs the block-level nested-CV grid search with a locked
holdout and 1-SE selection. Imported by the thin per-target entrypoints
`contribution_handcrafted.py` / `switch_handcrafted.py`; the minimal
`contribution_baseline.py` / `switch_logit_baseline.py` are left untouched.

Design (see doc/plans/119-handcrafted-linear-baselines.md):
  * No src/ changes -- only create_torch_data + get_cross_validations are used;
    every derived feature is computed here from the raw [G, A, T] tensors.
  * All behavioural features are previous-round (t-1). Group-mean / gap / window
    features use PREVIOUS-round group membership (prev_agent_group), matching the
    `prev_*_mean_*` naming and the existing `group_prev_means` helper. Structural
    `group_size` (B7) uses current membership; `prev_group_size*` use prev.
  * payoff = 20 - contribution - punishment + common_good (reports/basics.md;
    common_good is already the per-capita share). Linear in {c, p, cg}, hence
    only ever used in the `compact` encoding, never alongside its components.

Runs locally (CPU torch, no PyG). Self-test:
    .venv/bin/python scripts/baselines/handcrafted_grid.py
"""
import os

os.environ.setdefault("DISABLE_PANDERA_IMPORT_WARNING", "True")
import numpy as np
import yaml

ENDOWMENT = 20.0  # per-round private endowment (reports/basics.md)


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
def load_config(path):
    with open(path) as fh:
        return yaml.safe_load(fh)


def config_feature_names(cfg):
    """Every distinct feature name referenced by any block/encoding in cfg."""
    names = set()
    for block in cfg["blocks"].values():
        for enc in ("components", "compact"):
            names.update(block.get(enc, []))
    return names


def load_episodes(cfg, root):
    """Load the experiment rows, dropping the pair-flip copies when
    `data.exclude_flipped` is set (train on the 50 real episodes, not 100)."""
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
    `measure`, grouped by `group` membership. Generalizes group_prev_means from
    contribution_baseline.py to any measure; both inputs are t-1 tensors, so the
    result is the previous-round peers' mean.

    A genuinely empty other group (everyone merged into one sub-group mid-game)
    is treated as a 0-sized, all-zero group -> `other = 0`, so the gap (peers -
    other) reflects the real 'other group emptied out' asymmetry. The round-0
    symmetric default (no real previous round) is handled separately in
    build_feature_pool."""
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


def _switched_last_choice(does_switch, switch_every):
    """'Did I switch last time I had the choice?' -- the agent's does_switch
    value at its most recent decision round (r % switch_every == 0, r != 0,
    r <= t), forward-filled. 0 before the first decision round. Uses the
    current round when t is itself a decision round (group membership is known
    before contributions, so this is a valid covariate, not leakage)."""
    ds = does_switch.astype(int)
    _, _, T = ds.shape
    out = np.zeros_like(ds, dtype=float)
    for t in range(T):
        last_dec = (t // switch_every) * switch_every  # largest multiple <= t
        if last_dec != 0:  # 0 => no decision has happened yet
            out[:, :, t] = ds[:, :, last_dec]
    return out


# --------------------------------------------------------------------------- #
# feature pool
# --------------------------------------------------------------------------- #
def build_feature_pool(d, switch_every):
    """Return {feature_name: [G, A, T] float array} for the full 30-feature pool.

    `d` is a create_torch_data data dict of tensors; `switch_every` is the
    decision cadence (for switched_last_choice). prev_common_good_mean_peers is
    computed internally (for peer payoff) but not exposed -- B2 omits it since
    group-level cg equals B1's prev_common_good."""
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

    # -- payoff (self / peers / other / gap) --
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

    return f


# --------------------------------------------------------------------------- #
# self-test: build the pool on the real data and validate against the config
# --------------------------------------------------------------------------- #
def _self_test():
    import random
    from pathlib import Path

    import torch as th

    from aimanager.generic.data import create_torch_data

    root = Path(__file__).resolve().parents[2]
    cfg = load_config(
        root / "configs/training/baselines/contribution/handcrafted_grid.yml"
    )
    seed = cfg["cv"]["seed"]
    th.random.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    df = load_episodes(cfg, root)
    data, _, pair_id = create_torch_data(df)
    G, A, T = data["contribution"].shape
    print(f"episodes={G} (exclude_flipped={cfg['data'].get('exclude_flipped')}), "
          f"agents={A}, rounds={T}, pairs={len(set(pair_id.tolist()))}")

    pool = build_feature_pool(data, cfg["data"]["switch_every"])
    wanted = config_feature_names(cfg)
    missing = wanted - set(pool)
    extra = set(pool) - wanted
    assert not missing, f"config features not built: {sorted(missing)}"
    print(f"\nbuilt {len(pool)} features; config references {len(wanted)}; "
          f"internal-only (not in config): {sorted(extra) or 'none'}")

    print("\n{:<30} {:>8} {:>8} {:>8}  {}".format(
        "feature", "min", "max", "mean", "finite?"))
    for name in sorted(pool):
        a = pool[name]
        ok = np.isfinite(a).all()
        flag = "OK" if ok else "!! NON-FINITE"
        print("{:<30} {:>8.2f} {:>8.2f} {:>8.2f}  {}".format(
            name, a.min(), a.max(), a.mean(), flag))
        assert ok, f"{name} has non-finite values"
    print("\nall features finite and shape [G, A, T] =", pool["prev_contribution"].shape)


if __name__ == "__main__":
    _self_test()
