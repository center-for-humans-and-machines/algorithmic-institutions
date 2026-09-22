"""Integration test for the hand-crafted baseline feature pipeline (issue #119,
re-anchored per #123).

Verifies that build_feature_pool (src) produces exactly the features it claims,
by comparing every feature for one agent across a whole episode against an
INDEPENDENT pandas reference computed here from the frozen raw fixture -- no src
code is used to build the reference, so this is a genuine cross-check. Two
families (notes/baseline_feature_defs.md): the CURRENT family (no prefix,
switch target at the pre-switch decision row) aggregates round-t values over
the round-t roster; the PREV family (contribution target) aggregates t-1
values over the t-1 roster of the CURRENT group ids.

Two src sides are checked against that one reference:
  * build_feature_pool  -- the batch pipeline (CV / training).
  * LinearAHAdapter      -- the simulation adapter, which rebuilds features ONE
    ROUND AT A TIME from env state (#121). Replayed twice: as the contribution
    model (called before round t is played -> prev family + membership-derived
    features checked) and as the switch model (called at the END of round t
    with realised current values -> all features checked).

Target: episode 70 (global_group_id 'rokqh2fp #2'), player 6, from the non-flipped
originals of experiments/2group_8agent_50ep.csv. The fixture under fixtures/ was
extracted with pure pandas (see git history for the one-off extract script):
  * episode_raw.csv        -- all 8 agents' raw per-round fields (wide)

Run:  .venv/bin/python -m pytest tests/baselines/test_baseline_features.py
Eyeball:  .venv/bin/python tests/baselines/test_baseline_features.py
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("DISABLE_PANDERA_IMPORT_WARNING", "True")
import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]  # tests/baselines -> repo root
DATA = HERE / "fixtures"
RAW = ROOT / "experiments/2group_8agent_50ep.csv"
TARGET, EPISODE_ID, EXPERIMENT, SWITCH_EVERY = 6, 70, "ah_group_switching", 4

# prev family (contribution target)
B1 = ["prev_contribution", "prev_punishment", "prev_payoff"]
B2 = [
    "prev_contribution_mean_group",
    "prev_punishment_mean_group",
    "prev_payoff_mean_group",
    "prev_group_size",
    "prev_common_good",
]
B3 = [
    "prev_contribution_mean_other",
    "prev_punishment_mean_other",
    "prev_common_good_other",
    "prev_payoff_mean_other",
    "prev_group_size_other",
]
B4 = [
    "prev_contribution_mean_gap",
    "prev_punishment_mean_gap",
    "prev_common_good_gap",
    "prev_payoff_mean_gap",
    "prev_group_size_delta",
]
B5 = [
    "prev_win_contribution_mean_group",
    "prev_win_punishment_mean_group",
    "prev_win_common_good",
    "prev_win_payoff_mean_group",
    "prev_win_group_size",
]
B6 = [
    "prev_win_contribution_mean_other",
    "prev_win_punishment_mean_other",
    "prev_win_common_good_other",
    "prev_win_payoff_mean_other",
    "prev_win_group_size_other",
]
# current family (switch target at the decision row)
C1 = ["contribution", "punishment", "payoff"]
C2 = [
    "contribution_mean_group",
    "punishment_mean_group",
    "payoff_mean_group",
    "common_good",
    "group_size",
]
C3 = [
    "contribution_mean_other",
    "punishment_mean_other",
    "payoff_mean_other",
    "common_good_other",
    "group_size_other",
]
C4 = [
    "contribution_mean_gap",
    "punishment_mean_gap",
    "payoff_mean_gap",
    "common_good_gap",
    "group_size_delta",
]
C5 = [
    "win_contribution_mean_group",
    "win_punishment_mean_group",
    "win_common_good",
    "win_payoff_mean_group",
    "win_group_size",
]
C6 = [
    "win_contribution_mean_other",
    "win_punishment_mean_other",
    "win_common_good_other",
    "win_payoff_mean_other",
    "win_group_size_other",
]
# structural (shared)
B7 = ["round_number", "rounds_since_switch", "switched_last_choice", "is_first"]

PREV_FAMILY = B1 + B2 + B3 + B4 + B5 + B6
CUR_FAMILY = C1 + C2 + C3 + C4 + C5 + C6
ALL_FEATURES = PREV_FAMILY + CUR_FAMILY + B7
# what the contribution adapter can compute before round t is played: the prev
# family plus membership-derived current features (sizes / counters)
CONTRIB_SAFE = (
    PREV_FAMILY
    + B7
    + [
        "group_size",
        "group_size_other",
        "group_size_delta",
        "win_group_size",
        "win_group_size_other",
    ]
)


# --------------------------------------------------------------------------- #
# independent (pandas-only) reference
# --------------------------------------------------------------------------- #
def _dataset_defaults():
    df = pd.read_csv(RAW)
    df = df[df["experiment_name"] == EXPERIMENT]
    df = df[~df["global_group_id"].str.contains("(flipped)", regex=False)].copy()
    valid = df["player_no_input"] == 0
    c_def = float(np.rint(df.loc[valid, "contribution"].median()))
    p_def = float(np.rint(df.loc[df["manager_no_input"] == 0, "punishment"].median()))
    n_valid = (
        df.assign(cv=valid.astype(int))
        .groupby(["episode_id", "round_number", "group_id"])["cv"]
        .transform("sum")
    )
    cg_def = float((df["common_good"] / n_valid)[valid].median())
    return c_def, p_def, cg_def


def _percap_series(bank):
    """Own EXPERIENCED per-capita cg at each round (own group's pool / n_valid)."""
    b = bank.set_index("round_number")
    T = len(b)
    percap = np.zeros(T)
    for t in range(T):
        g = b.loc[t, f"p{TARGET}_grp"]
        members = [j for j in range(8) if b.loc[t, f"p{j}_grp"] == g]
        nv = sum(int(b.loc[t, f"p{j}_noinput"] == 0) for j in members)
        percap[t] = float(b.loc[t, f"p{TARGET}_common_good"]) / nv if nv else 0.0
    return percap


def _add_self(ref, bank, c_def, p_def, cg_def):
    c = bank[f"p{TARGET}_contribution"].to_numpy(float)
    p = bank[f"p{TARGET}_punishment"].to_numpy(float)
    percap = _percap_series(bank)
    ref["ref_contribution"] = c
    ref["ref_punishment"] = p
    ref["ref_payoff"] = 20.0 - c - p + percap
    ref["ref_prev_contribution"] = np.concatenate([[c_def], c[:-1]])
    ref["ref_prev_punishment"] = np.concatenate([[p_def], p[:-1]])
    ref["ref_prev_payoff"] = (
        20.0
        - ref["ref_prev_contribution"]
        - ref["ref_prev_punishment"]
        + np.concatenate([[cg_def], percap[:-1]])
    )


def _roster_side(b, t, gid, loo_self):
    """(c_mean, p_mean, cg, size) of group `gid` at round `t` from the raw bank;
    self excluded from the means when loo_self."""
    roster = [j for j in range(8) if b.loc[t, f"p{j}_grp"] == gid]
    mem = [j for j in roster if not (loo_self and j == TARGET)]

    def mean(field):
        return float(np.mean([b.loc[t, f"p{j}_{field}"] for j in mem])) if mem else 0.0

    if roster:
        nv = sum(int(b.loc[t, f"p{j}_noinput"] == 0) for j in roster)
        pool = float(b.loc[t, f"p{roster[0]}_common_good"])
        cg = pool / nv if nv else 0.0
    else:
        cg = 0.0
    return mean("contribution"), mean("punishment"), cg, float(len(roster))


def _add_cur(ref, bank):
    """Current family: round-t values over the round-t roster keyed to the
    target's group at t; self excluded on the group side."""
    b = bank.set_index("round_number")
    T = len(b)
    per = {}
    for t in range(T):
        g_t = b.loc[t, f"p{TARGET}_grp"]
        c_g, p_g, cg_g, n_g = _roster_side(b, t, g_t, loo_self=True)
        c_o, p_o, cg_o, n_o = _roster_side(b, t, 1 - g_t, loo_self=False)
        per[t] = {
            "c_grp": c_g,
            "p_grp": p_g,
            "cg_grp": cg_g,
            "size_grp": n_g,
            "c_oth": c_o,
            "p_oth": p_o,
            "cg_oth": cg_o,
            "size_oth": n_o,
        }

    def col(key):
        return ref["round_number"].map(lambda t: per[int(t)][key])

    ref["ref_contribution_mean_group"] = col("c_grp")
    ref["ref_punishment_mean_group"] = col("p_grp")
    ref["ref_common_good"] = col("cg_grp")
    ref["ref_group_size"] = col("size_grp")
    ref["ref_contribution_mean_other"] = col("c_oth")
    ref["ref_punishment_mean_other"] = col("p_oth")
    ref["ref_common_good_other"] = col("cg_oth")
    ref["ref_group_size_other"] = col("size_oth")
    ref["ref_payoff_mean_group"] = (
        20.0
        - ref["ref_contribution_mean_group"]
        - ref["ref_punishment_mean_group"]
        + ref["ref_common_good"]
    )
    ref["ref_payoff_mean_other"] = (
        20.0
        - ref["ref_contribution_mean_other"]
        - ref["ref_punishment_mean_other"]
        + ref["ref_common_good_other"]
    )
    ref["ref_contribution_mean_gap"] = (
        ref["ref_contribution_mean_group"] - ref["ref_contribution_mean_other"]
    )
    ref["ref_punishment_mean_gap"] = (
        ref["ref_punishment_mean_group"] - ref["ref_punishment_mean_other"]
    )
    ref["ref_common_good_gap"] = ref["ref_common_good"] - ref["ref_common_good_other"]
    ref["ref_payoff_mean_gap"] = (
        ref["ref_payoff_mean_group"] - ref["ref_payoff_mean_other"]
    )
    ref["ref_group_size_delta"] = ref["ref_group_size"] - ref["ref_group_size_other"]


def _add_obs(ref, bank, c_def, p_def, cg_def):
    """Observational prev Group/Other reference: t-1 values over the t-1 ROSTER
    of the target's CURRENT group ids. Group side excludes self (when self was
    in that roster); other side is the full roster mean; cg is the roster's
    single shared per-capita value. Round 0: defaults, other = group, gaps 0,
    sizes = current sizes."""
    b = bank.set_index("round_number")
    T = len(b)
    gsize0 = int(
        sum(b.loc[0, f"p{j}_grp"] == b.loc[0, f"p{TARGET}_grp"] for j in range(8))
    )
    per = {
        0: {
            "c_grp": c_def,
            "p_grp": p_def,
            "cg_grp": cg_def,
            "c_oth": c_def,
            "p_oth": p_def,
            "cg_oth": cg_def,
            "size_grp": float(gsize0),
            "size_oth": float(8 - gsize0),
        }
    }
    for t in range(1, T):
        g_t = b.loc[t, f"p{TARGET}_grp"]
        c_g, p_g, cg_g, n_g = _roster_side(b, t - 1, g_t, loo_self=True)
        c_o, p_o, cg_o, n_o = _roster_side(b, t - 1, 1 - g_t, loo_self=False)
        per[t] = {
            "c_grp": c_g,
            "p_grp": p_g,
            "cg_grp": cg_g,
            "size_grp": n_g,
            "c_oth": c_o,
            "p_oth": p_o,
            "cg_oth": cg_o,
            "size_oth": n_o,
        }

    def col(key):
        return ref["round_number"].map(lambda t: per[int(t)][key])

    ref["ref_prev_contribution_mean_group"] = col("c_grp")
    ref["ref_prev_punishment_mean_group"] = col("p_grp")
    ref["ref_prev_common_good"] = col("cg_grp")
    ref["ref_prev_contribution_mean_other"] = col("c_oth")
    ref["ref_prev_punishment_mean_other"] = col("p_oth")
    ref["ref_prev_common_good_other"] = col("cg_oth")
    ref["ref_prev_group_size"] = col("size_grp")
    ref["ref_prev_group_size_other"] = col("size_oth")
    ref["ref_prev_payoff_mean_group"] = (
        20.0
        - ref["ref_prev_contribution_mean_group"]
        - ref["ref_prev_punishment_mean_group"]
        + ref["ref_prev_common_good"]
    )
    ref["ref_prev_payoff_mean_other"] = (
        20.0
        - ref["ref_prev_contribution_mean_other"]
        - ref["ref_prev_punishment_mean_other"]
        + ref["ref_prev_common_good_other"]
    )
    ref["ref_prev_contribution_mean_gap"] = (
        ref["ref_prev_contribution_mean_group"]
        - ref["ref_prev_contribution_mean_other"]
    )
    ref["ref_prev_punishment_mean_gap"] = (
        ref["ref_prev_punishment_mean_group"] - ref["ref_prev_punishment_mean_other"]
    )
    ref["ref_prev_common_good_gap"] = (
        ref["ref_prev_common_good"] - ref["ref_prev_common_good_other"]
    )
    ref["ref_prev_payoff_mean_gap"] = (
        ref["ref_prev_payoff_mean_group"] - ref["ref_prev_payoff_mean_other"]
    )
    ref["ref_prev_group_size_delta"] = (
        ref["ref_prev_group_size"] - ref["ref_prev_group_size_other"]
    )


def _switch_series(bank):
    """Arrival marker: the target's membership changed vs the previous round."""
    g = bank[f"p{TARGET}_grp"].to_numpy()
    ds = np.zeros(len(g), bool)
    ds[1:] = g[1:] != g[:-1]
    return ds


def _window(series, ds):
    """Windows that include the reset round's value (current family + sizes)."""
    out = np.zeros(len(series))
    s, c = 0.0, 0
    for t in range(len(series)):
        if ds[t]:
            s, c = 0.0, 0
        s += series[t]
        c += 1
        out[t] = s / c
    return out


def _window0(series, ds):
    """Prev value windows: 0 at the arrival round (strictly within-tenure)."""
    out = np.zeros(len(series))
    s, c = 0.0, 0
    for t in range(len(series)):
        if ds[t]:
            s, c = 0.0, 0
            out[t] = 0.0
            continue
        s += series[t]
        c += 1
        out[t] = s / c
    return out


def _add_prev_windows(ref, ds, side):
    """B5/B6: windows over the OBSERVATIONAL prev series -- the win_* windows
    as of t-1. Value windows are strictly within-tenure (0 at the arrival
    round; payoff windowed as its own series so it is 0 there too); size
    windows keep their arrival value."""
    sfx = "group" if side == "group" else "other"
    cgcol = "ref_prev_common_good" if side == "group" else "ref_prev_common_good_other"
    sizecol = "ref_prev_group_size" if side == "group" else "ref_prev_group_size_other"
    names = {
        f"prev_win_contribution_mean_{sfx}": _window0(
            ref[f"ref_prev_contribution_mean_{sfx}"].to_numpy(float), ds
        ),
        f"prev_win_punishment_mean_{sfx}": _window0(
            ref[f"ref_prev_punishment_mean_{sfx}"].to_numpy(float), ds
        ),
        (
            "prev_win_common_good" if side == "group" else "prev_win_common_good_other"
        ): _window0(ref[cgcol].to_numpy(float), ds),
        f"prev_win_payoff_mean_{sfx}": _window0(
            ref[f"ref_prev_payoff_mean_{sfx}"].to_numpy(float), ds
        ),
        (
            "prev_win_group_size" if side == "group" else "prev_win_group_size_other"
        ): _window(ref[sizecol].to_numpy(float), ds),
    }
    for name, arr in names.items():
        ref[f"ref_{name}"] = arr


def _add_cur_windows(ref, ds, side):
    """C5/C6: windows over the CURRENT series, INCLUDING the current round;
    the arrival round starts the new window with the joined group's outcome."""
    sfx = "group" if side == "group" else "other"
    cgcol = "ref_common_good" if side == "group" else "ref_common_good_other"
    sizecol = "ref_group_size" if side == "group" else "ref_group_size_other"
    names = {
        f"win_contribution_mean_{sfx}": _window(
            ref[f"ref_contribution_mean_{sfx}"].to_numpy(float), ds
        ),
        f"win_punishment_mean_{sfx}": _window(
            ref[f"ref_punishment_mean_{sfx}"].to_numpy(float), ds
        ),
        ("win_common_good" if side == "group" else "win_common_good_other"): _window(
            ref[cgcol].to_numpy(float), ds
        ),
        f"win_payoff_mean_{sfx}": _window(
            ref[f"ref_payoff_mean_{sfx}"].to_numpy(float), ds
        ),
        ("win_group_size" if side == "group" else "win_group_size_other"): _window(
            ref[sizecol].to_numpy(float), ds
        ),
    }
    for name, arr in names.items():
        ref[f"ref_{name}"] = arr


def _add_b7(ref, ds):
    T = len(ref)
    ref["ref_round_number"] = np.arange(T, dtype=float)
    rss = np.zeros(T)
    cnt = 0
    for t in range(T):
        cnt = 0 if (t == 0 or ds[t]) else cnt + 1
        rss[t] = cnt
    ref["ref_rounds_since_switch"] = rss
    slc = np.zeros(T)
    for t in range(T):
        last_arr = (t // SWITCH_EVERY) * SWITCH_EVERY
        if last_arr > 0:
            slc[t] = float(ds[last_arr])
    ref["ref_switched_last_choice"] = slc
    ref["ref_is_first"] = (np.arange(T) == 0).astype(float)


def build_reference():
    bank = pd.read_csv(DATA / "episode_raw.csv")
    c_def, p_def, cg_def = _dataset_defaults()
    ref = pd.DataFrame({"round_number": bank["round_number"]})
    ds = _switch_series(bank)
    _add_self(ref, bank, c_def, p_def, cg_def)
    _add_cur(ref, bank)
    _add_obs(ref, bank, c_def, p_def, cg_def)
    for side in ("group", "other"):
        _add_prev_windows(ref, ds, side)
        _add_cur_windows(ref, ds, side)
    _add_b7(ref, ds)
    return ref


# --------------------------------------------------------------------------- #
# pipeline side (src)
# --------------------------------------------------------------------------- #
def pipeline_features():
    import random
    import torch as th

    sys.path.insert(0, str(ROOT / "scripts/baselines"))
    from handcrafted_grid import build_feature_pool
    from aimanager.generic.data import create_torch_data

    random.seed(38381)
    np.random.seed(38381)
    th.manual_seed(38381)
    df = pd.read_csv(RAW)
    df = df[df["experiment_name"] == EXPERIMENT]
    df = df[~df["global_group_id"].str.contains("(flipped)", regex=False)]
    pair = int(df.loc[df["episode_id"] == EPISODE_ID, "pair_id"].iloc[0])
    data, _, pair_id = create_torch_data(df, switch_every=SWITCH_EVERY)
    pool = build_feature_pool(data, SWITCH_EVERY)
    g = int(np.where(np.asarray(pair_id) == pair)[0][0])
    return {name: np.asarray(pool[name][g, TARGET, :], float) for name in ALL_FEATURES}


@pytest.fixture(scope="module")
def compared():
    ref = build_reference()
    pipe = pipeline_features()
    return {
        name: (ref[f"ref_{name}"].to_numpy(float), pipe[name]) for name in ALL_FEATURES
    }


@pytest.mark.parametrize("feature", ALL_FEATURES)
def test_feature_matches_pipeline(compared, feature):
    ref, pipe = compared[feature]
    bad = np.where(~np.isclose(ref, pipe, atol=1e-6))[0]
    assert len(bad) == 0, f"{feature} mismatch at rounds {bad.tolist()}"


# --------------------------------------------------------------------------- #
# sim-adapter side (src): the simulation feeds the LinearAHAdapter ONE ROUND AT
# A TIME from env state, and the adapter rebuilds features incrementally from the
# accumulated history. Replay the episode through it (states derived from the
# realised per-round values, as the env would provide) and collect the per-round
# feature vector, to confirm the incremental reconstruction matches the pipeline.
# The contribution model is called BEFORE round t is played (prev family +
# membership-derived features available); the switch model at the END of round t
# (current values realised) -> every feature available.
# --------------------------------------------------------------------------- #
def episode_states(target):
    """Teacher-forced env-like states for the fixture episode, one per round:
    ``prev_*`` = the realised round t-1 values, ``agent_group`` = post-arrival
    membership. The switch target additionally gets round t's realised current
    values (it is called at the END of round t). Returns
    ``(states, n_agents, default_values)``; reused by the mlp adapter test."""
    import random

    import torch as th

    sys.path.insert(0, str(ROOT / "scripts/baselines"))
    from aimanager.generic.data import create_torch_data

    random.seed(38381)
    np.random.seed(38381)
    th.manual_seed(38381)
    df = pd.read_csv(RAW)
    df = df[df["experiment_name"] == EXPERIMENT]
    df = df[~df["global_group_id"].str.contains("(flipped)", regex=False)]
    pair = int(df.loc[df["episode_id"] == EPISODE_ID, "pair_id"].iloc[0])
    data, default_values, pair_id = create_torch_data(df, switch_every=SWITCH_EVERY)
    g = int(np.where(np.asarray(pair_id) == pair)[0][0])
    A, T = data["contribution"].shape[1], data["contribution"].shape[2]

    states = []
    for t in range(T):
        state = {  # env-like state: prev_* = realised t-1, agent_group = current
            "round_number": th.full((1, A, 1), t, dtype=th.int64),
            "prev_contribution": data["prev_contribution"][g, :, t].reshape(1, A, 1),
            "prev_punishment": data["prev_punishment"][g, :, t].reshape(1, A, 1),
            "prev_common_good": data["prev_common_good"][g, :, t].reshape(1, A, 1),
            "prev_agent_group": data["prev_agent_group"][g, :, t].reshape(1, A, 1),
            "agent_group": data["agent_group"][g, :, t].reshape(1, A, 1),
        }
        if target == "does_switch":
            # switch model is called at the END of round t: current realised
            state.update(
                {
                    "contribution": data["contribution"][g, :, t].reshape(1, A, 1),
                    "punishment": data["punishment"][g, :, t].reshape(1, A, 1),
                    "common_good": data["common_good"][g, :, t].reshape(1, A, 1),
                }
            )
        states.append(state)
    return states, A, default_values


def adapter_features(target):
    import torch as th

    sys.path.insert(0, str(ROOT / "scripts/baselines"))
    from aimanager.simulation.linear_ah import LinearAHAdapter

    states, A, default_values = episode_states(target)
    T = len(states)

    # only the fields LinearAHAdapter reads to rebuild features are needed
    bundle = {
        "model": "ridge",
        "estimator": None,
        "scaler": None,
        "features": [],
        "target": target,
        "n_levels": 0,
        "switch_every": SWITCH_EVERY,
        "default_values": {
            k: (float(v) if hasattr(v, "__float__") else v)
            for k, v in default_values.items()
        },
    }
    ad = LinearAHAdapter(
        bundle, n_agents=A, n_contributions=21, device=th.device("cpu")
    )

    features = ALL_FEATURES if target == "does_switch" else CONTRIB_SAFE
    collected = {name: np.zeros(T) for name in features}
    for t in range(T):
        if t == 0:
            ad._reset_history()
        ad._record(states[t], t)
        pool = ad._build_pool(t)
        for name in features:
            collected[name][t] = pool[name][0, TARGET, t]
    return collected


@pytest.fixture(scope="module")
def adapter_compared():
    ref = build_reference()
    ada = adapter_features("contribution")
    return {
        name: (ref[f"ref_{name}"].to_numpy(float), ada[name]) for name in CONTRIB_SAFE
    }


@pytest.mark.parametrize("feature", CONTRIB_SAFE)
def test_adapter_matches_reference(adapter_compared, feature):
    # atol=1e-4: the adapter rebuilds via float32 tensors + float64 round-0
    # defaults, so common_good-derived features differ by ~1e-7 (float32 eps).
    ref, ada = adapter_compared[feature]
    bad = np.where(~np.isclose(ref, ada, atol=1e-4))[0]
    assert len(bad) == 0, f"adapter {feature} mismatch at rounds {bad.tolist()}"


@pytest.fixture(scope="module")
def switch_adapter_compared():
    ref = build_reference()
    ada = adapter_features("does_switch")
    return {
        name: (ref[f"ref_{name}"].to_numpy(float), ada[name]) for name in ALL_FEATURES
    }


@pytest.mark.parametrize("feature", ALL_FEATURES)
def test_switch_adapter_matches_reference(switch_adapter_compared, feature):
    ref, ada = switch_adapter_compared[feature]
    bad = np.where(~np.isclose(ref, ada, atol=1e-4))[0]
    assert len(bad) == 0, f"switch adapter {feature} mismatch at rounds {bad.tolist()}"


# --------------------------------------------------------------------------- #
# punishment-manager side (src): the rounds-driven LinearAHAdapter entry (#127).
# Replay the episode as the sim would (rounds < t complete, round t punishments
# still None); this path also RECOMPUTES per-capita common good (env formula),
# so the replay cross-checks that reconstruction against the recorded data.
# --------------------------------------------------------------------------- #
def manager_features():
    import torch as th  # noqa: F401  (linear_ah imports torch)

    sys.path.insert(0, str(ROOT / "scripts/baselines"))
    from aimanager.simulation.linear_ah import LinearAHAdapter

    bank = pd.read_csv(DATA / "episode_raw.csv").set_index("round_number")
    T = len(bank)
    c_def, p_def, cg_def = _dataset_defaults()
    bundle = {  # only the fields the adapter reads to rebuild features
        "model": "ridge",
        "estimator": None,
        "scaler": None,
        "features": [],
        "target": "punishment",
        "n_levels": 31,
        "switch_every": SWITCH_EVERY,
        "default_values": {
            "contribution": c_def,
            "punishment": p_def,
            "common_good": cg_def,
        },
    }
    mgr = LinearAHAdapter(bundle)

    def cell(t, j, field):
        v = bank.loc[t, f"p{j}_{field}"]
        return None if pd.isna(v) else float(v)

    full = [
        {
            "contribution": [cell(t, j, "contribution") for j in range(8)],
            "contribution_valid": [
                bool(bank.loc[t, f"p{j}_noinput"] == 0) for j in range(8)
            ],
            "punishment": [cell(t, j, "punishment") for j in range(8)],
            "punishment_valid": [True] * 8,
            "agent_group": [int(bank.loc[t, f"p{j}_grp"]) for j in range(8)],
            "round": t,
        }
        for t in range(T)
    ]

    collected = {name: np.zeros(T) for name in CONTRIB_SAFE}
    for t in range(T):
        cur = {**full[t], "punishment": [None] * 8, "punishment_valid": [False] * 8}
        pool = mgr._pool_from_rounds(full[:t] + [cur])
        for name in CONTRIB_SAFE:
            collected[name][t] = pool[name][0, TARGET, t]
    return collected


@pytest.fixture(scope="module")
def manager_compared():
    ref = build_reference()
    man = manager_features()
    return {
        name: (ref[f"ref_{name}"].to_numpy(float), man[name]) for name in CONTRIB_SAFE
    }


@pytest.mark.parametrize("feature", CONTRIB_SAFE)
def test_punishment_manager_matches_reference(manager_compared, feature):
    ref, man = manager_compared[feature]
    bad = np.where(~np.isclose(ref, man, atol=1e-4))[0]
    assert len(bad) == 0, f"manager {feature} mismatch at rounds {bad.tolist()}"


if __name__ == "__main__":  # manual eyeball: dump ref vs pipe side by side
    ref = build_reference()
    pipe = pipeline_features()
    for name in ALL_FEATURES:
        ref[f"pipe_{name}"] = pipe[name]
    out = HERE / "reference_features_p6.csv"
    ref.to_csv(out, index=False)
    fails = [
        n
        for n in ALL_FEATURES
        if not np.allclose(ref[f"ref_{n}"], ref[f"pipe_{n}"], atol=1e-6)
    ]
    print(
        f"wrote {out}  ({len(ALL_FEATURES)} features, "
        f"{'ALL MATCH' if not fails else 'FAILS: ' + str(fails)})"
    )
