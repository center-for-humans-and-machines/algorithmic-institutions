"""Integration test for the hand-crafted baseline feature pipeline (issue #119).

Verifies that build_feature_pool (src) produces exactly the features it claims,
by comparing every feature (blocks B1-B7 + the lag_* switch twins) for one agent
across a whole episode against an INDEPENDENT pandas reference computed here from
frozen raw fixtures -- no src code is used to build the reference, so this is a
genuine cross-check. Group means/windows aggregate t-1 VALUES over CURRENT-round
membership (reference: episode_raw.csv rosters at t); the lag_* twins keep t-1
membership (reference: the frozen t-1 roster fixtures).

Two src sides are checked against that one reference:
  * build_feature_pool  -- the batch pipeline (CV / training).
  * LinearAHAdapter      -- the simulation adapter, which rebuilds features ONE
    ROUND AT A TIME from env state (#121); the episode is replayed through it and
    the per-round features collected, confirming the incremental reconstruction
    matches the batch pipeline / reference.

Target: episode 70 (global_group_id 'rokqh2fp #2'), player 6, from the non-flipped
originals of experiments/2group_8agent_50ep.csv. The fixtures under fixtures/ were
extracted with pure pandas (see git history for the one-off extract scripts):
  * episode_raw.csv        -- all 8 agents' raw per-round fields (wide)
  * episode_peers_p6.csv   -- player 6's t-1 own-group roster per round
  * episode_other_p6.csv   -- player 6's t-1 other-group roster per round

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
ROOT = HERE.parents[1]              # tests/baselines -> repo root
DATA = HERE / "fixtures"
RAW = ROOT / "experiments/2group_8agent_50ep.csv"
TARGET, EPISODE_ID, EXPERIMENT, SWITCH_EVERY = 6, 70, "ah_group_switching", 4

B1 = ["prev_contribution", "prev_punishment", "prev_payoff"]
# Group/Other are OBSERVATIONAL (notes/baseline_feature_defs.md): t-1 values over the
# t-1 roster of the agent's CURRENT group ids; self excluded on the group side.
B2 = ["prev_contribution_mean_group", "prev_punishment_mean_group",
      "prev_payoff_mean_group", "prev_group_size", "prev_common_good"]
B3 = ["prev_contribution_mean_other", "prev_punishment_mean_other",
      "prev_common_good_other", "prev_payoff_mean_other",
      "prev_group_size_other"]
B4 = ["prev_contribution_mean_gap", "prev_punishment_mean_gap",
      "prev_common_good_gap", "prev_payoff_mean_gap", "prev_group_size_delta"]
_WIN5 = ["win_contribution_mean_peers", "win_punishment_mean_peers",
         "win_common_good_peers", "win_payoff_mean_peers", "win_group_size"]
_WIN6 = ["win_contribution_mean_other", "win_punishment_mean_other",
         "win_common_good_other", "win_payoff_mean_other", "win_group_size_other"]
B5 = _WIN5 + [f"prev_{w}" for w in _WIN5]
B6 = _WIN6 + [f"prev_{w}" for w in _WIN6]
B7 = ["round_number", "group_size", "rounds_since_switch",
      "switched_last_choice", "prev_switched_last_choice"]
# lag_* twins (switch target): t-1 values over t-1 membership -- the group as it
# stood before the round-t decision. Reference: the frozen t-1 roster fixtures.
LAG = ["lag_contribution_mean_peers", "lag_punishment_mean_peers",
       "lag_payoff_mean_group",
       "lag_contribution_mean_other", "lag_punishment_mean_other",
       "lag_common_good_mean_other", "lag_payoff_mean_other",
       "lag_contribution_mean_gap", "lag_punishment_mean_gap",
       "lag_common_good_mean_gap", "lag_payoff_mean_gap"]
ALL_FEATURES = B1 + B2 + B3 + B4 + B5 + B6 + B7 + LAG


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
    n_valid = df.assign(cv=valid.astype(int)).groupby(
        ["episode_id", "round_number", "group_id"])["cv"].transform("sum")
    cg_def = float((df["common_good"] / n_valid)[valid].median())
    return c_def, p_def, cg_def


def _add_b1(ref, bank, c_def, p_def, cg_def):
    b = bank.set_index("round_number")
    T = len(b)
    c = bank[f"p{TARGET}_contribution"].to_numpy(float)
    p = bank[f"p{TARGET}_punishment"].to_numpy(float)
    # own EXPERIENCED per-capita cg at each round (own group's pool / n_valid)
    percap = np.zeros(T)
    for t in range(T):
        g = b.loc[t, f"p{TARGET}_grp"]
        members = [j for j in range(8) if b.loc[t, f"p{j}_grp"] == g]
        nv = sum(int(b.loc[t, f"p{j}_noinput"] == 0) for j in members)
        percap[t] = float(b.loc[t, f"p{TARGET}_common_good"]) / nv if nv else 0.0
    ref["ref_prev_contribution"] = np.concatenate([[c_def], c[:-1]])
    ref["ref_prev_punishment"] = np.concatenate([[p_def], p[:-1]])
    ref["ref_prev_payoff"] = 20.0 - ref["ref_prev_contribution"] \
        - ref["ref_prev_punishment"] + np.concatenate([[cg_def], percap[:-1]])


def _add_b2(ref, peers, c_def, p_def, cg_def):
    """lag_* own-group twins from the frozen t-1 roster fixture."""
    per = {}
    for t, grp in peers.groupby("round_number"):
        pe = grp[grp["is_self"] == 0]
        nv = int((grp["noinput_tm1"] == 0).sum())
        percap = float(grp["common_good_pool_tm1"].iloc[0]) / nv if nv else 0.0
        mc, mp = grp["contribution_tm1"].mean(), grp["punishment_tm1"].mean()
        per[int(t)] = {
            "lag_contribution_mean_peers":
                float(pe["contribution_tm1"].mean()) if len(pe) else 0.0,
            "lag_punishment_mean_peers":
                float(pe["punishment_tm1"].mean()) if len(pe) else 0.0,
            "lag_payoff_mean_group": 20.0 - mc - mp + percap,
        }
    per[0] = {"lag_contribution_mean_peers": c_def,
              "lag_punishment_mean_peers": p_def,
              "lag_payoff_mean_group": 20.0 - c_def - p_def + cg_def}
    for name in per[0]:
        ref[f"ref_{name}"] = ref["round_number"].map(lambda t: per[int(t)][name])


def _add_b3(ref, other, c_def, p_def, cg_def):
    """lag_* other-group twins from the frozen t-1 roster fixture."""
    per = {}
    for t in range(1, 24):
        grp = other[other["round_number"] == t]
        if len(grp) == 0:
            per[t] = {"lag_contribution_mean_other": 0.0,
                      "lag_punishment_mean_other": 0.0,
                      "lag_common_good_mean_other": 0.0,
                      "lag_payoff_mean_other": 20.0}
            continue
        nv = int((grp["noinput_tm1"] == 0).sum())
        percap = float(grp["common_good_pool_tm1"].iloc[0]) / nv if nv else 0.0
        mc, mp = grp["contribution_tm1"].mean(), grp["punishment_tm1"].mean()
        per[t] = {"lag_contribution_mean_other": float(mc),
                  "lag_punishment_mean_other": float(mp),
                  "lag_common_good_mean_other": percap,
                  "lag_payoff_mean_other": 20.0 - mc - mp + percap}
    per[0] = {"lag_contribution_mean_other": c_def,
              "lag_punishment_mean_other": p_def,
              "lag_common_good_mean_other": cg_def,
              "lag_payoff_mean_other": 20.0 - c_def - p_def + cg_def}
    for name in per[0]:
        ref[f"ref_{name}"] = ref["round_number"].map(lambda t: per[int(t)][name])


def _add_b4(ref, peers, other):
    """lag_* gap twins from the frozen t-1 roster fixtures."""
    per = {}
    for t in range(1, 24):
        pg = peers[peers["round_number"] == t]
        og = other[other["round_number"] == t]
        pe = pg[pg["is_self"] == 0]
        own_nv = int((pg["noinput_tm1"] == 0).sum())
        own_cg = float(pg["common_good_pool_tm1"].iloc[0]) / own_nv if own_nv else 0.0
        peers_c = float(pe["contribution_tm1"].mean()) if len(pe) else 0.0
        peers_p = float(pe["punishment_tm1"].mean()) if len(pe) else 0.0
        peers_payoff = 20.0 - peers_c - peers_p + own_cg
        if len(og):
            oth_nv = int((og["noinput_tm1"] == 0).sum())
            oth_cg = float(og["common_good_pool_tm1"].iloc[0]) / oth_nv if oth_nv else 0.0
            oth_c, oth_p = (float(og["contribution_tm1"].mean()),
                            float(og["punishment_tm1"].mean()))
        else:
            oth_cg = oth_c = oth_p = 0.0
        oth_payoff = 20.0 - oth_c - oth_p + oth_cg
        per[t] = {"lag_contribution_mean_gap": peers_c - oth_c,
                  "lag_punishment_mean_gap": peers_p - oth_p,
                  "lag_common_good_mean_gap": own_cg - oth_cg,
                  "lag_payoff_mean_gap": peers_payoff - oth_payoff}
    per[0] = {n: 0.0 for n in per[1]}
    for name in per[1]:
        ref[f"ref_{name}"] = ref["round_number"].map(lambda t: per[int(t)][name])


def _switch_series(bank):
    g = bank[f"p{TARGET}_grp"].to_numpy()
    ds = np.zeros(len(g), bool)
    ds[1:] = g[1:] != g[:-1]
    return ds


def _window(series, ds):
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
    """Value windows: 0 at the arrival round (strictly within-tenure)."""
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


def _shift1(a):
    out = np.roll(a, 1)
    out[0] = a[0]
    return out


def _add_windows(ref, ds, side):
    """Shared window builder for B5/B6 over the OBSERVATIONAL series. Value
    windows are strictly within-tenure (0 at the arrival round; payoff is
    windowed as its own series so it is 0 there too); size windows keep their
    arrival value. prev_win = one-round shift."""
    src = "group" if side == "peers" else "other"
    cgcol = ("ref_prev_common_good" if side == "peers"
             else "ref_prev_common_good_other")
    sizecol = ("ref_prev_group_size" if side == "peers"
               else "ref_prev_group_size_other")
    wc = _window0(ref[f"ref_prev_contribution_mean_{src}"].to_numpy(float), ds)
    wp = _window0(ref[f"ref_prev_punishment_mean_{src}"].to_numpy(float), ds)
    wcg = _window0(ref[cgcol].to_numpy(float), ds)
    wpay = _window0(ref[f"ref_prev_payoff_mean_{src}"].to_numpy(float), ds)
    wsize = _window(ref[sizecol].to_numpy(float), ds)
    size_name = "win_group_size" if side == "peers" else "win_group_size_other"
    for name, arr in {f"win_contribution_mean_{side}": wc,
                      f"win_punishment_mean_{side}": wp,
                      f"win_common_good_{side}": wcg,
                      f"win_payoff_mean_{side}": wpay,
                      size_name: wsize}.items():
        ref[f"ref_{name}"] = arr
        ref[f"ref_prev_{name}"] = _shift1(arr)


def _add_b5(ref, ds):   # own-group since-switch window (peers)
    _add_windows(ref, ds, "peers")


def _add_b6(ref, ds):   # other-group since-switch window
    _add_windows(ref, ds, "other")


def _add_obs(ref, bank, c_def, p_def, cg_def):
    """Observational Group/Other reference (notes/baseline_feature_defs.md): t-1
    values over the t-1 ROSTER of the target's CURRENT group ids. Group side
    excludes self (when self was in that roster); other side is the full
    roster mean; cg is the roster's single shared per-capita value. Round 0:
    defaults, other = group, gaps 0, sizes = current sizes."""
    b = bank.set_index("round_number")
    T = len(b)
    gsize0 = int(sum(b.loc[0, f"p{j}_grp"] == b.loc[0, f"p{TARGET}_grp"]
                     for j in range(8)))
    per = {0: {"c_grp": c_def, "p_grp": p_def, "cg_grp": cg_def,
               "c_oth": c_def, "p_oth": p_def, "cg_oth": cg_def,
               "size_grp": float(gsize0), "size_oth": float(8 - gsize0)}}
    for t in range(1, T):
        g_t = b.loc[t, f"p{TARGET}_grp"]

        def side(gid, loo_self):
            roster = [j for j in range(8) if b.loc[t - 1, f"p{j}_grp"] == gid]
            mem = [j for j in roster if not (loo_self and j == TARGET)]

            def mean(field):
                return (float(np.mean([b.loc[t - 1, f"p{j}_{field}"]
                                       for j in mem])) if mem else 0.0)

            if roster:
                nv = sum(int(b.loc[t - 1, f"p{j}_noinput"] == 0)
                         for j in roster)
                pool = float(b.loc[t - 1, f"p{roster[0]}_common_good"])
                cg = pool / nv if nv else 0.0
            else:
                cg = 0.0
            return mean("contribution"), mean("punishment"), cg, float(len(roster))

        c_g, p_g, cg_g, n_g = side(g_t, loo_self=True)
        c_o, p_o, cg_o, n_o = side(1 - g_t, loo_self=False)
        per[t] = {"c_grp": c_g, "p_grp": p_g, "cg_grp": cg_g, "size_grp": n_g,
                  "c_oth": c_o, "p_oth": p_o, "cg_oth": cg_o, "size_oth": n_o}

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
        20.0 - ref["ref_prev_contribution_mean_group"]
        - ref["ref_prev_punishment_mean_group"] + ref["ref_prev_common_good"])
    ref["ref_prev_payoff_mean_other"] = (
        20.0 - ref["ref_prev_contribution_mean_other"]
        - ref["ref_prev_punishment_mean_other"]
        + ref["ref_prev_common_good_other"])
    ref["ref_prev_contribution_mean_gap"] = (
        ref["ref_prev_contribution_mean_group"]
        - ref["ref_prev_contribution_mean_other"])
    ref["ref_prev_punishment_mean_gap"] = (
        ref["ref_prev_punishment_mean_group"]
        - ref["ref_prev_punishment_mean_other"])
    ref["ref_prev_common_good_gap"] = (
        ref["ref_prev_common_good"] - ref["ref_prev_common_good_other"])
    ref["ref_prev_payoff_mean_gap"] = (
        ref["ref_prev_payoff_mean_group"] - ref["ref_prev_payoff_mean_other"])
    ref["ref_prev_group_size_delta"] = (
        ref["ref_prev_group_size"] - ref["ref_prev_group_size_other"])


def _add_b7(ref, bank, ds):
    b = bank.set_index("round_number")
    T = len(b)
    ref["ref_round_number"] = np.arange(T, dtype=float)
    ref["ref_group_size"] = [
        float(sum(b.loc[t, f"p{j}_grp"] == b.loc[t, f"p{TARGET}_grp"] for j in range(8)))
        for t in range(T)]
    rss = np.zeros(T)
    cnt = 0
    for t in range(T):
        cnt = 0 if (t == 0 or ds[t]) else cnt + 1
        rss[t] = cnt
    ref["ref_rounds_since_switch"] = rss

    def slc(strict):
        out = np.zeros(T)
        for t in range(T):
            last_dec = ((t - 1 if strict else t) // SWITCH_EVERY) * SWITCH_EVERY
            if last_dec > 0:
                out[t] = float(ds[last_dec])
        return out
    ref["ref_switched_last_choice"] = slc(False)
    ref["ref_prev_switched_last_choice"] = slc(True)


def build_reference():
    bank = pd.read_csv(DATA / "episode_raw.csv")
    peers = pd.read_csv(DATA / "episode_peers_p6.csv")
    other = pd.read_csv(DATA / "episode_other_p6.csv")
    c_def, p_def, cg_def = _dataset_defaults()
    ref = pd.DataFrame({"round_number": bank["round_number"]})
    _add_b1(ref, bank, c_def, p_def, cg_def)
    _add_b2(ref, peers, c_def, p_def, cg_def)         # -> lag_* (t-1 rosters)
    _add_b3(ref, other, c_def, p_def, cg_def)         # -> lag_*
    _add_b4(ref, peers, other)                        # -> lag_* gaps
    _add_obs(ref, bank, c_def, p_def, cg_def)         # Group/Other/gaps/sizes
    ds = _switch_series(bank)
    _add_b5(ref, ds)
    _add_b6(ref, ds)
    _add_b7(ref, bank, ds)
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

    random.seed(38381); np.random.seed(38381); th.manual_seed(38381)
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
    return {name: (ref[f"ref_{name}"].to_numpy(float), pipe[name]) for name in ALL_FEATURES}


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
# --------------------------------------------------------------------------- #
def adapter_features():
    import random

    import torch as th

    sys.path.insert(0, str(ROOT / "scripts/baselines"))
    from aimanager.generic.data import create_torch_data
    from aimanager.simulation.linear_ah import LinearAHAdapter

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

    # only the fields LinearAHAdapter reads to rebuild features are needed
    bundle = {
        "model": "ridge", "estimator": None, "scaler": None, "features": [],
        "target": "contribution", "n_levels": 0, "switch_every": SWITCH_EVERY,
        "default_values": {k: (float(v) if hasattr(v, "__float__") else v)
                           for k, v in default_values.items()},
    }
    ad = LinearAHAdapter(bundle, n_agents=A, n_contributions=21,
                         device=th.device("cpu"))

    collected = {name: np.zeros(T) for name in ALL_FEATURES}
    for t in range(T):
        state = {  # env-like state: prev_* = realised t-1, agent_group = current
            "round_number": th.full((1, A, 1), t, dtype=th.int64),
            "prev_contribution": data["prev_contribution"][g, :, t].reshape(1, A, 1),
            "prev_punishment": data["prev_punishment"][g, :, t].reshape(1, A, 1),
            "prev_common_good": data["prev_common_good"][g, :, t].reshape(1, A, 1),
            "prev_agent_group": data["prev_agent_group"][g, :, t].reshape(1, A, 1),
            "agent_group": data["agent_group"][g, :, t].reshape(1, A, 1),
        }
        if t == 0:
            ad._reset_history()
        ad._record(state, t)
        pool = ad._build_pool(t)
        for name in ALL_FEATURES:
            collected[name][t] = pool[name][0, TARGET, t]
    return collected


@pytest.fixture(scope="module")
def adapter_compared():
    ref = build_reference()
    ada = adapter_features()
    return {name: (ref[f"ref_{name}"].to_numpy(float), ada[name])
            for name in ALL_FEATURES}


@pytest.mark.parametrize("feature", ALL_FEATURES)
def test_adapter_matches_reference(adapter_compared, feature):
    # atol=1e-4: the adapter rebuilds via float32 tensors + float64 round-0
    # defaults, so common_good-derived features differ by ~1e-7 (float32 eps).
    ref, ada = adapter_compared[feature]
    bad = np.where(~np.isclose(ref, ada, atol=1e-4))[0]
    assert len(bad) == 0, f"adapter {feature} mismatch at rounds {bad.tolist()}"


if __name__ == "__main__":  # manual eyeball: dump ref vs pipe side by side
    ref = build_reference()
    pipe = pipeline_features()
    for name in ALL_FEATURES:
        ref[f"pipe_{name}"] = pipe[name]
    out = HERE / "reference_features_p6.csv"
    ref.to_csv(out, index=False)
    fails = [n for n in ALL_FEATURES
             if not np.allclose(ref[f"ref_{n}"], ref[f"pipe_{n}"], atol=1e-6)]
    print(f"wrote {out}  ({len(ALL_FEATURES)} features, "
          f"{'ALL MATCH' if not fails else 'FAILS: ' + str(fails)})")
