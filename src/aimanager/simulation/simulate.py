"""Simulation Script.

Runs the simulation based on the provided config file.
Extracted from notebooks/test_manager/simulate_mixed_comparison.ipynb.

Usage:
    python src/simulation/simulate.py <config_path>
"""

import argparse
import math
import os
import random
import sys
from collections import Counter
from itertools import count
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch as th
import yaml

from aimanager.simulation.linear_ah import load_ah_model
from aimanager.manager.api_manager import MultiManager
from aimanager.manager.environment import ArtificialHumanEnv
from aimanager.manager.memory import Memory
from aimanager.utils.array_to_df import using_multiindex
from aimanager.utils.utils import make_dir

# pytorch geometric meta module has changed place
# since the saving of the training data, this points to the new location
import torch_geometric.nn.models.meta as meta_module

sys.modules["torch_geometric.nn.meta"] = meta_module


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_output_dir(config: dict, config_path: str) -> str:
    """Determine output directory from config or config path."""
    if "output_dir" in config:
        return config["output_dir"]
    return os.path.splitext(config_path)[0]


def mem_to_df(recorder, name: str) -> pd.DataFrame:
    """Convert Memory recorder to DataFrame."""
    columns = ["episode", "participant_code", "round_number"]

    punishments = using_multiindex(
        recorder.memory["punishment"].squeeze(1).numpy(),
        columns=columns,
        value_name="punishment",
    )
    common_good = using_multiindex(
        recorder.memory["common_good"].squeeze(1).numpy(),
        columns=columns,
        value_name="common_good",
    )
    contributions = using_multiindex(
        recorder.memory["contribution"].squeeze(1).numpy(),
        columns=columns,
        value_name="contribution",
    )
    agent_group = using_multiindex(
        recorder.memory["agent_group"].squeeze(1).numpy(),
        columns=columns,
        value_name="agent_group",
    )
    contribution_valid = using_multiindex(
        recorder.memory["contribution_valid"].squeeze(1).numpy(),
        columns=columns,
        value_name="contribution_valid",
    )

    df_sim = (
        punishments.merge(common_good)
        .merge(contributions)
        .merge(agent_group)
        .merge(contribution_valid)
    )

    # Calculate payoff: endowment (20) - contribution - punishment + common_good
    df_sim["payoff"] = (
        20 - df_sim["contribution"] - df_sim["punishment"] + df_sim["common_good"]
    )

    df_sim["participant_code"] = (
        df_sim["participant_code"].astype(str) + "_" + df_sim["episode"].astype(str)
    )
    df_sim["group_id"] = df_sim["agent_group"].astype(int)

    df_sim["run"] = name
    return df_sim


def make_round(
    contributions,
    round_num,
    groups,
    episode_group_idx,
    agent_group=None,
    contribution_valid=None,
):
    """Create a round dictionary; `contribution_valid` is the env's input flag."""
    if agent_group is None:
        agent_group = [0] * len(contributions)
    if contribution_valid is None:
        contribution_valid = [c is not None for c in contributions]
    return {
        "contribution": contributions,
        "contribution_valid": [bool(v) for v in contribution_valid],
        "punishment_valid": [False] * len(contributions),
        "punishment": [None] * len(contributions),
        "group": groups,
        "agent_group": agent_group,
        "round": round_num,
        "episode_group_idx": episode_group_idx,
    }


def add_punishments(round_dict, punishments):
    """Add punishments to a round dictionary."""
    return {
        **round_dict,
        "punishment": punishments,
        "punishment_valid": [p is not None for p in punishments],
    }


#: Largest episode_batch_size: ~9 GB of A100 memory for the env and AH
#: models (#232), leaving room for the managers.
MAX_EPISODE_BATCH_SIZE = 20000


def get_episode_batch_size(config: dict) -> int:
    """`episode_batch_size` from the config: 1 (default) plays episodes one at
    a time, above 1 in lockstep batches of that many."""
    size = config.get("episode_batch_size", 1)
    if isinstance(size, bool) or not isinstance(size, int):
        raise ValueError(f"episode_batch_size must be an integer, got {size!r}")
    if not 1 <= size <= MAX_EPISODE_BATCH_SIZE:
        raise ValueError(
            f"episode_batch_size must be in 1..{MAX_EPISODE_BATCH_SIZE}, got {size}"
        )
    return size


def load_humans(config: dict, humans: str, device):
    """The contribution, valid and (optional) switch models of an AH block."""
    basedir = config.get("basedir", ".")
    ah_config = config["artificial_humans"][humans]
    kwargs = dict(
        device=device,
        n_agents=config["n_agents"],
        n_contributions=config["n_contributions"],
    )
    # .joblib -> linear-baseline adapter, .pt -> GNN (a config may mix both,
    # e.g. GNN valid_model + linear contribution/switch models). See #121.
    ah = load_ah_model(os.path.join(basedir, ah_config["contribution_model"]), **kwargs)
    ah_val = load_ah_model(os.path.join(basedir, ah_config["valid_model"]), **kwargs)
    ah_switch = None
    if "switch_model" in ah_config:
        ah_switch = load_ah_model(
            os.path.join(basedir, ah_config["switch_model"]), **kwargs
        )
    return ah, ah_val, ah_switch


def make_env(config: dict, ah, ah_val, ah_switch, batch_size: int, device):
    """The simulation's environment around an AH block's models."""
    return ArtificialHumanEnv(
        artifical_humans=ah,
        artifical_humans_valid=ah_val,
        artifical_humans_switch=ah_switch,
        switch_every=config.get("switch_every", None),
        n_agents=config["n_agents"],
        n_contributions=config["n_contributions"],
        n_punishments=config["n_punishments"],
        n_rounds=config["n_rounds"],
        n_groups=config["n_groups"],
        batch_size=batch_size,
        device=device,
        agent_groups=config.get("agent_groups", None),
        reward_mode=config.get("reward_mode", "sum"),
        timeout_contribution=config.get("timeout_contribution", "default"),
    )


#: The env state entries a batched run records, one [B, A, T] tensor each.
BATCHED_KEYS = (
    "punishment",
    "common_good",
    "contribution",
    "agent_group",
    "contribution_valid",
)


def run_batched(config: dict, runs: dict, mm, device, episode_batch_size: int):
    """Play the runs' episodes in lockstep batches of `episode_batch_size`.

    The episodes of all runs that share an AH block are chunked into batches,
    pairings mixed within a batch. Every manager in a batch punishes the
    whole batch; each agent takes the punishment of the manager its episode's
    pairing assigns to its current agent_group. Batch i is seeded with
    `seed + i` when the config sets a seed.

    Returns one (episodes, record) per batch: `episodes` lists the batch's
    (run name, episode index) pairs in batch order, `record` maps each of
    BATCHED_KEYS to a [B, A, n_rounds] tensor on `device`.
    """
    n_episodes = config["n_episodes"]
    seed = config.get("seed")
    # one side per agent group: (group_0 manager, group_1 manager) per run
    sides = {
        name: (
            (run["pairing"]["group_0"], run["pairing"]["group_1"])
            if run["pairing"] is not None
            else (run["groups"][0], run["groups"][0])
        )
        for name, run in runs.items()
    }

    unbatched = sorted(
        {
            mm.manager_info[m]["type"]
            for pair in sides.values()
            for m in pair
            if not hasattr(mm.managers[m], "batched_punish")
        }
    )
    if unbatched:
        raise ValueError(
            f"managers of type {unbatched} have no batched form;"
            " set episode_batch_size: 1"
        )

    batches = []
    for humans in dict.fromkeys(run["humans"] for run in runs.values()):
        ah, ah_val, ah_switch = load_humans(config, humans, device)
        episodes = [
            (name, e)
            for name, run in runs.items()
            if run["humans"] == humans
            for e in range(n_episodes)
        ]
        for start in range(0, len(episodes), episode_batch_size):
            if seed is not None:
                batch_seed = seed + len(batches)
                random.seed(batch_seed)
                np.random.seed(batch_seed)
                th.manual_seed(batch_seed)
                if th.cuda.is_available():
                    th.cuda.manual_seed_all(batch_seed)

            batch = episodes[start : start + episode_batch_size]
            print(f"Start batch {len(batches)}: {len(batch)} episodes")
            used = list(dict.fromkeys(m for name, _ in batch for m in sides[name]))
            # [B, 2]: index into `used` of each episode's group_0 / group_1 manager
            side_idx = th.tensor(
                [[used.index(m) for m in sides[name]] for name, _ in batch],
                device=device,
            )

            env = make_env(config, ah, ah_val, ah_switch, len(batch), device)
            state = env.reset()
            record = {k: [] for k in BATCHED_KEYS}
            while True:
                # [B, A, 1]: index into `used` of each agent's manager
                agent_manager = side_idx.gather(1, state["agent_group"].squeeze(-1))
                agent_manager = agent_manager.unsqueeze(-1)
                punishment = th.zeros_like(state["punishment"])
                for i, m in enumerate(used):
                    punishment = th.where(
                        agent_manager == i,
                        mm.managers[m].batched_punish(state),
                        punishment,
                    )
                state = env.punish(punishment)
                for k in BATCHED_KEYS:
                    record[k].append(state[k].clone())

                state, _, done = env.step()
                if done:
                    break

            batches.append((batch, {k: th.cat(v, -1) for k, v in record.items()}))
    return batches


def batches_to_dfs(batches, runs: dict, config: dict) -> list:
    """Split run_batched's batches back into one DataFrame per run.

    Rebuilds the per-episode recorder's store ([n_episodes, 1, A,
    n_episode_steps] per key, episode e in row e) and converts it with
    mem_to_df, so the frames match a per-episode run's in columns, dtypes,
    episode numbering and order (runs in config order).
    """
    n_episodes = config["n_episodes"]
    n_steps = config["n_episode_steps"]
    store = {}
    for episodes, record in batches:
        record = {k: v.cpu() for k, v in record.items()}
        rows = {}
        for i, (name, e) in enumerate(episodes):
            rows.setdefault(name, ([], []))
            rows[name][0].append(i)
            rows[name][1].append(e)
        for name, (idx, eps) in rows.items():
            if name not in store:
                store[name] = {
                    k: th.zeros((n_episodes, 1, v.shape[1], n_steps), dtype=v.dtype)
                    for k, v in record.items()
                }
            for k, v in record.items():
                store[name][k][eps, 0, :, : v.shape[-1]] = v[idx]
    return [mem_to_df(SimpleNamespace(memory=store[name]), name=name) for name in runs]


def run_simulation(config: dict, output_dir: str) -> list:
    """Run the simulation and return list of DataFrames."""
    # Extract config parameters
    artificial_humans = config["artificial_humans"]
    managers_config = config["managers"]
    n_agents = config["n_agents"]
    n_episode_steps = config["n_episode_steps"]
    n_episodes = config["n_episodes"]
    basedir = config.get("basedir", ".")

    # Setup device
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    rec_device = th.device("cpu")

    print(f"Using device: {device}")

    # Add model_path to managers config, and resolve a rule-based manager's
    # rule and params files against basedir. Entries without a path
    # (e.g. type: dummy) pass through untouched.
    managers = {
        k: {
            **v,
            **({"model_path": os.path.join(basedir, v["path"])} if "path" in v else {}),
            **{
                key: os.path.join(basedir, v[key])
                for key in ("rule", "params")
                if key in v
            },
        }
        for k, v in managers_config.items()
    }
    print(f"Managers: {managers}")

    # Create MultiManager
    mm = MultiManager(managers, n_steps=n_episode_steps)

    # Fix autoregressive bug
    for k, man in mm.managers.items():
        if "autoregressive" in managers[k]:
            man.model.autoregressive = managers[k]["autoregressive"]

    # Create runs.
    # Two modes:
    #   - pairings (new): one run per (pairing, AH); per-agent manager
    #     assignment is rebuilt each round from state["agent_group"]
    #     (dynamic dispatch — matches training-time switch-predictor
    #     semantics).
    #   - legacy: one run per (manager, AH) with a static per-agent
    #     manager list (same manager on every agent).
    pairings = config.get("pairings")
    if pairings is not None:
        runs = {
            f"ah {h} managed by {p['name']}": {
                "pairing": p,
                "groups": None,
                "humans": h,
            }
            for p in pairings
            for h in artificial_humans.keys()
        }
    else:
        runs = {
            f"ah {h} managed by {m}": {
                "pairing": None,
                "groups": [m] * n_agents,
                "humans": h,
            }
            for m in managers.keys()
            for h in artificial_humans.keys()
        }

    episode_batch_size = get_episode_batch_size(config)
    if episode_batch_size > 1:
        n_batches = sum(
            math.ceil(n_runs * n_episodes / episode_batch_size)
            for n_runs in Counter(run["humans"] for run in runs.values()).values()
        )
        print(
            f"Batched: {len(runs)} runs x {n_episodes} episodes in {n_batches}"
            f" batches of up to {episode_batch_size}"
        )
        batches = run_batched(config, runs, mm, device, episode_batch_size)
        return batches_to_dfs(batches, runs, config)

    dfs = []
    for name, run in runs.items():
        print(f"Start run {name}")
        groups = run["groups"]
        pairing = run["pairing"]

        ah, ah_val, ah_switch = load_humans(config, run["humans"], device)
        env = make_env(config, ah, ah_val, ah_switch, 1, device)

        # Create recorder
        recorder = Memory(
            n_episodes=n_episodes,
            n_episode_steps=n_episode_steps,
            output_file=None,
            device=rec_device,
        )

        # Run episodes
        for e in range(n_episodes):
            state = env.reset()
            episode_group_idx = random.randint(0, 1000000)
            rounds = []

            for round_number in count():
                contributions = state["contribution"].squeeze().tolist()
                current_agent_group = (
                    state["agent_group"].squeeze().tolist()
                    if "agent_group" in state
                    else None
                )
                if pairing is not None:
                    # Dispatch each agent to its pairing-side manager based
                    # on its *current* agent_group, so post-switch agents
                    # are punished by the other side's manager.
                    group_map = [pairing["group_0"], pairing["group_1"]]
                    groups = [group_map[g] for g in current_agent_group]
                round_dict = make_round(
                    contributions,
                    round_number,
                    groups,
                    episode_group_idx,
                    agent_group=current_agent_group,
                    contribution_valid=state["contribution_valid"].reshape(-1).tolist(),
                )
                punishments = mm.get_punishments(rounds + [round_dict])[0]
                punishments_tensor = th.tensor(
                    punishments, dtype=th.int64, device=device
                )
                state = env.punish(punishments_tensor.unsqueeze(-1).unsqueeze(0))

                # record what was charged (punish zeroes timed-out players)
                round_dict = add_punishments(
                    round_dict, state["punishment"].reshape(-1).tolist()
                )
                rounds.append(round_dict)

                recorder.add(
                    **{
                        k: v if len(v.shape) == 3 else v.unsqueeze(-1)
                        for k, v in state.items()
                    },
                    episode_step=round_number,
                    episode=e,
                )

                state, reward, done = env.step()
                if done:
                    break

            recorder.next_episode(e)

        dfs.append(mem_to_df(recorder, name=name))

    return dfs


def load_pilot_data(config: dict, basedir: str) -> pd.DataFrame:
    """Load pilot experiment data — opt-in via `pilot_data_file`."""
    if "pilot_data_file" not in config:
        return None
    data_file = os.path.join(basedir, config["pilot_data_file"])

    if not os.path.exists(data_file):
        print(f"Warning: Pilot data file not found: {data_file}")
        return None

    df_pilot = pd.read_csv(data_file)

    experiment_name_map = config.get(
        "pilot_experiment_name_map",
        {
            "trail_rounds_2": "pilot human manager",
            "random_1": "pilot rule based manager",
        },
    )

    if "experiment_name" in df_pilot.columns:
        df_pilot["run"] = df_pilot["experiment_name"].map(experiment_name_map)
        df_pilot["run"] = df_pilot["run"].fillna(df_pilot["experiment_name"])
    else:
        df_pilot["run"] = "pilot"
    # common_good is stored as per-group pool (sum_c*1.6 - sum_p).
    # Divide by the number of valid contributors per group per round
    # so per-capita stays correct even when groups become imbalanced.
    valid = (df_pilot["player_no_input"] == 0).astype(int)
    n_valid = (
        valid.groupby(
            [df_pilot["episode_id"], df_pilot["round_number"], df_pilot["group_id"]]
        )
        .transform("sum")
        .clip(lower=1)
    )
    df_pilot["common_good"] = df_pilot["common_good"] / n_valid
    # Calculate payoff: endowment (20) - contribution - punishment + common_good
    df_pilot["payoff"] = (
        20 - df_pilot["contribution"] - df_pilot["punishment"] + df_pilot["common_good"]
    )
    df_pilot = df_pilot[
        [
            "round_number",
            "common_good",
            "contribution",
            "participant_code",
            "punishment",
            "payoff",
            "run",
            "global_group_id",
            "group_id",
        ]
    ]

    df_pilot["episode"] = df_pilot["global_group_id"]
    return df_pilot


def create_plots(
    df: pd.DataFrame,
    output_dir: str,
    managers_config: dict,
    figure_name: str = "",
    pairings: list = None,
    n_groups: int = 1,
) -> None:
    """Create and save comparison plots."""
    make_dir(output_dir)

    df["episode"] = df["run"] + "__" + df["episode"].astype(str)

    # `payoff_sum` is the per-group sum of per-agent payoffs. Place the
    # value on the first agent-row of each (episode, round, group) and
    # NaN on the rest, so downstream means over agent-rows (seaborn
    # line plots and aggregates) skip duplicates and reduce to one
    # value per episode -- the correct unweighted per-episode mean.
    _gkeys = ["run", "episode", "round_number", "group_id"]
    df["payoff_sum"] = (
        df.groupby(_gkeys)["payoff"].transform("sum").where(~df.duplicated(_gkeys))
    )

    if "group_id" in df.columns:
        df_focus = df[df["group_id"] == 0]
    else:
        df_focus = df

    dfm = df_focus.melt(
        id_vars=["episode", "round_number", "participant_code", "run"],
        value_vars=[
            "punishment",
            "contribution",
            "common_good",
            "payoff",
            "payoff_sum",
        ],
    )

    # Plot 1: Manager comparison
    manager_keys = list(managers_config.keys())

    # keep any run that looks like "ah ... managed by {manager}"
    w = dfm["run"].str.startswith("ah ") & dfm["run"].apply(
        lambda s: any(s.endswith(f"managed by {m}") for m in manager_keys)
    )

    if w.any():
        dfg = dfm[w].copy()

        # turn "ah {ah_name} managed by {m}" into "{ah_name} (managed by {m})"
        def to_label(s: str) -> str:
            # expected pattern: "ah {ah_name} managed by {manager}"
            if not s.startswith("ah ") or " managed by " not in s:
                return s
            body = s[len("ah ") :]  # "{ah_name} managed by {manager}"
            ah_name, manager = body.split(" managed by ", 1)
            return f"{ah_name} (managed by {manager})"

        dfg["run"] = dfg["run"].apply(to_label)

        g = sns.relplot(
            data=dfg,
            x="round_number",
            y="value",
            col="variable",
            hue="run",  # legend will now show "{ah_name} (managed by {manager})"
            kind="line",
            facet_kws={"sharey": False, "sharex": True},
            height=3,
            aspect=1,
        )
        g.fig.suptitle(f"Manager Comparison: {figure_name}", y=1.02)
        g.set(ylim=(0, None))
        g.savefig(os.path.join(output_dir, "comparison_manager.jpg"))
        plt.close()
        print(f"Saved: {os.path.join(output_dir, 'comparison_manager.jpg')}")

    # Plot 1b: Pairing-side comparison.
    # When the sim uses `pairings:`, each pairing's group_0 and group_1
    # are controlled by different managers. Plot 1 hues by run only,
    # averaging the two sides together; this plot splits them so the
    # trained side and opponent side appear as separate lines.
    if pairings and "group_id" in df.columns:
        pairings_by_name = {p["name"]: p for p in pairings}

        def _side(name, key):
            return pairings_by_name[name][key] if name in pairings_by_name else None

        df_p = df[df["run"].str.contains(" managed by ", na=False)].copy()
        pairing_name = df_p["run"].str.rsplit(" managed by ", n=1).str[1]
        df_p["pairing"] = pairing_name
        g0 = pairing_name.map(lambda n: _side(n, "group_0"))
        g1 = pairing_name.map(lambda n: _side(n, "group_1"))
        df_p["manager_side"] = g0.where(df_p["group_id"] == 0, g1)
        df_p = df_p[df_p["manager_side"].notna()]

        if len(df_p):
            df_p["label"] = df_p["pairing"] + " / " + df_p["manager_side"]
            # `payoff_sum` already per (run, episode, round, group_id)
            # from the top of create_plots, which is the correct
            # per-side sum here too (group_id distinguishes sides).
            dfp_m = df_p.melt(
                id_vars=[
                    "episode",
                    "round_number",
                    "participant_code",
                    "label",
                ],
                value_vars=[
                    "punishment",
                    "contribution",
                    "common_good",
                    "payoff",
                    "payoff_sum",
                ],
            )
            g = sns.relplot(
                data=dfp_m,
                x="round_number",
                y="value",
                col="variable",
                hue="label",
                kind="line",
                facet_kws={"sharey": False, "sharex": True},
                height=3,
                aspect=1.1,
                col_wrap=2,
            )
            g.fig.suptitle(
                f"Pairing-side Comparison: {figure_name}",
                y=1.02,
            )
            g.set(ylim=(0, None))
            out = os.path.join(output_dir, "comparison_pairing_side.jpg")
            g.savefig(out, bbox_inches="tight")
            plt.close()
            print(f"Saved: {out}")

    # Plot 2: Pilot comparison (if pilot data exists). Pilot rows are
    # anything that isn't a simulation run (sim runs start with "ah ").
    pilot_runs = [r for r in dfm["run"].unique() if not r.startswith("ah ")]
    w_pilot = dfm["run"].isin(pilot_runs)

    if w_pilot.any():
        dfg = dfm[w_pilot].copy()
        g = sns.relplot(
            data=dfg,
            x="round_number",
            y="value",
            col="variable",
            hue="run",
            kind="line",
            facet_kws={"sharey": False, "sharex": True},
            height=3,
            aspect=1,
        )
        g.set(ylim=(0, None))
        g.savefig(os.path.join(output_dir, "comparison_pilot.jpg"))
        plt.close()
        print(f"Saved: {os.path.join(output_dir, 'comparison_pilot.jpg')}")

    # Plot 3: Pilot + simulation overlay (direct comparison).
    # Only emit if pilot data is actually present — otherwise this plot
    # collapses to a copy of Plot 1.
    sim_runs = [r for r in dfm["run"].unique() if r.startswith("ah ")]
    overlay_runs = pilot_runs + sim_runs
    w_overlay = dfm["run"].isin(overlay_runs)

    if pilot_runs and w_overlay.any():
        dfg = dfm[w_overlay].copy()
        g = sns.relplot(
            data=dfg,
            x="round_number",
            y="value",
            col="variable",
            hue="run",
            kind="line",
            facet_kws={"sharey": False, "sharex": True},
            height=3,
            aspect=1,
        )
        g.fig.suptitle(f"Pilot vs Simulation: {figure_name}", y=1.02)
        g.set(ylim=(0, None))
        g.savefig(os.path.join(output_dir, "comparison_pilot_vs_sim.jpg"))
        plt.close()
        print(f"Saved: {os.path.join(output_dir, 'comparison_pilot_vs_sim.jpg')}")

    # Plot 4: Group-size evolution per run (sim + pilot comparison)
    if n_groups > 1 and "group_id" in df.columns:
        # an empty group has no rows: fill it in as size 0
        per_episode_sizes = (
            df.groupby(["run", "episode", "round_number", "group_id"])[
                "participant_code"
            ]
            .nunique()
            .unstack("group_id", fill_value=0)
            .stack()
            .rename("group_size")
            .reset_index()
        )
        max_agents = (
            df.groupby(["run", "episode", "round_number"])["participant_code"]
            .nunique()
            .max()
        )

        g = sns.relplot(
            data=per_episode_sizes,
            x="round_number",
            y="group_size",
            col="group_id",
            hue="run",
            kind="line",
            facet_kws={"sharey": True, "sharex": True},
            height=4,
            aspect=1.3,
        )
        g.fig.suptitle(f"Group size per round: {figure_name}", y=1.02)
        g.set(ylim=(0, max_agents))
        g.set_axis_labels("round_number", "group_size")
        global_group_size_path = os.path.join(
            output_dir, "group_size_evolution_global.jpg"
        )
        g.savefig(global_group_size_path, bbox_inches="tight")
        plt.close()
        print(f"Saved: {global_group_size_path}")

    # Plot 5: Number of switches per round (mean over episodes)
    if n_groups > 1 and "group_id" in df.columns:
        switch_df = df.sort_values(
            ["run", "episode", "participant_code", "round_number"]
        ).copy()
        player_key = (
            switch_df["participant_code"].astype(str)
            + "__"
            + switch_df["episode"].astype(str)
        )
        switch_df["player_episode"] = player_key
        switch_df["prev_group_id"] = switch_df.groupby(["run", "player_episode"])[
            "group_id"
        ].shift(1)
        switch_df["switched"] = (
            switch_df["group_id"] != switch_df["prev_group_id"]
        ) & switch_df["prev_group_id"].notna()
        per_episode_switches = (
            switch_df.groupby(["run", "episode", "round_number"], as_index=False)[
                "switched"
            ]
            .sum()
            .rename(columns={"switched": "switch_count"})
        )
        switch_counts = per_episode_switches.groupby(
            ["run", "round_number"], as_index=False
        )["switch_count"].mean()
        plt.figure(figsize=(9, 4))
        sns.lineplot(
            data=switch_counts,
            x="round_number",
            y="switch_count",
            hue="run",
        )
        plt.title("Number of switches per round")
        plt.tight_layout()
        switch_path = os.path.join(output_dir, "switch_count_per_round.jpg")
        plt.savefig(switch_path)
        plt.close()
        print(f"Saved: {switch_path}")

    # Plot 6: Group switching heatmap (if agent_group data exists)
    if n_groups > 1 and "agent_group" in df.columns:
        # Only use simulation runs (not pilot data)
        sim_runs = [r for r in df["run"].unique() if r.startswith("ah ")]
        df_sim = df[df["run"].isin(sim_runs)].copy()

        if len(df_sim) > 0:
            # Extract agent index from participant_code ("3_12" -> 3)
            df_sim["agent"] = (
                df_sim["participant_code"].str.split("_").str[0].astype(int)
            )

        for run_name in sim_runs:
            run_df = df_sim[df_sim["run"] == run_name]
            episodes = run_df["episode"].unique()
            # Sort numerically by episode suffix
            episodes = sorted(
                episodes,
                key=lambda e: int(str(e).rsplit("__", 1)[-1]) if "__" in str(e) else e,
            )
            n_show = min(4, len(episodes))
            show_episodes = episodes[:n_show]

            fig, axes = plt.subplots(1, n_show, figsize=(4 * n_show, 4), sharey=True)
            if n_show == 1:
                axes = [axes]

            cmap = sns.color_palette(["#4393c3", "#d6604d"], as_cmap=True)
            for ax, ep in zip(axes, show_episodes):
                ep_df = run_df[run_df["episode"] == ep]
                heatmap_data = ep_df.pivot(
                    index="agent",
                    columns="round_number",
                    values="agent_group",
                ).astype(int)
                sns.heatmap(
                    heatmap_data,
                    cmap=cmap,
                    vmin=0,
                    vmax=1,
                    cbar=False,
                    ax=ax,
                    linewidths=0.5,
                    linecolor="white",
                )
                ep_label = str(ep).rsplit("__", 1)[-1] if "__" in str(ep) else str(ep)
                ax.set_title(f"Episode {ep_label}")
                ax.set_xlabel("Round")
                if ax == axes[0]:
                    ax.set_ylabel("Agent")
                else:
                    ax.set_ylabel("")

            label = run_name.replace("ah ", "").replace(" managed by ", " / ")
            fig.suptitle(f"Group Membership — {label}", y=1.02)
            fig.tight_layout()
            fname = run_name.replace(" ", "_").replace("/", "_") + "_groups.png"
            fig.savefig(
                os.path.join(output_dir, fname),
                dpi=150,
                bbox_inches="tight",
            )
            plt.close(fig)
            print(f"Saved: {os.path.join(output_dir, fname)}")

    # Save aggregates — restricted to the focus manager's group (id 0)
    # so reported numbers match the displayed lines in Plot 1.
    aggregates = (
        df_focus.groupby(["run", "round_number"])
        .agg(
            {
                "punishment": "mean",
                "contribution": "mean",
                "common_good": "mean",
                "payoff": "mean",
                "payoff_sum": "mean",
            }
        )
        .reset_index()
    )
    aggregates_path = os.path.join(output_dir, "aggregates.csv")
    aggregates.to_csv(aggregates_path, index=False)
    print(f"Saved: {aggregates_path}")


def run_cli(config, config_path):
    """Entry point for the unified CLI.

    Args:
        config: Parsed YAML config dict.
        config_path: Path to the config file (used to derive output_dir).
    """
    output_dir = get_output_dir(config, config_path)
    basedir = config.get("basedir", ".")
    get_episode_batch_size(config)  # fail on a bad value before any work

    print(f"Config: {config_path}")
    print(f"Output directory: {output_dir}")

    # Optional seeding for reproducible sims (eps-greedy sampling,
    # switch predictor sampling, episode_group_idx draw).
    seed = config.get("seed")
    if seed is not None:
        print(f"Seeding with seed {seed}")
        random.seed(seed)
        np.random.seed(seed)
        th.manual_seed(seed)
        if th.cuda.is_available():
            th.cuda.manual_seed_all(seed)

    # Create output directory
    make_dir(output_dir)

    # Run simulation
    dfs = run_simulation(config, output_dir)

    df_sim = pd.concat(dfs).reset_index(drop=True)

    # Persist per-agent per-round simulation frame — opt-in via
    # `save_per_round` for downstream trajectory plotting.
    if config.get("save_per_round", False):
        per_round_path = os.path.join(output_dir, "per_round.parquet")
        df_sim.to_parquet(per_round_path, index=False)
        print(f"Saved: {per_round_path}")

    # Load pilot data if available
    df_pilot = load_pilot_data(config, basedir)

    # Combine dataframes
    if df_pilot is not None:
        df = pd.concat([df_sim, df_pilot]).reset_index(drop=True)
    else:
        df = df_sim

    # Create plots
    create_plots(
        df,
        output_dir,
        config["managers"],
        config["figure_name"],
        pairings=config.get("pairings"),
        n_groups=config.get("n_groups", 1),
    )

    print("Simulation complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Run simulation from config file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "config_path",
        type=str,
        help="Path to simulation config YAML file",
    )

    args = parser.parse_args()

    # Validate config file exists
    if not os.path.exists(args.config_path):
        print(
            f"Error: Config file not found: {args.config_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Load config
    config = load_config(args.config_path)
    run_cli(config, args.config_path)


if __name__ == "__main__":
    main()
