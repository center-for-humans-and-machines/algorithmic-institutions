"""Simulation Script.

Runs the simulation based on the provided config file.
Extracted from notebooks/test_manager/simulate_mixed_comparison.ipynb.

Usage:
    python src/simulation/simulate.py <config_path>
"""

import argparse
import os
import random
import sys
from itertools import count

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch as th
import yaml

# pytorch geometric meta module has changed place
# since the saving of the training data, this points to the new location
import torch_geometric.nn.models.meta as meta_module
sys.modules['torch_geometric.nn.meta'] = meta_module

from aimanager.artificial_humans import GraphNetwork
from aimanager.manager.api_manager import MultiManager
from aimanager.manager.environment import ArtificialHumanEnv
from aimanager.manager.memory import Memory
from aimanager.utils.array_to_df import using_multiindex
from aimanager.utils.utils import make_dir


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
    group = using_multiindex(
        recorder.memory["group"].squeeze(1).numpy(),
        columns=columns,
        value_name="group",
    )

    df_sim = punishments.merge(common_good).merge(contributions).merge(group)

    df_sim["participant_code"] = (
        df_sim["participant_code"].astype(str) + "_" + df_sim["episode"].astype(str)
    )

    df_sim["run"] = name
    return df_sim


def make_round(contributions, round_num, groups, episode_group_idx):
    """Create a round dictionary."""
    return {
        "contribution": contributions,
        "contribution_valid": [c is not None for c in contributions],
        "punishment_valid": [False] * len(contributions),
        "punishment": [None] * len(contributions),
        "group": groups,
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


def run_simulation(config: dict, output_dir: str) -> list:
    """Run the simulation and return list of DataFrames."""
    # Extract config parameters
    artificial_humans = config["artificial_humans"]
    managers_config = config["managers"]
    n_episode_steps = config["n_episode_steps"]
    n_episodes = config["n_episodes"]
    basedir = config.get("basedir", ".")

    # Setup device
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    rec_device = th.device("cpu")

    print(f"Using device: {device}")

    # Add model_path to managers config
    managers = {
        k: {**v, 'model_path': os.path.join(basedir, v['path'])}
        for k, v in managers_config.items()
    }
    print(f"Managers: {managers}")

    # Create MultiManager
    mm = MultiManager(managers, n_steps=n_episode_steps)

    # Fix autoregressive bug
    for k, man in mm.managers.items():
        if "autoregressive" in managers[k]:
            man.model.autoregressive = managers[k]["autoregressive"]

    # Create runs: all combinations of managers and artificial humans
    runs = {
        f"ah {h} managed by {m}": {"groups": [m] * 4, "humans": h}
        for m in managers.keys()
        for h in artificial_humans.keys()
    }

    dfs = []
    for name, run in runs.items():
        print(f"Start run {name}")
        groups = run["groups"]
        n_agents = len(groups)

        # Load artificial humans
        hm_path = os.path.join(
            basedir, artificial_humans[run["humans"]]["contribution_model"]
        )
        hmv_path = os.path.join(
            basedir, artificial_humans[run["humans"]]["valid_model"]
        )

        ah = GraphNetwork.load(hm_path, device=device)
        ah_val = GraphNetwork.load(hmv_path, device=device)

        # Create environment
        env = ArtificialHumanEnv(
            artifical_humans=ah,
            artifical_humans_valid=ah_val,
            n_agents=n_agents,
            n_contributions=21,
            n_punishments=31,
            n_rounds=n_episode_steps,
            batch_size=1,
            device=device,
        )

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
                round_dict = make_round(
                    contributions, round_number, groups, episode_group_idx
                )
                punishments = mm.get_punishments(rounds + [round_dict])[0]
                round_dict = add_punishments(round_dict, punishments)
                rounds.append(round_dict)

                punishments_tensor = th.tensor(
                    punishments, dtype=th.int64, device=device
                )
                state = env.punish(punishments_tensor.unsqueeze(-1).unsqueeze(0))

                recorder.add(
                    **{
                        k: v if len(v.shape) == 3 else v.unsqueeze(-1)
                        for k, v in state.items()
                    },
                    episode_step=round_number,
                )

                state, reward, done = env.step()
                if done:
                    break

            recorder.next_episode(e)

        dfs.append(mem_to_df(recorder, name=name))

    return dfs


def load_pilot_data(basedir: str) -> pd.DataFrame:
    """Load pilot experiment data."""
    data_file = "experiments/pilot_random1_player_round_slim.csv"
    data_file = os.path.join(basedir, data_file)

    if not os.path.exists(data_file):
        print(f"Warning: Pilot data file not found: {data_file}")
        return None

    df_pilot = pd.read_csv(data_file)

    experiment_name_map = {
        "trail_rounds_2": "pilot human manager",
        "random_1": "pilot rule based manager",
    }

    df_pilot["run"] = df_pilot["experiment_name"].map(experiment_name_map)
    df_pilot["common_good"] = df_pilot["common_good"] / 4
    df_pilot = df_pilot[
        [
            "round_number",
            "common_good",
            "contribution",
            "participant_code",
            "punishment",
            "run",
            "global_group_id",
        ]
    ]

    df_pilot["episode"] = df_pilot["global_group_id"]
    return df_pilot


def create_plots(df: pd.DataFrame, output_dir: str, managers_config: dict) -> None:
    """Create and save comparison plots."""
    make_dir(output_dir)

    df["episode"] = df["run"] + "__" + df["episode"].astype(str)

    dfm = df.melt(
        id_vars=["episode", "round_number", "participant_code", "run"],
        value_vars=["punishment", "contribution", "common_good"],
    )

    # Plot 1: Manager comparison
    manager_runs = [
        f"ah full managed by {m}" for m in managers_config.keys()
    ]
    w = dfm["run"].isin(manager_runs)

    if w.any():
        dfg = dfm[w].copy()
        run_map = {
            f"ah full managed by {m}": f"{m} manager"
            for m in managers_config.keys()
        }
        dfg["run"] = dfg["run"].map(run_map)

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
        g.savefig(os.path.join(output_dir, "comparison_manager.jpg"))
        plt.close()
        print(f"Saved: {os.path.join(output_dir, 'comparison_manager.jpg')}")

    # Plot 2: Pilot comparison (if pilot data exists)
    pilot_runs = ["pilot human manager", "pilot rule based manager"]
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

    # Save aggregates
    aggregates = (
        df.groupby(["run", "round_number"])
        .agg({
            "punishment": "mean",
            "contribution": "mean",
            "common_good": "mean",
        })
        .reset_index()
    )
    aggregates_path = os.path.join(output_dir, "aggregates.csv")
    aggregates.to_csv(aggregates_path, index=False)
    print(f"Saved: {aggregates_path}")


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
        print(f"Error: Config file not found: {args.config_path}", file=sys.stderr)
        sys.exit(1)

    # Load config
    config = load_config(args.config_path)
    output_dir = get_output_dir(config, args.config_path)
    basedir = config.get("basedir", ".")

    print(f"Config: {args.config_path}")
    print(f"Output directory: {output_dir}")

    # Create output directory
    make_dir(output_dir)

    # Run simulation
    dfs = run_simulation(config, output_dir)

    # Load pilot data if available
    df_pilot = load_pilot_data(basedir)

    # Combine dataframes
    if df_pilot is not None:
        df = pd.concat([*dfs, df_pilot]).reset_index(drop=True)
    else:
        df = pd.concat(dfs).reset_index(drop=True)

    # Create plots
    create_plots(df, output_dir, config["managers"])

    print("Simulation complete!")


if __name__ == "__main__":
    main()
