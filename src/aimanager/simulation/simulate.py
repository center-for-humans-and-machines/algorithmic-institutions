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
import pandas as pd
import seaborn as sns
import torch as th
import yaml

from aimanager.artificial_humans import GraphNetwork
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

    df_sim = punishments.merge(common_good).merge(contributions).merge(agent_group)

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


def make_round(contributions, round_num, groups, episode_group_idx, agent_group=None):
    """Create a round dictionary."""
    if agent_group is None:
        agent_group = [0] * len(contributions)
    return {
        "contribution": contributions,
        "contribution_valid": [c is not None for c in contributions],
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


def run_simulation(config: dict, output_dir: str) -> list:
    """Run the simulation and return list of DataFrames."""
    # Extract config parameters
    artificial_humans = config["artificial_humans"]
    managers_config = config["managers"]
    n_groups = config["n_groups"]
    n_agents = config["n_agents"]
    n_contributions = config["n_contributions"]
    n_punishments = config["n_punishments"]
    n_rounds = config["n_rounds"]
    n_episode_steps = config["n_episode_steps"]
    n_episodes = config["n_episodes"]
    basedir = config.get("basedir", ".")

    # Setup device
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    rec_device = th.device("cpu")

    print(f"Using device: {device}")

    # Add model_path to managers config
    managers = {
        k: {**v, "model_path": os.path.join(basedir, v["path"])}
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
        f"ah {h} managed by {m}": {"groups": [m] * n_agents, "humans": h}
        for m in managers.keys()
        for h in artificial_humans.keys()
    }

    dfs = []
    for name, run in runs.items():
        print(f"Start run {name}")
        groups = run["groups"]

        # Load artificial humans
        hm_path = os.path.join(
            basedir, artificial_humans[run["humans"]]["contribution_model"]
        )
        hmv_path = os.path.join(
            basedir, artificial_humans[run["humans"]]["valid_model"]
        )

        ah = GraphNetwork.load(hm_path, device=device)
        ah_val = GraphNetwork.load(hmv_path, device=device)

        # Load optional switch predictor
        ah_switch = None
        switch_every = config.get("switch_every", None)
        ah_config = artificial_humans[run["humans"]]
        if "switch_model" in ah_config:
            hms_path = os.path.join(basedir, ah_config["switch_model"])
            ah_switch = GraphNetwork.load(hms_path, device=device)

        agent_groups = config.get("agent_groups", None)
        reward_mode = config.get("reward_mode", "sum")

        # Create environment
        env = ArtificialHumanEnv(
            artifical_humans=ah,
            artifical_humans_valid=ah_val,
            artifical_humans_switch=ah_switch,
            switch_every=switch_every,
            n_agents=n_agents,
            n_contributions=n_contributions,
            n_punishments=n_punishments,
            n_rounds=n_rounds,
            n_groups=n_groups,
            batch_size=1,
            device=device,
            agent_groups=agent_groups,
            reward_mode=reward_mode,
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
                current_agent_group = (
                    state["agent_group"].squeeze().tolist()
                    if "agent_group" in state
                    else None
                )
                round_dict = make_round(
                    contributions,
                    round_number,
                    groups,
                    episode_group_idx,
                    agent_group=current_agent_group,
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
                    episode=e,
                )

                state, reward, done = env.step()
                if done:
                    break

            recorder.next_episode(e)

        dfs.append(mem_to_df(recorder, name=name))

    return dfs


def load_pilot_data(config: dict, basedir: str) -> pd.DataFrame:
    """Load pilot experiment data."""
    data_file = config.get(
        "pilot_data_file", "experiments/pilot_random1_player_round_slim.csv"
    )
    data_file = os.path.join(basedir, data_file)

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
    df: pd.DataFrame, output_dir: str, managers_config: dict, figure_name: str = ""
) -> None:
    """Create and save comparison plots."""
    make_dir(output_dir)

    df["episode"] = df["run"] + "__" + df["episode"].astype(str)

    dfm = df.melt(
        id_vars=["episode", "round_number", "participant_code", "run"],
        value_vars=["punishment", "contribution", "common_good", "payoff"],
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

    # Plot 3: Pilot + simulation overlay (direct comparison)
    sim_runs = [r for r in dfm["run"].unique() if r.startswith("ah ")]
    overlay_runs = pilot_runs + sim_runs
    w_overlay = dfm["run"].isin(overlay_runs)

    if w_overlay.any():
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
    if "group_id" in df.columns:
        per_episode_sizes = (
            df.groupby(
                ["run", "episode", "round_number", "group_id"],
                as_index=False,
            )["participant_code"]
            .nunique()
            .rename(columns={"participant_code": "group_size"})
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
    if "group_id" in df.columns:
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
    if "agent_group" in df.columns:
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

    # Save aggregates
    aggregates = (
        df.groupby(["run", "round_number"])
        .agg(
            {
                "punishment": "mean",
                "contribution": "mean",
                "common_good": "mean",
                "payoff": "mean",
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

    print(f"Config: {config_path}")
    print(f"Output directory: {output_dir}")

    # Create output directory
    make_dir(output_dir)

    # Run simulation
    dfs = run_simulation(config, output_dir)

    # Load pilot data if available
    df_pilot = load_pilot_data(config, basedir)

    # Combine dataframes
    if df_pilot is not None:
        df = pd.concat([*dfs, df_pilot]).reset_index(drop=True)
    else:
        df = pd.concat(dfs).reset_index(drop=True)

    # Create plots
    create_plots(df, output_dir, config["managers"], config["figure_name"])

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
