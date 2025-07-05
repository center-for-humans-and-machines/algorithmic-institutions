import numpy as np
from tqdm import tqdm, trange
from typing import Generator, Optional
import random
import itertools
import matplotlib.pyplot as plt
import pandas as pd
from scipy.special import expit as logit
import fire
import os
from ykutil import dict_with
from util import prepare_df, save_or_show


def switching_model(
    p_i,
    p_g_ai,
    p_g_h,
    c_g_ai,
    c_g_h,
    size_g_h,
    size_g_ai,
    I_g_ai,
    alpha=0,
    beta_1=0,
    beta_2=0,
    beta_3=0,
    beta_4=0,
    beta_5=0,
):
    """
    Model:
    p_{G_O} = p_{G_H}(I_{AI}) + p_{G_{AI}}(1-I_{AI})

    \mathbb{P}(AI_i) = logit(\alpha +
        \beta_1\frac{p_i-p_{G_O}}{|p_i|+|p_{G_O}|}\cdot(2I_{AI}-1) +
        \beta_2\frac{p_{G_{AI}}-p_{G_H}}{|p_{G_H}| + |p_{G_{AI}}|} +
        \beta_3\frac{c_{G_{AI}}-c_{G_H}}{|c_{G_H}| + |c_{G_{AI}}|} +
        \beta_4(|G_{H}|-|G_{AI}|)/8 +
        \beta_5(2I_{G_{AI}}-1)
    \mathbb{P}(AI_i) is the probability of player i selecting the AI institution

    Args:
        p_i: payoff of player i
        p_g_ai: payoff of G_{AI}
        p_g_h: payoff of G_H
        c_g_ai: common good of G_{AI}
        c_g_h: common good of G_H
        size_g_h: size of G_H
        size_g_ai: size of G_{AI}
        I_g_ai: indicator if player i was in G_{AI}
        alpha: intercept, AI aversion/preference
        beta_1 - beta_5: coefficients
    """

    # Calculate the linear combination
    G_O = p_g_h * I_g_ai + p_g_ai * (1 - I_g_ai)
    linear_combination = (
        alpha
        + beta_1 * (p_i - G_O) / max(abs(p_i) + abs(G_O), 1e-10) * (2 * I_g_ai - 1)
        + beta_2 * (p_g_ai - p_g_h) / max(abs(p_g_h) + abs(p_g_ai), 1e-10)
        + beta_3 * (c_g_ai - c_g_h) / max(abs(c_g_h) + abs(c_g_ai), 1e-10)
        + beta_4 * (size_g_h - size_g_ai) / 8
        + beta_5 * (2 * I_g_ai - 1)
    )

    # Apply logit function to get probability
    probability = logit(linear_combination)

    return probability


def get_pairs(
    df: pd.DataFrame,
) -> Generator[tuple[pd.DataFrame, pd.DataFrame], None, None]:
    group_to_manager_codes = {
        g: list(d["manager_code"].unique()) for g, d in df.groupby("groups")
    }
    combos = list(
        itertools.product(
            group_to_manager_codes["ai_governor"],
            group_to_manager_codes["human_governor"],
        )
    )
    random.shuffle(combos)
    for ai_code, h_code in combos:
        ai_df = df.loc[df["manager_code"] == ai_code]
        h_df = df.loc[df["manager_code"] == h_code]
        yield ai_df, h_df


def compute_statistics(df: pd.DataFrame) -> tuple[int, dict, float, float]:
    group_size = df["participant_codes"].nunique()
    player_payoffs = df.groupby("participant_codes")["payoff"].mean().to_dict()
    group_payoff = df["payoff"].mean()
    group_common_good = df["common_good"].mean()

    return group_size, player_payoffs, group_payoff, group_common_good


def evaluate_param_set(
    df: pd.DataFrame,
    params: dict,
    switching_point: int,
    limit: Optional[int] = None,
    only_last_round: bool = False,
) -> float:
    probs = []
    i = 0
    for ai_df, h_df in get_pairs(df):
        if limit is not None and i > limit:
            break
        if only_last_round:
            ai_df = ai_df[ai_df["round"] == switching_point]
            h_df = h_df[h_df["round"] == switching_point]
        else:
            ai_df = ai_df[ai_df["round"] <= switching_point]
            h_df = h_df[h_df["round"] <= switching_point]
        ai_size, ai_player_payoffs, ai_group_payoff, ai_group_common_good = (
            compute_statistics(ai_df)
        )
        h_size, h_player_payoffs, h_group_payoff, h_group_common_good = (
            compute_statistics(h_df)
        )
        with_indicator = [
            (player_payoff, True) for player_payoff in ai_player_payoffs.values()
        ] + [(player_payoff, False) for player_payoff in h_player_payoffs.values()]
        for player_payoff, indicator in with_indicator:
            prob = switching_model(
                p_i=player_payoff,
                p_g_ai=ai_group_payoff,
                p_g_h=h_group_payoff,
                c_g_ai=ai_group_common_good,
                c_g_h=h_group_common_good,
                size_g_h=h_size,
                size_g_ai=ai_size,
                I_g_ai=indicator,
                **params,
            )
            probs.append(prob)
        i += 1

    mean_prob = float(np.mean(probs))

    print(f"Mean probability: {mean_prob}, expected amount in AI group: {8*mean_prob}")
    return mean_prob


def modify_player_count(df: pd.DataFrame, target_player_count: int):
    player_codes = list(df["participant_codes"].unique())
    actual_player_count = len(player_codes)
    if actual_player_count > target_player_count:
        selected = random.sample(player_codes, target_player_count)
        df = df[df["participant_codes"].isin(selected)]
    elif actual_player_count < target_player_count:
        to_duplicate = random.sample(
            player_codes, target_player_count - actual_player_count
        )
        for code in to_duplicate:
            add_df = df[df["participant_codes"] == code].copy()
            add_df["participant_codes"] = f"{code}_simulated"
            df = pd.concat([df, add_df])

    return df


def whole_game_switching(
    df: pd.DataFrame,
    params: dict,
    switching_n: int,
    limit: Optional[int] = None,
    only_last_round: bool = False,
    players_per_group: int = 4,
) -> list[list[int]]:
    trajectories = []
    i = 0
    max_round = df["round"].max()
    for full_ai_df, full_h_df in tqdm(
        get_pairs(df), desc="Evaluating switching points", total=limit
    ):
        if limit is not None and i > limit:
            break
        num_ai_group = players_per_group
        num_human_group = players_per_group
        trajectory = [num_ai_group]
        trajectories.append(trajectory)
        for n in range(switching_n, max_round + 1, switching_n):
            if only_last_round:
                ai_df = full_ai_df[full_ai_df["round"] == n]
                h_df = full_h_df[full_h_df["round"] == n]
            else:
                ai_df = full_ai_df[
                    (n - switching_n < full_ai_df["round"]) & (full_ai_df["round"] <= n)
                ]
                h_df = full_h_df[
                    (n - switching_n < full_h_df["round"]) & (full_h_df["round"] <= n)
                ]
            ai_df = modify_player_count(ai_df, num_ai_group)
            h_df = modify_player_count(h_df, num_human_group)
            ai_size, ai_player_payoffs, ai_group_payoff, ai_group_common_good = (
                compute_statistics(ai_df)
            )
            h_size, h_player_payoffs, h_group_payoff, h_group_common_good = (
                compute_statistics(h_df)
            )
            with_indicator = [
                (player_payoff, True) for player_payoff in ai_player_payoffs.values()
            ] + [(player_payoff, False) for player_payoff in h_player_payoffs.values()]
            new_num_ai_group = 0
            assert (
                len(with_indicator) == players_per_group * 2
            ), f"{len(with_indicator)} {players_per_group}"
            for player_payoff, indicator in with_indicator:
                prob = switching_model(
                    p_i=player_payoff,
                    p_g_ai=ai_group_payoff,
                    p_g_h=h_group_payoff,
                    c_g_ai=ai_group_common_good,
                    c_g_h=h_group_common_good,
                    size_g_h=h_size,
                    size_g_ai=ai_size,
                    I_g_ai=indicator,
                    **params,
                )
                if random.random() < prob:
                    new_num_ai_group += 1
            num_ai_group = min(max(new_num_ai_group, 1), 2 * players_per_group - 1)
            num_human_group = players_per_group * 2 - num_ai_group
            trajectory.append(num_ai_group)
        i += 1

    return trajectories


def plot_mean_trajectory(
    trajectories: list[list[int]],
    label: str,
    color: str = "blue",
    with_ci: bool = True,
):
    x = range(len(trajectories[0]))
    y = np.array([np.mean([t[i] for t in trajectories]) for i in x])
    if with_ci:
        ci = 1.96 * np.array(
            [
                np.std([t[i] for t in trajectories]) / np.sqrt(len(trajectories))
                for i in x
            ]
        )
        plt.fill_between(x, y - ci, y + ci, alpha=0.2, color=color)
    plt.plot(x, y, label=label, color=color)
    plt.xticks(range(1, len(x) + 1, 2))
    plt.xlabel("Number of switch")
    plt.ylabel("Number of players in AI group")


def plot_some_trajectories(
    trajectories: list[list[int]],
    num_trajectories: int = 10,
    color="blue",
    alpha=0.1,
):
    plt.xticks(range(1, len(trajectories[0]) + 1, 2))
    for trajectory in random.sample(trajectories, num_trajectories):
        plt.plot(range(len(trajectory)), trajectory, alpha=alpha, color=color)


def switching_point_plot(
    df: pd.DataFrame,
    params: dict,
    max_switching_point: int = 24,
    limit: Optional[int] = 1000,
    num_players=8,
    label: Optional[str] = None,
    only_last_round: bool = False,
):
    group_sizes = []
    for switching_point in trange(
        1, max_switching_point + 1, desc="Evaluating switching points"
    ):
        mean_prob = evaluate_param_set(
            df, params, switching_point, limit, only_last_round
        )
        group_sizes.append(num_players * mean_prob)
    plt.plot(range(1, max_switching_point + 1), group_sizes, label=label)
    plt.xlabel("Switching point")
    plt.ylabel("Expected number of players in AI group")
    plt.xticks(range(1, max_switching_point + 1, 2))


def whole_game_switching_plot(
    df: pd.DataFrame,
    params: dict,
    switching_n: int,
    limit: Optional[int] = None,
    only_last_round: bool = False,
    label: Optional[str] = None,
    color: Optional[str] = None,
):
    trajectories = whole_game_switching(df, params, switching_n, limit, only_last_round)
    plot_mean_trajectory(trajectories, label, color=color)
    plot_some_trajectories(trajectories, num_trajectories=10, color=color)


def create_whole_game_switching_plot(df_path: str, plot_folder: str):
    os.makedirs(plot_folder, exist_ok=True)
    df = prepare_df(df_path)
    basic_params = {
        "alpha": 0,
        "beta_1": 0,
        "beta_2": 0,
        "beta_3": 0,
        "beta_4": 0,
        "beta_5": 0,
    }
    for switching_n in [2, 4, 8]:
        for beta_4 in [0, 10]:
            for beta_5 in [0, 10]:
                basic_params["beta_4"] = beta_4
                basic_params["beta_5"] = beta_5
                for only_last_round in [True, False]:
                    whole_game_switching_plot(
                        df,
                        dict_with(basic_params, beta_1=10),
                        switching_n=switching_n,
                        limit=1000,
                        only_last_round=only_last_round,
                        label="Individual payoff (beta_1=10)",
                        color="tab:blue",
                    )
                    whole_game_switching_plot(
                        df,
                        dict_with(basic_params, beta_2=10),
                        switching_n=switching_n,
                        limit=1000,
                        only_last_round=only_last_round,
                        label="Group payoff (beta_2=10)",
                        color="tab:orange",
                    )
                    whole_game_switching_plot(
                        df,
                        dict_with(basic_params, beta_3=10),
                        switching_n=switching_n,
                        limit=1000,
                        only_last_round=only_last_round,
                        label="Common good (beta_3=10)",
                        color="tab:green",
                    )
                    plt.legend()
                    plt.title(
                        f"{'only last round' if only_last_round else 'round average'}, switching_n={switching_n}, beta_4={basic_params['beta_4']}, beta_5={basic_params['beta_5']}"
                    )
                    plt.tight_layout()
                    save_or_show(
                        os.path.join(
                            plot_folder,
                            "only_last_round" if only_last_round else "round_avg",
                            f"switching_n_{switching_n}",
                            f"beta_4_{basic_params['beta_4']}_beta_5_{basic_params['beta_5']}.png",
                        )
                    )


def create_switching_plots(df_path: str, plot_folder: str):
    os.makedirs(plot_folder, exist_ok=True)
    df = prepare_df(df_path)
    basic_params = {
        "alpha": 0,
        "beta_1": 0,
        "beta_2": 0,
        "beta_3": 0,
        "beta_4": 0,
        "beta_5": 0,
    }
    for only_last_round in [True, False]:
        switching_point_plot(
            df,
            dict_with(basic_params, beta_1=10),
            limit=1000,
            label="Individual payoff (beta_1=10)",
            only_last_round=only_last_round,
        )
        switching_point_plot(
            df,
            dict_with(basic_params, beta_2=10),
            limit=1000,
            label="Group payoff (beta_2=10)",
            only_last_round=only_last_round,
        )
        switching_point_plot(
            df,
            dict_with(basic_params, beta_3=10),
            limit=1000,
            label="Common good (beta_3=10)",
            only_last_round=only_last_round,
        )
        plt.legend()
        plt.title(
            f"Switching point plot {'only last round' if only_last_round else 'round average'}"
        )
        save_or_show(
            os.path.join(
                plot_folder,
                f"switching_point_{'only_last_round' if only_last_round else 'round_avg'}.png",
            )
        )


def main():
    df = prepare_df("data/round.csv")
    params = {
        "alpha": 0,
        "beta_1": 5,
        "beta_2": 0,
        "beta_3": 0,
        "beta_4": 0,
        "beta_5": 0,
    }
    switching_point_plot(df, params, limit=1000)


if __name__ == "__main__":
    fire.Fire()
