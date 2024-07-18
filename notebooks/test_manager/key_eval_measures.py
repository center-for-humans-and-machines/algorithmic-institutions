import os
import random
from functools import reduce
from itertools import chain

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure

matplotlib.use("Agg")


def multimerge(dfs, on):
    return reduce(lambda left, right: pd.merge(left, right, on=on, how="outer"), dfs)


def load_data(data_folders, evals_path="evals"):
    def clean_run_name(run_name):
        return run_name.replace("ah full managed by ", "")

    already_runs = []
    dfs = []
    for data_folder in data_folders:
        df = pd.read_csv(os.path.join(evals_path, data_folder, "data.csv"))
        if len(already_runs) > 0:
            df = df[~df["run"].isin(already_runs)]
        already_runs.extend(df["run"].unique())

        dfs.append(df)

    df = pd.concat(dfs)
    df = df[~df["run"].str.contains("ah human")]
    df = df[~df["run"].str.contains("pilot")]
    df = df[~df["run"].str.contains("humanlike")]

    df["payoff"] = 20 - df["contribution"] - df["punishment"] + df["common_good"]

    df["run"] = df["run"].apply(clean_run_name)
    return df


def compute_measure(df: pd.DataFrame, measure: str, max_round=None):
    if max_round is not None:
        df = df[df["round_number"] <= max_round]

    dfs = (
        df[["episode", "participant_code", "run", measure]]
        .groupby(["episode", "participant_code", "run"])
        .sum()
        .reset_index()
    )

    dfm = dfs[["run", measure]].groupby("run").mean().reset_index()
    if max_round is not None:
        dfm = dfm.rename(columns={measure: f"{measure}_maxround_{max_round}"})
    return dfm


def compute_key_metrics(df: pd.DataFrame):
    return multimerge(
        [
            compute_measure(df, measure, max_round)
            for max_round in (4, 12, None)
            for measure in ("common_good", "payoff")
        ],
        on="run",
    )


abbreviation_map = {
    "gamma_07": "ci, γ=0.7",
    "gamma_09": "ci, γ=0.9",
    "gamma_08": "ci, γ=0.8",
    "gamma_05": "ci, γ=0.5",
    "group_payoff": "pg, γ=0.98",
    "group_payoff_heavy": "pgh, γ=0.98",
    "payoff_impact": "pi, γ=0.98",
    "gamma_1": "ci, γ=1",
    "gamma_0": "ci, γ=0",
    "true_common_good_gamma_08": "cg, γ=0.8",
}


def multibar(df: pd.DataFrame, folder="plots"):
    figs, axes = plt.subplots(2, 3, figsize=(20, 7))
    plot_order = [
        ["payoff_maxround_4", "payoff_maxround_12", "payoff"],
        ["common_good_maxround_4", "common_good_maxround_12", "common_good"],
    ]
    my_cmap = plt.get_cmap("brg")

    df["run"] = df["run"].apply(lambda x: abbreviation_map.get(x, x))

    ax: Axes
    for ax, colname in zip(chain(*axes), chain(*plot_order)):
        df.sort_values(colname, ascending=False, inplace=True)
        rescale = lambda y: (y - np.min(y)) / (np.max(y) - np.min(y))
        ax.bar(df["run"], df[colname], color=my_cmap(rescale(df[colname])))
        ax.set_xticklabels(df["run"], rotation=90)
        ax.set_ylabel(colname)

    plt.text(
        0.93,
        0.5,
        "p = payoff\ng = group\nh = heavy optim\ni = individual\nc = common good\nγ = gamma",
        color="black",
        bbox=dict(facecolor="none", edgecolor="black", boxstyle="round,pad=1"),
        transform=plt.gcf().transFigure,
    )
    plt.subplots_adjust(
        top=0.99, bottom=0.15, hspace=0.5, wspace=0.2, right=0.9, left=0.05
    )
    # plt.tight_layout()

    plt.savefig(f"{folder}/key_metrics.png")


if __name__ == "__main__":
    df = load_data(["05_all", "10_few", "11_one", "09_some"])
    metric_df = compute_key_metrics(df)
    print(metric_df)
    multibar(metric_df)
