import ast
import os
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt


def prepare_df(file_path: str = "data/round.csv", remove_outliers: bool = False):
    sessions = [
        "rzxe19eg",
        "fx4glgwj",
        "od6716ho",
        "7nki28wg",
        "hhl7yk2l",
        "2iu7mws9",
        "9uxydej0",
        "l4w1qzmz",
        "bzormoyg",
        "iowvzezr",
        "kes3nebx",
        "d9wukpqo",
        "wqehcexq",
    ]
    outlier_managers = [
        "3nylj27q",
        "6fcfgla2",
        "ad53hfkv",
        "v1r8itbk",
        "ysqk4y6b",
    ]  # Everyone with common good < 0
    df = pd.read_csv(file_path)
    df = df[df["session"].isin(sessions)]
    if remove_outliers:
        df = df[~df["manager_code"].isin(outlier_managers)]

    df["contributions"] = df["contributions"].apply(ast.literal_eval)
    df["groups"] = df["groups"].apply(ast.literal_eval)
    df["punishments"] = df["punishments"].apply(ast.literal_eval)
    df["participant_codes"] = df["participant_codes"].apply(ast.literal_eval)
    df["missing_inputs"] = df["missing_inputs"].apply(ast.literal_eval)

    mdf = df.groupby("manager_code")["round"].max().reset_index()
    broken_examples = list(mdf[mdf["round"] < 24]["manager_code"])
    print(f"Removing {len(broken_examples)} broken groups with less than 24 rounds")
    df = df[~df["manager_code"].isin(broken_examples)]

    dfe = df.explode(
        [
            "participant_codes",
            "contributions",
            "punishments",
            "groups",
            "missing_inputs",
        ]
    )
    dfe.loc[dfe["groups"] == "ai_governor", "manager"] = "pgh, γ=0.98"
    dfe.loc[dfe["groups"] == "human_governor", "manager"] = "human"

    missing = dfe["missing_inputs"]
    dfe.loc[missing, "contributions"] = 0
    dfe.loc[missing, "punishments"] = 0

    dfe["group_session"] = (
        dfe["groups"] + "_" + dfe["session"] + dfe["group_idx"].astype(str)
    )
    dfe["group_contributions"] = pd.to_numeric(
        dfe.groupby(["group_session", "round"])["contributions"].transform("sum")
    )
    dfe["group_punishments"] = pd.to_numeric(
        dfe.groupby(["group_session", "round"])["punishments"].transform("sum")
    )
    dfe["group_missing"] = pd.to_numeric(
        dfe.groupby(["group_session", "round"])["missing_inputs"].transform("sum")
    )

    dfe["common_good"] = dfe["group_contributions"] * 1.6 - dfe["group_punishments"]

    dfe["payoff"] = (
        20
        - dfe["contributions"]
        - dfe["punishments"]
        + dfe["common_good"] / (4 - dfe["group_missing"])
    )

    dfe.loc[missing, "payoff"] = 0

    for mn in ["contributions", "punishments", "common_good", "payoff"]:
        dfe[mn] = pd.to_numeric(dfe[mn])
        dfe[f"{mn}_mean"] = dfe.groupby("group_session")[mn].transform("mean")

    dfe["contribution_delta"] = dfe["contributions"] - dfe["contributions_mean"]
    dfe["payoff_delta"] = dfe["payoff"] - dfe["payoff_mean"]
    dfe["punishment_delta"] = dfe["punishments"] - dfe["punishments_mean"]
    dfe["condition"] = dfe["groups"] + dfe["manager_identity_transparent"].apply(
        lambda x: "_transparent" if x else "_opaque"
    )
    return dfe.reset_index(drop=True)


def save_or_show(plot_path: Optional[str] = None):
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    if plot_path:
        plt.savefig(plot_path, dpi=200)
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    df = prepare_df()
    print(df.head())
    print(df.columns)
    print(df["participant_codes"][:10])
    print(type(df.groupby("participant_codes")["payoff"].mean()))
