import math

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def compute_data_statistics(df: pd.DataFrame):
    """Columns should be (experiment, condition, group, participant_id, value)"""
    out_dict = {}
    dfg = df.groupby(["experiment", "condition", "group"])
    group_means = dfg["value"].mean().reset_index()

    out_dict["between_group_sample_variance"] = group_means["value"].var()

    # out_dict["between_group_variance"] = correct_variance(
    #    group_means["value"].var(), len(group_means["value"])
    # )

    group_vars = dfg["value"].var().reset_index()
    # out_dict["within_group_variance"] = correct_variance(
    #     group_vars["value"].mean(), dfg["value"].count().mean()
    # )
    out_dict["within_group_variance"] = group_vars["value"].mean()

    out_dict["between_group_sample_std"] = math.sqrt(
        out_dict["between_group_sample_variance"]
    )
    out_dict["within_group_std"] = math.sqrt(out_dict["within_group_variance"])

    out_dict["condition_0_mean"] = df[df["condition"] == 0]["value"].mean()
    out_dict["condition_1_mean"] = df[df["condition"] == 1]["value"].mean()

    dfc = df.groupby(["experiment", "condition"])["value"].mean().reset_index()
    dfcp = dfc.pivot(
        index="experiment", columns="condition", values="value"
    ).reset_index(drop=True)
    out_dict["condition_0_higher"] = (dfcp[0] > dfcp[1]).sum()
    out_dict["condition_1_higher"] = (dfcp[1] > dfcp[0]).sum()

    return out_dict


def compute_data_statistics_algoinst_format(df: pd.DataFrame):
    df = df[df["experiment_name"] == "trail_rounds_2"]
    out_dict = {}
    df["group_missing"] = pd.to_numeric(
        df.groupby(["session", "global_group_id", "round_number", "episode"])[
            "player_no_input"
        ].transform("sum")
    )
    df["payoff"] = (
        20
        - df["contribution"]
        - df["punishment"]
        + df["common_good"] / (4 - df["group_missing"])
    )
    dfg = (
        df.groupby(["global_group_id", "participant_code"])["payoff"]
        .mean()
        .reset_index()
    )

    out_dict["between_group_sample_variance"] = (
        df.groupby("global_group_id")["payoff"].mean().reset_index()["payoff"].var()
    )

    md = smf.mixedlm("payoff ~ 1", dfg, groups=dfg["global_group_id"])
    mdf = md.fit(method=["lbfgs"])
    out_dict["between_group_variance"] = float(
        mdf.summary().tables[1].loc["Group Var", "Coef."]
    )

    out_dict["within_group_variance"] = (
        dfg.groupby("global_group_id")["payoff"].var().reset_index()["payoff"].mean()
    )

    # out_dict["within_group_variance"] = correct_variance(
    #    dfg.groupby("global_group_id")["payoff"].var().reset_index()["payoff"].mean(),
    #    dfg.groupby("global_group_id")["payoff"].count().mean(),
    # )

    out_dict["between_group_std"] = math.sqrt(out_dict["between_group_variance"])
    out_dict["between_group_sample_std"] = math.sqrt(
        out_dict["between_group_sample_variance"]
    )
    out_dict["within_group_std"] = math.sqrt(out_dict["within_group_variance"])

    return out_dict
