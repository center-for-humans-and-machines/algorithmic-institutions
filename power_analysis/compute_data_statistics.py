import ast
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
    out_dict["c0_between_group_sample_variance"] = group_means[
        group_means["condition"] == 0
    ]["value"].var()
    out_dict["c1_between_group_sample_variance"] = group_means[
        group_means["condition"] == 1
    ]["value"].var()

    # Pandas applies the Bessel correction by default
    group_vars = dfg["value"].var().reset_index()
    out_dict["within_group_variance"] = group_vars["value"].mean()

    out_dict["c0_within_group_variance"] = group_vars[group_vars["condition"] == 0][
        "value"
    ].mean()
    out_dict["c1_within_group_variance"] = group_vars[group_vars["condition"] == 1][
        "value"
    ].mean()

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


def compute_data_statistics_mongo_format(df: pd.DataFrame, sessions=["85dlorg9"]):
    out_dict = {}

    manager_map = {
        "wwjoqsbe": "ci, γ=1",
        "unc1fc2m": "ci, γ=1",
        "ka5v3qsi": "ci, γ=0.9",
        "85dlorg9": "pgh, γ=0.98",
    }
    df = df[df["session"].isin(sessions)]
    df.loc[:, "manager"] = df["session"].map(manager_map)
    df.loc[:, "contributions"] = df["contributions"].apply(ast.literal_eval)
    df.loc[:, "groups"] = df["groups"].apply(ast.literal_eval)
    df.loc[:, "punishments"] = df["punishments"].apply(ast.literal_eval)
    df.loc[:, "participant_codes"] = df["participant_codes"].apply(ast.literal_eval)
    df.loc[:, "missing_inputs"] = df["missing_inputs"].apply(ast.literal_eval)
    dfe = df.explode(
        [
            "participant_codes",
            "contributions",
            "punishments",
            "groups",
            "missing_inputs",
        ]
    )

    missing = dfe["missing_inputs"]
    dfe[missing]["contributions"] = 0
    dfe[missing]["punishments"] = 0
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

    dfe[missing]["payoff"] = 0

    dfg = (
        dfe.groupby(["group_session", "participant_codes"])["payoff"]
        .mean()
        .reset_index()
    )

    out_dict["between_group_sample_variance"] = (
        dfg.groupby("group_session")["payoff"].mean().reset_index()["payoff"].var()
    )
    out_dict["within_group_variance"] = (
        dfg.groupby("group_session")["payoff"].var().reset_index()["payoff"].mean()
    )

    md = sm.MixedLM.from_formula(
        "payoff ~ 1",
        re_formula="1",
        vc_formula={"participant_codes": "0 + C(participant_codes)"},
        groups="group_session",
        data=dfg,
    )
    res = md.fit(method=["lbfgs"])
    out_dict["between_group_variance"] = float(
        res.summary().tables[1].loc["group_session Var", "Coef."]
    )

    out_dict["between_group_std"] = math.sqrt(out_dict["between_group_variance"])
    out_dict["between_group_sample_std"] = math.sqrt(
        out_dict["between_group_sample_variance"]
    )
    out_dict["within_group_std"] = math.sqrt(out_dict["within_group_variance"])

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

    md = sm.MixedLM.from_formula(
        "payoff ~ 1",
        re_formula="1",
        vc_formula={"participant_code": "0 + C(participant_code)"},
        groups="global_group_id",
        data=dfg,
    )
    res = md.fit(method=["lbfgs"])
    out_dict["between_group_variance"] = float(
        res.summary().tables[1].loc["global_group_id Var", "Coef."]
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
