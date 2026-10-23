import argparse
import ast
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
import statsmodels.api as sm
from scipy import stats


@dataclass(frozen=True)
class GroupConfig:
    group_a: str
    group_b: str
    label_a: str
    label_b: str
    chose_a_value: str


KNOWN_GROUPS = {
    frozenset({"ai_governor", "human_punishment"}): GroupConfig(
        "ai_governor", "human_punishment", "AI", "Human", "algorithmic"
    ),
    frozenset({"governorA", "governorB"}): GroupConfig(
        "governorA", "governorB", "Governor A", "Governor B", "governorA"
    ),
}


def detect_groups(dfe: pd.DataFrame) -> GroupConfig:
    groups = sorted(dfe["groups"].dropna().unique())
    key = frozenset(groups)
    if key in KNOWN_GROUPS:
        return KNOWN_GROUPS[key]
    return GroupConfig(groups[0], groups[1], groups[0], groups[1], groups[0])


def _finish_plot(save_path: str | None, filename: str):
    if save_path is None:
        plt.show()
    else:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, filename))
    plt.close()


def load_df(sessions: list[str], filename: str) -> pd.DataFrame:
    df = pd.read_csv(filename)
    df = df[df["session"].isin(sessions)]
    return df


def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    df["contributions"] = df["contributions"].apply(ast.literal_eval)
    df["groups"] = df["groups"].apply(ast.literal_eval)
    df["punishments"] = df["punishments"].apply(ast.literal_eval)
    df["participant_codes"] = df["participant_codes"].apply(ast.literal_eval)
    df["missing_inputs"] = df["missing_inputs"].apply(ast.literal_eval)
    df["institution_chosen"] = df["institution_chosen"].apply(ast.literal_eval)
    return df


def explode_df(df: pd.DataFrame) -> pd.DataFrame:
    dfe = df.explode(
        [
            "participant_codes",
            "contributions",
            "punishments",
            "groups",
            "missing_inputs",
            "institution_chosen",
        ]
    )
    dfe["manager"] = dfe["groups"]

    missing = dfe["missing_inputs"]
    dfe.loc[missing, "contributions"] = 0
    dfe.loc[missing, "punishments"] = 0

    dfe["group_session"] = (
        dfe["groups"] + "_" + dfe["session"] + "_" + dfe["group_idx"].astype(str)
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

    return dfe


def compute_group_sizes_payoffs(dfe: pd.DataFrame, gcfg: GroupConfig) -> pd.DataFrame:
    group_sizes = dfe.groupby(["session", "round", "group_idx"])[
        "manager"
    ].value_counts()
    group_sizes = group_sizes.reset_index(name="count")
    pivoted = group_sizes.pivot_table(
        index=["session", "round", "group_idx"],
        columns="manager",
        values="count",
        fill_value=0,
    )
    pivoted = pivoted.rename(
        columns={gcfg.group_a: "group_a_count", gcfg.group_b: "group_b_count"}
    ).reset_index()
    dfe = dfe.merge(pivoted, on=["session", "round", "group_idx"], how="left")

    dfe.loc[dfe["manager"] == gcfg.group_a, "group_size"] = dfe["group_a_count"]
    dfe.loc[dfe["manager"] == gcfg.group_b, "group_size"] = dfe["group_b_count"]

    dfe["common_good_share"] = dfe["common_good"] / (
        dfe["group_size"] - dfe["group_missing"]
    ).clip(lower=1)

    dfe["payoff"] = (
        20 - dfe["contributions"] - dfe["punishments"] + dfe["common_good_share"]
    )

    missing = dfe["missing_inputs"]
    dfe.loc[missing, "payoff"] = 0

    for mn in ["contributions", "punishments", "common_good", "payoff"]:
        dfe[mn] = pd.to_numeric(dfe[mn])
        dfe[f"{mn}_mean"] = dfe.groupby("group_session")[mn].transform("mean")

    dfe["contribution_delta"] = dfe["contributions"] - dfe["contributions_mean"]
    dfe["payoff_delta"] = dfe["payoff"] - dfe["payoff_mean"]
    dfe["punishment_delta"] = dfe["punishments"] - dfe["punishments_mean"]
    return dfe


def plot_over_time(dfe: pd.DataFrame, save_path: str | None = None):
    for metric in ["contributions", "punishments", "common_good", "payoff"]:
        sns.lineplot(data=dfe, x="round", y=metric, hue="groups", errorbar=None)
        _finish_plot(save_path, f"{metric}_over_time.png")


def plot_individual_group_sizes_over_time(
    dfe: pd.DataFrame, gcfg: GroupConfig, save_path: str | None = None
):
    _, dfm = get_dfc_dfm(dfe)
    dfm = dfm[
        [
            "session",
            "group_session",
            "round",
            "group_idx",
            "group_a_count",
            "group_b_count",
            "punishments",
        ]
    ]
    dfm["session_idx"] = dfm["session"].astype(str) + "_" + dfm["group_idx"].astype(str)
    dfms = (
        dfm.groupby(["session", "round", "group_idx", "session_idx"])
        .mean(numeric_only=True)
        .reset_index()
    )
    print(dfms.head())

    sns.lineplot(
        data=dfms, x="round", y="group_a_count", errorbar=None, hue="session_idx"
    )
    plt.ylabel(f"{gcfg.label_a} count")
    _finish_plot(save_path, "group_a_count_over_time.png")


def basic_plots(dfe: pd.DataFrame, gcfg: GroupConfig, save_path: str | None = None):
    dfe = dfe.copy()
    dfe["session_idx"] = dfe["session"].astype(str) + "_" + dfe["group_idx"].astype(str)
    for metric in ["contributions", "punishments", "common_good", "payoff"]:
        sns.lineplot(data=dfe, x="round", y=metric, hue="session_idx", errorbar=None)
        plt.title(f"Average {metric.replace('_', ' ').title()} per session over rounds")
        _finish_plot(save_path, f"{metric}_session.png")

    label_map = {gcfg.group_a: gcfg.label_a, gcfg.group_b: gcfg.label_b}
    dfe["gsn"] = (
        dfe["groups"].map(label_map)
        + "_"
        + dfe["session"]
        + "_"
        + dfe["group_idx"].astype(str)
    )

    df_b = dfe[dfe["groups"] == gcfg.group_b]
    df_a = dfe[dfe["groups"] == gcfg.group_a]
    for metric in ["contributions", "punishments", "common_good", "payoff"]:
        sns.lineplot(
            data=df_b, x="round", y=metric, hue="gsn", errorbar=None, linestyle="--"
        )
        sns.lineplot(
            data=df_a, x="round", y=metric, hue="gsn", errorbar=None, linestyle="-"
        )
        plt.title(f"Average {metric.replace('_', ' ').title()} per group over rounds")
        _finish_plot(save_path, f"{metric}_group.png")

    for metric in ["contributions", "punishments", "common_good", "payoff"]:
        sns.lineplot(data=dfe, x="round", y=metric, hue="groups", errorbar=None)
        _finish_plot(save_path, f"{metric}_avg.png")


def get_dfc_dfm(dfe: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    dfc = dfe[
        [
            "session",
            "group_session",
            "group_idx",
            "participant_codes",
            "institution_chosen",
            "round",
            "groups",
            "contributions",
            "punishments",
            "common_good_share",
            "payoff",
            "group_a_count",
            "group_b_count",
        ]
    ]
    dfm = (
        dfc.groupby(
            [
                "session",
                "group_session",
                "round",
                "groups",
                "group_idx",
            ]
        )
        .mean(numeric_only=True)
        .reset_index()
    )
    return dfc, dfm


def prepare_switching_correlation_df(
    dfe: pd.DataFrame, gcfg: GroupConfig
) -> pd.DataFrame:
    dfc, dfm = get_dfc_dfm(dfe)

    dfm["round"] += 1
    dfm = dfm[dfm["round"] <= dfc["round"].max()]
    dfm = dfm.rename(
        columns={
            "group_a_count": "group_a_count_last_round",
            "group_b_count": "group_b_count_last_round",
        }
    )
    print("dfm shape:", dfm.shape)

    dfa = dfm[dfm["groups"] == gcfg.group_a]
    dfb = dfm[dfm["groups"] == gcfg.group_b]
    dfa = dfa.rename(
        columns={
            "contributions": "contributions_last_round_a",
            "punishments": "punishments_last_round_a",
            "common_good_share": "common_good_share_last_round_a",
            "payoff": "payoff_last_round_a",
        }
    )
    dfb = dfb.rename(
        columns={
            "contributions": "contributions_last_round_b",
            "punishments": "punishments_last_round_b",
            "common_good_share": "common_good_share_last_round_b",
            "payoff": "payoff_last_round_b",
        }
    )
    dfa = dfa.drop(columns=["groups", "group_session"])
    dfb = dfb.drop(columns=["groups", "group_session"])

    dfc = dfc.merge(dfa, on=["session", "round", "group_idx"], how="left")
    dfc = dfc.merge(dfb, on=["session", "round", "group_idx"], how="left")

    dfc["group_a_count_last_round"] = dfc["group_a_count_last_round_x"].combine_first(
        dfc["group_a_count_last_round_y"]
    )

    dfc["group_b_count_last_round"] = dfc["group_b_count_last_round_x"].combine_first(
        dfc["group_b_count_last_round_y"]
    )

    dfc = dfc.drop(
        columns=[
            "group_a_count_last_round_x",
            "group_a_count_last_round_y",
            "group_b_count_last_round_x",
            "group_b_count_last_round_y",
        ]
    )
    dfc2 = dfc[["session", "group_idx", "round", "groups", "participant_codes"]].copy()
    dfc2["round"] += 1
    dfc2 = dfc2[dfc2["round"] <= dfc["round"].max()]
    dfc2 = dfc2.rename(
        columns={
            "groups": "groups_last_round",
        }
    )
    dfc = dfc.merge(
        dfc2, on=["session", "group_idx", "round", "participant_codes"], how="left"
    )

    dfc["last_round_a"] = (dfc["groups_last_round"] == gcfg.group_a).astype(int)
    dfc["chose_a"] = (dfc["institution_chosen"] == gcfg.chose_a_value).astype(int)
    dfc["payoff_last_round_delta"] = (
        dfc["payoff_last_round_a"] - dfc["payoff_last_round_b"]
    )
    dfc["common_good_share_last_round_delta"] = (
        dfc["common_good_share_last_round_a"] - dfc["common_good_share_last_round_b"]
    )
    dfc["contributions_last_round_delta"] = (
        dfc["contributions_last_round_a"] - dfc["contributions_last_round_b"]
    )
    dfc["punishments_last_round_delta"] = (
        dfc["punishments_last_round_a"] - dfc["punishments_last_round_b"]
    )
    dfc["governor_count_delta"] = (
        dfc["group_a_count_last_round"] - dfc["group_b_count_last_round"]
    )

    return dfc


def add_past_round_averages_by_governor(
    dfc: pd.DataFrame, gcfg: GroupConfig
) -> pd.DataFrame:
    """
    For each (session, group_idx, round), add cumulative averages up to the previous round
    for each governor group for: contributions, punishments, common_good_share, payoff.

    Adds columns {metric}_past_avg_a, {metric}_past_avg_b, and {metric}_past_avg_delta.
    """
    metrics = [
        "contributions",
        "punishments",
        "common_good_share",
        "payoff",
    ]

    dfm = (
        dfc[["session", "group_idx", "round", "groups"] + metrics]
        .groupby(["session", "group_idx", "round", "groups"])
        .mean(numeric_only=True)
        .reset_index()
    )

    dfm = dfm.sort_values(["session", "group_idx", "groups", "round"])

    for col in metrics:
        dfm[f"{col}_past_avg"] = dfm.groupby(["session", "group_idx", "groups"])[
            col
        ].transform(lambda s: s.expanding().mean().shift(1))
        dfm[f"{col}_to_date_avg"] = dfm.groupby(["session", "group_idx", "groups"])[
            col
        ].transform(lambda s: s.expanding().mean())

    past_cols = [f"{m}_past_avg" for m in metrics]
    todate_cols = [f"{m}_to_date_avg" for m in metrics]

    dfa = dfm[dfm["groups"] == gcfg.group_a][
        ["session", "group_idx", "round"] + past_cols + todate_cols
    ].rename(
        columns={
            **{c: c.replace("_past_avg", "_past_avg_a") for c in past_cols},
            **{c: c.replace("_to_date_avg", "_to_date_avg_a") for c in todate_cols},
        }
    )

    dfb = dfm[dfm["groups"] == gcfg.group_b][
        ["session", "group_idx", "round"] + past_cols + todate_cols
    ].rename(
        columns={
            **{c: c.replace("_past_avg", "_past_avg_b") for c in past_cols},
            **{c: c.replace("_to_date_avg", "_to_date_avg_b") for c in todate_cols},
        }
    )

    dfc = dfc.merge(dfa, on=["session", "group_idx", "round"], how="left")
    dfc = dfc.merge(dfb, on=["session", "group_idx", "round"], how="left")

    dfc = dfc.sort_values(["session", "group_idx", "round"])
    past_a_cols = [f"{m}_past_avg_a" for m in metrics]
    past_b_cols = [f"{m}_past_avg_b" for m in metrics]
    todate_a_cols = [f"{m}_to_date_avg_a" for m in metrics]
    todate_b_cols = [f"{m}_to_date_avg_b" for m in metrics]

    prev_todate_a = dfc.groupby(["session", "group_idx"], group_keys=False)[
        todate_a_cols
    ].apply(lambda g: g.shift(1).ffill())
    prev_todate_b = dfc.groupby(["session", "group_idx"], group_keys=False)[
        todate_b_cols
    ].apply(lambda g: g.shift(1).ffill())

    for m in metrics:
        dfc[f"{m}_past_avg_a"] = dfc[f"{m}_past_avg_a"].fillna(
            prev_todate_a[f"{m}_to_date_avg_a"]
        )
        dfc[f"{m}_past_avg_b"] = dfc[f"{m}_past_avg_b"].fillna(
            prev_todate_b[f"{m}_to_date_avg_b"]
        )

    dfc[past_a_cols] = dfc.groupby(["session", "group_idx"])[past_a_cols].ffill()
    dfc[past_b_cols] = dfc.groupby(["session", "group_idx"])[past_b_cols].ffill()

    drop_cols = todate_a_cols + todate_b_cols
    dfc = dfc.drop(columns=[c for c in drop_cols if c in dfc.columns])

    dfc["contributions_past_avg_delta"] = (
        dfc["contributions_past_avg_a"] - dfc["contributions_past_avg_b"]
    )
    dfc["punishments_past_avg_delta"] = (
        dfc["punishments_past_avg_a"] - dfc["punishments_past_avg_b"]
    )
    dfc["common_good_share_past_avg_delta"] = (
        dfc["common_good_share_past_avg_a"] - dfc["common_good_share_past_avg_b"]
    )
    dfc["payoff_past_avg_delta"] = dfc["payoff_past_avg_a"] - dfc["payoff_past_avg_b"]

    return dfc


def first_round_switching_model(dfc: pd.DataFrame):
    dfc = dfc[dfc["round"] <= 2]
    independent_vars = [
        "contributions_last_round_delta",
        "punishments_last_round_delta",
        "last_round_a",
    ]
    dependent_vars = ["chose_a"]
    df = dfc.dropna(subset=independent_vars + dependent_vars)
    X = df[independent_vars]
    y = df[dependent_vars]
    X = sm.add_constant(X)
    est = sm.Logit(y, X).fit()
    print("First round switching model:")
    print(est.summary())


def switching_model(df: pd.DataFrame, normalize: bool = False, lense="contrib_punish"):
    if lense == "contrib_punish":
        independent_vars = [
            "contributions_last_round_delta",
            "punishments_last_round_delta",
            "contributions_past_avg_delta",
            "punishments_past_avg_delta",
            "governor_count_delta",
            "last_round_a",
        ]
    elif lense == "common_good":
        independent_vars = [
            "common_good_share_last_round_delta",
            "common_good_share_past_avg_delta",
            "governor_count_delta",
            "last_round_a",
        ]
    elif lense == "payoff":
        independent_vars = [
            "payoff_last_round_delta",
            "payoff_past_avg_delta",
            "governor_count_delta",
            "last_round_a",
        ]

    dependent_vars = ["chose_a"]

    df = df.dropna(subset=independent_vars + dependent_vars)

    X = df[independent_vars].copy()
    y = df[dependent_vars]
    if normalize:
        means = X.mean(numeric_only=True)
        stds = X.std(ddof=0, numeric_only=True)
        nonzero_mask = stds > 0
        dropped_cols = stds[~nonzero_mask].index.tolist()
        if len(dropped_cols) > 0:
            print(
                f"Dropping zero-variance columns before normalization: {dropped_cols}"
            )
        X = X.loc[:, nonzero_mask]
        X = (X - means[nonzero_mask]) / stds[nonzero_mask]
    X = sm.add_constant(X)
    est = sm.Logit(y, X).fit()
    print(f"Switching model: {lense}")
    print(est.summary())


def individual_correlations(df: pd.DataFrame):
    independent_vars = [
        "payoff_last_round_delta",
        "common_good_share_last_round_delta",
        "contributions_last_round_delta",
        "punishments_last_round_delta",
        "contributions_past_avg_delta",
        "punishments_past_avg_delta",
        "common_good_share_past_avg_delta",
        "payoff_past_avg_delta",
        "governor_count_delta",
        "last_round_a",
    ]
    dependent_vars = ["chose_a"]

    dropped_info = {}
    for col in independent_vars + dependent_vars:
        missing_mask = df[col].isna()
        dropped_rows = df[missing_mask]
        if not dropped_rows.empty:
            dropped_info[col] = dropped_rows.index.tolist()
    if dropped_info:
        print("Rows dropped due to NaNs in columns:")
        for col, idxs in dropped_info.items():
            print(f"  {col}: {idxs}")
    df = df.dropna(subset=independent_vars + dependent_vars)
    print(df.shape, df.columns)
    df = df[independent_vars + dependent_vars]
    print(df.corr()[dependent_vars])


def individual_correlations_statsmodels(df: pd.DataFrame):
    independent_vars = [
        "payoff_last_round_delta",
        "common_good_share_last_round_delta",
        "contributions_last_round_delta",
        "punishments_last_round_delta",
        "contributions_past_avg_delta",
        "punishments_past_avg_delta",
        "common_good_share_past_avg_delta",
        "payoff_past_avg_delta",
        "governor_count_delta",
        "last_round_a",
    ]
    dependent_vars = ["chose_a"]

    for iv in independent_vars:
        ivs = [iv]
        dfa = df.dropna(subset=ivs + dependent_vars)

        X = dfa[ivs]
        y = dfa[dependent_vars]
        X = sm.add_constant(X)
        est = sm.Logit(y, X).fit()
        coef = est.params[iv]
        pval = est.pvalues[iv]
        print(f"{iv}: coef={coef:.4f}, p-value={pval:.4g}")


PLOT_COMMANDS = [
    "over_time",
    "group_sizes",
    "basic",
]

ANALYSIS_COMMANDS = [
    "correlations",
    "correlations_first",
    "correlations_sm",
    "first_round_model",
    "switching_model",
]

ALL_COMMANDS = PLOT_COMMANDS + ANALYSIS_COMMANDS


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Experiment 2 data analysis and plotting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "commands:\n"
            "  plots:\n"
            "    over_time         line plots of metrics over rounds by group\n"
            "    group_sizes       governor A count over time per session\n"
            "    basic             session / group / avg breakdown plots\n"
            "  analyses:\n"
            "    correlations      bivariate correlations with chose_a\n"
            "    correlations_first  same as correlations, first round only\n"
            "    correlations_sm   individual logit models per variable\n"
            "    first_round_model logit model for first-round switching\n"
            "    switching_model   full switching logit model\n"
            "  meta:\n"
            "    all               run every command (saves plots to --save-path)\n"
            "    all_plots         run all plot commands\n"
            "    all_analyses      run all analysis commands\n"
        ),
    )
    parser.add_argument(
        "command",
        choices=ALL_COMMANDS + ["all", "all_plots", "all_analyses"],
        help="which plot or analysis to run (see list below)",
    )
    parser.add_argument(
        "--data-file",
        default="data/human_human_rounds_exp2.csv",
        help="path to the CSV data file (default: data/human_human_rounds_exp2.csv)",
    )
    parser.add_argument(
        "--sessions",
        nargs="+",
        required=True,
        help="session codes to include",
    )
    parser.add_argument(
        "--save-path",
        default="plots/exp2",
        help="directory to save plots into (default: plots/exp2)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="show plots interactively instead of saving to disk",
    )
    parser.add_argument(
        "--lense",
        default="contrib_punish",
        choices=["contrib_punish", "common_good", "payoff"],
        help="lense for switching_model (default: contrib_punish)",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="normalize independent variables in switching_model",
    )
    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()

    save_path: str | None = None if args.show else args.save_path

    cmds: list[str]
    if args.command == "all":
        cmds = ALL_COMMANDS
    elif args.command == "all_plots":
        cmds = PLOT_COMMANDS
    elif args.command == "all_analyses":
        cmds = ANALYSIS_COMMANDS
    else:
        cmds = [args.command]

    needs_dfc = bool(set(cmds) & set(ANALYSIS_COMMANDS))

    df = load_df(args.sessions, args.data_file)
    df = preprocess_df(df)
    print("df shape:", df.shape)
    dfe = explode_df(df)
    gcfg = detect_groups(dfe)
    print(
        f"Detected groups: {gcfg.label_a} ({gcfg.group_a})"
        f" vs {gcfg.label_b} ({gcfg.group_b})"
    )
    print("dfe shape:", dfe.shape)
    dfe = compute_group_sizes_payoffs(dfe, gcfg)

    if needs_dfc:
        dfc = prepare_switching_correlation_df(dfe, gcfg)
        dfc = add_past_round_averages_by_governor(dfc, gcfg)
        dfc.to_csv("data/switching_correlation_df.csv", index=False)
        print("dfc shape:", dfc.shape)

    for cmd in cmds:
        print(f"\n{'=' * 40}\n  {cmd}\n{'=' * 40}")
        if cmd == "over_time":
            plot_over_time(dfe, save_path)
        elif cmd == "group_sizes":
            plot_individual_group_sizes_over_time(dfe, gcfg, save_path)
        elif cmd == "basic":
            basic_plots(dfe, gcfg, save_path)
        elif cmd == "correlations":
            individual_correlations(dfc)
        elif cmd == "correlations_first":
            df1 = dfc[dfc["round"] <= 2]
            individual_correlations(df1)
        elif cmd == "correlations_sm":
            individual_correlations_statsmodels(dfc)
        elif cmd == "first_round_model":
            first_round_switching_model(dfc)
        elif cmd == "switching_model":
            switching_model(dfc, normalize=args.normalize, lense=args.lense)


if __name__ == "__main__":
    main()
