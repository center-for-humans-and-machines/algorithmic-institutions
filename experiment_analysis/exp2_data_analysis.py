import pandas as pd
import os
import numpy as np
import ast
import seaborn as sns
import statsmodels.api as sm
import requests
import matplotlib.pyplot as plt
from scipy import stats

group_map = {
    "ai_governor": "AI",
    "ai_manager": "AI",
    "human_punishment": "Human",
    "human_governor": "Human",
}


def load_df() -> pd.DataFrame:
    internal_pilot_sessions = ["4tv5t74g"]
    pilot_sessions = ["z1awq43i"]
    filename = "data/rounds_exp2.csv"

    df = pd.read_csv(filename)
    df = df[df["session"].isin(pilot_sessions)]
    df["session"].unique()
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
    dfe.loc[dfe["groups"] == "ai_governor", "manager"] = "ai_manager"
    dfe.loc[dfe["groups"] == "human_punishment", "manager"] = "human"

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


def compute_group_sizes_payoffs(
    dfe: pd.DataFrame, special_1_rule: bool = True
) -> pd.DataFrame:
    # Get group sizes for ai_manager and human groups
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
        columns={"ai_manager": "ai_governor_count", "human": "human_governor_count"}
    ).reset_index()
    dfe = dfe.merge(pivoted, on=["session", "round", "group_idx"], how="left")

    dfe.loc[dfe["manager"] == "ai_manager", "group_size"] = dfe["ai_governor_count"]
    dfe.loc[dfe["manager"] == "human", "group_size"] = dfe["human_governor_count"]

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


def plot_over_time(dfe: pd.DataFrame, save_path: str):
    sns.lineplot(data=dfe, x="round", y="contributions", hue="groups", errorbar=None)
    plt.savefig(os.path.join(save_path, "contributions_over_time.png"))
    sns.lineplot(data=dfe, x="round", y="punishments", hue="groups", errorbar=None)
    plt.savefig(os.path.join(save_path, "punishments_over_time.png"))
    sns.lineplot(data=dfe, x="round", y="common_good", hue="groups", errorbar=None)
    plt.savefig(os.path.join(save_path, "common_good_over_time.png"))
    sns.lineplot(data=dfe, x="round", y="payoff", hue="groups", errorbar=None)
    plt.savefig(os.path.join(save_path, "payoff_over_time.png"))


def plot_individual_group_sizes_over_time(dfe: pd.DataFrame, save_path: str):
    _, dfm = get_dfc_dfm(dfe)
    dfm = dfm[
        [
            "session",
            "group_session",
            "round",
            "group_idx",
            "ai_governor_count",
            "human_governor_count",
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
        data=dfms, x="round", y="ai_governor_count", errorbar=None, hue="session_idx"
    )
    plt.savefig(os.path.join(save_path, "ai_governor_count_over_time.png"))
    plt.close()


def basic_plots(dfe: pd.DataFrame, save_path: str):

    dfe["session_idx"] = dfe["session"].astype(str) + "_" + dfe["group_idx"].astype(str)
    sns.lineplot(
        data=dfe, x="round", y="contributions", hue="session_idx", errorbar=None
    )
    plt.title("Average Contributions per session over rounds")
    plt.savefig(os.path.join(save_path, "contributions_session.png"))
    plt.close()
    sns.lineplot(data=dfe, x="round", y="punishments", hue="session_idx", errorbar=None)
    plt.title("Average Punishments per session over rounds")
    plt.savefig(os.path.join(save_path, "punishments_session.png"))
    plt.close()
    sns.lineplot(data=dfe, x="round", y="common_good", hue="session_idx", errorbar=None)
    plt.title("Average Common good per session over rounds")
    plt.savefig(os.path.join(save_path, "common_good_session.png"))
    plt.close()
    sns.lineplot(data=dfe, x="round", y="payoff", hue="session_idx", errorbar=None)
    plt.title("Average Payoff per session over rounds")
    plt.savefig(os.path.join(save_path, "payoff_session.png"))
    plt.close()

    dfe["gsn"] = (
        dfe["groups"].map(group_map)
        + "_"
        + dfe["session"]
        + "_"
        + dfe["group_idx"].astype(str)
    )

    dfh = dfe[dfe["groups"] == "human_punishment"]
    dfai = dfe[dfe["groups"] == "ai_governor"]
    sns.lineplot(
        data=dfh,
        x="round",
        y="contributions",
        hue="gsn",
        errorbar=None,
        linestyle="--",
    )
    sns.lineplot(
        data=dfai,
        x="round",
        y="contributions",
        hue="gsn",
        errorbar=None,
        linestyle="-",
    )
    plt.title("Average Contributions per group over rounds")
    plt.savefig(os.path.join(save_path, "contributions_group.png"))
    plt.close()
    sns.lineplot(
        data=dfh,
        x="round",
        y="punishments",
        hue="gsn",
        errorbar=None,
        linestyle="--",
    )
    sns.lineplot(
        data=dfai,
        x="round",
        y="punishments",
        hue="gsn",
        errorbar=None,
        linestyle="-",
    )
    plt.title("Average Punishments per group over rounds")
    plt.savefig(os.path.join(save_path, "punishments_group.png"))
    plt.close()
    sns.lineplot(
        data=dfh,
        x="round",
        y="common_good",
        hue="gsn",
        errorbar=None,
        linestyle="--",
    )
    sns.lineplot(
        data=dfai,
        x="round",
        y="common_good",
        hue="gsn",
        errorbar=None,
        linestyle="-",
    )
    plt.title("Average Common good per group over rounds")
    plt.savefig(os.path.join(save_path, "common_good_group.png"))
    plt.close()
    sns.lineplot(
        data=dfh,
        x="round",
        y="payoff",
        hue="gsn",
        errorbar=None,
        linestyle="--",
    )
    sns.lineplot(
        data=dfai,
        x="round",
        y="payoff",
        hue="gsn",
        errorbar=None,
        linestyle="-",
    )
    plt.savefig(os.path.join(save_path, "payoff_group.png"))
    plt.title("Average Payoff per group over rounds")
    plt.close()

    sns.lineplot(data=dfe, x="round", y="contributions", hue="groups", errorbar=None)
    plt.savefig(os.path.join(save_path, "contributions_avg.png"))
    plt.close()
    sns.lineplot(data=dfe, x="round", y="punishments", hue="groups", errorbar=None)
    plt.savefig(os.path.join(save_path, "punishments_avg.png"))
    plt.close()
    sns.lineplot(data=dfe, x="round", y="common_good", hue="groups", errorbar=None)
    plt.savefig(os.path.join(save_path, "common_good_avg.png"))
    plt.close()
    sns.lineplot(data=dfe, x="round", y="payoff", hue="groups", errorbar=None)
    plt.savefig(os.path.join(save_path, "payoff_avg.png"))
    plt.close()


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
            "ai_governor_count",
            "human_governor_count",
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


def prepare_switching_correlation_df(dfe: pd.DataFrame) -> pd.DataFrame:
    dfc, dfm = get_dfc_dfm(dfe)

    dfm["round"] += 1
    dfm = dfm[dfm["round"] <= dfc["round"].max()]
    dfm = dfm.rename(
        columns={
            "ai_governor_count": "ai_governor_count_last_round",
            "human_governor_count": "human_governor_count_last_round",
        }
    )
    print("dfm shape:", dfm.shape)

    dfai = dfm[dfm["groups"] == "ai_governor"]
    dfh = dfm[dfm["groups"] == "human_punishment"]
    dfai = dfai.rename(
        columns={
            "contributions": "contributions_last_round_ai",
            "punishments": "punishments_last_round_ai",
            "common_good_share": "common_good_share_last_round_ai",
            "payoff": "payoff_last_round_ai",
        }
    )
    dfh = dfh.rename(
        columns={
            "contributions": "contributions_last_round_h",
            "punishments": "punishments_last_round_h",
            "common_good_share": "common_good_share_last_round_h",
            "payoff": "payoff_last_round_h",
        }
    )
    dfai = dfai.drop(columns=["groups", "group_session"])
    dfh = dfh.drop(columns=["groups", "group_session"])

    dfc = dfc.merge(dfai, on=["session", "round", "group_idx"], how="left")
    dfc = dfc.merge(dfh, on=["session", "round", "group_idx"], how="left")

    dfc["ai_governor_count_last_round"] = dfc[
        "ai_governor_count_last_round_x"
    ].combine_first(dfc["ai_governor_count_last_round_y"])

    dfc["human_governor_count_last_round"] = dfc[
        "human_governor_count_last_round_x"
    ].combine_first(dfc["human_governor_count_last_round_y"])

    # drop the old columns
    dfc = dfc.drop(
        columns=[
            "ai_governor_count_last_round_x",
            "ai_governor_count_last_round_y",
            "human_governor_count_last_round_x",
            "human_governor_count_last_round_y",
        ]
    )
    dfc2 = dfc[["session", "group_idx", "round", "groups", "participant_codes"]]
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

    dfc["last_round_algorithmic"] = (dfc["groups_last_round"] == "ai_governor").astype(
        int
    )
    dfc["chose_algorithmic"] = (dfc["institution_chosen"] == "algorithmic").astype(int)
    dfc["payoff_last_round_delta"] = (
        dfc["payoff_last_round_ai"] - dfc["payoff_last_round_h"]
    )
    dfc["common_good_share_last_round_delta"] = (
        dfc["common_good_share_last_round_ai"] - dfc["common_good_share_last_round_h"]
    )
    dfc["contributions_last_round_delta"] = (
        dfc["contributions_last_round_ai"] - dfc["contributions_last_round_h"]
    )
    dfc["punishments_last_round_delta"] = (
        dfc["punishments_last_round_ai"] - dfc["punishments_last_round_h"]
    )
    dfc["governor_count_delta"] = (
        dfc["ai_governor_count_last_round"] - dfc["human_governor_count_last_round"]
    )

    return dfc


def add_past_round_averages_by_governor(dfc: pd.DataFrame) -> pd.DataFrame:
    """
    For each (session, group_idx, round), add cumulative averages up to the previous round
    for each governor (ai_governor, human_punishment) for the following metrics:
      - contributions
      - punishments
      - common_good_share
      - payoff

    Resulting columns added to dfc:
      - contributions_past_avg_ai, contributions_past_avg_h
      - punishments_past_avg_ai, punishments_past_avg_h
      - common_good_share_past_avg_ai, common_good_share_past_avg_h
      - payoff_past_avg_ai, payoff_past_avg_h

    Notes:
      - The averages are computed over the mean group values per round for the given governor,
        then expanded cumulatively and shifted by one to exclude the current round.
      - First observed round for a (session, group_idx, governor) will be NaN since no past rounds.
    """
    metrics = [
        "contributions",
        "punishments",
        "common_good_share",
        "payoff",
    ]

    # Compute per-round group means by governor
    dfm = (
        dfc[["session", "group_idx", "round", "groups"] + metrics]
        .groupby(["session", "group_idx", "round", "groups"])
        .mean(numeric_only=True)
        .reset_index()
    )

    # Ensure proper ordering for cumulative calculations
    dfm = dfm.sort_values(["session", "group_idx", "groups", "round"])  # stable sort

    # Cumulative averages: up to previous round (past_avg) and including current (to_date_avg)
    for col in metrics:
        dfm[f"{col}_past_avg"] = dfm.groupby(["session", "group_idx", "groups"])[
            col
        ].transform(lambda s: s.expanding().mean().shift(1))
        dfm[f"{col}_to_date_avg"] = dfm.groupby(["session", "group_idx", "groups"])[
            col
        ].transform(lambda s: s.expanding().mean())

    # Split by governor and rename columns
    past_cols = [f"{m}_past_avg" for m in metrics]
    todate_cols = [f"{m}_to_date_avg" for m in metrics]

    dfai = dfm[dfm["groups"] == "ai_governor"][
        ["session", "group_idx", "round"] + past_cols + todate_cols
    ].rename(
        columns={
            **{c: c.replace("_past_avg", "_past_avg_ai") for c in past_cols},
            **{c: c.replace("_to_date_avg", "_to_date_avg_ai") for c in todate_cols},
        }
    )

    dfh = dfm[dfm["groups"] == "human_punishment"][
        ["session", "group_idx", "round"] + past_cols + todate_cols
    ].rename(
        columns={
            **{c: c.replace("_past_avg", "_past_avg_h") for c in past_cols},
            **{c: c.replace("_to_date_avg", "_to_date_avg_h") for c in todate_cols},
        }
    )

    # Merge back to participant-level rows for the same (session, group_idx, round)
    dfc = dfc.merge(dfai, on=["session", "group_idx", "round"], how="left")
    dfc = dfc.merge(dfh, on=["session", "group_idx", "round"], how="left")

    # Fill missing past averages using previous round's to-date averages, per (session, group_idx)
    dfc = dfc.sort_values(["session", "group_idx", "round"])  # stable sort
    past_ai_cols = [f"{m}_past_avg_ai" for m in metrics]
    past_h_cols = [f"{m}_past_avg_h" for m in metrics]
    todate_ai_cols = [f"{m}_to_date_avg_ai" for m in metrics]
    todate_h_cols = [f"{m}_to_date_avg_h" for m in metrics]

    # Previous round's to-date (includes last round where governor existed)
    prev_todate_ai = dfc.groupby(["session", "group_idx"], group_keys=False)[
        todate_ai_cols
    ].apply(lambda g: g.shift(1).ffill())
    prev_todate_h = dfc.groupby(["session", "group_idx"], group_keys=False)[
        todate_h_cols
    ].apply(lambda g: g.shift(1).ffill())

    # Fill NAs in past averages with prev to-date
    for m in metrics:
        dfc[f"{m}_past_avg_ai"] = dfc[f"{m}_past_avg_ai"].fillna(
            prev_todate_ai[f"{m}_to_date_avg_ai"]
        )
        dfc[f"{m}_past_avg_h"] = dfc[f"{m}_past_avg_h"].fillna(
            prev_todate_h[f"{m}_to_date_avg_h"]
        )

    # Optional: forward-fill any remaining gaps in past averages within group
    dfc[past_ai_cols] = dfc.groupby(["session", "group_idx"])[past_ai_cols].ffill()
    dfc[past_h_cols] = dfc.groupby(["session", "group_idx"])[past_h_cols].ffill()

    # Drop helper to-date columns to keep dfc tidy
    drop_cols = todate_ai_cols + todate_h_cols
    dfc = dfc.drop(columns=[c for c in drop_cols if c in dfc.columns])

    # AI - Human deltas for the past-average metrics
    dfc["contributions_past_avg_delta"] = (
        dfc["contributions_past_avg_ai"] - dfc["contributions_past_avg_h"]
    )
    dfc["punishments_past_avg_delta"] = (
        dfc["punishments_past_avg_ai"] - dfc["punishments_past_avg_h"]
    )
    dfc["common_good_share_past_avg_delta"] = (
        dfc["common_good_share_past_avg_ai"] - dfc["common_good_share_past_avg_h"]
    )
    dfc["payoff_past_avg_delta"] = dfc["payoff_past_avg_ai"] - dfc["payoff_past_avg_h"]

    return dfc


def first_round_switching_model(dfc: pd.DataFrame) -> pd.DataFrame:
    dfc = dfc[dfc["round"] <= 2]
    independent_vars = [
        "contributions_last_round_delta",
        "punishments_last_round_delta",
        "last_round_algorithmic",
    ]
    dependent_vars = ["chose_algorithmic"]
    df = dfc.dropna(subset=independent_vars + dependent_vars)
    X = df[independent_vars]
    y = df[dependent_vars]
    X = sm.add_constant(X)
    est = sm.Logit(y, X).fit()
    print(f"First round switching model:")
    print(est.summary())


def switching_model(
    df: pd.DataFrame, normalize: bool = False, lense="contrib_punish"
) -> pd.DataFrame:
    if lense == "contrib_punish":
        independent_vars = [
            "contributions_last_round_delta",
            "punishments_last_round_delta",
            "contributions_past_avg_delta",
            "punishments_past_avg_delta",
            "governor_count_delta",
            "last_round_algorithmic",
        ]
    elif lense == "common_good":
        independent_vars = [
            "common_good_share_last_round_delta",
            "common_good_share_past_avg_delta",
            "governor_count_delta",
            "last_round_algorithmic",
        ]
    elif lense == "payoff":
        independent_vars = [
            "payoff_last_round_delta",
            "payoff_past_avg_delta",
            "governor_count_delta",
            "last_round_algorithmic",
        ]

    dependent_vars = ["chose_algorithmic"]

    df = df.dropna(subset=independent_vars + dependent_vars)

    X = df[independent_vars].copy()
    y = df[dependent_vars]
    if normalize:
        # Standardize to mean 0, std 1; drop zero-variance columns to avoid singular matrix
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


def individual_correlations(df: pd.DataFrame) -> pd.DataFrame:
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
        "last_round_algorithmic",
    ]
    dependent_vars = ["chose_algorithmic"]

    # Debug: print which rows are dropped because of which var
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


def individual_correlations_statsmodels(df: pd.DataFrame) -> pd.DataFrame:
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
        "last_round_algorithmic",
    ]
    dependent_vars = ["chose_algorithmic"]

    for iv in independent_vars:
        ivs = [iv]
        dfa = df.dropna(subset=ivs + dependent_vars)

        X = dfa[ivs]
        y = dfa[dependent_vars]
        X = sm.add_constant(X)
        est = sm.Logit(y, X).fit()
        # Get p-value and coefficient for the independent variable
        coef = est.params[iv]
        pval = est.pvalues[iv]
        print(f"{iv}: coef={coef:.4f}, p-value={pval:.4g}")


if __name__ == "__main__":
    save_path = "plots/exp2"
    os.makedirs(save_path, exist_ok=True)
    df = load_df()
    df = preprocess_df(df)
    print("df shape:", df.shape)
    dfe = explode_df(df)
    print("dfe shape:", dfe.shape)
    dfe = compute_group_sizes_payoffs(dfe)

    dfc = prepare_switching_correlation_df(dfe)
    dfc = add_past_round_averages_by_governor(dfc)
    dfc.to_csv("data/switching_correlation_df.csv", index=False)
    print("dfc shape:", dfc.shape)
    # switching_model(dfc, normalize=True, lense="contrib_punish")
    # switching_model(dfc, normalize=True, lense="common_good")
    # switching_model(dfc, normalize=True, lense="payoff")
    # first_round_switching_model(dfc)
    print("individual correlations")
    individual_correlations(dfc)
    df1 = dfc[dfc["round"] <= 2]
    print("Individual correlations for first round")
    individual_correlations(df1)
    plot_individual_group_sizes_over_time(dfe, save_path)
    basic_plots(dfe, save_path)
