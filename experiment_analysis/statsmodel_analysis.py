import pandas as pd
import statsmodels.api as sm
import ast
import fire
import numpy as np
from scipy.stats import norm
from util import prepare_df


def wald_test(cov, params, keys):
    """
    Test H0: β1 + β2 = 0 vs H1: β1 + β2 ≠ 0
    """
    # Sum of the two parameters
    sum_params = sum(params[k] for k in keys)

    # Variance of the sum: Var(β1 + β2) = Var(β1) + Var(β2) + 2*Cov(β1, β2)
    var_sum = (
        cov.loc[keys[0], keys[0]]
        + cov.loc[keys[1], keys[1]]
        + 2 * cov.loc[keys[0], keys[1]]  # Note: + not - for sum
    )

    # Z-statistic
    z_stat = sum_params / np.sqrt(var_sum)

    # Two-tailed p-value
    p_value = 2 * (1 - norm.cdf(abs(z_stat)))

    return float(p_value)


def simple_model(df: pd.DataFrame, y_col: str):
    # df["human_condition_revealed"] = (
    #    df["manager_identity_transparent"] & (df["manager"] == "human")
    # ).astype(int)
    # df["ai_condition_revealed"] = (
    #    df["manager_identity_transparent"] & (df["manager"] == "pgh, γ=0.9")
    # ).astype(int)
    df["transparent"] = (df["manager_identity_transparent"]).astype(int)
    df["is_human_governor"] = (df["manager"] == "human").astype(int)
    df["interaction"] = df["is_human_governor"] * df["transparent"]
    independent_vars = [
        "is_human_governor",
        "transparent",
        "interaction",
    ]
    X = df[independent_vars]
    y = df[y_col]
    X = sm.add_constant(X)
    print("\nX column types:")
    print(X.dtypes)
    print("\ny type:")
    print(y.dtype)
    print(X.head())
    print(y.head())
    est = sm.OLS(y, X).fit()
    print(est.summary())

    # Extract coefficients and p-values as dicts
    coefficients = est.params.to_dict()
    p_values = est.pvalues.to_dict()

    cov = est.cov_params()
    p_values["any_transparency_effect"] = wald_test(
        cov, coefficients, ["transparent", "interaction"]
    )

    return p_values, coefficients


def limit_rounds(df: pd.DataFrame, max_round: int = 24):
    df = df[df["round"] <= max_round]
    return df


def limit_40(df: pd.DataFrame):
    top_40_sessions = []
    for condition, dfc in df.groupby("condition"):
        top_40_sessions.extend(list(dfc["group_session"].unique())[:40])
    df = df[df["group_session"].isin(top_40_sessions)]
    return df


def group_df(df: pd.DataFrame):
    dfg = (
        df.groupby(["manager", "condition", "session", "manager_identity_transparent"])
        .mean(numeric_only=True)
        .reset_index()
    )
    return dfg


def main(
    limit_round: int = -1, remove_outliers: bool = False, do_limit_40: bool = True
):
    dfe = prepare_df(remove_outliers=remove_outliers)
    if limit_round != -1:
        dfe = limit_rounds(dfe, limit_round)
    if do_limit_40:
        dfe = limit_40(dfe)
    dfg = group_df(dfe)
    all_p_values = {}
    all_coefficients = {}
    for dependent_var in ["payoff", "contributions", "common_good"]:
        p_values, coefficients = simple_model(dfg, dependent_var)
        all_p_values[dependent_var] = p_values
        all_coefficients[dependent_var] = coefficients
    for key in all_p_values:
        print(f"Dependent variable: {key}")
        print(f"P-values: {all_p_values[key]}")
        print(f"Coefficients: {all_coefficients[key]}")


if __name__ == "__main__":
    fire.Fire(main)
