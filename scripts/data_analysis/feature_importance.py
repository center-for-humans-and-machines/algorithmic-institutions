"""Analyze feature importance for a target variable.

Usage:
    python scripts/data_analysis/feature_importance.py <config_yaml>
    python scripts/data_analysis/feature_importance.py <config_yaml> \
        --output-dir plots/data_analysis

Reads the training config YAML to extract:
- data_file: path to the CSV dataset
- model_args.x_encoding: feature names
- model_args.y_name: target variable

Derives prev_* features from the raw CSV the same way the
training pipeline does (lag-1 shift per episode and player).

Runs:
1. Mutual information (discrete target)
2. Random forest feature importance
3. Spearman correlation
4. Overall ranking table
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import spearmanr


def derive_features(df):
    """Add prev_* columns matching the training pipeline."""
    df = df.sort_values(
        ["episode_id", "player_id", "round_number"]
    ).copy()

    grp = df.groupby(["episode_id", "player_id"])

    df["prev_contribution"] = grp["contribution"].shift(1)
    df["prev_punishment"] = grp["punishment"].shift(1)
    df["prev_contribution_valid"] = (
        grp["player_no_input"].shift(1).map({0: 1, 1: 0})
    )

    if "group_id" in df.columns:
        df["agent_group"] = df["group_id"].astype(int)
        df["prev_agent_group"] = grp["group_id"].shift(1)

    # Fill first-round NaNs with defaults
    c_med = df.loc[
        df["player_no_input"] == 0, "contribution"
    ].median()
    p_med = df.loc[
        df["player_no_input"] == 0, "punishment"
    ].median()
    df["prev_contribution"] = df["prev_contribution"].fillna(
        c_med
    )
    df["prev_punishment"] = df["prev_punishment"].fillna(p_med)
    df["prev_contribution_valid"] = (
        df["prev_contribution_valid"].fillna(0).astype(int)
    )
    if "prev_agent_group" in df.columns:
        df["prev_agent_group"] = (
            df["prev_agent_group"].fillna(0).astype(int)
        )

    return df


def run_analysis(df, features, target):
    """Run feature importance analyses and print results."""
    valid = df[df["player_no_input"] == 0].copy()
    X = valid[features].values
    y = valid[target].astype(int).values

    print(f"Samples: {len(valid)}, Features: {features}")
    print(f"Target: {target} ({len(np.unique(y))} classes)")
    print()

    # 1. Mutual information
    mi = mutual_info_classif(
        X, y, discrete_features="auto", random_state=42
    )
    mi_df = pd.DataFrame(
        {"feature": features, "mutual_info": mi}
    ).sort_values("mutual_info", ascending=False)
    print("=== Mutual Information ===")
    for _, row in mi_df.iterrows():
        bar = "#" * int(row["mutual_info"] * 40)
        print(
            f"  {row['feature']:<30s} "
            f"{row['mutual_info']:.4f}  {bar}"
        )
    print()

    # 2. Random forest importance
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X, y)
    rf_imp = rf.feature_importances_
    rf_df = pd.DataFrame(
        {"feature": features, "rf_importance": rf_imp}
    ).sort_values("rf_importance", ascending=False)
    print("=== Random Forest Importance ===")
    for _, row in rf_df.iterrows():
        bar = "#" * int(row["rf_importance"] * 80)
        print(
            f"  {row['feature']:<30s} "
            f"{row['rf_importance']:.4f}  {bar}"
        )
    print(f"  OOB-like train accuracy: {rf.score(X, y):.4f}")
    print()

    # 3. Spearman correlation
    print("=== Spearman Correlation ===")
    sp_rows = []
    for i, feat in enumerate(features):
        rho, pval = spearmanr(X[:, i], y)
        sp_rows.append(
            {"feature": feat, "rho": rho, "p_value": pval}
        )
    sp_df = pd.DataFrame(sp_rows).sort_values(
        "rho", ascending=False, key=abs
    )
    for _, row in sp_df.iterrows():
        sig = "***" if row["p_value"] < 0.001 else (
            "**" if row["p_value"] < 0.01 else (
                "*" if row["p_value"] < 0.05 else ""
            )
        )
        print(
            f"  {row['feature']:<30s} "
            f"rho={row['rho']:+.4f}  p={row['p_value']:.2e} "
            f"{sig}"
        )
    print()

    # 4. Summary table
    summary = mi_df.merge(rf_df).merge(sp_df)
    summary["rank_mi"] = (
        summary["mutual_info"]
        .rank(ascending=False)
        .astype(int)
    )
    summary["rank_rf"] = (
        summary["rf_importance"]
        .rank(ascending=False)
        .astype(int)
    )
    summary["rank_sp"] = (
        summary["rho"]
        .abs()
        .rank(ascending=False)
        .astype(int)
    )
    summary["avg_rank"] = (
        summary[["rank_mi", "rank_rf", "rank_sp"]].mean(axis=1)
    )
    summary = summary.sort_values("avg_rank")
    print("=== Overall Ranking ===")
    print(
        summary[
            [
                "feature",
                "mutual_info",
                "rf_importance",
                "rho",
                "avg_rank",
            ]
        ].to_string(index=False, float_format="{:.4f}".format)
    )
    print()


def parse_config(config_path):
    """Extract data_file, features, and target from a YAML
    training config."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    params = cfg.get("params", cfg)
    data_file = params["data_file"]
    model_args = params["model_args"]
    target = model_args["y_name"]

    # Extract feature names from x_encoding
    features = []
    for enc in model_args["x_encoding"]:
        name = enc.get("name", enc.get("etype"))
        if name:
            features.append(name)

    return data_file, features, target


def main():
    parser = argparse.ArgumentParser(
        description="Feature importance analysis"
    )
    parser.add_argument(
        "config",
        help="Path to training config YAML",
    )
    args = parser.parse_args()

    data_file, features, target = parse_config(args.config)

    # Resolve data_file relative to repo root
    repo_root = Path(__file__).resolve().parents[2]
    csv_path = repo_root / data_file
    if not csv_path.exists():
        csv_path = Path(data_file)

    print(f"Config: {args.config}")
    print(f"Data: {csv_path}")
    print(f"Target: {target}")
    print(f"Features from config: {features}")
    print()

    df = pd.read_csv(csv_path)
    df = derive_features(df)

    # Also add round_number as a feature
    if "round_number" not in features:
        features.append("round_number")

    missing = [f for f in features if f not in df.columns]
    if missing:
        print(f"Error: columns not found: {missing}")
        return

    run_analysis(df, features, target)


if __name__ == "__main__":
    main()
