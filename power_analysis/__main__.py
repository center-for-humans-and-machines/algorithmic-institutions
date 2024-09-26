import argparse
import os

import pandas as pd
import requests

from power_analysis.compute_data_statistics import (
    compute_data_statistics,
    compute_data_statistics_algoinst_format,
    compute_data_statistics_mongo_format,
)
from power_analysis.sampling import pandas_transform, sample_groups
from power_analysis.significance_test import check_percent_significant
from power_analysis.simple import solve_power

if __name__ == "__main__":
    pd.options.mode.chained_assignment = None
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_groups", type=int, default=60)
    parser.add_argument("--experiments", type=int, default=1000)
    parser.add_argument("--consider_new_pilot_data", action="store_true")
    args = parser.parse_args()

    dfp = pd.read_csv("experiments/pilot_random1_player_round_slim.csv")
    mongo_df = pd.read_csv("experiments/aimanager_pilot/aimanager_pilot.csv")

    stats = compute_data_statistics_algoinst_format(dfp)
    mongo_stats = compute_data_statistics_mongo_format(mongo_df)
    print(
        "Simplified power analysis for upper bound. Upper bound required groups per condition for power 0.8",
        solve_power(
            mean_diff=2,
            condition_1_variance=stats["between_group_sample_variance"],
            condition_2_variance=(
                mongo_stats["between_group_sample_variance"]
                if args.consider_new_pilot_data
                else None
            ),
        ),
    )
    data = sample_groups(
        group_size=4,
        n_groups_per_condition=args.sample_groups,
        mean_diff=2,
        between_group_std=stats["between_group_std"],
        within_group_std=stats["within_group_std"],
        experiments=args.experiments,
        condition2_between_group_std=(
            mongo_stats["between_group_std"] if args.consider_new_pilot_data else None
        ),
        condition2_within_group_std=(
            mongo_stats["within_group_std"] if args.consider_new_pilot_data else None
        ),
    )
    df = pandas_transform(data)
    new_stats = compute_data_statistics(df)
    stat_df = pd.DataFrame.from_records(
        [stats, mongo_stats, new_stats], index=["old_pilot", "pgh", "sampled"]
    )
    print(stat_df)
    stat_df.to_csv("data/stats.csv")
    print(
        f"Percent of experiments significant (power): {check_percent_significant(df)}"
    )
