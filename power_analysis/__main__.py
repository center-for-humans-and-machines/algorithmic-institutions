import argparse

import pandas as pd

from power_analysis.compute_data_statistics import (
    compute_data_statistics,
    compute_data_statistics_algoinst_format,
)
from power_analysis.sampling import pandas_transform, sample_groups
from power_analysis.significance_test import check_percent_significant
from power_analysis.simple import solve_power

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_groups", type=int, default=110)
    parser.add_argument("--experiments", type=int, default=1000)
    args = parser.parse_args()

    dfp = pd.read_csv("experiments/pilot_random1_player_round_slim.csv")
    stats = compute_data_statistics_algoinst_format(dfp)
    print(
        "Simplified power analysis for upper bound. Upper bound required groups per condition for power 0.8",
        solve_power(
            n_samples_per_condition=43,
            mean_diff=2,
            sample_variance=stats["between_group_sample_variance"],
        ),
    )
    data = sample_groups(
        group_size=4,
        n_groups_per_condition=args.sample_groups,
        mean_diff=2,
        between_group_std=stats["between_group_std"],
        within_group_std=stats["within_group_std"],
        experiments=args.experiments,
    )
    df = pandas_transform(data)
    new_stats = compute_data_statistics(df)
    print(f"Stats in pilot dataset: {stats}, Stats in simulated dataset: {new_stats}")
    print(
        f"Percent of experiments significant (power): {check_percent_significant(df)}"
    )
