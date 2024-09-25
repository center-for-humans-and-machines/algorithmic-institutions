import pandas as pd

from power_analysis.compute_data_statistics import (
    compute_data_statistics_algoinst_format,
)
from power_analysis.simple import solve_power

if __name__ == "__main__":

    dfp = pd.read_csv("data/pilot_old.csv")
    stats = compute_data_statistics_algoinst_format(dfp)
    print(stats)
    print(
        "Super simple power analysis",
        solve_power(
            n_samples_per_condition=22,
            mean_diff=2,
            sample_variance=stats["between_group_sample_variance"],
        ),
    )
