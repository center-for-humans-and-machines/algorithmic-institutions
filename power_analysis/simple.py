# import required modules
from math import sqrt

import fire
from statsmodels.stats.power import TTestIndPower


def solve_power(
    n_samples_per_condition, mean_diff, sample_variance, alpha=0.05, power=0.8
):
    # calculation of effect size
    # size of samples in pilot study
    # Bessel's correction
    v = sample_variance * (n_samples_per_condition / (n_samples_per_condition - 1))

    # calculate the pooled standard deviation
    s = sqrt(v)

    # calculate the effect size
    # (Cohen's d)
    d = mean_diff / s
    print(f"Effect size: {d}")

    # perform power analysis to find sample size
    # for given effect
    obj = TTestIndPower()
    return obj.solve_power(
        effect_size=d, alpha=alpha, power=power, ratio=1, alternative="two-sided"
    )


if __name__ == "__main__":
    fire.Fire(solve_power)
