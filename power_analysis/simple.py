# import required modules
from math import sqrt

import fire
from statsmodels.stats.power import TTestIndPower


def solve_power(
    mean_diff, condition_1_variance, alpha=0.05, power=0.8, condition_2_variance=None
):
    condition_2_variance = (
        condition_1_variance if condition_2_variance is None else condition_2_variance
    )
    c1v, c2v = condition_1_variance, condition_2_variance
    # calculation of effect size
    # size of samples in pilot study

    # calculate the pooled standard deviation
    s = sqrt((c1v + c2v) / 2)

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
