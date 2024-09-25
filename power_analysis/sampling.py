import math
from functools import reduce
from operator import mul

import numpy as np
import pandas as pd


def sample_groups(
    group_size: int,
    n_groups_per_condition: int,
    mean_diff: float,
    between_group_std: float,
    within_group_std: float,
    experiments: int,
):
    r"""
    Let \sigma_b^2 be the between-group variance, \sigma_w^2 be the within-group variance, and g be the group size.
    Let \alpha_j be the normally distributed random effect for the j-th group with variance \sigma_b^2.
    Let \epsilon_ij be the normally distributed random effect for each data sample with variance \sigma_w^2.
    Let \mu be the mean of the data samples and \beta_c be the fixed effect for condition C.
    Then the i-th data sample in the j-th group for condition C1 is given by:
    Y_ij = \mu + \beta_c + \alpha_j + \epsilon_ij

    We can ignore the fixed offset \mu
    """
    sampled_group_effects = np.random.normal(
        0, between_group_std, n_groups_per_condition * 2 * experiments
    ).reshape((experiments, 2, n_groups_per_condition))
    sampled_group_effects = np.repeat(
        sampled_group_effects[..., np.newaxis], group_size, axis=3
    )
    sampled_individual_effects = np.random.normal(
        0, within_group_std, group_size * 2 * experiments * n_groups_per_condition
    ).reshape((experiments, 2, n_groups_per_condition, group_size))

    effect = np.zeros((experiments, 2, n_groups_per_condition, group_size))
    effect[:, 1] = mean_diff

    return effect + sampled_group_effects + sampled_individual_effects


def pandas_transform(np_data):
    indices = np.array(list(np.ndindex(np_data.shape)))
    return pd.DataFrame(
        {
            "experiment": indices[:, 0],
            "condition": indices[:, 1],
            "group": indices[:, 2],
            "participant_id": indices[:, 3],
            "value": np_data.flatten(),
        }
    )
