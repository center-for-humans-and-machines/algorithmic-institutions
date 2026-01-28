import pandas as pd
import numpy as np


def syn_con_pun(n_contribution, n_punishment):

    prev_contribution = pd.Categorical(
        np.arange(n_contribution), categories=np.arange(n_contribution), ordered=True
    )
    prev_punishment = pd.Categorical(
        np.arange(n_punishment), categories=np.arange(n_punishment), ordered=True
    )

    df_c = pd.DataFrame({
        'prev_contribution': prev_contribution,
        'merge_dummy': 0,
        'round': 1,
    })
    df_p = pd.DataFrame({
        'prev_punishment': prev_punishment,
        'merge_dummy': 0,
        'round': 1,
    })
    df = df_c.merge(df_p)
    df['sample_idx'] = np.arange(len(df))
    df = df.drop(columns='merge_dummy')
    return df