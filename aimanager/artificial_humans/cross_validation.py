
import random
import numpy as np

def split_xy(df, filter_nan=True):
    index = ['session', 'global_group_id', 'episode', 'participant_code', 'round_number']
    df = df.set_index(index).sort_index()
    prev_df = df.groupby(index[:-1]).shift(1)
    rename = {
        c: 'prev_' + c
        for c in prev_df.columns
    }
    prev_df = prev_df.rename(columns=rename)
    if filter_nan:
        w_is_not_null = ~prev_df['prev_contribution'].isnull()
        x = prev_df[w_is_not_null]
        y = df[w_is_not_null]['contribution']
    else:
        x = prev_df
        y = df['contribution']
    return x, y


def get_fraction_of_groups(x_df, y_sr, fraction):
    group_ids = list(x_df.index.levels[1])
    sel_group_ids = np.random.choice(group_ids, size=int(len(group_ids)*fraction), replace=False)
    x_df = x_df.loc[(slice(None),sel_group_ids),:]
    y_sr = y_sr.loc[(slice(None),sel_group_ids)]
    assert x_df.index.equals(y_sr.index)
    assert len(x_df) == len(y_sr)
    return x_df, y_sr


def get_cross_validations(x_df, y_sr, n_splits):
    group_ids = list(x_df.index.levels[1])
    random.shuffle(group_ids)
    groups = [group_ids[i::n_splits] for i in range(n_splits)]

    for i in range(n_splits):
        test_groups = groups[i]
        train_groups = [gg for g in groups for gg in g]
        x_test_df = x_df.loc[(slice(None),test_groups),:]
        x_train_df = x_df.loc[(slice(None),train_groups),:]
        y_test_sr = y_sr.loc[(slice(None),test_groups)]
        y_train_sr = y_sr.loc[(slice(None),train_groups)]
        assert list(x_test_df.reset_index().global_group_id.unique()) == test_groups
        assert list(y_test_sr.reset_index().global_group_id.unique()) == test_groups
        assert list(x_train_df.reset_index().global_group_id.unique()) == train_groups
        assert list(y_train_sr.reset_index().global_group_id.unique()) == train_groups
        assert len(y_test_sr) == len(x_test_df)
        assert len(x_train_df) == len(y_train_sr)
        yield x_train_df, y_train_sr, x_test_df, y_test_sr