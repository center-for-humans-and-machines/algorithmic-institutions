import numpy as np


def int_to_ordinal(arr, n_levels):
    """
    Turns a integer series into an ordinal encoding. 
    """
    encoding = np.array(
        [[1]*i + [0]*(n_levels - i - 1)
        for i in range(n_levels)]
    )

    return encoding[arr]

def ordinal_to_int(arr):
    """
    Get the position of the first 0
    """
    n_levels = arr.shape[-1] + 1
    integers = np.array([i for i in range(n_levels-1,0,-1)])
    arr = (1-arr) * integers[np.newaxis]
    return n_levels - arr.max(1) - 1

def outer(a, b):
    return np.einsum('ij,ik->ijk',a, b).reshape(a.shape[0], a.shape[1]*b.shape[1])

def int_encode(data, column=None, ordinal=False):
    if column:
        data = data[column]
    if ordinal:
        n_levels = len(data.cat.categories)
        return int_to_ordinal(data.astype(int).values, n_levels)
    else:
        return data.astype(int).values[:,np.newaxis]

def single_encoder(df, **kwargs):
    return int_encode(df, **kwargs)

def interaction_encoder(df, a, b):
    a_val = single_encoder(df, **a)
    b_val = single_encoder(df, **b)
    return outer(a_val, b_val)

def encoder(df, etype='single', **kwargs):
    if etype == 'interaction':
        return interaction_encoder(df, **kwargs)
    else:
        return single_encoder(df, **kwargs)

def joined_encoder(df, encodings):
    encoding = [
        encoder(df, **e)
        for e in encodings
    ]
    return np.concatenate(encoding, axis=1)   