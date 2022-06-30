import pandas as pd


def using_multiindex(A, columns, value_columns=None, value_name='value'):
    shape = A.shape
    if value_columns is not None:
        assert len(columns) == len(shape) - 1
        new_shape = (-1,len(value_columns))
    else:
        assert len(columns) == len(shape)
        new_shape = (-1,)
        value_columns = [value_name]

    index = pd.MultiIndex.from_product(
        [range(s) for s,c in zip(shape, columns)], names=columns)
    df = pd.DataFrame(A.reshape(*new_shape), columns=value_columns, index=index)
    df = df.reset_index()
    return df


def map_columns(df, **map_columns):
    for col, names in map_columns.items():
        df[col] = df[col].map({idx: name for idx, name in enumerate(names)})
    return df


def to_alphabete(df, columns):
    import string
    string_mapper = {x: y for x, y in enumerate(string.ascii_uppercase, 0)}
    for col in columns:
        df[col] = df[col].map(string_mapper)
    return df


def to_alphabete_sm(sr):
    # to_alphabete method used in solution_manipulation
    import string
    string_mapper = {j+i*len(string.ascii_uppercase): l1 + l2 for i, l1 in enumerate(
        string.ascii_uppercase, 0) for j, l2 in enumerate(string.ascii_uppercase, 0)}
    return sr.map(string_mapper)


def add_labels(df, labels):
    label_df = pd.DataFrame(data=labels, dtype='category', index=df.index)
    return pd.concat([label_df, df], axis=1)