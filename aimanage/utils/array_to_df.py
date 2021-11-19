import pandas as pd


def using_multiindex(A, columns):
    shape = A.shape
    index = pd.MultiIndex.from_product(
        [range(s) for s in shape], names=columns)
    df = pd.DataFrame({'value': A.flatten()}, index=index).reset_index()
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