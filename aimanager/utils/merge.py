import os
import pandas as pd


def find_files_by_name(_dir, name):
    for root, dirs, files in os.walk(_dir):
        for file in files:
            if file == name:
                yield os.path.join(root, file)


def read_pandas(f):
    ext = os.path.splitext(f)[1]
    if ext == '.parquet':
        return pd.read_parquet(f)
    elif ext == '.csv':
        return pd.read_csv(f)


def merge_files(files):
    return pd.concat(
        read_pandas(f)
        for f in files
    )


def merge_files_by_name(folder, name):
    files = list(find_files_by_name(folder, name))
    return merge_files(files)