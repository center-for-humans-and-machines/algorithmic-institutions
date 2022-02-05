import os
import pandas as pd


def find_files_by_name(_dir, name):
    for root, dirs, files in os.walk(_dir):
        for file in files:
            if file == name:
                yield os.path.join(root, file)


def merge_files(files):
    return pd.concat(
        pd.read_parquet(f)
        for f in files
    )


def merge_files_by_name(folder, name):
    files = list(find_files_by_name(folder, name))
    return merge_files(files)