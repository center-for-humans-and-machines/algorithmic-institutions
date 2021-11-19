import os
import pandas as pd
import numpy as np
import pickle
import json
import datetime
import sys

import os

class MyEncoder(json.JSONEncoder):
    """
    Taken from:
    https://stackoverflow.com/questions/27050108/convert-numpy-type-to-python
    """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, datetime.datetime):
            return obj.__str__()
        else:
            return super(MyEncoder, self).default(obj)


def make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def store_json(obj, path, name):
    make_dir(path)
    filename = os.path.join(path, name + '.json')
    with open(filename, 'w') as f:
        json.dump(obj, f, cls=MyEncoder)


def store_obj(obj, path, name):
    make_dir(path)
    filename = os.path.join(path, name + '.pkl')
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


def store_df(obj, path, name):
    make_dir(path)
    filename = os.path.join(path, name + '.parquet.gzip')
    # with open(filename, 'wb') as f:
    obj.to_parquet(filename, compression='gzip')


def load_json(filename):
    with open(filename) as f:
        return json.load(f)


def load_df(filename):
    with open(filename, 'rb') as f:
        return pd.read_parquet(f)


def load_obj(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)
