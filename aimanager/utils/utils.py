import json
import yaml
import sys
import os


# basics


def save_json(obj, filename):
    with open(filename, 'w') as f:
        json.dump(obj, f)


def load_json(filename):
    with open(filename) as f:
        data = json.load(f)
    return data


def save_yaml(obj, filename):
    with open(filename, 'w') as f:
        yaml.dump(obj, f)


def load_yaml(filename):
    with open(filename) as f:
        data = yaml.safe_load(f)
    return data


def make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def get_all_files(folder):
    for file in os.listdir(folder):
        yield os.path.join(folder, file)


def check_file(fname):
    return os.path.isfile(fname)