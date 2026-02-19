import importlib
import yaml
import subprocess
import sys
import os
import socket


def ensure_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def get_method(name):
    splitted = name.split(".")
    module_name = ".".join(splitted[:-1])
    method_name = splitted[-1]
    module = importlib.import_module(module_name)
    func = getattr(module, method_name)
    return func


def get_file(func):
    module = func.__module__
    module = sys.modules[module]
    return module.__file__


def save_yaml(obj, filename):
    with open(filename, "w") as f:
        yaml.dump(obj, f)


def load_yaml(filename):
    with open(filename) as f:
        data = yaml.safe_load(f)
    return data


def get_commit(entry):
    func = get_method(entry)
    folder = os.path.dirname(get_file(func))
    commit = subprocess.check_output(
        ["git rev-parse HEAD"], shell=True, cwd=folder
    ).decode()
    return commit


def get_repro(entry):
    func = get_method(entry)
    folder = os.path.dirname(get_file(func))
    repo = subprocess.check_output(
        ["git config --get remote.origin.url"], shell=True, cwd=folder
    ).decode()[:-1]
    return repo


def get_worker_info():
    return socket.gethostname()
