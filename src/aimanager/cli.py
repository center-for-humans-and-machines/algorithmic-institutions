"""Unified CLI for AI Manager pipelines.

Usage:
    python -m aimanager train-ah <config>
    python -m aimanager train-manager <config>
    python -m aimanager simulate <config>
"""

import argparse
import sys
import warnings

import yaml


# -- Config validation tables ----------------------------------------

REQUIRED_KEYS = {
    "train-ah": [
        "data_file",
        "model_args",
        "train_args",
        "optimizer_args",
    ],
    "train-manager": [
        "artificial_humans",
        "manager_args",
        "env_args",
        "n_update_steps",
    ],
    "simulate": [
        "artificial_humans",
        "managers",
        "n_episodes",
        "n_episode_steps",
    ],
}

# key -> (likely mode, warn if used with these modes)
CROSS_MODE_KEYS = {
    "managers": ("simulate", {"train-ah", "train-manager"}),
    "manager_args": ("train-manager", {"train-ah", "simulate"}),
    "n_update_steps": ("train-manager", {"train-ah", "simulate"}),
    "train_args": ("train-ah", {"train-manager", "simulate"}),
}


# -- Helpers ---------------------------------------------------------


def load_and_validate(config_path, mode):
    """Load YAML config, check required keys, warn on cross-mode keys.

    Returns the parsed config dict or exits with an error message.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Check required keys
    missing = [k for k in REQUIRED_KEYS[mode] if k not in config]
    if missing:
        print(
            f"Error: config {config_path} is missing required "
            f"keys for '{mode}': {', '.join(missing)}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Warn on cross-mode key presence
    for key, (likely_mode, warn_modes) in CROSS_MODE_KEYS.items():
        if key in config and mode in warn_modes:
            warnings.warn(
                f"Config contains '{key}' (associated with "
                f"'{likely_mode}'), but mode is '{mode}'. "
                f"Possible config/mode mismatch.",
                stacklevel=2,
            )

    return config


# -- Dispatch --------------------------------------------------------


def dispatch_train_ah(config, config_path):
    from aimanager.artificial_humans.train import main

    main(config)


def dispatch_train_manager(config, config_path):
    from aimanager.rl_manager import main

    main(config)


def dispatch_simulate(config, config_path):
    from aimanager.simulation.simulate import run_cli

    run_cli(config, config_path)


DISPATCH = {
    "train-ah": dispatch_train_ah,
    "train-manager": dispatch_train_manager,
    "simulate": dispatch_simulate,
}


# -- Entry point -----------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        prog="python -m aimanager",
        description="Unified CLI for AI Manager pipelines",
    )
    sub = parser.add_subparsers(dest="command")
    sub.required = True

    for cmd in DISPATCH:
        p = sub.add_parser(cmd)
        p.add_argument(
            "config",
            type=str,
            help="Path to YAML config file",
        )

    args = parser.parse_args()

    config = load_and_validate(args.config, args.command)
    DISPATCH[args.command](config, args.config)
