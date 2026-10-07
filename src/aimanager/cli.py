"""Unified CLI for AI Manager pipelines.

Usage:
    python -m aimanager train-ah <config>
    python -m aimanager train-manager <config>
    python -m aimanager simulate <config>
    python -m aimanager evaluate <config>
    python -m aimanager validate-rule <rule.yml> [--min-params N]
        [--max-params N] [--sobol-points N]
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
    # evaluate reads a finished simulation's output, so it takes the
    # same simulation config
    "evaluate": [
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


def dispatch_evaluate(config, config_path):
    from aimanager.evaluation_suite.evaluate import run_cli

    run_cli(config, config_path)


DISPATCH = {
    "train-ah": dispatch_train_ah,
    "train-manager": dispatch_train_manager,
    "simulate": dispatch_simulate,
    "evaluate": dispatch_evaluate,
}


def validate_rule(args):
    """Check a rule config for a sweep; exit 1 with the reason if it fails."""
    from aimanager.manager.rule import validate_rule

    try:
        design = validate_rule(
            args.rule, args.min_params, args.max_params, args.sobol_points
        )
    except ValueError as e:
        print(f"Invalid: {e}", file=sys.stderr)
        sys.exit(1)
    print(
        f"Valid: {args.rule}, {len(design)} distinct design points"
        f" of {args.sobol_points} Sobol points"
    )


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

    p = sub.add_parser("validate-rule", help="Check a rule config for a sweep")
    p.add_argument("rule", help="Path to the rule YAML")
    p.add_argument("--min-params", type=int, default=1)
    p.add_argument("--max-params", type=int, default=4)
    p.add_argument("--sobol-points", type=int, default=256)

    args = parser.parse_args()
    if args.command == "validate-rule":
        validate_rule(args)
        return

    config = load_and_validate(args.config, args.command)
    DISPATCH[args.command](config, args.config)
