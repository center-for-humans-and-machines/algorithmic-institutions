"""Unified CLI for AI Manager pipelines.

Usage:
    python -m aimanager train-ah <config>
    python -m aimanager train-manager <config>
    python -m aimanager simulate <config>
    python -m aimanager evaluate <config>
    python -m aimanager validate-rule <rule.yml> [--min-params N]
        [--max-params N|none] [--sobol-points N] [--setting exp2|exp1|exp2_multi|exp2_diff]
"""

import argparse
import os
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


def unwrap_orchestrator_config(config, mode):
    """Return the dict the CLI trains.

    Orchestrator YAMLs keep training keys under ``params`` and SLURM
    fields under ``exec``. The SSH path flattens that before sbatch.
    ``python -m aimanager`` accepts either shape so a command-server
    ``program_call`` can point at the repo config. A top-level ``grid``
    or ``include`` is still the SSH orchestrator's job.
    """
    required = REQUIRED_KEYS[mode]
    if all(key in config for key in required):
        return config
    if "grid" in config or "include" in config:
        print(
            f"Error: config for '{mode}' has 'grid' or 'include'. "
            "The SSH orchestrator expands those "
            "(src/aimanager/artificial_humans/run.py).",
            file=sys.stderr,
        )
        sys.exit(1)
    params = config.get("params")
    if isinstance(params, dict):
        # The orchestrator names the job from the job's own labels before
        # the params' labels (run.py:add_job_ids)
        if "labels" in config and not params.get("job_id"):
            params = {**params, "job_id": _job_id_from_labels(config["labels"])}
        return params
    return config


def _job_id_from_labels(labels):
    """``<k>_<v>__<k>_<v>``, the id artificial_humans/run.py gives a job."""
    return "__".join(f"{k}_{v}" for k, v in (labels or {}).items())


def default_job_id(config, config_path):
    """Return the config with a ``job_id``, as the SSH orchestrator sets it.

    train-ah names its outputs ``<output_dir>/model/<job_id>.pt``. The
    orchestrator (``artificial_humans/run.py:add_job_ids``) joins the
    labels into the id; a config without labels is named after its file.
    A ``job_id`` already in the config is kept.
    """
    if config.get("job_id"):
        return config
    job_id = _job_id_from_labels(config.get("labels"))
    if not job_id:
        job_id = os.path.splitext(os.path.basename(config_path))[0]
    return {**config, "job_id": job_id}


def load_and_validate(config_path, mode):
    """Load YAML config, check required keys, warn on cross-mode keys.

    Returns the parsed config dict or exits with an error message.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    config = unwrap_orchestrator_config(config, mode)
    if mode == "train-ah":
        config = default_job_id(config, config_path)

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
            args.rule, args.min_params, args.max_params, args.sobol_points,
            setting=args.setting,
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
    p.add_argument(
        "--max-params",
        type=lambda v: None if v == "none" else int(v),
        default=4,
        help="a number, or none for no upper bound",
    )
    p.add_argument("--sobol-points", type=int, default=256)
    p.add_argument(
        "--setting",
        default="exp2",
        help="the game the rule is for (manager/rule.py SETTINGS): exp2, exp1, exp2_multi or exp2_diff",
    )

    args = parser.parse_args()
    if args.command == "validate-rule":
        validate_rule(args)
        return

    config = load_and_validate(args.config, args.command)
    DISPATCH[args.command](config, args.config)
