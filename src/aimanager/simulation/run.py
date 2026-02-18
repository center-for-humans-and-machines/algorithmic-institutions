"""Simulation Orchestrator.

Entry point for running simulations. Creates output directories and submits
SLURM jobs via sbatch.

Usage:
    python src/simulation/run.py <config_path> [--local]

Examples:
    python src/simulation/run.py configs/simulation/01_compare.yml
    python src/simulation/run.py configs/simulation/01_compare.yml --local
"""

import argparse
import os
import subprocess
import sys

import yaml


def make_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_output_dir(config: dict, config_path: str) -> str:
    """Determine output directory from config or config path."""
    if "output_dir" in config:
        return config["output_dir"]
    # Default: use config path without extension
    return os.path.splitext(config_path)[0]


def validate_config(config: dict, config_path: str) -> None:
    """Validate that required config keys are present."""
    required_keys = ["artificial_humans", "managers", "n_episodes", "n_episode_steps"]
    missing = [k for k in required_keys if k not in config]
    if missing:
        raise ValueError(
            f"Config {config_path} missing required keys: {missing}"
        )


def run_local(config_path: str) -> None:
    """Run simulation locally without SLURM."""
    print(f"Running simulation locally with config: {config_path}")
    result = subprocess.run(
        [sys.executable, "-m", "src.simulation.simulate", config_path],
        check=True,
    )
    return result.returncode


def submit_slurm_job(config_path: str) -> int:
    """Submit simulation job to SLURM via sbatch."""
    script_path = "scripts/run_simulation.sh"
    print(f"Submitting SLURM job with config: {config_path}")
    result = subprocess.run(
        ["sbatch", script_path, config_path],
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Run simulation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "config_path",
        type=str,
        help="Path to simulation config YAML file",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run locally instead of submitting to SLURM",
    )

    args = parser.parse_args()

    # Validate config file exists
    if not os.path.exists(args.config_path):
        print(f"Error: Config file not found: {args.config_path}", file=sys.stderr)
        sys.exit(1)

    # Load and validate config
    config = load_config(args.config_path)
    validate_config(config, args.config_path)

    # Create output directory
    output_dir = get_output_dir(config, args.config_path)
    make_dir(output_dir)
    print(f"Output directory: {output_dir}")

    # Run simulation
    if args.local:
        returncode = run_local(args.config_path)
    else:
        returncode = submit_slurm_job(args.config_path)

    sys.exit(returncode)


if __name__ == "__main__":
    main()
