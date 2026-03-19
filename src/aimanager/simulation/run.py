"""Simulation Orchestrator.

Creates per-job directories under .log/simulation/{config_name}/{job_id}/,
archives the config and SLURM script, then submits via sbatch.

Usage:
    python src/aimanager/simulation/run.py <config_path>
"""

import os
import shutil
import subprocess
import sys
import uuid

import yaml


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def read_file(filename):
    with open(filename, "r") as f:
        return f.read()


def write_file(string, filename):
    with open(filename, "w") as f:
        f.write(string)


def ensure_dir(directory):
    os.makedirs(directory, exist_ok=True)


def config_name_from_path(config_path):
    """Derive config name from path, e.g.
    configs/simulation/ah_testing/group.yml -> simulation/ah_testing/group
    """
    path = os.path.normpath(config_path)
    name = os.path.splitext(path)[0]
    parts = name.split(os.sep)
    # Drop leading 'configs' directory if present
    if parts and parts[0] == "configs":
        parts = parts[1:]
    return os.path.join(*parts)


def run(config_path):
    config = load_config(config_path)
    config_name = config_name_from_path(config_path)
    job_id = str(uuid.uuid4())[:8]

    run_dir = os.path.join(".log", config_name, job_id)
    log_file = os.path.join(run_dir, "log.log")
    job_file = os.path.join(run_dir, "config.yml")
    script_file = os.path.join(run_dir, "run.sh")

    ensure_dir(run_dir)

    # Archive config
    shutil.copy2(config_path, job_file)

    # Fill SLURM template
    template = read_file("scripts/run_simulation.sh")
    script_str = template.format(
        log_file=log_file,
        job_id=job_id,
        config_path=config_path,
    )
    write_file(script_str, script_file)

    # Submit
    start_command = f"sbatch {script_file}"
    print(start_command)
    subprocess.run(
        start_command, stdout=subprocess.PIPE, shell=True, check=True
    )


if __name__ == "__main__":
    config_path = sys.argv[1]
    run(config_path)
