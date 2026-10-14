"""Evolution-strategies manager training orchestrator.

Same job as `run.py` -- archive the config and the SLURM script under
.log/training/manager_es/{config_name}/{job_id}/ and submit -- with the ES
SLURM template and the ES entry point. Kept separate from `run.py` rather than
parameterising it, so the three sibling exploration arms of this comparison
touch no shared launcher.

Usage:
    python src/aimanager/manager/run_es.py <config_path>
"""

import os
import shutil
import subprocess
import sys
import uuid

from aimanager.manager.run import (
    config_name_from_path,
    ensure_dir,
    read_file,
    write_file,
)

TEMPLATE = "scripts/manager/run_training_es.sh"


def run(config_path):
    config_name = config_name_from_path(config_path)
    job_id = str(uuid.uuid4())[:8]

    run_dir = os.path.join(".log", "training", "manager_es", config_name, job_id)
    log_file = os.path.join(run_dir, "log.log")
    job_file = os.path.join(run_dir, "config.yml")
    script_file = os.path.join(run_dir, "run.sh")

    ensure_dir(run_dir)
    shutil.copy2(config_path, job_file)

    script_str = read_file(TEMPLATE).format(
        log_file=log_file,
        job_id=job_id,
        config_path=config_path,
        experiment_name=config_name,
    )
    write_file(script_str, script_file)

    start_command = f"sbatch {script_file}"
    print(start_command)
    print(f"log: {log_file}")
    subprocess.run(start_command, stdout=subprocess.PIPE, shell=True, check=True)


if __name__ == "__main__":
    run(sys.argv[1])
