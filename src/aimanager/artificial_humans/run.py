import sys
from time import sleep
import os
from aimanager.artificial_humans.utils import load_yaml, save_yaml
from aimanager.artificial_humans.grid import parse_grid
from aimanager.artificial_humans.merge import deepermerge
from toolz import keyfilter
import uuid
import subprocess


def pick(whitelist, d):
    return keyfilter(lambda k: k in whitelist, d)


def omit(blacklist, d):
    return keyfilter(lambda k: k not in blacklist, d)


def get_uuid():
    return str(uuid.uuid4())


def read_file(filename):
    with open(filename, "r") as f:
        return f.read()


def write_file(string, filename):
    with open(filename, "w") as f:
        f.write(string)


def create_script(script_name, command, **kwargs):
    script_file = f"scripts/artificial_humans/{script_name}.sh"
    script_str = read_file(script_file)
    command = command.format(**kwargs)
    kwargs = {"in_dir": "", **kwargs, "command": command}
    script_str = script_str.format(**kwargs)
    return script_str


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def queue_job(job):
    job_id = job["job_id"]
    temp_dir = job["temp_dir"]
    data_dir = job["data_dir"]

    run_dir = f"{temp_dir}/{job_id}"
    job_file = f"{run_dir}/job.yml"
    script_file = f"{run_dir}/run.sh"
    log_file = f"{run_dir}/log.log"
    output_path = f"{run_dir}"
    job_only = job["exec"].get("job_only", False)

    experiment_name = os.path.splitext(job.get("exp_name", ""))[0] or job_id

    script_str = create_script(
        **job["exec"],
        job_id=job_id,
        job_file=job_file,
        log_file=log_file,
        output_path=output_path,
        run_dir=run_dir,
        experiment_name=experiment_name,
    )

    if job.get("params_only"):
        job = job["params"]

    if "output_path" not in job:
        job["output_path"] = os.path.join(os.getcwd(), output_path)
    job["job_id"] = job_id
    job["temp_dir"] = os.path.join(os.getcwd(), temp_dir)
    job["data_dir"] = os.path.join(os.getcwd(), data_dir)

    ensure_dir(run_dir)
    save_yaml(job, job_file)

    write_file(script_str, script_file)

    if not job_only:
        start_command = f"sbatch {script_file}"
        print(start_command)
        sleep(1)
        subprocess.run(start_command, stdout=subprocess.PIPE, shell=True, check=True)


def add_job_ids(jobs):
    ids = set()
    for idx, job in enumerate(jobs):
        labels = job.get("labels", job["params"].get("labels", {}))
        job_id = "__".join(f"{k}_{v}" for k, v in labels.items())
        ids.add(job_id)
        yield {**job, "job_id": job_id}


def get_subdirs(_dir):
    return [o for o in os.listdir(_dir) if os.path.isdir(os.path.join(_dir, o))]


def parse_parent(base_job, strategy, folder_name, base_dir, parent_name):
    parent_folder = os.path.join(base_dir, folder_name)
    for job_name in get_subdirs(parent_folder):
        parent_job_folder = os.path.join(parent_folder, job_name)
        parent_job_file = os.path.join(parent_job_folder, f"{parent_name}.yml")
        parent_job = load_yaml(parent_job_file)
        labels = {**base_job.get("labels", {}), **parent_job.get("labels", {})}
        _exec = {"in_dir": parent_job_folder, **base_job["exec"]}
        yield {**base_job, "labels": labels, "exec": _exec}


def include_files(exp):
    if "include" in exp:
        include = exp.pop("include")
        for inc in include:
            new = load_yaml(inc)
            exp = deepermerge(exp, new)
    return exp


def run(exp_file):
    exp = load_yaml(exp_file)
    exp = include_files(exp)
    exp_name = os.path.basename(exp_file)
    exp_dir = os.path.splitext(exp_file)[0]
    exp_dir = os.path.normpath(exp_dir)
    exp_dir = exp_dir.split(os.sep)
    temp_dir = os.path.join(exp["exec"]["temp_dir"], *exp_dir[1:])
    data_dir = os.path.join(exp["exec"]["data_dir"], *exp_dir[1:])

    if "grid" in exp:
        base_job = omit(["grid"], exp)
        jobs = parse_grid(exp["grid"], base_job)
    else:
        jobs = [exp]

    exp_id = get_uuid()
    jobs = [
        {
            **t,
            "exp_id": exp_id,
            "temp_dir": temp_dir,
            "data_dir": data_dir,
            "exp_name": exp_name,
        }
        for t in add_job_ids(jobs)
    ]

    for j in jobs:
        queue_job(j)


if __name__ == "__main__":
    config_path = sys.argv[1]
    run(config_path)
