import wandb
import subprocess
import click
import yaml
import os
import json


def read_file(filename):
    with open(filename, "r") as f:
        return f.read()


def write_file(string, filename):
    with open(filename, "w") as f:
        f.write(string)


def create_script(job_file_target, run_type="sweep", project_name=None, **kwargs):
    script_str = read_file("slurm_template.sh")
    # Define command based on the run_type
    if run_type.lower() == "sweep":
        command = f'wandb agent {kwargs["sweep_id"]} --project {project_name} --entity chm-hci --count 10'
    elif run_type.lower() == "run":
        command = f'wandb launch --project={project_name} --entity=chm-hci -c {kwargs["config_file"]}'
    else:
        raise ValueError(
            f'Invalid run_type: {run_type}. Please select either "sweep" or "run".'
        )
    # Add command to kwargs
    kwargs["command"] = command
    # Format the script string
    script_str = script_str.format(**kwargs)
    write_file(script_str, job_file_target)


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_project_name():
    return os.path.basename(os.getcwd())


@click.group(chain=False)
def cli():
    pass


@cli.command("create")
@click.argument("config_yaml")
def sweep(config_yaml):
    project_name = get_project_name()
    click.echo("Project name: " + project_name)
    sweep_name = os.path.splitext(os.path.basename(config_yaml))[0]
    with open(config_yaml) as file:
        config_dict = yaml.load(file, Loader=yaml.FullLoader)
    config_dict["name"] = sweep_name
    sweep_id = wandb.sweep(config_dict, project=project_name)
    click.echo("Sweep ID: " + sweep_id)


@cli.command("local")
@click.argument("sweep_id")
def local(sweep_id):
    project_name = get_project_name()
    print(project_name)
    wandb.agent(sweep_id, project=project_name, entity="chm-hci")


@cli.command("slurm")
@click.argument("sweep_id")
@click.argument("n_nodes")
def run(sweep_id, n_nodes):
    for node in range(int(n_nodes)):
        project_name = get_project_name()
        job_folder = os.path.join("jobs", sweep_id, "node_" + str(node))
        make_dir(job_folder)
        job_name = f"{project_name}_{sweep_id}_{node}"
        log_file = os.path.join(job_folder, "log.txt")
        script_file = os.path.join(job_folder, "script.sh")
        create_script(
            script_file,
            sweep_id=sweep_id,
            job_name=job_name,
            log_file=log_file,
            project_name=project_name,
        )
        subprocess.run(["sbatch", script_file])


# @cli.command("slurm_single")
# @click.argument("config_yaml")
# def run_single(config_yaml):
#     # Load project name and job parameters from the config file
#     with open(config_yaml) as file:
#         config_dict = yaml.load(file, Loader=yaml.FullLoader)

#     project_name = get_project_name()
#     run_name = os.path.splitext(os.path.basename(config_yaml))[0]
#     job_name = f"{project_name}_{run_name}"
#     job_folder = os.path.join("jobs", run_name)

#     make_dir(job_folder)

#     script_file = os.path.join(job_folder, "script.sh")
#     log_file = os.path.join(job_folder, "log.txt")
#     config_json_file = os.path.join(job_folder, "config.json")

#     # Save config as json
#     with open(config_json_file, "w") as f:
#         json.dump(config_dict, f)

#     # Get run type from config dictionary
#     run_type = config_dict.get(
#         "run_type", "sweep"
#     )  # default to "sweep" if "run_type" is not provided

#     # Create the script
#     create_script(
#         script_file,
#         run_type=run_type,  # pass run_type to create_script
#         job_name=job_name,
#         log_file=log_file,
#         project_name=project_name,
#         config_file=config_json_file,
#     )

#     subprocess.run(["sbatch", script_file])


if __name__ == "__main__":
    cli()
