import wandb
import subprocess
import click
import yaml
import os


def read_file(filename):
    with open(filename, "r") as f:
        return f.read()


def write_file(string, filename):
    with open(filename, "w") as f:
        f.write(string)


def create_script(job_file_target, **kwargs):
    script_str = read_file("slurm_template.sh")
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


if __name__ == "__main__":
    cli()
