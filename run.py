"""Run Notebook.

Usage:
  run.py run <parameter_file_name> [--prepare-only] [--inplace]
  run.py merge <run_folder>
"""
import os
from docopt import docopt
import papermill as pm
import yaml


def make_dir(path):
    try:
        os.makedirs(path)
    except FileExistsError:
        pass


arguments = docopt(__doc__)

if arguments["run"]:
    parameter_file_name = arguments["<parameter_file_name>"]
    with open(parameter_file_name, "r") as f:
        parameter = yaml.load(f, Loader=yaml.FullLoader)
    prepare_only = arguments["--prepare-only"]
    import_path = parameter.pop("__notebook__")
    if "__output_dir__" in parameter:
        output_dir = parameter.pop("__output_dir__")
    else:
        parameter_file_name_wo_ext = parameter_file_name.split(".")[0]
        output_dir = parameter_file_name_wo_ext
    make_dir(output_dir)
    parameter["output_dir"] = output_dir
    notebook_base_name = import_path.split("/")[-1].split(".")[0]
    if not arguments["--inplace"]:
        output_path = f"{output_dir}/{notebook_base_name}.ipynb"
        parameter["basedir"] = "."
    else:
        basedir = "/".join(".." for _ in range(len(import_path.split("/")) - 1))
        parameter["output_dir"] = f"{basedir}/{output_dir}"
        parameter["basedir"] = basedir
        output_path = import_path
else:
    import_path = "notebooks/merge.ipynb"
    output_path = "notebooks/merge.ipynb"
    parameter = {"run": arguments["<run_folder>"]}


pm.execute_notebook(
    import_path,
    output_path,
    parameters=parameter,
    prepare_only=prepare_only,
)
