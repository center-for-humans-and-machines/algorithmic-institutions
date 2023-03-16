"""Run Notebook.

Usage:
  run_notebook.py update <notebook_file> <parameter_file>
  run_notebook.py run <notebook_file> [<parameter_file>]
  run_notebook.py merge <run_folder>
"""
from docopt import docopt
import papermill as pm
import yaml


arguments = docopt(__doc__)


import_path = arguments["<notebook_file>"]

print(arguments)

parameter_path = arguments["<parameter_file>"]
if parameter_path is not None:
    with open(parameter_path, "r") as f:
        parameter = yaml.load(f, Loader=yaml.FullLoader)
    run_name = parameter_path.split("/")[-1]
else:
    parameter = None
    run_name = None

if arguments["update"]:
    output_path = import_path
    cwd = None
    prepare_only = True
elif arguments["run"]:
    temp_folder = "temp/notebooks"
    parameter_file = parameter_path.split("/")[-1]
    output_path = import_path.replace(".ipynb", ".{run_name}.ipynb")
    cwd = "/".join(import_path.split("/")[:-1])
    prepare_only = False
elif arguments["merge"]:
    output_path = import_path
    parameter = {"run": arguments["<run_folder>"]}
else:
    raise ValueError("Invalid arguments")


pm.execute_notebook(
    import_path,
    output_path,
    parameters=parameter,
    cwd=cwd,
    prepare_only=prepare_only,
)
