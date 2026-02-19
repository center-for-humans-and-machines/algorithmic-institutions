# Reproducing RL Manager Model

To train the RL Manager model you need some human clone models to be trained first before running these. However, you can also use existing artifacts in the `artifacts/` folder.

The following commands need to be run while the virtual environment is activated. Refer [here](../README.md) to set it up.

## Training Models
The training runs are configured with config files. Examples can be found [here](../configs/).

### Manager
You can directly run this to train the model, passing the path to the config file to use as the sole argument:
```bash
sbatch scripts/run_training.sh configs/training/01_rnn_node.yml
```
This will keep a log of the experiment under `.logs` folder.

### Artificial Humans
Artificial humans training has a dedicated pipeline which can be run through a python interface. This will document experiment docs under `temp` folder.
```bash
python src/aimanager/artificial_humans/run.py configs/training/artificial_humans/script_22.yml
```

## Simulation
Simulations make use of both kinds of models. To compare results of an updated model to an older one one needs to create config files.

1. [Comparing managers](../configs/simulation/manager_testing/01_compare.yml)
2. [Comparing artificial humans](../configs/simulation/ah_testing/01_compare.yml)

The following will run the simulations and save the results as configured in the yml file.
```bash
# multiple managers
python src/aimanager/simulation/run.py configs/simulation/manager_testing/01_compare.yml
```

This will save results at the `plots` folder.
