# Reproducing RL Manager Model

To train the RL Manager model you need some human clone models to be trained first before running these. However, you can also use existing artifacts in the `artifacts/` folder.

The following commands need to be run while the virtual environment is activated. Refer [here](../README.md) to set it up.

## Training a Model
The training is configured in a config file. An example config file can be found [here](../configs/training/01_rnn_node.yml)
You can directly run this to train the model, passing the path to the config file to use as the sole argument:
```bash
sbatch scripts/run_training.sh configs/training/01_rnn_node.yml
```
## Run Simulation to Evaluate
The following will run the simulations and save the results as configured in the yml file.
```bash
python src/simulation/run.py configs/simulation/01_compare.yml
```

