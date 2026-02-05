# Reproducing RL Manager Model

To train the RL Manager model you need some human clone models to be trained first before running these. However, you can also use existing artifacts in the `artifacts/` folder.
This project used to use notebooks and `papermill` package to train and evaluate models. However, the up-to-date and the recommended version uses a script pipeline.
The following commands need to be run while the virtual environment is activated. Refer [here](../README.md) to set it up.

## Using Notebooks (Old Version)
This version uses the sub-module `djx` refer to [the main documentation](../README.md) to set that up.
### Training a Model
Copy the data into a data folder as that is the way paths are formatted in the configuration files with this setup.
```bash
cp -r artifacts/behavioral_cloning/* data/behavioral_cloning/*
```
```bash
djx run/manager/17_exp2_group_payoff_heavy_optimize.yml
```
This would take circa 5 hours on a single GPU machine.

### Run Simulation to Evaluate
To evaluate the newly trained RL Manager model, you can run the following  command on interactive GPU node:
```bash
python run.py run notebooks/test_manager/simulate_mixed/12_b_heavies_mixed.yml
```
This would compare the newly trained RL Manager model against existing behavioral cloning models and old RL Manager model.

You should expect similar performance of the newly trained RL Manager model against the old RL Manager model as shown in the report.

## Using Scripts 

### Training a Model
You can directly run this to train passing the path to the config file to use as the sole argument:
```bash
sbatch scripts/run_training.sh configs/training/01_rnn_node.yml
```
### Run Simulation to Evaluate
The following will run the simulations and save the results as configured in the yml file.
```bash
python src/simulation/run.py configs/simulation/01_compare.yml
```

