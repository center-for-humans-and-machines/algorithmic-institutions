# Reproducing RL Manager Model

To train the RL Manager model you need some human clone models to be trained first before running these. However, you can also use existing artifacts in the `artifacts/` folder.
```bash
cp -r artifacts/behavioral_cloning/* data/behavioral_cloning/*
```
Then you can run the following command to train 
```bash
djx run/manager/17_exp2_group_payoff_heavy_optimize.yml
```
This would take circa 5 hours on a single GPU machine.


## Evaluating New RL Manager Model
To evaluate the newly trained RL Manager model, you can run the following  command on interactive GPU node:
```python
python run.py run notebooks/test_manager/simulate_mixed/12_b_heavies_mixed.yml
```
This would compare the newly trained RL Manager model against existing behavioral cloning models and old RL Manager model.

You should expect similar performance of the newly trained RL Manager model against the old RL Manager model as shown in the report.

