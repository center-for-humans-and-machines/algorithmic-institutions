This project had an experiment 1 in which an RL trained policy model based on a GNN was trained to optimize the payoffs of RL trained human clone contributors. This experiment was without group switching.

Now for experiment 2 we ended up developing a policy finder pipeline for a rule-based policy model that we used instead of a neural network policy. We now want to see if the autoresearch policy finder approach would have also worked for experiment 1:

Your tasks are:
1. Identify the policy model and human clone model used in experiment 1
2. Adapt the policy finder to experiment 1 and select the correct possible input features
3. Run the policy finder pipeline to optimize a policy against experiment 1 human clones. The target metric to optimize should be payoffs of the clone models.
4. Compare performance with the old RL policy.
5. Create a policy finder artifact for this run and make sure to include a comparison with the RL GNN policy.