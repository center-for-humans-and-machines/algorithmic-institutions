# Artificial Human Manager

We additionally build a model to mimick the behavior of human manager in the game. We reutilised the
same general architecture used for the artificial human contributors to create such artificial human manager.

We use the training hyperparameter which we found to be working well for the artificial
human participants (learning rate of 3.e-4, weight decay of 1.e-5, clamping of
gradients at 1), with the exception of the batch size, which we increased to 20.
The original batch size of 10 lead to failing convergence.

## Architecture

We run a binary grid over the components a) edge model and b) recurrent unit. We
found no significant improvement in performance for adding a recurrent unit and
adding a the edge model. The best average cross validated performance was
archived by a model with an node and a edge model. Therefore we settled on this
model for the artificial human manager.

![Effect size](../notebooks/evalutation/plots/human_like_manager_05_model/model_comparision.jpg)

## Simulation

We simulated the 10000 groups each of 4 artificial human contributors and a artificial human manager. The simulations show relative
contribution and punishment levels that are relative stable across rounds.

![Final loss](../notebooks/test_manager/plots/simulate_ah_hm/comparison_pilot.jpg)

We also investigate the behavior of larger and smaller groups and found relative
low deviation in both, behavior of the group as well as the behavior of the
manager. A detailed discussion of the comparision of artificial groups and real
group follows in the section "Comparision".

Finally, we compare the average punishments conditioned on the contribution of
the respective group member, between human manager and our human like manager
model. We found both to match within statistical errors.

![Final loss](../notebooks/test_manager/plots/simulate_ah_hm/comparison_pilot_policy.jpg)
