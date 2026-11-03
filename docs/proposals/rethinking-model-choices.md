# Re-thinking our model choices for algorithmic institutions.

## Status Assessment
We know that our dataset has a lot of the features that we care about as they can be found using linear models (e.g. adaption to new group mean contributions when switching). Still, our graph neural network fails to grasp them.

Our effective dataset size is 50. This is a lot in terms of participant hours we paid because each game needs 10 people for 45 minutes, but it is a tiny training dataset for machine learning.

## Properties of our GNN
Our GNN is powerful and agnostic enough to be able to learn any existing causal relationship between past events and future contributions in the dataset. This can be cool when making an argument about the model really "learning from scratch", because we only encode very limited prior information into our model choice. However, in practice, with a small dataset size, this just means that the model fails to learn basic relationships because it is just not possible to distinguish which of the huge space of possible causal relationships actually holds from such a limited dataset.

### Inductive biases of the GNN
The GNN matches our problems structure in that it represents the participants in a permutation agnostic way and in that it also represents the Groups as permuation agnostic. Arguably, RNNs inherently have an inductive recency bias, which matches our a-priori knowledge about the task. Finally, the GNN aggregation functions (mean/sum) match our intuition that participants care more about the mean contribution in their group than about the contribution dynamics of individuals.

One inductive bias that our GNN notably lacks is in the representation of contribution numbers. Contribution numbers are encoded categorically and thus we don't bias the GNN to treat 7 as more similar to 8 than to 20. Avoiding this bias seems like an odd choice to me and I think that this might be contributing to our overfitting / causality assignment problem.

### The GNN might be too expressive and we might not have enough data to fit it
Our GNN is set up in a way that it can learn any relationship. If we had a huge dataset in which players consistently only contribute if a group member was punished by exactly 4 exactly 6 rounds ago and they were punished by 12 exactly 2 rounds ago, then our GNN would be able to learn this relationship. But this makes credit assignment so hard. There is just too many possible hypothesis for what caused a players contribution to be X in round Y and 50 game trajectories are not enough to pin it down. Reducing the model hidden size or adding regularization can limit the complexity of relationships expressable by the model, but it can not fix the core issue.

## GNN alternatives
Instead of training an almost infinitely expressive GNN, we could design a model that explicitly encodes our a priori knowledge about the task. The canonincal approach is to hand-engineer a set of features that we expect to correlate with the target variable (contribution / switch). Then, we train a model (linear (ridge regression) or GBM (xgboost)) to predict the target variable. Through cross-validation, we can subset/select features that end up being useful.

Features I could come up with right now:
* Round Number
* Group Size
* My Contribution/Punishment/Payoff/Common_Good N rounds earlier than the current one. (for some small N)
* Average Contribution/Punishment/Payoff/Common_Good in my/other group N rounds earlier than the current one. (for some small N)
* Average Contribution/Punishment/Payoff/Common_Good in my/other group since the last switching choice.
* Did I switch last time I had the choice?
* Rounds since last switching option.

We should brainstorm for more features and only part of them will be useful for prediction. We should subset them through cross-validation (one annoying thing: To be proper, we'll need to keep an additional test set that we don't use for CV/feature selection).

I'd represent all numbers in the features as continuous variables and not categorically. Similarly, I'd optimize the MSE to the ground truth contribution for the contribution prediction model and I'd optimize the BCE for the switching prediction.

## What we lose with feature-based approach
1. Maximal expressivity. If for example the behavior of the individual group members through rounds matters and not just the averages, then our model can by design not learn this.
2. Our model becomes less hypothesis-blind: Our feature hand-selection encodes part of our hypotheses.
3. Relation to literature. Some other work used GNNs. But as I heard the literature conclusion is not "GNNs should clearly be the choice for problems like this", so I don't think it matters too much?

## My Recommendation
Ditch the GNN (and thus most of the architectural code we have) and let claude code build up the feature based approach from scratch. Then, start with linear models / ridge regression and try xgboost later.

Alternatively, if we have strong reasons not to ditch the GNN or want to try a little more before completely flipping the table, I'd suggest to change the contribution modelling in the GNN from a categorical to a continuous variable.