# Method

We follow in our our approach
(https://www.nature.com/articles/s41562-022-01383-x) adapeted to the case of
an governing agent imposing personalised punishments. The training process consist of the
following steps and components:

1. We collected in pilot studies human behavior on the task.
2. We trained a behavioral clone of human behavior on the task. We thereby
   trained seperate models for, a) contributions of participants, b) valid
   responses of participants, and c) punishment of a human governour.
3. We trained a optimal governour by training an RL based agorithm to govern
   behavioral clones of human contributor.

## Data Collection

to be filled

## Behavioral Cloning of Contributors

We train behavioral clones that are predicting behavior of human participants
based on historic contributions and punishments of all participants in a group.
We formulate the problem as classification problem, where the model is
predicting the probability of a participant to contribute a certain amount. We
model the influence of the other group members on the contribution of a single
participant by a graph neural network operating on a fully connected network. We
use a recurrent unit to allow for
temporal relationships. We found that both, social influence and temporal
relationships are important for the predictive performance of the model. Details
on the model architecture and its evaluation can be found in the supplementary material.

## Behavioral Cloning of Governour

We also train a behavioral clone predicting the punishment of a human
governour. While the different contributions in a single round are decided upon
independently by the different participants, the punishment is
determined by a single governour. Therefore the punishments within a single
round cannot be considered as independent. For this reason, we model the
punshiments autogressively. Besides the
autoregressive nature, the model architecture is identical to the one predicting
contributions. We found a strong improvement in predictive performance when
including the autoregressive component. Details on the model architecture and
its evaluation can be found in the supplementary material.

## Training an optimal governour

We train through reinforcment learning an governour to optimal punish the
behavioral clones of human contributors to maximize the common good. We use a deep Q-learning to
train the optimal governour maximising the cummulative common good accross all
rounds without discounting. The
architecture of the model is identical to the one predicting contributions, with
the acceptation that we omit the softmax operation and an additional bias term
to account for the cummulative nature of the rewards. Details on
the model architecture and its evaluation can be found in the supplementary
material.

# Supplementary Material

## Model architecture

Our model is structured into three distinct parts, following the graph network
(https://arxiv.org/abs/1806.01261) formalism. We use the same general
architecture for all three models, however we adapt the input features and the
output layer to the specific task. We represent the group as a fully connected
graph composed of four nodes, with each node corresponding to a group member.
Accounting for the direction of influence, we obtain 12 directed edges between
the nodes. Our architecture design ensures permutation symmetry in the
relationships between individuals and incorporates Gated Recurrent Units (GRUs)
to facilitate learning temporal relationships.

As the first module within the graph neural network, an edge model computes a
representation of the relationship between two nodes. To achieve this, we first
construct an edge feature vector by concatenating the features of the source and
target node of the particular edge. The edge model, a single-layer perceptron, is then applied to each of the 12
directed edges.

Second, a node model is applied to each of the four nodes. The output of the
edge model is averaged across all incoming edges of the target node.
Subsequently, the resulting vector is concatenated
with the features of the target node to form the input of the node model. The
node model consists of a single-layer perceptron, followed by a GRU and a final
linear layer. By positioning the GRUs after aggregating pairwise interactions,
we assume that temporal relationships are relevant at the global and individual
levels, but not at the pairwise relationship level.

Depending on the application, the output of the edge model serves as the
contribution, punishment, or Q-value of the node. For predictive models, a
softmax operation is applied to the output of the final layer. All linear layers
(except the final layer) and the GRU share the same number of output units,
referred to as 'hidden units' hereafter. We employ Rectified Linear Units (ReLU)
as the activation function for all layers.

```mermaid
flowchart TD
    subgraph E[Edge Model]
        E0([&]) --> E1[Linear] --> E2[RelU]
    end
    subgraph N[Node Model]
        N0([&]) --> N1[Linear] --> N2[RelU] --> GRU1[GRU] --> N3[Linear]
    end
    subgraph B[Bias Model]
        B0[Linear] --> B2[RelU] --> B3[Linear]
    end

I1[Features Agent A] --> E0
I2[Features Agent B] --> E0
R[Features Round] --> B0

E2 --> M1[Add & Norm] --> N0
I1 --> N0

N3 --> F1
B3 --> F1
F1([+]) --> F2[Softmax] --> F3[Output A]


style F3 stroke-width:0px
style I1 stroke-width:0px
style I2 stroke-width:0px
style R stroke-width:0px
```

## Behavioral Cloning of Contributors

We use supervised learning to train a model to predict the contribution of a
single participant in a given round. The model is trained on behavioral data
collected in pilot studies with a human and a rule based governour. In each round, we predict a multinominal distribution over the 21 possible contributions
for each of the 4 group members. We optimize the model on the cross-entropy
loss. Cases in which participants do not enter a contribution are masked and do
not enter the loss.

Three features enter the model: the contribution of the participant in the
previous round, the punishment of the participant in the previous round, and a
binary variable indicating whether the participant entered a valid contribution
in the previous round. The fist two features are scaled to the range 0 to 1. The latter feature is necessary to account for the fact
that some participants fail to enter a contribution in some rounds. We
investigate the individual importance of the input features on the model
performance, by individually shuffling them and calculating the resulting
increase in loss. We find the model to be dominantly rely on previous
contribution. However, all three features do contribute. We
investigated including additional features, such as the average contributions of the
other participants, but we found that models including the edge model
did not benefit from these additional features. We therefore decided to use the
minimal set of features described above.

We used the neural network architecture described above to train the model.
We train the models using an Adam Optimizer with a learning rate of 0.003.
Gradients are clamped at absolute 1. We train batches of the full episodes of 10
groups. We train for 10000 epochs in total. We used 5 hidden units in all layers.

We investigate the effect of different components of the architecture on the
models cross validated predictive performance.

![Model Comparision](../notebooks/evalutation/predictive_models_autoreg/20_contribution_model_v3/model_comparision_full.jpg)

We found a significant reduction in the cross-validated loss when including the
social influence through the edge model and when including temporal dependencies
through the recurrent unit. The model with both components performed best and we
therefore used it for all further analyses and to train the optimal governour.

We investigated formulating the problem as a regression instead of a
classification. However, shrinkage resulted in missing out on the extremes (i.e.
contributions of 0 or 20). Also predicting contributions as point values does
not allow to capture mixed strategies, i.e. cases in which participants in a
given situation randomly decide between different contributions.

We investigated the influences of the number of hidden units, the batch size
and the learning rate on the model performance. We found that the model to be
robust to these hyperparameters within a reasonable range. With 5 hidden units we choose the
smallest number of hidden units without compromising the model performance.
Thereby we intent to reduce the risk of overfitting, specifically given that
this model is used to train the governour.

## Behavioral Cloning of Gouverneur

Like the model for the contributors, we use supervised learning to train a model
to predict the punishment of a single participant in a given round by a human
governour. The model is trained on behavioral data collected a pilot study with
a human governour. The human governour is punishing all participants on the same
screen in random order. To capture this and the related correlation between the
punishments, we train the model to predict the punishment of all participants in
the same round autorergressively. Toward that goal, during training, we
randomly select a subset of the participants to be predicted and add the
punishment of the other participants as input features. For inference, we
likewise iteratively predict the punishment of each participant and previously
predicted punishments as features.

We use the same architecture as for the model of the contributors. In total 6
feature enter the model: the contribution of the participant in the current
round, the punishment of the participant in the previous round, 



      - name: contribution
        n_levels: 21
        encoding: numeric
      - name: prev_punishment
        n_levels: 31
        encoding: numeric
      - etype: bool
        name: contribution_valid
      - etype: bool
        name: prev_punishment_valid
      - name: punishment_masked
        n_levels: 31
        encoding: numeric
      - etype: bool
        name: autoreg_mask


Features / architecture

parameters selected

hyperparameter optimization

## RL Gouverneur

formulation of the problem

Environment / Reward / Definition of a round

Method / Deep Q learning

The cummulativly expected future rewards (Q-values) are decreasing over time,
as the repeated game is approaching its end. To enable the model to account for
this, we added a bias term specificly depending on the round number.

In the case of the Q-value model, we add an additional bias model. The bias
consists of a single-layer perceptron followed by a ReLU activation function and
a final linear layer. Its output is added to the output of the node model.

parameters selected

hyperparameter optimization

## Simulation and Evaluation

### Simulation

### Evaluation

comparision of average punishment over contribution levels

change over rounds

change dependent on previous behavior of the par

## Evaluation

For all evaluations in the following, we report (if applicable) averages on
cross-validated test sets (k=10). Thereby always a complete group (and their full
episode) is randomly assigned to one of the six folds, to prevent correlation
between folds.

### Features

As input we include the `previous contribution` and the
`previous punishment` of each group member. When the corresponding human
in the pilot did not entered a contribution or punishment we imputed these
values with the corresponding median. Additionally, we include binary
variable that indicate the corresponding validity. We found
including the `round number` and the `previous common good` as features to not
increase performance.

### Hyperparameter

We perform an initial hyperparameter optimisation with the full model as
depicted above.
We did a hyperparameter scan over the number of hidden units, the batch size and
the learning rate. We found that 5 hidden units were performaning best in
avoiding overfitting. Furthermore, we choose a batch size of 10 and a learning
rate of 3.e-4.

![Hidden Size](../notebooks/evalutation/plots/artificial_humans_05_hidden_size/model_comparision.jpg)

### Architecture

We investigate the effect of different components of the architecture on the
models cross validated predictive performance.

![Learning Rate](../notebooks/evalutation/plots/artificial_humans_04_3_model/model_comparision.jpg)

We found a significant improvement for independently adding the node model and for adding the
edge model. However, the model with rnn unit performed significant better then the one with the edge
model. We found only weak evidence for an improvement of adding the edge model
to an model with rnn unit.
Nevertheless, given its superior performance, we use the full model to train the RL manager.

## Model evalution

### Feature importance

We investigate the individual importance of the input features 'previous
contributions', 'previous punishments' and 'previous entry valid' on the model
performance, by individually shuffling them and calculating the resulting loss
in predictive performance.

We find the model to be dominantly rely on previous contribution. However, all
three features do contribute.

![Shuffle Feature](../notebooks/evalutation/plots/artificial_humans_04_model/shuffle_feature_importance.jpg)

### Confusion Matrix

The confusion matrix shows that our model well captures most of the variance and mostly only confuses between close or adjacent contribution levels. The model appears to predominantly predict contributions that are multiples of 5. Looking at the distribution of actual contributions, this appears to be a feature of the behavior of the participants, that (in particular in early rounds) predominantly chose contributions of 5, 10, 15, or 20. The model however appears to well capture the distribution of contributions on average.

![Confusion Matrix](../notebooks/evalutation/plots/artificial_humans_04_model/confusion_matrix.jpg)
_Confusion matrix between predicted and actual contribution (average accross the test sets). For the predictions we are weighting each contribution level with the corresponding probabilty assigned by the model. This is different to a confusion matrix most used for classification problems, where only the class with the highest predicted probability is
considered._

We investigate if the empirical frequency of each contribution level corresponds
to the modeled contribution probability. Both distributions match well and we do
not see any systematic diviations.

![Histogram](../notebooks/evalutation/plots/artificial_humans_04_model/action_histogram.jpg)

## Valid response model

Additional to the model predicting contributions, we also train a second
independent model to predict, wheather an agent is making a valid contribution.
We use the same general architecture, however we now use a binar target (valid
contribution) and we train the model on the full recorded dataset (including
invalid responses).

We use a model only including the boolean information of the previous round
being valid. We are adding a recurrent unit, however, unlike for the model on
contributions, we do not add a edge model.

Our model archives a roc score of 0.61, which suggest a low
predictive power. More importantly, for our purpose, the frequency of a participant
to not enter a valid solution is well represented.

![Confusion Matrix](../notebooks/evalutation/plots/artificial_humans_02_3_valid/action_histogram.jpg)

# Results

## Rule based manager (supplimental)
