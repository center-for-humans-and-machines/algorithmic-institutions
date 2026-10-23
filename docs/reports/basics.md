# Revisit Experimental setup

## Game

Four contributor and a manager are forming a group. Each group is playing 16
rounds. At the beginning of the round contributors receive 20 points from the
bank into their private account. Contributors can pay from 0 up to 20 points
into a common pool. The common pool is multiplied by a factor of 1.6. The
manager can punish individual contributors with from 0 up to 30 points. The
punishment is deducted from the contributors private account. Additionally the
punishment is deducted form the common pool. This renders punishments costly for
everyone. At the end of the round the common pool is splitted equally between
the contributors independently of their payments and added to their respective
private accounts. The contributors get a payout proportionally
to their private accounts. The manager is receiving a payout proportionally to
the common pool.

## Pilot Data

### Pilot 1 - Human Manager

We collected ~80 episodes a 8 rounds with a human manager. In this setup a group
would be playing in two episode with the manager changing.

### Pilot 2 - Rule Based Manager

We collected ~45 episodes a 16 rounds with a rule based manager.

In pilot two we utilized a rule to determine the punishments. The rule has the form:
$$pun = (20-cont) \cdot s + (cont != 20) \cdot  c - b$$

The factors s,c and b were distinct for each contributor and round. However,
they were sampled from a multivariate normal distribution ensuring correlation
over the rounds as well as a correlation between rounds.

## Models

### Artificial Humans

We used supervised deep learning to create a model of artificial humans. The
artificial humans are train on pilot 1 and 2.

**Simple Artificial Humans**

Inputs

- previous contribution
- previous punishment
- previous valid

Simple two layer neural network architecture.

**Complex Artificial Humans**

Inputs

- previous contribution
- previous punishment
- previous valid
- previous common good

Graph neural network architecture to learn social influence and RNN to learn
temporal relationships.

### Artificial Human Manager

We used supervised deep learning to create a model of artificial human manager. The
artificial human manager are train on pilot 1 only.

### Optimal Manager

We intend to train an optimal manager using reinforcement learning, that
maximize the common good of a group of artificial humans.
