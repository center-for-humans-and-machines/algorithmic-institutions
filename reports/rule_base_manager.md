# Rule Based Manager

## Rule

We use the same rule, that we also utilized in pilot 2.
$pun = (20-cont) \cdot s + (cont != 20) \cdot  c - b$

However, unlike pilot 3 we generally keep the three factors $s$, $c$ and $b$
constant throughout an episode and also identically across the four agents.

## Importance of factors

We calculate the cumulative common good across all 16 rounds for a grid of
different factors $s$, $c$ and $b$. We also compare results for two different
artificial humans (complex and simple).

![Common Good S C](../notebooks/manager_evaluation/plots/v1_2_1/common_good_s_c.jpg)

> The average common good over the full episode for different punishment slopes $c$ and binary
> punishment of any defectors $s$. We show
> results for complex and simple artificial humans on the left and right
> respectively.

![Common Good S B](../notebooks/manager_evaluation/plots/v1_2_1/common_good_s_b.jpg)

> The average common good over the full episode for different punishment slopes
> $c$ and grace values $b$. We show
> results for complex and simple artificial humans on the left and right
> respectively.

### Conclusion

- Slope is the most important factor.
- Therefore we set in the following c == 0 and b == 0.
- A slope around 1.2 appears optimal for the complex AH.
- For simple AH there does not appear to a clear maximum.

## Evolution over the rounds

**Contribution and punishments over rounds**

![Con Pun Over Rounds](../notebooks/manager_evaluation/plots/v1_2_1/over_rounds.jpg)

> The evolution of contributions and punishments on the y-axis are depicted over
> 16 rounds on the x-axis.
> The left and right column depict complex and simple artificial humans
> respectively. We only vary the slope factor $s$. The other two factors, $c$
> and $b$ are set to zero.

**Common Good and Cumulative Common Good**

![Common Good Over Rounds](../notebooks/manager_evaluation/plots/v1_2_1/over_rounds_cg.jpg)

> We show in the first row the common good (averaged over multiple simulations).
> The second row depicts the cumulative average common good. Thereby, we
> calculate the average common good over the rounds up to the round indicated
> by the y-axis.
> The left and right column depict complex and simple artificial humans
> respectively. We only vary the slope factor $s$. The other two factors, $c$
> and $b$ are set to zero.

**Common Good and Cumulative Common Good**

![Manager Reward Over Rounds](../notebooks/manager_evaluation/plots/v1_2_1/over_rounds_rew.jpg)

> We show the manager reward in the first row and the cumulative manager reward
> in the second row. The reward in each round is corresponding to the common
> good of the
> The left and right column depict complex and simple artificial humans
> respectively. We only vary the slope factor $s$. The other two factors, $c$
> and $b$ are set to zero.

The complex and simple artificial humans lead to qualitatively different
evolution of contributions, punishments and consequently common good. For simple
AH are relative insensitive to very large punishment / contribution ratios $s$.
Also punishments do, surprisingly, not increase with larger $s$ in later rounds.
this surprising observation might be an effect of the punishment being caped at
30 and a punishment ration $s$ of, for instance 1.5 and 2, are therefore
indistinguishable for contribution of 0.

For both AH a punishment policy ($s > 0$) does lead to lower reduced common good
initially and only on the long run to a increase in the common good. For
instance a optimal policy of $s = 1.2$ (AH: complex) leads to a reduction of the
average common good of ~10 points. The reason for this is twofold. On the on
hand the cost punishments is entering directly in the current round, while the benefit from an increase of the
of on increased contribution as a result of the punishment is only entering in
the next round. On the other hand,

Cumulatively this
An optimal policy of $s = 1.2$ (AH: complex) and $s = 2$ (AH: simple) is only in
later rounds leading to an superior common good through higher contributions.

## Applying the rule only partially

### Apply the rule only to 1,2,3,4 agents

![Different Agents](../notebooks/manager_evaluation/plots/v1_agents_1/metric_ah.jpg)

### Apply the rule only to some rounds - Total over episode

![Some Rounds](../notebooks/manager_evaluation/plots/v1_rounds_1/metric_ah.jpg)

### Apply the rule only to some rounds - Per round

![Some Rounds](../notebooks/manager_evaluation/plots/v1_rounds_1/metric_ah.jpg)

## RL Manager

### Sampling

Currently I am using an epsilon greedy sampling approach. Thereby 10% of
all individual punishment are sampled at random. Thereby I am assuming
independence between agents and independence between rounds. That means, there
is no correlation in the sampling of random actions between the four group
members. There is also no correlation in the sampling of random actions in the
16 rounds. Naturally, there is also no correlation between the contribution
and the randomly sampled actions.

Taken together, exploration might be very slow, if multiple actions need to come
together in order to gain a positive outcome.

Next steps

- massively increase sampling?
- sampling smarter?

### Output

We compare two different ways to formulate the problem. A direct policy (A) is
directly determining the punishment given a contribution. Alternatively, the
policy could control a factor (Punishment factor), that multiplied with the
contribution results in a punishment.

```mermaid
flowchart TD
subgraph A
    C((Contributions))
    P((Punishments))

    M[Manager Weights]
    Q[Action Value]

    C ---> Q ----> P
    M ---> Q
end

subgraph B
    Cf((Contributions))
    Ff((PunishmentFactor))
    Pf[Punishments]

    Mf[Manager Weights]
    Qf[Action Value]

    Cf ---> Qf ---> Ff ----> Pf
    Mf ---> Qf
    Cf ----> Pf
end
```

Pro Direct (A)

- Higher generalizability
- Simpler Architecture

Pro Factor (B)

- Random exploration is in terms of slope
- Faster sampling of

### Deepmind Approach

In the Deepmind paper, the game is slightly different. Group member contribute
to the common pool. Then the "mechansim" is deciding how the common pool is
split and payed back to the group member.

**Stochastic Computation Graph**

```mermaid
flowchart LR

AH[Artificial Human Weights]
M[Manager Weights]

subgraph round 1
    P1
    C1
    CG1
end

subgraph round 2
    P2
    C2
    CG2
end

C1((Contribution 1))
C2((Contribution 2))

CG1[Common Good 1]
CG2[Common Good 2]

P1[Payout 1]
P2[Payout 2]

C1 ---> C2

C1 ---> P1
C2 ---> P2

C1 ---> CG1
C2 ---> CG2

P1 ---> C2

CG1 ---> P1
CG2 ---> P2


AH ---> C1
AH ---> C2

M ---> P1
M ---> P2

```

**Surrogate Loss**

> SCG framework generalizes the policy gradient theorem and allowed us to obtain a low-variance estimator of the policy gradient by auto-differentiating through the environment and mechanism policy, while compensating for the non-differentiable operations (the discrete contributions of the players).

$$
L = J + \bot (J) \times \sum_{i} \sum_{t=2}^{10} logp(\bot(c_i^t))
$$

$J$: actual objective; $\bot$: stop gradient; $i$: contributor idx; $t$: round
number; $c$: contribution

### Applying SCG to our setup?

**General Setup**

```mermaid
flowchart LR

AH[Artificial Humans]
M[Manager]

subgraph round 1
    P1
    C1
    CG1
end

subgraph round 2
    P2
    C2
    CG2
end


P1((Punishment 1))
P2((Punishment 2))

C1((Contribution 1))
C2((Contribution 2))

CG1[Common Good 1]
CG2[Common Good 2]

C1 ---> P1
C2 ---> P2

C1 ---> C2

P1 ---> C2

P1 ---> CG1
C1 ---> CG1
P2 ---> CG2
C2 ---> CG2

AH ---> C1
AH ---> C2

M ---> P1
M ---> P2

```

**Reformulating Reward**

```mermaid
flowchart LR

AH[Artificial Humans]
M[Manager]

subgraph round 1
    P1
    C1
    CG1
end

subgraph round 2
    P2
    C2
    CG2
end


P1((Punishment 1))
P2((Punishment 2))

C1((Contribution 1))
C2((Contribution 2))

CG1[Common Good 1]
CG2[Common Good 2]


R1[Reward 1]

C1 ---> P1
C2 ---> P2

C1 ---> C2

P1 ---> C2

P1 ---> CG1
C1 ---> CG1
P2 ---> CG2
C2 ---> CG2

AH ---> C1
AH ---> C2

M ---> P1
M ---> P2

P1 ---> R1
C2 ---> R1

```

####

```mermaid
flowchart LR

AH[Artificial Humans]
M(Manager)

subgraph round 1
    P1
    C1
    CG1
    CD1
end

subgraph round 2
    P2
    C2
    CG2
    CD2
end

P1((Punishment 1))
P2((Punishment 2))

C1((Contribution 1))
C2((Contribution 2))

CD1[Contribution Dist 1]
CD2[Contribution Dist 2]

CG1[Common Good 1]
CG2[Common Good 2]


MR1[Manager Reward 1]

C1 ---> P1
C1 ---> CD2
P1 ---> CD2
C2 ---> P2

P1 ---> CG1
C1 ---> CG1
P2 ---> CG2
C2 ---> CG2

AH ---> CD1
AH ---> CD2

CD1 ---> C1
CD2 ---> C2

M ---> P1
M ---> P2

CD2 ---> MR1
P1 ---> MR1
```

# Main Questions

## DQN or Policy Gradient?

Which RL method to use?

## DQN parameter

- One update step every episode
- learning rate: 0.003
- Optimizer: RMSprop
- epsilon: 0.1
- update frequency of target policy: every 20 episodes
- Size of replay memory: 1000
- Batch Size: 20
- Number of episodes: 10000

## Next steps

- Changing method or further debugging?
- Parallelize environment (sampling in batches)
