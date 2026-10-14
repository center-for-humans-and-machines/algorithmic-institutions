# Behaviour versus evaluated punishment

Mean punishment per member per round over the RL manager's own group. `behaviour` is the rollout that fills the replay buffer, `evaluated` the fully deterministic rollout with every exploration mechanism disabled. `gap` is the punishment the behaviour policy adds that the evaluated policy never asked for.

| run | update_step | behaviour | evaluated | gap | ratio |
|---|---|---|---|---|---|
| rl_anneal_control_pilot | 0 | 4.9170 | 4.2149 | 0.7021 | 1.17 |
| rl_anneal_control_pilot | 280 | 7.2078 | 6.5768 | 0.6311 | 1.10 |
| rl_anneal_local_pilot | 0 | 4.1827 | 4.2149 | -0.0323 | 0.99 |
| rl_anneal_local_pilot | 280 | 5.5233 | 5.6153 | -0.0921 | 0.98 |

The arm passes if its `gap` at the last step is far below its own gap at the first step and below the control's gap at the last step. It says nothing about whether the arm improves the policy.

```json
{
  "rl_anneal_control_pilot": {
    "first": {
      "eps-greedy": 4.917026937007904,
      "gap": 0.7021143734455109,
      "greedy": 4.214912563562393,
      "ratio": 1.1665786330931838
    },
    "first_step": 0,
    "last": {
      "eps-greedy": 7.2078257004419966,
      "gap": 0.6310597062110901,
      "greedy": 6.5767659942309065,
      "ratio": 1.09595289033617
    },
    "last_step": 280,
    "n_eval_points": 15
  },
  "rl_anneal_local_pilot": {
    "first": {
      "eps-greedy": 4.182653258244197,
      "gap": -0.03225930531819632,
      "greedy": 4.214912563562393,
      "ratio": 0.9923463880135793
    },
    "first_step": 0,
    "last": {
      "eps-greedy": 5.523268183072408,
      "gap": -0.09207810958226581,
      "greedy": 5.6153462926546736,
      "ratio": 0.983602416523677
    },
    "last_step": 280,
    "n_eval_points": 15
  }
}
```
