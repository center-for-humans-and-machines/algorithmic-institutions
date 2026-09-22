# Behaviour versus evaluated punishment

Mean punishment per member per round over the RL manager's own group. `behaviour` is the rollout that fills the replay buffer, `evaluated` the fully deterministic rollout with every exploration mechanism disabled. `gap` is the punishment the behaviour policy adds that the evaluated policy never asked for.

| run | update_step | behaviour | evaluated | gap | ratio |
|---|---|---|---|---|---|
| rl_anneal_local_s42 | 0 | 4.1827 | 4.2149 | -0.0323 | 0.99 |
| rl_anneal_local_s42 | 3980 | 2.0027 | 1.8701 | 0.1327 | 1.07 |
| rl_anneal_local_s43 | 0 | 2.6225 | 0.0000 | 2.6225 | n/a |
| rl_anneal_local_s43 | 3980 | 1.3024 | 1.2087 | 0.0937 | 1.08 |
| rl_anneal_local_s44 | 0 | 4.4570 | 1.8351 | 2.6220 | 2.43 |
| rl_anneal_local_s44 | 3980 | 1.7616 | 1.6626 | 0.0991 | 1.06 |
| rl_anneal_local_s45 | 0 | 14.0183 | 4.9125 | 9.1057 | 2.85 |
| rl_anneal_local_s45 | 3980 | 3.7022 | 3.0742 | 0.6280 | 1.20 |
| rl_anneal_local_s46 | 0 | 8.0904 | 6.9709 | 1.1195 | 1.16 |
| rl_anneal_local_s46 | 3980 | 2.2238 | 1.8134 | 0.4103 | 1.23 |
| rl_new_clones_s42 | 0 | 4.9170 | 4.2149 | 0.7021 | 1.17 |
| rl_new_clones_s42 | 3980 | 2.7879 | 1.6201 | 1.1678 | 1.72 |
| rl_new_clones_s43 | 0 | 3.5283 | 0.0000 | 3.5283 | n/a |
| rl_new_clones_s43 | 3980 | 1.8958 | 0.8178 | 1.0780 | 2.32 |
| rl_new_clones_s44 | 0 | 5.3929 | 2.0307 | 3.3622 | 2.66 |
| rl_new_clones_s44 | 3980 | 2.2213 | 0.9677 | 1.2536 | 2.30 |
| rl_new_clones_s45 | 0 | 13.7117 | 4.9125 | 8.7992 | 2.79 |
| rl_new_clones_s45 | 3980 | 2.9971 | 1.7579 | 1.2393 | 1.70 |
| rl_new_clones_s46 | 0 | 8.4252 | 6.8024 | 1.6228 | 1.24 |
| rl_new_clones_s46 | 3980 | 3.2625 | 2.0410 | 1.2215 | 1.60 |

The arm passes if its `gap` at the last step is far below its own gap at the first step and below the control's gap at the last step. It says nothing about whether the arm improves the policy.

```json
{
  "rl_anneal_local_s42": {
    "first": {
      "eps-greedy": 4.182653258244197,
      "gap": -0.03225930531819632,
      "greedy": 4.214912563562393,
      "ratio": 0.9923463880135793
    },
    "first_step": 0,
    "last": {
      "eps-greedy": 2.002744287252426,
      "gap": 0.13267708321412397,
      "greedy": 1.8700672040383022,
      "ratio": 1.0709477621593575
    },
    "last_step": 3980,
    "n_eval_points": 200
  },
  "rl_anneal_local_s43": {
    "first": {
      "eps-greedy": 2.622548601900538,
      "gap": 2.622548601900538,
      "greedy": 0.0,
      "ratio": NaN
    },
    "first_step": 0,
    "last": {
      "eps-greedy": 1.302410123248895,
      "gap": 0.09367243200540543,
      "greedy": 1.2087376912434895,
      "ratio": 1.0774960793263921
    },
    "last_step": 3980,
    "n_eval_points": 200
  },
  "rl_anneal_local_s44": {
    "first": {
      "eps-greedy": 4.457041382789612,
      "gap": 2.6219831506411237,
      "greedy": 1.8350582321484883,
      "ratio": 2.4288283089367155
    },
    "first_step": 0,
    "last": {
      "eps-greedy": 1.761639470855395,
      "gap": 0.09907661378383636,
      "greedy": 1.6625628570715587,
      "ratio": 1.0595927025329737
    },
    "last_step": 3980,
    "n_eval_points": 200
  },
  "rl_anneal_local_s45": {
    "first": {
      "eps-greedy": 14.01825483640035,
      "gap": 9.105719526608786,
      "greedy": 4.912535309791565,
      "ratio": 2.8535682600508645
    },
    "first_step": 0,
    "last": {
      "eps-greedy": 3.702216992775599,
      "gap": 0.6279823482036591,
      "greedy": 3.07423464457194,
      "ratio": 1.2042727445390233
    },
    "last_step": 3980,
    "n_eval_points": 200
  },
  "rl_anneal_local_s46": {
    "first": {
      "eps-greedy": 8.090405821800232,
      "gap": 1.1195327440897627,
      "greedy": 6.970873077710469,
      "ratio": 1.1606015102569425
    },
    "first_step": 0,
    "last": {
      "eps-greedy": 2.2237589756647744,
      "gap": 0.4103349745273588,
      "greedy": 1.8134240011374156,
      "ratio": 1.226276355816394
    },
    "last_step": 3980,
    "n_eval_points": 200
  },
  "rl_new_clones_s42": {
    "first": {
      "eps-greedy": 4.917026937007904,
      "gap": 0.7021143734455109,
      "greedy": 4.214912563562393,
      "ratio": 1.1665786330931838
    },
    "first_step": 0,
    "last": {
      "eps-greedy": 2.7879055440425873,
      "gap": 1.167780930797259,
      "greedy": 1.6201246132453282,
      "ratio": 1.720796981448258
    },
    "last_step": 3980,
    "n_eval_points": 200
  },
  "rl_new_clones_s43": {
    "first": {
      "eps-greedy": 3.528305026392142,
      "gap": 3.528305026392142,
      "greedy": 0.0,
      "ratio": NaN
    },
    "first_step": 0,
    "last": {
      "eps-greedy": 1.8958127051591873,
      "gap": 1.077975805460786,
      "greedy": 0.8178368996984015,
      "ratio": 2.318081644223093
    },
    "last_step": 3980,
    "n_eval_points": 200
  },
  "rl_new_clones_s44": {
    "first": {
      "eps-greedy": 5.392872770627339,
      "gap": 3.3621612687905627,
      "greedy": 2.0307115018367767,
      "ratio": 2.655656781255974
    },
    "first_step": 0,
    "last": {
      "eps-greedy": 2.2212808430194855,
      "gap": 1.2535555958747864,
      "greedy": 0.9677252471446991,
      "ratio": 2.2953631204450207
    },
    "last_step": 3980,
    "n_eval_points": 200
  },
  "rl_new_clones_s45": {
    "first": {
      "eps-greedy": 13.711716731389364,
      "gap": 8.7991814215978,
      "greedy": 4.912535309791565,
      "ratio": 2.791169094308483
    },
    "first_step": 0,
    "last": {
      "eps-greedy": 2.9971150954564414,
      "gap": 1.2392623325188956,
      "greedy": 1.7578527629375458,
      "ratio": 1.7049864235773453
    },
    "last_step": 3980,
    "n_eval_points": 200
  },
  "rl_new_clones_s46": {
    "first": {
      "eps-greedy": 8.425249020258585,
      "gap": 1.6228010455767308,
      "greedy": 6.802447974681854,
      "ratio": 1.2385613313937367
    },
    "first_step": 0,
    "last": {
      "eps-greedy": 3.26251874367396,
      "gap": 1.221533139546712,
      "greedy": 2.0409856041272483,
      "ratio": 1.5985015950512083
    },
    "last_step": 3980,
    "n_eval_points": 200
  }
}
```
