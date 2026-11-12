"""Hand check of v2's timing and targeting on one made-up state; then its intensity on the zero group's
contributions in the reference sims (share of member-rounds targeted, points per member-round)."""
import torch as th, numpy as np, pandas as pd
th.clamp(th.ones(2), min=1.0); th.ones(2).sum(dim=-1, keepdim=True); th.remainder(th.tensor(3.), 4.0); th.tensor(3.) < 20
from aimanager.manager.rule import load_rule, run_rule, dry_run_inputs
dry_run_inputs()
code, params = load_rule('configs/managers/rule_based/pf-20261112-5-v2.yml', {'D': 5.0, 'lo': 1, 'hi': 8, 'w': 1, 'blocks': 4})
cnow = th.tensor([[0., 1, 8, 9, 5, 5, 5, 5]]); grp = th.tensor([[1., 1, 1, 1, 0, 0, 0, 0]]); val = th.ones(1, 8)
for t in [0, 1, 3, 4, 8, 12, 15, 16, 20, 23]:
    T = t + 1
    inp = dict(c=cnow.unsqueeze(-1).repeat(1, 1, T), valid=val.unsqueeze(-1).repeat(1, 1, T),
               in_group=grp.unsqueeze(-1).repeat(1, 1, T), p=th.zeros(1, 8, t), t=th.tensor(float(t)))
    print(t, run_rule(code, params, inp).tolist())
d = pd.read_parquet('plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet',
                    columns=['run', 'round_number', 'contribution', 'contribution_valid', 'group_id'])
z = d[d.run.str.contains('zero_vs_ah') & (d.group_id == 0) & d.contribution_valid]
print('zero group member contributions, share by value 0..20:', np.round(np.bincount(z.contribution, minlength=21) / len(z), 3))
for lo, hi in [(1, 5), (1, 8), (1, 11), (0, 8)]:
    s = ((z.contribution >= lo) & (z.contribution <= hi)).mean()
    print(f'lo={lo} hi={hi}: targeted share {s:.3f}; with w=1, blocks=4 (4 of 24 rounds) and D=5: {5 * s * 4 / 24:.2f} pts/member-round')
