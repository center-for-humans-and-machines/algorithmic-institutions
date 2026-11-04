"""Check the rule on a hand-made round and measure how much it would punish on the reference sims' zero-managed groups."""
import sys; sys.path.insert(0, 'src')
import torch as th, pandas as pd, numpy as np
from aimanager.manager.rule import load_rule, run_rule
R = 'configs/managers/rule_based/pf-20261104-2.yml'
def run(params, c, valid, ing, t):
    code, prm = load_rule(R, params)
    T = t + 1
    inp = dict(c=c, valid=valid, in_group=ing, p=th.zeros(c.shape[0], 8, t), t=th.tensor(float(t)))
    return run_rule(code, prm, inp)
c = th.tensor([[20, 10, 4, 0, 5, 5, 5, 5.]]).unsqueeze(-1).repeat(1, 1, 6)
valid = th.ones_like(c); valid[0, 3, -1] = 0  # player 3 timed out
ing = th.zeros_like(c); ing[0, :4] = 1
P = dict(k=1.0, tol=1.0, cap=5.0, w_dec=0.5, t_stop=20)
print('round 5:', run(P, c, valid, ing, 5))   # mean of valid members 11.33: p1 0.33, p2 6.33 -> capped 5
print('round 7 (decision):', run(P, c[..., :8].repeat(1, 1, 2)[..., :8], valid.repeat(1, 1, 2)[..., :8], ing.repeat(1, 1, 2)[..., :8], 7))
print('round 21 (stopped):', run(P, c.repeat(1, 1, 4)[..., :22], valid.repeat(1, 1, 4)[..., :22], ing.repeat(1, 1, 4)[..., :22], 21))

# punishment volume on the zero-managed groups of the reference sims, for a few points
d = pd.read_parquet('plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet', columns=['run', 'episode', 'round_number', 'agent_group', 'contribution', 'contribution_valid'])
d = d[d.run.str.contains('by zero_vs_ah') & (d.agent_group == 0) & d.contribution_valid]
d['mean_c'] = d.groupby(['episode', 'round_number']).contribution.transform('mean')
d['short'] = d.mean_c - d.contribution
for k, tol, cap in [(0.3, 2, 5), (0.5, 2, 5), (1.0, 3, 8), (1.5, 0, 20)]:
    p = np.floor(np.minimum(cap, k * np.clip(d.short - tol, 0, None)))
    print(f'k={k} tol={tol} cap={cap}: mean p per member-round {p.mean():.2f}, share punished {(p > 0).mean():.2f}')
