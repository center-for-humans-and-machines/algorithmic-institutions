"""Replay the rule on the zero group's histories in zero_vs_ah (offline, no feedback) to see the dose
it would deal per group and round, and who gets it, at a few param sets. A sanity check only:
the real players would react."""
import numpy as np
import pandas as pd
import torch as th
from aimanager.manager.rule import load_rule, rule_inputs, run_rule

RULE = 'configs/managers/rule_based/pf-20261104-11.yml'
d = pd.read_parquet('plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet')
d = d[d.run.str.endswith('zero_vs_ah') & (d.episode < 300)].copy()
d['pid'] = d.participant_code.str.split('_').str[0].astype(int)
E = d.episode.nunique()
shape = (E, 8, 24)
def arr(col):
    a = np.zeros(shape, dtype=np.float32)
    a[d.episode.values, d.pid.values, d.round_number.values] = d[col].values
    return th.tensor(a)
C, V, P, G = arr('contribution'), arr('contribution_valid'), arr('punishment'), arr('group_id')
sets = {
    'mid': dict(margin=3, slope=0.5, horizon=20, pre_switch=0.3, cap=8),
    'mild': dict(margin=5, slope=0.3, horizon=16, pre_switch=0, cap=4),
    'harsh': dict(margin=0, slope=1.5, horizon=40, pre_switch=1, cap=30),
}
for name, params in sets.items():
    code, params = load_rule(RULE, params)
    tot, byc = [], []
    for t in range(24):
        inp = rule_inputs(C[..., :t + 1], V[..., :t + 1], P[..., :t], G[..., :t + 1].long(), 0)
        pun = run_rule(code, params, inp).clamp(0, 30).floor() * inp['in_group'][..., -1]
        tot.append(pun.sum(-1).mean().item())
        byc.append((pun.flatten().numpy(), C[..., t].flatten().numpy(), (inp['in_group'][..., -1] * V[..., t]).flatten().numpy()))
    p_all = np.concatenate([b[0] for b in byc]); c_all = np.concatenate([b[1] for b in byc]); m_all = np.concatenate([b[2] for b in byc]) > 0
    s = pd.Series(p_all[m_all]).groupby(pd.cut(c_all[m_all], [-1, 0, 5, 10, 15, 20])).mean().round(2)
    print(name, 'group punishment per round:', np.round(tot, 1).tolist())
    print('   mean by contribution', s.to_dict())
