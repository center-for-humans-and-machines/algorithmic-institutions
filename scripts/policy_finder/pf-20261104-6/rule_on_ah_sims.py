"""Replay the rule on the ah_vs_ah sim histories (open loop): how much would it punish, and whom,
compared with ah's own punishments in those same states."""
import pandas as pd, numpy as np, torch as th, sys
from aimanager.manager.rule import load_rule, rule_inputs, run_rule
d = pd.read_parquet('plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet')
d = d[d.run.str.contains('managed by ah_vs_ah')]
d = d[d.episode < 300]
d['pid'] = d.participant_code.str.split('_').str[0].astype(int)
E = d.episode.nunique()
def arr(col):
    return th.tensor(d.pivot_table(index=['episode','pid'], columns='round_number', values=col).values.reshape(E, 8, 24)).float()
C, V, P, G = arr('contribution'), arr('contribution_valid'), arr('punishment'), arr('group_id')
settings = [dict(margin=2, floor=6, rate=0.5, cap=4, decision_mult=0.0), dict(margin=0, floor=8, rate=1.0, cap=8, decision_mult=0.5), dict(margin=4, floor=3, rate=0.3, cap=2, decision_mult=0.0)]
for s in settings:
    code, params = load_rule('configs/managers/rule_based/pf-20261104-6.yml', s)
    tot, ahtot, cnt = [], [], []
    rows = []
    for t in range(24):
        inp = rule_inputs(C[..., :t+1], V[..., :t+1], P[..., :t], G[..., :t+1], 0)
        pun = th.floor(run_rule(code, params, inp).clamp(0, 30))
        ig = (G[..., t] == 0) & (V[..., t] == 1)
        rows.append(pd.DataFrame({'t': t, 'rule': pun[ig].numpy(), 'ah': P[..., t][ig].numpy(), 'c': C[..., t][ig].numpy()}))
    r = pd.concat(rows)
    print(s); print('  mean per member-round rule %.2f ah %.2f; share punished rule %.2f ah %.2f' % (r.rule.mean(), r.ah.mean(), (r.rule>0).mean(), (r.ah>0).mean()))
    print('  by c bin rule', r.groupby(pd.cut(r.c,[-1,0,4,9,14,20])).rule.mean().round(2).tolist(), 'ah', r.groupby(pd.cut(r.c,[-1,0,4,9,14,20])).ah.mean().round(2).tolist())
