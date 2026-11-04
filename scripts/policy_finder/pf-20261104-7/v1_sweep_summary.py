"""Per design point of v1's sweep: punishment spent, contributions early/late, members, pool. Group 0 is the rule's."""
import json, re
import numpy as pd_np
import pandas as pd
base = 'plots/simulation/policy_finder/pf-20261104-7_sweep/'
sw = json.load(open(base + 'sweep.json'))
par = {p['name']: dict(p['params'], pool=p['pool'], members=p['members']) for p in sw['points']}
out = []
for part in ('p1of2', 'p2of2'):
    d = pd.read_parquet(base + f'per_round_{part}.parquet', columns=['run', 'episode', 'round_number', 'agent_group', 'contribution', 'punishment'])
    d = d[d.agent_group == 0]
    d['blk'] = d.round_number // 4
    neps = d.groupby('run').episode.nunique()
    g = d.groupby(['run', 'blk']).agg(c=('contribution', 'mean'), p=('punishment', 'mean'), n=('contribution', 'size')).reset_index()
    g['n'] = g['n'] / g.run.map(neps) / 4
    w = g.pivot(index='run', columns='blk', values=['c', 'p', 'n'])
    w.columns = [f'{a}{b}' for a, b in w.columns]
    out.append(w)
W = pd.concat(out)
W.index = [re.search(r'by (.*)_vs_ah', r).group(1) for r in W.index]
P = pd.DataFrame(par).T
W = W.join(P)
W.to_csv('/tmp/v1_sweep_summary.csv')
pd.set_option('display.width', 250)
W['ptot'] = W[[f'p{b}' for b in range(6)]].mean(axis=1)
print(W.sort_values('pool', ascending=False).head(20).round(2).to_string())
print(W[['pool', 'members', 'ptot', 'c0', 'c2', 'c5', 'n1', 'n5']].astype(float).corr().round(2))
