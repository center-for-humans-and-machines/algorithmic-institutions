"""v2: in v1's sweep, how contributions behave around the punishment stop, by stop round,
for strong points (P>=15, T>=12, 0.7<=g<=2). Also the payoff accounting before/after stop."""
import json, pandas as pd, numpy as np
base = 'plots/simulation/policy_finder/pf-20261105-11_sweep/'
s = json.load(open(base + 'sweep.json'))
par = {p['name']: dict(p['params'], pool=p['pool']) for p in s['points']}
dd = []
for k in (1, 2):
    d = pd.read_parquet(base + f'per_round_p{k}of2.parquet', columns=['run', 'round_number', 'contribution', 'punishment', 'contribution_valid', 'payoff'])
    d = d[d.contribution_valid]
    dd.append(d.groupby(['run', 'round_number']).agg(c=('contribution', 'mean'), p=('punishment', 'mean'), pay=('payoff', 'mean')).reset_index())
g = pd.concat(dd)
g['name'] = g.run.str.split().str[-1]
P = pd.DataFrame(par).T.astype(float)
g = g.join(P, on='name')
strong = g[(g['P'] >= 15) & (g['T'] >= 12) & (g['g'] >= 0.7) & (g['g'] <= 2)]
strong = strong.assign(stop=23 - strong.E)  # last punished round
print('n points', strong.name.nunique())
piv = strong.pivot_table(index='round_number', columns='stop', values='c', aggfunc='mean')
print(piv.round(2).to_string())
print(strong.groupby('stop').agg(pool=('pool', 'mean'), n=('name', 'nunique')).assign(n=lambda x: x.n))
# decay per round after stop, by stop
for st, gg in strong.groupby('stop'):
    m = gg.groupby('round_number').c.mean()
    if st < 22:
        print(st, 'c at stop', round(m[st], 2), 'decay/round after', round((m[23] - m[st + 1]) / (23 - st - 1), 3),
              'drift/round in rounds 8..stop', round((m[st] - m[8]) / (st - 8), 3), 'p at stop', round(gg[gg.round_number == st].p.mean(), 2))
