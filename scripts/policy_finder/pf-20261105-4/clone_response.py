"""Clone (artificial human) response to punishment in the humanlike reference sim,
plus persistence over 1-3 rounds, and the human pilots for comparison."""
import pandas as pd, numpy as np
import warnings; warnings.filterwarnings('ignore')

def prep(d, pid, ep, valid, rnd='round_number'):
    d = d.sort_values([ep, pid, rnd]).copy()
    d['valid'] = valid(d)
    d.loc[d.valid == 0, 'contribution'] = np.nan
    g = d.groupby([ep, pid])
    for k in (1, 2, 3):
        d[f'c{k}'] = g.contribution.shift(-k)
    s = d.groupby([ep, rnd]).contribution.agg(['sum', 'count'])
    d = d.join(s, on=[ep, rnd])
    d['others'] = (d['sum'] - d.contribution) / (d['count'] - 1)
    d['dev'] = d.contribution - d.others
    return d

def table(x, label):
    x = x[x.valid == 1]
    x = x.assign(cbin=pd.cut(x.contribution, [-1, 5, 10, 15, 19, 20]),
                 pbin=pd.cut(x.punishment, [-1, 0, 2, 5, 10, 30]))
    for k in (1, 3):
        print(f'{label}: c(t+{k}) - c(t) by own c (rows) x punishment (cols)')
        print(x.assign(dc=x[f'c{k}'] - x.contribution)
              .pivot_table(index='cbin', columns='pbin', values='dc', aggfunc='mean').round(2))
    print(x.pivot_table(index='cbin', columns='pbin', values='contribution', aggfunc='count'))
    x = x.assign(dbin=pd.cut(x.dev, [-21, -10, -5, -2, 2, 21]))
    print(f'{label}: c(t+1)-c(t) by deviation from others x punishment')
    print(x.assign(dc=x.c1 - x.contribution)
          .pivot_table(index='dbin', columns='pbin', values='dc', aggfunc='mean').round(2))

sim = pd.read_parquet('plots/simulation/29_EXP1_reference/per_round.parquet')
for run in sim.run.unique():
    s = prep(sim[sim.run == run], 'participant_code', 'episode',
             lambda d: d.contribution_valid.astype(int))
    table(s, run)
