import pandas as pd, numpy as np
d=pd.read_parquet('plots/simulation/29_EXP1_reference/per_round.parquet')
print(d.columns.tolist()); print(d.head())
d=d[d.contribution_valid]
for run,g in d.groupby('run'):
    pr=g.groupby(['episode','round_number']).payoff.mean().groupby('episode').sum()
    print(run, 'score', pr.mean(), pr.std()/np.sqrt(len(pr)), 'members', g.groupby(['episode','round_number']).size().mean())
    print(' valid rate', len(g)/ (1000*24*4))
    print(' contrib dist', g.contribution.describe().to_dict())
    print(' punish frac>0', (g.punishment>0).mean(), 'mean', g.punishment.mean())
