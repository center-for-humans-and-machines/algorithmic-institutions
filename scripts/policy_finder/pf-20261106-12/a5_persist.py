import sys; sys.path.insert(0, '/opt/algorithmic-institutions/scripts/policy_finder/pf-20261106-12')
from load import load
import numpy as np, pandas as pd, statsmodels.api as sm
d, gs = load()
d = d.sort_values(['game','player_id','round_number'])
g = d.groupby(['game','player_id'])
d['loo'] = (d.cmean*d.n - d.contribution)/(d.n-1).replace(0,np.nan)
d['below'] = np.maximum(0, d.loo - d.contribution).fillna(0)
d['above'] = np.maximum(0, d.contribution - d.loo).fillna(0)
for k in range(1,5):
    d['ck'] = g.contribution.shift(-k); d['gk'] = g.group_id.shift(-k)
    x = d[(d.valid==1)&d.ck.notna()&(d.gk==d.group_id)&(d.n>1)].copy()
    x['pb'] = x.punishment*(x.below>0); x['pa'] = x.punishment*(x.below==0)
    x['pb_sm'] = np.minimum(x.pb, 5)
    X = sm.add_constant(x[['contribution','loo','pb','pb_sm','pa']])
    m = sm.OLS(x.ck, X).fit()
    print(k, m.params.round(3).to_dict())
