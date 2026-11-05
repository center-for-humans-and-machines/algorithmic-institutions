import sys; sys.path.insert(0, '/opt/algorithmic-institutions/scripts/policy_finder/pf-20261106-5')
from load import load
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
d = load()
x = d[d.next_c.notna() & (d.next_group == d.group_id) & (d.player_no_input == 0)].copy()
x['dc'] = x.next_c - x.contribution
x['rel'] = x.contribution - x.gmean_c
x['pun'] = (x.punishment > 0).astype(int)
x['pb'] = pd.cut(x.punishment, [-1, 0, 2, 5, 10, 30])
x['cb'] = pd.cut(x.contribution, [-1, 0, 5, 10, 15, 19, 20])
print(x.pivot_table(index='cb', columns='pb', values='dc', aggfunc='mean', observed=False).round(2))
print(x.pivot_table(index='cb', columns='pb', values='dc', aggfunc='size', observed=False))
m = smf.ols('dc ~ contribution + gmean_c + punishment + I(punishment**2) + gmean_p', x).fit()
print(m.summary().tables[1])
# How did punishment depend on contribution (human managers)
print(d.groupby(pd.cut(d.contribution,[-1,0,5,10,15,19,20]), observed=False).punishment.agg(['mean', lambda s:(s>0).mean()]).round(2))
# rel contribution
d['rel'] = d.contribution - d.gmean_c
print(d.groupby(pd.cut(d.rel,[-20,-8,-4,-1,1,4,20]), observed=False).punishment.agg(['mean', lambda s:(s>0).mean()]).round(2))
