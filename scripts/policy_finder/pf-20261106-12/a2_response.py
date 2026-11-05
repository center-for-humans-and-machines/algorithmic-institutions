import sys; sys.path.insert(0, '/opt/algorithmic-institutions/scripts/policy_finder/pf-20261106-12')
from load import load
import numpy as np, pandas as pd
d, gs = load()
# within-group (no switch) contribution response
x = d[(d.valid==1) & d.c_next.notna() & (d.grp_next==d.group_id)].copy()
x = x[~x.round_number.isin([3,7,11,15,19])]
x['dc'] = x.c_next - x.contribution
x['cb'] = pd.cut(x.contribution, [-1,2,6,10,14,17,20])
x['pb'] = pd.cut(x.punishment, [-1,0,2,5,10,30])
print('mean dc by own contribution (rows) and punishment (cols):')
print(x.pivot_table(index='cb', columns='pb', values='dc', aggfunc='mean', observed=True).round(2))
print(x.pivot_table(index='cb', columns='pb', values='dc', aggfunc='size', observed=True))
# relative to group mean
x['devb'] = pd.cut(x.contribution - x.cmean, [-21,-6,-2,2,6,21])
print('dc by deviation from group mean x punishment')
print(x.pivot_table(index='devb', columns='pb', values='dc', aggfunc='mean', observed=True).round(2))
# regression
import numpy.linalg as la
X = np.column_stack([np.ones(len(x)), x.contribution, x.cmean, x.punishment, np.minimum(x.punishment,3), x.pmean])
b = la.lstsq(X, x.c_next, rcond=None)[0]
print('c_next ~ 1, c, cmean, p, min(p,3), group pmean:', b.round(3))
