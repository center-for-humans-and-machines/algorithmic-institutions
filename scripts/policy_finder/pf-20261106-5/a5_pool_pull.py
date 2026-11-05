import sys; sys.path.insert(0, '/opt/algorithmic-institutions/scripts/policy_finder/pf-20261106-5')
from load import load
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
d = load()
x = d[d.round_number.isin([3,7,11,15,19])].copy()
G = d.groupby(['game','round_number','group_id']).agg(pool=('common_good','first'), n=('player_id','size'), mp=('punishment','mean')).reset_index()
o = G.rename(columns={'group_id':'og','pool':'opool','n':'on','mp':'omp'})
x = x.merge(o, on=['game','round_number']); x = x[x.og != x.group_id]
x['sw'] = x.switch.astype(int)
x['pc'] = x.common_good / x.gsize; x['opc'] = x.opool / x.on
x = x[x.on > 0]
for f in ['sw ~ common_good + opool + punishment', 'sw ~ pc + opc + punishment', 'sw ~ pc + opc + common_good + opool + punishment + gmean_p + omp + gsize']:
    m = smf.logit(f + ' + C(round_number)', x).fit(disp=0); print(f, 'llf %.1f' % m.llf); print(m.params.round(3).to_string())
# who switches: contribution class
x['cb'] = pd.cut(x.contribution, [-1, 4, 10, 15, 20])
print(x.groupby('cb', observed=True).sw.mean().round(3))
