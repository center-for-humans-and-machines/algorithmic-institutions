import sys; sys.path.insert(0, '/opt/algorithmic-institutions/scripts/policy_finder/pf-20261106-5')
from load import load
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
d = load()
# block (4 rounds ending at decision) features per player
d['block'] = d.round_number // 4
blk = d.groupby(['game','player_id','block'])
f = blk.agg(p_sum=('punishment','sum'), p_last=('punishment','last'), c_mean=('contribution','mean'),
            gid=('group_id','last'), pay=('payoff','mean'), switch=('switch','last'), n=('gsize','last'),
            gc=('gmean_c','last'), gp=('gmean_p','mean'), gpay=('gpay','mean')).reset_index()
f = f[f.block < 5]
# other group's stats per game/block
gb = d.groupby(['game','block','group_id']).agg(op=('punishment','mean'), oc=('contribution','mean'), opay=('payoff','mean'), on=('gsize','last')).reset_index()
f = f.merge(gb.rename(columns={'group_id':'ogid'}), on=['game','block'])
f = f[f.ogid != f.gid]
# also own group block mean pun
own = gb.rename(columns={'group_id':'gid','op':'mp','oc':'mc','opay':'mpay','on':'mn'})
f = f.merge(own, on=['game','block','gid'])
f['sw'] = f.switch.astype(int)
f['dpay'] = f.opay - f.mpay
f['dpun'] = f.op - f.mp
f['dc'] = f.oc - f.mc
f['punished'] = (f.p_sum > 0).astype(int)
print('n', len(f), 'switch rate', f.sw.mean())
print(f.groupby(pd.cut(f.p_sum, [-1,0,3,8,20,200])).sw.agg(['mean','size']))
print(f.groupby(pd.cut(f.mp, [-1,0,0.5,1.5,3,6,100])).sw.agg(['mean','size']))
print(f.groupby(pd.cut(f.dpay, [-100,-5,-2,0,2,5,100])).sw.agg(['mean','size']))
print(f.groupby('n').sw.agg(['mean','size']))
m = smf.logit('sw ~ p_sum + mp + op + c_mean + mc + oc + n + C(block)', f).fit(disp=0)
print(m.summary().tables[1])
m = smf.logit('sw ~ p_sum + dpay + n + C(block)', f).fit(disp=0)
print(m.summary().tables[1])
