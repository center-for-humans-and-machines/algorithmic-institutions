import sys; sys.path.insert(0, '/opt/algorithmic-institutions/scripts/policy_finder/pf-20261106-12')
from load import load
import numpy as np, pandas as pd
d, gs = load()
print('games', d.game.nunique(), 'rows', len(d))
print('mean contribution by round:'); print(d.groupby('round_number').contribution.mean().round(1).values)
print('mean punishment by round:'); print(d.groupby('round_number').punishment.mean().round(2).values)
print('pool per group by round (empty=0):')
full = gs.set_index(['game','round_number','group_id']).pool.unstack(fill_value=0).stack()
print(gs.groupby('round_number').pool.sum().div(d.game.nunique()*2).round(1).values)
print('group size distribution, round 23:', gs[gs.round_number==23].n.value_counts().sort_index().to_dict())
# per game, group: avg pool over episode (empty counts 0), and manager stats
d['dev'] = d.contribution - d.cmean
ep = []
for (gm, gid), x in d.groupby(['game','group_id']):
    pools = gs[(gs.game==gm)&(gs.group_id==gid)].pool.sum()/24
    act = x[x.valid==1]
    pun_rate = (act.punishment>0).mean()
    ep.append(dict(game=gm, g=gid, pool=pools, pmean=act.punishment.mean(), prate=pun_rate,
        members=len(x)/24, cmean=act.contribution.mean(),
        corr=np.corrcoef(act.contribution, act.punishment)[0,1] if act.punishment.std()>0 else 0))
ep = pd.DataFrame(ep)
print(ep.describe().round(2))
print(ep[['pool','pmean','prate','members','cmean','corr']].corr().round(2))
ep['pbin'] = pd.cut(ep.pmean, [-0.01, 0.3, 1, 2, 4, 30])
print(ep.groupby('pbin')[['pool','members','cmean','prate']].agg(['mean','count']).round(2))
