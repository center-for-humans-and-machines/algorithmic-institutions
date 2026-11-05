# switching: what predicts leaving a group at a decision
import sys; sys.path.insert(0, '/opt/algorithmic-institutions/scripts/policy_finder/pf-20261106-7')
from load import load
import numpy as np, pandas as pd
df = load()
df['pay_rel']=0
# per group-round stats
g = df.groupby(['game','round_number','group_id']).agg(gc=('contribution','mean'), gp=('punishment','mean'), n=('player_id','size'), pool=('common_good','first')).reset_index()
df = df.merge(g, on=['game','round_number','group_id'])
rows=[]
for (game,pid), d in df.groupby(['game','player_id']):
    d=d.set_index('round_number')
    for r in [3,7,11,15,19]:
        blk=d.loc[r-3:r]
        cur=d.loc[r,'group_id']; nxt=d.loc[r+1,'group_id']
        # other group stats in block
        o = g[(g.game==game)&(g.round_number.between(r-3,r))]
        own = o[o.group_id==cur]; oth=o[o.group_id!=cur]
        rows.append(dict(game=game,pid=pid,r=r,switch=int(cur!=nxt),
            my_c=blk.contribution.mean(), my_p=blk.punishment.mean(), my_pay=blk.payoff.mean(),
            own_c=own.gc.mean(), own_p=own.gp.mean(), own_n=own.n.mean(),
            oth_c=oth.gc.mean() if len(oth) else np.nan, oth_p=oth.gp.mean() if len(oth) else np.nan, oth_n=oth.n.mean() if len(oth) else 0,
            last_p=d.loc[r,'punishment']))
s=pd.DataFrame(rows)
s.to_csv('/tmp/switch.csv',index=False)
print('switch rate', s.switch.mean(), s.groupby('r').switch.mean().round(3).to_dict())
s['dp']=s.own_p-s.oth_p; s['dc']=s.own_c-s.oth_c; s['dn']=s.own_n-s.oth_n
for col in ['my_p','last_p','dp','dc','dn','my_c']:
    q=pd.qcut(s[col].rank(method='first'),5,labels=False)
    print(col, s.groupby(q).agg(x=(col,'mean'),sw=('switch','mean')).round(2).values.tolist())
import statsmodels.formula.api as smf
m=smf.logit('switch ~ my_p + dp + dc + dn + my_c', data=s.dropna()).fit(disp=0); print(m.summary2().tables[1].round(3))
