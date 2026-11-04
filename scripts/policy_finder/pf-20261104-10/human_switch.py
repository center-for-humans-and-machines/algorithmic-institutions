"""Human data: who leaves a group at a switch point, vs punishment received in the block,
own contribution, and own/other group's pool and punishment level."""
import pandas as pd, numpy as np
d = pd.read_csv('experiments/2group_8agent_50ep.csv')
d = d.drop_duplicates(['session','episode','round_number','participant_code'])
d['game'] = d.session.astype(str)+'_'+d.episode.astype(str)
d = d.sort_values(['game','player_id','round_number'])
d['block'] = d.round_number//4
g = d.groupby(['game','player_id'])
d['grp_next'] = g.group_id.shift(-1)
# block-level summaries per player
d['pun_pos'] = (d.punishment>0).astype(float)
pb = d.groupby(['game','player_id','block']).agg(group=('group_id','last'), p=('punishment','mean'),
      ppos=('pun_pos','mean'), c=('contribution','mean'), grp_next=('grp_next','last')).reset_index()
pb = pb[pb.block<5]
pb['left'] = (pb.grp_next!=pb.group).astype(float)
# group-level per block: mean punishment, mean payoff, pool, size
gb = d.groupby(['game','block','group_id']).agg(gp=('punishment','mean'), gpay=('payoff','mean'),
      gc=('contribution','mean'), n=('player_id','nunique')).reset_index()
pb = pb.merge(gb, left_on=['game','block','group'], right_on=['game','block','group_id'])
ob = gb.rename(columns={'group_id':'og','gp':'op','gpay':'opay','gc':'oc','n':'on'})
pb['og'] = 1-pb.group
pb = pb.merge(ob, on=['game','block','og'], how='left')
print('leave rate', pb.left.mean().round(3), 'n', len(pb))
pb['pbin'] = pd.cut(pb.p,[-1,0,1,3,6,30])
print(pb.groupby('pbin', observed=True).left.agg(['mean','count']).round(3))
pb['dpay'] = pd.cut(pb.opay-pb.gpay, [-30,-3,-1,1,3,30])
print('by other-own mean payoff diff'); print(pb.groupby('dpay', observed=True).left.agg(['mean','count']).round(3))
pb['dpun'] = pd.cut(pb.gp-pb.op, [-30,-2,-0.5,0.5,2,30])
print('by own-other group mean punishment'); print(pb.groupby('dpun', observed=True).left.agg(['mean','count']).round(3))
pb['cbin']=pd.cut(pb.c,[-1,5,10,15,20])
print(pb.pivot_table(index='cbin',columns=pb.p>0,values='left',aggfunc=['mean','count'],observed=True).round(3))
print(pb.groupby('block').left.mean().round(3))
# logistic-ish linear regression
X = np.c_[np.ones(len(pb)), pb.p, pb.c, (pb.opay-pb.gpay).fillna(0), pb.n]
print('left ~ 1,p,c,opay-gpay,n', np.linalg.lstsq(X, pb.left, rcond=None)[0].round(4))
