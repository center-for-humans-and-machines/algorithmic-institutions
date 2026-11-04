"""Human data: switching of unpunished players vs punishment of others; pull of other group."""
import pandas as pd, numpy as np
import warnings; warnings.filterwarnings('ignore')
d = pd.read_csv('experiments/2group_8agent_50ep.csv')
d = d.drop_duplicates(['session','episode','round_number','participant_code'])
d['game'] = d.session.astype(str)+'_'+d.episode.astype(str)
d = d.sort_values(['game','participant_code','round_number'])
g = d.groupby(['game','participant_code'])
d['grp_next'] = g.group_id.shift(-1)
d['switch'] = (d.grp_next != d.group_id).astype(float)
d['phase'] = d.round_number//4
# phase-level aggregates per player and per group
pp = d.groupby(['game','participant_code','phase']).agg(ownp=('punishment','sum'), ownc=('contribution','mean'),
    grp=('group_id','last'), switch=('switch','last')).reset_index()
# group stats in the phase, by group (use last-round group membership rows each round)
gs = d.groupby(['game','phase','group_id']).agg(gp=('punishment','sum'), gc=('contribution','mean'),
    gpay=('payoff','mean'), gn=('contribution','size')).reset_index()
gs['gn'] = gs.gn/4
pp = pp.merge(gs, left_on=['game','phase','grp'], right_on=['game','phase','group_id'])
oth = gs.rename(columns={'group_id':'og','gp':'ogp','gc':'ogc','gpay':'ogpay','gn':'ogn'})
pp['og'] = 1-pp.grp
pp = pp.merge(oth, on=['game','phase','og'], how='left')
pp = pp[pp.phase<5]
pp['others_p'] = (pp.gp-pp.ownp)/4  # per round punishment of others
q = pp[pp.ownp==0]
q['ob']=pd.cut(q.others_p,[-1,0,2,5,10,100])
print('unpunished players: switch rate vs per-round punishment of others in own group')
print(q.groupby('ob').switch.agg(['mean','count']).round(2))
pp['dpay'] = pp.ogpay.fillna(0)-pp.gpay
pp['db']=pd.cut(pp.dpay,[-50,-4,-1,1,4,50])
print('switch rate vs other group mean payoff minus own')
print(pp.groupby('db').switch.agg(['mean','count']).round(2))
pp['dc'] = pp.ogc - pp.gc
pp['dcb']=pd.cut(pp.dc,[-21,-3,-1,1,3,21])
print('switch vs other group mean contribution minus own')
print(pp.groupby('dcb').switch.agg(['mean','count']).round(2))
pp['ogp_b'] = pd.cut(pp.ogp/4/pp.ogn.clip(lower=1),[-1,0,0.5,1.5,3,100])
print('switch vs other group per-member per-round punishment')
print(pp.groupby('ogp_b').switch.agg(['mean','count']).round(2))
# who switches by own contribution
pp['cb']=pd.cut(pp.ownc,[-1,4,8,12,16,20])
print(pp.groupby('cb').switch.agg(['mean','count']).round(2))
