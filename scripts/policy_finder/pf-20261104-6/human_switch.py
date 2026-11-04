"""Human data: what drives leaving a group at a switch decision."""
import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
d = pd.read_csv('experiments/2group_8agent_50ep.csv')
d = d.drop_duplicates(['session','episode','round_number','participant_code']).copy()
d['game'] = d.session.astype(str)+'_'+d.episode.astype(str)
d = d.sort_values(['game','player_id','round_number'])
d['valid'] = 1 - d.player_no_input
d['block'] = d.round_number // 4
# per group-round pool and size
gr = d.groupby(['game','round_number','group_id']).agg(n=('player_id','size'), pool=('common_good','first'),
        gc=('contribution','mean'), gp=('punishment','mean')).reset_index()
print('common_good check: total? compare 1.6*sum c - sum p')
chk = d.groupby(['game','round_number','group_id']).apply(lambda x: 1.6*(x.contribution*x.valid).sum()-(x.punishment*x.valid).sum()).rename('calc').reset_index()
m = gr.merge(chk); print(m[['pool','calc','n']].head(8))
# block summary per player: last round of block in which decision is made (rounds 3,7,11,15,19)
pl = d.groupby(['game','player_id','block']).agg(grp=('group_id','last'), c=('contribution','mean'), p=('punishment','sum'),
        p_last=('punishment','last'), payoff=('payoff','sum')).reset_index()
pl['grp_next'] = pl.groupby(['game','player_id']).grp.shift(-1)
pl = pl[pl.grp_next.notna()]
pl['left'] = (pl.grp_next != pl.grp).astype(int)
# group-level block stats for own and other group
gb = d.groupby(['game','block','group_id']).agg(gpay=('payoff','mean'), gc=('contribution','mean'), gpun=('punishment','mean'),
       n=('player_id', lambda x: x.nunique())).reset_index()
pl = pl.merge(gb.rename(columns={'group_id':'grp'}), on=['game','block','grp'])
gb2 = gb.rename(columns={'group_id':'grp','gpay':'opay','gc':'oc','gpun':'opun','n':'on'}); gb2['grp']=1-gb2.grp
pl = pl.merge(gb2, on=['game','block','grp'], how='left')
print('\noverall leave rate', pl.left.mean().round(3), 'by block', pl.groupby('block').left.mean().round(3).to_dict())
print('\nleave rate by punishment received in the block (sum)')
print(pl.groupby(pd.cut(pl.p,[-1,0,3,8,15,30,200])).left.agg(['mean','count']).round(3))
print('\nleave rate by punishment in the last round before the decision')
print(pl.groupby(pd.cut(pl.p_last,[-1,0,3,8,30])).left.agg(['mean','count']).round(3))
print('\nleave rate by own mean contribution')
print(pl.groupby(pd.cut(pl.c,[-1,3,7,11,15,20])).left.agg(['mean','count']).round(3))
pl['dpay'] = pl.gpay - pl.opay
print('\nleave rate by own-group minus other-group mean payoff (block)')
print(pl.groupby(pd.cut(pl.dpay,[-50,-5,-2,0,2,5,50])).left.agg(['mean','count']).round(3))
print('\nleave rate by own-group mean punishment minus other group')
pl['dpun'] = pl.gpun - pl.opun
print(pl.groupby(pd.cut(pl.dpun,[-30,-2,-0.5,0.5,2,30])).left.agg(['mean','count']).round(3))
import statsmodels.formula.api as smf
try:
    r = smf.logit('left ~ p + c + dpay + dpun + n', data=pl.dropna()).fit(disp=0); print(r.summary().tables[1])
except Exception as e: print(e)
# by contribution x punishment received
pl['cb'] = pd.cut(pl.c,[-1,5,10,15,20]); pl['pb']=pd.cut(pl.p,[-1,0,5,15,200])
print(pl.pivot_table(index='cb', columns='pb', values='left', aggfunc='mean').round(3))
print(pl.pivot_table(index='cb', columns='pb', values='left', aggfunc='count'))
pl.to_pickle('/tmp/pl.pkl')
