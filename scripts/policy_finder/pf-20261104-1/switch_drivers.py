"""What drives artificial humans to switch groups (reference sims).
Per player and switch decision: switched? vs. features of the preceding block."""
import pandas as pd, numpy as np
d = pd.read_parquet('plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet')
d = d.sort_values(['run','episode','participant_code','round_number'])
d['blk'] = d.round_number // 4
g = d.groupby(['run','episode','participant_code','blk'])
f = g.agg(grp=('agent_group','last'), c=('contribution','mean'), p=('punishment','mean'),
          pay=('payoff','mean'), p_any=('punishment', lambda x: (x>0).mean())).reset_index()
# group-level aggregates per block
gb = d.groupby(['run','episode','blk','agent_group'])
G = gb.agg(gp=('punishment','mean'), gc=('contribution','mean'), gpay=('payoff','mean'), n=('punishment','size')).reset_index()
G['n'] = G.n/4
f = f.merge(G.rename(columns={'agent_group':'grp'}), on=['run','episode','blk','grp'])
O = G.copy(); O['grp'] = 1-O.agent_group
f = f.merge(O[['run','episode','blk','grp','gpay','gc','gp','n']].rename(columns={'gpay':'opay','gc':'oc','gp':'op','n':'on'}), on=['run','episode','blk','grp'], how='left')
nxt = d[d.round_number % 4 == 0][['run','episode','participant_code','blk','agent_group']].copy()
nxt['blk'] -= 1
f = f.merge(nxt.rename(columns={'agent_group':'next'}), on=['run','episode','participant_code','blk'])
f['sw'] = (f.next != f.grp).astype(float)
print('overall switch rate', f.sw.mean().round(3))
f['mgr'] = [ (r.split('by ')[1].split('_vs_')[g]) for r,g in zip(f.run, f.grp)]
print(f.groupby('mgr').sw.mean().round(3))
for col, bins in [('p',[-.1,0,.5,1,2,4,8,30]),('gp',[-.1,0,.5,1,2,4,30]),('c',[-.1,2,5,10,15,20]),('n',[0,1,2,3,4,5,6,8])]:
    print(col); print(f.groupby(pd.cut(f[col],bins)).sw.agg(['mean','size']).round(3))
f['dpay'] = f.gpay - f.opay
print('dpay'); print(f.groupby(pd.cut(f.dpay,[-30,-6,-3,-1,1,3,6,30])).sw.agg(['mean','size']).round(3))
import statsmodels.formula.api as smf
try:
    m = smf.logit('sw ~ p + gp + c + gc + oc + n + dpay', data=f.dropna()).fit(disp=0); print(m.summary2().tables[1].round(3))
except Exception as e: print(e)
f.to_parquet('/tmp/switch_feat.parquet')
