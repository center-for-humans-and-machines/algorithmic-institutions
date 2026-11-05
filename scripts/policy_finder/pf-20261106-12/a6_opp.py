import sys; sys.path.insert(0, '/opt/algorithmic-institutions/scripts/policy_finder/pf-20261106-12')
from load import load
import numpy as np, pandas as pd, statsmodels.api as sm
d, gs = load()
d['blk'] = d.round_number//4
# group block mean punishment
gb = d.groupby(['game','blk','group_id']).punishment.mean().rename('gpblk').reset_index()
ob = gb.copy(); ob['group_id']=1-ob.group_id; ob=ob.rename(columns={'gpblk':'opblk'})
dec = d[d.round_number.isin([3,7,11,15,19])].copy()
dec['sw'] = (dec.grp_next != dec.group_id).astype(float)
dec = dec.merge(gb, on=['game','blk','group_id']).merge(ob, on=['game','blk','group_id'], how='left')
og = gs.copy(); og['group_id'] = 1-og.group_id; og['ocg_pc']=og.pool/og.n
dec = dec.merge(og[['game','round_number','group_id','ocg_pc']], on=['game','round_number','group_id'], how='left')
dec['cg_pc']=dec.pool/dec.n
dec = dec.dropna(subset=['opblk','ocg_pc'])
X = dec[['punishment','gpblk','opblk','cg_pc','ocg_pc','contribution']].copy(); X[['cg_pc','ocg_pc']]/=10
m = sm.Logit(dec.sw, sm.add_constant(X)).fit(disp=0); print(m.summary2().tables[1].round(3))
# human managers: punishment in decision rounds vs others
d['dec'] = d.round_number%4==3
print(d[d.valid==1].groupby('dec').punishment.mean().round(2))
