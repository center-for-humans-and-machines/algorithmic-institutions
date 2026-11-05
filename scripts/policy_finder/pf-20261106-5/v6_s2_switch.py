# v6: what drives the artificial humans' switching, pooled over sweep games (v4, v5 parts)
import sys, numpy as np, pandas as pd, pyarrow.parquet as pq
import statsmodels.formula.api as smf
base = '/opt/algorithmic-institutions/plots/simulation/policy_finder/'
files = sys.argv[1:] or ['pf-20261106-5-v5_sweep/per_round_p1of2.parquet']
out = []
for f in files:
    d = pq.read_table(base + f, columns=['episode', 'participant_code', 'round_number', 'punishment', 'contribution',
                                         'contribution_valid', 'payoff', 'group_id', 'run'], filters=[('episode', '<', 12)]).to_pandas()
    d['run'] = d.run.astype('category').cat.codes
    d['pid'] = d.participant_code.str.split('_').str[-1].astype(int)
    d = d.drop(columns='participant_code')
    d['blk'] = d.round_number // 4
    k = ['run', 'episode', 'blk', 'group_id']
    gb = d.groupby(k).agg(gp=('punishment', 'mean'), gc=('contribution', 'mean'), gpay=('payoff', 'mean')).reset_index()
    gn = d[d.round_number % 4 == 3].groupby(['run', 'episode', 'blk', 'group_id']).size().rename('gn').reset_index()
    pl = d.groupby(['run', 'episode', 'pid', 'blk']).agg(op=('punishment', 'mean'), oc=('contribution', 'mean'), opay=('payoff', 'mean'),
                                                          c3=('contribution', 'last'), p3=('punishment', 'last'), g=('group_id', 'last')).reset_index()
    nxt = d[d.round_number % 4 == 0][['run', 'episode', 'pid', 'blk', 'group_id']].copy(); nxt['blk'] -= 1
    pl = pl.merge(nxt.rename(columns={'group_id': 'gnext'}), on=['run', 'episode', 'pid', 'blk'])
    pl['sw'] = (pl.gnext != pl.g).astype(int)
    own = gb.rename(columns={'group_id': 'g'}).merge(gn.rename(columns={'group_id': 'g'}), on=['run', 'episode', 'blk', 'g'])
    oth = own.copy(); oth['g'] = 1 - oth.g
    oth = oth.rename(columns={'gp': 'xp', 'gc': 'xc', 'gpay': 'xpay', 'gn': 'xn'})
    pl = pl.merge(own, on=['run', 'episode', 'blk', 'g']).merge(oth, on=['run', 'episode', 'blk', 'g'], how='left')
    pl['xn'] = pl.xn.fillna(0)
    out.append(pl)
pl = pd.concat(out)
pl = pl[pl.blk < 5]
print(len(pl), pl.sw.mean().round(3))
print(pl.groupby('blk').sw.mean().round(3).to_dict())
pl['dpay'] = pl.gpay - pl.xpay
m = smf.logit('sw ~ op + gp + xp + oc + gc + xc + gn + xn + C(blk)', data=pl.dropna()).fit(disp=0)
print(m.summary2().tables[1].round(4))
m2 = smf.logit('sw ~ opay + gpay + xpay + oc + gn + xn + C(blk)', data=pl.dropna()).fit(disp=0)
print(m2.summary2().tables[1].round(4))
pl.to_parquet('/tmp/v6_switch.parquet')
