import pandas as pd, numpy as np, statsmodels.formula.api as smf
d=pd.read_parquet('plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet')
d['c']=d.contribution.where(d.contribution_valid.astype(bool),0)
d=d.sort_values(['run','episode','participant_code','round_number'])
g=d.groupby(['run','episode','participant_code'])
d['ng']=g.group_id.shift(-1); d['nc']=g.c.shift(-1); d['n2c']=g.c.shift(-2)
k=['run','episode','round_number','group_id']
agg=d.groupby(k).agg(sc=('c','sum'),sp=('punishment','sum'),n=('c','size'),nmin=('c',lambda s:(s<=2).sum())).reset_index()
d=d.merge(agg,on=k)
d['oc']=(d.sc-d.c)/(d.n-1).clip(lower=1); d['op']=(d.sp-d.punishment)/(d.n-1).clip(lower=1)
d['lowc']=(d.c<10).astype(float)
y=d[d.run.str.contains('ah_vs_ah')&(d.round_number<22)&(d.ng==d.group_id)&(d.n>1)]
for tgt in ['nc','n2c']:
    m=smf.ols(f'{tgt} ~ c + punishment + punishment:lowc + oc + op + C(round_number)',data=y).fit()
    print(tgt, m.params[['c','punishment','punishment:lowc','oc','op']].round(3).to_dict())
z=d[d.run.str.contains('zero_vs_zero')&(d.round_number<22)&(d.ng==d.group_id)&(d.n>1)]
m=smf.ols('nc ~ c + oc + C(round_number)',data=z).fit(); print('zero_vs_zero', m.params[['c','oc']].round(3).to_dict())
# per member pool by run/group across rounds
a=agg.copy(); a['ppm']=(1.6*a.sc-a.sp)/a.n
print(a.groupby(['run','group_id']).agg(ppm=('ppm','mean'),n=('n','mean')).round(2))
