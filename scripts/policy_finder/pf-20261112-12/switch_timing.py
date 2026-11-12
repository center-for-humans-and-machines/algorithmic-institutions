"""Which round of the block's punishment matters for switching (sims)?"""
import pandas as pd, numpy as np
import statsmodels.formula.api as smf
d=pd.read_parquet('plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet')
d=d[d.run.str.contains('ah_vs_ah|ah_vs_zero')]  # ah managed somewhere; keep all
d=d.sort_values(['run','episode','participant_code','round_number'])
d['blk']=d.round_number//4; d['pos']=d.round_number%4
w=d.pivot_table(index=['run','episode','participant_code','blk'],columns='pos',values=['punishment','contribution','common_good'])
w.columns=[f'{a[:1]}{b}' for a,b in w.columns]
w=w.reset_index()
g=d.groupby(['run','episode','participant_code','blk']).agent_group.first().rename('g').reset_index()
gn=g.copy(); gn['blk']-=1; gn=gn.rename(columns={'g':'gn'})
w=w.merge(g).merge(gn)
w['sw']=(w.g!=w.gn).astype(int)
m=smf.logit('sw ~ p0+p1+p2+p3+c0+c1+c2+c3+C(blk)',data=w).fit(disp=0)
print(m.params.round(3))
# previous blocks' punishment (memory)?
w=w.sort_values(['run','episode','participant_code','blk'])
w['pprev']=w.groupby(['run','episode','participant_code'])[['p0','p1','p2','p3']].shift(1).mean(axis=1)
m=smf.logit('sw ~ p0+p1+p2+p3+pprev+c3+C(blk)',data=w.dropna(subset=['pprev'])).fit(disp=0)
print(m.params.round(3))
