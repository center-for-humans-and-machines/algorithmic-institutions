"""Reference sims (ah_vs_zero, zero_vs_ah): composition vs treatment, persistence of punishment."""
import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
d = pd.read_parquet('plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet')
d = d[d.run.str.contains('ah_vs_zero|zero_vs_ah')].copy()
d['mgr'] = np.where((d.agent_group==0) == d.run.str.contains('by ah_vs'), 'ah', 'zero')
d = d.sort_values(['run','episode','participant_code','round_number'])
# players who never switched
key = ['run','episode','participant_code']
nsw = d.groupby(key).agent_group.nunique()
stay = nsw[nsw==1].reset_index()[key]
s = d.merge(stay, on=key)
print('never-switchers, mean contribution by round and manager')
print(s.groupby(['mgr','round_number']).contribution.mean().unstack().round(2).iloc[:, ::2])
# round-0 contribution of never-switchers vs leavers (selection)
r0 = d[d.round_number==0].merge(nsw.rename('k').reset_index(), on=key)
print('round-0 contribution, stayers(k=1) vs switchers(k=2), by initial manager'); print(r0.groupby(['mgr','k']).contribution.mean().unstack().round(2))
# persistence: within ah group, player punished at round t (t in 4..18) vs not, contribution at t+1..t+3, same c_t bin, staying
g = d.groupby(key)
for k in (1,2,3,4):
    d[f'c{k}'] = g.contribution.shift(-k); d[f'grp{k}'] = g.agent_group.shift(-k)
x = d[(d.mgr=='ah') & d.round_number.between(4,18) & (d.grp4==d.agent_group) & d.contribution_valid & (d.contribution<=10)]
x['pb'] = pd.cut(x.punishment,[-1,0,3,8,30])
x['cb'] = pd.cut(x.contribution,[-1,0,5,10])
print('ah group, c<=10, staying 4 rounds: mean c_{t+k} - c_t')
print(x.groupby(['cb','pb'])[['c1','c2','c3','c4']].mean().sub(x.groupby(['cb','pb']).contribution.mean(), axis=0).round(2))
