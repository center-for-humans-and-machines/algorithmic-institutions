"""Reference sims: leaving vs punishment in the decision round (r%4==3) vs earlier block rounds,
and vs own contribution; logistic fit."""
import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
d = pd.read_parquet('plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet')
pair = d.run.str.split('managed by ').str[-1].str.split('_vs_')
d['mgr'] = np.where(d.agent_group==0, pair.str[0], pair.str[1])
d = d.sort_values(['run','episode','participant_code','round_number'])
d['gn'] = d.groupby(['run','episode','participant_code']).agent_group.shift(-1)
d['block']=d.round_number//4; d['last']=(d.round_number%4==3)
# group means per round for both groups
gr = d.groupby(['run','episode','round_number','agent_group']).agg(gp=('punishment','mean'), gpay=('payoff','mean')).reset_index()
d = d.merge(gr, on=['run','episode','round_number','agent_group'])
o = gr.rename(columns={'agent_group':'og','gp':'op','gpay':'opay'}); d['og']=1-d.agent_group
d = d.merge(o, on=['run','episode','round_number','og'], how='left')
d['opay']=d.opay.fillna(d.gpay)
d = d[(d.block<5)&(d.mgr=='ah')]
b = d.groupby(['run','episode','participant_code','block']).apply(lambda x: pd.Series({
  'p_last': x.punishment[x['last']].sum(), 'p_early': x.punishment[~x['last']].sum()/3,
  'c': x.contribution.mean(), 'dpay': (x.opay-x.gpay).mean(), 'left': float(x.gn.iloc[-1]!=x.agent_group.iloc[-1])}), include_groups=False).reset_index()
from sklearn.linear_model import LogisticRegression
X = b[['p_last','p_early','c','dpay']].values
m = LogisticRegression().fit(X, b.left)
print('logit left ~ p_last, p_early(mean), c, other-own payoff:', m.coef_.round(3), m.intercept_.round(3))
b['pl']=pd.cut(b.p_last,[-1,0,3,30]); b['pe']=pd.cut(b.p_early,[-1,0,1,3,30])
print(b.pivot_table(index='pe',columns='pl',values='left',aggfunc='mean',observed=True).round(3))
