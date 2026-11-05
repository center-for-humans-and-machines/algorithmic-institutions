import sys; sys.path.insert(0,'/opt/algorithmic-institutions/scripts/policy_finder/pf-20261106-4')
from load import load
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
df, gs = load()
d = pd.read_pickle('/tmp/sw.pkl')
d['p_dec'] = d.punishment
d['p_rest'] = d.pblk - d.punishment
# punishment in earlier blocks
df['block'] = df.round_number//4
prior = df.groupby(['game','player_id','block']).punishment.sum().groupby(level=[0,1]).cumsum().reset_index()
prior['block'] += 1; prior = prior.rename(columns={'punishment':'p_prior'})
d = d.merge(prior, on=['game','player_id','block'], how='left').fillna({'p_prior':0})
d['p_prior_pr'] = d.p_prior/(d.block*4)
d['dcg'] = d.cg_b - d.cgo_b
d2 = d.dropna(subset=['dpc'])
for f in ['sw ~ p_dec + p_rest + p_prior_pr + dpc + n + round_number', 'sw ~ p_dec + p_rest + dcg + round_number', 'sw ~ p_dec + p_rest + dpc + round_number','sw ~ p_dec + p_rest + dpc + dmp + round_number']:
    m = smf.logit(f, d2).fit(disp=0); print(f, 'AIC', round(m.aic,1)); print(m.params.round(3).to_dict())
