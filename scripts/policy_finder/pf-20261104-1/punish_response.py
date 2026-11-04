"""Contribution response to punishment, in the reference sims (AH model) and in the human data."""
import pandas as pd, numpy as np
def prep(d, pid):
    d = d.sort_values(['src','episode',pid,'round_number']).copy()
    k = d.groupby(['src','episode',pid])
    d['c_next'] = k.contribution.shift(-1); d['g_next'] = k.group_id.shift(-1) if 'group_id' in d else k.agent_group.shift(-1)
    gcol = 'agent_group' if 'agent_group' in d else 'group_id'
    d['gmean'] = d.groupby(['src','episode','round_number',gcol]).contribution.transform('mean')
    d['others'] = (d.gmean*d.groupby(['src','episode','round_number',gcol]).contribution.transform('size') - d.contribution)/(d.groupby(['src','episode','round_number',gcol]).contribution.transform('size')-1).replace(0,np.nan)
    d = d[d.c_next.notna() & (d.round_number % 4 != 3)]  # same group next round
    d['dc'] = d.c_next - d.contribution
    return d
s = pd.read_parquet('plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet')
s = s[s.run.str.contains('ah_vs_ah')].copy(); s['src']='sim'
h = pd.read_csv('experiments/2group_8agent_50ep.csv').drop_duplicates(['session','episode','round_number','participant_code']); h['src']=h.session.astype(str)
for name, d, pid in [('SIM ah_vs_ah', s, 'participant_code'), ('HUMAN', h, 'participant_code')]:
    d = prep(d, pid)
    d['cb'] = pd.cut(d.contribution, [-1,4,9,14,19,20], labels=['0-4','5-9','10-14','15-19','20'])
    d['pb'] = pd.cut(d.punishment, [-1,0,2,5,10,30], labels=['0','1-2','3-5','6-10','11+'])
    print('=====', name, 'mean dc by contribution bin x punishment bin')
    print(d.pivot_table(index='cb', columns='pb', values='dc', aggfunc='mean', observed=True).round(2))
    print(d.pivot_table(index='cb', columns='pb', values='dc', aggfunc='size', observed=True))
    import statsmodels.formula.api as smf
    m = smf.ols('c_next ~ contribution + others + punishment + I(punishment**2)', data=d.dropna(subset=['others'])).fit()
    print(m.params.round(3))
