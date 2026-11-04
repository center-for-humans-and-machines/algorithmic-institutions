"""Do punishments of group-mates move a player's next contribution (ah_vs_ah sim; humans)?"""
import pandas as pd, numpy as np, statsmodels.formula.api as smf
s = pd.read_parquet('plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet')
s = s[s.run.str.contains('ah_vs_ah')].rename(columns={'agent_group':'g'}); s['game']=s.episode
h = pd.read_csv('experiments/2group_8agent_50ep.csv').drop_duplicates(['session','episode','round_number','participant_code']).rename(columns={'group_id':'g'})
h['game'] = h.session.astype(str)+'_'+h.episode.astype(str)
for name,d in [('sim',s),('human',h)]:
    d = d.sort_values(['game','participant_code','round_number']).copy()
    k = ['game','round_number','g']
    n = d.groupby(k).contribution.transform('size')
    d['oc'] = (d.groupby(k).contribution.transform('sum')-d.contribution)/(n-1).replace(0,np.nan)
    d['op'] = (d.groupby(k).punishment.transform('sum')-d.punishment)/(n-1).replace(0,np.nan)
    d['cn'] = d.groupby(['game','participant_code']).contribution.shift(-1)
    d = d[(d.round_number%4!=3)].dropna(subset=['cn','oc','op'])
    m = smf.ols('cn ~ contribution + oc + punishment + op', data=d).fit()
    print(name, m.params.round(3).to_dict())
