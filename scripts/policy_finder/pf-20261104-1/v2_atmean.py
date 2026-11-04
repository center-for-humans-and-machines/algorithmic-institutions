"""v2: does punishing a member who is at or above the mates' mean raise or lower its next contribution?
ah_vs_ah reference sims and human data; dc by gap-to-mates bin x punished or not, mid contributors (5-16)."""
import pandas as pd, numpy as np
import importlib.util, sys
spec = importlib.util.spec_from_file_location('pr', 'scripts/policy_finder/pf-20261104-1/punish_response.py')
src = open('scripts/policy_finder/pf-20261104-1/punish_response.py').read().split('s = pd.read_parquet')[0]
exec(src)
s = pd.read_parquet('plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet')
s = s[s.run.str.contains('ah_vs_ah')].copy(); s['src'] = 'sim'
h = pd.read_csv('experiments/2group_8agent_50ep.csv').drop_duplicates(['session', 'episode', 'round_number', 'participant_code']); h['src'] = h.session.astype(str)
for name, d in [('SIM ah_vs_ah', s), ('HUMAN', h)]:
    d = prep(d, 'participant_code').dropna(subset=['others'])
    d = d[(d.contribution >= 5) & (d.contribution <= 16)]
    d['gap'] = pd.cut(d.others - d.contribution, [-20, -4, -1, 1, 4, 20])
    d['pb'] = pd.cut(d.punishment, [-1, 0, 3, 8, 30])
    print('=====', name)
    print(d.pivot_table(index='gap', columns='pb', values='dc', aggfunc='mean', observed=True).round(2).to_string())
    print(d.pivot_table(index='gap', columns='pb', values='dc', aggfunc='size', observed=True).to_string())
