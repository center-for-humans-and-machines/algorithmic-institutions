"""v4: next-round contribution change by own contribution bin x punishment bin, humans (one copy per game) and the
reference sims' ah-managed groups (observational; mid-block rounds only so no switch in between)."""
import numpy as np, pandas as pd
h = pd.read_csv('experiments/2group_8agent_50ep.csv')
h = h.drop_duplicates(['session', 'episode', 'round_number', 'participant_code'])
h = h.sort_values(['session', 'episode', 'participant_code', 'round_number'])
h['c_next'] = h.groupby(['session', 'episode', 'participant_code']).contribution.shift(-1)
s = pd.read_parquet('plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet')
s = s[s.run.str.contains('ah_vs_ah')].sort_values(['episode', 'participant_code', 'round_number'])
s['c_next'] = s.groupby(['episode', 'participant_code']).contribution.shift(-1)
for lab, d in [('humans', h), ('sims ah_vs_ah', s)]:
    d = d[(d.round_number % 4 != 3) & (d.round_number < 23) & d.c_next.notna()].copy()
    d['cb'] = pd.cut(d.contribution, [-1, 4, 10, 15, 20], labels=['0-4', '5-10', '11-15', '16-20'])
    d['pb'] = pd.cut(d.punishment, [-1, 0, 3, 7, 30], labels=['0', '1-3', '4-7', '8+'])
    d['dc'] = d.c_next - d.contribution
    t = d.groupby(['cb', 'pb'], observed=True).dc.agg(['mean', 'count']).unstack('pb')
    print(lab); print(t.round(2))
