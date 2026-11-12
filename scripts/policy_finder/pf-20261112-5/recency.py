"""Does punishment late in a block weigh more on the switch than early in it? (sims and humans)"""
import pandas as pd, numpy as np
import statsmodels.formula.api as smf

def build(d, key):
    d = d.sort_values(key + ['round_number'])
    g = d.groupby(key)
    d['ng'] = g.group_id.shift(-1)
    for k in range(4):
        d[f'p{k}'] = g.punishment.shift(3 - k)  # p3 = decision round itself, p0 = block start
    d['prev_block_p'] = sum(g.punishment.shift(4 + j) for j in range(4)) / 4
    dec = d[d.round_number.isin([7, 11, 15, 19]) & d.ng.notna()].copy()
    dec['sw'] = (dec.ng != dec.group_id).astype(int)
    return dec

h = pd.read_csv('experiments/2group_8agent_50ep.csv').drop_duplicates(['session', 'episode', 'round_number', 'participant_code'])
hd = build(h.copy(), ['session', 'episode', 'participant_code'])
s = pd.read_parquet('plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet')
s = s[s.run.str.contains('ah_vs_ah|ah_vs_zero|zero_vs_ah')]
sd = build(s.copy(), ['run', 'episode', 'participant_code'])
for name, dec in [('humans', hd), ('sims', sd)]:
    m = smf.logit('sw ~ p0 + p1 + p2 + p3 + prev_block_p + C(round_number)', dec).fit(disp=0)
    print(name, len(dec)); print(m.params.filter(like='p').round(3).to_dict()); print(m.bse.filter(like='p').round(3).to_dict())
