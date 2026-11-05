"""Dose-response of next (and 3-ahead) contribution to own punishment, with controls.
Sim (humanlike) and the human pilots (both experiments)."""
import pandas as pd, numpy as np
import statsmodels.formula.api as smf
bins = [-1, 0, 1, 2, 3, 5, 8, 12, 30]
def prep(d, ep, pid, rnd, valid):
    d = d.sort_values([ep, pid, rnd]).copy()
    d['c'] = np.where(d[valid] == 1, d.contribution, 0); d['v'] = d[valid].astype(int)
    g = d.groupby([ep, rnd])
    d['n'] = g.v.transform('sum')
    d['oth_c'] = (g.c.transform('sum') - d.c) / (d.n - d.v).clip(lower=1)
    d['oth_p'] = (g.punishment.transform('sum') - d.punishment) / (d.n - d.v).clip(lower=1)
    k = d.groupby([ep, pid])
    d['c_prev'] = k.c.shift(1)
    for lag in (1, 3):
        d[f'c{lag}'] = k.c.shift(-lag); d[f'v{lag}'] = k.v.shift(-lag)
    d['pb'] = pd.cut(d.punishment, bins).astype(str)
    return d[(d.v == 1)]
s = pd.read_parquet('plots/simulation/29_EXP1_reference/per_round.parquet')
s = s[s.run.str.contains('humanlike')]
s['pid'] = s.participant_code.str.split('_').str[0]
s['valid'] = s.contribution_valid.astype(int)
s = prep(s, 'episode', 'pid', 'round_number', 'valid')
h = pd.read_csv('experiments/pilot_random1_player_round_slim.csv')
h['valid'] = 1 - h.player_no_input
h = prep(h, 'episode_id', 'participant_code', 'round_number', 'valid')
for name, d in [('sim humanlike', s), ('humans', h)]:
    for lo, hi in [(0, 9), (10, 16), (17, 20)]:
        x = d[(d.c >= lo) & (d.c <= hi) & (d.v1 == 1)]
        m = smf.ols("c1 ~ c + oth_c + C(pb, Treatment('(-1, 0]'))", x).fit()
        coefs = {k.split('[T.')[1][:-1]: round(v, 2) for k, v in m.params.items() if 'pb' in k}
        x3 = d[(d.c >= lo) & (d.c <= hi) & (d.v3 == 1)]
        m3 = smf.ols("c3 ~ c + oth_c + C(pb, Treatment('(-1, 0]'))", x3).fit()
        coefs3 = {k.split('[T.')[1][:-1]: round(v, 2) for k, v in m3.params.items() if 'pb' in k}
        print(name, f'c {lo}-{hi}', 'n', len(x), '\n  t+1', coefs, '\n  t+3', coefs3)
