"""Pilot random_1: per-group punishment rule (fitted) vs group outcomes."""
import pandas as pd, numpy as np
pd.set_option('display.width', 200)
h = pd.read_csv('experiments/pilot_random1_player_round_slim.csv')
r = h[(h.experiment_name == 'random_1')]
rows = []
for gid, x in r.groupby('global_group_id'):
    v = x[x.player_no_input == 0]
    b, a = np.polyfit(v.contribution, v.punishment, 1) if v.contribution.nunique() > 1 else (np.nan, np.nan)
    thr = -a / b if b < 0 else np.nan  # contribution where fitted punishment hits 0
    late = v[v.round_number >= 8]
    rows.append(dict(g=gid, slope=-b, thr=thr, p=v.punishment.mean(), c_early=v[v.round_number < 4].contribution.mean(),
                     c_late=late.contribution.mean(), pay=x[x.player_no_input == 0].payoff.mean(),
                     pay_late=late.payoff.mean(), to=x.player_no_input.mean()))
g = pd.DataFrame(rows)
print(g.describe().round(2))
print(g[['slope', 'thr', 'p', 'c_early', 'c_late', 'pay', 'pay_late']].corr().round(2))
g['sb'] = pd.qcut(g.slope, 3)
print(g.groupby('sb', observed=True)[['p', 'c_early', 'c_late', 'pay', 'pay_late']].mean().round(2))
g['tb'] = pd.qcut(g.thr, 3)
print(g.groupby('tb', observed=True)[['slope', 'p', 'c_early', 'c_late', 'pay', 'pay_late']].mean().round(2))
