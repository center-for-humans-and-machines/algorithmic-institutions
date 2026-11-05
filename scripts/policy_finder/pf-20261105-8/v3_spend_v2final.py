"""v3: where v2's final rule spends its punishment, by round and recipient's contribution, and what
the recipients do next round."""
import pandas as pd, numpy as np
f = pd.read_parquet('plots/simulation/policy_finder/pf-20261105-8-v2_sweep/final/per_round.parquet')
f = f[f.run.str.endswith('_gp')].copy()
f['player'] = f.participant_code.str.split('_').str[0].astype(int)
f = f.sort_values(['episode', 'player', 'round_number'])
f['cnext'] = f.groupby(['episode', 'player']).contribution.shift(-1)
v = f[f.contribution_valid.astype(bool)]
v = v.assign(cb=pd.cut(v.contribution, [-1, 2, 7, 12, 15, 17, 19, 20]))
ng = v.episode.nunique()
for rr, lab in [((0, 0), 'round 0'), ((1, 3), 'rounds 1-3'), ((4, 8), 'rounds 4-8')]:
    s = v[(v.round_number >= rr[0]) & (v.round_number <= rr[1])]
    g = s.groupby('cb', observed=True)
    print(lab); print(pd.DataFrame({'share': g.size() / len(s), 'p_mean': g.punishment.mean(),
          'spend/player/game': g.punishment.sum() / ng / 4, 'dc_next': (g.cnext.mean() - g.contribution.mean())}).round(2).to_string())
