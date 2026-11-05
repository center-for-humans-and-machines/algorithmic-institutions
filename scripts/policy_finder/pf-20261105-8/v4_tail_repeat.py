"""v4: under v3's final rule, how is the tail's punishment spread over fresh and repeated
free riders, and what do they contribute next?"""
import pandas as pd, numpy as np
f = pd.read_parquet('plots/simulation/policy_finder/pf-20261105-8-v3_sweep/final/per_round.parquet')
for run in ['pf-20261105-8-v3_gp', 'pf-20261105-8-v3_s172', 'pf-20261105-8-v3_s046']:
    d = f[f.run.str.endswith(run)].copy()
    d['player'] = d.participant_code.str.split('_').str[0].astype(int)
    d = d.sort_values(['episode', 'player', 'round_number'])
    n = d.episode.nunique()
    C = d.contribution.values.reshape(n, 4, 24).astype(float); V = d.contribution_valid.values.reshape(n, 4, 24).astype(bool)
    P = d.punishment.values.reshape(n, 4, 24).astype(float) * V
    h = {'pf-20261105-8-v3_gp': (5, 17), 'pf-20261105-8-v3_s172': (8, 17), 'pf-20261105-8-v3_s046': (5, 15)}[run]
    a, b = h
    tp = P[:, :, a:b]
    tot = tp.sum()
    # number of earlier tail punishments of this player
    hit = (tp > 0).astype(int)
    prior = np.cumsum(hit, 2) - hit
    print(run, 'tail pts/player/game', round(tot / n / 4, 2), 'share of games with any tail hit', round(((tp > 0).any((1, 2))).mean(), 3))
    for k in [0, 1, 2, 3, 5]:
        m = (hit == 1) & ((prior == k) if k < 5 else (prior >= 5))
        nxt = C[:, :, a + 1:b + 1][m[:, :, :b - a - (1 if b == 24 else 0)]] if False else None
        # next-round contribution of hit players
        mm = m[:, :, :-1]
        nx = C[:, :, a + 1:b][mm]
        print(f'  prior hits {k}{"+" if k==5 else ""}: share of tail pts {tp[m].sum()/tot:.2f}, n {m.sum()}, next c mean {np.nanmean(nx):.1f}, next c==0 {np.mean(nx==0):.2f}')
