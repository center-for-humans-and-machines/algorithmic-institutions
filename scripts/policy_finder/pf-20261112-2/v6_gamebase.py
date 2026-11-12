"""v6: a per-game never-punish baseline. A game in which a point never fines replays never-punish
exactly (same seeds per sweep index, v4 entry 1/v5 entry 1: unfired diffs 0.0). So for each
(idx, seat, episode), any version's unfired game there gives the never-punish lead of that game.
Pair every fired game with it; report coverage."""
import pandas as pd, numpy as np, json
G = pd.read_parquet('/tmp/v6_games.parquet')
U = G[G.spend == 0]
chk = U.groupby(['idx', 'seat', 'episode']).lead.agg(['min', 'max', 'size'])
print('unfired games per cell', chk['size'].describe().round(2).to_dict(), 'max spread', (chk['max'] - chk['min']).max())
base = U.groupby(['idx', 'seat', 'episode']).agg(lead0=('lead', 'first'), n0=('n_end', 'first'))
print('cells with a baseline: %.3f of %d' % (len(base) / (256 * 500), 256 * 500))
H = G.join(base, on=['idx', 'seat', 'episode'])
H['d'] = np.where(H.spend == 0, 0.0, H.lead - H.lead0)
H['dn'] = np.where(H.spend == 0, 0.0, H.n_end - H.n0)
H.to_parquet('/tmp/v6_H.parquet')
F = H[H.spend > 0]
print('fired games with baseline: %.3f' % F.lead0.notna().mean())
pt = H.groupby(['ver', 'idx']).apply(lambda x: pd.Series({
    'fired': (x.spend > 0).mean(), 'cov': x[x.spend > 0].lead0.notna().mean() if (x.spend > 0).any() else 1.0,
    'd_fired': x[(x.spend > 0)].d.mean(), 'se_fired': x[(x.spend > 0)].d.sem(),
    'spend': x.spend.mean(), 'dn_fired': x[(x.spend > 0)].dn.mean()}), include_groups=False).reset_index()
pt['d_all'] = pt.d_fired.fillna(0) * pt.fired  # assumes uncovered fired games like covered
pt.to_csv('scripts/policy_finder/pf-20261112-2/v6_gamebase_points.csv', index=False)
print(pt.groupby('ver')[['fired', 'cov', 'd_fired', 'd_all', 'spend']].mean().round(3).to_string())
