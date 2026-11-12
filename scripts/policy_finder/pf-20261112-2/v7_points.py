"""v7: per-game never-punish baseline (as v6_gamebase: any version's unfired game at the same
(idx, seat, episode)) from the lean tables of v7_games_lean, then v5 and v6 points scored by the
paired gain, d_all = mean over all 500 games of (lead - baseline lead), fired games without a
baseline imputed with the point's covered mean. Pools v5 (n_min = 2) with v6. Writes /tmp/v7_pts.parquet
and /tmp/v7_H.parquet."""
import pandas as pd, numpy as np, json, glob
vs = ['pf-20261112-2', 'pf-20261112-2-v2', 'pf-20261112-2-v3', 'pf-20261112-2-v4', 'pf-20261112-2-v5', 'pf-20261112-2-v6']
G = pd.concat([pd.read_parquet(f'/tmp/v7_games_{v}.parquet') for v in vs])
U = G[G.spend == 0]
base = U.groupby(['idx', 'seat', 'episode']).agg(lead0=('lead', 'median'), n0=('n_end', 'median'), nb=('lead', 'size'))
print('cells with baseline %.3f' % (len(base) / 128000))
H = G.join(base, on=['idx', 'seat', 'episode'])
H['fired'] = H.spend > 0
H['d'] = np.where(H.fired, H.lead - H.lead0, 0.0)
H['dn'] = np.where(H.fired, H.n_end - H.n0, 0.0)
H.to_parquet('/tmp/v7_H.parquet')
rows = []
for v in ['pf-20261112-2-v5', 'pf-20261112-2-v6']:
    sw = json.load(open(f'plots/simulation/policy_finder/{v}_sweep/sweep.json'))
    par = pd.DataFrame([dict(idx=int(p['name'].split('_s')[-1]), raw=p['pool'], **p['params']) for p in sw['points']])
    if 'n_min' not in par:
        par['n_min'] = 2
    h = H[H.ver == v]
    f = h[h.fired]
    pt = f.groupby('idx').agg(nf=('d', 'size'), d_f=('d', lambda s: s.dropna().mean()), se_f=('d', lambda s: s.dropna().sem()),
                              cov=('lead0', lambda s: s.notna().mean()), dn_f=('dn', lambda s: s.dropna().mean()))
    sp = h.groupby('idx').agg(spend=('spend', 'mean'), **{f'sp{b}': (f'sp{b}', 'mean') for b in range(6)})
    pt = par.set_index('idx').join(sp).join(pt).reset_index()
    pt['fired'] = pt.nf.fillna(0) / 500
    pt['d_all'] = pt.d_f.fillna(0) * pt.fired
    pt['se_all'] = pt.se_f.fillna(0) * pt.fired
    pt['ver'] = v
    rows.append(pt)
P = pd.concat(rows)
P.to_parquet('/tmp/v7_pts.parquet')
for v, x in P.groupby('ver'):
    on = x[x.fine >= 1]
    print(v, 'fining points', len(on), 'd_all %.3f +- %.3f' % (on.d_all.mean(), np.sqrt((on.se_all ** 2).sum()) / len(on)),
          'fired %.2f spend %.1f cov %.3f' % (on.fired.mean(), on.spend.mean(), on['cov'].mean()))
