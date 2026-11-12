"""v7: v6_crn plus v6. (v6: copy of v5_crn with v5 added); v4: the finals' slots share seeds across versions. Per game lead for each final point,
keyed by (slot, seat, episode); check whether games without fines replay identically."""
import pandas as pd, numpy as np, json
vs = ['pf-20261112-2', 'pf-20261112-2-v2', 'pf-20261112-2-v3', 'pf-20261112-2-v4', 'pf-20261112-2-v5', 'pf-20261112-2-v6']
rows = []
for v in vs:
    d = json.load(open(f'plots/simulation/policy_finder/{v}_sweep/final/sweep.json'))
    names = [p['name'] for p in d['points']]
    df = pd.read_parquet(f'plots/simulation/policy_finder/{v}_sweep/final/per_round.parquet')
    r = df.run.str.extract(r'by (\S+)_vs_(\S+)$')
    first = r[0].str.startswith('pf')
    df['pt'] = np.where(first, r[0], r[1]); df['seat'] = np.where(first, 0, 1)
    df['mine'] = df.group_id == df.seat
    df['pool'] = 1.6 * df.contribution - df.punishment
    df['sgn'] = np.where(df.mine, 1, -1)
    df['lead'] = df.pool * df.sgn
    df['pme'] = df.punishment * df.mine
    g = df.groupby(['pt', 'seat', 'episode']).agg(lead=('lead', 'sum'), spend=('pme', 'sum'), cme=('contribution', lambda s: 0)).reset_index()
    g['lead'] /= 24
    # first round my manager fines
    f = df[df.pme > 0].groupby(['pt', 'seat', 'episode']).round_number.min().rename('t_first')
    g = g.merge(f, on=['pt', 'seat', 'episode'], how='left')
    g['slot'] = g.pt.map({n: i for i, n in enumerate(names)})
    g['ver'] = v
    # final size of my group
    last = df[df.round_number == 23].groupby(['pt', 'seat', 'episode']).mine.sum().rename('n_end')
    g = g.merge(last, on=['pt', 'seat', 'episode'])
    rows.append(g.drop(columns='cme'))
G = pd.concat(rows)
G.to_parquet('/tmp/v7_crn.parquet')
print(G.groupby(['slot', 'ver']).agg(lead=('lead', 'mean'), spend=('spend', 'mean'), fired=('t_first', lambda s: s.notna().mean())).round(3))

# v6: paired against never-punish (v1's gp slot = never-punish on the same seeds)
base = G[(G.ver == 'pf-20261112-2') & (G.slot == 0)].set_index(['seat', 'episode']).lead
for v in vs:
    for s in sorted(G[G.ver == v].slot.unique()):
        x = G[(G.ver == v) & (G.slot == s)].set_index(['seat', 'episode'])
        d = x.lead - base.reindex(x.index)
        fired = x.t_first.notna()
        print(v, s, 'paired %.3f +- %.3f' % (d.mean(), d.std() / np.sqrt(len(d))),
              'fired %.2f' % fired.mean(), 'per fired %.3f +- %.3f' % (d[fired].mean(), d[fired].std() / np.sqrt(max(fired.sum(), 1))),
              'unfired maxabs %.3f' % (d[~fired].abs().max() if (~fired).any() else 0), 'spend %.2f' % x.spend.mean())
