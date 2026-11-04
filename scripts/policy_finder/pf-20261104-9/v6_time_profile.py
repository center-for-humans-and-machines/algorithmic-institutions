"""v6: per-round pool, members and mean c of the rule's group: earlier finals vs zero_vs_ah (reference)."""
import pandas as pd, numpy as np, glob
def prof(fn, run_filter=None):
    d = pd.read_parquet(fn, columns=['run', 'episode', 'round_number', 'group_id', 'contribution', 'punishment', 'contribution_valid'])
    if run_filter: d = d[d.run.str.contains(run_filter)]
    d = d[d.group_id == 0]
    g = d.groupby(['episode', 'round_number']).agg(c=('contribution', 'sum'), p=('punishment', 'sum'), n=('contribution', 'size'))
    g['pool'] = 1.6 * g.c - g.p
    full = g.reindex(pd.MultiIndex.from_product([sorted(d.episode.unique()), range(24)]), fill_value=0)
    r = full.groupby(level=1).mean()
    r['cpm'] = r.c / r.n.replace(0, np.nan); r['ppm'] = r.p / r.n.replace(0, np.nan)
    return r[['pool', 'n', 'cpm', 'ppm']]
z = prof('plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet', 'zero_vs_ah')
out = {'zero': z}
for v in ['pf-20261104-9-v3', 'pf-20261104-9-v5']:
    out[v] = prof(f'plots/simulation/policy_finder/{v}_sweep/final/per_round.parquet', '_gp')
print(pd.concat(out, axis=1).round(2).to_string())
for k, r in out.items(): print(k, 'mean pool', r.pool.mean().round(2), 'rounds 0-3', r.pool[:4].mean().round(1), '4-11', r.pool[4:12].mean().round(1), '12-23', r.pool[12:].mean().round(1))
