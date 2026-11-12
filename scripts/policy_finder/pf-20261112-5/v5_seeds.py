"""v5: are the final runs' seeds shared across versions (per slot)? Paired per-episode leads of each version's gp."""
import pandas as pd, numpy as np
from aimanager.simulation.pool_scores import against_anchor, add_pool
V = ['pf-20261112-5', 'pf-20261112-5-v2', 'pf-20261112-5-v3', 'pf-20261112-5-v4']
leads = {}
for v in V:
    d = add_pool(pd.read_parquet(f'plots/simulation/policy_finder/{v}_sweep/final/per_round.parquet'))
    s = against_anchor(d, 'ah')
    for m, f in s.items():
        f = f.reset_index() if 'episode' not in f.columns else f
        leads[m] = f
for m, f in leads.items():
    print(m, len(f), list(f.columns)[:8], round((f.pool - f.other).mean(), 2))
    break
gp = {v: leads[f'{v}_gp'] for v in V}
L = {v: (f.pool - f.other).values for v, f in gp.items()}
for a in V:
    for b in V:
        if a < b:
            dd = L[b] - L[a]
            print(a, b, 'diff %.2f se %.2f corr %.2f' % (dd.mean(), dd.std() / np.sqrt(len(dd)), np.corrcoef(L[a], L[b])[0, 1]))
# slot 2 across versions (different params though)
s2 = {v: [m for m in leads if m.startswith(v + '_s')] for v in V}
print(s2)
