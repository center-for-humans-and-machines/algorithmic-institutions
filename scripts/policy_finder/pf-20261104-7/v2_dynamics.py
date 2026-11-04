"""Per-round dynamics of v1's and v2's final points vs reference zero/ah: members, contribution, punishment, pool of my group (vs ah)."""
import pandas as pd
from aimanager.simulation.pool_scores import add_pool, pairing_of
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
B = 'plots/simulation/policy_finder/'
srcs = [B + 'pf-20261104-7_sweep/final/per_round.parquet', B + 'pf-20261104-7-v2_sweep/final/per_round.parquet',
        'plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet']
rows = []
for f in srcs:
    df = add_pool(pd.read_parquet(f))
    for run in df.run.unique():
        g0, g1 = pairing_of(run)
        sub = df[df.run == run]
        for grp, (m, o) in enumerate(((g0, g1), (g1, g0))):
            if o != 'ah' or (m == 'ah' and grp == 1):
                continue
            s = sub[sub.agent_group == grp]
            neps = sub.episode.nunique()
            r = s.groupby('round_number').agg(n=('pool', 'size'), c=('contribution', 'mean'), p=('punishment', 'mean'), pool=('pool', 'sum'))
            r['n'] /= neps; r['pool'] /= neps
            r['mgr'] = m.replace('pf-20261104-7', '')
            rows.append(r.reset_index())
R = pd.concat(rows)
for v in ['n', 'c', 'p', 'pool']:
    print(v); print(R.pivot_table(index='mgr', columns='round_number', values=v).round(1).to_string())
print(R.groupby('mgr').pool.mean().round(2))
