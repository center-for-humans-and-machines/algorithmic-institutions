"""Which round of the block drives the switch? Logistic-ish OLS of switch on punishment in each of
the 4 rounds of the block, ah_vs_ah sims and humans."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
for src in ['sim', 'human']:
    if src == 'sim':
        d = pd.read_parquet('plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet')
        d = d[d.run.str.endswith('ah_vs_ah')]
        key = ['episode', 'participant_code']
    else:
        d = pd.read_csv('experiments/2group_8agent_50ep.csv')
        d = d.drop_duplicates(['session', 'episode', 'round_number', 'participant_code'])
        key = ['session', 'episode', 'participant_code']
    d = d.copy()
    d['blk'] = d.round_number // 4
    d['pos'] = d.round_number % 4
    w = d[d.round_number < 20].pivot_table(index=key + ['blk'], columns='pos', values='punishment')
    w.columns = [f'p{i}' for i in w.columns]
    c = d[d.round_number < 20].groupby(key + ['blk']).contribution.mean().rename('cm')
    g3 = d[d.pos == 3].set_index(key + ['blk']).group_id
    g0 = d[(d.pos == 0) & (d.blk > 0)].assign(blk=lambda x: x.blk - 1).set_index(key + ['blk']).group_id
    sw = (g3 != g0.reindex(g3.index)).astype(float)[g3.index.isin(g0.index)].rename('sw')
    x = w.join(c).join(sw, how='inner').dropna()
    X = np.column_stack([np.ones(len(x)), x.p0, x.p1, x.p2, x.p3, x.cm])
    beta = np.linalg.lstsq(X, x.sw.values, rcond=None)[0]
    print(src, 'switch prob per point in block round 0..3:', np.round(beta[1:5], 4), 'per contr point', round(beta[5], 4))
    # previous blocks' punishment (memory)
