"""AH: weight of punishment in each round of a block (relative to the switch decision) on exit."""
import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
d = pd.read_parquet('plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet')
d['mgr'] = [r.split('by ')[1].split('_vs_')[int(a)] for r, a in zip(d.run, d.agent_group)]
d = d.sort_values(['run', 'episode', 'participant_code', 'round_number'])
key = ['run', 'episode', 'participant_code']
g = d.groupby(key)
d['grp_next'] = g.agent_group.shift(-1)
d['switch'] = (d.grp_next != d.agent_group) & d.grp_next.notna()
gg = d.groupby(['run', 'episode', 'round_number', 'agent_group'])
d['gsize'] = gg.contribution.transform('size')
d['oth_p'] = (gg.punishment.transform('sum') - d.punishment) / (d.gsize - 1).replace(0, np.nan)
for k in range(4):
    d[f'p_l{k}'] = g.punishment.shift(k)
    d[f'op_l{k}'] = g.oth_p.shift(k)
    d[f'c_l{k}'] = g.contribution.shift(k)
    d[f'grp_l{k}'] = g.agent_group.shift(k)
sw = d[d.round_number.isin([7, 11, 15, 19]) & (d.mgr == 'ah')].copy()
stay = np.all([sw[f'grp_l{k}'] == sw.agent_group for k in range(4)], axis=0)
sw = sw[stay].fillna(0)
X = np.column_stack([np.ones(len(sw))] + [sw[f'p_l{k}'] for k in range(4)] + [sw[f'op_l{k}'] for k in range(4)] + [sw[f'c_l{k}'] for k in range(4)] + [sw.gsize])
b = np.linalg.lstsq(X, sw.switch.astype(float).values, rcond=None)[0]
print('n', len(sw))
print('const', b[0].round(3))
print('own p, lag 0 (decision round) .. 3:', b[1:5].round(4))
print('others mean p, lag 0..3:', b[5:9].round(4))
print('own c, lag 0..3:', b[9:13].round(4)); print('size', b[13].round(4))
# first decision, round 3
f = d[(d.round_number == 3) & (d.mgr == 'ah')].fillna(0)
X = np.column_stack([np.ones(len(f))] + [f[f'p_l{k}'] for k in range(4)] + [f[f'op_l{k}'] for k in range(4)] + [f[f'c_l{k}'] for k in range(4)])
b = np.linalg.lstsq(X, f.switch.astype(float).values, rcond=None)[0]
print('\nround 3 decision: const', b[0].round(3), 'own p lag0..3', b[1:5].round(4), 'others p', b[5:9].round(4), 'own c', b[9:13].round(4))
