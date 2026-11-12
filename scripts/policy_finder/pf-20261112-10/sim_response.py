"""AH responses in the reference sims: switching and next contribution vs punishment received."""
import pandas as pd, numpy as np
import warnings; warnings.filterwarnings('ignore')
d = pd.read_parquet('plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet')
d = d.sort_values(['run', 'episode', 'participant_code', 'round_number'])
g = d.groupby(['run', 'episode', 'participant_code'])
d['c_next'] = g.contribution.shift(-1); d['grp_next'] = g.agent_group.shift(-1)
d['dc'] = d.c_next - d.contribution
d['blk'] = d.round_number // 4
# who manages the player's group
def mgr(row_run, grp):
    a, b = row_run.split('by ')[1].split('_vs_'); return np.where(grp == 0, a, b)
d['mgr'] = np.concatenate([mgr(r, x.agent_group.values) for r, x in d.groupby('run', sort=False)])
# 1. contribution response within a block, in ah-managed groups
x = d[(d.mgr == 'ah') & d.contribution_valid.astype(bool) & d.c_next.notna() & (d.round_number % 4 != 3)].copy()
x['pbin'] = pd.cut(x.punishment, [-1, 0, 2, 5, 10, 30]); x['cbin'] = pd.cut(x.contribution, [-1, 0, 5, 10, 15, 19, 20])
print('sim dc by own contribution x punishment (ah groups)'); print(x.pivot_table('dc', 'cbin', 'pbin').round(2)); print(x.pivot_table('dc', 'cbin', 'pbin', aggfunc='size'))
z = d[(d.mgr == 'zero') & d.contribution_valid.astype(bool) & d.c_next.notna() & (d.round_number % 4 != 3)].copy()
z['cbin'] = pd.cut(z.contribution, [-1, 0, 5, 10, 15, 19, 20])
print('zero groups dc by contribution'); print(z.groupby('cbin').dc.mean().round(2))
# ah's punishment as function of contribution
d['cbin'] = pd.cut(d.contribution, [-1, 0, 5, 10, 15, 19, 20])
print('ah punishment by contribution bin x block'); print(d[d.mgr == 'ah'].pivot_table('punishment', 'cbin', 'blk').round(2))
# 2. switching at decision points vs punishment in block
b = d[d.round_number % 4 == 3].copy()
b['left'] = (b.grp_next != b.agent_group).astype(float)
bp = d.groupby(['run', 'episode', 'participant_code', 'blk']).agg(pblk=('punishment', 'mean'), cblk=('contribution', 'mean')).reset_index()
b = b.merge(bp, on=['run', 'episode', 'participant_code', 'blk'])
b = b[b.grp_next.notna()]
b['pbin'] = pd.cut(b.pblk, [-1, 0, 1, 3, 6, 30]); b['plbin'] = pd.cut(b.punishment, [-1, 0, 2, 5, 10, 30])
print('leave rate by manager'); print(b.groupby('mgr').left.mean().round(3))
print('leave rate by block mean punishment (ah groups)'); print(b[b.mgr == 'ah'].groupby('pbin').left.agg(['mean', 'size']).round(3))
print('leave rate by last-round punishment (ah groups)'); print(b[b.mgr == 'ah'].groupby('plbin').left.agg(['mean', 'size']).round(3))
b['cb'] = pd.cut(b.cblk, [-1, 5, 10, 15, 20])
print('leave rate by contribution x manager'); print(b.pivot_table('left', 'cb', 'mgr').round(3))
print('leave rate by block x manager'); print(b.pivot_table('left', 'blk', 'mgr').round(3))
