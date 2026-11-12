"""Human data: who leaves a group at a switch, and why."""
import pandas as pd, numpy as np
import warnings; warnings.filterwarnings('ignore')
d = pd.read_csv('experiments/2group_8agent_50ep.csv')
d = d.drop_duplicates(['session', 'episode', 'round_number', 'participant_code'])
d['game'] = d.session + '_' + d.episode.astype(str)
d = d.sort_values(['game', 'participant_code', 'round_number'])
d['blk'] = d.round_number // 4
# per player-block summaries
gb = d.groupby(['game', 'participant_code', 'blk'])
b = gb.agg(grp=('group_id', 'last'), c=('contribution', 'mean'), p=('punishment', 'mean'),
           p_last=('punishment', 'last'), pay=('payoff', 'mean'), cg=('common_good', 'mean')).reset_index()
b = b.sort_values(['game', 'participant_code', 'blk'])
b['grp_next'] = b.groupby(['game', 'participant_code']).grp.shift(-1)
b = b[b.grp_next.notna()]
b['left'] = (b.grp_next != b.grp).astype(int)
# group-level block summaries
gg = d.groupby(['game', 'blk', 'group_id']).agg(gp=('punishment', 'mean'), gc=('contribution', 'mean'),
    gpay=('payoff', 'mean'), n=('participant_code', 'nunique')).reset_index()
gg['n'] = gg.n
other = gg.copy(); other['group_id'] = 1 - other.group_id
b = b.merge(gg.rename(columns={'group_id': 'grp'}), on=['game', 'blk', 'grp'])
b = b.merge(other.rename(columns={'group_id': 'grp', 'gp': 'op', 'gc': 'oc', 'gpay': 'opay', 'n': 'on'}), on=['game', 'blk', 'grp'], how='left')
print('overall leave rate %.3f' % b.left.mean(), 'n', len(b))
b['pbin'] = pd.cut(b.p, [-1, 0, 1, 3, 6, 30])
print('\nleave rate by own mean punishment in block'); print(b.groupby('pbin').left.agg(['mean', 'size']).round(3))
b['gpd'] = pd.cut(b.gp - b.op, [-30, -2, -0.5, 0.5, 2, 30])
print('\nleave rate by own group mean punishment minus other group'); print(b.groupby('gpd').left.agg(['mean', 'size']).round(3))
b['payd'] = pd.cut(b.gpay - b.opay, [-50, -4, -1, 1, 4, 50])
print('\nleave rate by own group mean payoff minus other'); print(b.groupby('payd').left.agg(['mean', 'size']).round(3))
b['cbin'] = pd.cut(b.c, [-1, 5, 10, 15, 20])
print('\nleave rate by own contribution x own punishment'); print(b.pivot_table('left', 'cbin', 'pbin', aggfunc='mean').round(2))
print('\nleave rate by own group size'); print(b.groupby('n').left.agg(['mean', 'size']).round(3))
import statsmodels.formula.api as smf
b['dp'] = b.gp - b.op; b['dpay'] = b.gpay - b.opay; b['dc'] = b.gc - b.oc
m = smf.logit('left ~ p + dp + dpay + c + dc + n', b.dropna()).fit(disp=0); print(m.summary().tables[1])
