"""Logit of leaving at a switch in the reference sims: own vs groupmates' punishment, timing."""
import pandas as pd, numpy as np
import statsmodels.formula.api as smf
import warnings; warnings.filterwarnings('ignore')
d = pd.read_parquet('plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet')
d = d.sort_values(['run', 'episode', 'participant_code', 'round_number'])
d['grp_next'] = d.groupby(['run', 'episode', 'participant_code']).agent_group.shift(-1)
d['blk'] = d.round_number // 4; d['pos'] = d.round_number % 4
key = ['run', 'episode', 'blk']
# per player-block: own punishment per position in block, own contribution
w = d.pivot_table(index=key + ['participant_code'], columns='pos', values='punishment').add_prefix('p')
w['c'] = d.groupby(key + ['participant_code']).contribution.mean()
last = d[d.pos == 3].set_index(key + ['participant_code'])
w['grp'] = last.agent_group; w['left'] = (last.grp_next != last.agent_group).astype(float); w = w[last.grp_next.notna()]
w['pown'] = w[['p0', 'p1', 'p2', 'p3']].mean(axis=1)
w = w.reset_index()
gg = w.groupby(key + ['grp']).agg(gp_sum=('pown', 'sum'), n=('pown', 'size'), gc=('c', 'mean')).reset_index()
w = w.merge(gg, on=key + ['grp'])
w['pmates'] = (w.gp_sum - w.pown) / (w.n - 1).replace(0, np.nan)
oth = gg.rename(columns={'gp_sum': 'o_sum', 'n': 'on', 'gc': 'oc'}); oth['grp'] = 1 - oth.grp
w = w.merge(oth, on=key + ['grp'], how='left').fillna({'o_sum': 0, 'on': 0, 'oc': 0})
w['pother'] = w.o_sum / w.on.replace(0, np.nan); w['pother'] = w.pother.fillna(0)
w['dcg'] = w.gc - w.oc
w = w.dropna(subset=['pmates'])
m = smf.logit('left ~ p0 + p1 + p2 + p3 + pmates + pother + c + dcg + n + C(blk)', w).fit(disp=0)
print(m.summary().tables[1])
# unpunished players: leave rate by mates' punishment
u = w[w.pown == 0].copy(); u['mb'] = pd.cut(u.pmates, [-1, 0, 1, 2, 4, 30])
print(u.groupby('mb').left.agg(['mean', 'size']).round(3))
