"""In the reference sims: how artificial humans' contributions and switches respond to
the ah manager's punishment (the stack the rule faces)."""
import numpy as np, pandas as pd
d = pd.read_parquet('plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet')
d = d[d.run.isin(['ah group_switching managed by ah_vs_zero', 'ah group_switching managed by zero_vs_ah', 'ah group_switching managed by ah_vs_ah'])].copy()
d = d.sort_values(['run', 'episode', 'participant_code', 'round_number'])
k = ['run', 'episode', 'participant_code']
d['pc'] = d.punishment * d.contribution_valid
g = d.groupby(['run', 'episode', 'round_number', 'agent_group'])
d['n'] = g.contribution.transform('size')
d['next_c'] = d.groupby(k).contribution.shift(-1)
d['next_v'] = d.groupby(k).contribution_valid.shift(-1)
d['next_g'] = d.groupby(k).agent_group.shift(-1)
d['dc'] = d.next_c - d.contribution
d['c2'] = d.groupby(k).contribution.shift(-2)
d['c3'] = d.groupby(k).contribution.shift(-3)
# manager label of group
def mgr(r):
    a, b = r.split('by ')[1].split('_vs_')
    return a, b
d['mgr'] = [mgr(r)[int(gg)] for r, gg in zip(d.run, d.agent_group)]
v = d[(d.contribution_valid == 1)]
s = v[(v.next_g == v.agent_group) & (v.next_v == 1) & ~v.round_number.isin([3, 7, 11, 15, 19, 23]) & (v.mgr == 'ah')].copy()
s['pbin'] = pd.cut(s.punishment, [-1, 0, 2, 5, 10, 30])
s['cbin'] = pd.cut(s.contribution, [-1, 0, 5, 10, 15, 20])
print('ah-managed: dc next round by c bin x p bin')
print(s.pivot_table(index='cbin', columns='pbin', values='dc', aggfunc='mean', observed=False).round(2))
print(s.pivot_table(index='cbin', columns='pbin', values='dc', aggfunc='count', observed=False))
s['dc3'] = s.c3 - s.contribution
print('dc 3 rounds later')
print(s.pivot_table(index='cbin', columns='pbin', values='dc3', aggfunc='mean', observed=False).round(2))
# zero managed baseline dc by c bin
z = v[(v.next_g == v.agent_group) & (v.next_v == 1) & ~v.round_number.isin([3, 7, 11, 15, 19, 23]) & (v.mgr == 'zero')].copy()
print('zero-managed dc by c bin', z.groupby(pd.cut(z.contribution, [-1, 0, 5, 10, 15, 20]), observed=False).dc.mean().round(2).tolist())
# switching
e = d.copy()
e['p_last'] = e.pc
e['p4'] = e.groupby(k).pc.transform(lambda x: x.rolling(4, min_periods=1).mean())
sw = e[e.round_number.isin([3, 7, 11, 15, 19]) & e.next_g.notna()].copy()
sw['left'] = (sw.next_g != sw.agent_group).astype(int)
for m in ['ah', 'zero']:
    q = sw[sw.mgr == m]
    print(m, 'leave rate', q.left.mean().round(3))
q = sw[sw.mgr == 'ah']
print(q.groupby(pd.cut(q.p4, [-1, 0, 1, 3, 6, 30]), observed=False).left.agg(['mean', 'count']).round(3))
print('by last-round p', q.groupby(pd.cut(q.p_last, [-1, 0, 2, 5, 10, 30]), observed=False).left.agg(['mean', 'count']).round(3))
# punished in rounds before last but not in last
q2 = q[q.p_last == 0]
print('p_last=0, by p4', q2.groupby(pd.cut(q2.p4, [-1, 0, 1, 3, 30]), observed=False).left.agg(['mean', 'count']).round(3))
print('by own c', q.groupby(pd.cut(q.contribution, [-1, 0, 5, 10, 15, 20]), observed=False).left.agg(['mean', 'count']).round(3))
# group punishment total (others) effect: leave for those not punished themselves
gp = e.groupby(['run', 'episode', 'round_number', 'agent_group']).pc.sum().rename('gpsum')
q = q.join(gp, on=['run', 'episode', 'round_number', 'agent_group'])
q3 = q[q.p4 == 0]
print('unpunished members, by group punishment sum last round', q3.groupby(pd.cut(q3.gpsum, [-1, 0, 3, 8, 15, 200]), observed=False).left.agg(['mean', 'count']).round(3))
