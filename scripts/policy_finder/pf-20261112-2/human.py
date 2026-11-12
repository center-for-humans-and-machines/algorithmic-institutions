"""Human data: manager punishment policy, contribution response, switching, per-capita pool vs punishment."""
import pandas as pd, numpy as np
h = pd.read_csv('experiments/2group_8agent_50ep.csv')
h = h.drop_duplicates(['session', 'episode', 'round_number', 'participant_code'])
print('games', h.groupby(['session', 'episode']).ngroups, 'rows', len(h))
h = h.sort_values(['session', 'episode', 'participant_code', 'round_number'])
key = ['session', 'episode', 'participant_code']
G = h.groupby(key)
h['valid'] = 1 - h.player_no_input
h['c_next'] = G.contribution.shift(-1); h['g_next'] = G.group_id.shift(-1)
h['v_next'] = G.valid.shift(-1)
h['gmean'] = h.groupby(['session', 'episode', 'round_number', 'group_id']).contribution.transform('mean')
x = h[(h.valid == 1) & (h.v_next == 1) & (h.g_next == h.group_id) & (h.manager_no_input == 0)].copy()
x['dc'] = x.c_next - x.contribution
x['cb'] = pd.cut(x.contribution, [-1, 0, 4, 8, 12, 16, 20])
x['pb'] = pd.cut(x.punishment, [-1, 0, 2, 5, 10, 30])
print(x.pivot_table(index='cb', columns='pb', values='dc', aggfunc='mean', observed=True).round(2))
print(x.pivot_table(index='cb', columns='pb', values='dc', aggfunc='size', observed=True))
print('human mgr punishment by c'); print(x.groupby('cb', observed=True).punishment.agg(['mean', lambda s: (s > 0).mean()]).round(2))
print('mean punishment by round'); print(h[h.valid == 1].groupby('round_number').punishment.mean().round(2).values)
# per-capita pool per group-round, vs mean punishment of manager in game
gr = h[h.valid == 1].groupby(['session', 'episode', 'group_id', 'round_number']).agg(c=('contribution', 'mean'), p=('punishment', 'mean'), n=('valid', 'size')).reset_index()
gr['pc'] = 1.6 * gr.c - gr.p
gg = gr.groupby(['session', 'episode', 'group_id']).agg(c=('c', 'mean'), p=('p', 'mean'), pc=('pc', 'mean'), n=('n', 'mean')).reset_index()
gg['pbin'] = pd.qcut(gg.p, 4)
print(gg.groupby('pbin', observed=True)[['c', 'p', 'pc', 'n']].mean().round(2))
# switching in humans
h['sw'] = (h.g_next != h.group_id).astype(float)
s = h[h.round_number.isin([3, 7, 11, 15, 19]) & h.g_next.notna()].copy()
h['blk'] = h.round_number // 4
bp = h.groupby(key + ['blk']).agg(bp=('punishment', 'mean'), bc=('contribution', 'mean')).reset_index()
s['blk'] = s.round_number // 4
s = s.merge(bp, on=key + ['blk'])
s['bpb'] = pd.cut(s.bp, [-1, 0, 1, 2, 4, 30]); s['bcb'] = pd.cut(s.bc, [-1, 6, 12, 20])
print('human switch rate'); print(s.pivot_table(index='bcb', columns='bpb', values='sw', aggfunc='mean', observed=True).round(2))
print(s.pivot_table(index='bcb', columns='bpb', values='sw', aggfunc='size', observed=True))
